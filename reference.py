"""FROZEN REFERENCE (do not optimise): YOLO11l on every frame via plain Ultralytics predict.

This is the simple first version of run.py, kept unchanged to generate ground-truth files
that the optimised run.py is scored against with evaluate.py:
    python reference.py --input <folder of .parquet files> --output ref.jsonl

Development-only options (not needed for grading):
    --max-videos N      stop after N videos in total
    --max-seconds S     only process the first S seconds of each video (truncation, not an error)
    --batch B           frames per model call (default 16)
    --weights PATH      YOLO weights (default yolo11l.pt next to this script)

Per-second rules (from the brief):
    * second k holds frames with timestamp in [k, k+1), measured from the first decoded frame
    * an object is present if it is detected in at least half of that second's frames
    * people = median per-frame person count, rounded down on halves
    * a line is written for every second up to and including the second of the last frame
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from ultralytics import YOLO

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]
IMGSZ = 640
CONF = 0.4
IOU = 0.7


# --------------------------------------------------------------------------- per-second aggregation
class SecondAggregator:
    """Collects per-frame detection counts and turns them into one record per second."""

    def __init__(self) -> None:
        # second -> list of per-frame count vectors (len(TARGETS),)
        self.frames: dict[int, list[np.ndarray]] = defaultdict(list)
        self.max_second = -1

    def add(self, second: int, counts: np.ndarray) -> None:
        self.frames[second].append(counts)
        if second > self.max_second:
            self.max_second = second

    @staticmethod
    def _median_floor(values: np.ndarray) -> int:
        v = np.sort(values)
        n = len(v)
        if n % 2:
            return int(v[n // 2])
        return int((int(v[n // 2 - 1]) + int(v[n // 2])) // 2)

    def records(self, video_id: str):
        for s in range(self.max_second + 1):
            fr = self.frames.get(s)
            if not fr:  # gap in timestamps: no frames in this second
                yield {"video": video_id, "second": s, "objects": [], "people": 0}
                continue
            m = np.stack(fr)                    # (n_frames, 5)
            n = m.shape[0]
            present = (m > 0).sum(axis=0) * 2 >= n
            objects = [name for name, p in zip(TARGETS, present) if p]
            people = self._median_floor(m[:, 0])
            yield {"video": video_id, "second": s, "objects": objects, "people": people}


# --------------------------------------------------------------------------- decoding
def rotate_frame(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the display-matrix rotation (PyAV reports degrees counter-clockwise)."""
    k = (rotation // 90) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def iter_frames(mp4_bytes: bytes, max_seconds: float | None):
    """Yield (relative_time_seconds, BGR ndarray) for every decoded frame."""
    with av.open(io.BytesIO(mp4_bytes)) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        t0 = None
        last_t = None
        for frame in container.decode(stream):
            t = frame.time
            if t is None:  # no timestamp: extrapolate from frame rate
                rate = float(stream.average_rate or 30)
                t = (last_t + 1.0 / rate) if last_t is not None else 0.0
            if t0 is None:
                t0 = t
            rel = t - t0
            last_t = t
            if max_seconds is not None and rel >= max_seconds:
                break
            img = frame.to_ndarray(format="bgr24")
            rot = int(getattr(frame, "rotation", 0) or 0)
            if rot:
                img = rotate_frame(img, rot)
            yield rel, img


# --------------------------------------------------------------------------- model
class Detector:
    def __init__(self, weights: str, batch: int) -> None:
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        names = self.model.names
        name_to_id = {v: k for k, v in names.items()}
        self.class_ids = [name_to_id[n] for n in TARGETS]
        self.batch = batch
        self.infer_time = 0.0

    def counts(self, images: list[np.ndarray]) -> list[np.ndarray]:
        t = time.perf_counter()
        results = self.model.predict(
            images, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=self.class_ids,
            device=self.device, verbose=False,
        )
        self.infer_time += time.perf_counter() - t
        out = []
        for r in results:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            out.append(np.array([(cls == c).sum() for c in self.class_ids], dtype=np.int32))
        return out


# --------------------------------------------------------------------------- per video
def process_video(video_id: str, mp4_bytes, det: Detector, out, max_seconds):
    agg = SecondAggregator()
    n_frames = 0
    error = None
    batch_imgs, batch_secs = [], []

    def flush():
        nonlocal batch_imgs, batch_secs
        if batch_imgs:
            for s, c in zip(batch_secs, det.counts(batch_imgs)):
                agg.add(s, c)
            batch_imgs, batch_secs = [], []

    try:
        if not mp4_bytes:
            raise ValueError("empty mp4 bytes")
        for rel, img in iter_frames(mp4_bytes, max_seconds):
            batch_imgs.append(img)
            batch_secs.append(int(math.floor(rel)))
            n_frames += 1
            if len(batch_imgs) >= det.batch:
                flush()
                if n_frames % (det.batch * 20) == 0:
                    print(f"  {video_id}: {n_frames} frames, t={rel:.1f}s", file=sys.stderr, flush=True)
        flush()
        if n_frames == 0:
            raise ValueError("no decodable frames")
    except Exception as e:  # damaged video: keep what we have, then an error line
        try:
            flush()
        except Exception:
            pass
        error = f"{type(e).__name__}: {e}"[:200] or "could not decode video"

    for rec in agg.records(video_id):
        out.write(json.dumps(rec) + "\n")
    if error is not None:
        out.write(json.dumps({"video": video_id, "error": error}) + "\n")
    out.flush()
    return n_frames, agg.max_second + 1, error


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default=str(Path(__file__).with_name("yolo11l.pt")))
    args = ap.parse_args()

    files = sorted(Path(args.input).glob("*.parquet"))
    det = Detector(args.weights, args.batch)
    print(f"device={det.device} files={len(files)} classes={det.class_ids}", file=sys.stderr)

    t_start = time.perf_counter()
    total_frames = total_seconds = n_videos = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            if args.max_videos is not None and n_videos >= args.max_videos:
                break
            try:
                pf = pq.ParquetFile(path)
                n_rows = pf.metadata.num_rows
            except Exception as e:
                out.write(json.dumps({"video": f"{path.name}:0", "error": f"unreadable parquet: {e}"[:200]}) + "\n")
                continue
            row = 0
            try:
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                    for mp4 in batch.column(0).to_pylist():
                        if args.max_videos is not None and n_videos >= args.max_videos:
                            break
                        vid = f"{path.name}:{row}"
                        t = time.perf_counter()
                        nf, ns, err = process_video(vid, mp4, det, out, args.max_seconds)
                        dt = time.perf_counter() - t
                        total_frames += nf
                        total_seconds += ns
                        n_videos += 1
                        row += 1
                        print(f"{vid}: {nf} frames, {ns} s, {dt:.1f}s wall, {nf / max(dt, 1e-9):.1f} fps"
                              + (f", ERROR {err}" if err else ""), file=sys.stderr)
                    if args.max_videos is not None and n_videos >= args.max_videos:
                        break
            except Exception as e:  # parquet broke mid-file: error lines for remaining rows
                for r in range(row, n_rows):
                    out.write(json.dumps({"video": f"{path.name}:{r}", "error": f"row read failed: {e}"[:200]}) + "\n")

    wall = time.perf_counter() - t_start
    print(f"done: {n_videos} videos, {total_frames} frames, {total_seconds} video-seconds, "
          f"{wall:.1f}s wall ({total_frames / max(wall, 1e-9):.1f} fps), "
          f"inference {det.infer_time:.1f}s ({100 * det.infer_time / max(wall, 1e-9):.0f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
