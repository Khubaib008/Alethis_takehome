"""Extract INT8 calibration frames from the FineVideo Parquet files (run once, commit the output).

    python make_calibration.py --input finevideo/data --out calib/images --per-file 9

Picks random videos in every Parquet file and random timestamps in each, decodes those frames
(rotation applied, same decoder as run.py) and saves them as JPEGs. prepare.py points the
TensorRT INT8 calibrator at this folder, so the engine is calibrated on the footage it will see.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from pipeline.decode import rotate_frame


def frame_at(mp4: bytes, t: float):
    with av.open(io.BytesIO(mp4)) as c:
        v = c.streams.video[0]
        start = v.start_time or 0
        c.seek(int(start + t / float(v.time_base)), stream=v)
        for fr in c.decode(v):
            if fr.time is not None and fr.time >= t + float(start * v.time_base):
                img = fr.to_ndarray(format="bgr24")
                rot = int(getattr(fr, "rotation", 0) or 0)
                return rotate_frame(img, rot) if rot else img
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="finevideo/data")
    ap.add_argument("--out", default="calib/images")
    ap.add_argument("--per-file", type=int, default=9, help="frames per Parquet file")
    ap.add_argument("--videos-per-file", type=int, default=3)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(Path(args.input).glob("*.parquet")):
        try:
            col = pq.read_table(path, columns=["mp4"]).column("mp4")
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue
        rows = rng.choice(len(col), size=min(args.videos_per_file, len(col)), replace=False)
        per_video = max(1, args.per_file // len(rows))
        for row in rows:
            mp4 = col[int(row)].as_py()
            try:
                with av.open(io.BytesIO(mp4)) as c:
                    dur = (c.duration or 0) / 1e6
            except Exception:
                continue
            for t in np.sort(rng.uniform(0, max(dur - 1, 0.1), per_video)):
                try:
                    img = frame_at(mp4, float(t))
                except Exception:
                    img = None
                if img is None:
                    continue
                name = f"{path.stem}_r{int(row)}_t{t:07.2f}.jpg"
                cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                n += 1
        print(f"{path.name}: total {n} frames", flush=True)
    size = sum(f.stat().st_size for f in out.glob("*.jpg")) / 2**20
    print(f"wrote {n} calibration frames ({size:.1f} MB) to {out.resolve()}")


if __name__ == "__main__":
    main()
