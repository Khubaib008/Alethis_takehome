"""CPU side: decoder worker processes. Every frame is decoded and sent on (nothing is skipped).

Each worker owns a block of shared memory split into fixed-size slots (640*640*3 bytes). Per frame:
  1. decode with PyAV (FFmpeg); skip packets the decoder rejects instead of abandoning the video,
  2. BGR path (default, pixel-identical to OpenCV): convert to BGR, apply rotation, and do the
     Ultralytics letterbox *resize* (cv2 INTER_LINEAR) if needed - padding is left to the GPU;
     YUV path (--gpu-color, opt-in): copy the raw yuv420p planes (half the bytes) and let the GPU
     convert; approximate (within ~3 grey levels of FFmpeg), used only for plain yuv420p frames,
  3. write the frame into a free slot; frame descriptors are sent to the main process in groups.

Long videos are split at keyframes into segments that several workers decode in parallel, so one
long video cannot keep a single worker (and the end of the run) busy while the rest sit idle.
Segment k decodes frames with start_k <= pts < end_k; decoding from a keyframe is bit-exact.

Must stay light: imported by spawned workers, so no torch / ultralytics here.
"""

from __future__ import annotations

import io
import math
import queue
import time
from multiprocessing import shared_memory

import av
import cv2
import numpy as np

IMGSZ = 640
SLOT_BYTES = IMGSZ * IMGSZ * 3
GROUP = 8                 # frames per message to the main process
# decode-worker timing stages (seconds, summed per segment and reported in the "d" message)
STAGES = ("load", "plan", "decode", "convert", "resize_rotate", "copy", "send", "slot_wait")
MAX_BAD_PACKETS = 200     # give up on a video after this many undecodable packets
HEARTBEAT_S = 5.0

# frame codes in descriptors
BGR, YUV709, YUV601 = 0, 1, 2


def frame_nbytes(code: int, nh: int, nw: int) -> int:
    return nh * nw * 3 if code == BGR else nh * nw * 3 // 2


def letterbox_params(h: int, w: int, imgsz: int = IMGSZ):
    """Same arithmetic as ultralytics LetterBox(auto=False, scaleup=True, center=True)."""
    r = min(imgsz / h, imgsz / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    top, left = round(dh - 0.1), round(dw - 0.1)
    return nw, nh, top, left


def rotate_frame(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the display-matrix rotation (PyAV reports degrees counter-clockwise)."""
    k = (rotation // 90) % 4
    return img if k == 0 else np.ascontiguousarray(np.rot90(img, k=k))


def _decode_tolerant(container, stream):
    """Decode packet by packet, skipping packets the decoder rejects (like ffmpeg/OpenCV do)."""
    bad = 0
    for packet in container.demux(stream):  # ends with an empty packet that flushes the decoder
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            bad += 1
            if bad > MAX_BAD_PACKETS:
                raise
            continue
        yield from frames


def _open(mp4_bytes: bytes, threads: int):
    container = av.open(io.BytesIO(mp4_bytes))
    if not container.streams.video:
        container.close()
        raise ValueError("no video stream")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = threads
    return container, stream


def iter_frames(mp4_bytes: bytes, threads: int, start=None, end=None):
    """Yield (absolute time in seconds, av.VideoFrame) for frames with start <= pts < end
    (pts in stream time_base; None = open). Frames without timestamps get one extrapolated."""
    container, stream = _open(mp4_bytes, threads)
    with container:
        if start is not None:
            container.seek(start, stream=stream, backward=True, any_frame=False)
        last_t = None
        for frame in _decode_tolerant(container, stream):
            pts = frame.pts
            if pts is None:
                if start is not None:  # cannot place it inside a segment
                    continue
                rate = float(stream.average_rate or 30)
                t = (last_t + 1.0 / rate) if last_t is not None else 0.0
            else:
                if start is not None and pts < start:
                    continue  # leading frames of an open GOP: belong to the previous segment
                if end is not None and pts >= end:
                    break     # decoder outputs in presentation order: segment complete
                t = float(frame.time)
            last_t = t
            yield t, frame


def plan_segments(mp4_bytes: bytes, seg_frames: int):
    """Split points for long videos. Returns [(start_pts, end_pts), ...] ([(None, None)] = whole).
    Cheap estimate first (container duration x frame rate); only long videos get a packet scan."""
    if seg_frames <= 0:
        return [(None, None)], None
    try:
        with av.open(io.BytesIO(mp4_bytes)) as c:
            if not c.streams.video:
                return [(None, None)], None
            v = c.streams.video[0]
            est = (c.duration or 0) / 1e6 * float(v.average_rate or 30)
            if est <= seg_frames * 1.5:
                return [(None, None)], None
            pts, keys = [], []
            for p in c.demux(v):
                if p.size == 0 or p.pts is None:
                    continue
                pts.append(p.pts)
                if p.is_keyframe:
                    keys.append(p.pts)
            tb = float(v.time_base)
    except Exception:
        return [(None, None)], None
    n = len(pts)
    if n <= seg_frames * 1.5 or len(keys) < 2:
        return [(None, None)], None
    pts_sorted = np.sort(np.asarray(pts))
    keys = np.unique(np.asarray(keys))
    k = math.ceil(n / seg_frames)
    bounds = []
    for j in range(1, k):
        target = pts_sorted[j * n // k]
        i = np.searchsorted(keys, target, side="right") - 1   # last keyframe at or before target
        if i <= 0:
            continue
        b = int(keys[i])
        if not bounds or b > bounds[-1]:
            bounds.append(b)
    if not bounds:
        return [(None, None)], None
    edges = [None] + bounds + [None]
    return list(zip(edges[:-1], edges[1:])), float(pts_sorted[0]) * tb


def _load(name, size) -> bytes:
    """Copy a video's mp4 bytes out of the reader's shared-memory block (None = empty video)."""
    if not name or not size:
        return b""
    shm = shared_memory.SharedMemory(name=name)
    try:
        return bytes(shm.buf[:size])
    finally:
        shm.close()


def worker_main(wid, task_q, seg_q, result_q, free_q, shm_name, n_slots, opts, stop_event):
    """Tasks:  ("v", video_id, shm_name, size)                        whole video from the reader
               ("s", video_id, seg, shm_name, size, start, end, t0)  one segment of a long video
    (mp4 bytes live in the reader's shared memory; tasks carry only its name)
    Messages to result_q:
      ("s", wid, vid, seg)                      started a segment
      ("p", vid, n_segments)                    how many segments the video was split into
      ("t0", vid, t)                            time of the first decoded frame (segment 0)
      ("F", wid, vid, [(slot, t, code, nh, nw, top, left), ...], {stage: seconds since last msg})
      ("d", wid, vid, seg, n_frames, error, {stage: seconds})      segment finished + timings
      ("h", wid)                                heartbeat while waiting for free slots
      ("x", wid)                                exiting
    """
    cv2.setNumThreads(1)
    threads = opts["decode_threads"]
    max_seconds = opts["max_seconds"]
    gpu_color = opts["gpu_color"]
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = np.ndarray((n_slots, SLOT_BYTES), dtype=np.uint8, buffer=shm.buf)
    local_free: list[int] = []
    group: list[tuple] = []
    cur = {"vid": None}
    pc = time.perf_counter
    T = dict.fromkeys(STAGES, 0.0)  # per-segment stage timings, sent with the "d" message

    def take_timing() -> dict:
        d = dict(T)
        for k in T:
            T[k] = 0.0
        return d

    def flush():
        if group:
            t = pc()
            result_q.put(("F", wid, cur["vid"], list(group), take_timing()))
            group.clear()
            T["send"] += pc() - t  # reported with the next message

    def get_slot() -> int:
        while not local_free:
            flush()  # never sit on unsent frames while waiting (would deadlock)
            t = pc()
            try:
                local_free.extend(free_q.get(timeout=HEARTBEAT_S))
            except queue.Empty:
                result_q.put(("h", wid))
            T["slot_wait"] += pc() - t
        return local_free.pop()

    def emit(t: float, frame) -> None:
        t_a = pc()
        rot = int(getattr(frame, "rotation", 0) or 0)
        h, w = frame.height, frame.width
        if rot % 180:
            h, w = w, h
        nw, nh, top, left = letterbox_params(h, w)
        code = BGR
        if (gpu_color and rot == 0 and (w, h) == (nw, nh) and w % 2 == 0 and h % 2 == 0
                and frame.format.name == "yuv420p" and frame.color_range != 2):
            cs = frame.colorspace
            code = YUV709 if cs == 1 else (YUV601 if cs in (2, 5, 6) else BGR)
        if code == BGR:
            img = frame.to_ndarray(format="bgr24")
            t_b = pc()
            T["convert"] += t_b - t_a
            if rot:
                img = rotate_frame(img, rot)
            if (w, h) != (nw, nh):
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            data = np.ascontiguousarray(img).reshape(-1)
            T["resize_rotate"] += pc() - t_b
        else:
            data = np.ascontiguousarray(frame.to_ndarray(format="yuv420p")).reshape(-1)
            T["convert"] += pc() - t_a
        slot = get_slot()
        t_c = pc()
        slots[slot, :data.size] = data
        group.append((slot, t, code, nh, nw, top, left))
        T["copy"] += pc() - t_c
        if len(group) >= GROUP:
            flush()

    def run_segment(vid, seg, data, start, end, t0_hint):
        result_q.put(("s", wid, vid, seg))
        cur["vid"] = vid
        n, err = 0, None
        try:
            if not data:
                raise ValueError("empty mp4 bytes")
            t0 = t0_hint
            it = iter_frames(data, threads, start, end)
            while True:
                t_d = pc()
                try:
                    t, frame = next(it)
                except StopIteration:
                    T["decode"] += pc() - t_d
                    break
                T["decode"] += pc() - t_d
                if t0 is None:
                    t0 = t
                    if seg == 0:
                        result_q.put(("t0", vid, t))
                if max_seconds is not None and t - t0 >= max_seconds:
                    break
                emit(t, frame)
                n += 1
            if n == 0 and seg == 0:
                raise ValueError("no decodable frames")
        except Exception as e:  # damaged video: report, keep going
            err = f"{type(e).__name__}: {e}"[:200]
        flush()
        result_q.put(("d", wid, vid, seg, n, err, take_timing()))

    try:
        while not stop_event.is_set():
            try:
                task = seg_q.get_nowait()          # segments of long videos first
            except queue.Empty:
                try:
                    task = task_q.get(timeout=0.2)
                except queue.Empty:
                    continue
            if task[0] == "v":
                _, vid, name, size = task
                result_q.put(("s", wid, vid, 0))  # announce before planning, so a crash here is noticed
                t_p = pc()
                data = _load(name, size)
                T["load"] += pc() - t_p
                t_p = pc()
                segs, t0_hint = plan_segments(data, opts["seg_frames"]) if data else ([(None, None)], None)
                T["plan"] += pc() - t_p
                result_q.put(("p", vid, len(segs)))
                for i in range(1, len(segs)):
                    seg_q.put(("s", vid, i, name, size, segs[i][0], segs[i][1], t0_hint))
                run_segment(vid, 0, data, segs[0][0], segs[0][1], None)
            else:
                _, vid, seg, name, size, start, end, t0_hint = task
                t_p = pc()
                data = _load(name, size)
                T["load"] += pc() - t_p
                run_segment(vid, seg, data, start, end, t0_hint)
            data = None
    finally:
        del slots
        shm.close()
        result_q.put(("x", wid))
