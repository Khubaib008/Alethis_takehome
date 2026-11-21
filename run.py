"""Video indexing at speed: YOLO11l on every frame, aggregated per second.

    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline (every frame of every video is decoded and run through the model)
  reader thread      : reads Parquet rows (mp4 bytes) -> task queue
  decode workers     : N CPU processes; long videos are split at keyframes into segments that
                       several workers decode in parallel; frames go to shared-memory slots and
                       are announced to the main process in groups
  main thread        : batches frames, keeps per-video state, watchdog (timeouts, respawn),
                       vectorised per-second aggregation, buffered JSONL writing
  inference threads  : M model instances, each with its own CUDA stream; frames DMA'd straight from
                       pinned shared memory -> GPU (optional YUV->BGR), pad 640x640, normalise,
                       YOLO (async TensorRT) -> conf filter -> batched NMS -> per-frame counts
N, M, FFmpeg threads and slots are planned from the machine (pipeline/resources.py).

Development options:
    --max-videos N        stop after N videos       --max-seconds S    first S seconds of each video
    --workers N           decode processes          --gpu-instances M  model instances
    --decode-threads T    FFmpeg threads per worker --batch B          frames per model call
    --weights PATH        .pt or .engine            --fp32             disable FP16 (.pt on GPU)
    --seg-frames F        split videos longer than ~1.5*F frames (0 = never split)
    --gpu-color           send YUV planes, convert on GPU (faster on few CPUs; not bit-exact)
    --worker-timeout S    restart a decode worker silent for S seconds (hung on a bad file)
    --fake-model          no model, zero detections (pipeline test / decode benchmark without a GPU)
    --log-every S         progress line interval     --metrics-json P  save the timing report data

At the end a timing report is printed (stderr): time per stage for decode workers, model instances,
main thread and reader, overall fps, x-real-time, GPU memory and a bottleneck verdict.
"""

from __future__ import annotations

# Keep top-level imports light: spawned decode workers re-import this module.
import argparse
import os
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count_videos(files, max_videos):
    """Rows per file from Parquet footers only (cheap). Unreadable files count as 1."""
    import pyarrow.parquet as pq

    n = 0
    for f in files:
        try:
            n += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            n += 1
    return n if max_videos is None else min(n, max_videos)


class Reader(threading.Thread):
    """Feeds ("v", video_id, shm_name, size) tasks; unreadable files/rows go to `errors`.

    Each video's mp4 bytes are written once into their own shared-memory block, so tasks (and the
    segments of long videos) carry only a name instead of tens of MB of pickled bytes. The main
    process owns the blocks and frees each one when its video is finalized."""

    def __init__(self, files, task_q, max_videos):
        super().__init__(daemon=True)
        self.files, self.task_q, self.max_videos = files, task_q, max_videos
        self.errors: deque = deque()
        self.sent = 0
        self.finished = threading.Event()
        self.shms: dict = {}
        self.timing = {"read": 0.0, "shm": 0.0, "blocked": 0.0}

    def submit(self, vid, mp4):
        from multiprocessing import shared_memory

        if not mp4:
            task = ("v", vid, None, 0)
        else:
            t = time.perf_counter()
            shm = shared_memory.SharedMemory(create=True, size=len(mp4))
            shm.buf[:len(mp4)] = mp4
            self.shms[vid] = shm
            task = ("v", vid, shm.name, len(mp4))
            self.timing["shm"] += time.perf_counter() - t
        t = time.perf_counter()
        self.task_q.put(task)  # blocks while all workers are busy (bounded queue)
        self.timing["blocked"] += time.perf_counter() - t

    def release(self, vid):
        shm = self.shms.pop(vid, None)
        if shm is not None:
            shm.close()
            shm.unlink()

    def run(self):
        import pyarrow.parquet as pq

        mv = self.max_videos
        try:
            for path in self.files:
                if mv is not None and self.sent >= mv:
                    break
                try:
                    pf = pq.ParquetFile(path)
                    n_rows = pf.metadata.num_rows
                except Exception as e:
                    self.errors.append((f"{path.name}:0", f"unreadable parquet: {e}"[:200]))
                    continue
                row = 0
                try:
                    batches = pf.iter_batches(batch_size=1, columns=["mp4"])
                    while True:
                        t = time.perf_counter()
                        batch = next(batches, None)
                        rows = batch.column(0).to_pylist() if batch is not None else None
                        self.timing["read"] += time.perf_counter() - t
                        if rows is None:
                            break
                        for mp4 in rows:
                            if mv is not None and self.sent >= mv:
                                break
                            self.submit(f"{path.name}:{row}", mp4)
                            row += 1
                            self.sent += 1
                        if mv is not None and self.sent >= mv:
                            break
                except Exception as e:  # file broke mid-way: error lines for the rows not yet read
                    for r in range(row, n_rows):
                        self.errors.append((f"{path.name}:{r}", f"row read failed: {e}"[:200]))
        finally:
            self.finished.set()


def inference_thread(det, batch_q, done_q, pools, busy):
    """One model instance. Items: (wid, vid, slot, t, code, nh, nw, top, left).
    * pinned shared memory (CUDA): GPU copies straight from the slots; slots freed after the batch.
    * otherwise: memcpy into this instance's staging buffer, free the slots, then run."""
    from pipeline.decode import frame_nbytes

    while True:
        items = batch_q.get()
        if items is None:
            break
        t = time.perf_counter()
        try:
            by_worker = defaultdict(list)
            if all(pools[it[0]].host is not None for it in items):
                frames = []
                for (wid, vid, slot, ft, code, nh, nw, top, left) in items:
                    nb = frame_nbytes(code, nh, nw)
                    frames.append((pools[wid].host[slot, :nb], code, nh, nw, top, left))
                    by_worker[wid].append(slot)
                counts = det.run_shared(frames)
            else:
                t_s = time.perf_counter()
                off, metas = 0, []
                for (wid, vid, slot, ft, code, nh, nw, top, left) in items:
                    nb = frame_nbytes(code, nh, nw)
                    det.stage(off, pools[wid].view[slot, :nb])
                    metas.append((off, code, nh, nw, top, left))
                    by_worker[wid].append(slot)
                    off += nb
                for wid, sl in by_worker.items():  # staged: slots can be reused right away
                    pools[wid].free_q.put(sl)
                by_worker.clear()
                busy["stage_copy"] += time.perf_counter() - t_s
                counts = det.run(metas, off)
            for wid, sl in by_worker.items():      # one put per worker per batch
                pools[wid].free_q.put(sl)
            done_q.put((items, counts))
        except BaseException as e:  # surface to the main thread
            done_q.put((items, e))
        busy["busy"] += time.perf_counter() - t


class Pool:
    """One decode worker: process + its shared-memory slot block + free-slot queue."""

    def __init__(self, ctx, wid, n_slots, task_q, seg_q, result_q, opts, stop_event):
        import numpy as np
        from multiprocessing import shared_memory

        from pipeline.decode import SLOT_BYTES, worker_main

        self.wid = wid
        self.shm = shared_memory.SharedMemory(create=True, size=n_slots * SLOT_BYTES)
        self.view = np.ndarray((n_slots, SLOT_BYTES), dtype=np.uint8, buffer=self.shm.buf)
        self.host = None  # torch tensor over the block once page-locked for CUDA
        self.free_q = ctx.Queue()
        self.free_q.put(list(range(n_slots)))
        self.proc = ctx.Process(target=worker_main, daemon=True,
                                args=(wid, task_q, seg_q, result_q, self.free_q, self.shm.name, n_slots,
                                      opts, stop_event))
        self.proc.start()
        self.alive = True       # False once exited / killed
        self.last_seen = time.perf_counter()

    def close(self):
        self.host = None
        del self.view
        self.shm.close()
        self.shm.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--decode-threads", type=int, default=None)
    ap.add_argument("--gpu-instances", type=int, default=None)
    ap.add_argument("--slots", type=int, default=24, help="minimum shared-memory frame slots per worker")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--seg-frames", type=int, default=2400)
    ap.add_argument("--gpu-color", action="store_true")
    ap.add_argument("--worker-timeout", type=float, default=120.0)
    ap.add_argument("--fake-model", action="store_true",
                    help="no model: zero detections; tests the pipeline / measures CPU decode throughput")
    ap.add_argument("--log-every", type=float, default=30.0, help="seconds between progress lines")
    ap.add_argument("--metrics-json", default=None, help="also write the timing metrics to this JSON file")
    args = ap.parse_args()

    import multiprocessing as mp

    import numpy as np
    import orjson

    from pipeline.aggregate import aggregate, records
    from pipeline.resources import make_plan

    t_start = time.perf_counter()
    files = sorted(Path(args.input).glob("*.parquet"))

    # ---- weights / batch (a static engine fixes the batch size)
    weights = args.weights
    engine = HERE / "yolo11l.engine"
    if weights is None and not args.fake_model:
        weights = str(engine) if engine.exists() else str(HERE / "yolo11l.pt")
    batch = args.batch
    if weights and weights.endswith(".engine") and Path(weights).with_suffix(".json").exists():
        import json
        batch = json.loads(Path(weights).with_suffix(".json").read_text())["batch"]

    plan = make_plan(n_videos=count_videos(files, args.max_videos), fake_model=args.fake_model,
                     workers=args.workers, decode_threads=args.decode_threads, slots=args.slots,
                     gpu_instances=args.gpu_instances, batch=batch)
    print(plan.describe(), file=sys.stderr, flush=True)
    n_gpu = plan.gpu_instances

    # ---- decode workers + reader start first, so decoding overlaps model loading.
    # Workers are single-threaded apart from FFmpeg: stop BLAS/OpenMP pools from oversubscribing.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=plan.workers * 2)
    seg_q = ctx.Queue()
    result_q = ctx.Queue()
    stop_event = ctx.Event()
    opts = {"decode_threads": plan.decode_threads, "max_seconds": args.max_seconds,
            "gpu_color": args.gpu_color, "seg_frames": args.seg_frames}
    pools: dict[int, Pool] = {}
    next_wid = [0]

    def spawn():
        wid = next_wid[0]
        next_wid[0] += 1
        pools[wid] = Pool(ctx, wid, plan.slots, task_q, seg_q, result_q, opts, stop_event)
        return pools[wid]

    for _ in range(plan.workers):
        spawn()
    reader = Reader(files, task_q, args.max_videos)
    reader.start()

    # ---- model instances, one thread each
    import torch
    from pipeline.infer import FakeDetector, GpuDetector, register_host_memory, unregister_host_memory

    cuda = torch.cuda.is_available() and not args.fake_model
    if cuda:
        torch.set_num_threads(min(4, torch.get_num_threads()))  # CPU torch work is tiny on the GPU path
    if args.fake_model:
        dets = [FakeDetector(batch) for _ in range(n_gpu)]
        weights = "FAKE (zero detections)"
    else:
        dets = [GpuDetector(weights, batch, half=False if args.fp32 else None) for _ in range(n_gpu)]

    def pin(pool):
        if cuda:
            h = register_host_memory([pool.view])
            pool.host = h[0] if h else None

    for p in pools.values():
        pin(p)
    B = dets[0].batch
    batch_q: queue.Queue = queue.Queue(maxsize=n_gpu * 2)
    done_q: queue.Queue = queue.Queue()
    busy = [{"busy": 0.0, "stage_copy": 0.0} for _ in range(n_gpu)]
    inf_threads = [threading.Thread(target=inference_thread, daemon=True, args=(d, batch_q, done_q, pools, busy[i]))
                   for i, d in enumerate(dets)]
    for t in inf_threads:
        t.start()
    transfer = "zero-copy pinned shared memory" if any(p.host is not None for p in pools.values()) else "staging copy"
    model_ready = time.perf_counter() - t_start
    print(f"device={dets[0].device} backend={dets[0].backend} dtype={dets[0].dtype} weights={Path(weights).name} "
          f"batch={B} transfer={transfer} gpu_color={args.gpu_color} model_ready={model_ready:.1f}s",
          file=sys.stderr, flush=True)

    # ---- timing accumulators
    dec_t: dict[str, float] = defaultdict(float)     # decode-worker stages (summed over workers)
    main_t = {"wait": 0.0, "handle": 0.0}
    gpu_mem = {"min_free": None, "total": 0}

    def sample_gpu_mem():
        if cuda:
            free, total = torch.cuda.mem_get_info()
            gpu_mem["total"] = total
            gpu_mem["min_free"] = free if gpu_mem["min_free"] is None else min(gpu_mem["min_free"], free)

    sample_gpu_mem()

    # ---- state (main thread only)
    videos: dict[str, dict] = {}
    current: dict[int, tuple] = {}            # wid -> (vid, seg)
    seg_recv: dict[tuple, int] = defaultdict(int)  # (wid, vid) -> frames received from that worker
    pending: list[tuple] = []
    in_flight = 0
    finalized = [0]
    stats = defaultdict(float)
    out = open(args.output, "wb", buffering=1 << 20)

    def state(vid):
        st = videos.get(vid)
        if st is None:
            st = videos[vid] = {"times": [], "counts": [], "t0": None, "n_seg": None, "segs_done": 0,
                                "received": 0, "processed": 0, "expected": 0, "errors": []}
        return st

    def write_error(vid, err):
        out.write(orjson.dumps({"video": vid, "error": err}) + b"\n")
        stats["errors"] += 1

    def finalize(vid):
        st = videos.get(vid)
        if st is None or st["n_seg"] is None or st["segs_done"] < st["n_seg"] or st["processed"] < st["expected"]:
            return
        t = time.perf_counter()
        if st["times"]:
            times = np.asarray(st["times"], dtype=np.float64)
            t0 = st["t0"] if st["t0"] is not None else times.min()
            secs = np.floor(times - t0).astype(np.int64)
            counts = np.vstack(st["counts"]).astype(np.int64)
            ok = secs >= 0
            present, people = aggregate(secs[ok], counts[ok])
            for rec in records(vid, present, people):
                out.write(orjson.dumps(rec) + b"\n")
            stats["seconds"] += len(people)
        if st["errors"]:
            write_error(vid, "; ".join(st["errors"])[:200])
        stats["videos"] += 1
        stats["agg_s"] += time.perf_counter() - t
        finalized[0] += 1
        del videos[vid]
        reader.release(vid)  # free the video's mp4 bytes

    def handle(msg):
        kind = msg[0]
        if kind == "F":
            _, wid, vid, entries, timing = msg
            for k, v in timing.items():
                dec_t[k] += v
            pools[wid].last_seen = time.perf_counter()
            pending.extend((wid, vid) + e for e in entries)
            state(vid)["received"] += len(entries)
            seg_recv[(wid, vid)] += len(entries)
        elif kind == "s":
            _, wid, vid, seg = msg
            pools[wid].last_seen = time.perf_counter()
            current[wid] = (vid, seg)
            state(vid)
        elif kind == "p":
            state(msg[1])["n_seg"] = msg[2]
            stats["segments"] += msg[2]
            finalize(msg[1])
        elif kind == "t0":
            state(msg[1])["t0"] = msg[2]
        elif kind == "d":
            _, wid, vid, seg, n, err, timing = msg
            pools[wid].last_seen = time.perf_counter()
            for k, v in timing.items():
                dec_t[k] += v
            current.pop(wid, None)
            seg_recv.pop((wid, vid), None)
            st = state(vid)
            st["segs_done"] += 1
            st["expected"] += n
            if err:
                st["errors"].append(err if st["n_seg"] in (None, 1) else f"segment {seg}: {err}")
            finalize(vid)
        elif kind == "h":
            pools[msg[1]].last_seen = time.perf_counter()
        elif kind == "x":
            pools[msg[1]].alive = False

    def collect_results(block: bool = False):
        nonlocal in_flight
        while in_flight:
            try:
                items, counts = done_q.get(timeout=0.5) if block else done_q.get_nowait()
            except queue.Empty:
                return
            block = False
            in_flight -= 1
            if isinstance(counts, BaseException):
                raise counts
            stats["batches"] += 1
            stats["frames"] += len(items)
            touched = set()
            for i, item in enumerate(items):
                st = videos.get(item[1])
                if st is None:  # late frame from a crashed worker whose video was already closed
                    continue
                st["times"].append(item[3])
                st["counts"].append(counts[i])
                st["processed"] += 1
                touched.add(item[1])
            for vid in touched:
                finalize(vid)

    def submit(items) -> bool:
        nonlocal in_flight
        try:
            batch_q.put_nowait(items)
        except queue.Full:
            return False
        in_flight += 1
        return True

    def watchdog():
        """Restart decode workers that crashed or hung; their segment is closed with an error."""
        now = time.perf_counter()
        for wid, p in list(pools.items()):
            if not p.alive:
                continue
            dead = not p.proc.is_alive()
            hung = not dead and wid in current and now - p.last_seen > args.worker_timeout
            if not (dead or hung):
                continue
            if hung:
                p.proc.kill()
                p.proc.join(timeout=5)
            p.alive = False
            reason = "decoder timed out" if hung else f"decoder crashed (exit {p.proc.exitcode})"
            print(f"worker {wid}: {reason}; restarting", file=sys.stderr, flush=True)
            if wid in current:
                vid, seg = current.pop(wid)
                st = state(vid)
                if st["n_seg"] is None:
                    st["n_seg"] = 1
                st["segs_done"] += 1
                st["expected"] += seg_recv.pop((wid, vid), 0)
                st["errors"].append(reason if st["n_seg"] == 1 else f"segment {seg}: {reason}")
                finalize(vid)
            if not stop_event.is_set():
                pin(spawn())

    def flush_reader_errors():
        while reader.errors:  # not counted in reader.sent, so they do not affect termination
            write_error(*reader.errors.popleft())

    # ---- main loop
    last_log = last_watch = time.perf_counter()
    pc = time.perf_counter
    try:
        while True:
            t_iter = pc()
            collect_results()
            got = False
            t_w = pc()
            try:
                msg = result_q.get(timeout=0.005 if pending else 0.1)
                main_t["wait"] += pc() - t_w
                handle(msg)
                got = True
                while len(pending) < B:
                    handle(result_q.get_nowait())
            except queue.Empty:
                if not got:
                    main_t["wait"] += pc() - t_w
            flush_reader_errors()
            while len(pending) >= B and submit(pending[:B]):
                del pending[:B]
            if pending and len(pending) < B and not got and in_flight < n_gpu:
                if submit(list(pending)):  # input went quiet and a model is idle: partial batch
                    pending.clear()
            now = time.perf_counter()
            if now - last_watch > 1.0:
                watchdog()
                last_watch = now
            if reader.finished.is_set() and finalized[0] >= reader.sent and not pending and not in_flight:
                break
            if now - last_log > args.log_every:
                el = now - t_start
                inf_util = sum(b["busy"] for b in busy) / (el * n_gpu)
                dec_active = sum(v for k, v in dec_t.items() if k != "slot_wait")
                sample_gpu_mem()
                mem = ""
                if gpu_mem["min_free"] is not None:
                    mem = f" gpu_mem_peak={(gpu_mem['total'] - gpu_mem['min_free']) / 2**30:.1f}GB"
                print(f"[{el:6.0f}s] videos {int(stats['videos'])}/{reader.sent} | frames {int(stats['frames']):,} "
                      f"| {stats['frames'] / el:,.0f} fps | {stats['seconds'] / el:.1f}x realtime "
                      f"| decode busy {100 * dec_active / (el * len(pools)):.0f}% "
                      f"slot-wait {100 * dec_t['slot_wait'] / (el * len(pools)):.0f}% "
                      f"| model busy {100 * inf_util:.0f}% | pending {len(pending)} in-flight {in_flight}{mem}",
                      file=sys.stderr, flush=True)
                last_log = now
            main_t["handle"] += pc() - t_iter
        flush_reader_errors()
    finally:
        stop_event.set()
        out.close()
        for _ in inf_threads:
            try:
                batch_q.put_nowait(None)
            except queue.Full:
                pass
        for t in inf_threads:
            t.join(timeout=5)
        for p in pools.values():
            p.proc.join(timeout=3)
            if p.proc.is_alive():
                p.proc.terminate()
        for p in pools.values():
            if p.host is not None:
                unregister_host_memory([p.host])
            p.close()
        for vid in list(reader.shms):
            reader.release(vid)

    wall = time.perf_counter() - t_start
    sample_gpu_mem()
    gpu_t: dict[str, float] = defaultdict(float)
    for d in dets:
        for k, v in d.clock.totals.items():
            gpu_t[k] += v
    gpu_t["stage_copy"] = sum(b["stage_copy"] for b in busy)
    metrics = {
        "wall": wall, "model_ready": model_ready,
        "frames": int(stats["frames"]), "seconds": int(stats["seconds"]), "videos": int(stats["videos"]),
        "segments": int(stats["segments"]), "errors": int(stats["errors"]), "batches": int(stats["batches"]),
        "batch": B, "workers": plan.workers, "decode_threads": plan.decode_threads,
        "restarted": next_wid[0] - plan.workers, "gpu_instances": n_gpu, "backend": dets[0].backend,
        "transfer": transfer, "gpu_color": args.gpu_color,
        "decode": dict(dec_t), "gpu": dict(gpu_t), "inference_busy": sum(b["busy"] for b in busy),
        "main": {"wait": main_t["wait"], "handle": max(main_t["handle"] - main_t["wait"] - stats["agg_s"], 0.0),
                 "aggregate": stats["agg_s"]},
        "reader": dict(reader.timing),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_peak_used_gb": (gpu_mem["total"] - gpu_mem["min_free"]) / 2**30 if gpu_mem["min_free"] is not None else 0,
        "gpu_torch_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else 0,
    }
    from pipeline.report import build
    print(build(metrics), file=sys.stderr, flush=True)
    if args.metrics_json:
        import json
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
