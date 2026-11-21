"""Timing report: where the time goes and what throughput was reached.

All stage times are summed seconds. Columns:
  % of capacity  share of (parallel units x wall time) spent in that stage
  ms/frame       stage time per processed frame (summed over units)
  capacity fps   frames/s the whole pool could sustain if that stage were the only work
                 = frames / (stage time / units). The smallest capacity among the real work
                 stages is the ceiling that pool can reach.
GPU stage times come from CUDA events on each instance's stream (device time). With several
instances their spans overlap on one GPU, so per-instance sums can exceed wall time.
"""

from __future__ import annotations

from .decode import STAGES as DECODE_STAGES
from .infer import GPU_STAGES

DECODE_LABELS = {
    "load": "load mp4 from shared memory",
    "plan": "segment planning (packet scan)",
    "decode": "demux + decode (FFmpeg)",
    "convert": "colour convert to BGR / YUV copy",
    "resize_rotate": "rotate / letterbox resize",
    "copy": "copy into shared-memory slot",
    "send": "send frame messages",
    "slot_wait": "WAIT for free slot (GPU side behind)",
}
GPU_LABELS = {
    "stage_copy": "host staging memcpy (CPU)",
    "h2d": "host -> GPU transfer",
    "preprocess": "pad / colour / normalise",
    "forward": "model forward",
    "postprocess": "conf filter + NMS + counts",
}


def _row(label, total, units, wall, frames, capacity=True):
    cap = units * wall
    pct = 100 * total / cap if cap else 0.0
    ms = 1000 * total / frames if frames else 0.0
    fps = frames / (total / units) if total > 0 and units else float("inf")
    fps_s = f"{fps:>10.0f}" if capacity and fps != float("inf") else f"{'-':>10}"
    return f"  {label:<40}{total:>10.1f}{pct:>9.1f}%{ms:>10.2f}{fps_s}"


def build(m: dict) -> str:
    wall, frames = m["wall"], max(m["frames"], 1)
    nw, ng = m["workers"], m["gpu_instances"]
    L = []
    L.append("=" * 84)
    L.append("TIMING REPORT")
    L.append("=" * 84)
    L.append(f"wall time            {wall:10.1f} s   (model ready after {m['model_ready']:.1f} s)")
    L.append(f"frames processed     {m['frames']:10d}     -> {m['frames'] / wall:,.1f} fps overall")
    L.append(f"video processed      {m['seconds']:10d} s   -> {m['seconds'] / wall:,.1f}x real time")
    L.append(f"videos / segments    {m['videos']:>10d} / {m['segments']}   errors {m['errors']}")
    L.append(f"batches              {m['batches']:10d}     avg {m['frames'] / max(m['batches'], 1):.1f} of {m['batch']} frames")
    if m.get("gpu_name"):
        L.append(f"GPU                  {m['gpu_name']}   peak used {m['gpu_peak_used_gb']:.2f} GB "
                 f"(torch tensors {m['gpu_torch_peak_gb']:.2f} GB)")
    restarted = " (+%d restarted)" % m["restarted"] if m["restarted"] else ""
    L.append(f"setup                {nw} decode workers x {m['decode_threads']} FFmpeg thread(s){restarted}, "
             f"{ng} model instance(s), backend {m['backend']}, transfer {m['transfer']}, gpu_color={m['gpu_color']}")

    hdr = f"  {'stage':<40}{'total s':>10}{'% cap':>10}{'ms/frame':>10}{'cap. fps':>10}"
    # ---- decode workers
    d = m["decode"]
    L.append("")
    L.append(f"CPU DECODE WORKERS  (capacity {nw} x {wall:.1f} s = {nw * wall:.1f} s)")
    L.append(hdr)
    for s in DECODE_STAGES:
        L.append(_row(DECODE_LABELS[s], d.get(s, 0.0), nw, wall, frames, capacity=s != "slot_wait"))
    active = sum(d.get(s, 0.0) for s in DECODE_STAGES if s != "slot_wait")
    idle = max(nw * wall - active - d.get("slot_wait", 0.0), 0.0)
    L.append(_row("idle (no task / startup / shutdown)", idle, nw, wall, frames, capacity=False))
    L.append(_row("= active work", active, nw, wall, frames))

    # ---- inference
    g = m["gpu"]
    L.append("")
    L.append(f"MODEL INSTANCES  (capacity {ng} x {wall:.1f} s = {ng * wall:.1f} s; device time from CUDA events)")
    L.append(hdr)
    L.append(_row(GPU_LABELS["stage_copy"], g.get("stage_copy", 0.0), ng, wall, frames))
    for s in GPU_STAGES:
        L.append(_row(GPU_LABELS[s], g.get(s, 0.0), ng, wall, frames))
    L.append(_row("= instance threads busy (wall)", m["inference_busy"], ng, wall, frames))
    L.append(_row("idle (waiting for full batches)", max(ng * wall - m["inference_busy"], 0.0), ng, wall, frames,
                  capacity=False))

    # ---- main thread + reader
    mt, rd = m["main"], m["reader"]
    L.append("")
    L.append("MAIN THREAD / READER")
    L.append(hdr)
    L.append(_row("main: waiting for frame messages", mt.get("wait", 0.0), 1, wall, frames, capacity=False))
    L.append(_row("main: message handling + batching", mt.get("handle", 0.0), 1, wall, frames))
    L.append(_row("main: aggregate + write JSONL", mt.get("aggregate", 0.0), 1, wall, frames))
    L.append(_row("reader: Parquet read", rd.get("read", 0.0), 1, wall, frames))
    L.append(_row("reader: write mp4 to shared memory", rd.get("shm", 0.0), 1, wall, frames))
    L.append(_row("reader: blocked (workers busy)", rd.get("blocked", 0.0), 1, wall, frames, capacity=False))

    # ---- verdict
    dec_util = active / max(nw * wall, 1e-9)
    wait_share = d.get("slot_wait", 0.0) / max(nw * wall, 1e-9)
    inf_util = m["inference_busy"] / max(ng * wall, 1e-9)
    top_dec = max((s for s in DECODE_STAGES if s != "slot_wait"), key=lambda s: d.get(s, 0.0))
    top_gpu = max(GPU_STAGES, key=lambda s: g.get(s, 0.0))
    if inf_util > 0.8 and wait_share > 0.2:
        verdict = (f"MODEL-BOUND: instances busy {inf_util:.0%}, decoders wait for slots {wait_share:.0%} of the "
                   f"time. Largest GPU stage: {GPU_LABELS[top_gpu]}. Try INT8, more instances or larger batch.")
    elif dec_util > 0.75 and inf_util < 0.7:
        verdict = (f"DECODE-BOUND: decoders busy {dec_util:.0%}, instances only {inf_util:.0%}. Largest CPU stage: "
                   f"{DECODE_LABELS[top_dec]}. Try more workers or --gpu-color.")
    elif mt.get("handle", 0.0) / wall > 0.7:
        verdict = "MAIN-THREAD-BOUND: message handling dominates; increase GROUP size in pipeline/decode.py."
    else:
        verdict = (f"BALANCED / startup-dominated: decoders busy {dec_util:.0%}, instances busy {inf_util:.0%} "
                   f"(short runs are dominated by model loading and the tail of the last video).")
    L.append("")
    L.append("BOTTLENECK: " + verdict)
    L.append("=" * 84)
    return "\n".join(L)
