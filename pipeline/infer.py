"""GPU side: colour conversion (optional), padding, normalisation, model, confidence filter, NMS
and per-frame counts.

Everything after the host->device copy stays on the device; only an (n, 5) int tensor of per-frame
counts comes back. Works on CPU too (slowly) so it can be tested without a GPU.

Matches ultralytics predict defaults:
  * fixed 640x640 letterbox, padding value 114, BGR -> RGB, /255
  * per box: best class over all 80 classes, keep if score > 0.4 (strict, like Ultralytics),
    then keep only the 5 target classes (filtering *after* argmax, as `classes=` does)
  * class-aware NMS, IoU 0.7
Not replicated: max_det=300 / max_nms=30000 caps (only matter for >300 boxes in one frame).

Model backends: our async TensorRT runner (pipeline/trt.py) for .engine files, falling back to
Ultralytics AutoBackend; AutoBackend for .pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

from .decode import BGR, IMGSZ, SLOT_BYTES, YUV709, frame_nbytes

TARGETS = ["person", "car", "dog", "laptop", "cell phone"]
CONF = 0.4
IOU = 0.7
PAD_VALUE = 114  # Ultralytics letterbox padding colour (gray). 0 would be black.

# limited-range YCbCr -> RGB coefficients (Kr, Kb) per frame code
_KRKB = {YUV709: (0.2126, 0.0722), 2: (0.299, 0.114)}


def yuv420_to_bgr(buf: torch.Tensor, h: int, w: int, code: int) -> torch.Tensor:
    """buf: flat uint8 yuv420p planes (Y, U, V). Returns (h, w, 3) uint8 BGR.
    Limited range, chroma repeated 2x2 (closest match to FFmpeg's unscaled converter, within
    ~3 grey levels; not bit-exact - only used with --gpu-color)."""
    kr, kb = _KRKB[code]
    kg = 1.0 - kr - kb
    hw, q = h * w, (h // 2) * (w // 2)
    y = (buf[:hw].view(h, w).float() - 16.0) * (255.0 / 219.0)
    u = buf[hw:hw + q].view(h // 2, w // 2).float() - 128.0
    v = buf[hw + q:hw + 2 * q].view(h // 2, w // 2).float() - 128.0
    u = u.repeat_interleave(2, 0).repeat_interleave(2, 1)
    v = v.repeat_interleave(2, 0).repeat_interleave(2, 1)
    s = 255.0 / 224.0
    b = y + (s * 2 * (1 - kb)) * u
    g = y - (s * 2 * kb * (1 - kb) / kg) * u - (s * 2 * kr * (1 - kr) / kg) * v
    r = y + (s * 2 * (1 - kr)) * v
    return torch.stack((b, g, r), -1).add_(0.5).floor_().clamp_(0, 255).to(torch.uint8)


def register_host_memory(arrays) -> list:
    """Page-lock (pin) the decoders' shared-memory blocks for CUDA, once per process, so the GPU
    can DMA frames straight out of them. Returns host tensors over the blocks, or [] if unavailable
    (no CUDA, or registration refused) - callers then fall back to the staging-buffer copy."""
    if not torch.cuda.is_available():
        return []
    cudart = torch.cuda.cudart()
    done = []
    for a in arrays:
        err = cudart.cudaHostRegister(a.ctypes.data, a.nbytes, 0)
        if int(err) != 0:
            unregister_host_memory(done)
            return []
        done.append(a)
    return [torch.from_numpy(a) for a in arrays]


def unregister_host_memory(arrays) -> None:
    if not arrays or not torch.cuda.is_available():
        return
    cudart = torch.cuda.cudart()
    for a in arrays:
        ptr = a.data_ptr() if isinstance(a, torch.Tensor) else a.ctypes.data
        cudart.cudaHostUnregister(ptr)


GPU_STAGES = ("h2d", "preprocess", "forward", "postprocess")


class StageClock:
    """Times the GPU stages of each batch. On CUDA it records events on the instance's stream
    (true device time, no extra synchronisation: read after the batch's final sync); on CPU it
    uses wall-clock marks. Totals in seconds per stage, plus the device-side batch span."""

    def __init__(self, stream):
        from collections import defaultdict
        self.stream = stream
        self.totals = defaultdict(float)
        if stream is not None:
            self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(GPU_STAGES) + 1)]
        else:
            self.t = [0.0] * (len(GPU_STAGES) + 1)

    def mark(self, i: int) -> None:
        if self.stream is not None:
            self.ev[i].record(self.stream)
        else:
            import time
            self.t[i] = time.perf_counter()

    def collect(self) -> None:
        """Call after the stream has been synchronised (the counts copy does that)."""
        for k, name in enumerate(GPU_STAGES):
            if self.stream is not None:
                self.totals[name] += self.ev[k].elapsed_time(self.ev[k + 1]) / 1000.0
            else:
                self.totals[name] += self.t[k + 1] - self.t[k]


class FakeDetector:
    """Drop-in stand-in with no model: stages frames like the real one, returns zero counts.

    Used with `run.py --fake-model` to test the pipeline and measure CPU throughput
    (Parquet read + decode + shared memory + batching + aggregation) without a GPU."""

    def __init__(self, batch: int):
        self.batch = batch
        self.device = torch.device("cpu")
        self.dtype = torch.uint8
        self.backend = "fake"
        self.staging_np = np.empty(batch * SLOT_BYTES, dtype=np.uint8)
        self.clock = StageClock(None)  # stays at zero: no GPU work

    def stage(self, i_off: int, frame_flat: np.ndarray) -> None:
        self.staging_np[i_off:i_off + frame_flat.size] = frame_flat

    def run(self, metas, used_bytes: int) -> np.ndarray:
        return np.zeros((len(metas), len(TARGETS)), dtype=np.int32)


class GpuDetector:
    def __init__(self, weights: str, batch: int, device: str | None = None, half: bool | None = None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        cuda = self.device.type == "cuda"
        if half is None:
            half = cuda
        self.trt = None
        self.model = None
        names = None
        if str(weights).endswith(".engine") and cuda:
            try:
                from .trt import TrtModel
                self.trt = TrtModel(weights, self.device)
                batch = self.trt.batch
                names = self.trt.names
                self.dtype = self.trt.in_dtype if self.trt.in_dtype.is_floating_point else torch.float32
                self.backend = "tensorrt-async"
            except Exception as e:  # fall back to Ultralytics' runner
                print(f"TrtModel unavailable ({e}); using AutoBackend", file=sys.stderr)
                self.trt = None
        if self.trt is None:
            from ultralytics.nn.autobackend import AutoBackend
            meta_path = Path(weights).with_suffix(".json")
            if str(weights).endswith(".engine") and meta_path.exists():
                batch = json.loads(meta_path.read_text())["batch"]
            self.model = AutoBackend(weights, device=self.device, fp16=half and cuda, fuse=True, verbose=False)
            self.dtype = torch.float16 if getattr(self.model, "fp16", False) else torch.float32
            names = self.model.names
            self.backend = "autobackend"
        if names is None:  # engine without metadata: names from the .pt next to it
            from ultralytics import YOLO
            names = YOLO(str(Path(weights).with_suffix(".pt"))).names
        self.batch = batch
        self.is_engine = str(weights).endswith(".engine")

        name_to_id = {v: k for k, v in names.items()}
        lut = torch.full((max(names) + 1,), -1, dtype=torch.long)
        for t, name in enumerate(TARGETS):
            lut[name_to_id[name]] = t
        self.lut = lut.to(self.device)
        self.class_ids = [name_to_id[n] for n in TARGETS]

        # own CUDA stream per instance, so several instances overlap on one GPU
        self.stream = torch.cuda.Stream(self.device) if cuda else None
        # host staging buffer (pinned on CUDA for fast async copies) and device buffers
        self.staging = torch.empty(batch * SLOT_BYTES, dtype=torch.uint8, pin_memory=cuda)
        self.staging_np = self.staging.numpy()
        self.dev_flat = torch.empty(batch * SLOT_BYTES, dtype=torch.uint8, device=self.device) if cuda else None
        self.canvas = torch.empty((batch, IMGSZ, IMGSZ, 3), dtype=torch.uint8, device=self.device)
        self.warmup()
        self.clock = StageClock(self.stream)  # created after warmup so warmup is not counted

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trt is not None:
            return self.trt(x, self.stream)
        if self.is_engine and self.stream is not None:
            self.stream.synchronize()  # AutoBackend's TensorRT call is not ordered on our stream
        out = self.model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    def warmup(self) -> None:
        x = torch.zeros((self.batch, 3, IMGSZ, IMGSZ), dtype=self.dtype, device=self.device)
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _null()
        with ctx, torch.inference_mode():
            for _ in range(3 if self.stream is not None else 1):
                self._forward(x)
        if self.stream is not None:
            self.stream.synchronize()

    # ------------------------------------------------------------------ host -> device
    def stage(self, i_off: int, frame_flat: np.ndarray) -> None:
        """Copy one frame (flat uint8) into the pinned staging buffer at offset."""
        self.staging_np[i_off:i_off + frame_flat.size] = frame_flat

    def run(self, metas, used_bytes: int) -> np.ndarray:
        """Staging path. metas: per frame (offset, code, nh, nw, top, left) into the staging buffer.
        Returns an (n, 5) int32 numpy array of per-frame target counts."""
        ctx = torch.cuda.stream(self.stream) if self.stream is not None else _null()
        with ctx, torch.inference_mode():
            self.clock.mark(0)
            dev = self.staging[:used_bytes].to(self.device, non_blocking=True)
            self.clock.mark(1)
            return self._infer(dev, metas)

    def run_shared(self, frames) -> np.ndarray:
        """Zero-extra-copy path: frames are (host uint8 tensor over a cudaHostRegister-ed shared-memory
        slot, code, nh, nw, top, left). Each is DMA-copied straight to the GPU; no staging memcpy.
        The caller may free the slots once this returns (the result copy synchronises the stream)."""
        with torch.cuda.stream(self.stream), torch.inference_mode():
            self.clock.mark(0)
            off, metas = 0, []
            for host, code, nh, nw, top, left in frames:
                nb = host.numel()
                self.dev_flat[off:off + nb].copy_(host, non_blocking=True)
                metas.append((off, code, nh, nw, top, left))
                off += nb
            self.clock.mark(1)
            return self._infer(self.dev_flat, metas)

    def _infer(self, dev: torch.Tensor, metas) -> np.ndarray:
        n = len(metas)
        canvas = self.canvas
        canvas.fill_(PAD_VALUE)
        for i, (off, code, nh, nw, top, left) in enumerate(metas):
            nb = frame_nbytes(code, nh, nw)
            if code == BGR:
                img = dev[off:off + nb].view(nh, nw, 3)
            else:
                img = yuv420_to_bgr(dev[off:off + nb], nh, nw, code)
            canvas[i, top:top + nh, left:left + nw] = img
        if self.trt is None and not (self.is_engine and not getattr(self.model, "dynamic", False)):
            canvas = canvas[:n]  # a static engine needs the full batch; padding rows are ignored
        # BGR -> RGB, HWC -> CHW, scale to [0, 1]
        x = canvas.flip(-1).permute(0, 3, 1, 2).to(self.dtype).div_(255.0).contiguous()
        self.clock.mark(2)
        out = self._forward(x)
        self.clock.mark(3)
        counts = self._counts(out[:n])
        self.clock.mark(4)
        result = counts.cpu().numpy()  # synchronises the stream
        self.clock.collect()
        return result

    # ------------------------------------------------------------------ post-processing
    def _counts(self, pred: torch.Tensor) -> torch.Tensor:
        """pred: (n, 4 + nc, anchors) raw head output, boxes xywh in input pixels."""
        n = pred.shape[0]
        pred = pred.transpose(1, 2)                         # (n, A, 4 + nc)
        conf, cls = pred[..., 4:].max(-1)                   # best class per box
        tgt = self.lut[cls]                                 # -1 if not a target class
        keep = (conf > CONF) & (tgt >= 0)
        b_idx, a_idx = keep.nonzero(as_tuple=True)
        counts = torch.zeros(n * len(TARGETS), dtype=torch.int32, device=pred.device)
        if b_idx.numel():
            xywh = pred[b_idx, a_idx, :4].float()
            boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
            scores = conf[b_idx, a_idx].float()
            group = b_idx * len(TARGETS) + tgt[b_idx, a_idx]  # (frame, class) -> class-aware per frame
            kept = torchvision.ops.batched_nms(boxes, scores, group, IOU)
            counts.index_add_(0, group[kept], torch.ones_like(kept, dtype=torch.int32))
        return counts.view(n, len(TARGETS))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
