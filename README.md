# Video indexing at speed — YOLO11l on every frame of FineVideo

For every second of every video in a folder of FineVideo Parquet files, `run.py` records which of
**person, car, dog, laptop, cell phone** are present and how many people are visible, using
Ultralytics **YOLO11l** (`yolo11l.pt`, 640 input, confidence > 0.4, class-aware NMS IoU 0.7).
**Every frame is processed**; nothing is sampled or skipped.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions; torch/torchvision come from PyTorch's CUDA 12.6 wheel index
(declared inside the file).

## Preparation command (run once, needs internet, not timed)

```bash
python prepare.py
```

Downloads the official `yolo11l.pt` and builds a TensorRT **INT8** engine (`yolo11l.engine`,
static batch 32, 640x640), calibrated on 450 FineVideo frames shipped in `calib/images/`.
Variants: `python prepare.py --precision fp16` (FP16 engine) or `--precision none` (weights only;
`run.py` then uses PyTorch FP16). If an engine exists, `run.py` uses it automatically.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

No internet is needed. Output: one JSON line per second per video, plus one `error` line for any
video that cannot be (fully) decoded. A timing report is printed to stderr at the end.

## How it works

```
reader thread ──► decode workers (CPU processes) ──► main thread ──► model instances (GPU threads)
 Parquet rows      decode every frame, rotate,        batches,          pad 640x640 (114), BGR->RGB,
 -> shared mem     letterbox-resize -> shared-mem     per-video state,  /255, YOLO11l (async TensorRT,
                   slots; long videos split at        watchdog,         own CUDA stream each), conf>0.4
                   keyframes across workers           aggregation       + best class, batched NMS,
                                                                        per-frame counts only
```

* **Decode workers** (`pipeline/decode.py`): one process per usable core (minus 2). PyAV/FFmpeg
  decoding; BGR conversion is pixel-identical to OpenCV. Damaged packets are skipped (like FFmpeg);
  undecodable videos produce an error line. Long videos are split at keyframes into segments that
  several workers decode in parallel (verified bit-identical to decoding whole).
* **Shared memory, no pickled frames**: mp4 bytes and decoded frames live in shared memory; on CUDA
  the frame slots are page-locked so the GPU copies straight out of them.
* **Model instances** (`pipeline/infer.py`, `pipeline/trt.py`): up to 3 copies of the engine, each in
  its own thread and CUDA stream, so transfers / pre- / post-processing overlap with inference.
  Post-processing mirrors Ultralytics exactly (score > 0.4 on the best of all 80 classes, then the
  5 target classes, class-aware NMS 0.7); only 5 integers per frame return to the CPU.
* **Aggregation** (`pipeline/aggregate.py`): seconds are `floor(t - t_first_frame)` from frame
  timestamps; an object is present if found in at least half of the second's frames; people is the
  median per-frame count, rounded down on halves; every second up to the last frame gets a line.
* **Resource planning** (`pipeline/resources.py`): worker count, FFmpeg threads, frame slots and
  model instances are derived from usable CPUs (affinity/cgroups), free RAM and free GPU memory.
* **Robustness**: a watchdog restarts decode workers that crash or hang (`--worker-timeout`), and the
  affected video keeps its finished seconds plus an error line.

Useful options (all optional): `--workers`, `--gpu-instances`, `--batch`, `--weights`,
`--gpu-color` (convert YUV on the GPU: faster with few CPUs, colours within ~3 levels of FFmpeg),
`--metrics-json PATH`, `--max-videos`, `--max-seconds`, `--fake-model` (no model; decode benchmark).

## Checking correctness

* `reference.py` — the first, simple version (Ultralytics `predict` on every frame, FP32), kept
  unchanged as the ground-truth generator.
* `evaluate.py --pred results.jsonl --ref reference.jsonl` — the grader's three checks
  (objects F1 >= 0.95, people exact >= 85 %, people within 1 >= 95 %) plus per-object and per-video detail.
* `make_broken_data.py` — builds damaged / unusual inputs (empty, null, garbage, truncated,
  corrupted, rotated, audio-only, corrupt Parquet, missing column, zero rows).

## Files

| Path | Purpose |
|---|---|
| `run.py` | the submission script |
| `pipeline/` | decode workers, GPU inference, TensorRT runner, aggregation, planning, timing report |
| `prepare.py` | weights download + TensorRT engine export |
| `reference.py`, `evaluate.py` | ground truth and scoring |
| `make_broken_data.py`, `make_calibration.py` | test data and INT8 calibration frames |
| `calib/images/` | 450 INT8 calibration frames from the 50 FineVideo files |
| `results.jsonl.gz` | output of the 50-file run |
| `HARDWARE.md` | machine used for that run |
| `WRITEUP.md` | write-up |
| `ai_chats/` | AI assistant chat log |
