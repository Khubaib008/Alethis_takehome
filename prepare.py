"""One-off preparation step (has internet, not timed):

    python prepare.py                      # INT8 TensorRT engine (default)
    python prepare.py --precision fp16     # FP16 TensorRT engine instead
    python prepare.py --precision none     # just download weights; run.py uses yolo11l.pt

1. downloads the official yolo11l.pt next to this script
2. if a CUDA GPU is available, exports a static-batch TensorRT engine (yolo11l.engine) plus
   yolo11l.json recording batch size and precision. run.py uses the engine automatically.

INT8 is calibrated on real FineVideo frames shipped in calib/images (see make_calibration.py).
INT8 shifts confidences more than FP16, so check it with evaluate.py against reference.py output;
fall back to --precision fp16 if it fails the thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_calib_yaml(names: dict) -> Path:
    """Dataset yaml for the INT8 calibrator: absolute paths, images only (no labels needed)."""
    img_dir = HERE / "calib" / "images"
    n = len(list(img_dir.glob("*.jpg")))
    if n == 0:
        raise SystemExit(f"no calibration images in {img_dir}; run make_calibration.py first")
    y = HERE / "calib" / "calib.yaml"
    lines = [f"path: {(HERE / 'calib').as_posix()}", f"train: {img_dir.as_posix()}",
             f"val: {img_dir.as_posix()}", "names:"]
    lines += [f"  {k}: {json.dumps(v)}" for k, v in sorted(names.items())]
    y.write_text("\n".join(lines) + "\n")
    print(f"calibration set: {n} images")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["int8", "fp16", "none"], default="int8")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.chdir(HERE)
    import torch
    from ultralytics import YOLO

    model = YOLO("yolo11l.pt")  # downloads into HERE if missing
    print("weights:", HERE / "yolo11l.pt")

    if args.precision == "none" or not torch.cuda.is_available():
        print("no CUDA GPU (or --precision none): skipping TensorRT export; run.py will use yolo11l.pt")
        return

    kw = dict(format="engine", imgsz=640, batch=args.batch, dynamic=False, nms=False, device=0, workspace=8)
    if args.precision == "int8":
        kw.update(quantize=8, data=str(write_calib_yaml(model.names)), fraction=1.0)
    else:
        kw.update(quantize=16)
    path = model.export(**kw)
    (HERE / "yolo11l.json").write_text(json.dumps({"batch": args.batch, "imgsz": 640, "precision": args.precision}))
    print("engine:", path)


if __name__ == "__main__":
    main()
