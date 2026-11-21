"""Minimal TensorRT runner that enqueues on the caller's CUDA stream (execute_async_v3).

Ultralytics' AutoBackend runs engines with a blocking call that is not ordered on our stream, so
several model instances cannot overlap. This runner binds torch tensors directly and enqueues on
the instance's own stream: copies, pre-processing, the engine and NMS of one instance overlap with
the others, and nothing synchronises until the per-frame counts are copied back.

Reads engines written by Ultralytics export (4-byte little-endian metadata length + JSON metadata,
then the serialized engine), or plain serialized engines.
"""

from __future__ import annotations

import json

import torch


class TrtModel:
    def __init__(self, path: str, device: torch.device):
        import tensorrt as trt

        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        meta = {}
        with open(path, "rb") as f:
            blob = f.read()
        with trt.Runtime(logger) as runtime:
            try:
                n = int.from_bytes(blob[:4], "little")
                meta = json.loads(blob[4:4 + n].decode("utf-8"))
                engine_bytes = blob[4 + n:]
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                engine_bytes = blob
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine {path}")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int8: torch.int8,
                    trt.int32: torch.int32, trt.bool: torch.bool}
        self.in_dtype = to_torch[self.engine.get_tensor_dtype(self.inp)]
        self.in_shape = tuple(self.engine.get_tensor_shape(self.inp))
        out_shape = tuple(self.engine.get_tensor_shape(self.out))
        if -1 in self.in_shape or -1 in out_shape:
            raise RuntimeError("dynamic-shape engines are not supported by TrtModel; export with dynamic=False")
        self.output = torch.empty(out_shape, dtype=to_torch[self.engine.get_tensor_dtype(self.out)], device=device)
        self.batch = self.in_shape[0]
        raw = meta.get("names")
        self.names = {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else None

    def __call__(self, x: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        if x.dtype != self.in_dtype:
            x = x.to(self.in_dtype)
        x = x.contiguous()
        self.context.set_tensor_address(self.inp, x.data_ptr())
        self.context.set_tensor_address(self.out, self.output.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self._keep = x  # keep the input alive until the stream has consumed it
        return self.output
