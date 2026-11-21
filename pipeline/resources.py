"""Work out how many parallel instances this machine can run.

Three kinds of parallel units:
  * decode workers   - CPU processes (one video each at a time). Limited by usable CPU cores
                       (affinity + cgroup quota, so containers/VMs are respected), by free RAM, and by
                       the number of videos (no point in more workers than videos).
  * decode threads   - FFmpeg threads inside each worker. With many workers, 1 thread each scales
                       best (process parallelism beats FFmpeg frame-threading); with few workers,
                       spread the spare cores across them.
  * GPU instances    - model copies, each in its own thread + CUDA stream, so copies / pre- and
                       post-processing of one batch overlap with the model of another. Limited by
                       free GPU memory; capped because a single GPU saturates after a few.

Every number can be overridden from the command line; the plan is printed at start-up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .decode import SLOT_BYTES

RESERVED_CORES = 2            # main process: queue handling, staging copies, aggregation, GPU feeding
WORKER_BASE_BYTES = 250 << 20  # per decode worker: interpreter + FFmpeg buffers + mp4 bytes in flight
RAM_FRACTION = 0.7            # never plan to use more than this share of currently free RAM
GPU_INSTANCE_BYTES = 3 << 30  # rough per-instance VRAM (model/engine + batch activations)
GPU_FRACTION = 0.8
MAX_GPU_INSTANCES = 3


@dataclass
class Plan:
    usable_cpus: int
    free_ram_gb: float
    workers: int
    decode_threads: int
    slots: int
    gpu_instances: int
    gpu_name: str
    gpu_free_gb: float
    limited_by: str

    def describe(self) -> str:
        return (f"plan: {self.workers} decode workers x {self.decode_threads} thread(s), {self.slots} slots each, "
                f"{self.gpu_instances} inference instance(s) | cpus={self.usable_cpus} "
                f"free_ram={self.free_ram_gb:.1f}GB gpu={self.gpu_name} gpu_free={self.gpu_free_gb:.1f}GB "
                f"| workers limited by {self.limited_by}")


def usable_cpus() -> int:
    """CPUs this process may actually use: affinity mask, then cgroup v2/v1 quota."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        n = os.cpu_count() or 1
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:  # cgroup v1
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, q // p))
        except (OSError, ValueError):
            pass
    return max(1, n)


def free_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 8 << 30  # unknown: assume 8 GB


def gpu_info():
    """(name, free bytes) of GPU 0, or (None, 0)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            return torch.cuda.get_device_name(0), int(free)
    except Exception:
        pass
    return None, 0


def make_plan(*, n_videos: int | None, fake_model: bool, workers: int | None = None,
              decode_threads: int | None = None, slots: int = 24, gpu_instances: int | None = None,
              batch: int = 32) -> Plan:
    cpus = usable_cpus()
    ram = free_ram_bytes()
    gpu_name, gpu_free = gpu_info()
    on_gpu = gpu_name is not None and not fake_model

    # --- decode workers
    if on_gpu or fake_model:
        by_cpu = max(1, cpus - RESERVED_CORES)
    else:  # CPU inference: torch needs most of the cores, decoding is comparatively cheap
        by_cpu = max(1, cpus // 4)
    per_worker = WORKER_BASE_BYTES + slots * SLOT_BYTES
    by_ram = max(1, int(ram * RAM_FRACTION) // per_worker)
    limits = {"cpu cores": by_cpu, "free RAM": by_ram}
    if n_videos is not None:
        limits["number of videos"] = max(1, n_videos)
    limited_by = min(limits, key=limits.get)
    n_workers = workers or limits[limited_by]
    if workers:
        limited_by = "--workers"

    # --- FFmpeg threads per worker: spare cores spread over workers (1 when workers fill the machine)
    threads = decode_threads or max(1, min(4, by_cpu // n_workers))

    # --- GPU inference instances
    if gpu_instances:
        n_gpu = gpu_instances
    elif on_gpu:
        n_gpu = max(1, min(MAX_GPU_INSTANCES, int(gpu_free * GPU_FRACTION) // GPU_INSTANCE_BYTES))
    else:
        n_gpu = 1

    # enough frame slots in total to keep every inference instance fed with full batches
    slots = max(slots, -(-(batch * (n_gpu + 1)) // n_workers) + 2)

    return Plan(cpus, ram / 2**30, n_workers, threads, slots, n_gpu,
                gpu_name or "none", gpu_free / 2**30, limited_by)
