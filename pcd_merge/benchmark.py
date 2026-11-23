"""Compare CPU and GPU speed on large synthetic scans.

``PCD Merge.exe --benchmark`` (or ``python -m pcd_merge.cli --benchmark``)
builds a few large synthetic scans, aligns and merges them once on the CPU
and once with the GPU (when the GPU add-on and an NVIDIA card are present),
and writes the timings to ``%TEMP%\\pcd-merge-benchmark.log``. The bundled
program opens that log in Notepad when it is done.

Options: ``--points N`` (per scan, default 10 million), ``--clouds K``
(2 to 4, default 3).
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import traceback

import numpy as np


def _log_path() -> str:
    return os.environ.get("PCD_MERGE_BENCHMARK_LOG") or os.path.join(
        tempfile.gettempdir(), "pcd-merge-benchmark.log")


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    return f"{name}, {os.cpu_count()} logical cores"


def run(points: int = 10_000_000, clouds: int = 3, log_path: str | None = None) -> int:
    log_path = log_path or _log_path()
    lines: list[str] = []

    def log(msg: str = "") -> None:
        lines.append(msg)
        if sys.stdout is not None:
            print(msg, flush=True)
        with open(log_path, "w", encoding="utf-8") as f:  # keep it current
            f.write("\n".join(lines) + "\n")

    code = 1
    try:
        from . import __version__, gpu
        from .merge import merge_clouds
        from .pcd_io import PointCloud
        from .registration import AlignSettings, align, voxel_down
        from .synthetic import make_scan, make_scene, make_transform, transform_error

        st = gpu.status()
        log(f"PCD Merge {__version__} benchmark, Python {sys.version.split()[0]}")
        log(f"CPU: {_cpu_name()}")
        log(f"GPU: {st.device if st.available else 'not used: ' + st.reason}")
        log("")
        poses = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6)),
                 make_transform(-1.2, 0.0, (2, -7, 1.4)), make_transform(0.9, 0.01, (-3, 6, 1.5))]
        poses = poses[: max(2, min(4, clouds))]
        log(f"Making {len(poses)} synthetic scans of about {points / 1e6:.0f} million points "
            "each (takes a minute)…")
        t = time.perf_counter()
        scene = make_scene(seed=0, density=120 * points / 75_000)
        scans = [PointCloud(make_scan(scene, p[:2, 3], 18, p, seed=i))
                 for i, p in enumerate(poses)]
        del scene
        total = sum(len(c) for c in scans)
        log(f"  {total:,} points in {time.perf_counter() - t:.0f} s")
        log("")

        modes = [("CPU", False)] + ([("GPU", True)] if st.available else [])
        results = {}
        biggest = max(scans, key=len).finite_xyz()
        for label, use_gpu in modes:
            gpu.set_enabled(use_gpu)
            if use_gpu:  # warm up: the first GPU call compiles its kernels
                voxel_down(biggest[:2_000_000], 0.5)
            t = time.perf_counter()
            voxel_down(biggest, 0.125)
            down = time.perf_counter() - t

            t = time.perf_counter()
            r = align([c.finite_xyz for c in scans], AlignSettings(seed=1))
            al = time.perf_counter() - t
            err = max(transform_error(r.transforms[i], np.linalg.inv(poses[0]) @ poses[i])[0]
                      for i in range(len(poses)))
            t = time.perf_counter()
            merge_clouds(scans, r.transforms)
            mg = time.perf_counter() - t
            ok = r.ok and err < 0.2
            results[label] = (down, al, mg)
            log(f"{label}:")
            log(f"  downsample largest scan ({len(biggest):,} pts)  {down:7.2f} s")
            log(f"  align all scans                          {al:7.2f} s"
                f"   (max rotation error {err:.3f} deg{'' if ok else ', WRONG'})")
            log(f"  merge at full resolution                 {mg:7.2f} s")
            if not ok:
                raise RuntimeError(f"{label} alignment is wrong")
            if use_gpu and not gpu.enabled():
                log(f"  note: {gpu.status().reason}")
        gpu.set_enabled(True)
        if "GPU" in results:
            c, g = results["CPU"], results["GPU"]
            log("")
            log(f"GPU speed-up: downsampling {c[0] / g[0]:.1f}x, "
                f"align {c[1] / g[1]:.2f}x, total {sum(c) / sum(g):.2f}x")
        else:
            log("")
            log("Only the CPU was measured (see the GPU line at the top).")
        code = 0
    except Exception:
        log("")
        log("BENCHMARK FAILED")
        log(traceback.format_exc())
    if getattr(sys, "frozen", False) and hasattr(os, "startfile"):
        try:
            os.startfile(log_path)  # open the results in Notepad
        except OSError:
            pass
    return code


def gpu_info() -> int:
    """Print (and log) whether the GPU would be used, for checks and support."""
    from . import gpu

    st = gpu.status()
    msg = st.describe()
    path = os.path.join(tempfile.gettempdir(), "pcd-merge-gpu.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.write("add-on folders searched: " + "; ".join(gpu.addon_dirs()) + "\n")
    if sys.stdout is not None:
        print(msg)
    return 0
