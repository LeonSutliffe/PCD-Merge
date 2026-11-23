"""Optional NVIDIA GPU acceleration (CUDA via CuPy), with CPU fallback.

The Windows program does not include CuPy or the CUDA libraries, because
they would add a few hundred MB to every download. They come as a separate
**GPU add-on** zip. Unzip it so that this folder exists::

    %LOCALAPPDATA%\\PCD Merge\\gpu\\        (survives app updates)

or put the ``gpu`` folder next to ``PCD Merge.exe``. With the add-on and an
NVIDIA graphics card (with a current driver) present, the full-resolution
voxel downsampling in :mod:`registration` runs on the GPU. Anything missing
or failing means the CPU path is used, as before. ``PCD_MERGE_GPU=0`` turns
the GPU off; when running from source, ``pip install cupy-cuda12x`` is enough.

Layout of the add-on folder (built by the windows build workflow)::

    gpu/site/       cupy, cuda-pathfinder, fastrlock (Python packages)
    gpu/cuda/bin/   CUDA runtime and NVRTC DLLs
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass

import numpy as np

# Clouds smaller than this are downsampled on the CPU: copying them to the
# GPU costs more than it saves.
MIN_POINTS = 1_000_000
# Points sent to the GPU at a time (bounds GPU memory: ~100 bytes per point).
CHUNK = 8_000_000


@dataclass
class Status:
    available: bool
    device: str  # e.g. "NVIDIA GeForce RTX 3070 (8.0 GB)"
    reason: str  # why the GPU is not used, when it isn't

    def describe(self) -> str:
        if self.available:
            return f"GPU: {self.device}"
        return f"CPU only ({self.reason})"


_lock = threading.Lock()
_status: Status | None = None
_cp = None  # the cupy module once loaded
_forced_off = False


def addon_dirs() -> list[str]:
    """Folders searched for the GPU add-on, in order."""
    dirs = []
    if os.environ.get("PCD_MERGE_GPU_DIR"):
        dirs.append(os.environ["PCD_MERGE_GPU_DIR"])
    if getattr(sys, "frozen", False):
        dirs.append(os.path.join(os.path.dirname(sys.executable), "gpu"))
    base = os.environ.get("LOCALAPPDATA")
    if base:
        dirs.append(os.path.join(base, "PCD Merge", "gpu"))
    return dirs


def _activate_addon() -> str | None:
    """Make an unzipped add-on importable. Returns its folder, if found."""
    for d in addon_dirs():
        site = os.path.join(d, "site")
        if not os.path.isdir(site):
            continue
        if site not in sys.path:
            sys.path.insert(0, site)
        cuda = os.path.join(d, "cuda")
        bindir = os.path.join(cuda, "bin")
        if os.path.isdir(bindir):
            # CuPy finds NVRTC through CUDA_PATH when it is not in a normal
            # site-packages folder; Windows needs the DLL folder registered.
            os.environ.setdefault("CUDA_PATH", cuda)
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bindir)
                except OSError:
                    pass
        return d
    return None


def _detect() -> Status:
    if os.environ.get("PCD_MERGE_GPU", "").strip() in ("0", "off", "no", "false"):
        return Status(False, "", "turned off by PCD_MERGE_GPU")
    addon = _activate_addon()
    try:
        import cupy as cp
    except Exception as e:  # not installed, or its DLLs are missing
        if addon is None and getattr(sys, "frozen", False):
            return Status(False, "", "GPU add-on not installed")
        return Status(False, "", f"CuPy could not be loaded: {type(e).__name__}: {e}")
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return Status(False, "", "no NVIDIA GPU found")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        mem = props.get("totalGlobalMem", 0) / 1e9
        # Compile and run a tiny kernel now, so a broken setup (old driver,
        # missing NVRTC) is found here rather than in the middle of a job.
        _voxel_sums(cp, cp.asarray(np.random.default_rng(0).uniform(0, 1, (1000, 3))),
                    np.zeros(3), 0.25, np.array([4, 4, 4]))
        cp.cuda.Stream.null.synchronize()
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        if "InsufficientDriver" in msg or "NoDevice" in msg:
            msg = "no NVIDIA GPU with a current driver found"
        return Status(False, "", f"GPU not usable: {msg}")
    global _cp
    _cp = cp
    return Status(True, f"{name} ({mem:.1f} GB)", "")


def status() -> Status:
    """Whether the GPU will be used, detected once and cached."""
    global _status
    with _lock:
        if _forced_off:
            return Status(False, "", "turned off")
        if _status is None:
            _status = _detect()
        return _status


def enabled() -> bool:
    return status().available


def set_enabled(on: bool) -> None:
    """Turn the GPU on or off for this run (e.g. to compare speeds)."""
    global _forced_off
    with _lock:
        _forced_off = not on


def _disable(reason: str) -> None:
    global _status
    with _lock:
        _status = Status(False, "", reason)


# --------------------------------------------------------------------------
# Voxel downsampling


def _voxel_sums(xp, pts, origin, voxel, dims):
    """Per-voxel key, coordinate sum and count for one chunk.

    ``xp`` is cupy (or numpy, for the tests). Coordinates are made relative
    to ``origin`` first so the sums keep full float64 precision.
    """
    local = pts - xp.asarray(origin)
    idx = xp.floor(local / voxel).astype(xp.int64)
    d = xp.asarray(dims, dtype=xp.int64)
    key = (idx[:, 0] * d[1] + idx[:, 1]) * d[2] + idx[:, 2]
    uniq, inv = xp.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    count = xp.bincount(inv, minlength=len(uniq)).astype(xp.float64)
    sums = xp.empty((len(uniq), 3), dtype=xp.float64)
    for a in range(3):
        sums[:, a] = xp.bincount(inv, weights=local[:, a], minlength=len(uniq))
    return uniq, sums, count


def voxel_down_np(xyz: np.ndarray, voxel: float, xp=None, chunk: int = CHUNK) -> np.ndarray:
    """Centroid of the points in each occupied voxel, on the same grid as
    Open3D's ``voxel_down_sample`` (origin half a voxel below the minimum).

    Runs on the GPU with ``xp=cupy`` (the default when available), or with
    numpy. Large clouds go to the GPU in chunks; partial sums are combined
    exactly, so the result does not depend on the chunk size.
    """
    if xp is None:
        if not enabled():
            raise RuntimeError("GPU not available")
        xp = _cp
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros((0, 3))
    lo = xyz.min(axis=0)
    origin = lo - voxel * 0.5
    dims = np.floor((xyz.max(axis=0) - origin) / voxel).astype(np.int64) + 1
    keys, sums, counts = [], [], []
    for start in range(0, len(xyz), chunk):
        part = xp.asarray(xyz[start : start + chunk])
        k, s, c = _voxel_sums(xp, part, origin, voxel, dims)
        to_host = (lambda a: a.get()) if hasattr(k, "get") else (lambda a: a)
        keys.append(to_host(k))
        sums.append(to_host(s))
        counts.append(to_host(c))
        del part, k, s, c
    if len(keys) == 1:
        k, s, c = keys[0], sums[0], counts[0]
    else:  # combine chunks that share voxels
        allk = np.concatenate(keys)
        k, inv = np.unique(allk, return_inverse=True)
        alls, allc = np.concatenate(sums), np.concatenate(counts)
        c = np.bincount(inv, weights=allc, minlength=len(k))
        s = np.stack([np.bincount(inv, weights=alls[:, a], minlength=len(k))
                      for a in range(3)], axis=1)
    if xp is not np and _cp is not None:
        _cp.get_default_memory_pool().free_all_blocks()
    return s / c[:, None] + origin


def try_voxel_down(xyz: np.ndarray, voxel: float) -> np.ndarray | None:
    """GPU voxel downsampling if it is available and worth it, else None.

    A failure (for example running out of GPU memory) switches the GPU off
    for the rest of the run, and the caller falls back to the CPU.
    """
    if len(xyz) < MIN_POINTS or not enabled():
        return None
    try:
        return voxel_down_np(xyz, voxel)
    except Exception as e:  # noqa: BLE001 - any GPU problem means: use the CPU
        _disable(f"GPU failed, using CPU: {type(e).__name__}: {e}".splitlines()[0])
        return None
