"""GPU path, exercised with numpy standing in for CuPy (CI has no GPU)."""

import numpy as np
import pytest

from pcd_merge import gpu
from pcd_merge.registration import AlignSettings, align, to_o3d, voxel_down
from pcd_merge.synthetic import make_scan, make_scene, make_transform, transform_error


def _sorted_rows(a):
    return a[np.lexsort(a.T[::-1])]


@pytest.mark.parametrize("chunk", [10**9, 7_000])
def test_gpu_voxel_down_matches_open3d(chunk):
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-5, 5, (60_000, 3)) + [500_000.0, 4_000_000.0, 30.0]  # UTM-like
    ref = np.asarray(to_o3d(xyz).voxel_down_sample(0.5).points)
    got = gpu.voxel_down_np(xyz, 0.5, xp=np, chunk=chunk)
    assert got.shape == ref.shape
    np.testing.assert_allclose(_sorted_rows(got), _sorted_rows(ref), rtol=0, atol=1e-6)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Pretend a GPU is present, with numpy doing CuPy's job."""
    monkeypatch.setattr(gpu, "_cp", np)
    monkeypatch.setattr(gpu, "_status", gpu.Status(True, "Fake GPU (numpy)", ""))
    monkeypatch.setattr(gpu, "_forced_off", False)
    monkeypatch.setattr(gpu, "MIN_POINTS", 1000)
    calls = []
    real = gpu.voxel_down_np
    monkeypatch.setattr(gpu, "voxel_down_np", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_alignment_uses_gpu_path_when_available(fake_gpu):
    poses = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6))]
    scene = make_scene(seed=0, density=60)
    scans = [make_scan(scene, p[:2, 3], 18, p, seed=i) for i, p in enumerate(poses)]
    xyz = [np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64) for s in scans]
    r = align(xyz, AlignSettings(seed=1))
    assert fake_gpu, "GPU downsampling was not used"
    rot, trans = transform_error(r.transforms[1], np.linalg.inv(poses[0]) @ poses[1])
    assert rot < 0.2 and trans < 0.03


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(gpu, "_cp", np)
    monkeypatch.setattr(gpu, "_status", gpu.Status(True, "Fake GPU", ""))
    monkeypatch.setattr(gpu, "_forced_off", False)
    monkeypatch.setattr(gpu, "MIN_POINTS", 10)

    def boom(*a, **k):
        raise MemoryError("out of GPU memory")

    monkeypatch.setattr(gpu, "voxel_down_np", boom)
    xyz = np.random.default_rng(0).uniform(0, 10, (5000, 3))
    out = voxel_down(xyz, 1.0)  # must still work, on the CPU
    assert len(out.points) == len(to_o3d(xyz).voxel_down_sample(1.0).points)
    st = gpu.status()
    assert not st.available and "out of GPU memory" in st.reason


def test_status_without_gpu_is_reported_not_raised():
    st = gpu.status()
    assert isinstance(st.describe(), str)
    gpu.set_enabled(False)
    try:
        assert not gpu.enabled()
    finally:
        gpu.set_enabled(True)
