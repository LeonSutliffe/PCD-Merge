"""Checks for the large-scan paths: chunked reading, writing, downsampling and merging."""

import numpy as np
import pytest

from pcd_merge import merge as merge_mod
from pcd_merge import pcd_io
from pcd_merge.merge import merge_clouds, transform_cloud
from pcd_merge.pcd_io import PCDError, PointCloud, read_pcd, write_pcd
from pcd_merge.registration import AlignSettings, align, to_o3d, voxel_down
from pcd_merge.synthetic import make_transform


def _cloud(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                           ("normal_x", "<f4"), ("normal_y", "<f4"), ("normal_z", "<f4")])
    for k in ("x", "y", "z"):
        d[k] = rng.uniform(-10, 10, n)
    d["intensity"] = rng.uniform(0, 100, n)
    d["normal_z"] = 1
    return d


def test_chunked_binary_read_and_write_with_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(pcd_io, "_CHUNK_BYTES", 4096)  # force many chunks
    d = _cloud(10_000)
    p = str(tmp_path / "a.pcd")
    wrote = []
    write_pcd(p, d, progress=wrote.append)
    seen = []
    c = read_pcd(p, progress=seen.append)
    np.testing.assert_array_equal(c.data, d)
    for steps in (wrote, seen):
        assert len(steps) > 10
        assert steps == sorted(steps) and steps[-1] == 1.0


def test_truncated_binary_is_reported(tmp_path):
    p = tmp_path / "t.pcd"
    write_pcd(str(p), _cloud(1000))
    p.write_bytes(p.read_bytes()[:-100])
    with pytest.raises(PCDError, match="truncated"):
        read_pcd(str(p))


def test_chunked_ascii_read(tmp_path, monkeypatch):
    monkeypatch.setattr(pcd_io, "_CHUNK_BYTES", 1000)
    d = _cloud(2000)
    p = str(tmp_path / "a.pcd")
    write_pcd(p, d, binary=False)
    c = read_pcd(p)
    for name in d.dtype.names:
        np.testing.assert_allclose(c.data[name], d[name], rtol=1e-6)


def test_ascii_with_bad_value(tmp_path):
    p = tmp_path / "bad.pcd"
    p.write_text("FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\n"
                 "POINTS 2\nDATA ascii\n1 2 3\n4 oops 6\n")
    with pytest.raises(PCDError):
        read_pcd(str(p))


def test_chunked_voxel_down_matches_open3d():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-5, 5, (60_000, 3))
    ref = np.asarray(to_o3d(xyz).voxel_down_sample(0.5).points)
    got = np.asarray(voxel_down(xyz, 0.5, chunk=7_000).points)
    assert len(got) == len(ref)  # one point per occupied voxel
    # Same voxels: snap both to Open3D's grid (origin half a voxel below the
    # minimum) and compare
    lo = xyz.min(axis=0) - 0.25
    key = lambda p: np.unique(np.floor((p - lo) / 0.5).astype(int), axis=0)  # noqa: E731
    np.testing.assert_array_equal(key(got), key(ref))


def test_streamed_merge_matches_transform_cloud(monkeypatch):
    monkeypatch.setattr(merge_mod, "_CHUNK", 3_000)
    a, b = _cloud(10_000, 1), _cloud(7_000, 2)
    b["x"][5] = np.nan  # dropped from the output
    ta, tb = make_transform(0.4, 0.02, (1, 2, 3)), make_transform(-1.0, 0.0, (5, -1, 0))
    steps = []
    out, missing = merge_clouds([PointCloud(a), PointCloud(b)], [ta, tb],
                                add_source_field=True,
                                progress=lambda _m, f: steps.append(f))
    expect = np.concatenate([transform_cloud(PointCloud(a), ta), transform_cloud(PointCloud(b), tb)])
    assert missing == [] and len(out) == len(expect) == 16_999
    for name in expect.dtype.names:
        np.testing.assert_array_equal(out[name], expect[name])
    assert (out["source"][:10_000] == 0).all() and (out["source"][10_000:] == 1).all()
    assert steps[-1] == 1.0 and len(steps) > 3


def test_align_accepts_functions_and_reports_progress():
    from pcd_merge.synthetic import make_scan, make_scene

    poses = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6))]
    scene = make_scene(seed=0, density=60)
    scans = [make_scan(scene, p[:2, 3], 18, p, seed=i) for i, p in enumerate(poses)]
    calls = []
    srcs = [PointCloud(s).finite_xyz for s in scans]  # bound methods, made on demand
    r = align(srcs, AlignSettings(seed=1), progress=lambda m, f: calls.append(f))
    assert r.ok
    assert calls == sorted(calls) and calls[-1] == 1.0
    assert len(calls) >= 8  # several updates within the registration step


def test_quality_from_alignment_clouds_matches_full_clouds():
    from pcd_merge.quality import assess
    from pcd_merge.synthetic import make_scan, make_scene

    poses = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6))]
    scene = make_scene(seed=0, density=60)
    xyz = [PointCloud(make_scan(scene, p[:2, 3], 18, p, seed=i)).finite_xyz()
           for i, p in enumerate(poses)]
    r = align(xyz, AlignSettings(seed=1))
    full = assess(xyz, r.transforms, r.voxel_size)
    fast = assess(r.fine, r.transforms, r.voxel_size)
    assert [p.grade for p in fast.pairs] == [p.grade for p in full.pairs]
    for a, b in zip(fast.pairs, full.pairs):
        assert a.overlap == pytest.approx(b.overlap, abs=1e-9)
        assert a.rmse == pytest.approx(b.rmse, rel=1e-9)
