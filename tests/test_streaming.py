"""Out-of-core merge and export: scans streamed from disk a chunk at a time."""

import tracemalloc

import numpy as np
import pytest

from pcd_merge import merge as merge_mod
from pcd_merge.cli import main
from pcd_merge.exporters import open_writer, write_chunks, write_cloud
from pcd_merge.files import read_manifest, save_merged
from pcd_merge.merge import StreamedMerge, merge_clouds
from pcd_merge.pcd_io import read_pcd, write_pcd
from pcd_merge.readers import MemorySource, open_scans, pack_rgb, read_cloud
from pcd_merge.synthetic import make_scan, make_scene, make_transform

pytest.importorskip("laspy")


def _scan(n, seed, with_rgb=True):
    rng = np.random.default_rng(seed)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]
    if with_rgb:
        fields.append(("rgb", "<f4"))
    d = np.zeros(n, dtype=fields)
    for k in "xyz":
        d[k] = rng.uniform(-10, 10, n)
    d["intensity"] = rng.integers(0, 255, n)
    if with_rgb:
        d["rgb"] = pack_rgb(rng.integers(0, 256, n), 0, 255)
    d["x"][::997] = np.nan  # invalid points are dropped
    return d


def _transforms():
    return [np.eye(4), make_transform(30.0, 0.0, (5, -2, 1))]


@pytest.mark.parametrize("ext", [".pcd", ".ply", ".las", ".laz", ".e57"])
def test_streamed_save_equals_in_memory_merge(tmp_path, ext, monkeypatch):
    monkeypatch.setattr(merge_mod, "_CHUNK", 777)  # many chunks per scan
    paths = []
    for k, n in enumerate((5000, 3001)):
        p = str(tmp_path / f"s{k}.pcd")
        write_pcd(p, _scan(n, k, with_rgb=k == 0))
        paths.append(p)
    sources = [s for p in paths for s in open_scans(p)]
    clouds = [read_pcd(p) for p in paths]
    out = str(tmp_path / f"merged{ext}")
    n, partial, notes, stats = save_merged(sources, _transforms(), out, add_source=True)
    assert stats is None and partial == ["rgb"]
    ref, _ = merge_clouds(clouds, _transforms(), add_source_field=True)
    assert n == len(ref)
    got = read_cloud(out).data
    assert len(got) == len(ref)
    tol = 2e-4 if ext in (".las", ".laz") else 1e-5
    for k in "xyz":
        assert np.allclose(got[k], ref[k], atol=tol)
    assert np.allclose(got["intensity"], ref["intensity"])
    man = read_manifest(out)
    assert [e["points"] for e in man] == [int(np.isfinite(c.data["x"]).sum()) for c in clouds]
    assert [e["source"] for e in man] == [0, 1]


def test_options_needing_every_point_still_work_from_sources(tmp_path):
    p = str(tmp_path / "a.pcd")
    write_pcd(p, _scan(4000, 1))
    src = open_scans(p) * 2
    n, *_ = save_merged(src, _transforms(), str(tmp_path / "m.pcd"), min_spacing=0.5)
    assert 0 < n < 8000


def test_streamed_merge_memory_stays_at_chunk_size(tmp_path, monkeypatch):
    """Merging ~1.2M points to LAZ must not allocate anything near the full
    cloud (about 40 MB as float64 xyz); chunks are 100k points."""
    monkeypatch.setattr(merge_mod, "_CHUNK", 100_000)
    paths = []
    for k in range(3):
        p = str(tmp_path / f"s{k}.pcd")
        write_pcd(p, _scan(400_000, k))
        paths.append(p)
    sources = [open_scans(p)[0] for p in paths]
    import pcd_merge.readers as readers

    monkeypatch.setattr(readers, "CHUNK_POINTS", 100_000)
    tracemalloc.start()
    save_merged(sources, [np.eye(4)] * 3, str(tmp_path / "m.las"))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 25e6, f"peak {peak / 1e6:.0f} MB"


def test_streamed_writer_counts_are_filled_in(tmp_path):
    d = _scan(1000, 2)
    d = d[np.isfinite(d["x"])]
    for ext in (".pcd", ".ply"):
        p = str(tmp_path / f"a{ext}")
        w = open_writer(p, d.dtype)
        w.write(d[:400])
        w.write(d[400:])
        assert w.close() == []
        c = read_cloud(p)
        assert len(c) == len(d) and np.allclose(c.data["x"], d["x"])


def test_failed_stream_leaves_no_file(tmp_path):
    d = _scan(100, 1)

    def chunks():
        yield d
        raise RuntimeError("disk gone")

    with pytest.raises(RuntimeError):
        write_chunks(str(tmp_path / "a.laz"), d.dtype, lambda: chunks())
    assert list(tmp_path.iterdir()) == []


def test_memory_source_streams_a_loaded_cloud():
    from pcd_merge.pcd_io import PointCloud

    c = PointCloud(_scan(2500, 3), path="x.pcd")
    s = MemorySource(c)
    assert s.count == 2500 and sum(len(k) for k in s.chunks(1000)) == 2500
    m = StreamedMerge([s, s], [np.eye(4), np.eye(4)])
    assert sum(len(k) for k in m) == sum(m.written) == 2 * int(np.isfinite(c.data["x"]).sum())


def test_cli_aligns_mixed_formats(tmp_path):
    scene = make_scene(seed=0, density=60)
    poses = [make_transform(0.0, 0.0, (-4, 0, 1.5)), make_transform(5.0, 0.02, (3, 2, 1.5))]
    scans = [make_scan(scene, p[:2, 3], 16, p, seed=i) for i, p in enumerate(poses)]
    a, b = str(tmp_path / "a.laz"), str(tmp_path / "b.e57")
    write_cloud(a, scans[0])
    write_cloud(b, scans[1])
    out = str(tmp_path / "m.ply")
    assert main([a, b, "-o", out, "--seed", "1"]) == 0
    assert len(read_cloud(out)) == len(scans[0]) + len(scans[1])
