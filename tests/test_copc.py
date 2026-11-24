"""COPC viewing files: valid COPC (octree index) that is also plain LAZ."""

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
pytest.importorskip("lazrs")

from pcd_merge import copc  # noqa: E402
from pcd_merge.exporters import format_of, write_chunks, write_cloud  # noqa: E402
from pcd_merge.readers import pack_rgb, read_cloud  # noqa: E402


def _cloud(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.zeros(n, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f4"),
                           ("rgb", "<f4"), ("ring", "<u2")])
    d["x"] = rng.uniform(500_000, 500_080, n)  # UTM-like coordinates
    d["y"] = rng.uniform(6_000_000, 6_000_040, n)
    d["z"] = rng.uniform(0, 12, n) * (rng.random(n) < 0.4)
    d["intensity"] = rng.integers(0, 4000, n)
    d["rgb"] = pack_rgb(np.full(n, 10), np.full(n, 200), np.full(n, 30))
    d["ring"] = np.arange(n) % 32
    return d


def test_levels_fill_cubes_evenly():
    w = copc.level_weights(10_000_000, 100_000)
    assert len(w) == 5 and np.isclose(w.sum(), 1) and np.all(np.diff(w) > 0)
    assert len(copc.level_weights(50_000, 100_000)) == 1  # small: one cube
    k = copc.node_keys(np.array([[0.1, 0.9, 0.5], [0.99, 0.01, 0.2]]), np.array([0, 2]),
                       np.zeros(3), 1.0)
    assert copc.unpack_key(int(k[0])) == (0, 0, 0, 0)
    assert copc.unpack_key(int(k[1])) == (2, 3, 0, 0)


def test_copc_file_is_indexed_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(copc, "NODE_POINTS", 5_000)
    d = _cloud()
    path = str(tmp_path / "site.copc.laz")
    assert format_of(path) == ".copc.laz"
    notes = write_cloud(path, d)
    assert any("ring" in n for n in notes)
    with laspy.CopcReader.open(path) as r:
        info = r.copc_info
        assert r.header.point_count == len(d)
        assert np.allclose(info.center[:2], [500_040, 6_000_020], atol=0.01)
        assert info.halfsize == pytest.approx(40, rel=1e-3)
        everything = r.query()
        assert len(everything) == len(d)
        assert sorted(np.asarray(everything["ring"]).tolist()) == sorted(d["ring"].tolist())
        coarse = r.query(level=0)
        assert 0 < len(coarse) < len(d) / 10  # a light overview to start from
        # a box query gives exactly the points inside it
        lo, hi = np.array([500_000, 6_000_000]), np.array([500_010, 6_000_010])
        q = r.query(bounds=laspy.copc.Bounds(mins=lo, maxs=hi))
        inside = (d["x"] <= hi[0]) & (d["y"] <= hi[1])
        assert len(q) == inside.sum()
        assert np.asarray(everything.red[:1]).tolist() == [10 * 257]
    # plain LAZ readers see the same points
    back = read_cloud(path).data
    assert len(back) == len(d)
    assert np.allclose(np.sort(back["x"]), np.sort(d["x"]), atol=1e-3)
    assert not list(tmp_path.glob("*.part*")) and not list(tmp_path.glob("*.records"))


def test_streamed_and_empty(tmp_path):
    d = _cloud(30_000, seed=1)
    path = str(tmp_path / "s.copc.laz")
    write_chunks(path, d.dtype, lambda: (d[i:i + 7_000] for i in range(0, len(d), 7_000)),
                 total=len(d))
    with laspy.CopcReader.open(path) as r:
        assert len(r.query()) == len(d)
    empty = str(tmp_path / "e.copc.laz")
    write_cloud(empty, d[:0])
    with laspy.CopcReader.open(empty) as r:
        assert len(r.query()) == 0
