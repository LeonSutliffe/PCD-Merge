"""Reading PLY, LAS, LAZ and E57 scans (and streaming them in chunks)."""

import os

import numpy as np
import pytest

from pcd_merge.exporters import write_cloud
from pcd_merge.files import find_files
from pcd_merge.pcd_io import write_pcd
from pcd_merge.readers import (INPUT_FORMATS, ScanReadError, find_scan_files, open_scans,
                               pack_rgb, read_cloud, read_clouds, sample_points)

pytest.importorskip("laspy")


def _cloud(n=3000, seed=1, offset=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    d = np.zeros(n, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f4"),
                           ("rgb", "<f4"), ("timestamp", "<f8"), ("ring", "<u2")])
    for k, o in zip("xyz", offset):
        d[k] = o + rng.uniform(-20, 20, n)
    d["intensity"] = rng.integers(0, 4000, n)
    d["rgb"] = pack_rgb(rng.integers(0, 256, n), rng.integers(0, 256, n), rng.integers(0, 256, n))
    d["timestamp"] = 1.7e9 + rng.uniform(0, 5, n)
    d["ring"] = rng.integers(0, 32, n)
    return d


def _rgb(col):
    v = col.view(np.uint32)
    return np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], 1)


@pytest.mark.parametrize("ext", [".pcd", ".ply", ".las", ".laz", ".e57"])
def test_round_trip_every_format(tmp_path, ext):
    d = _cloud(offset=(500_000.0, 4_000_000.0, 100.0) if ext in (".las", ".laz") else (0, 0, 0))
    path = str(tmp_path / f"scan{ext}")
    write_cloud(path, d)
    cloud = read_cloud(path)
    assert len(cloud) == len(d)
    tol = 2e-4 if ext in (".las", ".laz") else 1e-9
    for k in "xyz":
        assert np.allclose(cloud.data[k], d[k], atol=tol)
    assert np.allclose(cloud.data["intensity"], d["intensity"])
    assert np.array_equal(_rgb(cloud.data["rgb"]), _rgb(d["rgb"]))
    if ext in (".pcd", ".ply"):
        assert np.array_equal(cloud.data["ring"], d["ring"])
    if ext in (".las", ".laz"):
        assert np.allclose(cloud.data["gps_time"], d["timestamp"])
        assert np.array_equal(cloud.data["ring"], d["ring"])  # kept as an extra field
    if ext == ".e57":
        assert np.allclose(cloud.data["timestamp"], d["timestamp"])


@pytest.mark.parametrize("ext", [".pcd", ".ply", ".las", ".laz", ".e57"])
def test_chunks_match_full_read(tmp_path, ext):
    d = _cloud(n=10_001)
    path = str(tmp_path / f"scan{ext}")
    write_cloud(path, d)
    (src,) = open_scans(path)
    assert src.count == len(d)
    parts = list(src.chunks(size=3000))
    assert len(parts) == 4
    joined = np.concatenate(parts)
    assert joined.dtype == src.dtype
    full = src.read().data
    for k in joined.dtype.names:
        assert np.array_equal(joined[k], full[k])


def test_las_header_gives_bounds_without_reading_points(tmp_path):
    d = _cloud(offset=(10.0, 20.0, 30.0))
    path = str(tmp_path / "a.las")
    write_cloud(path, d)
    (src,) = open_scans(path)
    lo, hi = src.bounds
    assert np.allclose(lo, [d[k].min() for k in "xyz"], atol=1e-3)
    assert np.allclose(hi, [d[k].max() for k in "xyz"], atol=1e-3)


def test_las_rescaled_intensity_comes_back_exactly(tmp_path):
    d = _cloud()
    d["intensity"] = np.linspace(0, 1, len(d))  # not whole numbers: LAS rescales it
    path = str(tmp_path / "a.laz")
    write_cloud(path, d)
    assert np.allclose(read_cloud(path).data["intensity"], d["intensity"], atol=1e-6)


def _ply(path, header_extra, body: bytes, props, n):
    head = ["ply", header_extra, f"element vertex {n}"] + [f"property {t} {nm}" for t, nm in props]
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["element face 0", "property list uchar int vertex_indices",
                                   "end_header"]) + "\n").encode())
        f.write(body)


def test_ply_ascii_and_big_endian_with_colour(tmp_path):
    props = [("float", "x"), ("float", "y"), ("float", "z"), ("uchar", "red"),
             ("uchar", "green"), ("uchar", "blue"), ("float", "scalar_intensity")]
    rows = [(1, 2, 3, 255, 0, 10, 0.5), (4, 5, 6, 0, 128, 255, 0.25)]
    p1 = str(tmp_path / "a.ply")
    _ply(p1, "format ascii 1.0", "".join(" ".join(map(str, r)) + "\n" for r in rows).encode(),
         props, 2)
    dt = np.dtype([(nm, ">" + {"float": "f4", "uchar": "u1"}[t]) for t, nm in props])
    p2 = str(tmp_path / "b.ply")
    _ply(p2, "format binary_big_endian 1.0", np.array(rows, dtype=dt).tobytes(), props, 2)
    for p in (p1, p2):
        c = read_cloud(p)
        assert c.data["x"].tolist() == [1, 4]
        assert np.allclose(c.data["intensity"], [0.5, 0.25])
        assert _rgb(c.data["rgb"]).tolist() == [[255, 0, 10], [0, 128, 255]]


def _two_scan_e57(path):
    import uuid

    import pye57

    e57 = pye57.E57(path, mode="w")
    rng = np.random.default_rng(0)
    for k in range(2):
        n = 500 + k
        data = {"cartesianX": rng.uniform(-5, 5, n), "cartesianY": rng.uniform(-5, 5, n),
                "cartesianZ": rng.uniform(0, 3, n), "intensity": rng.uniform(0, 1, n)}
        e57.write_scan_raw(data, name=f"Station {k + 1}", rotation=np.array([0.0, 0, 0, 1]),
                           translation=np.array([10.0 * k, 0, 0]))
    e57.close()
    return uuid


def test_multi_scan_e57_gives_one_source_per_scan_with_pose(tmp_path):
    path = str(tmp_path / "site.e57")
    _two_scan_e57(path)
    sources = open_scans(path)
    assert [s.count for s in sources] == [500, 501]
    assert sources[0].name == "site.e57 [1/2: Station 1]"
    assert sources[0].key != sources[1].key
    assert sources[1].viewpoint[:3] == (10.0, 0.0, 0.0)
    clouds = read_clouds(path)
    assert [c.name for c in clouds] == [s.name for s in sources]
    # quaternion (0,0,0,1) = 180 degrees about z, then +10 m in x
    raw = sources[1].read().data
    assert raw["x"].min() > 4.9 and raw["x"].max() < 15.1


def test_sample_points_streams_a_random_subset(tmp_path):
    d = _cloud(n=50_000)
    d["x"][:10] = np.nan
    path = str(tmp_path / "a.pcd")
    write_pcd(path, d)
    (src,) = open_scans(path)
    s = sample_points(src, 5000)
    assert len(s) == 5000 and np.isfinite(s).all()


def test_find_scan_files_every_format_but_not_project_folders(tmp_path):
    for name in ("b.LAS", "a.pcd", "c.e57", "d.ply", "e.laz", "notes.txt", "a.pcd.json"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "Site.pcdproj" / "cache").mkdir(parents=True)
    (tmp_path / "Site.pcdproj" / "cache" / "x.pcd").write_bytes(b"")
    found = [os.path.basename(f) for f in find_scan_files([str(tmp_path)])]
    assert found == ["a.pcd", "b.LAS", "c.e57", "d.ply", "e.laz"]
    assert set(INPUT_FORMATS) == {".pcd", ".ply", ".las", ".laz", ".e57"}
    assert find_files([str(tmp_path)], (".pcd",)) == [str(tmp_path / "a.pcd")]


def test_unsupported_and_broken_files_raise_a_clear_error(tmp_path):
    bad = tmp_path / "a.ply"
    bad.write_bytes(b"not a ply")
    with pytest.raises(ScanReadError):
        open_scans(str(bad))
    with pytest.raises(ScanReadError):
        open_scans(str(tmp_path / "a.xyz"))
