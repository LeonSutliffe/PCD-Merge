import numpy as np
import pytest

from pcd_merge.cli import main
from pcd_merge.merge import merge_clouds, transform_cloud
from pcd_merge.pcd_io import PointCloud, read_pcd, write_pcd
from pcd_merge.registration import AlignSettings, align

from pcd_merge.synthetic import make_scan, make_scene, make_transform, transform_error

POSES = [
    make_transform(0.3, 0.02, (-5, 0, 1.5)),
    make_transform(2.0, 0.03, (4, 3, 1.6)),     # rotated ~100 degrees
    make_transform(-1.2, 0.0, (2, -7, 1.4)),
]


@pytest.fixture(scope="module")
def scans():
    scene = make_scene(seed=0, density=120)
    return [make_scan(scene, p[:2, 3], 18, p, seed=i) for i, p in enumerate(POSES)]


def _xyz(d):
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def _check(result, n):
    for i in range(n):
        truth = np.linalg.inv(POSES[0]) @ POSES[i]
        rot, trans = transform_error(result.transforms[i], truth)
        assert rot < 0.2, f"cloud {i}: rotation error {rot:.3f} deg"
        assert trans < 0.03, f"cloud {i}: translation error {trans:.3f}"
    assert result.ok


def test_two_clouds(scans):
    r = align([_xyz(s) for s in scans[:2]], AlignSettings(seed=1))
    _check(r, 2)
    assert r.warnings() == []


def test_three_clouds_pose_graph(scans):
    r = align([_xyz(s) for s in scans], AlignSettings(seed=2))
    _check(r, 3)


def test_unrelated_cloud_is_flagged(scans):
    rng = np.random.default_rng(5)
    junk = rng.uniform(-50, 50, (20000, 3)) + 500  # noise far away, matches nothing
    r = align([_xyz(scans[0]), _xyz(scans[1]), junk], AlignSettings(seed=3))
    assert not r.connected[2] or r.cloud_fitness[2] < 0.3
    assert r.warnings()


def test_merge_keeps_every_point_and_field(scans):
    clouds = [PointCloud(s.copy()) for s in scans[:2]]
    clouds[1].data["x"][:5] = np.nan  # invalid points are dropped
    t = make_transform(0.5, 0, (1, 2, 3))
    merged, partial = merge_clouds(clouds, [np.eye(4), t], add_source_field=True)
    assert len(merged) == len(scans[0]) + len(scans[1]) - 5
    assert partial == []
    assert merged.dtype.names == ("x", "y", "z", "intensity", "ring", "source")
    part = merged[merged["source"] == 1]
    np.testing.assert_array_equal(part["intensity"], scans[1]["intensity"][5:])
    exp = _xyz(scans[1][5:]) @ t[:3, :3].T + t[:3, 3]
    np.testing.assert_allclose(_xyz(part), exp, atol=1e-4)


def test_merge_field_union_and_normals():
    a = np.zeros(3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                           ("normal_x", "<f4"), ("normal_y", "<f4"), ("normal_z", "<f4")])
    a["normal_x"] = 1
    b = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<u2")])
    b["intensity"] = 9
    t = make_transform(np.pi / 2)
    merged, partial = merge_clouds([PointCloud(a), PointCloud(b)], [t, np.eye(4)])
    assert set(partial) == {"normal_x", "normal_y", "normal_z"}
    assert merged["intensity"].dtype == np.float32
    np.testing.assert_allclose(merged["intensity"][3:], 9)
    np.testing.assert_allclose(merged["normal_y"][:3], 1, atol=1e-6)  # x rotated 90deg -> y


def test_transform_uses_float64_for_large_coordinates():
    d = np.zeros(1, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    d["x"], d["y"] = 500000.123, 4000000.456
    out = transform_cloud(PointCloud(d), make_transform(0, 0, (0.001, 0, 0)))
    assert abs(out["x"][0] - 500000.124) < 1e-6


def test_cli_end_to_end(tmp_path, scans):
    paths = []
    for i, s in enumerate(scans[:2]):
        p = str(tmp_path / f"scan{i}.pcd")
        write_pcd(p, s, compressed=(i == 1))
        paths.append(p)
    out = str(tmp_path / "merged.pcd")
    tf = str(tmp_path / "tf.json")
    assert main(paths + ["-o", out, "--seed", "4", "--transforms", tf]) == 0
    m = read_pcd(out)
    assert len(m) == len(scans[0]) + len(scans[1])
    assert m.field_names == ("x", "y", "z", "intensity", "ring")


def test_cli_exports_las_thinned(tmp_path, scans):
    laspy = pytest.importorskip("laspy")
    paths = []
    for i, s in enumerate(scans[:2]):
        p = str(tmp_path / f"scan{i}.pcd")
        write_pcd(p, s)
        paths.append(p)
    out = str(tmp_path / "merged.laz")
    assert main(paths + ["-o", out, "--seed", "4", "--min-spacing", "0.2"]) == 0
    las = laspy.read(out)
    assert 0 < len(las.points) < len(scans[0]) + len(scans[1])
    assert "ring" in las.point_format.extra_dimension_names
    assert main(paths + ["-o", str(tmp_path / "merged.xyz")]) == 2
