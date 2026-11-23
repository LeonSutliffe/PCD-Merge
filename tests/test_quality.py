import numpy as np
import pytest

from pcd_merge.quality import assess, error_colours, grade_pair, surface_distance, NO_OVERLAP, _surface
from pcd_merge.registration import auto_voxel_size
from pcd_merge.synthetic import make_scan, make_scene, make_transform

POSES = [
    make_transform(0.3, 0.02, (-5, 0, 1.5)),
    make_transform(2.0, 0.03, (4, 3, 1.6)),
]


@pytest.fixture(scope="module")
def setup():
    scene = make_scene(seed=0, density=80)
    scans = [make_scan(scene, p[:2, 3], 18, p, seed=i) for i, p in enumerate(POSES)]
    xyz = [np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64) for s in scans]
    truth = [np.eye(4), np.linalg.inv(POSES[0]) @ POSES[1]]
    return xyz, truth, auto_voxel_size(xyz)


def test_correct_alignment_is_good(setup):
    xyz, truth, voxel = setup
    q = assess(xyz, truth, voxel, preview_xyz=[x[::7] for x in xyz])
    assert q.grade == "Good"
    p = q.pairs[0]
    assert p.overlap > 0.5
    assert p.rmse < 0.03  # 1 cm sensor noise per scan
    assert len(q.point_error) == 2
    assert len(q.point_error[1]) == len(xyz[1][::7])
    # most preview points overlap and sit close to the other scan
    e = q.point_error[1]
    assert np.isfinite(e).mean() > 0.5
    assert np.nanmedian(e) < 0.02


def test_misalignment_is_flagged(setup):
    xyz, truth, voxel = setup
    off = make_transform(np.radians(2.0), 0.0, (0.15, 0.1, 0.05)) @ truth[1]
    good = assess(xyz, truth, voxel)
    bad = assess(xyz, [truth[0], off], voxel)
    assert bad.grade != "Good"
    assert bad.pairs[0].rmse > 2 * good.pairs[0].rmse


def test_unrelated_clouds_do_not_overlap(setup):
    xyz, truth, voxel = setup
    far = make_transform(0, 0, (500, 0, 0)) @ truth[1]
    q = assess(xyz, [truth[0], far], voxel, preview_xyz=xyz)
    assert q.grade == "Poor"
    assert not q.pairs[0].overlaps
    assert np.isnan(q.point_error[0]).all()
    assert "no overlap" in "\n".join(q.lines())
    d = q.to_dict()
    assert d["clouds"][0]["rmse"] is None


def test_point_to_plane_ignores_sampling_spacing():
    # Two samplings of the same flat plane, 5 cm apart in-plane: the nearest
    # neighbour is far, but the surface distance is ~0.
    g = np.mgrid[0:5:0.1, 0:5:0.1].reshape(2, -1).T
    a = np.c_[g, np.zeros(len(g))]
    b = a + [0.05, 0.05, 0.0]
    surf = _surface(a, np.eye(4), 0.1)
    d = surface_distance(b, surf, 0.15)
    assert np.nanmax(d) < 1e-6
    d = surface_distance(b + [0, 0, 0.03], surf, 0.15)
    assert np.allclose(d[np.isfinite(d)], 0.03, atol=1e-6)


def test_grade_and_colours():
    assert grade_pair(0.8, 0.01, 0.2) == "Good"
    assert grade_pair(0.8, 0.05, 0.2) == "Fair"
    assert grade_pair(0.8, 0.5, 0.2) == "Poor"
    assert grade_pair(0.05, 0.01, 0.2) == "Poor"
    c = error_colours(np.array([0.0, 0.5, 1.0, 5.0, np.nan]), 1.0)
    assert c.shape == (5, 4) and c.dtype == np.float32
    assert c[0, 2] > c[0, 0]  # 0 error is blue
    assert c[2, 0] > c[2, 2] and np.allclose(c[2], c[3])  # max and beyond are red
    assert np.allclose(c[4, :3], NO_OVERLAP)
