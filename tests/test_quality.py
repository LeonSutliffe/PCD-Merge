import numpy as np
import pytest

from pcd_merge.quality import (CHANGED, NO_OVERLAP, _surface, assess, error_colours, grade_pair,
                               surface_distance)
from pcd_merge.registration import auto_voxel_size
from pcd_merge.synthetic import _box, _cylinder, make_scan, make_scene, make_transform

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
    assert p.offset < 0.01
    assert p.agreement > 0.9
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
    assert bad.grade == "Poor"
    assert bad.pairs[0].offset > 0.15  # close to the real error (~0.3 m at the edges)
    assert bad.score < good.score


@pytest.mark.parametrize("shift", [0.05, 0.10])
def test_small_misalignment_is_flagged(setup, shift):
    xyz, truth, voxel = setup
    off = make_transform(0, 0, (shift, 0, 0)) @ truth[1]
    q = assess(xyz, [truth[0], off], voxel)
    assert q.grade != "Good"
    assert q.pairs[0].offset == pytest.approx(shift, abs=0.015)


def _moved_object_scans(seed=9):
    """The same street, but parked cars, a van, people and crates moved between
    the scans (and the van left)."""
    scene = make_scene(seed=0, density=80)
    rng = np.random.default_rng(seed)

    def things(cars, people, crates, van):
        parts = [_box(rng, (*c, 0), (4.5, 1.8, 1.5), 2400) for c in cars]
        parts += [_cylinder(rng, p, 0.25, 1.8, 500) for p in people]
        parts += [_box(rng, (*c, 0), (1.2, 0.8, 1.0), 400) for c in crates]
        if van:
            parts.append(_box(rng, (2, 8, 0), (6, 2.2, 2.6), 5000))
        return np.concatenate(parts)

    s0 = np.concatenate([scene, things([(-1, -2), (3, 1), (-6, -6)], [(0, 3), (-3, 1)],
                                       [(2, -4), (-2, 6)], van=True)])
    s1 = np.concatenate([scene, things([(-1, 0.5), (3.5, 2.2), (-6, -8)], [(1.5, 3.5), (-1, -1)],
                                       [(5, -3), (-2.5, 6.5)], van=False)])
    scans = [make_scan(s, p[:2, 3], 18, p, seed=i) for i, (s, p) in enumerate(zip((s0, s1), POSES))]
    return [np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64) for s in scans]


def test_moved_objects_do_not_lower_the_grade(setup):
    _, truth, _ = setup
    xyz = _moved_object_scans()
    voxel = auto_voxel_size(xyz)
    q = assess(xyz, truth, voxel, preview_xyz=xyz)
    assert q.grade == "Good"
    p = q.pairs[0]
    assert p.offset < 0.01
    assert p.changed > 0.003  # the moved things are found ...
    assert p.rmse < 0.03  # ... and kept out of the fit
    # and drawn as changed in the heatmap
    assert (q.point_error[0] > q.scale_max).sum() > 100
    # A real misalignment of the same scans is still caught.
    off = make_transform(np.radians(0.5), 0.0, (0.05, 0, 0)) @ truth[1]
    assert assess(xyz, [truth[0], off], voxel).grade != "Good"


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
    assert grade_pair(0.8, 0.01, 0.95, 0.2) == "Good"
    assert grade_pair(0.8, 0.07, 0.95, 0.2) == "Fair"
    assert grade_pair(0.8, 0.5, 0.95, 0.2) == "Poor"
    assert grade_pair(0.05, 0.01, 0.95, 0.2) == "Poor"
    assert grade_pair(0.8, 0.01, 0.5, 0.2) == "Fair"  # half the overlap disagrees
    assert grade_pair(0.8, 0.01, 0.2, 0.2) == "Poor"
    assert grade_pair(0.8, float("nan"), 0.95, 0.2) == "Poor"
    c = error_colours(np.array([0.0, 0.5, 1.0, 5.0, np.nan]), 1.0)
    assert c.shape == (5, 4) and c.dtype == np.float32
    assert c[0, 2] > c[0, 0]  # 0 error is blue
    assert c[2, 0] > c[2, 2] and c[2, 1] < 0.3  # the top of the scale is red
    assert np.allclose(c[3, :3], CHANGED)  # beyond it: changed
    assert np.allclose(c[4, :3], NO_OVERLAP)
