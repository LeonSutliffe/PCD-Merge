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
    surf, _ = _surface(a, np.eye(4), 0.1)
    d = surface_distance(b, surf, 0.15)
    assert np.nanmax(d) < 1e-6
    d = surface_distance(b + [0, 0, 0.03], surf, 0.15)
    assert np.allclose(d[np.isfinite(d)], 0.03, atol=1e-6)


def test_grade_and_colours():
    assert grade_pair(0.8, 0.01, 0.95, 0.2) == "Good"
    assert grade_pair(0.8, 0.07, 0.95, 0.2) == "Fair"
    assert grade_pair(0.8, 0.5, 0.95, 0.2) == "Poor"
    assert grade_pair(0.03, 0.01, 0.95, 0.2) == "Poor"
    assert grade_pair(0.10, 0.01, 0.95, 0.2) == "Fair"  # little overlap: less evidence
    assert grade_pair(0.8, 0.01, 0.5, 0.2) == "Fair"  # half the overlap disagrees
    assert grade_pair(0.8, 0.01, 0.2, 0.2) == "Poor"
    assert grade_pair(0.8, float("nan"), 0.95, 0.2) == "Poor"
    c = error_colours(np.array([0.0, 0.5, 1.0, 5.0, np.nan]), 1.0)
    assert c.shape == (5, 4) and c.dtype == np.float32
    assert c[0, 2] > c[0, 0]  # 0 error is blue
    assert c[2, 0] > c[2, 2] and c[2, 1] < 0.3  # the top of the scale is red
    assert np.allclose(c[3, :3], CHANGED)  # beyond it: changed
    assert np.allclose(c[4, :3], NO_OVERLAP)


# --------------------------------------------------------------------------
# Ray-cast scans of a building: walls have two faces, scans only partly
# overlap, and a long corridor gives ICP room to slide.


def _box_mesh(centre, size):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(np.asarray(centre, float) - [size[0] / 2, size[1] / 2, 0])
    return m


def _building(seed=0, length=40):
    """A corridor along x with rooms either side, 15 cm walls with doors,
    floor, ceiling and some furniture."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    half = length / 2
    t = 0.15
    parts = [_box_mesh((0, 0, -0.1), (length, 20, 0.1)), _box_mesh((0, 0, 3.0), (length, 20, 0.1)),
             _box_mesh((0, 10, 0), (length, t, 3)), _box_mesh((0, -10, 0), (length, t, 3)),
             _box_mesh((half, 0, 0), (t, 20, 3)), _box_mesh((-half, 0, 0), (t, 20, 3))]
    for y in (1, -1):  # corridor walls, a door per room
        for x0 in np.arange(-half, half, 8):
            parts += [_box_mesh((x0 + 1.5, y, 0), (3, t, 3)), _box_mesh((x0 + 6, y, 0), (4, t, 3)),
                      _box_mesh((x0 + 3.5, y, 2.2), (1, t, 0.8))]
    for x in np.arange(-half + 8, half, 8):  # walls between rooms
        parts += [_box_mesh((x, 5.5, 0), (t, 9, 3)), _box_mesh((x, -5.5, 0), (t, 9, 3))]
    for _ in range(int(length)):
        c = (rng.uniform(-half + 1, half - 1), rng.choice([-1, 1]) * rng.uniform(2, 9), 0)
        parts.append(_box_mesh(c, (rng.uniform(.5, 2), rng.uniform(.5, 1.5), rng.uniform(.4, 2))))
    mesh = o3d.geometry.TriangleMesh()
    for p in parts:
        mesh += p
    return mesh


def _raycast_scan(mesh, pose, noise=0.01, step=0.4, seed=0):
    """Points a scanner at ``pose`` (scan -> world) sees, with occlusion, in
    the scan's own frame."""
    import open3d as o3d

    rng = np.random.default_rng(seed)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    az, el = np.meshgrid(np.radians(np.arange(0, 360, step)), np.radians(np.arange(-60, 89, step)))
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
    rays = np.c_[np.tile(pose[:3, 3], (len(d), 1)), d].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    ok = np.isfinite(hit) & (hit < 60)
    r = hit[ok] + rng.normal(0, noise, ok.sum())
    return (d[ok] * r[:, None]) @ pose[:3, :3]


BUILDING_SCANNERS = [(-16, 0, 1.5), (-8, 0, 1.5), (0, 0, 1.5), (8, 0, 1.5), (-15, 5, 1.5),
                     (-15, -5, 1.4)]


@pytest.fixture(scope="module")
def building():
    mesh = _building()
    poses = [make_transform(0.5 * i, 0.0, p) for i, p in enumerate(BUILDING_SCANNERS)]
    xyz = [_raycast_scan(mesh, t, seed=i) for i, t in enumerate(poses)]
    return xyz, poses, auto_voxel_size(xyz)


def test_correct_building_alignment_is_good(building):
    # Scans in a corridor and in rooms off it: pairs share anything from most
    # of their points to a sliver seen through a door, and the corridor walls
    # are seen from both sides. None of that is misalignment.
    xyz, poses, voxel = building
    q = assess(xyz, poses, voxel)
    assert q.grade == "Good", "\n".join(q.lines())
    counted = q.counted()
    for p in counted:
        assert p.offset < 0.01
        assert p.agreement > 0.9  # the other face of a wall is not "changed"
    # pairs that only share a sliver are listed but not counted
    slivers = [p for p in q.pairs if p.overlaps and all(p is not c for c in counted)]
    assert slivers and all(p.overlap < 0.15 for p in slivers)
    assert "not counted" in "\n".join(q.lines())


def test_building_misalignment_is_flagged(building):
    xyz, poses, voxel = building
    for shift in [(0.10, 0, 0), (0, 0.10, 0)]:
        moved = list(poses)
        moved[4] = make_transform(0, 0, shift) @ poses[4]  # a room scan
        q = assess(xyz, moved, voxel)
        assert q.grade != "Good", shift
        worst = max(p.offset for p in q.counted() if 4 in (p.a, p.b))
        assert worst == pytest.approx(0.10, abs=0.03)


def test_long_corridor_does_not_read_as_offset():
    # Noisy scans of a long corridor: along it, few surfaces pin the scans
    # down, so an ICP slides there. That sliding is not misalignment.
    mesh = _building(seed=1, length=72)
    pos = [(-24, 0, 1.5), (-12, 0, 1.5), (0, 0, 1.5)]
    poses = [make_transform(0.3 * i, 0.0, p) for i, p in enumerate(pos)]
    xyz = [_raycast_scan(mesh, t, noise=0.03, seed=i) for i, t in enumerate(poses)]
    voxel = auto_voxel_size(xyz)
    q = assess(xyz, poses, voxel)
    assert q.grade == "Good", "\n".join(q.lines())
    assert all(p.offset < 0.2 * voxel for p in q.counted())


def test_noise_widens_the_offset_limits():
    # 7 cm offset with a 30 cm voxel: Fair for clean scans, Good when the
    # scans themselves are 5 cm thick (a gap thinner than that can't be seen).
    assert grade_pair(0.5, 0.07, 0.95, 0.3) == "Fair"
    assert grade_pair(0.5, 0.07, 0.95, 0.3, noise=0.05) == "Good"
    assert grade_pair(0.5, 0.20, 0.95, 0.3, noise=0.05) == "Poor"
    assert grade_pair(0.5, 0.07, 0.95, 0.3, noise=np.nan) == "Fair"
