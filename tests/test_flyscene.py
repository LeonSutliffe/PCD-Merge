import numpy as np
import pytest

from pcd_merge.flyscene import (FlyCamera, build_scene, default_speed, plan_frame,
                                start_camera)
from pcd_merge.pcd_io import PointCloud
from pcd_merge.synthetic import make_scan, make_scene, make_transform

POSES = [make_transform(0.0, 0.0, (-5, 0, 1.5)), make_transform(1.0, 0.0, (4, 3, 1.6))]


@pytest.fixture(scope="module")
def clouds():
    scene = make_scene(seed=0, density=60)
    return [PointCloud(make_scan(scene, p[:2, 3], 18, p, seed=i), path=f"scan{i}.pcd")
            for i, p in enumerate(POSES)]


@pytest.fixture(scope="module")
def scene(clouds):
    # make_scan returns points in the scanner's frame: POSES put them back in the world.
    return build_scene(clouds, POSES)


def test_scene_is_chunked_and_moved(clouds, scene):
    assert len(scene) == sum(len(c) for c in clouds)
    assert scene.chunk_count.sum() == len(scene)
    assert np.all(np.diff(scene.chunk_start) == scene.chunk_count[:-1])
    pos = scene.vertices["pos"]
    # Every point lies within its chunk's bounding sphere.
    chunk = np.repeat(np.arange(len(scene.chunk_count)), scene.chunk_count)
    d = np.linalg.norm(pos - scene.chunk_centre[chunk], axis=1)
    assert np.all(d <= scene.chunk_radius[chunk] + 1e-4)
    # The two scans were moved into one frame: the scanner positions are where
    # the poses put them.
    assert [n for n, _ in scene.scan_positions] == ["scan0.pcd", "scan1.pcd"]
    for (_, p), pose in zip(scene.scan_positions, POSES):
        assert np.allclose(p + scene.origin, pose[:3, 3], atol=1e-3)
    assert np.all(scene.chunk_spacing > 0)
    assert scene.colour_modes == ["Intensity", "Height", "Per scan"]
    for mode in scene.colour_modes:
        assert scene.colours(mode).shape == (len(scene), 4)


def test_point_cap(clouds):
    small = build_scene(clouds, POSES, max_points=20_000)
    assert 15_000 < len(small) < 25_000
    assert small.total_points == sum(len(c) for c in clouds)


def test_single_cloud_without_transform(clouds):
    s = build_scene(clouds[:1])
    assert len(s) == len(clouds[0])
    assert "Per scan" not in s.colour_modes
    assert np.allclose(s.scan_positions[0][1] + s.origin, 0, atol=1e-3)


def test_plan_culls_and_respects_budget(scene):
    cam = start_camera(scene)
    cam.pos = np.asarray(scene.scan_positions[0][1], dtype=np.float64)
    mvp = cam.projection_matrix(1.5) @ cam.view_matrix()
    focal = cam.focal_px(800)
    full = plan_frame(scene, cam.pos, mvp, focal, budget=10**9, px_target=0.01)
    assert 0 < full.points < len(scene)  # what is behind the camera is skipped
    tight = plan_frame(scene, cam.pos, mvp, focal, budget=20_000, px_target=0.01)
    assert tight.points < full.points
    assert tight.points < 40_000  # rounding to levels and 1-point minimums
    # Drawn ranges stay inside the chunks they come from.
    starts = set(scene.chunk_start.tolist())
    for _, firsts, counts in tight.levels:
        assert set(firsts.tolist()) <= starts
        assert np.all(counts >= 1)


def test_far_chunks_are_thinned(scene):
    cam = FlyCamera(far=1000)
    cam.pos = np.array([0.0, 0.0, 200.0])
    cam.look_at([0, 0, 0])
    mvp = cam.projection_matrix(1.0) @ cam.view_matrix()
    plan = plan_frame(scene, cam.pos, mvp, cam.focal_px(600), budget=10**9, px_target=1.5)
    assert plan.points < 0.2 * len(scene)
    assert all(level > 1 for level, _, _ in plan.levels)


def test_camera_moves_level_and_looks_forward():
    cam = FlyCamera()
    cam.yaw, cam.pitch = 90.0, 45.0
    cam.move(2.0, 0.0, 0.0)  # walking: stays at the same height
    assert np.allclose(cam.pos, [0, 2, 0], atol=1e-9)
    cam.move(0.0, 1.0, 0.5)  # facing +Y, right is +X
    assert np.allclose(cam.pos, [1, 2, 0.5], atol=1e-9)
    cam.move(np.sqrt(2), 0.0, 0.0, follow_look=True)
    assert np.allclose(cam.pos, [1, 3, 1.5], atol=1e-9)
    # A point straight ahead ends up on the view axis, in front (negative z).
    ahead = np.append(cam.pos + 5 * cam.forward(), 1.0)
    v = cam.view_matrix() @ ahead
    assert np.allclose(v[:2], 0, atol=1e-9) and v[2] == pytest.approx(-5)
    cam.look(0, 1000)  # pitch is clamped short of straight down
    assert cam.pitch == -89.0


def test_default_speed(scene):
    assert default_speed(scene) == 2.0


# --------------------------------------------------------------------------
# Editing and measuring in the walk-through


def test_pick_finds_the_nearest_point_under_the_mouse(scene):
    from pcd_merge.flyscene import pick

    cam = start_camera(scene)
    w, h = 800, 600
    mvp = cam.projection_matrix(w / h) @ cam.view_matrix()
    pos = scene.vertices["pos"].astype(np.float64)
    clip = pos @ mvp[:3, :3].T + mvp[:3, 3]
    ww = pos @ mvp[3, :3] + mvp[3, 3]
    ndc = clip[:, :2] / ww[:, None]
    seen = np.flatnonzero((ww > 0.5) & np.all(np.abs(ndc) < 0.8, axis=1))
    target = seen[len(seen) // 2]
    i = pick(scene, mvp, ndc[target], (w, h), radius_px=3.0)
    assert i is not None
    # the nearest of the points within 3 px of the click
    near = np.flatnonzero((ww > 1e-9) & (np.linalg.norm(
        (ndc - ndc[target]) * [w / 2, h / 2], axis=1) <= 3.0))
    assert ww[i] == pytest.approx(ww[near].min())
    # hidden points are never picked
    skip = np.zeros(len(scene), bool)
    skip[near] = True
    j = pick(scene, mvp, ndc[target], (w, h), radius_px=3.0, skip=skip)
    assert j is None or j not in near
    # nothing there (looking at the sky)
    assert pick(scene, mvp, [0.0, 5.0], (w, h)) is None


def test_scene_edits_hide_deleted_points_and_hidden_parts(clouds):
    from pcd_merge.edits import Edit, EditStack, HalfSpace, Sphere
    from pcd_merge.flyscene import SceneEdits, region_mask, world

    sc = build_scene(clouds, POSES)
    sc.scan_keys = ["a", "b"]
    xyz = world(sc)
    centre = xyz[len(xyz) // 3]
    ball = Sphere(centre, 1.5)
    inside = np.linalg.norm(xyz - centre, axis=1) <= 1.5
    assert np.array_equal(region_mask(sc, ball), inside)  # (chunk culling loses nothing)

    st = EditStack()
    st.add(Edit("delete", ball))
    ed = SceneEdits(sc)
    ed.update(st)
    assert np.array_equal(ed.deleted, inside)
    ed.apply()
    assert np.array_equal(sc.hidden, inside)
    chunk = np.repeat(np.arange(len(sc.chunk_count)), sc.chunk_count)
    assert np.array_equal(sc.chunk_live, np.bincount(chunk[~inside], minlength=len(sc.chunk_count)))
    ed.apply(show_deleted=True)  # shown (dark red) to be restored
    assert sc.hidden is None

    # an edit for one scan only; a restore; undo
    only_b = Edit("delete", Sphere(centre, 3.0), scans=["b"])
    st.add(only_b)
    ed.update(st)
    in3 = np.linalg.norm(xyz - centre, axis=1) <= 3.0
    assert np.array_equal(ed.deleted, inside | (in3 & (sc.scan_id == 1)))
    st.add(Edit("restore", Sphere(centre, 0.5)))
    ed.update(st)
    assert not ed.deleted[np.linalg.norm(xyz - centre, axis=1) <= 0.5].any()
    st.undo()
    st.undo()
    ed.update(st)
    assert np.array_equal(ed.deleted, inside)
    assert len(ed._masks) == 3  # each region tested once

    # parts: points outside every visible part are hidden
    z = float(np.median(xyz[:, 2]))
    upper, lower = st.split(HalfSpace([0, 0, 1], z), ("up", "down"))
    lower.visible = False
    ed.update(st)
    assert np.array_equal(ed.outside_parts, xyz[:, 2] < z)
    # merged files (no scan key) are never edited
    sc.scan_keys = ["a", None]
    ed.update(st)
    assert not ed.deleted[sc.scan_id == 1].any() and not ed.outside_parts[sc.scan_id == 1].any()


def test_hidden_chunks_are_not_drawn(clouds):
    sc = build_scene(clouds, POSES)
    cam = FlyCamera(pos=np.array([0.0, -40.0, 10.0]))
    cam.look_at([0, 0, 0])
    mvp = cam.projection_matrix(4 / 3) @ cam.view_matrix()
    before = plan_frame(sc, cam.pos, mvp, 500, 10_000_000).points
    sc.chunk_live = np.zeros(len(sc.chunk_count), np.int64)
    sc.chunk_live[::2] = 1
    after = plan_frame(sc, cam.pos, mvp, 500, 10_000_000).points
    assert 0 < after < before


def test_near_points_and_chunk_ranges(scene):
    from pcd_merge.flyscene import chunk_ranges, near_points, world

    idx = chunk_ranges(scene, [0, 2])
    want = np.r_[np.arange(scene.chunk_start[0], scene.chunk_start[0] + scene.chunk_count[0]),
                 np.arange(scene.chunk_start[2], scene.chunk_start[2] + scene.chunk_count[2])]
    assert np.array_equal(idx, want)
    xyz = world(scene)
    c = xyz[100]
    got = near_points(scene, c, 2.0)
    want = xyz[np.all(np.abs(xyz - c) <= 2.0, axis=1)]
    assert len(got) == len(want)
