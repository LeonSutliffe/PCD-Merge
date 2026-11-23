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
