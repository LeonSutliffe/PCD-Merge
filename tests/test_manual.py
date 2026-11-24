"""Manual (point-pair) alignment."""

import numpy as np
import pytest

from pcd_merge.manual import ManualAlignError, manual_result, refine, rigid_from_pairs
from pcd_merge.synthetic import make_scan, make_scene, make_transform


def _rot(deg, axis=2):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    r = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def test_three_pairs_give_the_transform_back():
    rng = np.random.default_rng(1)
    fixed = rng.uniform(-10, 10, (4, 3))
    r = _rot(37) @ _rot(5, 0)
    moving = (fixed - [3, -2, 1]) @ r  # fixed = moving @ r.T + shift
    fit = rigid_from_pairs(moving[:3], fixed[:3])
    assert fit.rms < 1e-9
    got = moving @ fit.transform[:3, :3].T + fit.transform[:3, 3]
    assert np.allclose(got, fixed, atol=1e-9)
    # a mis-click shows up as the worst pair
    bad = fixed.copy()
    bad[2] += [0.5, 0, 0]
    fit = rigid_from_pairs(moving, bad)
    assert fit.worst == 2 and fit.rms > 0.05


def test_level_needs_two_pairs_and_only_turns_about_z():
    fixed = np.array([[0, 0, 0], [5, 0, 0.1], [0, 3, 2.0]])  # moving: these turned +80°
    moving = (fixed - [1, 2, 0.5]) @ _rot(-80)
    fit = rigid_from_pairs(moving[:2], fixed[:2], level=True)
    r = fit.transform[:3, :3]
    assert np.allclose(r[2], [0, 0, 1]) and np.allclose(r, _rot(-80))
    assert np.allclose(moving @ r.T + fit.transform[:3, 3], fixed, atol=1e-9)


def test_bad_picks_are_refused():
    with pytest.raises(ManualAlignError):
        rigid_from_pairs([[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]])  # too few
    with pytest.raises(ManualAlignError):
        line = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        rigid_from_pairs(line, line)  # in a line
    with pytest.raises(ManualAlignError):
        rigid_from_pairs([[0, 0, 0]] * 3, [[0, 0, 0]] * 2)


def test_refine_tightens_rough_picks():
    scene = make_scene(seed=0, density=60)
    true = make_transform(25.0, 0.0, (3, 2, 0.5))
    fixed = make_scan(scene, (0, 0), 16, np.eye(4), seed=0)
    fixed = np.c_[fixed["x"], fixed["y"], fixed["z"]].astype(float)
    # the moving scan is the same site, moved by inv(true)
    inv = np.linalg.inv(true)
    moving = fixed @ inv[:3, :3].T + inv[:3, 3]
    rough = true.copy()
    rough[:3, 3] += [0.15, -0.1, 0.05]  # a few clicks' worth of error
    t, fitness = refine(moving, fixed, rough, voxel=0.1)
    assert np.abs(t[:3, 3] - true[:3, 3]).max() < 0.02 and fitness > 0.9


def test_manual_result_keeps_the_other_scans():
    r = manual_result(None, 3, [np.eye(4)] * 3, moved=2, fixed=0, fitness=0.8, voxel=0.1)
    assert r.connected == [True, False, True] and r.reference == 0 and r.voxel_size == 0.1
    r2 = manual_result(r, 3, [np.eye(4)] * 3, moved=1, fixed=0, fitness=0.7, voxel=9)
    assert r2.ok and r2.cloud_fitness == [1.0, 0.7, 0.8] and r2.voxel_size == 0.1
