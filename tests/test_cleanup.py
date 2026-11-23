import numpy as np
import pytest

from pcd_merge.cleanup import (CleanupOptions, dedup_mask, moving_mask, noise_mask,
                               plausible_origin)
from pcd_merge.merge import merge_clouds, merge_clouds_with_stats, transform_cloud
from pcd_merge.pcd_io import PointCloud
from pcd_merge.synthetic import _box, _cylinder, make_scan, make_scene, make_transform

POSES = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6))]


def _xyz(d):
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


@pytest.fixture(scope="module")
def setup():
    """Two scans of the same scene; a person and a car are only in the first."""
    scene = make_scene(seed=0, density=120)
    rng = np.random.default_rng(9)
    dyn = np.r_[_cylinder(rng, (0, -2), 0.25, 1.7, 1500), _box(rng, (-2, 6, 0), (4, 1.8, 1.5), 4000)]
    s0 = make_scan(np.r_[scene, dyn], POSES[0][:2, 3], 18, POSES[0], seed=0)
    s1 = make_scan(scene, POSES[1][:2, 3], 18, POSES[1], seed=1)
    w = _xyz(s0) @ POSES[0][:3, :3].T + POSES[0][:3, 3]
    person = (np.hypot(w[:, 0], w[:, 1] + 2) < 0.35) & (w[:, 2] > 0.05)
    car = (abs(w[:, 0] + 2) < 2.1) & (abs(w[:, 1] - 6) < 1.0) & (w[:, 2] > 0.05)
    transforms = [np.eye(4), np.linalg.inv(POSES[0]) @ POSES[1]]
    return [PointCloud(s0), PointCloud(s1)], transforms, person | car


def test_off_by_default_keeps_every_point(setup):
    clouds, tf, _ = setup
    merged, _, stats = merge_clouds_with_stats(clouds, tf, cleanup=CleanupOptions())
    assert stats is None
    assert len(merged) == sum(len(c) for c in clouds)


def test_moving_objects_removed_static_kept(setup):
    clouds, tf, dynamic = setup
    parts = [_xyz(transform_cloud(c, t)) for c, t in zip(clouds, tf)]
    keep = moving_mask(parts, [t[:3, 3] for t in tf], margin=0.1)
    assert (~keep[0] & dynamic).sum() > 0.75 * dynamic.sum()
    assert (~keep[0] & ~dynamic).sum() <= 0.001 * (~dynamic).sum()
    assert keep[1].all()


def test_moving_skips_clouds_without_a_scanner_position(setup):
    clouds, tf, _ = setup
    parts = [_xyz(transform_cloud(c, t)) for c, t in zip(clouds, tf)]
    assert plausible_origin(parts[1], tf[1][:3, 3])
    far = np.array([500.0, 0, 0])
    skipped = []
    keep = moving_mask(parts, [tf[0][:3, 3], far], margin=0.1, skipped=skipped)
    assert skipped == [1]
    assert keep[0].all()


def test_dedup_keeps_one_scan_per_cell_and_original_points(setup):
    clouds, tf, _ = setup
    opts = CleanupOptions(dedup=True)
    merged, _, stats = merge_clouds_with_stats(clouds, tf, add_source_field=True, cleanup=opts)
    full, _ = merge_clouds(clouds, tf, add_source_field=True)
    assert stats.dedup_cell > 0
    assert len(merged) == len(full) - sum(stats.duplicate)
    assert 0.1 * len(full) < sum(stats.duplicate) < 0.5 * len(full)
    # every kept point is an untouched original point with all fields
    key = lambda a: set(map(tuple, np.c_[_xyz(a), a["intensity"], a["source"]]))
    assert key(merged) <= key(full)
    # one source per cell
    ijk = np.floor((_xyz(merged) - _xyz(merged).min(0)) / stats.dedup_cell).astype(int)
    cells = {}
    for c, s in zip(map(tuple, ijk), merged["source"]):
        cells.setdefault(c, set()).add(int(s))
    # cells are re-gridded from a different minimum, so allow a little slack
    assert sum(len(v) > 1 for v in cells.values()) < 0.1 * len(cells)


def test_dedup_leaves_single_scan_areas_alone():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, (5000, 3))
    b = rng.uniform(100, 110, (3000, 3))
    keep = dedup_mask([a, b], 0.5)
    assert keep[0].all() and keep[1].all()
    # identical coverage: the denser cloud wins
    c = rng.uniform(0, 10, (2000, 3))
    keep = dedup_mask([a, c], 2.0)
    assert keep[0].all() and not keep[1].any()


def test_noise_filter_removes_outliers_only():
    rng = np.random.default_rng(2)
    g = rng.uniform(-10, 10, (40000, 2))
    surface = np.c_[g, 0.01 * rng.normal(size=len(g))]
    outliers = rng.uniform(-10, 10, (200, 3)) + [0, 0, 5]
    keep = noise_mask(np.r_[surface, outliers])
    assert keep[:len(surface)].mean() > 0.999
    assert (~keep[len(surface):]).mean() > 0.8


def test_stats_report_lines(setup):
    clouds, tf, _ = setup
    opts = CleanupOptions(dedup=True, denoise=True, remove_moving=True)
    _, _, stats = merge_clouds_with_stats(clouds, tf, cleanup=opts)
    lines = stats.lines(["a.pcd", "b.pcd"])
    assert any("moving" in l for l in lines) and any("duplicate" in l for l in lines)


def test_cli_cleanup_flags(tmp_path, setup):
    from pcd_merge.cli import main
    from pcd_merge.pcd_io import read_pcd, write_pcd

    clouds, _, _ = setup
    paths = []
    for i, c in enumerate(clouds):
        p = str(tmp_path / f"scan{i}.pcd")
        write_pcd(p, c.data)
        paths.append(p)
    out = str(tmp_path / "merged.pcd")
    assert main(paths + ["-o", out, "--seed", "4", "--dedup", "--denoise", "--remove-moving"]) == 0
    m = read_pcd(out)
    assert 0.5 * sum(len(c) for c in clouds) < len(m) < 0.9 * sum(len(c) for c in clouds)
    assert m.field_names == ("x", "y", "z", "intensity", "ring")


def test_clean_masks_match_clean_and_weights_pick_the_denser_full_cloud(setup):
    from pcd_merge.cleanup import clean, clean_masks

    clouds, tf, _ = setup
    parts = [transform_cloud(c, t) for c, t in zip(clouds, tf)]
    origins = [t[:3, 3] for t in tf]
    opts = CleanupOptions(dedup=True, denoise=True, remove_moving=True)
    keep, stats = clean_masks([_xyz(p) for p in parts], origins, opts)
    cleaned, stats2 = clean([p.copy() for p in parts], origins, opts)
    for p, k, c in zip(parts, keep, cleaned):
        np.testing.assert_array_equal(p[k], c)
    assert stats.duplicate == stats2.duplicate and stats.moving == stats2.moving
    # Same coverage, but cloud 1 stands for a 10x denser full cloud: it wins.
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 10, (5000, 3)), rng.uniform(0, 10, (2000, 3))
    k = dedup_mask([a, b], 2.0, weights=[1.0, 10.0])
    assert not k[0].any() and k[1].all()
