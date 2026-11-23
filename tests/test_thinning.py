import numpy as np
import pytest

from pcd_merge.thinning import min_spacing_indices, thin


def _min_pairwise(p):
    d2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min())


@pytest.mark.parametrize("spacing", [0.05, 0.3, 1.0, 4.0])
def test_minimum_spacing_is_guaranteed_and_coverage_is_good(spacing):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-5, 5, (3000, 3))
    idx = min_spacing_indices(xyz, spacing)
    kept = xyz[idx]
    assert np.all(np.diff(idx) > 0)
    if len(kept) > 1:
        assert _min_pairwise(kept) >= spacing
    # every original point has a kept point nearby (no big holes)
    d = np.sqrt(((xyz[:, None, :] - kept[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    assert d.max() < 2 * spacing


def test_keeps_original_records_and_drops_nan():
    rng = np.random.default_rng(1)
    d = np.zeros(2000, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("intensity", "<f4"), ("ring", "<u2")])
    for k in "xyz":
        d[k] = rng.uniform(0, 3, len(d))
    d["intensity"] = np.arange(len(d))
    d["ring"] = rng.integers(0, 64, len(d))
    d["x"][:10] = np.nan
    out = thin(d, 0.25)
    assert 0 < len(out) < len(d)
    assert np.all(np.isfinite(out["x"]))
    src = d[out["intensity"].astype(int)]  # intensity is the original row number
    np.testing.assert_array_equal(src, out)


def test_tiny_spacing_keeps_everything_distinct():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    assert list(min_spacing_indices(xyz, 1e-3)) == [0, 1, 2, 3]
    assert len(min_spacing_indices(np.repeat(xyz[:1], 5, axis=0), 0.1)) == 1
