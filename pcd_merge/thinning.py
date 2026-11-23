"""Thin a cloud so that no two points are closer than a minimum spacing.

Used on export to make files smaller. Unlike a voxel filter, which averages
points, this keeps a subset of the *original* points, so every field
(intensity, ring, time, colour, ...) stays exactly as measured.

Method (all vectorised, no per-point Python loop):

1. Put points in a grid of cubes with side ``spacing`` and keep, in each
   cube, the point nearest the cube's centre. This gives an even spread.
2. Points in neighbouring cubes can still be closer than ``spacing``. The
   cubes are split into 8 groups by the parity of their (i, j, k) index;
   two cubes in the same group are at least one cube apart, so their points
   are always at least ``spacing`` apart. Groups are processed one after
   another, and a point is dropped if it is too close to an already accepted
   point in any of its 26 neighbouring cubes.

The result is guaranteed to have a minimum distance of ``spacing`` between
any two points.
"""

from __future__ import annotations

import numpy as np

_OFFSETS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                     if (a, b, c) != (0, 0, 0)], dtype=np.int64)


def min_spacing_indices(xyz: np.ndarray, spacing: float) -> np.ndarray:
    """Indices (sorted) of a subset of ``xyz`` with no two points closer than ``spacing``.

    Points with non-finite coordinates are dropped.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    idx_all = np.flatnonzero(np.all(np.isfinite(xyz), axis=1))
    if len(idx_all) == 0:
        return idx_all
    pts = xyz[idx_all]
    origin = pts.min(axis=0)
    cell = np.floor((pts - origin) / spacing).astype(np.int64)
    dims = cell.max(axis=0) + 3  # room for the -1/+1 neighbours
    cell += 1
    if float(np.prod(dims.astype(np.float64))) > 2**62:
        raise ValueError("spacing is too small for the size of this cloud")

    def key(c):
        return (c[..., 0] * dims[1] + c[..., 1]) * dims[2] + c[..., 2]

    # 1. one point per cube: the one closest to the cube centre
    keys = key(cell)
    centre_d2 = np.sum((pts - origin - (cell - 1 + 0.5) * spacing) ** 2, axis=1)
    order = np.lexsort((centre_d2, keys))
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    rep = order[first]  # one per cube, sorted by key
    rep_keys = keys[rep]
    rep_cell = cell[rep]
    rep_pts = pts[rep]

    # 2. resolve conflicts between neighbouring cubes, one parity group at a time
    accepted = np.zeros(len(rep), dtype=bool)
    parity = (rep_cell[:, 0] & 1) * 4 + (rep_cell[:, 1] & 1) * 2 + (rep_cell[:, 2] & 1)
    s2 = spacing * spacing
    for group in range(8):
        cand = np.flatnonzero(parity == group)
        if len(cand) == 0:
            continue
        ok = np.ones(len(cand), dtype=bool)
        for off in _OFFSETS:
            # a neighbour's group is this group with the odd offset axes
            # flipped; only groups already processed can hold accepted points
            if group ^ int((off[0] & 1) * 4 + (off[1] & 1) * 2 + (off[2] & 1)) > group:
                continue
            nk = key(rep_cell[cand] + off)
            pos = np.searchsorted(rep_keys, nk)
            pos = np.minimum(pos, len(rep_keys) - 1)
            hit = (rep_keys[pos] == nk) & accepted[pos]
            if not hit.any():
                continue
            h = np.flatnonzero(hit)
            d2 = np.sum((rep_pts[cand[h]] - rep_pts[pos[h]]) ** 2, axis=1)
            ok[h[d2 < s2]] = False
        accepted[cand[ok]] = True

    return np.sort(idx_all[rep[accepted]])


def thin(data: np.ndarray, spacing: float) -> np.ndarray:
    """Return the records of structured array ``data`` kept by ``min_spacing_indices``."""
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    return data[min_spacing_indices(xyz, spacing)]
