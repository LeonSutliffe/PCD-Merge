"""Small geometry helpers shared by the Model tools: thinning to a grid,
neighbours on a grid, connected groups, local surface shape."""

from __future__ import annotations

import numpy as np


def grid_keys(pts: np.ndarray, cell: float, lo: np.ndarray | None = None) -> np.ndarray:
    """Integer grid cell of each point (N, d)."""
    lo = pts.min(axis=0) if lo is None else lo
    return np.floor((pts - lo) / cell).astype(np.int64)


def pack(ijk: np.ndarray) -> np.ndarray:
    """One int64 per cell (cells must be >= -1 and below 2**20 per axis)."""
    k = ijk + 1
    out = np.zeros(len(k), dtype=np.int64)
    for c in range(k.shape[1]):
        out = (out << 21) | k[:, c]
    return out


def thin(pts: np.ndarray, cell: float, average: bool = False):
    """One point per grid cell: (thinned points, index of each kept point in
    ``pts``). With ``average`` the cell's mean is used instead of its first
    point (the index is then that of its first point)."""
    if not len(pts):
        return pts, np.zeros(0, dtype=np.int64)
    keys = pack(grid_keys(pts, cell))
    uniq, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    if not average:
        return pts[first], first
    sums = np.zeros((len(uniq), pts.shape[1]))
    np.add.at(sums, inv, pts)
    counts = np.bincount(inv, minlength=len(uniq))
    return sums / counts[:, None], first


def grid_pairs(ijk: np.ndarray, reach: int = 1):
    """Pairs (a, b) of points whose cells are within ``reach`` cells of each
    other on every axis (each pair once). Where several points share a
    cell, they are chained together and the cell's first point stands for
    it towards its neighbours."""
    n, d = ijk.shape
    keys = pack(ijk)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    same = np.nonzero(sk[1:] == sk[:-1])[0]
    rng = np.arange(-reach, reach + 1)
    offs = np.stack(np.meshgrid(*([rng] * d), indexing="ij"), -1).reshape(-1, d)
    # half of them: the first non-zero coordinate is positive
    keep = []
    for o in offs:
        nz = np.nonzero(o)[0]
        keep.append(len(nz) and o[nz[0]] > 0)
    a_all, b_all = [], []
    for o in offs[np.array(keep, bool)]:
        q = pack(ijk + o)
        pos = np.searchsorted(sk, q)
        pos[pos >= n] = n - 1
        hit = sk[pos] == q
        a_all.append(np.nonzero(hit)[0])
        b_all.append(order[pos[hit]])
    if len(same):
        a_all.append(order[same])
        b_all.append(order[same + 1])
    if not a_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(a_all), np.concatenate(b_all)


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected groups of ``n`` items joined by the pairs (a, b): a label
    per item (the smallest index in its group)."""
    parent = np.arange(n)
    if not len(a):
        return parent
    while True:
        pa, pb = parent[a], parent[b]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        m = lo != hi
        if not m.any():
            return parent
        np.minimum.at(parent, hi[m], lo[m])
        while True:  # point every item at its root
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp


TILE_POINTS = 1_000_000  # bigger sets are done in slabs, for progress and Cancel


def local_covariances(pts: np.ndarray, radius: float, max_nn: int = 30,
                      progress=None) -> np.ndarray:
    """Per point, the 3x3 covariance of its neighbours within ``radius``.

    A big set is done in slabs along its longest side (each with a margin
    of ``radius`` so the points at its edges still see all their
    neighbours), calling ``progress(fraction)`` after each; it may raise to
    stop."""
    import open3d as o3d

    def one(p):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(p, float)))
        pc.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,
                                                                     max_nn=max_nn))
        return np.asarray(pc.covariances)

    n = len(pts)
    if n <= TILE_POINTS:
        out = one(pts) if n else np.zeros((0, 3, 3))
        if progress:
            progress(1.0)
        return out
    ax = int(np.argmax(np.ptp(pts, axis=0)))
    order = np.argsort(pts[:, ax], kind="stable")
    x = pts[order, ax]
    out = np.empty((n, 3, 3))
    for a in range(0, n, TILE_POINTS):
        b = min(a + TILE_POINTS, n)
        lo = np.searchsorted(x, x[a] - radius, side="left")
        hi = np.searchsorted(x, x[b - 1] + radius, side="right")
        cov = one(pts[order[lo:hi]])
        out[order[a:b]] = cov[a - lo:b - lo]
        if progress:
            progress(b / n)
    return out


def local_shape(pts: np.ndarray, radius: float, max_nn: int = 30):
    """Per point: the normal of the local surface (smallest axis of the
    neighbours' spread), its eigenvalues (ascending) and main direction."""
    w, v = np.linalg.eigh(local_covariances(pts, radius, max_nn))  # ascending
    return v[:, :, 0], w, v[:, :, 2]


def pca(pts: np.ndarray):
    """Centre, axes (rows, largest spread first) and spreads of ``pts``."""
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    return c, v[:, ::-1].T, w[::-1]


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def fit_circle(xy: np.ndarray, refine: bool = True):
    """Least-squares circle through 2D points: (centre, radius, rms).
    Without ``refine`` only the quick algebraic fit is made."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return np.array([cx, cy]), 0.0, np.inf
    c = np.array([cx, cy])
    r = np.sqrt(r2)
    # a few Gauss-Newton steps on the true (geometric) distance
    for _ in range(10 if refine else 0):
        d = xy - c
        dist = np.linalg.norm(d, axis=1)
        ok = dist > 1e-12
        if ok.sum() < 3:
            break
        J = np.column_stack([-d[ok, 0] / dist[ok], -d[ok, 1] / dist[ok], -np.ones(ok.sum())])
        res = dist[ok] - r
        try:
            step = np.linalg.solve(J.T @ J, -(J.T @ res))
        except np.linalg.LinAlgError:
            break
        c = c + step[:2]
        r = r + step[2]
        if np.abs(step).max() < 1e-7:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - c, axis=1) - r) ** 2)))
    return c, float(abs(r)), rms
