"""Floor plans and sections: cut a thin slice through the site and trace the
walls in it as straight 2D lines.

A **plan** is a horizontal slice at a height (usually about 1.2 m above a
floor, through the walls, doors and windows). A **section** is a vertical
slice along a line drawn across the site.

Tracing, in short:

1. The slice is flattened to 2D and thinned to one point per centimetre.
2. Each point gets the direction of the line its neighbours make; points in
   clutter (no clear direction) are dropped.
3. Neighbouring points along the same line are joined into runs, and each
   run becomes one straight segment (curved walls become several).
4. Segments almost on a common axis are squared up to it, pieces of one wall
   separated by a small gap are joined, and ends near a corner are extended
   or trimmed to meet exactly.

Doors and other openings wider than *close gaps* stay open, as on a drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geom import components, grid_keys, grid_pairs, local_covariances, thin


@dataclass
class Drawing:
    """2D lines traced from one slice.

    A plan is drawn in the project's X, Y (metres) at height ``z``. A section
    is drawn with X = distance along the section line from its start ``a``
    and Y = height.
    """

    kind: str  # "plan" or "section"
    segments: np.ndarray  # (S, 2, 2)
    points: np.ndarray  # (M, 2) the slice, thinned, for showing and tracing by hand
    circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (C, 3): x, y, r
    z: float = 0.0  # plan: the cut height
    a: tuple = (0.0, 0.0)  # section: start of the line (X, Y)
    b: tuple = (1.0, 0.0)  # section: end of the line
    name: str = ""
    settings: dict = field(default_factory=dict)

    def to_3d(self, pts2d: np.ndarray) -> np.ndarray:
        """Drawing coordinates back to the project frame (to show in 3D)."""
        pts2d = np.asarray(pts2d, float).reshape(-1, 2)
        if self.kind == "plan":
            return np.column_stack([pts2d, np.full(len(pts2d), self.z)])
        a, d = np.asarray(self.a, float), self.direction()
        xy = a + pts2d[:, :1] * d
        return np.column_stack([xy, pts2d[:, 1]])

    def direction(self) -> np.ndarray:
        d = np.asarray(self.b, float) - np.asarray(self.a, float)
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([1.0, 0.0])

    def total_length(self) -> float:
        if not len(self.segments):
            return 0.0
        return float(np.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=1).sum())


# ---------------------------------------------------------------- slicing


def plan_band(z: float, thickness: float):
    return z - thickness / 2, z + thickness / 2


def section_within(a, b, thickness: float):
    """Mask function for the points within ``thickness / 2`` of the vertical
    plane through a -> b (X, Y), between a and b."""
    a = np.asarray(a, float)[:2]
    d = np.asarray(b, float)[:2] - a
    length = float(np.linalg.norm(d))
    if length <= 0:
        raise ValueError("The section line has no length.")
    d /= length
    n = np.array([-d[1], d[0]])

    def within(xyz):
        rel = xyz[:, :2] - a
        u = rel @ d
        return (np.abs(rel @ n) <= thickness / 2) & (u >= 0) & (u <= length)

    return within


def section_coords(xyz: np.ndarray, a, b) -> np.ndarray:
    a = np.asarray(a, float)[:2]
    d = np.asarray(b, float)[:2] - a
    d /= np.linalg.norm(d)
    return np.column_stack([(xyz[:, :2] - a) @ d, xyz[:, 2]])


# ---------------------------------------------------------------- tracing


@dataclass
class TraceSettings:
    cell: float = 0.01  # thin the slice to one point per cell (m)
    radius: float = 0.10  # neighbourhood for a point's direction (m)
    tolerance: float = 0.015  # how far a point may be off its line (m)
    min_length: float = 0.25  # shortest line kept (m)
    close_gaps: float = 0.30  # join pieces of one line up to this far apart (m)
    snap: float = 0.30  # extend or trim ends to meet at corners within this (m)
    square: bool = True  # square up lines within a few degrees of the main axes
    square_degrees: float = 3.0
    reach: float = 0.08  # largest gap between points along a line (m)


def trace(pts2d: np.ndarray, s: TraceSettings | None = None, progress=None) -> np.ndarray:
    """Straight segments (S, 2, 2) along the walls in a 2D slice."""
    s = s or TraceSettings()
    pts2d = np.asarray(pts2d, float).reshape(-1, 2)
    if len(pts2d) < 10:
        return np.zeros((0, 2, 2))
    if progress:
        progress("Tracing lines", 0.1)
    lo = pts2d.min(axis=0)
    p, _ = thin(pts2d - lo, s.cell, average=True)
    p3 = np.column_stack([p, np.zeros(len(p))])
    cov = local_covariances(p3, s.radius, 40)[:, :2, :2]
    w, v = np.linalg.eigh(cov)
    normal = v[:, :, 0]
    across = np.sqrt(np.maximum(w[:, 0], 0))
    linear = (w[:, 1] > 0) & (across < s.tolerance) & (w[:, 0] < 0.1 * w[:, 1])
    if progress:
        progress("Tracing lines", 0.4)
    idx = np.nonzero(linear)[0]
    if len(idx) < 5:
        return np.zeros((0, 2, 2))
    q, nq = p[idx], normal[idx]
    reach = max(1, int(math.ceil(s.reach / s.cell)))
    a, b = grid_pairs(grid_keys(q, s.cell, lo=np.zeros(2)), reach)
    cos_ok = np.abs(np.einsum("ij,ij->i", nq[a], nq[b])) > math.cos(math.radians(12))
    off = np.abs(np.einsum("ij,ij->i", q[b] - q[a], nq[a])) < s.tolerance
    ok = cos_ok & off
    labels = components(len(q), a[ok], b[ok])
    if progress:
        progress("Tracing lines", 0.6)
    segs = []
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    starts = np.r_[0, np.nonzero(np.diff(sl))[0] + 1]
    ends = np.r_[starts[1:], len(sl)]
    min_pts = max(5, int(s.min_length / max(s.cell, 1e-6) * 0.3))
    for st, en in zip(starts, ends):
        if en - st < min_pts:
            continue
        segs.extend(_fit_run(q[order[st:en]], s))
    segs = np.array(segs, float).reshape(-1, 2, 2)
    if not len(segs):
        return segs
    if s.square:
        segs = square_up(segs, s.square_degrees)
    segs = join_collinear(segs, s.close_gaps, max(s.tolerance * 1.5, 0.02))
    segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) >= s.min_length]
    segs = extend_ends(segs, p, s.tolerance, s.reach)
    segs = snap_corners(segs, s.snap)
    segs = np.concatenate([segs, cap_ends(segs)])
    if progress:
        progress("Tracing lines", 1.0)
    return segs + lo


def find_circles(pts2d: np.ndarray, segs: np.ndarray, s: TraceSettings | None = None,
                 smallest: float = 0.03, largest: float = 1.0) -> np.ndarray:
    """Round things in the slice (columns, pipes, posts): (C, 3) x, y, radius.
    Looks at the points the straight lines don't explain."""
    s = s or TraceSettings()
    pts2d = np.asarray(pts2d, float).reshape(-1, 2)
    if len(pts2d) < 10:
        return np.zeros((0, 3))
    lo = pts2d.min(axis=0)
    p, _ = thin(pts2d - lo, s.cell, average=True)
    left = np.ones(len(p), dtype=bool)
    for a, b in np.asarray(segs, float).reshape(-1, 2, 2) - lo:
        d = b - a
        L = np.linalg.norm(d)
        if L <= 0:
            continue
        d /= L
        rel = p - a
        t = rel @ d
        r = np.abs(rel @ np.array([-d[1], d[0]]))
        left &= ~((r <= 2 * s.tolerance) & (t >= -0.05) & (t <= L + 0.05))
    q = p[left]
    if len(q) < 10:
        return np.zeros((0, 3))
    reach = max(1, int(math.ceil(s.reach / s.cell)))
    a, b = grid_pairs(grid_keys(q, s.cell, lo=np.zeros(2)), reach)
    labels = components(len(q), a, b)
    out = []
    from .geom import fit_circle

    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    for lab, cnt in zip(ids[counts >= 12], counts[counts >= 12]):
        lo_ = np.searchsorted(sl, lab)
        g = q[order[lo_:lo_ + cnt]]
        c, r, rms = fit_circle(g)
        if not smallest <= r <= largest or rms > max(s.tolerance * 0.7, 0.05 * r):
            continue
        ang = np.arctan2(g[:, 1] - c[1], g[:, 0] - c[0])
        covered = len(np.unique(np.floor((ang + np.pi) / (np.pi / 18)))) / 36.0
        if covered < 0.3:  # less than about a third of the way round: not a circle
            continue
        out.append([c[0] + lo[0], c[1] + lo[1], r])
    return np.array(out, float).reshape(-1, 3)


def _fit_run(pts: np.ndarray, s: TraceSettings, depth: int = 0) -> list:
    """One run of points: a segment, or several if the run bends."""
    if len(pts) < 5:
        return []
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    d = v[:, 1]
    n = v[:, 0]
    t = (pts - c) @ d
    r = (pts - c) @ n
    rms = float(np.sqrt(np.mean(r * r)))
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 < s.min_length:
        return []
    if rms > s.tolerance and depth < 8:
        # bends (a curved wall, or two walls joined at a shallow angle): split
        # at the point furthest from the line and fit each side
        k = int(np.argmax(np.abs(r)))
        cut = t[k] if t0 + s.min_length / 2 < t[k] < t1 - s.min_length / 2 else (t0 + t1) / 2
        return (_fit_run(pts[t < cut], s, depth + 1) + _fit_run(pts[t >= cut], s, depth + 1))
    # the ends: past the last points by nothing; a stray point should not
    # stretch the line, so ignore the outermost few if they are far apart
    ts = np.sort(t)
    gaps = np.diff(ts)
    big = np.nonzero(gaps > s.close_gaps)[0]
    if len(big):  # the run has a hole wider than close gaps (an occluded stretch): split
        out, prev = [], 0
        for g in big:
            piece = ts[prev:g + 1]
            if piece[-1] - piece[0] >= s.min_length:
                out.append([c + piece[0] * d, c + piece[-1] * d])
            prev = g + 1
        piece = ts[prev:]
        if len(piece) and piece[-1] - piece[0] >= s.min_length:
            out.append([c + piece[0] * d, c + piece[-1] * d])
        return out
    return [[c + t0 * d, c + t1 * d]]


def _angles(segs: np.ndarray) -> np.ndarray:
    d = segs[:, 1] - segs[:, 0]
    return np.arctan2(d[:, 1], d[:, 0])


def main_axis(segs: np.ndarray) -> float:
    """The building's main direction (radians, 0..90°): the angle most of the
    line length runs along, modulo 90°."""
    lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    ang = np.degrees(_angles(segs)) % 90.0
    hist, _ = np.histogram(ang, bins=180, range=(0, 90), weights=lengths)
    k = np.ones(5)
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], k, mode="same")[2:-2]
    best = int(np.argmax(smooth))
    sel = np.abs(((ang - (best + 0.5) * 0.5) + 45) % 90 - 45) < 1.0
    if sel.any():
        diffs = ((ang[sel] - (best + 0.5) * 0.5) + 45) % 90 - 45
        return math.radians((best + 0.5) * 0.5 + float(np.average(diffs, weights=lengths[sel])))
    return math.radians((best + 0.5) * 0.5)


def square_up(segs: np.ndarray, degrees: float = 3.0) -> np.ndarray:
    """Turn lines within ``degrees`` of the main axes (or across them) onto
    them exactly, about their middle."""
    if not len(segs):
        return segs
    axis = main_axis(segs)
    out = segs.copy()
    ang = _angles(segs)
    rel = (ang - axis + math.pi / 4) % (math.pi / 2) - math.pi / 4
    turn = np.abs(rel) <= math.radians(degrees)
    for i in np.nonzero(turn)[0]:
        target = ang[i] - rel[i]
        m = segs[i].mean(axis=0)
        half = np.linalg.norm(segs[i, 1] - segs[i, 0]) / 2
        d = np.array([math.cos(target), math.sin(target)])
        out[i] = [m - half * d, m + half * d]
    return out


def join_collinear(segs: np.ndarray, gap: float, offset: float) -> np.ndarray:
    """Join segments on (nearly) the same line whose ends are within ``gap``
    of each other (or overlap). Longest first: each takes in the pieces
    along it, again until none is left within reach."""
    segs = np.asarray(segs, float).reshape(-1, 2, 2)
    if len(segs) < 2:
        return segs
    order = np.argsort(-np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1))
    arr = segs[order].copy()
    alive = np.ones(len(arr), dtype=bool)
    sin2 = math.sin(math.radians(2.0))
    for i in range(len(arr)):
        if not alive[i]:
            continue
        while True:
            a0, a1 = arr[i]
            d = a1 - a0
            L = float(np.linalg.norm(d))
            if L <= 0:
                break
            d /= L
            n = np.array([-d[1], d[0]])
            od = arr[:, 1] - arr[:, 0]
            ol = np.linalg.norm(od, axis=1)
            par = np.abs(od @ n) <= np.maximum(ol, 1e-9) * sin2
            dist = np.abs((arr - a0) @ n).max(axis=1) <= offset
            t = (arr - a0) @ d
            near = (t.min(axis=1) <= L + gap) & (t.max(axis=1) >= -gap)
            hit = alive & par & dist & near
            hit[i] = False
            if not hit.any():
                break
            tt = np.r_[0.0, L, t[hit].ravel()]
            arr[i] = [a0 + tt.min() * d, a0 + tt.max() * d]
            alive[hit] = False
    return arr[alive]


def extend_ends(segs: np.ndarray, pts: np.ndarray, tol: float, reach: float,
                most: float = 0.25) -> np.ndarray:
    """Push each end out along its line over the slice points that continue
    it (up to ``most``): near a corner or a door frame the points have no
    single direction, so the traced line stops a little short."""
    if not len(segs) or not len(pts):
        return segs
    cell = max(most, 0.1)
    keys = np.floor(pts / cell).astype(np.int64)
    packed = keys[:, 0] * 4_000_003 + keys[:, 1]
    order = np.argsort(packed)
    sp = packed[order]
    out = segs.copy()
    for i, (a, b) in enumerate(segs):
        d = b - a
        L = np.linalg.norm(d)
        if L <= 0:
            continue
        d /= L
        n = np.array([-d[1], d[0]])
        for end, (p0, sign) in enumerate(((a, -1.0), (b, 1.0))):
            k = np.floor(p0 / cell).astype(np.int64)
            idx = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (k[0] + dx) * 4_000_003 + (k[1] + dy)
                    lo, hi = np.searchsorted(sp, q), np.searchsorted(sp, q, side="right")
                    idx.append(order[lo:hi])
            idx = np.concatenate(idx)
            if not len(idx):
                continue
            rel = pts[idx] - p0
            t = sign * (rel @ d)
            r = np.abs(rel @ n)
            t = np.sort(t[(r <= tol) & (t > 0) & (t <= most)])
            reached = 0.0
            for v in t:
                if v - reached > reach:
                    break
                reached = v
            if reached > 0:
                out[i, end] = p0 + sign * reached * d
    return out


def cap_ends(segs: np.ndarray, thickest: float = 0.6, tol: float = 0.06) -> np.ndarray:
    """Close the ends of walls: where the two faces of a wall stop at the
    same place (a door or the end of a wall), and nothing else meets them
    there, draw the short line across between them."""
    if len(segs) < 2:
        return np.zeros((0, 2, 2))
    d = segs[:, 1] - segs[:, 0]
    L = np.linalg.norm(d, axis=1)
    du = d / np.maximum(L, 1e-12)[:, None]
    ends = segs.reshape(-1, 2)  # end k belongs to segment k // 2

    def free(k):
        """Nothing else passes through end k."""
        p, i = ends[k], k // 2
        rel = p - segs[:, 0]
        t = np.einsum("ij,ij->i", rel, du)
        dist = np.linalg.norm(rel - t[:, None] * du, axis=1)
        on = (dist < 0.03) & (t > -0.03) & (t < L + 0.03)
        on[i] = False
        return not on.any()

    caps, used = [], set()
    sin2 = math.sin(math.radians(2))
    for i in range(len(segs)):
        n = np.array([-du[i, 1], du[i, 0]])
        par = np.abs(du[:, 0] * du[i, 1] - du[:, 1] * du[i, 0]) <= sin2
        gaps = np.abs((segs[:, 0] - segs[i, 0]) @ n)
        # an end of j level with an end of i, along the line
        along = np.abs((ends.reshape(-1, 2, 2)[:, :, None, :] - segs[i][None, None, :, :])
                       @ du[i]).min(axis=(1, 2))
        cand = par & (gaps >= 0.03) & (gaps <= thickest) & (along <= tol)
        cand[: i + 1] = False
        for j in np.nonzero(cand)[0]:
            for ei in (0, 1):
                for ej in (0, 1):
                    ki, kj = 2 * i + ei, 2 * j + ej
                    if ki in used or kj in used:
                        continue
                    along = abs((ends[kj] - ends[ki]) @ du[i])
                    if along <= tol and free(ki) and free(kj):
                        caps.append([ends[ki], ends[kj]])
                        used.update((ki, kj))
    return np.array(caps, float).reshape(-1, 2, 2)


def snap_corners(segs: np.ndarray, snap: float) -> np.ndarray:
    """Extend or trim each end to where its line meets another line, when
    that point is within ``snap`` of the end and of the other segment."""
    if len(segs) < 2 or snap <= 0:
        return segs
    out = segs.copy()
    p0, p1 = segs[:, 0], segs[:, 1]
    d = p1 - p0
    L = np.linalg.norm(d, axis=1)
    du = d / np.maximum(L, 1e-12)[:, None]
    sin30 = math.sin(math.radians(30))
    for i in range(len(segs)):
        # where line i meets every line j: p0_i + s du_i = p0_j + t du_j
        cross = du[i, 0] * du[:, 1] - du[i, 1] * du[:, 0]
        ok = np.abs(cross) > sin30
        ok[i] = False
        if not ok.any():
            continue
        rel = p0[ok] - p0[i]
        c = cross[ok]
        s_ = (rel[:, 0] * du[ok, 1] - rel[:, 1] * du[ok, 0]) / c
        t_ = (rel[:, 0] * du[i, 1] - rel[:, 1] * du[i, 0]) / c
        on_j = (t_ >= -snap) & (t_ <= L[ok] + snap)  # the meeting point is on segment j
        if not on_j.any():
            continue
        s_i = s_[on_j]
        for end, target in ((0, 0.0), (1, L[i])):
            dist = np.abs(s_i - target)
            k = int(np.argmin(dist))
            if dist[k] <= snap:
                new = p0[i] + s_i[k] * du[i]
                # don't let both ends collapse onto one point
                if np.linalg.norm(new - out[i, 1 - end]) > 0.5 * L[i]:
                    out[i, end] = new
    return out


# ---------------------------------------------------------------- whole jobs


def make_plan(sources, edits, z: float, thickness: float = 0.10,
              settings: TraceSettings | None = None, name: str = "", progress=None) -> Drawing:
    """Slice the site at height ``z`` and trace it."""
    from .points import gather

    zmin, zmax = plan_band(z, thickness)
    pts = gather(sources, edits, zmin=zmin, zmax=zmax,
                 progress=(lambda m, f: progress(m, 0.7 * f)) if progress else None)
    return plan_from_points(pts.xyz, z, thickness, settings, name, progress)


def plan_from_points(xyz, z, thickness=0.10, settings=None, name="", progress=None) -> Drawing:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    lo, hi = plan_band(z, thickness)
    xyz = xyz[(xyz[:, 2] >= lo) & (xyz[:, 2] <= hi)]
    s = settings or TraceSettings()
    segs = trace(xyz[:, :2], s, (lambda m, f: progress(m, 0.7 + 0.3 * f)) if progress else None)
    circles = find_circles(xyz[:, :2], segs, s)
    shown, _ = thin(xyz[:, :2], 0.02) if len(xyz) else (np.zeros((0, 2)), None)
    return Drawing("plan", segs, shown, circles, z=float(z), name=name,
                   settings={"thickness": thickness, **vars(s)})


def make_section(sources, edits, a, b, thickness: float = 0.10,
                 settings: TraceSettings | None = None, name: str = "",
                 progress=None) -> Drawing:
    """A vertical slice along a -> b (X, Y), traced."""
    from .points import gather

    pts = gather(sources, edits, within=section_within(a, b, thickness),
                 progress=(lambda m, f: progress(m, 0.7 * f)) if progress else None)
    return section_from_points(pts.xyz, a, b, thickness, settings, name, progress)


def section_from_points(xyz, a, b, thickness=0.10, settings=None, name="",
                        progress=None) -> Drawing:
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    xyz = xyz[section_within(a, b, thickness)(xyz)]
    uv = section_coords(xyz, a, b) if len(xyz) else np.zeros((0, 2))
    s = settings or TraceSettings()
    segs = trace(uv, s, (lambda m, f: progress(m, 0.7 + 0.3 * f)) if progress else None)
    circles = find_circles(uv, segs, s)
    shown, _ = thin(uv, 0.02) if len(uv) else (np.zeros((0, 2)), None)
    return Drawing("section", segs, shown, circles, a=tuple(map(float, np.asarray(a)[:2])),
                   b=tuple(map(float, np.asarray(b)[:2])), name=name,
                   settings={"thickness": thickness, **vars(s)})
