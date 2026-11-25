"""Fit simple solids to the scan: flat surfaces become walls, floor slabs and
roofs; round things become columns and pipes.

1. The site is thinned to one point per small cube (*detail*, 5 cm by
   default) and every point gets the direction of its local surface.
   Normals are turned to face the scanner that saw the point, so the two
   faces of a wall point away from each other.
2. Neighbouring points on the same flat surface are grown into patches,
   and each big enough patch becomes a plane: a **face**, with its outline
   (a rectangle) in that plane. Nearly level faces are made exactly level,
   nearly upright ones exactly upright and square to the building.
3. Faces become solids. The two faces of a wall (seen from the rooms on
   either side) give the wall's real thickness; a ceiling with a floor just
   above it gives the slab's. A face seen from one side only gets a set
   thickness behind it.
4. With *leave out furniture* (on by default), faces that are not part of
   the building are dropped: level faces away from any floor or ceiling
   height (shelves, tables), and upright faces that neither reach the
   ceiling nor stand nearly a storey high (shelving ends and backs,
   cupboards). A wall that reaches the ceiling but was hidden lower down
   by something standing in front of it is carried down to the floor, and
   a floor hidden under shelving is stretched to the ceiling above it.
5. What is left over is searched for cylinders: points whose normals all
   lie across one direction (the axis) and which sit on a circle around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geom import components, fit_circle, grid_keys, grid_pairs, local_covariances, unit
from .model import Element, box, cylinder

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class FitSettings:
    detail: float = 0.05  # one point per cube of this size (m)
    min_area: float = 0.4  # smallest flat surface kept (m²)
    wall_thickness: float = 0.15  # for walls seen from one side only (m)
    slab_thickness: float = 0.20  # for floors, ceilings and roofs seen from one side (m)
    thickest_wall: float = 0.6  # two faces further apart are not one wall (m)
    planes: bool = True
    cylinders: bool = True
    square_degrees: float = 3.0
    max_points: int = 6_000_000
    skip_clutter: bool = True  # leave out shelves, furniture and other things in the rooms

    @property
    def tolerance(self) -> float:
        return max(0.01, 0.3 * self.detail)


@dataclass
class Face:
    kind: str  # "wall", "floor", "ceiling", "slope"
    normal: np.ndarray  # unit, towards the scanner (into the room)
    centre: np.ndarray  # a point on the plane
    u: np.ndarray  # in-plane axes
    v: np.ndarray
    u_range: tuple
    v_range: tuple
    points: int
    rms: float

    def corner(self, a: float, b: float) -> np.ndarray:
        return self.centre + a * self.u + b * self.v

    @property
    def offset(self) -> float:
        return float(self.normal @ self.centre)

    @property
    def area(self) -> float:
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])

    def bounds(self):
        """Lowest and highest corner (x, y, z) of its outline."""
        cs = np.array([self.corner(a, b) for a in self.u_range for b in self.v_range])
        return cs.min(axis=0), cs.max(axis=0)


def _near(faces, reach):
    """For each face, the faces whose outlines come within ``reach`` of it."""
    if not faces:
        return []
    b = [f.bounds() for f in faces]
    lo = np.array([x[0] for x in b])
    hi = np.array([x[1] for x in b])
    out = []
    for i in range(len(faces)):
        m = np.all((lo <= hi[i] + reach) & (hi >= lo[i] - reach), axis=1)
        m[i] = False
        out.append(np.nonzero(m)[0])
    return out


# ---------------------------------------------------------------- local surfaces


def surface_normals(xyz: np.ndarray, sensors: np.ndarray | None, detail: float,
                    groups: np.ndarray | None = None, progress=None):
    """Per point: unit normal (towards its scanner where known), how far the
    neighbours are off a plane (m), and whether the normal was oriented.

    With ``groups`` (the scan of each point) neighbours are only taken from
    the same scan, so the two faces of a thin wall, seen from the rooms on
    either side, don't blur into one."""
    report = progress or (lambda _m, _f: None)
    if groups is None or len(np.unique(groups)) <= 1:
        cov = local_covariances(xyz, 2.5 * detail, 30,
                                progress=lambda f: report("Finding surfaces", 0.7 * f))
    else:
        cov = np.empty((len(xyz), 3, 3))
        order = np.argsort(groups, kind="stable")
        gs, starts = np.unique(groups[order], return_index=True)
        ends = np.r_[starts[1:], len(order)]
        done = 0
        for a, b in zip(starts, ends):
            sel = order[a:b]
            base = done / len(xyz)
            share = len(sel) / len(xyz)
            cov[sel] = local_covariances(
                xyz[sel], 2.5 * detail, 30,
                progress=lambda f, base=base, share=share: report("Finding surfaces",
                                                                  0.7 * (base + share * f)))
            done += len(sel)
            report("Finding surfaces", 0.7 * done / len(xyz))
    w = np.empty((len(xyz), 3))
    vecs = np.empty((len(xyz), 3, 3))
    step = 1_000_000
    for a in range(0, len(xyz), step):  # in slices, so the bar moves and Cancel works
        w[a:a + step], vecs[a:a + step] = np.linalg.eigh(cov[a:a + step])
        report("Finding surfaces", 0.7 + 0.3 * min(a + step, len(xyz)) / len(xyz))
    del cov
    n = vecs[:, :, 0]
    off = np.sqrt(np.maximum(w[:, 0], 0))
    flat = (w[:, 1] > 0) & (w[:, 0] < 0.1 * w[:, 1])
    oriented = np.zeros(len(xyz), dtype=bool)
    if sensors is not None:
        ok = np.isfinite(sensors).all(axis=1)
        to = sensors[ok] - xyz[ok]
        flip = np.einsum("ij,ij->i", to, n[ok]) < 0
        nn = n[ok]
        nn[flip] *= -1
        n[ok] = nn
        oriented[ok] = True
    return n, off, flat, oriented


def grow_patches(xyz, normals, off, flat, oriented, detail, tol, degrees=10.0, progress=None):
    """Labels of flat patches (-1: not on a flat surface)."""
    report = progress or (lambda _f: None)
    good = flat & (off < tol)
    idx = np.nonzero(good)[0]
    labels = np.full(len(xyz), -1, dtype=np.int64)
    if len(idx) < 3:
        return labels
    p, n = xyz[idx], normals[idx]
    a, b = grid_pairs(grid_keys(p, detail), 1)
    report(0.4)
    dot = np.einsum("ij,ij->i", n[a], n[b])
    both = oriented[idx][a] & oriented[idx][b]
    dot = np.where(both, dot, np.abs(dot))
    ok = dot > math.cos(math.radians(degrees))
    ok &= np.abs(np.einsum("ij,ij->i", p[b] - p[a], n[a])) < tol
    ok &= np.abs(np.einsum("ij,ij->i", p[b] - p[a], n[b])) < tol
    report(0.6)
    labels[idx] = components(len(idx), a[ok], b[ok])
    report(1.0)
    return labels


def main_angle(normals: np.ndarray, weights: np.ndarray) -> float:
    """The building's main direction in plan (radians): what most wall area
    faces, modulo 90°."""
    if not len(normals):
        return 0.0
    ang = np.degrees(np.arctan2(normals[:, 1], normals[:, 0])) % 90.0
    hist, _ = np.histogram(ang, bins=90, range=(0, 90), weights=weights)
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5), mode="same")[2:-2]
    peak = float(np.argmax(smooth)) + 0.5
    diff = (ang - peak + 45) % 90 - 45
    near = np.abs(diff) <= 2.0
    if near.any() and weights[near].sum() > 0:
        peak += float(np.average(diff[near], weights=weights[near]))
    return math.radians(peak)


def _plane(pts: np.ndarray, normal_hint: np.ndarray):
    c = pts.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    n = v[:, 0]
    if n @ normal_hint < 0:
        n = -n
    return c, n


def make_faces(xyz, normals, labels, s: FitSettings, levels=None) -> list[Face]:
    tol = s.tolerance
    cell_area = s.detail * s.detail
    faces = []
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    for lab, cnt in zip(ids, counts):
        if cnt * cell_area < s.min_area:
            continue
        lo, hi = np.searchsorted(sl, lab), np.searchsorted(sl, lab, side="right")
        sel = order[lo:hi]
        pts = xyz[sel]
        hint = unit(normals[sel].sum(axis=0))
        c, n = _plane(pts, hint)
        r = (pts - c) @ n
        keep = np.abs(r) <= 3 * tol
        if keep.sum() < 10:
            continue
        pts = pts[keep]
        c, n = _plane(pts, hint)
        rms = float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))
        faces.append((c, n, pts, rms))
    # the building's axes, from the upright faces
    upright = [(n, len(p)) for c, n, p, _ in faces if abs(n[2]) < math.sin(math.radians(10))]
    axis = main_angle(np.array([n for n, _ in upright]).reshape(-1, 3),
                      np.array([k for _, k in upright], float)) if upright else 0.0
    ax = np.array([math.cos(axis), math.sin(axis), 0.0])
    out = []
    sq = math.radians(s.square_degrees)
    for c, n, pts, rms in faces:
        nz = n[2]
        if abs(nz) > math.cos(math.radians(10)):
            kind = "floor" if nz > 0 else "ceiling"
            n = np.array([0.0, 0.0, 1.0 if nz > 0 else -1.0])
            u, v = ax.copy(), np.cross(UP, ax)
        elif abs(nz) < math.sin(math.radians(10)):
            kind = "wall"
            n = unit([n[0], n[1], 0.0])
            ang = math.atan2(n[1], n[0])
            rel = (ang - axis + math.pi / 4) % (math.pi / 2) - math.pi / 4
            if abs(rel) <= sq:
                ang -= rel
                n = np.array([math.cos(ang), math.sin(ang), 0.0])
            u, v = np.cross(UP, n), UP.copy()
        else:
            kind = "slope"
            u = unit(np.cross(UP, n))
            v = np.cross(n, u)
        c = c - ((c - pts.mean(axis=0)) @ n) * n  # (unchanged, kept on the points' plane)
        rel = pts - c
        a, b = rel @ u, rel @ v
        ur = (float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5)))
        vr = (float(np.percentile(b, 0.5)), float(np.percentile(b, 99.5)))
        out.append(Face(kind, n, c, u, v, ur, vr, len(pts), rms))
    out = merge_faces(out, s)
    if s.skip_clutter and levels:
        out = merge_faces(building_faces(out, levels, s, xyz), s)
    return snap_edges(out)


# ---------------------------------------------------------------- furniture and shelving

NEAR_LEVEL = 0.15  # a level face this close to a floor or ceiling height is on it (m)
REACHES = 0.35  # a wall whose top is this close to the ceiling reaches it (m)
TALL = 0.85  # ... or that stands this share of the storey's height
SHORTEST_PIPE = 1.0  # shorter pipes are left out with the furniture (m)


def _storeys(levels):
    """(floor, top) per level, lowest first: top is the ceiling, or the next
    floor, or None for the top level without a ceiling."""
    lv = sorted(levels, key=lambda v: v.floor)
    out = []
    for k, v in enumerate(lv):
        top = v.ceiling
        if top is None and k + 1 < len(lv):
            top = lv[k + 1].floor
        out.append((v.floor, top))
    return out


def _storey_of(lo: float, hi: float, storeys):
    """The storey a face from height ``lo`` to ``hi`` lies in the most."""
    best, k_best = -1.0, None
    for k, (f, t) in enumerate(storeys):
        top = t if t is not None else max(hi, f + 3.0)
        ov = min(hi, top) - max(lo, f)
        if ov > best:
            best, k_best = ov, k
    return k_best


def _hidden_below(f: Face, bottom: float, xyz: np.ndarray) -> bool:
    """Whether the part of wall ``f`` below its points (down to ``bottom``)
    was hidden by something in front of it, rather than open (a doorway or
    an opening with a lintel, seen through)."""
    z0 = f.corner(0, f.v_range[0])[2]
    rel = xyz - f.centre
    a = rel @ f.u
    d = rel @ f.normal  # > 0: in front, on the scanner's side
    m = (a >= f.u_range[0]) & (a <= f.u_range[1]) & (xyz[:, 2] >= bottom + 0.05) \
        & (xyz[:, 2] <= z0 - 0.05)
    front = np.count_nonzero(m & (d > 0.05) & (d < 6.0))
    behind = np.count_nonzero(m & (d < -0.05) & (d > -4.0))
    return front > max(20, behind)


def building_faces(faces: list[Face], levels, s: FitSettings, xyz=None) -> list[Face]:
    """The faces that belong to the building itself (see step 4 at the top)."""
    storeys = _storeys(levels)
    floors = [f for f, _t in storeys]
    ceilings = [v.ceiling for v in levels if v.ceiling is not None]
    near = max(NEAR_LEVEL, 3 * s.tolerance)
    keep: list[Face] = []
    for f in faces:
        lo, hi = f.bounds()
        if f.kind == "floor":
            if any(abs(f.centre[2] - z) <= near for z in floors):
                keep.append(f)
        elif f.kind == "ceiling":
            k = _storey_of(lo[2], hi[2], storeys)
            fl, top = storeys[k]
            if any(abs(f.centre[2] - z) <= near for z in ceilings) or \
                    (top is None and f.centre[2] - fl >= 1.8):
                keep.append(f)
        elif f.kind == "wall":
            k = _storey_of(lo[2], hi[2], storeys)
            fl, top = storeys[k]
            height = hi[2] - lo[2]
            reaches = top is not None and hi[2] >= top - REACHES
            tall = height >= (TALL * (top - fl) if top is not None else 2.0)
            if not (reaches or tall):
                continue  # shelving, cupboards, partitions part of the way up
            if reaches and lo[2] > fl + 0.3 and xyz is not None and _hidden_below(f, fl, xyz):
                dz = lo[2] - fl  # carry it down to the floor
                f = Face(f.kind, f.normal, f.centre, f.u, f.v, f.u_range,
                         (f.v_range[0] - dz, f.v_range[1]), f.points, f.rms)
            keep.append(f)
        else:  # sloping: a roof if big enough
            if f.area >= max(4 * s.min_area, 1.5):
                keep.append(f)
    # a floor partly hidden under shelving: stretch it to the ceiling above it
    ceil = [c for c in keep if c.kind == "ceiling"]
    out = []
    for f in keep:
        if f.kind == "floor":
            k = _storey_of(f.centre[2], f.centre[2] + 0.5, storeys)
            top = storeys[k][1]
            for c in ceil:
                if top is None or abs(c.centre[2] - top) > near:
                    continue
                (a0, a1), (b0, b1) = _ranges_in(f, c)
                inside = _overlap(f.u_range, (a0, a1)) * _overlap(f.v_range, (b0, b1))
                if inside >= 0.5 * f.area and (a1 - a0) * (b1 - b0) <= 3 * f.area:
                    f = Face(f.kind, f.normal, f.centre, f.u, f.v,
                             (min(f.u_range[0], a0), max(f.u_range[1], a1)),
                             (min(f.v_range[0], b0), max(f.v_range[1], b1)), f.points, f.rms)
        out.append(f)
    return out


def snap_edges(faces: list[Face], reach: float = 0.3) -> list[Face]:
    """Stretch (or trim) each face's outline to the nearest face it meets:
    near a corner, or where a wall meets the floor, the points have no
    single direction, so the patch stops a few centimetres short."""
    out = []
    near = _near(faces, reach)
    for i, f in enumerate(faces):
        ranges = [list(f.u_range), list(f.v_range)]
        best = {}  # (axis, end) -> (distance, value)
        for j in near[i]:
            g = faces[j]
            gu, gv = g.normal @ f.u, g.normal @ f.v
            (a0, a1), (b0, b1) = _ranges_in(f, g)
            # g's plane crosses f's along a line of constant u (axis 0) or v (axis 1)
            for axis, along, other in ((0, gu, gv), (1, gv, gu)):
                if abs(along) < 0.97 or abs(other) > 0.1:
                    continue
                at = float(g.normal @ (g.centre - f.centre)) / along
                rng = ranges[axis]
                # g must reach along f's edge, and the line must lie within g
                span, mine = ((b0, b1), ranges[1]) if axis == 0 else ((a0, a1), ranges[0])
                if _overlap(span, (mine[0] - reach, mine[1] + reach)) <= 0:
                    continue
                gr = (a0, a1) if axis == 0 else (b0, b1)
                if not gr[0] - reach <= at <= gr[1] + reach:
                    continue
                mid = (rng[0] + rng[1]) / 2
                k = 0 if at < mid else 1
                dist = abs(at - rng[k])
                if dist <= reach and dist < best.get((axis, k), (np.inf,))[0]:
                    best[(axis, k)] = (dist, at)
        for (axis, k), (_d, at) in best.items():
            ranges[axis][k] = at
        out.append(Face(f.kind, f.normal, f.centre, f.u, f.v, tuple(ranges[0]),
                        tuple(ranges[1]), f.points, f.rms))
    return out


def _ranges_in(face: Face, other: Face):
    """``other``'s rectangle in ``face``'s u, v axes."""
    cs = np.array([other.corner(a, b) for a in other.u_range for b in other.v_range]) - face.centre
    a, b = cs @ face.u, cs @ face.v
    return (a.min(), a.max()), (b.min(), b.max())


def merge_faces(faces: list[Face], s: FitSettings, gap: float = 0.3) -> list[Face]:
    """Join faces on one plane whose outlines touch (a wall split by a pillar
    in front of it, a floor split by a doorway)."""
    tol = s.tolerance
    faces = sorted(faces, key=lambda f: -f.points)
    if not faces:
        return faces
    alive = np.ones(len(faces), dtype=bool)
    b = [f.bounds() for f in faces]
    lo = np.array([x[0] for x in b])
    hi = np.array([x[1] for x in b])
    normals = np.array([f.normal for f in faces])
    kinds = np.array([f.kind for f in faces])
    for i in range(len(faces)):
        if not alive[i]:
            continue
        grown = True
        while grown:
            grown = False
            fi = faces[i]
            cand = alive & (kinds == fi.kind) & (normals @ fi.normal >= math.cos(math.radians(3)))
            cand &= np.all((lo <= hi[i] + gap) & (hi >= lo[i] - gap), axis=1)
            cand[i] = False
            for j in np.nonzero(cand)[0]:
                fj = faces[j]
                if abs((fj.centre - fi.centre) @ fi.normal) > 2.5 * tol:
                    continue
                (a0, a1), (b0, b1) = _ranges_in(fi, fj)
                if a0 > fi.u_range[1] + gap or a1 < fi.u_range[0] - gap:
                    continue
                if b0 > fi.v_range[1] + gap or b1 < fi.v_range[0] - gap:
                    continue
                ur = (min(fi.u_range[0], a0), max(fi.u_range[1], a1))
                vr = (min(fi.v_range[0], b0), max(fi.v_range[1], b1))
                tot = fi.points + fj.points
                rms = math.sqrt((fi.rms ** 2 * fi.points + fj.rms ** 2 * fj.points) / tot)
                fi = faces[i] = Face(fi.kind, fi.normal, fi.centre, fi.u, fi.v, ur, vr, tot, rms)
                lo[i], hi[i] = fi.bounds()
                alive[j] = False
                grown = True
    return [f for f, a in zip(faces, alive) if a]


# ---------------------------------------------------------------- faces to solids


def _overlap(r1, r2) -> float:
    return max(0.0, min(r1[1], r2[1]) - max(r1[0], r2[0]))


def walls_from_faces(faces: list[Face], s: FitSettings) -> list[Element]:
    walls = [f for f in faces if f.kind == "wall"]
    used = set()
    pairs = []
    near = _near(walls, s.thickest_wall)
    for i, fi in enumerate(walls):
        for j in near[i]:
            if j <= i:
                continue
            fj = walls[j]
            if fi.normal @ fj.normal > -math.cos(math.radians(5)):
                continue
            gap = (fj.centre - fi.centre) @ fi.normal  # fj is behind fi: negative
            if not -s.thickest_wall <= gap <= -0.03:
                continue
            (a0, a1), (b0, b1) = _ranges_in(fi, fj)
            ou = _overlap(fi.u_range, (a0, a1))
            ov = _overlap(fi.v_range, (b0, b1))
            short = min(fi.u_range[1] - fi.u_range[0], a1 - a0)
            low = min(fi.v_range[1] - fi.v_range[0], b1 - b0)
            if ou >= 0.5 * short and ov >= 0.5 * low:
                pairs.append((ou * ov, i, j, -gap, (a0, a1), (b0, b1)))
    out = []
    for _score, i, j, thick, ur_j, vr_j in sorted(pairs, reverse=True):
        if i in used or j in used:
            continue
        used.update((i, j))
        fi = walls[i]
        ur = (min(fi.u_range[0], ur_j[0]), max(fi.u_range[1], ur_j[1]))
        vr = (min(fi.v_range[0], vr_j[0]), max(fi.v_range[1], vr_j[1]))
        out.append(_wall(fi, ur, vr, thick, fi.points + walls[j].points,
                         max(fi.rms, walls[j].rms)))
    for i, fi in enumerate(walls):
        if i not in used:
            out.append(_wall(fi, fi.u_range, fi.v_range, s.wall_thickness, fi.points, fi.rms,
                             one_side=True))
    return out


def _wall(f: Face, ur, vr, thick, points, rms, one_side=False) -> Element:
    origin = f.corner(ur[0], vr[0])
    e = box("wall", origin, f.u, -f.normal, UP, (ur[1] - ur[0], thick, vr[1] - vr[0]),
            rms=rms, points=points)
    if one_side:
        e.name = "Wall (one side seen)"
    return e


def slabs_from_faces(faces: list[Face], s: FitSettings) -> list[Element]:
    floors = [f for f in faces if f.kind == "floor"]
    ceilings = [f for f in faces if f.kind == "ceiling"]
    used_f, used_c = set(), set()
    pairs = []
    for i, c in enumerate(ceilings):
        for j, f in enumerate(floors):
            gap = f.centre[2] - c.centre[2]
            if not 0.05 <= gap <= 1.0:
                continue
            (a0, a1), (b0, b1) = _ranges_in(f, c)
            ou, ov = _overlap(f.u_range, (a0, a1)), _overlap(f.v_range, (b0, b1))
            smaller = min(f.area, c.area)
            if ou * ov >= 0.5 * smaller:
                pairs.append((ou * ov, i, j, gap, (a0, a1), (b0, b1)))
    out = []
    for _score, i, j, gap, ur_c, vr_c in sorted(pairs, reverse=True):
        if i in used_c or j in used_f:
            continue
        used_c.add(i)
        used_f.add(j)
        f = floors[j]
        ur = (min(f.u_range[0], ur_c[0]), max(f.u_range[1], ur_c[1]))
        vr = (min(f.v_range[0], vr_c[0]), max(f.v_range[1], vr_c[1]))
        out.append(_slab(f, ur, vr, f.centre[2] - gap, gap, f.points + ceilings[i].points,
                         max(f.rms, ceilings[i].rms)))
    for j, f in enumerate(floors):
        if j not in used_f:
            e = _slab(f, f.u_range, f.v_range, f.centre[2] - s.slab_thickness,
                      s.slab_thickness, f.points, f.rms)
            e.name = "Floor (top seen)"
            out.append(e)
    for i, c in enumerate(ceilings):
        if i not in used_c:
            e = _slab(c, c.u_range, c.v_range, c.centre[2], s.slab_thickness, c.points, c.rms)
            e.name = "Ceiling (underside seen)"
            out.append(e)
    return out


def _slab(f: Face, ur, vr, z0, thick, points, rms) -> Element:
    # a level face's u, v are horizontal and u x v points up
    corner = f.centre + ur[0] * f.u + vr[0] * f.v
    corner[2] = z0
    return box("slab", corner, f.u, f.v, UP, (ur[1] - ur[0], vr[1] - vr[0], thick),
               rms=rms, points=points)


def roofs_from_faces(faces: list[Face], s: FitSettings) -> list[Element]:
    out = []
    for f in faces:
        if f.kind != "slope":
            continue
        origin = f.corner(f.u_range[0], f.v_range[0])
        # extruded into the roof, away from the scanner
        out.append(box("roof", origin, f.u, f.v, -f.normal,
                       (f.u_range[1] - f.u_range[0], f.v_range[1] - f.v_range[0],
                        s.slab_thickness), rms=f.rms, points=f.points))
        # keep the frame right-handed: u x v must point along w
        e = out[-1]
        u, v, w = e.axes()
        if np.cross(u, v) @ w < 0:
            e.origin = (np.asarray(e.origin) + e.size[0] * u).tolist()
            e.u = (-u).tolist()
    return out


# ---------------------------------------------------------------- cylinders


def find_cylinders(xyz, normals, labels, s: FitSettings, sensors=None) -> list[Element]:
    """Columns and pipes among the points not on a flat surface."""
    rest = np.nonzero(labels < 0)[0]
    if len(rest) < 30:
        return []
    p, n = xyz[rest], normals[rest]
    sn = None if sensors is None else sensors[rest]
    a, b = grid_pairs(grid_keys(p, s.detail), 1)
    lab = components(len(p), a, b)
    ids, counts = np.unique(lab, return_counts=True)
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    out = []
    tol = s.tolerance
    # a tank 3 m across and 6 m high is about 10,000 points at 5 cm; far bigger
    # groups are the corners of whole rooms, joined up
    most = int(60_000 * (0.05 / s.detail) ** 2)
    for k, cnt in zip(ids[(counts >= 30) & (counts <= most)],
                      counts[(counts >= 30) & (counts <= most)]):
        lo = np.searchsorted(sl, k)
        sel = order[lo:lo + cnt]
        e = fit_cylinder(p[sel], n[sel], tol, sensors=None if sn is None else sn[sel])
        if e is not None:
            out.append(e)
    return out


def fit_cylinder(pts: np.ndarray, normals: np.ndarray, tol: float, smallest: float = 0.02,
                 largest: float = 1.5, sensors: np.ndarray | None = None) -> Element | None:
    """A cylinder through ``pts``, or None if they don't make one.

    The axis is the direction all the normals are square to; for thin pipes,
    whose normals are too rough for that, the direction the points run in.
    ``sensors`` (where the scanner of each point stood) rule out inside
    corners, which look like a quarter of a pipe from the room: a real pipe
    bulges towards the scanner."""
    candidates = []
    m = normals.T @ normals / len(normals)
    w, v = np.linalg.eigh(m)
    if w[1] >= 0.05 * w[2]:  # normals spread round an axis
        candidates.append(v[:, 0])
    c0 = pts.mean(axis=0)
    pw, pv = np.linalg.eigh(np.cov((pts - c0).T))
    if pw[2] > 6 * pw[1]:  # long and thin
        candidates.append(pv[:, 2])
    best = None
    for axis in candidates:
        e = _cylinder_along(pts, axis, tol, smallest, largest, sensors)
        if e is not None and (best is None or e.rms < best.rms):
            best = e
    return best


def _cylinder_along(pts, axis, tol, smallest, largest, sensors):
    # square to the world if nearly upright or level
    if abs(axis[2]) > math.cos(math.radians(5)):
        axis = UP.copy()
    elif abs(axis[2]) < math.sin(math.radians(5)):
        axis = unit([axis[0], axis[1], 0.0])
    e1 = unit(np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0]))
    e2 = np.cross(axis, e1)
    xy = np.column_stack([pts @ e1, pts @ e2])
    c, r, rms = fit_circle(xy, refine=False)  # quick look first: most groups are not round
    if not np.isfinite(rms) or not smallest / 2 <= r <= largest * 2 or rms > 0.5 * r:
        return None
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(2):
        xy = np.column_stack([pts[keep] @ e1, pts[keep] @ e2])
        c, r, rms = fit_circle(xy)
        if not np.isfinite(rms) or not smallest <= r <= largest:
            return None
        d = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        inl = d <= max(2 * tol, 0.08 * r)
        if inl.mean() < 0.6 or inl.sum() < 20:
            return None
        idx = np.nonzero(keep)[0]
        keep[idx[~inl]] = False
    sel = pts[keep]
    xy = np.column_stack([sel @ e1, sel @ e2])
    c, r, rms = fit_circle(xy)
    if rms > max(tol, 0.1 * r) or not smallest <= r <= largest:
        return None
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    if len(np.unique(np.floor((ang + np.pi) / (np.pi / 18)))) < 9:  # under a quarter round
        return None
    if sensors is not None:
        sn = sensors[keep]
        ok = np.isfinite(sn).all(axis=1)
        if ok.sum() >= 10:
            sxy = np.column_stack([sn[ok] @ e1, sn[ok] @ e2])
            out_dir = unit(xy[ok] - c)
            to_scanner = unit(sxy - c)
            if float(np.mean(np.einsum("ij,ij->i", out_dir, to_scanner))) < 0.2:
                return None  # hollow towards the scanner: an inside corner
    t = sel @ axis
    t0, t1 = float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))
    length = t1 - t0
    if length < max(0.3, 2 * r):
        return None
    base = c[0] * e1 + c[1] * e2 + t0 * axis
    kind = "column" if abs(axis[2]) > 0.99 and r >= 0.08 else "pipe"
    return cylinder(kind, base, axis, r, length, rms=rms, points=len(sel))


# ---------------------------------------------------------------- the whole job


def fit_model(site, s: FitSettings | None = None, progress=None, levels=None):
    """Elements fitted to ``site`` (:class:`~pcd_merge.cad.points.SitePoints`,
    already thinned to ``s.detail``). Returns (elements, faces).

    ``levels`` (the building's floors and ceilings) tell the building from
    the furniture; they are looked for in ``site`` when not given."""
    s = s or FitSettings()
    xyz = site.xyz
    if len(xyz) < 50:
        return [], []
    report = progress or (lambda _m, _f: None)
    report("Finding surfaces", 0.02)
    sensors = site.sensor_xyz()
    if s.skip_clutter and not levels:
        from .levels import find_levels

        report("Looking for floors and ceilings", 0.02)
        levels = find_levels(xyz, site.sensor_z())
    normals, off, flat, oriented = surface_normals(
        xyz, sensors, s.detail, site.scan,
        progress=lambda m, f: report(m, 0.05 + 0.4 * f))
    report("Growing flat patches", 0.45)
    labels = grow_patches(xyz, normals, off, flat, oriented, s.detail, s.tolerance,
                          progress=lambda f: report("Growing flat patches", 0.45 + 0.25 * f))
    elements: list[Element] = []
    faces: list[Face] = []
    report("Fitting planes", 0.7)
    faces = make_faces(xyz, normals, labels, s, levels)
    # points of patches too small to keep are left for the cylinder search
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    small = ids[counts * s.detail * s.detail < s.min_area]
    labels = np.where(np.isin(labels, small), -1, labels)
    if s.planes:
        elements += walls_from_faces(faces, s)
        elements += slabs_from_faces(faces, s)
        elements += roofs_from_faces(faces, s)
    if s.cylinders:
        report("Looking for columns and pipes", 0.85)
        found = find_cylinders(xyz, normals, labels, s, sensors)
        if s.skip_clutter:  # the edges of shelf boards look like short thin pipes
            found = [e for e in found if e.kind != "pipe" or e.size[2] >= SHORTEST_PIPE]
        elements += found
    report("Fitted", 1.0)
    return elements, faces
