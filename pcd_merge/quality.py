"""How well do the aligned clouds agree? Scores per pair and per point.

Everything is measured after alignment, with the final transforms, so it
reflects what will actually be merged (for three or more clouds that is after
the pose graph, not the raw pairwise result).

Distances are point-to-plane: from a point to the local surface of the other
cloud, not to its nearest sample. That removes most of the sampling spacing
between two scans of the same surface, so what is left is sensor noise plus
misalignment.

Only points that have a neighbour in another cloud within ``threshold``, on a
surface facing the same way, count as overlapping (the other face of a wall,
seen from the next room, is a different surface). The rest are areas only one scan saw. They are not errors and
are left out of the scores (and drawn grey in the heatmap).

Things that changed between scans (a parked car that left, a door that
opened, people) give large distances too, but they say nothing about the
alignment. So the grade does not use the raw distances. It uses:

* the **offset**: how far apart the two scans' shared surfaces still are, in
  the worst-off regions, counting only what one rigid movement of the scan
  could fix (see :func:`_offset`). A good alignment gives ~0; a misaligned one
  gives the size of the misalignment. Changed objects barely move it, because
  no single movement explains them, and sliding along a long flat site
  (which no surface pins down) doesn't count either;
* the **agreement**: the share of overlapping points within
  ``change_threshold`` of the other cloud. Points beyond it are "changed"
  (drawn magenta in the heatmap) and left out of the fit RMSE. Only when
  most of the overlap disagrees (typical of a wrong alignment, not of a few
  moved objects) does this lower the grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .cleanup import plausible_origin
from .registration import to_o3d, voxel_down

reg = o3d.pipelines.registration

# Grading, as multiples of the voxel size used for alignment. The voxel size
# scales with the scans (about 1/400 of their extent or coarser), so the same
# limits work for indoor and outdoor scans.
GOOD_OFFSET = 0.20  # 95% of overlapping points would move less than this
FAIR_OFFSET = 0.50
# ... or, for noisy scans, these multiples of the noise (if larger).
GOOD_NOISE = 1.5
FAIR_NOISE = 3.0
# Pairs need this much overlap for Good / Fair. Two scans can be well aligned
# with little overlap; they just give less evidence. (Overlap only counts
# surfaces both scans saw from the same side, so it is lower than the share
# of points that are merely close to the other scan.)
GOOD_OVERLAP = 0.15
FAIR_OVERLAP = 0.05
GOOD_AGREEMENT = 0.60  # share of overlapping points that match (not changed)
FAIR_AGREEMENT = 0.40
# Points further than this from the other cloud count as changed. Raised
# automatically for noisy scans (several times the typical distance).
CHANGE_THRESHOLD = 0.30
# Two points are on the same surface only if their normals are within ~60°.
SAME_FACE = 0.5
# Offset: gaps are measured in regions CELL_VOXELS voxels a side, per surface
# direction, each needing MIN_GROUP matched points.
CELL_VOXELS = 20
MIN_GROUP = 30
MIN_AXIS_SHARE = 0.02  # share of the matched points a surface direction needs

GRADES = ("Good", "Fair", "Poor")


@dataclass
class PairQuality:
    a: int
    b: int
    overlap: float  # fraction of points with a close match (better direction)
    rmse: float  # RMS point-to-plane distance over the matching (unchanged) points
    median: float  # median distance over all overlapping points
    p90: float
    grade: str
    offset: float = 0.0  # estimated misalignment (95th percentile point shift)
    agreement: float = 1.0  # share of overlapping points that are not changed
    noise: float = np.nan  # spread of the distances within a region (both scans' noise)

    @property
    def overlaps(self) -> bool:
        return self.overlap > 0.02

    @property
    def changed(self) -> float:
        return 1.0 - self.agreement


@dataclass
class QualityReport:
    voxel_size: float
    threshold: float  # max distance still counted as overlap
    pairs: list[PairQuality]
    cloud_overlap: list[float]  # per cloud, vs. all the others
    cloud_rmse: list[float]
    # Per preview point: distance to the other clouds' surface, NaN where no
    # other cloud is within ``threshold``. One array per input cloud.
    point_error: list[np.ndarray] = field(repr=False, default_factory=list)
    change_threshold: float = np.inf  # distances above this are "changed"
    cloud_changed: list[float] = field(default_factory=list)

    @property
    def scale_max(self) -> float:
        """Upper end of the heatmap colour scale (red); beyond it is "changed"."""
        return self.change_threshold

    def counted(self) -> list[PairQuality]:
        """The pairs the overall grade is made of.

        With three or more scans, scans at opposite ends of a site often
        share only a sliver. Such pairs are measured but don't decide the
        grade, as long as every scan has at least one pair with real overlap
        (otherwise its best-overlapping pair counts)."""
        ps = [p for p in self.pairs if p.overlaps]
        main = [p for p in ps if p.overlap >= GOOD_OVERLAP]
        n = len(self.cloud_overlap)
        for i in range(n):
            if not any(i in (p.a, p.b) for p in main):
                mine = [p for p in ps if i in (p.a, p.b)]
                if mine:
                    best = max(mine, key=lambda p: p.overlap)
                    if all(best is not q for q in main):
                        main.append(best)
        return main

    @property
    def grade(self) -> str:
        """Overall grade: the worst of the pairs that count (see :meth:`counted`)."""
        graded = [p.grade for p in self.counted()]
        if not graded:
            return "Poor"
        return max(graded, key=GRADES.index)

    @property
    def score(self) -> float:
        """One number for comparing alignments of the same clouds, graded
        with the same voxel size (higher is better). The whole part is the
        grade (2 Good, 1 Fair, 0 Poor); the fraction rewards a smaller offset
        first, then more agreement, then more overlap (in 5% steps, so noise
        doesn't decide)."""
        ps = self.counted()
        if not ps:
            return 0.0
        offsets = [p.offset for p in ps]
        worst = max(offsets) / self.voxel_size if all(np.isfinite(offsets)) else np.inf
        agreement = min(p.agreement for p in ps)
        overlap = round(float(np.mean([p.overlap for p in ps])) / 0.05) * 0.05
        rank = 2 - GRADES.index(self.grade)
        return rank + 0.45 / (1.0 + worst) + 0.3 * agreement + 0.2 * min(overlap, 1.0)

    def lines(self, names: list[str] | None = None) -> list[str]:
        """Plain-text summary for the report panel or a saved report."""
        n = len(self.cloud_overlap)
        names = names or [f"cloud {i + 1}" for i in range(n)]
        out = [f"Alignment quality: {self.grade}", ""]
        out.append("Pairs:")
        counted = self.counted()
        for p in self.pairs:
            if not p.overlaps:
                out.append(f"  {p.a + 1} ↔ {p.b + 1}: no overlap")
                continue
            note = "" if any(p is q for q in counted) else " (little overlap, not counted)"
            out.append(f"  {p.a + 1} ↔ {p.b + 1}: {p.grade}, overlap {p.overlap:.0%}, "
                       f"offset ≈ {_fmt(p.offset)}{note}")
            out.append(f"         fit {_fmt(p.rmse)} rms, changed {p.changed:.0%}")
        out += ["", "Each cloud vs. the rest:"]
        for i in range(n):
            rm = _fmt(self.cloud_rmse[i]) if np.isfinite(self.cloud_rmse[i]) else "-"
            ch = f", changed {self.cloud_changed[i]:.0%}" if self.cloud_changed else ""
            out.append(f"  {i + 1} {names[i]}: overlap {self.cloud_overlap[i]:.0%}, "
                       f"fit {rm}{ch}")
        out += ["", "Offset: how far apart the scans' shared surfaces still are, where "
                "they are furthest apart (the part one movement of the scan could fix). "
                "Changed: overlapping points that differ by more than "
                f"{_fmt(self.change_threshold)} (moved objects, people, doors); "
                "they do not count against the grade unless most of the overlap differs."]
        return out

    def to_dict(self) -> dict:
        def num(x):
            return float(x) if np.isfinite(x) else None

        return {
            "grade": self.grade,
            "voxel_size": self.voxel_size,
            "overlap_threshold": self.threshold,
            "change_threshold": num(self.change_threshold),
            "pairs": [
                {"clouds": [p.a, p.b], "overlap": p.overlap, "offset": num(p.offset),
                 "rmse": num(p.rmse), "median": num(p.median), "p90": num(p.p90),
                 "changed": p.changed, "noise": num(p.noise), "grade": p.grade}
                for p in self.pairs
            ],
            "clouds": [
                {"overlap": o, "rmse": num(r),
                 "changed": (self.cloud_changed[i] if self.cloud_changed else None)}
                for i, (o, r) in enumerate(zip(self.cloud_overlap, self.cloud_rmse))
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, point_error=None) -> "QualityReport":
        """Inverse of :meth:`to_dict` (``point_error`` comes from a cache)."""
        def num(x, default=np.nan):
            return default if x is None else float(x)

        pairs = [PairQuality(a=p["clouds"][0], b=p["clouds"][1], overlap=p["overlap"],
                             rmse=num(p["rmse"]), median=num(p["median"]), p90=num(p["p90"]),
                             grade=p["grade"], offset=num(p["offset"]),
                             agreement=1.0 - p.get("changed", 0.0),
                             noise=num(p.get("noise")))
                 for p in d.get("pairs", [])]
        clouds = d.get("clouds", [])
        changed = [c["changed"] for c in clouds] if clouds and all(
            c.get("changed") is not None for c in clouds) else []
        return cls(voxel_size=d["voxel_size"], threshold=d["overlap_threshold"], pairs=pairs,
                   cloud_overlap=[c["overlap"] for c in clouds],
                   cloud_rmse=[num(c["rmse"]) for c in clouds],
                   point_error=list(point_error or []),
                   change_threshold=num(d.get("change_threshold"), np.inf),
                   cloud_changed=changed)


def _fmt(d: float) -> str:
    """Distance in readable units, assuming the scans are in metres."""
    if not np.isfinite(d):
        return "-"
    if d < 0.01:
        return f"{d * 1000:.1f} mm"
    if d < 1:
        return f"{d * 100:.1f} cm"
    return f"{d:.2f} m"


fmt_distance = _fmt


def grade_pair(overlap: float, offset: float, agreement: float, voxel: float,
               noise: float = 0.0) -> str:
    """Grade one pair. The offset limits are a share of the voxel size, or a
    multiple of the scans' own ``noise`` when that is larger: a gap thinner
    than the scans' noise can't be seen (or measured) and is not an error."""
    if not np.isfinite(offset):
        return "Poor"
    noise = noise if np.isfinite(noise) else 0.0
    if (overlap >= GOOD_OVERLAP and offset <= max(GOOD_OFFSET * voxel, GOOD_NOISE * noise)
            and agreement >= GOOD_AGREEMENT):
        return "Good"
    if (overlap >= FAIR_OVERLAP and offset <= max(FAIR_OFFSET * voxel, FAIR_NOISE * noise)
            and agreement >= FAIR_AGREEMENT):
        return "Fair"
    return "Poor"


def _surface(xyz, t: np.ndarray, voxel: float, origin=None):
    """Aligned, downsampled copy with normals: the 'surface' others are measured to.

    ``xyz`` may also be a function returning the coordinates, or the cloud
    already downsampled at ``voxel * 0.5`` (``AlignResult.fine``), which
    skips the pass over every full-resolution point.

    Returns ``(cloud, oriented)``. When the scan surrounds ``origin`` (its
    sensor position, in the scan's own frame) the normals are turned to face
    the sensor, so the two faces of a wall, seen from the two rooms, get
    opposite normals and are not mistaken for the same surface.
    """
    if isinstance(xyz, o3d.geometry.PointCloud):
        pc = o3d.geometry.PointCloud(xyz)
    else:
        if callable(xyz):  # see registration.align
            xyz = xyz()
        pc = voxel_down(xyz, voxel * 0.5)  # chunked: keeps the window responsive
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)[:3]
    oriented = bool(np.all(np.isfinite(origin))) and plausible_origin(np.asarray(pc.points),
                                                                     origin)
    pc.transform(t)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 1.5, max_nn=30))
    if oriented:
        pc.orient_normals_towards_camera_location(t[:3, :3] @ origin + t[:3, 3])
    return pc, oriented


def _nearest(points: np.ndarray, cloud: o3d.geometry.PointCloud, radius: float) -> np.ndarray:
    """Index of the nearest point of ``cloud`` within ``radius`` (-1 where none)."""
    idx = np.full(len(points), -1, dtype=np.int64)
    if len(points) == 0 or len(cloud.points) == 0:
        return idx
    ev = reg.evaluate_registration(to_o3d(points), cloud, radius, np.eye(4))
    corr = np.asarray(ev.correspondence_set)
    if len(corr):
        idx[corr[:, 0]] = corr[:, 1]
    return idx


def surface_distance(points: np.ndarray, surface: o3d.geometry.PointCloud,
                     threshold: float, normals: np.ndarray | None = None,
                     oriented: bool = False) -> np.ndarray:
    """Point-to-plane distance from each point to ``surface`` (NaN beyond ``threshold``).

    With ``normals`` (one per point) a point only counts when its surface
    faces the same way as the one it is measured to: a point on a wall is not
    compared with the floor next to it, and, when both are ``oriented``
    towards their scanners, not with the other face of the wall either.
    Those are different surfaces, not misalignment.
    """
    out = np.full(len(points), np.nan)
    if len(points) == 0 or len(surface.points) == 0:
        return out
    ev = reg.evaluate_registration(to_o3d(points), surface, threshold, np.eye(4))
    corr = np.asarray(ev.correspondence_set)
    if len(corr) == 0:
        return out
    q = np.asarray(surface.points)[corr[:, 1]]
    nrm = np.asarray(surface.normals)[corr[:, 1]]
    d = np.abs(np.einsum("ij,ij->i", points[corr[:, 0]] - q, nrm))
    if normals is not None and len(normals) == len(points):
        cos = np.einsum("ij,ij->i", np.asarray(normals)[corr[:, 0]], nrm)
        same = (cos > SAME_FACE) if oriented else (np.abs(cos) > SAME_FACE)
        corr, d = corr[same], d[same]
    out[corr[:, 0]] = d
    return out


def _gap_groups(points, normals, surface, threshold, cell):
    """Where, and by how much, ``points`` sit off ``surface``, region by region.

    Each overlapping point's signed point-to-plane distance to the other
    surface is grouped by where it is (a ``cell``-sized block) and which way
    its surface faces. Within a group, sensor noise averages out and what is
    left is how far that piece of surface sits in front of or behind the
    other scan's. Returns ``(centre, normal, gap, weight, spread)`` per group;
    the spread (robust standard deviation of the distances) is the two
    scans' noise there, and does not depend on the alignment.
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0))
    if len(points) == 0 or len(surface.points) == 0:
        return empty
    ev = reg.evaluate_registration(to_o3d(points), surface, threshold, np.eye(4))
    corr = np.asarray(ev.correspondence_set)
    if len(corr) == 0:
        return empty
    nrm = np.asarray(surface.normals)[corr[:, 1]]
    p = points[corr[:, 0]]
    cos = np.einsum("ij,ij->i", normals[corr[:, 0]], nrm)
    keep = np.abs(cos) > SAME_FACE
    p, nrm = p[keep], nrm[keep]
    if len(p) == 0:
        return empty
    # The same sign for every normal of a group: dominant component positive.
    axis = np.argmax(np.abs(nrm), axis=1)
    nrm = nrm * np.sign(nrm[np.arange(len(nrm)), axis] + 1e-12)[:, None]
    d = np.einsum("ij,ij->i", p - np.asarray(surface.points)[corr[keep, 1]], nrm)
    keys = np.c_[np.floor(p / cell).astype(np.int64), axis]
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(group.ravel(), kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    p, nrm, d = p[order], nrm[order], d[order]
    out = []
    for s, c in zip(starts, counts):
        if c < MIN_GROUP:
            continue
        g = d[s:s + c]
        m = np.median(g)
        # Take off what noise alone could give: twice the standard error of
        # the median (from the robust spread), so cluttered surfaces
        # (shelving, plants) don't read as a gap.
        spread = 1.4826 * np.median(np.abs(g - m))
        se = 1.2533 * spread / np.sqrt(c)
        n = nrm[s:s + c].mean(axis=0)
        out.append((p[s:s + c].mean(axis=0), n / max(np.linalg.norm(n), 1e-12),
                    np.sign(m) * max(0.0, abs(m) - 2.0 * se), float(c), spread))
    if not out:
        return empty
    c, n, m, w, sp = zip(*out)
    return np.array(c), np.array(n), np.array(m), np.array(w), np.array(sp)


def _offset(sa, na, surf_b, sb, nb, surf_a, voxel, change, threshold):
    """Estimated misalignment of two aligned clouds: how far apart their
    shared surfaces still are where they are furthest apart, counting only
    the part that one rigid movement could fix.

    The gaps between the two scans' surfaces are measured region by region
    (:func:`_gap_groups`); then the small rigid movement that best explains
    them is fitted, robustly, and the 90th percentile of the gaps *it*
    predicts at those regions is the offset.

    * Things that moved between scans (a pallet pushed back, a door) leave a
      gap in one region that no rigid movement of the whole scan explains,
      so they hardly count.
    * Directions that no surface pins down (sliding along a long, flat
      warehouse or corridor) don't count either: a movement there predicts no
      gap at any surface. The earlier measure (how far an ICP would still
      move the points) reported exactly that sliding as a large error.

    Returns ``(offset, noise)``: noise is the typical spread of the distances
    within a region, i.e. how thick the two scans' surfaces are together.
    """
    cell = voxel * CELL_VOXELS
    ca, nra, ma, wa, spa = _gap_groups(sb, nb, surf_a, threshold, cell)  # b off a
    cb, nrb, mb, wb, spb = _gap_groups(sa, na, surf_b, threshold, cell)  # a off b
    c = np.r_[ca, cb]
    n = np.r_[nra, nrb]
    gap = np.r_[ma, -mb]  # as movements of b relative to a
    w = np.r_[wa, wb]
    if len(w) < 6 or w.sum() < 10 * MIN_GROUP:
        return np.nan, np.nan
    noise = float(np.average(np.r_[spa, spb], weights=w))
    c = c - np.average(c, axis=0, weights=w)
    A = np.c_[np.cross(c, n), n]  # gap ~ A @ (rotation, translation)
    # Ridge so that movements no surface pins down stay at zero.
    scale = np.r_[np.full(3, max(np.sqrt(np.average((c ** 2).sum(1), weights=w)), 1e-9)),
                  np.ones(3)]
    As = A / scale
    wr = w / w.sum()
    floor = 0.05 * change
    x = np.zeros(6)
    # Iteratively reweighted least squares: plain least squares first (so a
    # gap shown only by the few surfaces facing one way is fitted, not
    # dismissed), then Huber, then Tukey to drop what moved.
    for it in range(12):
        r = gap - As @ x
        sigma = max(1.4826 * np.median(np.abs(r - np.median(r))), floor)
        if it == 0:
            k = wr.copy()
        elif it < 5:
            k = wr * np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(r), 1e-12))
        else:
            u = r / (4.685 * sigma)
            k = wr * np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        H = As.T @ (As * k[:, None])
        H += np.eye(6) * 1e-3 * max(np.trace(H) / 6, 1e-12)
        x = np.linalg.solve(H, As.T @ (k * gap))
    # The fitted movement's gap at each region; the 90th percentile per
    # surface direction (so a sideways shift shows in full on the walls even
    # when the floor is most of the overlap), and the worst direction.
    pred = np.abs(As @ x)
    axis = np.argmax(np.abs(n), axis=1)
    worst = 0.0
    for k in range(3):
        sel = axis == k
        if w[sel].sum() < max(MIN_AXIS_SHARE * w.sum(), 3 * MIN_GROUP):
            continue
        v, ww = pred[sel], w[sel]
        o = np.argsort(v)
        cum = np.cumsum(ww[o]) / ww.sum()
        worst = max(worst, float(v[o][min(np.searchsorted(cum, 0.9), len(o) - 1)]))
    return worst, noise


def _split(d: np.ndarray, change: float):
    """(overlap, rmse of matching points, median, p90, agreement) for distances."""
    hit = d[np.isfinite(d)]
    overlap = len(hit) / max(1, len(d))
    if len(hit) == 0:
        return overlap, np.nan, np.nan, np.nan, 0.0
    match = hit[hit <= change]
    rmse = float(np.sqrt(np.mean(match ** 2))) if len(match) else np.nan
    return (overlap, rmse, float(np.median(hit)), float(np.percentile(hit, 90)),
            len(match) / len(hit))


def assess(clouds_xyz: list[np.ndarray], transforms: list[np.ndarray], voxel: float,
           preview_xyz: list[np.ndarray] | None = None,
           origins: list | None = None) -> QualityReport:
    """Score an alignment.

    ``clouds_xyz`` and ``transforms`` are what was passed to and returned by
    :func:`registration.align`. ``preview_xyz`` (optional, untransformed, one
    array per cloud) are the points to colour in the heatmap. ``origins``
    (optional) are the scanner positions in each scan's own frame (the PCD
    VIEWPOINT); by default the scan's origin is tried.
    """
    n = len(clouds_xyz)
    if n != len(transforms):
        raise ValueError("one transform per cloud is required")
    transforms = [np.asarray(t, dtype=np.float64) for t in transforms]
    origins = list(origins) if origins is not None else [None] * n
    threshold = voxel * 1.5
    built = [_surface(x, t, voxel, o) for x, t, o in zip(clouds_xyz, transforms, origins)]
    surfaces = [s for s, _ in built]
    oriented = [o for _, o in built]
    samples = [np.asarray(s.points) for s in surfaces]
    normals = [np.asarray(s.normals) for s in surfaces]

    def dist(i, surf, j):
        return surface_distance(samples[i], surf, threshold, normals[i],
                                oriented[i] and (j is not None and oriented[j]))

    dists = {}
    for a in range(n):
        for b in range(a + 1, n):
            dists[a, b] = (dist(a, surfaces[b], b), dist(b, surfaces[a], a))
    # "Changed" threshold: fixed share of the voxel, or several times the
    # typical distance for scans noisier than that.
    pooled = np.concatenate([np.concatenate(v) for v in dists.values()]) if dists else []
    pooled = np.asarray(pooled)
    pooled = pooled[np.isfinite(pooled)]
    typical = float(np.median(pooled)) if len(pooled) else 0.0
    change = min(max(CHANGE_THRESHOLD * voxel, 4.5 * typical), threshold)

    pairs = []
    for (a, b), (ab, ba) in dists.items():
        # Overlap is asymmetric (a small scan inside a big one); use the
        # direction where it is larger, and pool the distances of both.
        ov = max(_split(ab, change)[0], _split(ba, change)[0])
        _, rmse, med, p90, agree = _split(np.concatenate([ab, ba]), change)
        offset = noise = np.nan
        if ov > 0.02:
            ia, ib = np.isfinite(ab), np.isfinite(ba)  # same surface, within reach
            offset, noise = _offset(samples[a][ia], normals[a][ia], surfaces[b],
                             samples[b][ib], normals[b][ib], surfaces[a],
                             voxel, change, threshold)
        grade = grade_pair(ov, offset, agree, voxel, noise)
        pairs.append(PairQuality(a, b, ov, rmse, med, p90, grade, offset, agree, noise))

    cloud_overlap, cloud_rmse, cloud_changed = [], [], []
    others = []
    for i in range(n):
        o = o3d.geometry.PointCloud()
        for j in range(n):
            if j != i:
                o += surfaces[j]
        others.append(o)
        both = all(oriented[j] for j in range(n))
        d = surface_distance(samples[i], o, threshold, normals[i], both)
        ov, rmse, _, _, agree = _split(d, change)
        cloud_overlap.append(ov)
        cloud_rmse.append(rmse)
        cloud_changed.append(1.0 - agree if ov > 0 else 0.0)

    point_error = []
    if preview_xyz is not None:
        both = all(oriented)
        for i, (xyz, t) in enumerate(zip(preview_xyz, transforms)):
            pts = np.asarray(xyz, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]
            # Each preview point takes the normal of its own scan's surface
            # there, so it is only compared with surfaces facing the same way.
            near = _nearest(pts, surfaces[i], voxel)
            nrm = np.zeros_like(pts)
            nrm[near >= 0] = normals[i][near[near >= 0]]
            if len(pts):
                nrm[near < 0] = (1.0, 0.0, 0.0)
            err = surface_distance(pts, others[i], threshold, nrm, both)
            if len(pts):  # no nearby own surface (should be rare): unfiltered
                err[near < 0] = surface_distance(pts[near < 0], others[i], threshold)
            point_error.append(err)

    return QualityReport(voxel, threshold, pairs, cloud_overlap, cloud_rmse, point_error,
                         change, cloud_changed)


# --------------------------------------------------------------------------
# Heatmap colours

# Blue (fits) -> cyan -> green -> yellow -> red (at scale_max, the "changed"
# threshold). Beyond it: magenta, "changed" (moved objects, or misalignment).
_STOPS = np.array([
    [0.15, 0.35, 0.95],
    [0.10, 0.80, 0.90],
    [0.20, 0.85, 0.30],
    [0.98, 0.85, 0.15],
    [0.95, 0.20, 0.15],
])
NO_OVERLAP = (0.45, 0.45, 0.48)
CHANGED = (0.85, 0.25, 0.85)


def error_colours(err: np.ndarray, scale_max: float, alpha: float = 1.0) -> np.ndarray:
    """RGBA (N, 4) float32 colours for distances; NaN (no overlap) is grey,
    above ``scale_max`` (changed) is magenta."""
    err = np.asarray(err, dtype=np.float64)
    out = np.empty((len(err), 4), np.float32)
    out[:, 3] = alpha
    out[:, :3] = NO_OVERLAP
    ok = np.isfinite(err)
    if ok.any():
        x = np.clip(err[ok] / max(scale_max, 1e-12), 0, 1) * (len(_STOPS) - 1)
        k = np.minimum(x.astype(int), len(_STOPS) - 2)
        f = (x - k)[:, None]
        out[ok, :3] = _STOPS[k] * (1 - f) + _STOPS[k + 1] * f
        out[ok & (err > scale_max), :3] = CHANGED
    return out


def colour_stops() -> list[tuple[float, tuple[float, float, float]]]:
    """(position 0..1, rgb) stops, for drawing a legend."""
    m = len(_STOPS) - 1
    return [(k / m, tuple(float(v) for v in _STOPS[k])) for k in range(len(_STOPS))]
