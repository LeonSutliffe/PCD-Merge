"""How well do the aligned clouds agree? Scores per pair and per point.

Everything is measured after alignment, with the final transforms, so it
reflects what will actually be merged (for three or more clouds that is after
the pose graph, not the raw pairwise result).

Distances are point-to-plane: from a point to the local surface of the other
cloud, not to its nearest sample. That removes most of the sampling spacing
between two scans of the same surface, so what is left is sensor noise plus
misalignment.

Only points that have a neighbour in another cloud within ``threshold`` count
as overlapping. The rest are areas only one scan saw. They are not errors and
are left out of the scores (and drawn grey in the heatmap).

Things that changed between scans (a parked car that left, a door that
opened, people) give large distances too, but they say nothing about the
alignment. So the grade does not use the raw distances. It uses:

* the **offset**: how far a robust ICP (Tukey loss, which ignores points that
  don't fit) would still move one cloud onto the other. A good alignment
  gives ~0; a misaligned one gives the size of the misalignment. Changed
  objects barely move it, because the loss ignores them;
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

from .registration import to_o3d, voxel_down

reg = o3d.pipelines.registration

# Grading, as multiples of the voxel size used for alignment. The voxel size
# scales with the scans (about 1/400 of their extent or coarser), so the same
# limits work for indoor and outdoor scans.
GOOD_OFFSET = 0.20  # 95% of overlapping points would move less than this
FAIR_OFFSET = 0.50
GOOD_OVERLAP = 0.30
FAIR_OVERLAP = 0.15
GOOD_AGREEMENT = 0.60  # share of overlapping points that match (not changed)
FAIR_AGREEMENT = 0.40
# Points further than this from the other cloud count as changed. Raised
# automatically for noisy scans (several times the typical distance).
CHANGE_THRESHOLD = 0.30

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

    @property
    def grade(self) -> str:
        """Overall grade: the worst of the pairs that should overlap."""
        graded = [p.grade for p in self.pairs if p.overlaps]
        if not graded:
            return "Poor"
        return max(graded, key=GRADES.index)

    @property
    def score(self) -> float:
        """One number for comparing alignments of the same clouds (higher is
        better): grade first, then the worst offset, then agreement."""
        ps = [p for p in self.pairs if p.overlaps]
        if not ps:
            return 0.0
        worst_offset = max(p.offset for p in ps) / self.voxel_size
        agreement = min(p.agreement for p in ps)
        rank = 2 - GRADES.index(self.grade)
        return rank + 0.5 / (1.0 + worst_offset) + 0.5 * agreement / 2

    def lines(self, names: list[str] | None = None) -> list[str]:
        """Plain-text summary for the report panel or a saved report."""
        n = len(self.cloud_overlap)
        names = names or [f"cloud {i + 1}" for i in range(n)]
        out = [f"Alignment quality: {self.grade}", ""]
        out.append("Pairs:")
        for p in self.pairs:
            if not p.overlaps:
                out.append(f"  {p.a + 1} ↔ {p.b + 1}: no overlap")
                continue
            out.append(f"  {p.a + 1} ↔ {p.b + 1}: {p.grade}, overlap {p.overlap:.0%}, "
                       f"offset ≈ {_fmt(p.offset)}")
            out.append(f"         fit {_fmt(p.rmse)} rms, changed {p.changed:.0%}")
        out += ["", "Each cloud vs. the rest:"]
        for i in range(n):
            rm = _fmt(self.cloud_rmse[i]) if np.isfinite(self.cloud_rmse[i]) else "-"
            ch = f", changed {self.cloud_changed[i]:.0%}" if self.cloud_changed else ""
            out.append(f"  {i + 1} {names[i]}: overlap {self.cloud_overlap[i]:.0%}, "
                       f"fit {rm}{ch}")
        out += ["", "Offset: how far the clouds still seem to be from the best fit. "
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
                 "changed": p.changed, "grade": p.grade}
                for p in self.pairs
            ],
            "clouds": [
                {"overlap": o, "rmse": num(r),
                 "changed": (self.cloud_changed[i] if self.cloud_changed else None)}
                for i, (o, r) in enumerate(zip(self.cloud_overlap, self.cloud_rmse))
            ],
        }


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


def grade_pair(overlap: float, offset: float, agreement: float, voxel: float) -> str:
    if not np.isfinite(offset):
        return "Poor"
    if (overlap >= GOOD_OVERLAP and offset <= GOOD_OFFSET * voxel
            and agreement >= GOOD_AGREEMENT):
        return "Good"
    if (overlap >= FAIR_OVERLAP and offset <= FAIR_OFFSET * voxel
            and agreement >= FAIR_AGREEMENT):
        return "Fair"
    return "Poor"


def _surface(xyz, t: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    """Aligned, downsampled copy with normals: the 'surface' others are measured to.

    ``xyz`` may also be a function returning the coordinates, or the cloud
    already downsampled at ``voxel * 0.5`` (``AlignResult.fine``), which
    skips the pass over every full-resolution point.
    """
    if isinstance(xyz, o3d.geometry.PointCloud):
        pc = o3d.geometry.PointCloud(xyz)
    else:
        if callable(xyz):  # see registration.align
            xyz = xyz()
        pc = voxel_down(xyz, voxel * 0.5)  # chunked: keeps the window responsive
    pc.transform(t)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 1.5, max_nn=30))
    return pc


def surface_distance(points: np.ndarray, surface: o3d.geometry.PointCloud,
                     threshold: float) -> np.ndarray:
    """Point-to-plane distance from each point to ``surface`` (NaN beyond ``threshold``)."""
    out = np.full(len(points), np.nan)
    if len(points) == 0 or len(surface.points) == 0:
        return out
    ev = reg.evaluate_registration(to_o3d(points), surface, threshold, np.eye(4))
    corr = np.asarray(ev.correspondence_set)
    if len(corr) == 0:
        return out
    q = np.asarray(surface.points)[corr[:, 1]]
    nrm = np.asarray(surface.normals)[corr[:, 1]]
    out[corr[:, 0]] = np.abs(np.einsum("ij,ij->i", points[corr[:, 0]] - q, nrm))
    return out


def _offset(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, voxel: float,
            change: float, threshold: float) -> float:
    """Estimated misalignment of ``src`` against ``tgt`` (both already aligned).

    Runs a robust point-to-plane ICP from where the clouds are now and returns
    the 95th percentile of how far it moves the overlapping points of ``src``.
    """
    est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=change))
    r = reg.registration_icp(src, tgt, threshold, np.eye(4), est,
                             reg.ICPConvergenceCriteria(max_iteration=60))
    corr = np.asarray(r.correspondence_set)
    if len(corr) < 10:
        return np.nan
    p = np.asarray(src.points)[np.unique(corr[:, 0])]
    t = r.transformation
    shift = np.linalg.norm(p @ t[:3, :3].T + t[:3, 3] - p, axis=1)
    return float(np.percentile(shift, 95))


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
           preview_xyz: list[np.ndarray] | None = None) -> QualityReport:
    """Score an alignment.

    ``clouds_xyz`` and ``transforms`` are what was passed to and returned by
    :func:`registration.align`. ``preview_xyz`` (optional, untransformed, one
    array per cloud) are the points to colour in the heatmap.
    """
    n = len(clouds_xyz)
    if n != len(transforms):
        raise ValueError("one transform per cloud is required")
    threshold = voxel * 1.5
    surfaces = [_surface(x, t, voxel) for x, t in zip(clouds_xyz, transforms)]
    samples = [np.asarray(s.points) for s in surfaces]

    dists = {}
    for a in range(n):
        for b in range(a + 1, n):
            dists[a, b] = (surface_distance(samples[a], surfaces[b], threshold),
                           surface_distance(samples[b], surfaces[a], threshold))
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
        offset = _offset(surfaces[a], surfaces[b], voxel, change, threshold) \
            if ov > 0.02 else np.nan
        grade = grade_pair(ov, offset, agree, voxel)
        pairs.append(PairQuality(a, b, ov, rmse, med, p90, grade, offset, agree))

    cloud_overlap, cloud_rmse, cloud_changed = [], [], []
    others = []
    for i in range(n):
        o = o3d.geometry.PointCloud()
        for j in range(n):
            if j != i:
                o += surfaces[j]
        others.append(o)
        d = surface_distance(samples[i], o, threshold)
        ov, rmse, _, _, agree = _split(d, change)
        cloud_overlap.append(ov)
        cloud_rmse.append(rmse)
        cloud_changed.append(1.0 - agree if ov > 0 else 0.0)

    point_error = []
    if preview_xyz is not None:
        for i, (xyz, t) in enumerate(zip(preview_xyz, transforms)):
            pts = np.asarray(xyz, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]
            point_error.append(surface_distance(pts, others[i], threshold))

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
