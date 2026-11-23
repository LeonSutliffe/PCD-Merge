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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from .registration import to_o3d, voxel_down

reg = o3d.pipelines.registration

# Pair grading, as multiples of the voxel size used for alignment. The voxel
# size scales with the scans (about 1/400 of their extent or coarser), so the
# same limits work for indoor and outdoor scans.
GOOD_RMSE = 0.15
FAIR_RMSE = 0.30
GOOD_OVERLAP = 0.30
FAIR_OVERLAP = 0.15

GRADES = ("Good", "Fair", "Poor")


@dataclass
class PairQuality:
    a: int
    b: int
    overlap: float  # fraction of points with a close match (better direction)
    rmse: float  # RMS point-to-plane distance over the overlapping points
    median: float
    p90: float
    grade: str

    @property
    def overlaps(self) -> bool:
        return self.overlap > 0.02


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

    @property
    def scale_max(self) -> float:
        """Upper end of the heatmap colour scale (red)."""
        return self.voxel_size * FAIR_RMSE * 1.5

    @property
    def grade(self) -> str:
        """Overall grade: the worst of the pairs that should overlap."""
        graded = [p.grade for p in self.pairs if p.overlaps]
        if not graded:
            return "Poor"
        return max(graded, key=GRADES.index)

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
                       f"rmse {_fmt(p.rmse)}")
            out.append(f"         median {_fmt(p.median)}, 90% within {_fmt(p.p90)}")
        out += ["", "Each cloud vs. the rest:"]
        for i in range(n):
            rm = _fmt(self.cloud_rmse[i]) if np.isfinite(self.cloud_rmse[i]) else "-"
            out.append(f"  {i + 1} {names[i]}: overlap {self.cloud_overlap[i]:.0%}, rmse {rm}")
        return out

    def to_dict(self) -> dict:
        return {
            "grade": self.grade,
            "voxel_size": self.voxel_size,
            "overlap_threshold": self.threshold,
            "pairs": [
                {"clouds": [p.a, p.b], "overlap": p.overlap, "rmse": p.rmse,
                 "median": p.median, "p90": p.p90, "grade": p.grade}
                for p in self.pairs
            ],
            "clouds": [
                {"overlap": o, "rmse": (r if np.isfinite(r) else None)}
                for o, r in zip(self.cloud_overlap, self.cloud_rmse)
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


def grade_pair(overlap: float, rmse: float, voxel: float) -> str:
    if overlap >= GOOD_OVERLAP and rmse <= GOOD_RMSE * voxel:
        return "Good"
    if overlap >= FAIR_OVERLAP and rmse <= FAIR_RMSE * voxel:
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


def _stats(d: np.ndarray, n_total: int):
    hit = d[np.isfinite(d)]
    overlap = len(hit) / max(1, n_total)
    if len(hit) == 0:
        return overlap, np.nan, np.nan, np.nan
    return (overlap, float(np.sqrt(np.mean(hit ** 2))), float(np.median(hit)),
            float(np.percentile(hit, 90)))


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

    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            ab = surface_distance(samples[a], surfaces[b], threshold)
            ba = surface_distance(samples[b], surfaces[a], threshold)
            # Overlap is asymmetric (a small scan inside a big one); use the
            # direction where it is larger, and pool the distances of both.
            ov = max(_stats(ab, len(ab))[0], _stats(ba, len(ba))[0])
            both = np.concatenate([ab, ba])
            _, rmse, med, p90 = _stats(both, len(both))
            grade = grade_pair(ov, rmse, voxel) if np.isfinite(rmse) else "Poor"
            pairs.append(PairQuality(a, b, ov, rmse, med, p90, grade))

    cloud_overlap, cloud_rmse = [], []
    others = []
    for i in range(n):
        o = o3d.geometry.PointCloud()
        for j in range(n):
            if j != i:
                o += surfaces[j]
        others.append(o)
        d = surface_distance(samples[i], o, threshold)
        ov, rmse, _, _ = _stats(d, len(d))
        cloud_overlap.append(ov)
        cloud_rmse.append(rmse)

    point_error = []
    if preview_xyz is not None:
        for i, (xyz, t) in enumerate(zip(preview_xyz, transforms)):
            pts = np.asarray(xyz, dtype=np.float64) @ t[:3, :3].T + t[:3, 3]
            point_error.append(surface_distance(pts, others[i], threshold))

    return QualityReport(voxel, threshold, pairs, cloud_overlap, cloud_rmse, point_error)


# --------------------------------------------------------------------------
# Heatmap colours

# Blue (fits) -> cyan -> green -> yellow -> red (off by scale_max or more).
_STOPS = np.array([
    [0.15, 0.35, 0.95],
    [0.10, 0.80, 0.90],
    [0.20, 0.85, 0.30],
    [0.98, 0.85, 0.15],
    [0.95, 0.20, 0.15],
])
NO_OVERLAP = (0.45, 0.45, 0.48)


def error_colours(err: np.ndarray, scale_max: float, alpha: float = 1.0) -> np.ndarray:
    """RGBA (N, 4) float32 colours for distances; NaN (no overlap) is grey."""
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
    return out


def colour_stops() -> list[tuple[float, tuple[float, float, float]]]:
    """(position 0..1, rgb) stops, for drawing a legend."""
    m = len(_STOPS) - 1
    return [(k / m, tuple(float(v) for v in _STOPS[k])) for k in range(len(_STOPS))]
