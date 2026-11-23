"""Optional clean-up applied while merging: overlap de-duplication, noise
removal and removal of moving objects.

Every step only *removes* original points; nothing is resampled, averaged or
moved, so the points that remain keep their exact coordinates and all of their
fields. All steps are off unless asked for.

* **Overlap de-duplication.** Where two or more scans cover the same spot, the
  merged cloud otherwise gets two interleaved copies of each surface (twice the
  points, slightly "fuzzy" walls). Space is split into cells; in every cell
  that more than one scan reaches, only the points of the scan that sampled
  that cell most densely are kept. That is normally the scan whose sensor was
  closest, so it is also the most accurate. Areas seen by one scan only are
  never touched, and the full detail of the best scan is kept everywhere.
* **Noise filter.** Statistical outlier removal per scan: a point is dropped if
  its average distance to its neighbours is far above typical for that scan
  (dust, rain, stray returns). Conservative by default.
* **Moving objects.** A point from scan A is dropped when another scan B looked
  straight *through* that spot (B's laser went past it and hit something
  further away) and B has no point there itself. That is the signature of a
  person or car that was there for A and gone for B. Things that B simply could
  not see because they were hidden behind something are kept, and scans that
  do not reach a spot never count as evidence against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CleanupOptions:
    dedup: bool = False
    dedup_cell: float | None = None  # None = automatic
    denoise: bool = False
    denoise_neighbors: int = 20
    denoise_std_ratio: float = 3.0
    remove_moving: bool = False
    moving_margin: float | None = None  # None = automatic (from the point spacing)

    @property
    def any(self) -> bool:
        return self.dedup or self.denoise or self.remove_moving


@dataclass
class CleanupStats:
    """How many points each step removed, per input cloud."""

    noise: list[int] = field(default_factory=list)
    moving: list[int] = field(default_factory=list)
    duplicate: list[int] = field(default_factory=list)
    dedup_cell: float | None = None
    no_sensor_position: list[int] = field(default_factory=list)

    def lines(self, names: list[str]) -> list[str]:
        out = []
        if self.no_sensor_position:
            out.append("Moving objects: no scanner position found for "
                       + ", ".join(names[i] for i in self.no_sensor_position)
                       + " (not a single-position scan?), so it was not used as evidence")
        for label, counts in (("noise points", self.noise),
                              ("points on moving objects", self.moving),
                              ("duplicate points in overlaps", self.duplicate)):
            if counts:
                detail = ", ".join(f"{n}: {c:,}" for n, c in zip(names, counts) if c)
                out.append(f"Removed {sum(counts):,} {label}"
                           + (f" ({detail})" if detail else ""))
        return out


def _xyz(d: np.ndarray) -> np.ndarray:
    return np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)


def _o3d_cloud(xyz: np.ndarray):
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    return pc


def point_spacing(xyz: np.ndarray, sample: int = 20000, seed: int = 0) -> float:
    """Median nearest-neighbour distance, estimated from a random sample."""
    if len(xyz) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), size=min(sample, len(xyz)), replace=False)
    sub = xyz[idx]
    d = np.asarray(_o3d_cloud(sub).compute_nearest_neighbor_distance())
    d = d[d > 0]
    if not len(d):
        return 0.0
    # The sample is sparser than the full cloud; scale back to full density.
    return float(np.median(d) * (len(sub) / len(xyz)) ** (1 / 2))


# --------------------------------------------------------------------------
# Noise


def noise_mask(xyz: np.ndarray, neighbors: int = 20, std_ratio: float = 3.0) -> np.ndarray:
    """True for points to keep (statistical outlier removal)."""
    keep = np.ones(len(xyz), dtype=bool)
    if len(xyz) <= neighbors:
        return keep
    _, idx = _o3d_cloud(xyz).remove_statistical_outlier(nb_neighbors=neighbors,
                                                        std_ratio=std_ratio)
    keep[:] = False
    keep[np.asarray(idx, dtype=np.int64)] = True
    return keep


# --------------------------------------------------------------------------
# Moving objects


class _RangeImage:
    """Closest return per (azimuth, elevation) cell, seen from a sensor."""

    def __init__(self, xyz: np.ndarray, origin: np.ndarray, az_step_deg=0.5, el_step_deg=1.0):
        self.origin = origin
        self.az_n = int(round(360 / az_step_deg))
        self.el_n = int(round(180 / el_step_deg))
        self.az_step = np.radians(az_step_deg)
        self.el_step = np.radians(el_step_deg)
        r, ia, ie = self._project(xyz)
        img = np.full((self.el_n, self.az_n), np.inf)
        order = np.argsort(-r)  # farthest first, so the closest write wins
        img[ie[order], ia[order]] = r[order]
        # Take the minimum over neighbouring cells as well: a point only counts
        # as "seen through" if the scan saw past it in all directions around it.
        # That keeps grazing ground and edges where beams are sparse.
        out = img.copy()
        for de in (-1, 0, 1):
            shifted_e = np.full_like(img, np.inf)
            if de < 0:
                shifted_e[:de] = img[-de:]
            elif de > 0:
                shifted_e[de:] = img[:-de]
            else:
                shifted_e = img
            for da in (-1, 0, 1):
                out = np.minimum(out, np.roll(shifted_e, da, axis=1))
        self.img = out
        self.max_range = float(r.max()) if len(r) else 0.0

    def _project(self, xyz):
        v = xyz - self.origin
        r = np.linalg.norm(v, axis=1)
        az = np.arctan2(v[:, 1], v[:, 0])
        el = np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-12), -1, 1))
        ia = np.floor((az + np.pi) / self.az_step).astype(np.int64) % self.az_n
        ie = np.clip(np.floor((el + np.pi / 2) / self.el_step).astype(np.int64), 0, self.el_n - 1)
        return r, ia, ie

    def seen_through(self, xyz: np.ndarray, margin: float) -> np.ndarray:
        """True where this sensor measured further than the point in every
        neighbouring direction, so its beams passed through that spot."""
        r, ia, ie = self._project(xyz)
        measured = self.img[ie, ia]
        return np.isfinite(measured) & (measured > r + margin + 0.02 * r) & (r < self.max_range)


def plausible_origin(xyz: np.ndarray, origin: np.ndarray, sectors: int = 36) -> bool:
    """Does the scan look like it was taken from ``origin``?

    A single-position LiDAR scan surrounds its sensor. Clouds that were already
    moved into a site or map frame, or built from a moving sensor, usually
    don't surround their PCD VIEWPOINT, and then the "seen through" test would
    be meaningless, so such a cloud is not used as evidence.
    """
    if len(xyz) < 100:
        return False
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    if np.any(origin[:2] < lo[:2]) or np.any(origin[:2] > hi[:2]):
        return False
    v = xyz[:, :2] - origin[:2]
    az = np.arctan2(v[:, 1], v[:, 0])
    hist = np.bincount(((az + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors,
                       minlength=sectors)
    return (hist > 0.002 * len(xyz)).mean() >= 0.5


def moving_mask(parts_xyz: list[np.ndarray], origins: list[np.ndarray],
                margin: float, progress=None, skipped: list[int] | None = None,
                ) -> list[np.ndarray]:
    """For each cloud, True for points to keep (see module docstring).

    Indices of clouds whose sensor position is not plausible are appended to
    ``skipped``; they are still checked against the others, but never used as
    evidence."""
    keeps = [np.ones(len(x), dtype=bool) for x in parts_xyz]
    n = len(parts_xyz)
    if n < 2:
        return keeps
    images = []
    for i, (x, o) in enumerate(zip(parts_xyz, origins)):
        if plausible_origin(x, o):
            images.append(_RangeImage(x, o))
        else:
            images.append(None)
            if skipped is not None:
                skipped.append(i)
    trees = [None] * n
    for a in range(n):
        if progress:
            progress(a / n)
        for b in range(n):
            if a == b or images[b] is None or not len(parts_xyz[a]):
                continue
            cand = np.flatnonzero(keeps[a])
            cand = cand[images[b].seen_through(parts_xyz[a][cand], margin)]
            if not len(cand):
                continue
            # Only if B has no point of its own there either.
            if trees[b] is None:
                trees[b] = _o3d_cloud(parts_xyz[b])
            d = np.asarray(_o3d_cloud(parts_xyz[a][cand]).compute_point_cloud_distance(trees[b]))
            keeps[a][cand[d > margin]] = False
    return keeps


# --------------------------------------------------------------------------
# Overlap de-duplication


def dedup_mask(parts_xyz: list[np.ndarray], cell: float) -> list[np.ndarray]:
    """For each cloud, True for points to keep: in cells reached by several
    clouds, keep only the cloud with the most points in that cell."""
    sizes = [len(x) for x in parts_xyz]
    if len(parts_xyz) < 2 or sum(sizes) == 0:
        return [np.ones(s, dtype=bool) for s in sizes]
    xyz = np.concatenate([x for x in parts_xyz if len(x)])
    src = np.repeat(np.arange(len(parts_xyz), dtype=np.int64), sizes)
    ijk = np.floor((xyz - xyz.min(axis=0)) / cell).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    if np.prod(dims.astype(np.float64)) < 2 ** 62:
        key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        _, voxel = np.unique(key, return_inverse=True)
    else:
        _, voxel = np.unique(ijk, axis=0, return_inverse=True)
    voxel = voxel.ravel()
    # Count points per (voxel, source) and pick the source with most points.
    n_src = len(parts_xyz)
    pair = voxel * n_src + src
    upair, inv, counts = np.unique(pair, return_inverse=True, return_counts=True)
    pv, ps = upair // n_src, upair % n_src
    # Sort by voxel, then count descending, then source ascending -> first per voxel wins.
    order = np.lexsort((ps, -counts, pv))
    first = np.ones(len(order), dtype=bool)
    first[1:] = pv[order][1:] != pv[order][:-1]
    winner_pair = np.zeros(len(upair), dtype=bool)
    winner_pair[order[first]] = True
    keep = winner_pair[inv.ravel()]
    return np.split(keep, np.cumsum(sizes)[:-1])


# --------------------------------------------------------------------------


def auto_dedup_cell(parts_xyz: list[np.ndarray]) -> float:
    """A cell several times the typical point spacing, so that two scans of the
    same surface land in the same cells, but small enough that a different
    scan can win on either side of a corner or an occlusion edge."""
    spacings = [point_spacing(x) for x in parts_xyz if len(x) > 1]
    s = max(spacings) if spacings else 0.0
    return max(4.0 * s, 1e-6) if s > 0 else 0.05


def clean(parts: list[np.ndarray], origins: list[np.ndarray], opts: CleanupOptions,
          progress=None) -> tuple[list[np.ndarray], CleanupStats]:
    """Apply the chosen steps to already-transformed clouds (structured arrays
    in the common frame). ``origins`` are the sensor positions in that frame.
    """
    stats = CleanupStats()
    several = len(parts) > 1
    steps = [s for s, on in (("noise", opts.denoise), ("moving", opts.remove_moving and several),
                             ("dedup", opts.dedup and several)) if on]

    def report(msg, frac, step):
        if progress:
            progress(msg, (steps.index(step) + frac) / len(steps))
    n = len(parts)
    xyz = [_xyz(p) for p in parts]

    if opts.denoise:
        stats.noise = []
        for i in range(n):
            report(f"Removing noise ({i + 1}/{n})", i / n, "noise")
            k = noise_mask(xyz[i], opts.denoise_neighbors, opts.denoise_std_ratio)
            stats.noise.append(int((~k).sum()))
            parts[i], xyz[i] = parts[i][k], xyz[i][k]

    if opts.remove_moving and n > 1:
        margin = opts.moving_margin
        if margin is None:
            s = max((point_spacing(x) for x in xyz if len(x) > 1), default=0.0)
            margin = max(5.0 * s, 0.1) if s > 0 else 0.3
        keeps = moving_mask(xyz, [np.asarray(o, float) for o in origins], margin,
                            lambda f: report("Removing moving objects", f, "moving"),
                            skipped=stats.no_sensor_position)
        stats.moving = [int((~k).sum()) for k in keeps]
        parts = [p[k] for p, k in zip(parts, keeps)]
        xyz = [x[k] for x, k in zip(xyz, keeps)]

    if opts.dedup and n > 1:
        report("Removing duplicate points in overlaps", 0.0, "dedup")
        cell = opts.dedup_cell or auto_dedup_cell(xyz)
        stats.dedup_cell = cell
        keeps = dedup_mask(xyz, cell)
        stats.duplicate = [int((~k).sum()) for k in keeps]
        parts = [p[k] for p, k in zip(parts, keeps)]

    return parts, stats
