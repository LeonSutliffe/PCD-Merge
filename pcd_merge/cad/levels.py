"""Find a building's levels (storeys): the heights of its floors and ceilings.

Floors and ceilings are large flat horizontal surfaces, so at their heights
a thin horizontal slice of the site covers a lot of ground, while at other
heights it only cuts through walls. The slice's covered area is worked out
every 2 cm up the building and the peaks are the floors and ceilings.

A peak is a **floor** when the scanners that saw it stood above it, and a
**ceiling** when they stood below it. Without scanner positions (a merged
cloud), the lowest peak is taken as a floor, and a peak just above a
ceiling (a slab's thickness, under 0.8 m) as the next floor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .geom import pack

STEP = 0.02  # height resolution of the search (m)
CELL = 0.10  # ground resolution of the area count (m)


@dataclass
class Level:
    name: str
    floor: float  # height of the floor's surface (m)
    ceiling: float | None = None  # height of the ceiling's underside, if found
    area: float = 0.0  # floor area found (m²)

    def cut_height(self, above: float = 1.2) -> float:
        """Where a floor plan cuts this level: ``above`` the floor, but below
        the ceiling."""
        z = self.floor + above
        if self.ceiling is not None and z > self.ceiling - 0.1:
            z = (self.floor + self.ceiling) / 2
        return z

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Level":
        return cls(str(d.get("name", "Level")), float(d["floor"]),
                   None if d.get("ceiling") is None else float(d["ceiling"]),
                   float(d.get("area", 0.0)))


@dataclass
class Peak:
    z: float
    area: float
    above: float  # share of its points seen from above (scanner higher): floor-like; NaN unknown


def area_profile(xyz: np.ndarray, step: float = STEP, cell: float | None = None):
    """For each height bin: (bin bottoms, area covered in m²). The ground
    cell is coarser for sparse points (a thinned sample of a big site), so
    a floor still fills its cells."""
    if cell is None:
        ext = np.ptp(xyz[:, :2], axis=0)
        per_point = float(ext[0] * ext[1]) / max(len(xyz), 1)
        cell = float(np.clip(np.sqrt(per_point * 40), CELL, 0.5))
    z = xyz[:, 2]
    z0 = float(np.floor(z.min() / step) * step)
    zi = np.floor((z - z0) / step).astype(np.int64)
    lo = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - lo) / cell).astype(np.int64)
    keys = np.unique(pack(np.column_stack([ij, zi])))
    counts = np.bincount(keys & ((1 << 21) - 1), minlength=int(zi.max()) + 2)
    # the packed key's last 21 bits are zi + 1
    counts = counts[1:]
    return z0 + step * np.arange(len(counts)), counts * cell * cell


def find_peaks(xyz: np.ndarray, sensor_z: np.ndarray | None = None, min_area: float | None = None,
               step: float = STEP) -> list[Peak]:
    """Heights where a horizontal surface covers much more ground than just
    above and below it."""
    if len(xyz) < 100:
        return []
    zs, area = area_profile(xyz, step)
    if len(area) < 5:
        return []
    # background: what walls alone cover (median over +-0.5 m)
    w = max(3, int(round(0.5 / step)))
    padded = np.r_[np.zeros(w), area, np.zeros(w)]
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1)
    background = np.median(windows, axis=1)
    excess = area - background
    if min_area is None:
        min_area = max(2.0, 0.15 * float(excess.max()))
    peaks = []
    k = 0
    while k < len(excess):
        if excess[k] < min_area:
            k += 1
            continue
        j = k
        while j + 1 < len(excess) and excess[j + 1] >= 0.25 * min_area:
            j += 1
        sel = slice(k, j + 1)
        wts = np.maximum(excess[sel], 0)
        zc = float(np.average(zs[sel] + step / 2, weights=wts)) if wts.sum() > 0 \
            else float(zs[k] + step / 2)
        peaks.append(Peak(zc, float(excess[sel].max()), np.nan))
        k = j + 1
    # merge peaks closer than 10 cm (a floor with a slight slope or noise)
    merged = []
    for p in peaks:
        if merged and p.z - merged[-1].z < 0.10:
            q = merged[-1]
            tot = q.area + p.area
            merged[-1] = Peak((q.z * q.area + p.z * p.area) / tot, max(q.area, p.area), np.nan)
        else:
            merged.append(p)
    if sensor_z is not None and len(sensor_z) == len(xyz):
        known = np.isfinite(sensor_z)
        for p in merged:
            near = known & (np.abs(xyz[:, 2] - p.z) <= 0.04)
            if near.sum() >= 20:
                p.above = float(np.mean(sensor_z[near] > p.z))
    return merged


def find_levels(xyz: np.ndarray, sensor_z: np.ndarray | None = None,
                min_area: float | None = None, min_height: float = 1.8) -> list[Level]:
    """The building's levels, lowest first."""
    peaks = find_peaks(np.asarray(xyz, float), sensor_z, min_area)
    if not peaks:
        return []
    if not any(np.isfinite(p.above) and p.above >= 0.5 for p in peaks):
        for p in peaks:  # no scanner stood above anything: the positions are unusable
            p.above = np.nan
    kinds = []
    prev = None
    for p in peaks:
        if np.isfinite(p.above):
            kind = "floor" if p.above >= 0.5 else "ceiling"
        elif prev is None:
            kind = "floor"
        elif prev[0] == "ceiling":
            kind = "floor" if p.z - prev[1] < 0.8 else "ceiling"
        else:  # above a floor: a ceiling if it is a storey's height up
            kind = "ceiling" if p.z - prev[1] >= min_height else "skip"
        if kind != "skip":
            kinds.append((kind, p))
            prev = (kind, p.z)
    levels: list[Level] = []
    for kind, p in kinds:
        if kind == "floor":
            if levels and levels[-1].ceiling is None and p.z - levels[-1].floor < min_height:
                continue  # a step, a table, a raised part of the same floor
            levels.append(Level("", p.z, None, p.area))
        elif levels and levels[-1].ceiling is None and p.z - levels[-1].floor >= min_height:
            levels[-1].ceiling = p.z
    for k, lv in enumerate(levels):
        lv.name = f"Level {k}"
    return levels
