"""The points the Model tools work on: every visible scan, moved into the
project frame, with deleted points and hidden parts left out, read a few
million points at a time so a whole site never has to fit in memory.

Each tool asks only for what it needs: a floor plan only keeps a thin
slice, the shape fitting keeps one point per small cube.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geom import pack


@dataclass
class Source:
    """One scan or merged cloud: something with ``chunks()`` (a
    :class:`~pcd_merge.readers.ScanSource` or a loaded cloud's source), the
    transform into the project frame, its key for the edits, and where its
    scanner stood (project frame, or None if not known)."""

    source: object
    transform: np.ndarray
    key: str | None = None
    sensor: np.ndarray | None = None
    name: str = ""
    use_edits: bool = True  # merged clouds: the edits were made on the scans, not on them


@dataclass
class SitePoints:
    xyz: np.ndarray  # (N, 3) project frame
    scan: np.ndarray  # (N,) index into ``sensors``
    sensors: list = field(default_factory=list)  # per source: scanner position or None

    def __len__(self) -> int:
        return len(self.xyz)

    def sensor_z(self) -> np.ndarray:
        """Height of the scanner that saw each point (NaN if not known)."""
        z = np.array([np.nan if s is None else float(s[2]) for s in self.sensors] or [np.nan])
        return z[self.scan] if len(self.scan) else np.zeros(0)

    def sensor_xyz(self) -> np.ndarray:
        s = np.array([[np.nan] * 3 if p is None else list(map(float, p[:3]))
                      for p in self.sensors] or [[np.nan] * 3])
        return s[self.scan] if len(self.scan) else np.zeros((0, 3))


def _xyz(chunk: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    xyz = np.empty((len(chunk), 3), dtype=np.float64)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = chunk["x"], chunk["y"], chunk["z"]
    return xyz @ t[:3, :3].T + t[:3, 3]


def visible_mask(edits, xyz: np.ndarray, key) -> np.ndarray:
    """Points not deleted, and inside a visible part (when there are parts)."""
    if edits is None:
        return np.ones(len(xyz), dtype=bool)
    keep = ~edits.deleted(xyz, key) if edits else np.ones(len(xyz), bool)
    if edits.parts:
        inside = np.zeros(len(xyz), dtype=bool)
        for p in edits.parts:
            if p.visible:
                inside |= p.contains(xyz)
        keep &= inside
    return keep


def gather(sources, edits=None, *, zmin=None, zmax=None, within=None, cell=None,
           per_source=False, progress=None) -> SitePoints:
    """Every point of ``sources`` in the project frame.

    ``zmin`` / ``zmax`` keep a horizontal slice; ``within(xyz)`` (a function
    giving a mask) keeps any other shape. ``cell`` keeps one point per cube
    of that size (the first one read), across all the sources, or in each
    source on its own with ``per_source`` (so each scan keeps its own view
    of every surface, for working out which side of a wall it saw).
    """
    xs, ss = [], []
    keys_all = None
    ref = None  # cubes are counted from here, so big (e.g. UTM) coordinates fit
    n = max(len(sources), 1)
    for k, src in enumerate(sources):
        if progress:
            progress(f"Reading {src.name or 'scan'}", k / n)
        if per_source:
            keys_all = None
        for chunk in src.source.chunks():
            xyz = _xyz(chunk, src.transform)
            ok = np.isfinite(xyz).all(axis=1)
            if zmin is not None:
                ok &= xyz[:, 2] >= zmin
            if zmax is not None:
                ok &= xyz[:, 2] <= zmax
            xyz = xyz[ok]
            if within is not None and len(xyz):
                xyz = xyz[within(xyz)]
            if len(xyz) and src.use_edits:
                xyz = xyz[visible_mask(edits, xyz, src.key)]
            if cell and len(xyz):
                if ref is None:
                    ref = np.floor(xyz[0])
                kk = pack(np.floor((xyz - ref) / cell).astype(np.int64) + (1 << 19))
                kk, first = np.unique(kk, return_index=True)
                xyz = xyz[first]
                if keys_all is not None:  # cubes already filled by earlier chunks
                    new = ~np.isin(kk, keys_all)
                    xyz, kk = xyz[new], kk[new]
                keys_all = kk if keys_all is None else np.union1d(keys_all, kk)
            if len(xyz):
                xs.append(xyz)
                ss.append(np.full(len(xyz), k, dtype=np.int32))
            if progress:
                progress(f"Reading {src.name or 'scan'}", k / n)
    if progress:
        progress("Read", 1.0)
    xyz = np.concatenate(xs) if xs else np.zeros((0, 3))
    scan = np.concatenate(ss) if ss else np.zeros(0, np.int32)
    return SitePoints(xyz, scan, [s.sensor for s in sources])


def from_arrays(clouds, sensors=None) -> SitePoints:
    """SitePoints from (N, 3) arrays already in the project frame (tests,
    and the preview samples the view draws)."""
    clouds = [np.asarray(c, float).reshape(-1, 3) for c in clouds]
    sensors = list(sensors) if sensors is not None else [None] * len(clouds)
    xyz = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    scan = np.concatenate([np.full(len(c), k, np.int32) for k, c in enumerate(clouds)]) \
        if clouds else np.zeros(0, np.int32)
    return SitePoints(xyz, scan, sensors)


class ArraySource:
    """A (N, 3) array that looks like a scan source (``chunks()``)."""

    def __init__(self, xyz, chunk: int = 2_000_000):
        xyz = np.asarray(xyz, float)
        self.data = np.zeros(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
        self.data["x"], self.data["y"], self.data["z"] = xyz.T
        self.chunk = chunk

    def chunks(self, size=None):
        size = size or self.chunk
        for s in range(0, len(self.data), size):
            yield self.data[s:s + size]
