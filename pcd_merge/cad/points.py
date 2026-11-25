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


class TooManyPoints(ValueError):
    """More points than the caller can take (``gather(limit=...)``)."""

    def __init__(self, count: int, limit: int):
        super().__init__(f"more than {limit:,} points")
        self.count, self.limit = count, limit


def _first_of_each(keys: np.ndarray):
    """(sorted unique keys, index of the first occurrence of each). Same as
    ``np.unique(keys, return_index=True)`` but by one stable sort, which is
    several times faster on big int64 arrays."""
    if not len(keys):
        return keys, np.zeros(0, np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    head = np.empty(len(sk), dtype=bool)
    head[0] = True
    np.not_equal(sk[1:], sk[:-1], out=head[1:])
    return sk[head], order[head]


class _Thinner:
    """One point per cube, the first one read, kept while chunks stream in.

    Each chunk is thinned on its own, then checked against the cubes kept so
    far with a binary search (``kept`` is sorted). New cubes wait in
    ``pending`` and are folded into ``kept`` once there are as many of them
    as kept cubes, so the work grows as n log n rather than with the square
    of the scan size."""

    def __init__(self):
        self.kept = np.zeros(0, np.int64)  # sorted
        self.pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # keys, xyz, src
        self.n_pending = 0
        self.out_xyz: list[np.ndarray] = []
        self.out_src: list[np.ndarray] = []

    def add(self, keys: np.ndarray, xyz: np.ndarray, src: int) -> None:
        if len(self.kept):
            pos = np.searchsorted(self.kept, keys)
            pos[pos == len(self.kept)] = 0
            new = self.kept[pos] != keys
            keys, xyz = keys[new], xyz[new]
        if not len(keys):
            return
        self.pending.append((keys, xyz, np.full(len(keys), src, np.int32)))
        self.n_pending += len(keys)
        if self.n_pending >= max(len(self.kept), 4_000_000):
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys = np.concatenate([p[0] for p in self.pending])
        xyz = np.concatenate([p[1] for p in self.pending])
        src = np.concatenate([p[2] for p in self.pending])
        self.pending, self.n_pending = [], 0
        uk, first = _first_of_each(keys)  # first: earliest read
        first.sort()  # keep the reading order
        self.out_xyz.append(xyz[first])
        self.out_src.append(src[first])
        if len(self.kept):  # two sorted runs with no keys in common
            merged = np.concatenate([self.kept, uk])
            merged.sort(kind="stable")
            self.kept = merged
        else:
            self.kept = uk

    def __len__(self) -> int:
        return len(self.kept) + self.n_pending  # an upper bound until flushed


def _chunk_count(src) -> int:
    n = getattr(src.source, "count", 0)
    if not n and hasattr(src.source, "data"):
        n = len(src.source.data)
    return int(n or 0)


def gather(sources, edits=None, *, zmin=None, zmax=None, within=None, cell=None,
           per_source=False, limit=None, progress=None) -> SitePoints:
    """Every point of ``sources`` in the project frame.

    ``zmin`` / ``zmax`` keep a horizontal slice; ``within(xyz)`` (a function
    giving a mask) keeps any other shape. ``cell`` keeps one point per cube
    of that size (the first one read), across all the sources, or in each
    source on its own with ``per_source`` (so each scan keeps its own view
    of every surface, for working out which side of a wall it saw).

    ``progress(message, fraction)`` is called after every chunk, with the
    share of all the points read so far (it may raise to cancel). With
    ``limit``, :class:`TooManyPoints` is raised as soon as more than that
    many points are kept, without reading the rest.
    """
    xs, ss = [], []
    ref = None  # cubes are counted from here, so big (e.g. UTM) coordinates fit
    counts = [_chunk_count(s) for s in sources]
    total = sum(counts)
    n = max(len(sources), 1)
    done = 0
    thinners: list[_Thinner] = []
    thin_all = _Thinner() if cell and not per_source else None
    kept_before = 0  # points kept by finished sources

    def kept_now() -> int:
        if not cell:
            return sum(len(x) for x in xs)
        return kept_before + sum(len(t) for t in thinners[-1:]) if per_source \
            else len(thin_all)

    for k, src in enumerate(sources):
        label = f"Reading {src.name or 'scan'}"
        if progress:
            progress(label, done / total if total else k / n)
        thin = None
        if cell:
            thin = _Thinner() if per_source else thin_all
            if per_source:
                thinners.append(thin)
        read_here = 0
        for chunk in src.source.chunks():
            read_here += len(chunk)
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
                kk, first = _first_of_each(kk)
                thin.add(kk, xyz[first], k)
            elif len(xyz):
                xs.append(xyz)
                ss.append(np.full(len(xyz), k, dtype=np.int32))
            if limit is not None and kept_now() > limit:
                if thin is not None:
                    thin.flush()
                if kept_now() > limit:
                    raise TooManyPoints(kept_now(), limit)
            if progress:
                if total:
                    frac = (done + min(read_here, counts[k] or read_here)) / total
                else:
                    frac = k / n
                progress(label, min(frac, 1.0))
        done += counts[k] or read_here
        if per_source and thin is not None:
            thin.flush()
            kept_before += len(thin)
    for t in thinners or ([thin_all] if thin_all is not None else []):
        t.flush()
        xs.extend(t.out_xyz)
        ss.extend(t.out_src)
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
        self.count = len(xyz)

    def chunks(self, size=None):
        size = size or self.chunk
        for s in range(0, len(self.data), size):
            yield self.data[s:s + size]
