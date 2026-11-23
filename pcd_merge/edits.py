"""Edit mode's data: deletions and parts, stored as regions in space.

Nothing here touches the scan files, and no point lists are stored. An edit
is an instruction such as "delete every point inside this sphere" or "restore
every point inside this lasso, as seen from this camera". The regions are in
the project frame (the aligned frame), so:

* an edit applies to **every full-resolution point** inside it, not only to
  the points that happened to be drawn when it was made (Edit mode draws a
  sample for speed);
* aligning again keeps the edits where they were in the building;
* undo is just dropping the last instruction, and the project stays small.

A point is deleted if the **last** edit whose region contains it is a
delete (a later restore brings it back). Each edit can be limited to some
scans (by scan key).

Regions: :class:`Sphere` (click, brush), :class:`Box`, :class:`ScreenShape`
(lasso and rectangle: a polygon on screen plus the camera it was drawn with,
selecting through the whole depth), :class:`Voxels` (grow-to-object: the
cells of one connected object) and :class:`HalfSpace` (one side of a plane,
used by splits).

A :class:`Part` is a named piece of the site (a floor, a wing) made by
splitting: the points inside all of its regions.
"""

from __future__ import annotations

import base64
import itertools
import uuid
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Regions


class Region:
    kind = ""

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of the (N, 3) project-frame points inside."""
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Axis-aligned box around the region, or None if unbounded."""
        return None

    def to_dict(self) -> dict:
        raise NotImplementedError

    def mask(self, xyz: np.ndarray) -> np.ndarray:
        """``contains`` with a quick bounding-box test first."""
        b = self.bounds()
        if b is None or not len(xyz):
            return self.contains(xyz)
        lo, hi = b
        near = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        out = np.zeros(len(xyz), dtype=bool)
        if near.any():
            out[near] = self.contains(xyz[near])
        return out


@dataclass
class Sphere(Region):
    centre: np.ndarray
    radius: float
    kind = "sphere"

    def __post_init__(self):
        self.centre = np.asarray(self.centre, dtype=np.float64)

    def contains(self, xyz):
        d = xyz - self.centre
        return np.einsum("ij,ij->i", d, d) <= self.radius * self.radius

    def bounds(self):
        return self.centre - self.radius, self.centre + self.radius

    def to_dict(self):
        return {"kind": self.kind, "centre": self.centre.tolist(), "radius": float(self.radius)}


@dataclass
class Box(Region):
    lo: np.ndarray
    hi: np.ndarray
    kind = "box"

    def __post_init__(self):
        lo, hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)

    def contains(self, xyz):
        return np.all((xyz >= self.lo) & (xyz <= self.hi), axis=1)

    def bounds(self):
        return self.lo, self.hi

    def to_dict(self):
        return {"kind": self.kind, "lo": self.lo.tolist(), "hi": self.hi.tolist()}


def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points. ``poly`` is (K, 2), not closed."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    for i in range(len(poly)):
        x1, y1 = px[i - 1], py[i - 1]
        x2, y2 = px[i], py[i]
        if y1 == y2:
            continue
        crosses = (y1 > y) != (y2 > y)
        xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xin)
    return inside


@dataclass
class ScreenShape(Region):
    """A lasso or rectangle drawn on the screen, through the whole depth of
    the view. ``polygon`` is in normalised device coordinates (-1..1) and
    ``mvp`` maps project-frame points to clip space with the camera of the
    moment it was drawn (with any drawing offset already folded in)."""

    polygon: np.ndarray
    mvp: np.ndarray
    kind = "screen"

    def __post_init__(self):
        self.polygon = np.asarray(self.polygon, float).reshape(-1, 2)
        self.mvp = np.asarray(self.mvp, float).reshape(4, 4)

    def contains(self, xyz):
        if len(self.polygon) < 3 or not len(xyz):
            return np.zeros(len(xyz), dtype=bool)
        clip = xyz @ self.mvp[:3, :3].T + self.mvp[:3, 3]
        w = xyz @ self.mvp[3, :3] + self.mvp[3, 3]
        front = w > 1e-9
        out = np.zeros(len(xyz), dtype=bool)
        if front.any():
            ndc = clip[front, :2] / w[front, None]
            lo, hi = self.polygon.min(axis=0), self.polygon.max(axis=0)
            box = np.all((ndc >= lo) & (ndc <= hi), axis=1)
            hit = np.zeros(len(ndc), dtype=bool)
            if box.any():
                hit[box] = points_in_polygon(ndc[box], self.polygon)
            out[front] = hit
        return out

    def to_dict(self):
        return {"kind": self.kind, "polygon": self.polygon.tolist(), "mvp": self.mvp.tolist()}


@dataclass
class Voxels(Region):
    """A set of grid cells (grow-to-object). ``keys`` are the cells' integer
    coordinates packed into int64, sorted."""

    cell: float
    origin: np.ndarray
    keys: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    kind = "voxels"

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.keys = np.unique(np.asarray(self.keys, np.int64))
        if self.lo is None and len(self.keys):
            ijk = unpack_keys(self.keys)
            self.lo = self.origin + ijk.min(axis=0) * self.cell
            self.hi = self.origin + (ijk.max(axis=0) + 1) * self.cell
        if self.lo is not None:
            self.lo, self.hi = np.asarray(self.lo, float), np.asarray(self.hi, float)

    def contains(self, xyz):
        if not len(self.keys):
            return np.zeros(len(xyz), dtype=bool)
        ijk = np.floor((xyz - self.origin) / self.cell).astype(np.int64)
        return np.isin(pack_keys(ijk), self.keys, assume_unique=False)

    def bounds(self):
        return (self.lo, self.hi) if self.lo is not None else None

    def to_dict(self):
        # the cells as base64 of little-endian int64: far smaller than a JSON list
        packed = base64.b64encode(self.keys.astype("<i8").tobytes()).decode("ascii")
        return {"kind": self.kind, "cell": float(self.cell), "origin": self.origin.tolist(),
                "keys": packed}

    @classmethod
    def from_dict(cls, d: dict) -> "Voxels":
        keys = d.get("keys", [])
        if isinstance(keys, str):
            keys = np.frombuffer(base64.b64decode(keys), dtype="<i8")
        return cls(float(d["cell"]), d["origin"], np.asarray(keys, np.int64))


@dataclass
class HalfSpace(Region):
    """Points with ``dot(normal, p) >= offset``."""

    normal: np.ndarray
    offset: float
    kind = "halfspace"

    def __post_init__(self):
        n = np.asarray(self.normal, float)
        self.normal = n / (np.linalg.norm(n) or 1.0)

    def contains(self, xyz):
        return xyz @ self.normal >= self.offset

    def flipped(self) -> "HalfSpace":
        return HalfSpace(-self.normal, -self.offset)

    def to_dict(self):
        return {"kind": self.kind, "normal": self.normal.tolist(), "offset": float(self.offset)}


class Union(Region):
    """Points inside any of several regions (a brush stroke, a selection
    made of several shapes)."""

    kind = "union"

    def __init__(self, regions):
        self.regions = list(regions)
        self._bounds = None
        boxes = [r.bounds() for r in self.regions]
        if self.regions and all(b is not None for b in boxes):
            self._bounds = (np.min([b[0] for b in boxes], axis=0),
                            np.max([b[1] for b in boxes], axis=0))

    def __eq__(self, other):
        return isinstance(other, Union) and other.to_dict() == self.to_dict()

    def contains(self, xyz):
        out = np.zeros(len(xyz), dtype=bool)
        for r in self.regions:
            todo = ~out
            if todo.any():
                out[todo] = r.mask(xyz[todo])
        return out

    def bounds(self):
        return self._bounds

    def to_dict(self):
        return {"kind": self.kind, "regions": [r.to_dict() for r in self.regions]}


def union(regions) -> Region:
    """One region for several (itself if there is only one)."""
    regions = list(regions)
    return regions[0] if len(regions) == 1 else Union(regions)


_KINDS = {c.kind: c for c in (Sphere, Box, ScreenShape, Voxels, HalfSpace, Union)}


def region_from_dict(d: dict) -> Region:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind not in _KINDS:
        raise ValueError(f"unknown kind of region: {kind!r}")
    if kind == "voxels":
        return Voxels.from_dict(d)
    if kind == "union":
        return Union([region_from_dict(r) for r in d["regions"]])
    return _KINDS[kind](**d)


_BITS = 21  # per axis: +-1M cells
_OFF = 1 << (_BITS - 1)


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    v = np.clip(ijk + _OFF, 0, (1 << _BITS) - 1).astype(np.int64)
    return (v[:, 0] << (2 * _BITS)) | (v[:, 1] << _BITS) | v[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    m = (1 << _BITS) - 1
    k = np.asarray(keys, np.int64)
    return np.stack([(k >> (2 * _BITS)) & m, (k >> _BITS) & m, k & m], 1) - _OFF


# --------------------------------------------------------------------------
# Edits


@dataclass
class Edit:
    action: str  # "delete" or "restore"
    region: Region
    scans: list | None = None  # scan keys it applies to; None = every scan
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def to_dict(self) -> dict:
        return {"id": self.id, "action": self.action, "label": self.label,
                "scans": self.scans, "region": self.region.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(action=d["action"], region=region_from_dict(d["region"]),
                   scans=d.get("scans"), label=d.get("label", ""), id=d.get("id") or
                   uuid.uuid4().hex[:10])


@dataclass
class Part:
    """A piece of the site made by a split: the points inside all ``regions``."""

    name: str
    regions: list
    visible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        m = np.ones(len(xyz), dtype=bool)
        for r in self.regions:
            m &= r.mask(xyz)
        return m

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "visible": self.visible,
                "regions": [r.to_dict() for r in self.regions]}

    @classmethod
    def from_dict(cls, d: dict) -> "Part":
        return cls(name=d["name"], regions=[region_from_dict(r) for r in d["regions"]],
                   visible=d.get("visible", True), id=d.get("id") or uuid.uuid4().hex[:10])


class EditStack:
    """The project's edits in order, with undo and redo."""

    def __init__(self, edits=None, parts=None):
        self.edits: list[Edit] = list(edits or [])
        self.parts: list[Part] = list(parts or [])
        self._redo: list = []

    # ---- changes (each is one undo step)

    def add(self, edit: Edit) -> None:
        self.edits.append(edit)
        self._redo.clear()

    def split(self, plane: HalfSpace, names=("Part A", "Part B"), within: Part | None = None):
        """Cut the site (or one part) in two along a plane. Returns the new parts."""
        base = list(within.regions) if within is not None else []
        a = Part(names[0], base + [plane])
        b = Part(names[1], base + [plane.flipped()])
        old = list(self.parts)
        if within is not None:
            self.parts = [p for p in self.parts if p.id != within.id]
        self.parts += [a, b]
        self.edits.append(("parts", old))  # undo marker
        self._redo.clear()
        return a, b

    def can_undo(self) -> bool:
        return bool(self.edits)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self):
        if not self.edits:
            return None
        e = self.edits.pop()
        if isinstance(e, tuple):  # a split: put the parts back as they were
            self._redo.append(("parts", list(self.parts)))
            self.parts = e[1]
            return "split"
        self._redo.append(e)
        return e

    def redo(self):
        if not self._redo:
            return None
        e = self._redo.pop()
        if isinstance(e, tuple):
            self.edits.append(("parts", list(self.parts)))
            self.parts = e[1]
            return "split"
        self.edits.append(e)
        return e

    # ---- queries

    @property
    def point_edits(self) -> list[Edit]:
        return [e for e in self.edits if isinstance(e, Edit)]

    def __bool__(self) -> bool:
        return bool(self.point_edits)

    def deleted(self, xyz: np.ndarray, scan_key: str | None = None) -> np.ndarray:
        """Which of these project-frame points of one scan are deleted."""
        out = np.zeros(len(xyz), dtype=bool)
        for e in self.point_edits:
            if e.scans is not None and scan_key not in e.scans:
                continue
            m = e.region.mask(xyz)
            if e.action == "delete":
                out |= m
            else:
                out &= ~m
        return out

    def keep(self, xyz: np.ndarray, scan_key: str | None = None,
             part: Part | None = None) -> np.ndarray:
        """Points that survive the edits (and lie in ``part``, if given)."""
        k = ~self.deleted(xyz, scan_key)
        if part is not None:
            k &= part.contains(xyz)
        return k

    # ---- saving

    def to_dict(self) -> dict:
        return {"edits": [e.to_dict() for e in self.point_edits],
                "parts": [p.to_dict() for p in self.parts]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "EditStack":
        d = d or {}
        return cls([Edit.from_dict(e) for e in d.get("edits", [])],
                   [Part.from_dict(p) for p in d.get("parts", [])])


# --------------------------------------------------------------------------
# Grow to object


def grow_object(xyz: np.ndarray, seed: np.ndarray, cell: float | None = None,
                max_radius: float = 6.0, ground: bool = True) -> Voxels:
    """Cells of the object at ``seed``: the connected cluster of occupied
    cells around it, not crossing the floor or a wall it stands against.

    ``xyz`` are the project-frame points to search (the drawn sample is
    enough: the result is a region, so it then applies to every point).
    Large flat surfaces near the seed (floor, walls) are left out first,
    unless the seed is on one of them, then the cluster is grown through
    face, edge and corner neighbours, at most ``max_radius`` from the seed.
    """
    seed = np.asarray(seed, float)
    near = np.linalg.norm(xyz - seed, axis=1) <= max_radius
    pts = xyz[near]
    if not len(pts):
        return Voxels(cell or 0.05, seed, np.zeros(0, np.int64))
    if cell is None:
        cell = _auto_cell(pts)
    if ground and len(pts) > 200:
        pts = _drop_big_planes(pts, seed, cell)
    origin = seed - cell * 0.5  # seed in the middle of its cell
    ijk = np.floor((pts - origin) / cell).astype(np.int64)
    occupied = set(map(tuple, np.unique(ijk, axis=0).tolist()))
    start = (0, 0, 0)
    if start not in occupied:
        # the click may land just beside the object: take the nearest cell
        if not occupied:
            return Voxels(cell, origin, np.zeros(0, np.int64))
        arr = np.array(list(occupied))
        start = tuple(arr[np.argmin(np.abs(arr).sum(axis=1))].tolist())
        if max(abs(v) for v in start) > 2:
            return Voxels(cell, origin, np.zeros(0, np.int64))
    limit = int(np.ceil(max_radius / cell))
    steps = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]
    seen = {start}
    todo = [start]
    while todo:
        c = todo.pop()
        for d in steps:
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if n in occupied and n not in seen and max(abs(v) for v in n) <= limit:
                seen.add(n)
                todo.append(n)
    return Voxels(cell, origin, pack_keys(np.array(list(seen), dtype=np.int64)))


def _auto_cell(pts: np.ndarray) -> float:
    """About three times the typical spacing of the points."""
    return _spacing(pts) * 3.0


def _spacing(pts: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        sample = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 2000), False)]
        d, _ = cKDTree(pts).query(sample, k=2)
        s = float(np.median(d[:, 1]))
    except ImportError:  # SciPy is optional: Open3D can do it too
        import open3d as o3d

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        s = float(np.median(np.asarray(pc.compute_nearest_neighbor_distance())))
    return max(s, 1e-4)


def _drop_big_planes(pts: np.ndarray, seed: np.ndarray, cell: float) -> np.ndarray:
    """Remove up to three large flat surfaces (floor, walls) near the seed,
    unless the seed itself is on one."""
    import open3d as o3d

    out = pts
    for _ in range(3):
        if len(out) < 200:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(out))
        args = dict(distance_threshold=cell * 0.6, ransac_n=3, num_iterations=300)
        try:
            plane, inliers = pc.segment_plane(**args, probability=0.999, seed=0)
        except TypeError:  # older Open3D
            plane, inliers = pc.segment_plane(**args)
        if len(inliers) < 0.25 * len(out):
            break
        a, b, c, d = plane
        if abs(a * seed[0] + b * seed[1] + c * seed[2] + d) <= cell * 0.6:
            break  # the user clicked the surface itself: keep it
        keep = np.ones(len(out), dtype=bool)
        keep[np.asarray(inliers)] = False
        out = out[keep]
    return out



# --------------------------------------------------------------------------
# Applying edits to whole clouds


def project_xyz(data: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(N, 3) float64 positions of a structured array's points after ``t``."""
    t = np.asarray(t, dtype=np.float64)
    xyz = np.empty((len(data), 3), dtype=np.float64)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = data["x"], data["y"], data["z"]
    return xyz @ t[:3, :3].T + t[:3, 3]


def filter_clouds(clouds, transforms, stack: EditStack | None, keys, part: Part | None = None):
    """The clouds with deleted points (and points outside ``part``) left
    out. Clouds are :class:`~pcd_merge.pcd_io.PointCloud` objects in their
    own frames; the edits are tested where ``transforms`` put them."""
    from .pcd_io import PointCloud

    if not stack and part is None:
        return list(clouds)
    out = []
    for c, t, key in zip(clouds, transforms, keys):
        keep = (stack or EditStack()).keep(project_xyz(c.data, t), key, part)
        out.append(c if keep.all() else
                   PointCloud(c.data[keep], c.path, c.viewpoint, dict(c.extra)))
    return out
