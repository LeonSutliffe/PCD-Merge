"""Data side of the first-person fly-through: no Qt or OpenGL in here.

Building-sized scans have tens of millions of points, far more than can be
drawn every frame. The fly-through therefore works in two steps:

1. :func:`build_scene` (run once, in the background) takes a random subset of
   up to ``max_points`` points from the clouds, moves them with the alignment
   transforms and sorts them into *chunks*: cubes of a regular grid. Inside a
   chunk the points are in random order, so the first k points of a chunk are
   a random k-point sample of it. For each chunk we also estimate how far apart
   its points are on the surfaces they sample (``spacing``).

2. :func:`plan_frame` (run every frame) decides how many points of each chunk
   to draw. Chunks outside the field of view are skipped. For the rest, the
   fraction drawn is chosen so that, at that chunk's distance, the drawn points
   are about ``px_target`` pixels apart on screen: close chunks are drawn in
   full, far chunks only sparsely, and the drawn points are made larger to fill
   the gaps. If that still exceeds the frame's point budget, everything is
   scaled down evenly. Fractions are rounded to powers of two so the whole
   frame is a dozen draw calls, one per level of detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MAX_POINTS = 16_000_000  # points kept on the graphics card
TARGET_CHUNK_POINTS = 2_000
LEVELS = 12  # fractions 1, 1/2, ..., 1/2048

COLOUR_MODES = ("Colour (RGB)", "Intensity", "Height", "Per scan")

PALETTE = [
    (0.95, 0.35, 0.25), (0.25, 0.65, 0.95), (0.35, 0.85, 0.35), (0.95, 0.80, 0.20),
    (0.75, 0.40, 0.90), (0.20, 0.85, 0.80), (0.95, 0.55, 0.75), (0.65, 0.65, 0.65),
]

# Interleaved vertex layout uploaded to the graphics card as is.
VERTEX_DTYPE = np.dtype([("pos", "<f4", 3), ("rgba", "u1", 4), ("spacing", "<f4")])


@dataclass
class Scene:
    vertices: np.ndarray  # VERTEX_DTYPE, sorted by chunk
    origin: np.ndarray  # world position of (0, 0, 0) in ``vertices``
    chunk_start: np.ndarray  # int64, first vertex of each chunk
    chunk_count: np.ndarray  # int64
    chunk_centre: np.ndarray  # (C, 3) float32, relative to origin
    chunk_radius: np.ndarray  # (C,) float32
    chunk_spacing: np.ndarray  # (C,) float32, typical point spacing on surfaces
    cell: float
    lo: np.ndarray  # bounding box (2nd/98th percentile), relative to origin
    hi: np.ndarray
    scan_positions: list = field(default_factory=list)  # (name, xyz relative to origin)
    names: list = field(default_factory=list)
    total_points: int = 0  # points in the source clouds
    # Kept to recolour without rebuilding:
    scan_id: np.ndarray | None = None
    intensity: np.ndarray | None = None
    rgb: np.ndarray | None = None
    # Editing in the walk-through (see :class:`SceneEdits`): the scan key of
    # each scan id (None: a merged file, which edits don't touch), whether
    # the points are in the project frame (aligned, where edits and
    # measurements live), and per vertex what is hidden.
    scan_keys: list = field(default_factory=list)
    project_frame: bool = True
    hidden: np.ndarray | None = None  # bool: not drawn at all
    chunk_live: np.ndarray | None = None  # per chunk: points not hidden
    edits: object = None  # a SceneEdits, once edits are applied

    def __len__(self) -> int:
        return len(self.vertices)

    @property
    def colour_modes(self) -> list[str]:
        modes = []
        if self.rgb is not None:
            modes.append("Colour (RGB)")
        if self.intensity is not None:
            modes.append("Intensity")
        modes.append("Height")
        if len(self.names) > 1:
            modes.append("Per scan")
        return modes

    def colours(self, mode: str) -> np.ndarray:
        """(N, 4) uint8 colours for a colour mode."""
        n = len(self.vertices)
        out = np.empty((n, 4), dtype=np.uint8)
        out[:, 3] = 255
        if mode == "Colour (RGB)" and self.rgb is not None:
            out[:, :3] = self.rgb
        elif mode == "Intensity" and self.intensity is not None:
            v = self.intensity
            lo, hi = np.percentile(v, [1, 99]) if n else (0.0, 1.0)
            v = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1) ** 0.7
            # Slightly warm grey reads better than pure grey on a dark background.
            out[:, 0] = (30 + 225 * v).astype(np.uint8)
            out[:, 1] = (28 + 222 * v).astype(np.uint8)
            out[:, 2] = (26 + 214 * v).astype(np.uint8)
        elif mode == "Per scan" and self.scan_id is not None:
            pal = (np.array(PALETTE) * 255).astype(np.uint8)
            out[:, :3] = pal[self.scan_id % len(pal)]
        else:
            z = self.vertices["pos"][:, 2]
            lo, hi = float(self.lo[2]), float(self.hi[2])
            out[:, :3] = height_colours(z, lo, hi)
        return out

    def set_colours(self, mode: str) -> None:
        self.vertices["rgba"] = self.colours(mode)


_HEIGHT_STOPS = np.array([
    (0.00, 40, 60, 160), (0.25, 40, 150, 200), (0.50, 80, 200, 110),
    (0.75, 235, 210, 60), (1.00, 230, 80, 50),
], dtype=np.float64)


def height_colours(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(z, dtype=np.float64) - lo) / max(hi - lo, 1e-12), 0, 1)
    s = _HEIGHT_STOPS
    return np.stack([np.interp(t, s[:, 0], s[:, k]) for k in (1, 2, 3)], 1).astype(np.uint8)


def _unpack_rgb(col: np.ndarray) -> np.ndarray:
    if col.dtype.itemsize != 4:
        col = col.astype(np.uint32)
    v = np.ascontiguousarray(col).view(np.uint32)
    return np.stack([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF], 1).astype(np.uint8)


def _rgb_of(data: np.ndarray) -> np.ndarray | None:
    names = data.dtype.names
    for n in ("rgb", "rgba"):
        if n in names and data[n].ndim == 1:
            return _unpack_rgb(data[n])
    if all(n in names for n in ("red", "green", "blue")):
        cols = [data[n].astype(np.float64) for n in ("red", "green", "blue")]
        top = max(float(c.max()) if len(c) else 0.0 for c in cols)
        scale = 255.0 / 65535.0 if top > 255 else 1.0
        return np.stack([np.clip(c * scale, 0, 255) for c in cols], 1).astype(np.uint8)
    return None


def _scanner_position(cloud, xyz_local: np.ndarray) -> np.ndarray | None:
    """The scanner's position in the cloud's own frame, if the PCD VIEWPOINT is
    believable (the scan surrounds it), else None."""
    from .cleanup import plausible_origin

    vp = np.asarray(getattr(cloud, "viewpoint", (0, 0, 0))[:3], dtype=np.float64)
    try:
        ok = plausible_origin(xyz_local, vp)
    except Exception:
        ok = False
    return vp if ok else None


def _no_progress(msg, frac):
    pass


def build_scene(clouds, transforms=None, max_points: int = DEFAULT_MAX_POINTS,
                progress=None, seed: int = 0, even: bool = False,
                chunk_points: int = TARGET_CHUNK_POINTS) -> Scene:
    """Prepare clouds (pcd_io.PointCloud, already loaded) for the fly-through.

    ``transforms`` (4x4 each) move every cloud into the common frame; None
    means they are already there (a merged cloud, or one scan on its own).
    ``even``: thin to ``max_points`` evenly in space (at most one point per
    small cube) instead of at random, so areas far from every scanner keep
    as much detail as those next to one (for the browser walk-through,
    whose point budget is much smaller than the scans).
    """
    progress = progress or _no_progress
    if transforms is None:
        transforms = [np.eye(4)] * len(clouds)
    rng = np.random.default_rng(seed)
    total = sum(len(c) for c in clouds)
    keep_frac = min(1.0, (4 if even else 1) * max_points / max(total, 1))
    want_rgb = all(_rgb_of(c.data[:1]) is not None for c in clouds)
    want_int = all(c.has_field("intensity") for c in clouds)

    parts, ids, ints, rgbs, positions = [], [], [], [], []
    for i, (c, t) in enumerate(zip(clouds, transforms)):
        progress(f"Preparing fly-through: {c.name}", 0.6 * i / len(clouds))
        d = c.data
        if keep_frac < 1.0:
            d = d[rng.random(len(d)) < keep_frac]
        xyz = np.empty((len(d), 3), dtype=np.float64)
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = d["x"], d["y"], d["z"]
        ok = np.isfinite(xyz).all(axis=1)
        if not ok.all():
            xyz, d = xyz[ok], d[ok]
        t = np.asarray(t, dtype=np.float64)
        vp = _scanner_position(c, xyz[:: max(1, len(xyz) // 200_000)])
        if vp is not None:
            positions.append((c.name, t[:3, :3] @ vp + t[:3, 3]))
        parts.append(xyz @ t[:3, :3].T + t[:3, 3])
        ids.append(np.full(len(xyz), i, dtype=np.uint8 if len(clouds) < 256 else np.uint16))
        if want_int:
            ints.append(np.asarray(d["intensity"], dtype=np.float32).reshape(len(d), -1)[:, 0])
        if want_rgb:
            rgbs.append(_rgb_of(d))

    xyz = np.concatenate(parts) if parts else np.zeros((0, 3))
    del parts
    ids = np.concatenate(ids) if ids else np.zeros(0, np.uint8)
    ints = np.concatenate(ints) if want_int and ints else None
    rgbs = np.concatenate(rgbs) if want_rgb and rgbs else None
    if even and len(xyz) > max_points:
        progress("Thinning evenly", 0.62)
        keep = even_subset(xyz, max_points, rng)
        xyz, ids = xyz[keep], ids[keep]
        ints = ints[keep] if ints is not None else None
        rgbs = rgbs[keep] if rgbs is not None else None
    n = len(xyz)
    if n == 0:
        raise ValueError("there are no points to show")
    progress("Preparing fly-through: sorting into chunks", 0.65)
    lo_w = np.percentile(xyz[:: max(1, n // 500_000)], 2, axis=0)
    hi_w = np.percentile(xyz[:: max(1, n // 500_000)], 98, axis=0)
    origin = (lo_w + hi_w) / 2
    rel = (xyz - origin).astype(np.float32)
    del xyz

    cell = _pick_cell(rel, rng, chunk_points)
    keys, inverse = _chunk_keys(rel, cell)
    # Sort by chunk, with random low bits so the order inside a chunk is random.
    sort_key = (inverse.astype(np.int64) << 24) | rng.integers(0, 1 << 24, n, dtype=np.int64)
    order = np.argsort(sort_key)
    del sort_key
    progress("Preparing fly-through: measuring point spacing", 0.8)
    inv_sorted = inverse[order]
    counts = np.bincount(inv_sorted, minlength=len(keys)).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rel = rel[order]

    centre, radius = _chunk_bounds(rel, inv_sorted, counts, len(keys))
    spacing = _chunk_spacing(rel, inv_sorted, counts, cell)

    verts = np.empty(n, dtype=VERTEX_DTYPE)
    verts["pos"] = rel
    verts["spacing"] = spacing[inv_sorted]
    del rel

    scene = Scene(
        vertices=verts, origin=origin, chunk_start=starts, chunk_count=counts,
        chunk_centre=centre, chunk_radius=radius, chunk_spacing=spacing, cell=cell,
        lo=(lo_w - origin).astype(np.float32), hi=(hi_w - origin).astype(np.float32),
        scan_positions=[(name, (p - origin).astype(np.float32)) for name, p in positions],
        names=[c.name for c in clouds], total_points=total,
        scan_id=ids[order],
        intensity=ints[order] if ints is not None else None,
        rgb=rgbs[order] if rgbs is not None else None,
    )
    progress("Preparing fly-through: colours", 0.95)
    scene.set_colours(scene.colour_modes[0])
    return scene


def _chunk_keys(rel: np.ndarray, cell: float):
    g = np.floor(rel / cell).astype(np.int64)
    g -= g.min(axis=0)
    dims = g.max(axis=0) + 1
    flat = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    return np.unique(flat, return_inverse=True)


def even_subset(xyz: np.ndarray, target: int, rng) -> np.ndarray:
    """Indices of about ``target`` points spread evenly: at most one per
    cube of a grid sized for that many occupied cubes (then a random cut
    if still too many)."""
    n = len(xyz)
    if n <= target:
        return np.arange(n)
    order = rng.permutation(n)  # which point of a cube is kept: a random one
    ext = float(np.max(np.ptp(xyz, axis=0))) or 1.0
    # Points sample surfaces: occupied cubes go as 1 / cell^2.
    sample = xyz[rng.integers(0, n, min(n, 1_000_000))]
    cell = ext / 100
    for _ in range(4):
        occ = len(np.unique(_grid_keys(sample, cell)))
        est = occ * (n / len(sample)) ** 0.5 if len(sample) < n else occ
        cell *= float(np.clip(np.sqrt(est / target), 0.3, 3.0))
    for _ in range(4):
        keys = _grid_keys(xyz[order], cell)
        _, first = np.unique(keys, return_index=True)
        if len(first) >= 0.8 * target or cell < ext * 1e-6:
            break
        cell *= 0.8 * np.sqrt(len(first) / target)
    keep = order[first]
    if len(keep) > target:
        keep = rng.choice(keep, target, replace=False)
    return np.sort(keep)


def _grid_keys(xyz: np.ndarray, cell: float) -> np.ndarray:
    g = np.floor((xyz - xyz.min(axis=0)) / cell).astype(np.int64)
    return (g[:, 0] << 42) ^ (g[:, 1] << 21) ^ g[:, 2]


def _pick_cell(rel: np.ndarray, rng, target: int = TARGET_CHUNK_POINTS) -> float:
    """Grid size giving about ``target`` points per occupied cube."""
    n = len(rel)
    ext = float(np.max(np.ptp(rel, axis=0))) or 1.0
    if n <= target:
        return ext + 1e-3
    sample = rel[rng.integers(0, n, 400_000)] if n > 400_000 else rel
    cell = ext / 32
    for _ in range(3):
        per = n / len(_chunk_keys(sample, cell)[0])
        # Points sample surfaces, so points per cube grows with the cube's area.
        cell *= float(np.clip(np.sqrt(target / max(per, 1e-9)), 0.25, 4.0))
    return max(cell, ext / 2000)


def _chunk_bounds(rel, inv, counts, c):
    sums = np.stack([np.bincount(inv, weights=rel[:, k], minlength=c) for k in range(3)], 1)
    centre = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
    d = np.linalg.norm(rel - centre[inv], axis=1)
    radius = np.zeros(c, dtype=np.float32)
    np.maximum.at(radius, inv, d)
    return centre, radius


def _chunk_spacing(rel, inv, counts, cell, sub: int = 16):
    """Typical distance between neighbouring points in each chunk.

    The chunk's surface area is estimated from how many cubes of a finer grid
    (``sub`` per chunk side) contain points; spacing = sqrt(area / points).
    """
    fine = cell / sub
    g = np.floor(rel / fine).astype(np.int64) % sub
    key = inv.astype(np.int64) * (sub ** 3) + (g[:, 0] * sub + g[:, 1]) * sub + g[:, 2]
    occupied = np.bincount(np.unique(key) // (sub ** 3), minlength=len(counts))
    area = occupied * fine * fine
    return np.sqrt(area / np.maximum(counts, 1)).astype(np.float32)


# --------------------------------------------------------------------------
# Per frame


@dataclass
class FramePlan:
    # One entry per level of detail actually used.
    levels: list  # of (size_factor, firsts int32 array, counts int32 array)
    points: int


def frustum_planes(mvp: np.ndarray) -> np.ndarray:
    """(6, 4) normalised planes ax+by+cz+d >= 0 inside, from a view-projection
    matrix acting on column vectors."""
    m = np.asarray(mvp, dtype=np.float64)
    planes = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1],
                       m[3] + m[2], m[3] - m[2]])
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    return planes


def plan_frame(scene: Scene, eye: np.ndarray, mvp: np.ndarray, focal_px: float,
               budget: int, px_target: float = 1.5) -> FramePlan:
    """How many points of each chunk to draw this frame (see module docstring)."""
    c = scene.chunk_centre.astype(np.float64)
    r = scene.chunk_radius.astype(np.float64)
    planes = frustum_planes(mvp)
    vis = np.all(c @ planes[:, :3].T + planes[:, 3] >= -r[:, None], axis=1)
    if scene.chunk_live is not None:
        vis &= scene.chunk_live > 0  # every point deleted or in a hidden part
    idx = np.flatnonzero(vis)
    if len(idx) == 0:
        return FramePlan([], 0)
    dist = np.maximum(np.linalg.norm(c[idx] - eye, axis=1) - r[idx], 1e-3)
    counts = scene.chunk_count[idx].astype(np.float64)
    sp = np.maximum(scene.chunk_spacing[idx].astype(np.float64), 1e-9)
    # Spacing on screen we want, turned into a spacing in the world at that
    # distance; a chunk's points, thinned to fraction f, are sp / sqrt(f) apart.
    want = dist * px_target / max(focal_px, 1.0)
    frac = np.minimum(1.0, (sp / want) ** 2)
    total = float(np.sum(frac * counts))
    if total > budget:
        frac *= budget / total
        # Chunks capped at 1 above don't change; one correction step is plenty.
        frac = np.minimum(frac, 1.0)
    level = np.clip(np.round(-np.log2(np.maximum(frac, 2.0 ** -(LEVELS - 1)))), 0, LEVELS - 1)
    level = level.astype(np.int64)
    n_draw = np.maximum(1, np.ceil(counts * 0.5 ** level)).astype(np.int64)
    out, drawn = [], 0
    starts = scene.chunk_start[idx]
    for lv in np.unique(level):
        m = level == lv
        firsts = starts[m].astype(np.int32)
        cnt = n_draw[m].astype(np.int32)
        drawn += int(cnt.sum())
        out.append((float(np.sqrt(2.0 ** lv)), firsts, cnt))
    return FramePlan(out, drawn)


# --------------------------------------------------------------------------
# Camera


@dataclass
class FlyCamera:
    """First-person camera, Z up. Yaw 0 looks along +X, pitch up is positive."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0  # degrees
    pitch: float = 0.0
    fov: float = 75.0  # vertical, degrees
    near: float = 0.05
    far: float = 1000.0

    def forward(self) -> np.ndarray:
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        return np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])

    def right(self) -> np.ndarray:
        y = np.radians(self.yaw)
        return np.array([np.sin(y), -np.cos(y), 0.0])

    def look(self, dx_deg: float, dy_deg: float) -> None:
        self.yaw = (self.yaw - dx_deg) % 360.0
        self.pitch = float(np.clip(self.pitch - dy_deg, -89.0, 89.0))

    def look_at(self, target) -> None:
        d = np.asarray(target, dtype=np.float64) - self.pos
        if np.linalg.norm(d) < 1e-9:
            return
        self.yaw = float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0
        self.pitch = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))

    def move(self, forward: float, right: float, up: float, follow_look: bool = False) -> None:
        """Move by distances along the view (horizontal unless ``follow_look``),
        to the right, and straight up."""
        f = self.forward()
        if not follow_look:
            f = np.array([f[0], f[1], 0.0])
            n = np.linalg.norm(f)
            f = f / n if n > 1e-9 else np.array([np.cos(np.radians(self.yaw)),
                                                 np.sin(np.radians(self.yaw)), 0.0])
        self.pos = self.pos + forward * f + right * self.right() + up * np.array([0, 0, 1.0])

    def view_matrix(self) -> np.ndarray:
        f = self.forward()
        s = np.cross(f, [0.0, 0.0, 1.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -m[:3, :3] @ self.pos
        return m

    def projection_matrix(self, aspect: float) -> np.ndarray:
        t = 1.0 / np.tan(np.radians(self.fov) / 2)
        n, f = self.near, self.far
        m = np.zeros((4, 4))
        m[0, 0] = t / max(aspect, 1e-6)
        m[1, 1] = t
        m[2, 2] = (f + n) / (n - f)
        m[2, 3] = 2 * f * n / (n - f)
        m[3, 2] = -1.0
        return m

    def focal_px(self, height_px: float) -> float:
        return height_px / 2 / np.tan(np.radians(self.fov) / 2)


def start_camera(scene: Scene) -> FlyCamera:
    """Where the walk starts: the first scanner position if known (that is
    inside the building, at tripod height), else above the middle of the scan
    looking at it."""
    diag = float(np.linalg.norm(scene.hi - scene.lo)) or 1.0
    cam = FlyCamera(near=max(diag * 1e-5, 0.01), far=max(diag * 4, 10.0))
    if scene.scan_positions:
        cam.pos = np.asarray(scene.scan_positions[0][1], dtype=np.float64).copy()
        centre = (scene.lo + scene.hi) / 2
        cam.look_at([centre[0], centre[1], cam.pos[2]])
    else:
        centre = (scene.lo.astype(np.float64) + scene.hi) / 2
        cam.pos = np.array([centre[0], scene.lo[1] - 0.3 * diag, centre[2] + 0.3 * diag])
        cam.look_at(centre)
    return cam


def default_speed(scene: Scene) -> float:
    """Walking pace in scan units per second (scans are usually in metres)."""
    diag = float(np.linalg.norm(scene.hi - scene.lo))
    if 5 <= diag <= 3000:
        return 2.0
    return max(diag / 30, 1e-3)


# --------------------------------------------------------------------------
# Editing and measuring in the walk-through
#
# The walk-through's points are a sample of the scans, already moved into
# the project frame. Edits are regions in that frame, so they are tested on
# the sample directly: no full-resolution points are needed to show them.

BLOCK = 2_000_000  # vertices tested at a time (keeps temporary arrays small)


def chunk_ranges(scene: Scene, chunks) -> np.ndarray:
    """Vertex indices of these chunks, in order."""
    chunks = np.asarray(chunks, dtype=np.int64)
    if not len(chunks):
        return np.zeros(0, dtype=np.int64)
    starts, counts = scene.chunk_start[chunks], scene.chunk_count[chunks]
    total = int(counts.sum())
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return np.arange(total, dtype=np.int64) + offs


def world(scene: Scene, idx=None) -> np.ndarray:
    """Project-frame (float64) positions of vertices ``idx`` (all if None)."""
    pos = scene.vertices["pos"] if idx is None else scene.vertices["pos"][idx]
    return pos.astype(np.float64) + scene.origin


def region_mask(scene: Scene, region) -> np.ndarray:
    """Which vertices are inside ``region`` (an edits Region, or a Part)."""
    n = len(scene)
    out = np.zeros(n, dtype=bool)
    test = region.mask if hasattr(region, "mask") else region.contains
    bounds = region.bounds() if hasattr(region, "bounds") else None
    if bounds is not None:
        lo, hi = (np.asarray(b, float) - scene.origin for b in bounds)
        c = scene.chunk_centre.astype(np.float64)
        r = scene.chunk_radius.astype(np.float64)[:, None]
        near = np.flatnonzero(np.all((c + r >= lo) & (c - r <= hi), axis=1))
        idx = chunk_ranges(scene, near)
        for k in range(0, len(idx), BLOCK):
            part = idx[k:k + BLOCK]
            out[part] = test(world(scene, part))
        return out
    for k in range(0, n, BLOCK):
        out[k:k + BLOCK] = test(world(scene, slice(k, k + BLOCK)))
    return out


class SceneEdits:
    """The project's edits applied to a walk-through scene.

    Each region is tested on the scene once and remembered (packed to one
    bit per point), so after an edit, an undo or a part shown or hidden only
    new regions cost anything.
    """

    def __init__(self, scene: Scene):
        self.scene = scene
        self._masks = {}  # id(region or part) -> (the object, packed mask)
        ids = scene.scan_id
        self._scan_of = ids if ids is not None else np.zeros(len(scene), np.uint8)
        self.deleted = np.zeros(len(scene), dtype=bool)
        self.outside_parts = np.zeros(len(scene), dtype=bool)

    def mask(self, obj) -> np.ndarray:
        hit = self._masks.get(id(obj))
        if hit is not None and hit[0] is obj:
            return np.unpackbits(hit[1], count=len(self.scene)).astype(bool)
        m = region_mask(self.scene, obj)
        self._masks[id(obj)] = (obj, np.packbits(m))
        return m

    def _scans_in(self, keys) -> np.ndarray:
        """Vertices of the scans with these keys (merged files never)."""
        want = np.array([k is not None and k in keys for k in self.scene.scan_keys]
                        or [False], dtype=bool)
        return want[np.minimum(self._scan_of, len(want) - 1)]

    def update(self, stack) -> None:
        """Recompute what is deleted and what lies outside the visible parts."""
        scene = self.scene
        n = len(scene)
        editable = self._scans_in(set(k for k in scene.scan_keys if k is not None))
        deleted = np.zeros(n, dtype=bool)
        outside = np.zeros(n, dtype=bool)
        if stack is not None:
            for e in stack.point_edits:
                m = self.mask(e.region) & editable
                if e.scans is not None:
                    m &= self._scans_in(set(e.scans))
                if e.action == "delete":
                    deleted |= m
                else:
                    deleted &= ~m
            if stack.parts:
                inside = np.zeros(n, dtype=bool)
                for p in stack.parts:
                    if p.visible:
                        inside |= self.mask(p)
                outside = ~inside & editable
        self.deleted, self.outside_parts = deleted, outside

    def hidden(self, show_deleted: bool = False) -> np.ndarray:
        return self.outside_parts if show_deleted else self.outside_parts | self.deleted

    def apply(self, show_deleted: bool = False) -> None:
        """Store what is hidden on the scene (for drawing and picking)."""
        h = self.hidden(show_deleted)
        self.scene.hidden = h if h.any() else None
        if self.scene.hidden is None:
            self.scene.chunk_live = None
        else:
            chunk = np.repeat(np.arange(len(self.scene.chunk_count)), self.scene.chunk_count)
            self.scene.chunk_live = np.bincount(chunk[~h], minlength=len(self.scene.chunk_count))


def pick(scene: Scene, mvp: np.ndarray, ndc, size_px, radius_px: float = 7.0,
         skip: np.ndarray | None = None):
    """The nearest point under a screen position, as an index into the
    vertices, or None. ``mvp`` maps scene positions (relative to the
    origin) to clip space; ``ndc`` is the position in device coordinates
    (-1..1) and ``size_px`` the view's (width, height) in pixels."""
    m = np.asarray(mvp, dtype=np.float64)
    half = np.array([size_px[0] / 2.0, size_px[1] / 2.0])
    target = np.asarray(ndc, dtype=np.float64)
    c = scene.chunk_centre.astype(np.float64)
    r = scene.chunk_radius.astype(np.float64)
    clip = c @ m[:3, :3].T + m[:3, 3]
    w = c @ m[3, :3] + m[3, 3]
    scale = max(abs(m[1, 1]), abs(m[0, 0])) * half.max()  # pixels per unit at depth 1
    inside = w <= r * 1.01  # the camera is in (or right next to) the chunk
    ok = w > 1e-9
    d = np.full(len(c), np.inf)
    d[ok] = np.linalg.norm((clip[ok, :2] / w[ok, None] - target) * half, axis=1)
    rad = np.where(ok, r * scale / np.maximum(w, 1e-9), np.inf)
    cand = np.flatnonzero(inside | (d <= rad + radius_px))
    if scene.chunk_live is not None:
        cand = cand[scene.chunk_live[cand] > 0]
    idx = chunk_ranges(scene, cand)
    if skip is not None and len(idx):
        idx = idx[~skip[idx]]
    best, best_w = None, np.inf
    for k in range(0, len(idx), BLOCK):
        part = idx[k:k + BLOCK]
        p = scene.vertices["pos"][part].astype(np.float64)
        pc = p @ m[:3, :3].T + m[:3, 3]
        pw = p @ m[3, :3] + m[3, 3]
        front = pw > 1e-9
        if not front.any():
            continue
        dd = (pc[front, :2] / pw[front, None] - target) * half
        near = np.einsum("ij,ij->i", dd, dd) <= radius_px * radius_px
        if near.any():
            ws = pw[front][near]
            j = int(np.argmin(ws))
            if ws[j] < best_w:
                best_w, best = ws[j], int(part[front][near][j])
    return best


def near_points(scene: Scene, centre, radius: float, skip: np.ndarray | None = None):
    """Project-frame positions of the vertices within ``radius`` (a cube) of
    ``centre`` (project frame): what grow-to-object searches."""
    lo = np.asarray(centre, float) - scene.origin - radius
    hi = lo + 2 * radius
    c = scene.chunk_centre.astype(np.float64)
    r = scene.chunk_radius.astype(np.float64)[:, None]
    idx = chunk_ranges(scene, np.flatnonzero(np.all((c + r >= lo) & (c - r <= hi), axis=1)))
    if skip is not None and len(idx):
        idx = idx[~skip[idx]]
    p = scene.vertices["pos"][idx].astype(np.float64)
    keep = np.all((p >= lo) & (p <= hi), axis=1)
    return p[keep] + scene.origin
