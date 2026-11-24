"""The browser walk-through: one .html file that anyone can open by
double-clicking it (Chrome, Edge, Firefox, Safari), with no app, no
internet and no web server. It opens in walk-through mode.

Browsers will not let a page opened from a file read other files next to
it, so everything is inside the one file: the viewer (plain WebGL, in
:mod:`webview_page`) and the points. The points are the walk-through's
(:func:`flyscene.build_scene`): thinned evenly to a budget the browser can
hold (a few million), sorted into chunks with random order inside each, so
the viewer can draw near chunks in full and far ones sparsely, exactly like
the app's walk-through.

Each chunk's points are stored as 16-bit offsets inside the chunk's box
(sub-millimetre for any chunk smaller than 65 m), with 8-bit colour and
intensity: 6-10 bytes a point, gzip-compressed and base64-encoded in the
page (about 9-12 bytes a point in the file: 5 million points, about 50 MB).
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import time

import numpy as np

from .flyscene import Scene, default_speed, start_camera

WEB_CHUNK_POINTS = 8_000  # bigger chunks than the app: fewer draw calls in a browser
# name: (points, what it is for)
SIZES = {
    "Small (2 million points, about 20 MB)": 2_000_000,
    "Standard (5 million points, about 50 MB)": 5_000_000,
    "Detailed (10 million points, about 100 MB)": 10_000_000,
}
DEFAULT_SIZE = "Standard (5 million points, about 50 MB)"


def quantise(scene: Scene):
    """Per chunk: lowest corner and step; per point: uint16 x, y, z."""
    pos = scene.vertices["pos"]
    c = len(scene.chunk_count)
    chunk = np.repeat(np.arange(c), scene.chunk_count)
    lo = np.full((c, 3), np.inf, np.float64)
    hi = np.full((c, 3), -np.inf, np.float64)
    np.minimum.at(lo, chunk, pos)
    np.maximum.at(hi, chunk, pos)
    step = np.maximum(hi - lo, 1e-6) / 65535.0
    q = np.rint((pos - lo[chunk]) / step[chunk])
    q = np.clip(q, 0, 65535).astype("<u2")
    return lo.astype(np.float32), step.astype(np.float32), q


def _intensity_u8(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, [1, 99]) if len(v) else (0.0, 1.0)
    g = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1) ** 0.7
    return np.rint(g * 255).astype(np.uint8)


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def pack(scene: Scene) -> tuple[dict, bytes]:
    """The page's description (JSON) and its binary point data."""
    lo, step, q = quantise(scene)
    c = len(scene.chunk_count)
    table = np.zeros((c, 11), np.float32)
    table[:, 0:3], table[:, 3:6] = lo, step
    table[:, 6:9] = scene.chunk_centre
    table[:, 9] = scene.chunk_radius
    table[:, 10] = scene.chunk_spacing
    parts = [table.astype("<f4").tobytes(), scene.chunk_count.astype("<u4").tobytes(),
             _pad4(q.tobytes())]
    has_rgb = scene.rgb is not None
    has_int = scene.intensity is not None
    if has_rgb:
        parts.append(_pad4(np.ascontiguousarray(scene.rgb, np.uint8).tobytes()))
    if has_int:
        parts.append(_pad4(_intensity_u8(np.asarray(scene.intensity, np.float64)).tobytes()))
    meta = {"points": int(len(scene)), "chunks": int(c), "has_rgb": has_rgb,
            "has_intensity": has_int, "origin": [float(v) for v in scene.origin],
            "lo": [float(v) for v in scene.lo], "hi": [float(v) for v in scene.hi]}
    return meta, b"".join(parts)


def page_html(scene: Scene, title: str = "Walk-through", measurements=(),
              note: str = "", app_version: str = "") -> str:
    """The whole page as text."""
    from .webview_page import PAGE

    meta, data = pack(scene)
    cam = start_camera(scene)
    origin = np.asarray(scene.origin, np.float64)
    meta.update({
        "title": title,
        "note": note,
        "made": time.strftime("%Y-%m-%d"),
        "app": f"PCD Merge {app_version}".strip(),
        "start": {"pos": [float(v) for v in cam.pos], "yaw": float(cam.yaw),
                  "pitch": float(cam.pitch)},
        "near": float(cam.near), "far": float(cam.far),
        "speed": float(default_speed(scene)),
        "scans": [[str(n), [float(v) for v in p]] for n, p in scene.scan_positions],
        "measurements": [
            {"name": m.name, "kind": m.kind,
             "points": [[float(v) for v in np.asarray(p, float) - origin] for p in m.points]}
            for m in measurements if getattr(m, "visible", True)],
    })
    blob = base64.b64encode(gzip.compress(data, compresslevel=6, mtime=0)).decode("ascii")
    step = 4 << 20  # several script blocks: each one is decoded on its own
    blocks = "\n".join(f'<script type="application/octet-stream" class="pts">{blob[k:k + step]}'
                       "</script>" for k in range(0, len(blob), step))
    meta_json = json.dumps(meta, separators=(",", ":")).replace("</", "<\\/")
    return (PAGE.replace("{{TITLE}}", html.escape(title))
            .replace("{{META}}", meta_json)
            .replace("{{DATA}}", blocks))


def write_page(path: str, scene: Scene, title: str = "Walk-through", measurements=(),
               note: str = "", app_version: str = "") -> int:
    """Write the page; returns its size in bytes."""
    text = page_html(scene, title, measurements, note, app_version)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(text.encode("utf-8"))


def web_scene(clouds, transforms, edits=None, keys=(), merged=(), max_points=5_000_000,
              progress=None) -> Scene:
    """The points of the browser walk-through: the scans (moved into the
    project frame by ``transforms``) without deleted points or hidden parts,
    plus merged files, thinned evenly to ``max_points``."""
    from .edits import filter_clouds, project_xyz
    from .pcd_io import PointCloud

    def report(msg, frac):
        if progress:
            progress(msg, frac)

    clouds = list(clouds)
    if edits is not None:
        report("Leaving out deleted points", 0.0)
        clouds = filter_clouds(clouds, transforms, edits, keys)
        vis = [p for p in edits.parts if p.visible]
        if edits.parts:
            out = []
            for c, t in zip(clouds, transforms):
                xyz = project_xyz(c.data, t)
                keep = np.zeros(len(xyz), bool)
                for p in vis:
                    keep |= p.contains(xyz)
                out.append(c if keep.all() else PointCloud(c.data[keep], c.path, c.viewpoint,
                                                           dict(c.extra)))
            clouds = out
    transforms = list(transforms)
    if merged:
        import os

        from .readers import read_cloud

        for k, path in enumerate(merged):
            report(f"Loading {os.path.basename(path)}", 0.1)
            clouds.append(read_cloud(path))
            transforms.append(np.eye(4))
    return build_web_scene(clouds, transforms, max_points, progress)


def build_web_scene(clouds, transforms, max_points, progress=None) -> Scene:
    from .flyscene import build_scene

    def report(msg, frac):
        if progress:
            progress(msg.replace("fly-through", "browser walk-through"), 0.1 + 0.8 * frac)

    return build_scene(clouds, transforms, max_points=max_points, progress=report,
                       even=True, chunk_points=WEB_CHUNK_POINTS)


def browser_walk_job(clouds, transforms, edits, keys, merged, path, max_points, title,
                     measurements, app_version="", *, progress):
    """Background job of the app's "Make browser walk-through" button."""
    scene = web_scene(clouds, transforms, edits, keys, merged, max_points, progress)
    progress("Writing the page", 0.92)
    size = write_page(path, scene, title, measurements, app_version=app_version)
    return path, size, len(scene)
