"""DXF files (AutoCAD R12 text format) for any CAD program: AutoCAD,
BricsCAD, DraftSight, LibreCAD, QCAD, FreeCAD, SketchUp, Rhino, Revit.

R12 is the oldest and most widely read DXF version and needs nothing but
plain text. Units are metres (DXF R12 files carry no unit: choose metres
when importing).

* :func:`write_drawings` writes traced plans and sections: the lines on a
  ``WALLS`` layer, round things on ``ROUND``, and the slice's points on
  ``SCAN`` (to trace the rest by hand, or turn off).
* :func:`write_model` writes the fitted solids in 3D as 3D faces, one layer
  per kind (walls, slabs, roofs, columns, pipes).
"""

from __future__ import annotations

import math
import re

import numpy as np

# AutoCAD colour numbers
COLOURS = {"WALLS": 7, "ROUND": 4, "SCAN": 8, "TEXT": 2,
           "WALL": 7, "SLAB": 3, "ROOF": 1, "COLUMN": 4, "PIPE": 6, "PLANE": 5}


def _num(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def layer_name(name: str) -> str:
    """Layer names: letters, digits, - and _ only, upper case."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").upper()
    return s[:31] or "0"


class DXF:
    def __init__(self):
        self.layers: dict[str, int] = {}
        self.body: list[str] = []
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)

    def _layer(self, name: str, colour: int | None = None) -> str:
        name = layer_name(name)
        if name not in self.layers:
            base = name.split("-")[-1]
            self.layers[name] = colour if colour is not None else COLOURS.get(base, 7)
        return name

    def _grow(self, *pts):
        for p in pts:
            p = np.asarray(p, float)
            p = np.r_[p, 0.0] if len(p) == 2 else p
            self.lo = np.minimum(self.lo, p)
            self.hi = np.maximum(self.hi, p)

    def _xyz(self, code: int, p) -> list[str]:
        p = list(map(float, p)) + [0.0] * (3 - len(p))
        return [str(code), _num(p[0]), str(code + 10), _num(p[1]), str(code + 20), _num(p[2])]

    def line(self, layer: str, a, b) -> None:
        self._grow(a, b)
        self.body += ["0", "LINE", "8", self._layer(layer)] + self._xyz(10, a) + self._xyz(11, b)

    def circle(self, layer: str, centre, r: float) -> None:
        c = np.asarray(centre, float)
        self._grow(c[:2] - r, c[:2] + r)
        self.body += ["0", "CIRCLE", "8", self._layer(layer)] + self._xyz(10, c) + \
            ["40", _num(r)]

    def point(self, layer: str, p) -> None:
        self.body += ["0", "POINT", "8", self._layer(layer)] + self._xyz(10, p)

    def points(self, layer: str, pts) -> None:
        pts = np.asarray(pts, float)
        if not len(pts):
            return
        self._grow(pts.min(axis=0), pts.max(axis=0))
        lay = self._layer(layer)
        pts = np.column_stack([pts, np.zeros(len(pts))]) if pts.shape[1] == 2 else pts
        for x, y, z in pts:
            self.body += ["0", "POINT", "8", lay, "10", _num(x), "20", _num(y), "30", _num(z)]

    def text(self, layer: str, p, height: float, text: str) -> None:
        text = text.replace("\n", " ")
        self.body += ["0", "TEXT", "8", self._layer(layer)] + self._xyz(10, p) + \
            ["40", _num(height), "1", text]

    def face(self, layer: str, quad) -> None:
        """A 3D face (3 or 4 corners)."""
        q = [np.asarray(p, float) for p in quad]
        if len(q) == 3:
            q.append(q[2])
        self._grow(*q)
        out = ["0", "3DFACE", "8", self._layer(layer)]
        for k, p in enumerate(q):
            out += self._xyz(10 + k, p)
        self.body += out

    def save(self, path: str) -> None:
        lo = np.where(np.isfinite(self.lo), self.lo, 0.0)
        hi = np.where(np.isfinite(self.hi), self.hi, 0.0)
        out = ["0", "SECTION", "2", "HEADER",
               "9", "$ACADVER", "1", "AC1009",
               "9", "$INSBASE", "10", "0.0", "20", "0.0", "30", "0.0",
               "9", "$EXTMIN"] + self._xyz(10, lo) + ["9", "$EXTMAX"] + self._xyz(10, hi) + \
              ["0", "ENDSEC",
               "0", "SECTION", "2", "TABLES",
               "0", "TABLE", "2", "LTYPE", "70", "1",
               "0", "LTYPE", "2", "CONTINUOUS", "70", "0", "3", "Solid line", "72", "65",
               "73", "0", "40", "0.0",
               "0", "ENDTAB",
               "0", "TABLE", "2", "LAYER", "70", str(len(self.layers) + 1),
               "0", "LAYER", "2", "0", "70", "0", "62", "7", "6", "CONTINUOUS"]
        for name, colour in self.layers.items():
            out += ["0", "LAYER", "2", name, "70", "0", "62", str(colour), "6", "CONTINUOUS"]
        out += ["0", "ENDTAB", "0", "ENDSEC",
                "0", "SECTION", "2", "BLOCKS", "0", "ENDSEC",
                "0", "SECTION", "2", "ENTITIES"] + self.body + ["0", "ENDSEC", "0", "EOF"]
        with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as f:
            f.write("\n".join(out) + "\n")


def write_drawings(path: str, drawings, points: bool = True, labels: bool = True) -> dict:
    """Plans and sections, each on its own layers (named after it when there
    are several). Returns counts of what was written."""
    dxf = DXF()
    n_lines = n_circles = n_points = 0
    many = len(drawings) > 1
    for d in drawings:
        pre = (layer_name(d.name) + "-") if many and d.name else ""
        for a, b in np.asarray(d.segments, float).reshape(-1, 2, 2):
            dxf.line(pre + "WALLS", a, b)
            n_lines += 1
        for x, y, r in np.asarray(d.circles, float).reshape(-1, 3):
            dxf.circle(pre + "ROUND", (x, y), r)
            n_circles += 1
        if points and len(d.points):
            dxf.points(pre + "SCAN", d.points)
            n_points += len(d.points)
        if labels and d.name:
            if len(d.segments):
                lo = np.asarray(d.segments, float).reshape(-1, 2).min(axis=0)
            elif len(d.points):
                lo = d.points.min(axis=0)
            else:
                lo = np.zeros(2)
            dxf.text(pre + "TEXT", (lo[0], lo[1] - 0.6), 0.25, _title(d))
    dxf.save(path)
    return {"lines": n_lines, "circles": n_circles, "points": n_points}


def _title(d) -> str:
    if d.kind == "plan":
        return f"{d.name}  (plan cut at {d.z:.2f} m)"
    return f"{d.name}  (section)"


def write_model(path: str, elements, segments: int = 24) -> int:
    """The fitted solids as 3D faces. Returns the number of solids written."""
    dxf = DXF()
    n = 0
    for e in elements:
        layer = e.kind.upper()
        for quad in faces_of(e, segments):
            dxf.face(layer, quad)
        n += 1
    dxf.save(path)
    return n


BOX_FACES = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


def faces_of(e, segments: int = 24) -> list:
    """An element's surface as quads (and triangles, as quads with a repeated corner)."""
    if e.shape != "cylinder":
        c = e.corners()
        return [[c[i] for i in f] for f in BOX_FACES]
    o = np.asarray(e.origin, float)
    u, v, w = e.axes()
    r, h = e.radius, float(e.size[2])
    th = np.linspace(0, 2 * math.pi, segments + 1)
    ring = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    out = []
    top = o + h * w
    for k in range(segments):
        a, b = ring[k], ring[k + 1]
        out.append([o + a, o + b, top + b, top + a])
        out.append([o, o + b, o + a])
        out.append([top, top + a, top + b])
    return out
