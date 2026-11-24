"""The building model: levels, traced drawings and fitted solids, saved in
the project folder (``model/model.json``) so it is there next time."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field

import numpy as np

from .levels import Level

KINDS = {
    "wall": "Wall",
    "slab": "Floor slab",
    "roof": "Roof",
    "column": "Column",
    "pipe": "Pipe",
    "plane": "Surface",
}


@dataclass
class Element:
    """One solid of the model.

    A **box** is ``origin + a*size[0]*u + b*size[1]*v + c*size[2]*w`` for
    a, b, c in 0..1: a rectangle in u, v pushed out along w (for a wall: its
    length, its thickness, and up its height).

    A **cylinder** has its base centre at ``origin``, its axis along ``w``,
    radius ``size[0]`` and length ``size[2]``.
    """

    kind: str
    shape: str  # "box" or "cylinder"
    origin: list
    u: list
    v: list
    w: list
    size: list
    name: str = ""
    level: int | None = None
    rms: float = 0.0  # how far the scan points were from the fitted surface (m)
    points: int = 0  # scan points that support it
    visible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def radius(self) -> float:
        return float(self.size[0])

    def axes(self):
        return (np.asarray(self.u, float), np.asarray(self.v, float),
                np.asarray(self.w, float))

    def corners(self) -> np.ndarray:
        """The 8 corners of a box (for a cylinder: of its bounding box)."""
        o = np.asarray(self.origin, float)
        u, v, w = self.axes()
        if self.shape == "cylinder":
            r = self.radius
            o = o - r * u - r * v
            su, sv, sw = 2 * r, 2 * r, float(self.size[2])
        else:
            su, sv, sw = map(float, self.size)
        out = []
        for c in (0, 1):
            for b in (0, 1):
                for a in (0, 1):
                    out.append(o + a * su * u + b * sv * v + c * sw * w)
        return np.array(out)

    def edges(self, segments: int = 24) -> np.ndarray:
        """Line segments (E, 2, 3) to draw it as a wireframe."""
        if self.shape == "cylinder":
            o = np.asarray(self.origin, float)
            u, v, w = self.axes()
            r, h = self.radius, float(self.size[2])
            th = np.linspace(0, 2 * np.pi, segments + 1)
            ring = r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
            out = []
            for base in (o, o + h * w):
                pts = base + ring
                out += [[pts[k], pts[k + 1]] for k in range(segments)]
            for k in range(0, segments, max(1, segments // 4)):
                out.append([o + ring[k], o + h * w + ring[k]])
            return np.array(out)
        c = self.corners()
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        return np.array([[c[i], c[j]] for i, j in pairs])

    def bottom(self) -> float:
        return float(self.corners()[:, 2].min())

    def level_height(self) -> float:
        """The height that says which level it is on: a floor slab belongs to
        the level it carries, everything else to the level it stands on."""
        z = self.corners()[:, 2]
        return float(z.max()) - 0.05 if self.kind == "slab" else float(z.min()) + 0.05

    def summary(self) -> str:
        if self.shape == "cylinder":
            return f"Ø {2 * self.radius * 1000:.0f} mm × {self.size[2]:.2f} m"
        a, b, c = self.size
        if self.kind == "wall":
            return f"{a:.2f} m long, {b * 1000:.0f} mm thick, {c:.2f} m high"
        if self.kind == "slab":
            return f"{a:.2f} × {b:.2f} m, {c * 1000:.0f} mm thick"
        return f"{a:.2f} × {b:.2f} m, {c * 1000:.0f} mm thick"

    def label(self) -> str:
        return self.name or KINDS.get(self.kind, self.kind.title())

    def to_dict(self) -> dict:
        d = dict(vars(self))
        for k in ("origin", "u", "v", "w", "size"):
            d[k] = [float(x) for x in d[k]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        known = {k: d[k] for k in ("kind", "shape", "origin", "u", "v", "w", "size", "name",
                                   "level", "rms", "points", "visible", "id") if k in d}
        return cls(**known)


def box(kind: str, origin, u, v, w, size, **kw) -> Element:
    return Element(kind, "box", list(map(float, origin)), list(map(float, u)),
                   list(map(float, v)), list(map(float, w)), list(map(float, size)), **kw)


def cylinder(kind: str, base, axis, radius: float, length: float, **kw) -> Element:
    w = np.asarray(axis, float)
    w = w / np.linalg.norm(w)
    u = np.cross(w, [0, 0, 1] if abs(w[2]) < 0.9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return Element(kind, "cylinder", list(map(float, base)), u.tolist(), v.tolist(), w.tolist(),
                   [float(radius), float(radius), float(length)], **kw)


@dataclass
class Model:
    levels: list = field(default_factory=list)  # [Level]
    elements: list = field(default_factory=list)  # [Element]
    settings: dict = field(default_factory=dict)

    def level_of(self, z: float) -> int | None:
        """The level a height belongs to (from just below its floor up to
        just below the next floor)."""
        if not self.levels:
            return None
        k = None
        for i, lv in enumerate(sorted(self.levels, key=lambda lv: lv.floor)):
            if z >= lv.floor - 0.25:
                k = i
        return 0 if k is None else k

    def assign_levels(self) -> None:
        for e in self.elements:
            e.level = self.level_of(e.level_height())

    def element(self, eid: str):
        return next((e for e in self.elements if e.id == eid), None)

    def counts(self) -> dict:
        out = {}
        for e in self.elements:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {"version": 1, "levels": [lv.to_dict() for lv in self.levels],
                "elements": [e.to_dict() for e in self.elements], "settings": self.settings}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Model":
        d = d or {}
        return cls([Level.from_dict(x) for x in d.get("levels", [])],
                   [Element.from_dict(x) for x in d.get("elements", [])],
                   dict(d.get("settings", {})))

    # ---- in the project folder

    @staticmethod
    def path(folder: str) -> str:
        return os.path.join(folder, "model", "model.json")

    def save(self, folder: str) -> None:
        p = self.path(folder)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, p)

    @classmethod
    def load(cls, folder: str) -> "Model":
        p = cls.path(folder)
        if not os.path.isfile(p):
            return cls()
        try:
            with open(p, encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except (OSError, ValueError, KeyError, TypeError):
            return cls()
