"""Measurements made in the 3D view: distances between two points and the
coordinates of single points. Saved in the project (project frame, the
same frame the edits use) and exportable as CSV.
"""

from __future__ import annotations

import csv
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

KINDS = ("distance", "point")


@dataclass
class Measurement:
    kind: str  # "distance" (two points) or "point" (one)
    points: list  # [[x, y, z], ...] in the project frame
    name: str = ""
    visible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind of measurement: {self.kind!r}")
        self.points = [list(map(float, p)) for p in self.points]
        need = 2 if self.kind == "distance" else 1
        if len(self.points) != need:
            raise ValueError(f"a {self.kind} needs {need} point(s)")

    # ---- values

    @property
    def a(self) -> np.ndarray:
        return np.asarray(self.points[0], float)

    @property
    def b(self) -> np.ndarray:
        return np.asarray(self.points[-1], float)

    @property
    def delta(self) -> np.ndarray:
        return self.b - self.a

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.delta))

    @property
    def horizontal(self) -> float:
        return float(np.hypot(*self.delta[:2]))

    @property
    def vertical(self) -> float:
        """Height difference, second point minus first (signed)."""
        return float(self.delta[2])

    @property
    def slope_deg(self) -> float:
        """Angle above the horizontal, in degrees."""
        if self.length == 0:
            return 0.0
        return math.degrees(math.atan2(self.vertical, self.horizontal))

    def summary(self, unit: str = "m") -> str:
        if self.kind == "point":
            x, y, z = self.a
            return f"({x:.3f}, {y:.3f}, {z:.3f})"
        return f"{fmt_length(self.length, unit)}"

    def details(self, unit: str = "m") -> list[str]:
        """Lines describing the measurement, for the panel."""
        if self.kind == "point":
            x, y, z = self.a
            return [f"X {x:.3f}", f"Y {y:.3f}", f"Z {z:.3f}"]
        return [f"Distance {fmt_length(self.length, unit)}",
                f"Horizontal {fmt_length(self.horizontal, unit)}",
                f"Height difference {self.vertical:+.3f} {unit}",
                f"Slope {self.slope_deg:+.1f}°"]

    def label(self) -> str:
        return self.name or (("Distance " if self.kind == "distance" else "Point ")
                             + self.summary())

    # ---- saving

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                "visible": self.visible, "points": self.points}

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        return cls(kind=d["kind"], points=d["points"], name=d.get("name", ""),
                   visible=d.get("visible", True), id=d.get("id") or uuid.uuid4().hex[:10])


def fmt_length(v: float, unit: str = "m") -> str:
    """3.421 m, 0.152 m: millimetre precision, which is what a scan gives."""
    return f"{v:.3f} {unit}"


CSV_COLUMNS = ["name", "kind", "x1", "y1", "z1", "x2", "y2", "z2", "distance",
               "horizontal", "height_difference", "slope_deg"]


def write_csv(path: str, measurements) -> None:
    """One row per measurement; empty cells where a value does not apply."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for m in measurements:
            a = m.a
            row = [m.label(), m.kind, *(f"{v:.4f}" for v in a)]
            if m.kind == "distance":
                row += [f"{v:.4f}" for v in m.b]
                row += [f"{m.length:.4f}", f"{m.horizontal:.4f}", f"{m.vertical:.4f}",
                        f"{m.slope_deg:.2f}"]
            else:
                row += [""] * 7
            w.writerow(row)
