"""Snapping: measurements made on the scan snap to the fitted model, so a
click near a wall's corner measures from the exact corner, and a click on a
wall measures from the wall's plane instead of one noisy scan point."""

from __future__ import annotations

import numpy as np

from .dxf import BOX_FACES


class Snapper:
    """Corners, edges and faces of the model (and the traced plan lines, at
    their cut height) to snap a picked point to."""

    def __init__(self, elements=(), drawings=()):
        corners, edges, faces = [], [], []
        for e in elements:
            if not e.visible:
                continue
            if e.shape == "cylinder":
                o = np.asarray(e.origin, float)
                w = np.asarray(e.w, float)
                top = o + float(e.size[2]) * w
                corners += [o, top]
                edges.append([o, top])  # the axis
            else:
                c = e.corners()
                corners += list(c)
                edges += [list(x) for x in e.edges()]
                for f in BOX_FACES:
                    p = c[list(f)]
                    n = np.cross(p[1] - p[0], p[3] - p[0])
                    if np.linalg.norm(n) > 0:
                        faces.append((p, n / np.linalg.norm(n)))
        for d in drawings:
            segs = np.asarray(d.segments, float).reshape(-1, 2, 2)
            if not len(segs):
                continue
            pts = d.to_3d(segs.reshape(-1, 2)).reshape(-1, 2, 3)
            corners += list(pts.reshape(-1, 3))
            edges += [list(x) for x in pts]
        self.corners = np.array(corners, float).reshape(-1, 3)
        self.edges = np.array(edges, float).reshape(-1, 2, 3)
        self.faces = faces

    def __bool__(self) -> bool:
        return bool(len(self.corners) or len(self.edges))

    def snap(self, p, radius: float, face_tol: float = 0.05):
        """(snapped point, what it snapped to: "corner", "edge", "face" or None)."""
        p = np.asarray(p, float)
        if len(self.corners):
            d = np.linalg.norm(self.corners - p, axis=1)
            k = int(np.argmin(d))
            if d[k] <= radius:
                return self.corners[k].copy(), "corner"
        if len(self.edges):
            a, b = self.edges[:, 0], self.edges[:, 1]
            ab = b - a
            L2 = np.einsum("ij,ij->i", ab, ab)
            t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(L2, 1e-12), 0, 1)
            q = a + t[:, None] * ab
            d = np.linalg.norm(q - p, axis=1)
            k = int(np.argmin(d))
            if d[k] <= radius * 0.6:
                return q[k], "edge"
        best = None
        for quad, n in self.faces:
            dist = float((p - quad[0]) @ n)
            if abs(dist) > face_tol:
                continue
            q = p - dist * n
            u, v = quad[1] - quad[0], quad[3] - quad[0]
            a = (q - quad[0]) @ u / (u @ u)
            b = (q - quad[0]) @ v / (v @ v)
            if -0.01 <= a <= 1.01 and -0.01 <= b <= 1.01 and (best is None or abs(dist) < best[0]):
                best = (abs(dist), q)
        if best is not None:
            return best[1], "face"
        return p, None
