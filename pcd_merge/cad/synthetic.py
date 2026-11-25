"""A small synthetic two-storey building, scanned from inside each room, for
the Model tools' tests and demos.

Outside 12 x 8 m, outer walls 0.30 m thick, one inner wall 0.12 m thick at
x = 5.00 .. 5.12 with a door 0.9 m wide. Floors at 0 and 3.0 m, ceilings at
2.7 and 5.7 m (slab 0.30 m). A round column (r 0.15 m) stands at (9, 4) on
the ground floor and a pipe (r 0.05 m) runs along X at y = 6, z = 2.4.
"""

from __future__ import annotations

import numpy as np

OUTER = (0.0, 12.0, 0.0, 8.0)
WALL = 0.30
INNER = (5.0, 5.12)
DOOR = (3.0, 3.9, 2.1)  # y from, y to, height
LEVELS = [(0.0, 2.7), (3.0, 5.7)]  # floor, ceiling
COLUMN = ((9.0, 4.0), 0.15)
PIPE = (6.0, 2.4, 0.05)  # y, z, radius (from x = 0.3 to 5.0)


def _rect(rng, origin, u, v, density):
    """Points on the rectangle origin + [0,1]u + [0,1]v."""
    area = np.linalg.norm(np.cross(u, v))
    n = rng.poisson(area * density)
    a, b = rng.random(n), rng.random(n)
    return np.asarray(origin, float) + a[:, None] * np.asarray(u, float) \
        + b[:, None] * np.asarray(v, float)


def _cylinder(rng, base, axis, radius, length, density, sensor=None, arc=(0, 2 * np.pi)):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    e1 = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    n = rng.poisson((arc[1] - arc[0]) * radius * length * density)
    th = rng.uniform(arc[0], arc[1], n)
    t = rng.uniform(0, length, n)
    out_dir = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    pts = np.asarray(base, float) + t[:, None] * axis + radius * out_dir
    if sensor is not None:  # only the side the scanner sees
        pts = pts[np.einsum("ij,ij->i", out_dir, np.asarray(sensor, float) - pts) > 0]
    return pts


# shelving units in the ground-floor room on the left, (x0, x1, y0, y1, height):
# two against the wall at y = 0.3 and a free-standing one in the middle
SHELVES = [(0.6, 2.6, 0.3, 0.8, 2.0), (2.8, 4.6, 0.3, 0.8, 2.0), (1.0, 4.0, 4.6, 5.2, 1.4)]


def _shelving(rng, unit, density):
    """A shelving unit: five boards, two ends and a back, as points."""
    x0, x1, y0, y1, h = unit
    w, d = x1 - x0, y1 - y0
    parts = []
    for z in np.linspace(0.05, h, 5):
        parts.append(_rect(rng, (x0, y0, z), (w, 0, 0), (0, d, 0), density))  # top of a board
    for x in (x0, x1):
        parts.append(_rect(rng, (x, y0, 0), (0, d, 0), (0, 0, h), density))
    parts.append(_rect(rng, (x0, y1, 0), (w, 0, 0), (0, 0, h), density))  # back, facing the room
    return np.concatenate(parts)


def _hidden(pts, sensor, boxes):
    """Points the scanner cannot see because a box is in the way."""
    sensor = np.asarray(sensor, float)
    d = pts - sensor
    hid = np.zeros(len(pts), dtype=bool)
    for x0, x1, y0, y1, h in boxes:
        lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, h])
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (lo - sensor) / d, (hi - sensor) / d
        tmin = np.nanmax(np.minimum(t0, t1), axis=1)
        tmax = np.nanmin(np.maximum(t0, t1), axis=1)
        hid |= (tmin < tmax) & (tmax > 0) & (tmin < 0.98)  # crosses the box before the point
    return hid


def make_building(density: float = 400.0, noise: float = 0.003, seed: int = 0,
                  shelving: bool = False):
    """Scans of the building: a list of ``(points (N, 3), scanner position)``,
    one per room and storey. With ``shelving`` the ground-floor room on the
    left is full of shelves, which hide parts of its walls and floor."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = OUTER
    ix0, iy0, ix1, iy1 = x0 + WALL, y0 + WALL, x1 - WALL, y1 - WALL
    a, b = INNER
    rooms = [(ix0, a), (b, ix1)]
    scans = []
    for lvl, (fz, cz) in enumerate(LEVELS):
        h = cz - fz
        for r, (rx0, rx1) in enumerate(rooms):
            w = rx1 - rx0
            d = iy1 - iy0
            parts = [
                _rect(rng, (rx0, iy0, fz), (w, 0, 0), (0, d, 0), density),  # floor
                _rect(rng, (rx0, iy0, cz), (w, 0, 0), (0, d, 0), density),  # ceiling
                _rect(rng, (rx0, iy0, fz), (w, 0, 0), (0, 0, h), density),  # y = iy0 wall
                _rect(rng, (rx0, iy1, fz), (w, 0, 0), (0, 0, h), density),  # y = iy1 wall
            ]
            outer_x = rx0 if r == 0 else rx1
            parts.append(_rect(rng, (outer_x, iy0, fz), (0, d, 0), (0, 0, h), density))
            inner_x = rx1 if r == 0 else rx0
            # the inner wall, with the door
            dy0, dy1, dh = DOOR
            parts.append(_rect(rng, (inner_x, iy0, fz), (0, dy0 - iy0, 0), (0, 0, h), density))
            parts.append(_rect(rng, (inner_x, dy1, fz), (0, iy1 - dy1, 0), (0, 0, h), density))
            parts.append(_rect(rng, (inner_x, dy0, fz + dh), (0, dy1 - dy0, 0), (0, 0, h - dh),
                               density))
            # door reveals (the wall's thickness, seen through the door)
            parts.append(_rect(rng, (a, dy0, fz), (b - a, 0, 0), (0, 0, dh), density / 2))
            parts.append(_rect(rng, (a, dy1, fz), (b - a, 0, 0), (0, 0, dh), density / 2))
            sensor = np.array([(rx0 + rx1) / 2 + 0.3, (iy0 + iy1) / 2 - 0.7, fz + 1.5])
            if lvl == 0 and r == 1:  # not inside the column
                sensor[:2] = (7.0, 2.0)
            if lvl == 0 and r == 1:
                (cx, cy), cr = COLUMN
                parts.append(_cylinder(rng, (cx, cy, fz), (0, 0, 1), cr, h, density, sensor))
            if lvl == 0 and r == 0:
                py, pz, pr = PIPE
                parts.append(_cylinder(rng, (rx0, py, pz), (1, 0, 0), pr, w, density, sensor))
            pts = np.concatenate(parts)
            if shelving and lvl == 0 and r == 0:
                pts = pts[~_hidden(pts, sensor, SHELVES)]
                for u in SHELVES:
                    sp = _shelving(rng, u, density)
                    pts = np.concatenate([pts, sp[~_hidden(sp, sensor, [u])]])
            pts += rng.normal(0, noise, pts.shape)
            scans.append((pts, sensor))
    return scans
