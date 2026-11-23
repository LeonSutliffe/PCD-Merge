"""Synthetic LiDAR-like scans of a small outdoor scene, for tests, the self-test and demos."""

from __future__ import annotations

import numpy as np


def _rot(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_transform(yaw, tilt=0.0, t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = _rot([0, 0, 1], yaw) @ _rot([1, 0.3, 0], tilt)
    T[:3, 3] = t
    return T


def _box(rng, centre, size, n):
    c = np.asarray(centre, float)
    s = np.asarray(size, float)
    areas = np.array([s[1] * s[2], s[1] * s[2], s[0] * s[2], s[0] * s[2], s[0] * s[1]])
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for face, k in enumerate(counts):
        u = rng.uniform(-0.5, 0.5, (k, 3)) * s
        axis, sign = [(0, -1), (0, 1), (1, -1), (1, 1), (2, 1)][face]
        u[:, axis] = sign * s[axis] / 2
        pts.append(u)
    p = np.concatenate(pts) + c
    p[:, 2] += s[2] / 2  # sit on the ground
    return p


def _cylinder(rng, centre, radius, height, n):
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, height, n)
    return np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), z], 1)


def make_scene(seed=0, density=60.0):
    """Ground, two walls, boxes and poles in a ~40 m square. Not symmetric."""
    rng = np.random.default_rng(seed)
    parts = []
    # ground with gentle undulation
    n = int(40 * 40 * density)
    g = rng.uniform(-20, 20, (n, 2))
    parts.append(np.c_[g, 0.15 * np.sin(g[:, 0] / 5.0) * np.cos(g[:, 1] / 7.0)])
    # L-shaped walls
    n = int(40 * 4 * density)
    parts.append(np.c_[rng.uniform(-20, 20, n), np.full(n, 20.0), rng.uniform(0, 4, n)])
    n = int(25 * 3 * density)
    parts.append(np.c_[np.full(n, -20.0), rng.uniform(-5, 20, n), rng.uniform(0, 3, n)])
    # boxes (buildings / vehicles)
    for centre, size in [((5, 5), (4, 2, 1.6)), ((-8, 10), (6, 6, 3)), ((12, -6), (3, 8, 2.5)),
                         ((-4, -10), (2, 2, 1)), ((0, 14), (10, 2, 2)), ((-12, -2), (2.5, 5, 1.8)),
                         ((9, 12), (3, 3, 4))]:
        area = 2 * (size[0] + size[1]) * size[2] + size[0] * size[1]
        parts.append(_box(rng, (*centre, 0.0), size, int(area * density)))
    # poles
    for centre, r, h in [((3, -3), 0.15, 5), ((-6, 4), 0.2, 6), ((15, 5), 0.3, 3),
                         ((-15, -12), 0.15, 4), ((7, -14), 0.25, 5)]:
        parts.append(_cylinder(rng, centre, r, h, int(2 * np.pi * r * h * density * 2)))
    return np.concatenate(parts)


def make_scan(scene, sensor_xy, radius, transform, seed=0, noise=0.01):
    """Points within ``radius`` of the sensor, with noise and intensity,
    expressed in a scan frame: ``scan = inv(transform) @ world``.

    So ``transform`` is the ground-truth pose mapping scan -> world.
    """
    rng = np.random.default_rng(seed)
    d = np.linalg.norm(scene[:, :2] - np.asarray(sensor_xy), axis=1)
    pts = scene[d < radius]
    # thin out with range like a real sensor
    keep = rng.uniform(size=len(pts)) < np.clip(6.0 / (d[d < radius] + 1e-3), 0.15, 1.0)
    pts = pts[keep] + rng.normal(0, noise, (keep.sum(), 3))
    inv = np.linalg.inv(transform)
    local = pts @ inv[:3, :3].T + inv[:3, 3]
    intensity = (50 + 40 * np.sin(pts[:, 0]) + pts[:, 2] * 10).astype(np.float32)
    data = np.zeros(len(local), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("intensity", "<f4"), ("ring", "<u2")])
    data["x"], data["y"], data["z"] = local[:, 0], local[:, 1], local[:, 2]
    data["intensity"] = intensity
    data["ring"] = (np.arange(len(local)) % 32).astype(np.uint16)
    return data


def transform_error(estimated, truth):
    """Rotation error (degrees) and translation error between two 4x4 poses."""
    d = np.linalg.inv(truth) @ estimated
    cos = np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos))), float(np.linalg.norm(d[:3, 3]))
