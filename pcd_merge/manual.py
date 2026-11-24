"""Manual alignment: the fallback when Align cannot find a match.

The user clicks the same spot (a corner, a pole, a door frame) on two scans,
three times or more. :func:`rigid_from_pairs` finds the rotation and shift
that bring the picked points of the moving scan onto those of the fixed one,
and :func:`refine` can then tighten it with ICP on the scans themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_PAIRS = 3


class ManualAlignError(ValueError):
    pass


@dataclass
class PairFit:
    transform: np.ndarray  # 4x4, maps the moving points onto the fixed ones
    residuals: np.ndarray  # per pair: distance left after the fit

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2))) if len(self.residuals) else 0.0

    @property
    def worst(self) -> int:
        """Index of the pair that fits worst (probably a mis-click)."""
        return int(np.argmax(self.residuals)) if len(self.residuals) else -1


def rigid_from_pairs(moving, fixed, level: bool = False) -> PairFit:
    """The rigid transform (no scaling) taking ``moving[k]`` to ``fixed[k]``,
    least squares over all pairs (Kabsch).

    ``level``: only turn about the vertical axis (for scans that are already
    level, as most tripod scans are). Two pairs are then enough.
    """
    a = np.asarray(moving, float).reshape(-1, 3)
    b = np.asarray(fixed, float).reshape(-1, 3)
    if len(a) != len(b):
        raise ManualAlignError("every point needs its partner")
    need = 2 if level else MIN_PAIRS
    if len(a) < need:
        raise ManualAlignError(f"pick at least {need} pairs of points")
    ca, cb = a.mean(0), b.mean(0)
    da, db = a - ca, b - cb
    if level:
        h = da[:, :2].T @ db[:, :2]
        spread = np.linalg.norm(da[:, :2], axis=1).max()
        if spread < 1e-6:
            raise ManualAlignError("the points are all in one spot: spread them out")
        ang = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = np.cos(ang), np.sin(ang)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        # three points in a line leave the turn about that line open
        sv = np.linalg.svd(da, compute_uv=False)
        if sv[1] < 1e-3 * max(sv[0], 1e-9) or sv[0] < 1e-6:
            raise ManualAlignError("the points are (nearly) in a straight line: pick one off "
                                   "to the side")
        u, _, vt = np.linalg.svd(da.T @ db)
        d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = cb - r @ ca
    res = np.linalg.norm(a @ r.T + t[:3, 3] - b, axis=1)
    return PairFit(t, res)


def refine(moving_xyz, fixed_xyz, init, voxel: float):
    """Tighten ``init`` (moving -> fixed) with point-to-plane ICP, from a
    search distance a little larger than the picking error down to the
    voxel size. Returns (transform, fitness): fitness is the share of moving
    points with a fixed point within ``voxel``."""
    import open3d as o3d

    reg = o3d.pipelines.registration

    def cloud(xyz, v):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, float)))
        pc = pc.voxel_down_sample(v)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 3, max_nn=30))
        return pc

    src, tgt = cloud(moving_xyz, voxel), cloud(fixed_xyz, voxel)
    t = np.asarray(init, float)
    est = reg.TransformationEstimationPointToPlane()
    for dist in (voxel * 6, voxel * 3, voxel * 1.5):
        r = reg.registration_icp(src, tgt, dist, t, est,
                                 reg.ICPConvergenceCriteria(max_iteration=40))
        t = r.transformation
    ev = reg.evaluate_registration(src, tgt, voxel, t)
    return np.asarray(t), float(ev.fitness)


def overlap(moving_xyz, fixed_xyz, t, voxel: float) -> float:
    """Share of the moving points (after ``t``) with a fixed point within ``voxel``."""
    import open3d as o3d

    def cloud(xyz):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, float)))
        return pc.voxel_down_sample(voxel)

    ev = o3d.pipelines.registration.evaluate_registration(
        cloud(moving_xyz), cloud(fixed_xyz), voxel, np.asarray(t, float))
    return float(ev.fitness)


def manual_result(previous, n: int, transforms, moved: int, fixed: int, fitness: float,
                  voxel: float):
    """An :class:`~pcd_merge.registration.AlignResult` with scan ``moved``
    placed by hand onto scan ``fixed``. Keeps what an earlier Align found for
    the other scans."""
    from .registration import AlignResult

    if previous is not None:
        connected = list(previous.connected)
        cloud_fit = list(previous.cloud_fitness)
        ref, anchored, pairs = previous.reference, previous.anchored, list(previous.pairs)
        voxel = previous.voxel_size
    else:
        connected, cloud_fit = [False] * n, [0.0] * n
        ref, anchored, pairs = fixed, False, []
        connected[fixed] = True
        cloud_fit[fixed] = 1.0
    connected[moved] = True
    cloud_fit[moved] = float(fitness)
    return AlignResult(transforms=[np.asarray(t, float) for t in transforms], pairs=pairs,
                       voxel_size=float(voxel), connected=connected,
                       cloud_fitness=cloud_fit, reference=ref, anchored=anchored)
