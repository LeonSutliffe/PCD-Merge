"""Automatic alignment of two or more LiDAR point clouds.

Pipeline (all on voxel-downsampled copies, the originals are never touched):

1. Voxel downsample, estimate normals, compute FPFH features.
2. Global registration with RANSAC on FPFH feature matches (no initial guess
   needed).
3. Point-to-plane ICP refinement, coarse then fine.
4. For three or more clouds: register every pair, build a pose graph from the
   most reliable pairs and optimise it so all clouds agree with each other.

The scans come from the same sensor, so only rigid transforms are estimated
(no scale).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration

ProgressFn = Callable[[str, float], None]
# A cloud's coordinates, or a function returning them. With functions, only
# one full-resolution cloud is held in memory at a time.
CloudSource = Union[np.ndarray, Callable[[], np.ndarray]]

# The automatic voxel size is estimated on at most this many points per cloud.
_VOXEL_ESTIMATE_POINTS = 2_000_000


@dataclass
class AlignSettings:
    voxel_size: float | None = None  # None = choose automatically
    target_points: int = 40_000  # used by the automatic voxel size
    ransac_iterations: int = 200_000
    ransac_confidence: float = 0.999
    ransac_attempts: int = 5  # independent RANSAC runs, best after ICP is kept
    # Stop early once this many RANSAC runs have all led to the same answer.
    ransac_agree: int = 3
    min_fitness: float = 0.15  # pairs below this are not trusted as graph edges
    seed: int | None = None


@dataclass
class PairResult:
    source: int
    target: int
    transformation: np.ndarray  # maps source into the target frame
    fitness: float  # fraction of source points with a close match (~overlap)
    rmse: float
    information: np.ndarray = field(repr=False, default=None)


@dataclass
class AlignResult:
    # transforms[i] maps cloud i into the frame of cloud 0 (the reference).
    transforms: list[np.ndarray]
    pairs: list[PairResult]
    voxel_size: float
    connected: list[bool]
    # per-cloud final fitness against all the other aligned clouds
    cloud_fitness: list[float]
    # each cloud downsampled at voxel_size * 0.5, in its own frame; lets the
    # quality report skip another pass over the full-resolution points
    fine: list = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return all(self.connected)

    def warnings(self, low_fitness: float = 0.3) -> list[str]:
        msgs = []
        for i, (c, f) in enumerate(zip(self.connected, self.cloud_fitness)):
            if not c:
                msgs.append(f"Cloud {i + 1} could not be matched to any other cloud.")
            elif f < low_fitness:
                msgs.append(
                    f"Cloud {i + 1}: only {f:.0%} of its points overlap the others; "
                    "check the result carefully."
                )
        return msgs


def _noop(_msg: str, _frac: float) -> None:
    pass


def to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    return pc


_DOWN_CHUNK = 2_000_000


def voxel_down(xyz: np.ndarray, voxel: float, chunk: int = _DOWN_CHUNK) -> o3d.geometry.PointCloud:
    """Voxel-downsample a cloud of any size in chunks of ``chunk`` points.

    Open3D holds Python's global lock while it works, so one call on a
    50-million-point cloud would freeze the app's window for several seconds.
    Chunks keep each pause short and avoid a full-size Open3D copy.

    Every chunk gets a sentinel point at the same far corner, which pins all
    chunks to one voxel grid; the sentinel sits alone in its voxel and is
    removed. Voxels split across chunks are joined by a final pass, so the
    output has exactly one point per occupied voxel, like Open3D's own.
    """
    n = len(xyz)
    if n <= chunk:
        return to_o3d(xyz).voxel_down_sample(voxel)
    lo = xyz.min(axis=0)
    buf = np.empty((chunk + 1, 3), dtype=np.float64)
    buf[0] = lo - 2 * voxel

    def down(m: int) -> np.ndarray:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(buf[: m + 1]))
        out = np.asarray(pc.voxel_down_sample(voxel).points)
        return out[(out >= lo - voxel).all(axis=1)]  # drop the sentinel

    parts = []
    for start in range(0, n, chunk):
        part = xyz[start : start + chunk]
        buf[1 : len(part) + 1] = part
        parts.append(down(len(part)))
    joined = np.concatenate(parts)
    if len(joined) + 1 > len(buf):
        buf = np.empty((len(joined) + 1, 3), dtype=np.float64)
        buf[0] = lo - 2 * voxel
    buf[1 : len(joined) + 1] = joined
    return to_o3d(down(len(joined)))


def _estimate_sample(xyz: np.ndarray) -> np.ndarray:
    """Random subset used to choose the voxel size quickly on huge clouds."""
    if len(xyz) <= _VOXEL_ESTIMATE_POINTS:
        return xyz
    idx = np.random.default_rng(0).integers(0, len(xyz), _VOXEL_ESTIMATE_POINTS)
    return xyz[np.sort(idx)]


def auto_voxel_size(clouds_xyz: list[np.ndarray], target_points: int = 40_000) -> float:
    """Pick a voxel size giving roughly ``target_points`` per downsampled cloud.

    Very large clouds are estimated from a random subset of their points,
    which is many times faster and gives practically the same voxel size.
    """
    sizes = []
    for xyz in clouds_xyz:
        pc = to_o3d(_estimate_sample(xyz))
        ext = pc.get_axis_aligned_bounding_box().get_extent()
        diag = float(np.linalg.norm(ext))
        if diag <= 0:
            continue
        v = diag / 400.0
        for _ in range(20):
            n = len(pc.voxel_down_sample(v).points)
            if n <= target_points:
                break
            v *= 1.4
        sizes.append(v)
    if not sizes:
        raise ValueError("point clouds are empty or degenerate")
    return float(np.median(sizes))


@dataclass
class _Prepared:
    down: o3d.geometry.PointCloud  # voxel-downsampled, with normals
    fine: o3d.geometry.PointCloud  # denser copy for the last ICP pass
    # Points that are not on the dominant plane (usually the ground), with
    # their features. Ground points all look alike, so they only add false
    # feature matches and make wrong alignments score well.
    struct: o3d.geometry.PointCloud
    struct_fpfh: o3d.pipelines.registration.Feature


def _prepare(xyz: np.ndarray, voxel: float) -> _Prepared:
    # One pass over the full-resolution points; the coarser grid is built
    # from the (much smaller) fine one.
    fine = voxel_down(xyz, voxel * 0.5)
    down = fine.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    fpfh = reg.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )
    fine.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 1.5, max_nn=30))

    keep = np.arange(len(down.points))
    if len(down.points) > 1000:
        _plane, inliers = down.segment_plane(
            distance_threshold=voxel * 1.5, ransac_n=3, num_iterations=2000
        )
        mask = np.ones(len(down.points), bool)
        mask[np.asarray(inliers, dtype=int)] = False
        # Only drop the plane if it is a big part of the scan and enough
        # structure remains to match on.
        if len(inliers) > 0.2 * len(down.points) and mask.sum() >= 500:
            keep = np.flatnonzero(mask)
    struct = down.select_by_index(keep)
    struct_fpfh = reg.Feature()
    struct_fpfh.data = np.asarray(fpfh.data)[:, keep]
    return _Prepared(down, fine, struct, struct_fpfh)


def _ransac(src: _Prepared, tgt: _Prepared, voxel: float, s: AlignSettings) -> np.ndarray:
    """One RANSAC run on feature matches; the result is a candidate start for ICP."""
    dist = voxel * 1.5
    r = reg.registration_ransac_based_on_feature_matching(
        src.struct,
        tgt.struct,
        src.struct_fpfh,
        tgt.struct_fpfh,
        True,  # mutual filter
        dist,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        reg.RANSACConvergenceCriteria(s.ransac_iterations, s.ransac_confidence),
    )
    return r.transformation


def _refine(src: _Prepared, tgt: _Prepared, init: np.ndarray, voxel: float) -> np.ndarray:
    est = reg.TransformationEstimationPointToPlane()
    t = init
    for dist, a, b, iters in (
        (voxel * 2.0, src.down, tgt.down, 50),
        (voxel * 0.8, src.down, tgt.down, 50),
        (voxel * 0.5, src.fine, tgt.fine, 30),
    ):
        r = reg.registration_icp(a, b, dist, t, est, reg.ICPConvergenceCriteria(max_iteration=iters))
        t = r.transformation
    return t


def _score(src: _Prepared, tgt: _Prepared, t: np.ndarray, voxel: float) -> float:
    """Candidate score: tight-threshold overlap of the non-ground structure."""
    ev = reg.evaluate_registration(src.struct, tgt.struct, voxel * 0.75, t)
    return ev.fitness


def register_pair(src: _Prepared, tgt: _Prepared, voxel: float, s: AlignSettings,
                  i: int, j: int, step: Callable[[float], None] | None = None) -> PairResult:
    # RANSAC can lock onto a wrong but plausible solution (e.g. rotated on a
    # ground plane). Refine every candidate and keep the one whose structure
    # (walls, poles, objects) agrees best after ICP.
    best_t, best_score = None, -1.0
    seen: list[np.ndarray] = []
    attempts = max(1, s.ransac_attempts)
    for k in range(attempts):
        t = _refine(src, tgt, _ransac(src, tgt, voxel, s), voxel)
        if step:
            step((k + 1) / attempts)
        if any(np.allclose(t, u, atol=voxel * 0.2) for u in seen):
            # Same solution as an earlier candidate. If every run so far has
            # ended at one answer, further runs will not change the result.
            if len(seen) == 1 and s.ransac_agree and k + 1 >= s.ransac_agree:
                break
            continue
        seen.append(t)
        score = _score(src, tgt, t, voxel)
        if score > best_score:
            best_t, best_score = t, score
    t = best_t
    ev = reg.evaluate_registration(src.down, tgt.down, voxel * 1.5, t)
    info = reg.get_information_matrix_from_point_clouds(src.down, tgt.down, voxel * 1.5, t)
    # Overlap is asymmetric; take the better direction as the pair's score.
    rev = reg.evaluate_registration(tgt.down, src.down, voxel * 1.5, np.linalg.inv(t))
    return PairResult(i, j, t, max(ev.fitness, rev.fitness), ev.inlier_rmse, info)


def _get(src: CloudSource) -> np.ndarray:
    return src() if callable(src) else src


def align(clouds_xyz: Sequence[CloudSource], settings: AlignSettings | None = None,
          progress: ProgressFn = _noop) -> AlignResult:
    """Align clouds into the frame of the first cloud.

    ``clouds_xyz`` are (N, 3) arrays, or functions returning them (so that
    huge clouds can be converted one at a time). Returns one 4x4 transform
    per cloud.
    """
    s = settings or AlignSettings()
    n = len(clouds_xyz)
    if n < 2:
        raise ValueError("need at least two point clouds")
    if s.seed is not None:
        o3d.utility.random.seed(int(s.seed))

    voxel = s.voxel_size
    if not voxel:
        samples = []
        for k, src in enumerate(clouds_xyz):
            progress(f"Choosing voxel size ({k + 1}/{n})", 0.06 * k / n)
            samples.append(_estimate_sample(_get(src)))
        progress("Choosing voxel size", 0.06)
        voxel = auto_voxel_size(samples, s.target_points)
        del samples

    prepared = []
    for k, src in enumerate(clouds_xyz):
        progress(f"Downsampling and computing features for cloud {k + 1}/{n}",
                 0.08 + 0.22 * k / n)
        prepared.append(_prepare(_get(src), voxel))

    pairs: list[PairResult] = []
    combos = list(itertools.combinations(range(n), 2))
    span = 0.6 / len(combos)
    for k, (i, j) in enumerate(combos):
        msg = f"Registering cloud {i + 1} with cloud {j + 1}"
        base = 0.3 + span * k
        progress(msg, base)
        # source i -> target j
        pairs.append(register_pair(prepared[i], prepared[j], voxel, s, i, j,
                                   step=lambda f, m=msg, b=base: progress(m, b + span * f)))

    progress("Optimising poses", 0.92)
    if n == 2:
        p = pairs[0]
        connected = [True, p.fitness >= s.min_fitness]
        # cloud 1 into frame of cloud 0: inverse of (0 -> 1)
        transforms = [np.eye(4), np.linalg.inv(p.transformation)]
    else:
        transforms, connected = _pose_graph(n, pairs, prepared, voxel, s)

    cloud_fitness = _cloud_fitness(prepared, transforms, voxel)
    progress("Done", 1.0)
    return AlignResult(transforms, pairs, voxel, connected, cloud_fitness,
                       fine=[p.fine for p in prepared])


def _pose_graph(n, pairs, prepared, voxel, s):
    """Spanning tree from the best pairs for initial poses, then optimise."""
    good = sorted((p for p in pairs if p.fitness >= s.min_fitness), key=lambda p: -p.fitness)
    # Kruskal maximum spanning tree on fitness
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = []
    for p in good:
        a, b = find(p.source), find(p.target)
        if a != b:
            parent[a] = b
            tree.append(p)

    # Node pose = world_from_node. Edge T maps source into target:
    # T ~= inv(pose_target) @ pose_source.
    poses: dict[int, np.ndarray] = {0: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for p in tree:
            if p.target in poses and p.source not in poses:
                poses[p.source] = poses[p.target] @ p.transformation
                changed = True
            elif p.source in poses and p.target not in poses:
                poses[p.target] = poses[p.source] @ np.linalg.inv(p.transformation)
                changed = True
    connected = [i in poses for i in range(n)]

    graph = reg.PoseGraph()
    for i in range(n):
        graph.nodes.append(reg.PoseGraphNode(poses.get(i, np.eye(4))))
    tree_ids = {id(p) for p in tree}
    for p in good:
        if not (connected[p.source] and connected[p.target]):
            continue
        graph.edges.append(
            reg.PoseGraphEdge(p.source, p.target, p.transformation, p.information,
                              uncertain=id(p) not in tree_ids)
        )
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=voxel * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(
            graph,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    ref_inv = np.linalg.inv(np.asarray(graph.nodes[0].pose))
    transforms = [ref_inv @ np.asarray(graph.nodes[i].pose) for i in range(n)]
    for i in range(n):
        if not connected[i]:
            transforms[i] = np.eye(4)
    return transforms, connected


def _cloud_fitness(prepared, transforms, voxel) -> list[float]:
    """For each cloud, the fraction of its points close to any other aligned cloud."""
    n = len(prepared)
    moved = []
    for p, t in zip(prepared, transforms):
        c = o3d.geometry.PointCloud(p.down)
        c.transform(t)
        moved.append(c)
    out = []
    for i in range(n):
        others = o3d.geometry.PointCloud()
        for j in range(n):
            if j != i:
                others += moved[j]
        ev = reg.evaluate_registration(moved[i], others, voxel * 1.5, np.eye(4))
        out.append(float(ev.fitness))
    return out
