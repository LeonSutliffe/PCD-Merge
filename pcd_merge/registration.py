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
from typing import Callable

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration

ProgressFn = Callable[[str, float], None]


@dataclass
class AlignSettings:
    voxel_size: float | None = None  # None = choose automatically
    target_points: int = 40_000  # used by the automatic voxel size
    ransac_iterations: int = 200_000
    ransac_confidence: float = 0.999
    ransac_attempts: int = 5  # independent RANSAC runs, best after ICP is kept
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


def auto_voxel_size(clouds_xyz: list[np.ndarray], target_points: int = 40_000) -> float:
    """Pick a voxel size giving roughly ``target_points`` per downsampled cloud."""
    sizes = []
    for xyz in clouds_xyz:
        pc = to_o3d(xyz)
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
    pc = to_o3d(xyz)
    down = pc.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    fpfh = reg.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100)
    )
    fine = pc.voxel_down_sample(voxel * 0.5)
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


def _global_candidates(src: _Prepared, tgt: _Prepared, voxel: float, s: AlignSettings):
    """Several independent RANSAC runs; each result is a candidate start for ICP."""
    dist = voxel * 1.5
    out = []
    for _ in range(max(1, s.ransac_attempts)):
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
        out.append(r.transformation)
    return out


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
                  i: int, j: int) -> PairResult:
    # RANSAC can lock onto a wrong but plausible solution (e.g. rotated on a
    # ground plane). Refine every candidate and keep the one whose structure
    # (walls, poles, objects) agrees best after ICP.
    best_t, best_score = None, -1.0
    seen: list[np.ndarray] = []
    for init in _global_candidates(src, tgt, voxel, s):
        t = _refine(src, tgt, init, voxel)
        if any(np.allclose(t, u, atol=voxel * 0.2) for u in seen):
            continue  # same solution as an earlier candidate
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


def align(clouds_xyz: list[np.ndarray], settings: AlignSettings | None = None,
          progress: ProgressFn = _noop) -> AlignResult:
    """Align clouds into the frame of the first cloud.

    ``clouds_xyz`` are (N, 3) arrays. Returns one 4x4 transform per cloud.
    """
    s = settings or AlignSettings()
    n = len(clouds_xyz)
    if n < 2:
        raise ValueError("need at least two point clouds")
    if s.seed is not None:
        o3d.utility.random.seed(int(s.seed))

    progress("Choosing voxel size", 0.0)
    voxel = s.voxel_size or auto_voxel_size(clouds_xyz, s.target_points)

    prepared = []
    for k, xyz in enumerate(clouds_xyz):
        progress(f"Downsampling and computing features for cloud {k + 1}/{n}", 0.05 + 0.25 * k / n)
        prepared.append(_prepare(xyz, voxel))

    pairs: list[PairResult] = []
    combos = list(itertools.combinations(range(n), 2))
    for k, (i, j) in enumerate(combos):
        progress(f"Registering cloud {i + 1} with cloud {j + 1}", 0.3 + 0.6 * k / len(combos))
        # source i -> target j
        pairs.append(register_pair(prepared[i], prepared[j], voxel, s, i, j))

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
    return AlignResult(transforms, pairs, voxel, connected, cloud_fitness)


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
