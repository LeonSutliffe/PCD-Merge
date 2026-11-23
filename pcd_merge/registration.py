"""Automatic alignment of two or more LiDAR point clouds.

Pipeline (all on voxel-downsampled copies, the originals are never touched):

1. Voxel downsample, estimate normals, compute FPFH features.
2. Global registration with RANSAC on FPFH feature matches (no initial guess
   needed).
3. Point-to-plane ICP refinement, coarse then fine.
4. For three or more clouds: register pairs, build a pose graph from the
   most reliable pairs and optimise it so all clouds agree with each other.
   Up to ``exhaustive_max`` clouds every pair is registered. Larger batches
   (a whole site) register neighbours in file order first, then only try
   other pairs until every scan is connected, and finally add loop closures
   between scans that the first pass already places next to each other
   (ICP only, no feature search), so the work grows roughly linearly.
5. Incremental mode (``anchor=True``): cloud 0 is an existing merged cloud
   that must not move; each new scan is matched against it first.

The scans come from the same sensor, so only rigid transforms are estimated
(no scale).
"""

from __future__ import annotations

import dataclasses
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
    # ...and neither are pairs whose non-ground structure agrees less than
    # this. Wrong matches between far-apart scans often overlap 50%+ of their
    # ground, but their walls and objects do not line up.
    min_structure: float = 0.25
    seed: int | None = None
    # Above this many clouds, not every pair is feature-matched (see module doc).
    exhaustive_max: int = 6
    # Cloud 0 is an existing merged cloud: keep it fixed as the reference, pick
    # the voxel size from the new scans only, and match new scans to it first.
    anchor: bool = False


@dataclass
class PairResult:
    source: int
    target: int
    transformation: np.ndarray  # maps source into the target frame
    fitness: float  # fraction of source points with a close match (~overlap)
    rmse: float
    information: np.ndarray = field(repr=False, default=None)
    # Tight overlap of the non-ground structure (walls, poles, objects). Wrong
    # matches can still overlap a lot of ground, so this is the stricter test.
    structure: float = 0.0

    def trusted(self, s: AlignSettings) -> bool:
        return self.fitness >= s.min_fitness and self.structure >= s.min_structure


@dataclass
class AlignResult:
    # transforms[i] maps cloud i into the frame of cloud 0 (the reference).
    transforms: list[np.ndarray]
    pairs: list[PairResult]
    voxel_size: float
    connected: list[bool]
    # per-cloud final fitness against all the other aligned clouds
    cloud_fitness: list[float]
    # Index of the cloud whose frame the result is in. Normally 0; another
    # cloud only if cloud 0 matched nothing while the others did.
    reference: int = 0
    # Cloud 0 is an existing merged cloud (incremental mode).
    anchored: bool = False

    @property
    def ok(self) -> bool:
        return all(self.connected)

    def warnings(self, low_fitness: float = 0.3) -> list[str]:
        msgs = []
        if self.reference != 0:
            msgs.append(f"Cloud 1 matched nothing, so cloud {self.reference + 1} is used as "
                        "the reference instead.")
        for i, (c, f) in enumerate(zip(self.connected, self.cloud_fitness)):
            if not c:
                msgs.append(f"Cloud {i + 1} could not be matched to any other cloud.")
            elif f < low_fitness and not (i == 0 and self.anchored):
                # (an existing merged site only partly overlaps the new scans)
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
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
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
    return _pair_result(src, tgt, best_t, voxel, i, j)


def _pair_result(src: _Prepared, tgt: _Prepared, t: np.ndarray, voxel: float,
                 i: int, j: int) -> PairResult:
    ev = reg.evaluate_registration(src.down, tgt.down, voxel * 1.5, t)
    info = reg.get_information_matrix_from_point_clouds(src.down, tgt.down, voxel * 1.5, t)
    # Overlap is asymmetric; take the better direction as the pair's score.
    rev = reg.evaluate_registration(tgt.down, src.down, voxel * 1.5, np.linalg.inv(t))
    sev = _score(src, tgt, t, voxel)
    srev = _score(tgt, src, np.linalg.inv(t), voxel)
    return PairResult(i, j, t, max(ev.fitness, rev.fitness), ev.inlier_rmse, info,
                      max(sev, srev))


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
    voxel = s.voxel_size
    if not voxel:
        # In incremental mode the existing merged cloud covers a whole site;
        # the new scans are what the sensor produces, so size from those.
        voxel = auto_voxel_size(clouds_xyz[1:] if s.anchor else clouds_xyz, s.target_points)

    prepared = []
    for k, xyz in enumerate(clouds_xyz):
        progress(f"Downsampling and computing features for cloud {k + 1}/{n}", 0.05 + 0.25 * k / n)
        prepared.append(_prepare(xyz, voxel))

    if n <= max(2, s.exhaustive_max):
        pairs: list[PairResult] = []
        combos = list(itertools.combinations(range(n), 2))
        for k, (i, j) in enumerate(combos):
            progress(f"Registering cloud {i + 1} with cloud {j + 1}", 0.3 + 0.6 * k / len(combos))
            # source i -> target j
            pairs.append(register_pair(prepared[i], prepared[j], voxel, s, i, j))
    else:
        pairs = _batch_pairs(prepared, voxel, s, progress)

    progress("Optimising poses", 0.92)
    reference = 0
    if n == 2:
        p = pairs[0]
        connected = [True, p.trusted(s)]
        # cloud 1 into frame of cloud 0: inverse of (0 -> 1)
        transforms = [np.eye(4), np.linalg.inv(p.transformation)]
        if not connected[1]:
            transforms[1] = np.eye(4)
    else:
        transforms, connected, reference = _pose_graph(n, pairs, prepared, voxel, s)

    cloud_fitness = _cloud_fitness(prepared, transforms, voxel)
    progress("Done", 1.0)
    return AlignResult(transforms, pairs, voxel, connected, cloud_fitness, reference, s.anchor)


class _Components:
    """Union-find over cloud indices."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        self.parent[a] = b
        return True

    def count(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})


def _batch_pairs(prepared: list[_Prepared], voxel: float, s: AlignSettings,
                 progress: ProgressFn) -> list[PairResult]:
    """Register enough pairs to connect a large set of scans, without trying all of them.

    1. Neighbours: each scan with the next one in file order (scanners number
       their files in capture order, so these usually overlap). In incremental
       mode, each new scan with the existing merged cloud instead.
    2. Bridging: while the scans fall into separate groups, try pairs across
       groups, closest in file order first, until everything is connected or
       every cross-group pair has been tried (e.g. a scan from another site).
    3. Loop closures: with rough poses from step 1-2, every further pair whose
       scans now overlap is refined with ICP from that pose. These extra
       edges let the pose graph spread out small errors along long chains.
    """
    n = len(prepared)
    pairs: list[PairResult] = []
    tried: set[tuple[int, int]] = set()
    comps = _Components(n)
    # Rough number of feature-matched pairs, for the progress bar only.
    budget = max(n, 1)

    def attempt(i: int, j: int, what: str):
        tried.add((min(i, j), max(i, j)))
        frac = min(len(pairs) / budget, 1.0)
        progress(f"{what}: cloud {i + 1} with cloud {j + 1}", 0.3 + 0.45 * frac)
        p = register_pair(prepared[i], prepared[j], voxel, s, i, j)
        pairs.append(p)
        if p.trusted(s):
            comps.union(i, j)

    first = [(k, 0) for k in range(1, n)] if s.anchor else [(k, k + 1) for k in range(n - 1)]
    for i, j in first:
        attempt(i, j, "Matching neighbours")

    while comps.count() > 1:
        if s.anchor:
            # Only the merged cloud's group matters; join scans to it.
            root = comps.find(0)
            cand = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if (comps.find(i) == root) != (comps.find(j) == root)]
        else:
            cand = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if comps.find(i) != comps.find(j)]
        cand = [c for c in cand if c not in tried]
        if not cand:
            break
        budget = len(pairs) + min(len(cand), n)
        i, j = min(cand, key=lambda c: (c[1] - c[0], c[0]))
        attempt(j, i, "Connecting groups")

    # Second chance for scans still on their own: RANSAC sometimes misses a
    # real overlap (repetitive walls, little structure). Retry their most
    # promising pairs with coarser features (they describe larger
    # neighbourhoods, which helped most in testing) and a wider search, then
    # refine at the normal resolution.
    if comps.count() > 1:
        retry = dataclasses.replace(s, ransac_attempts=s.ransac_attempts * 2)
        coarse_voxel = voxel * 1.5
        coarse: dict[int, _Prepared] = {}

        def coarse_of(k):
            if k not in coarse:
                coarse[k] = _prepare(np.asarray(prepared[k].fine.points), coarse_voxel)
            return coarse[k]

        def crosses(p):
            a, b = comps.find(p.source), comps.find(p.target)
            if s.anchor:  # only links to the merged cloud's group help
                root = comps.find(0)
                return (a == root) != (b == root)
            return a != b

        retried: set[tuple[int, int]] = set()
        budget = 3 * (comps.count() - 1) + 3
        while comps.count() > 1 and budget > 0:
            cand = [p for p in pairs if crosses(p)
                    and (min(p.source, p.target), max(p.source, p.target)) not in retried]
            if not cand:
                break
            old = max(cand, key=lambda p: p.fitness)
            a, b = old.source, old.target
            retried.add((min(a, b), max(a, b)))
            budget -= 1
            progress(f"Retrying cloud {a + 1} with cloud {b + 1}", 0.75)
            rough = register_pair(coarse_of(a), coarse_of(b), coarse_voxel, retry, a, b)
            t = _refine(prepared[a], prepared[b], rough.transformation, voxel)
            p = _pair_result(prepared[a], prepared[b], t, voxel, a, b)
            pairs.append(p)
            if p.trusted(s):
                comps.union(a, b)

    # Loop closures from the rough poses.
    good = [p for p in pairs if p.trusted(s)]
    root = 0 if s.anchor else _pick_root(n, good)
    poses = _tree_poses(n, good, root)
    boxes = {}
    for k, pose in poses.items():
        pts = np.asarray(prepared[k].down.points) @ pose[:3, :3].T + pose[:3, 3]
        boxes[k] = (pts.min(axis=0), pts.max(axis=0))
    closures = []
    for i, j in itertools.combinations(sorted(poses), 2):
        if (i, j) in tried:
            continue
        lo = np.maximum(boxes[i][0], boxes[j][0])
        hi = np.minimum(boxes[i][1], boxes[j][1])
        if np.any(hi - lo <= voxel):
            continue
        closures.append((i, j))
    for k, (i, j) in enumerate(closures):
        progress(f"Checking loop closure: cloud {i + 1} with cloud {j + 1}",
                 0.75 + 0.15 * k / len(closures))
        init = np.linalg.inv(poses[j]) @ poses[i]  # maps i into j
        ev = reg.evaluate_registration(prepared[i].down, prepared[j].down, voxel * 2.0, init)
        if ev.fitness < 0.5 * s.min_fitness:
            continue
        t = _refine(prepared[i], prepared[j], init, voxel)
        pairs.append(_pair_result(prepared[i], prepared[j], t, voxel, i, j))
    return pairs


def _pick_root(n: int, good: list[PairResult]) -> int:
    """Cloud 0, unless it matched nothing while other clouds did."""
    comps = _Components(n)
    for p in good:
        comps.union(p.source, p.target)
    sizes: dict[int, list[int]] = {}
    for i in range(n):
        sizes.setdefault(comps.find(i), []).append(i)
    if len(sizes[comps.find(0)]) > 1:
        return 0
    biggest = max(sizes.values(), key=lambda m: (len(m), -m[0]))
    return biggest[0] if len(biggest) > 1 else 0


def _tree_poses(n: int, good: list[PairResult], root: int) -> dict[int, np.ndarray]:
    """Poses (world_from_cloud, world = root) from a maximum spanning tree on structure."""
    comps = _Components(n)
    tree = []
    for p in sorted(good, key=lambda p: -p.structure):
        if comps.union(p.source, p.target):
            tree.append(p)
    # Edge T maps source into target: T ~= inv(pose_target) @ pose_source.
    poses: dict[int, np.ndarray] = {root: np.eye(4)}
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
    return poses


def _pose_graph(n, pairs, prepared, voxel, s):
    """Spanning tree from the best pairs for initial poses, then optimise."""
    good = [p for p in pairs if p.trusted(s)]
    root = 0 if s.anchor else _pick_root(n, good)
    poses = _tree_poses(n, good, root)
    tree_edges = _tree_edge_ids(n, good)
    connected = [i in poses for i in range(n)]

    graph = reg.PoseGraph()
    for i in range(n):
        graph.nodes.append(reg.PoseGraphNode(poses.get(i, np.eye(4))))
    for p in good:
        if not (connected[p.source] and connected[p.target]):
            continue
        graph.edges.append(
            reg.PoseGraphEdge(p.source, p.target, p.transformation, p.information,
                              uncertain=id(p) not in tree_edges)
        )
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=voxel * 1.5,
        edge_prune_threshold=0.25,
        reference_node=root,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(
            graph,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    ref_inv = np.linalg.inv(np.asarray(graph.nodes[root].pose))
    transforms = [ref_inv @ np.asarray(graph.nodes[i].pose) for i in range(n)]
    for i in range(n):
        if not connected[i]:
            transforms[i] = np.eye(4)
    return transforms, connected, root


def _tree_edge_ids(n: int, good: list[PairResult]) -> set[int]:
    comps = _Components(n)
    return {id(p) for p in sorted(good, key=lambda p: -p.structure)
            if comps.union(p.source, p.target)}


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
