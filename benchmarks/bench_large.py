"""Time loading, alignment and merging on large synthetic scans.

    python benchmarks/bench_large.py --points 5000000 --clouds 3

Writes the scans to a temporary folder (binary and binary_compressed), then
times each stage the app runs. Prints one line per stage so runs before and
after a change can be compared.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pcd_merge.merge import merge_clouds  # noqa: E402
from pcd_merge.pcd_io import read_pcd, write_pcd  # noqa: E402
from pcd_merge.registration import AlignSettings, align  # noqa: E402
from pcd_merge.synthetic import make_scan, make_scene, make_transform, transform_error  # noqa: E402

POSES = [
    make_transform(0.3, 0.02, (-5, 0, 1.5)),
    make_transform(2.0, 0.03, (4, 3, 1.6)),
    make_transform(-1.2, 0.0, (2, -7, 1.4)),
    make_transform(0.9, 0.01, (-3, 6, 1.5)),
]


class Timer:
    def __init__(self):
        self.rows = []

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                dt = time.perf_counter() - self.t
                timer.rows.append((name, dt))
                print(f"{name:<38}{dt:8.2f} s", flush=True)

        return _T()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=5_000_000, help="points per scan (approx.)")
    ap.add_argument("--clouds", type=int, default=3)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    tmp = args.dir or tempfile.mkdtemp(prefix="pcd-bench-")
    os.makedirs(tmp, exist_ok=True)
    poses = POSES[: args.clouds]
    paths = [os.path.join(tmp, f"scan{i}.pcd") for i in range(len(poses))]
    cpaths = [os.path.join(tmp, f"scan{i}_c.pcd") for i in range(len(poses))]
    if not all(os.path.exists(p) for p in paths + cpaths):
        # density so that each scan has roughly --points points
        density = 120 * args.points / 75_000
        scene = make_scene(seed=0, density=density)
        for i, p in enumerate(poses):
            d = make_scan(scene, p[:2, 3], 18, p, seed=i)
            write_pcd(paths[i], d)
            write_pcd(cpaths[i], d, compressed=True)
        del scene

    t = Timer()
    with t("load binary"):
        clouds = [read_pcd(p) for p in paths]
    with t("load binary_compressed"):
        [read_pcd(p) for p in cpaths]
    print(f"   points per scan: {', '.join(f'{len(c):,}' for c in clouds)}")
    with t("align (incl. extracting xyz)"):
        res = align([c.finite_xyz for c in clouds], AlignSettings(seed=1))
    for i in range(len(poses)):
        rot, tr = transform_error(res.transforms[i], np.linalg.inv(poses[0]) @ poses[i])
        print(f"   cloud {i}: rotation error {rot:.3f} deg, translation error {tr:.4f}")
    with t("merge"):
        merged, _ = merge_clouds(clouds, res.transforms)
    out = os.path.join(tmp, "merged.pcd")
    with t("write merged (binary)"):
        write_pcd(out, merged)
    total = sum(dt for name, dt in t.rows if name != "load binary_compressed")
    print(f"{'TOTAL (load+align+merge+write)':<38}{total:8.2f} s")
    try:
        import resource

        print(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    except ImportError:  # Windows
        pass


if __name__ == "__main__":
    main()
