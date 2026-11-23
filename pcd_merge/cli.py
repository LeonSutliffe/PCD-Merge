"""Command-line interface: ``pcd-merge a.pcd b.pcd [...] -o merged.pcd``.

Folders can be given instead of files (every .pcd inside, in natural order).
``--add-to merged.pcd new.pcd ...`` aligns new scans to an existing merged
cloud and appends them to it.

With no file arguments the desktop app is started instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _parse(argv):
    p = argparse.ArgumentParser(
        prog="pcd-merge",
        description="Automatically align two or more PCD point clouds and merge them. "
        "Run without arguments to open the desktop app.",
    )
    p.add_argument("files", nargs="*",
                   help="PCD files or folders of them; the first one is the reference frame")
    p.add_argument("--add-to", metavar="MERGED",
                   help="existing merged .pcd to add the new scans to; it stays fixed and "
                        "the scans are aligned to it")
    p.add_argument("-o", "--output",
                   help="merged cloud to write; the extension picks the format: "
                   ".pcd, .ply, .las, .laz or .e57")
    p.add_argument("--voxel", type=float, default=None,
                   help="voxel size for alignment in scan units (default: automatic)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable runs")
    p.add_argument("--transforms", help="write the estimated 4x4 transforms to this JSON file")
    p.add_argument("--source-field", action="store_true",
                   help="add a 'source' field saying which file each point came from")
    g = p.add_argument_group("merge clean-up (all off by default; only ever removes points)")
    g.add_argument("--dedup", nargs="?", type=float, const=0.0, default=None, metavar="CELL",
                   help="where scans overlap, keep only the densest scan's points in each "
                   "CELL-sized cube (default cell: automatic)")
    g.add_argument("--denoise", action="store_true",
                   help="remove isolated noise points (dust, rain, stray returns)")
    g.add_argument("--remove-moving", action="store_true",
                   help="remove people, cars etc. that another scan saw through "
                   "(needs scans in their scanner's own frame)")
    p.add_argument("--compressed", action="store_true",
                   help="write binary_compressed PCD (PCD output only)")
    p.add_argument("--min-spacing", type=float, default=None, metavar="D",
                   help="thin the merged cloud so no two points are closer than D "
                   "(scan units); default: keep every point")
    p.add_argument("--force", action="store_true",
                   help="merge even if some clouds could not be aligned (they keep their "
                        "original position)")
    p.add_argument("--skip-unaligned", action="store_true",
                   help="merge the clouds that aligned and leave out the ones that did not")
    p.add_argument("--selftest", action="store_true",
                   help="run a quick end-to-end check of this installation and exit")
    p.add_argument("--benchmark", action="store_true",
                   help="time alignment and merging on large synthetic scans, on the CPU "
                   "and (if available) the NVIDIA GPU, and exit")
    p.add_argument("--points", type=int, default=10_000_000,
                   help="points per synthetic scan for --benchmark (default 10 million)")
    p.add_argument("--clouds", type=int, default=3,
                   help="number of synthetic scans for --benchmark (2 to 4, default 3)")
    p.add_argument("--gpu-info", action="store_true",
                   help="say whether the NVIDIA GPU would be used, and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        from .selftest import run as selftest

        return selftest()
    if args.benchmark:
        from .benchmark import run as benchmark

        return benchmark(points=args.points, clouds=args.clouds)
    if args.gpu_info:
        from .benchmark import gpu_info

        return gpu_info()
    if not args.files and not args.add_to:
        from .gui import run

        return run()

    from .cleanup import CleanupOptions
    from .exporters import format_of
    from .files import already_merged, find_pcd_files, read_manifest, save_merged
    from .pcd_io import read_pcd
    from .registration import AlignSettings, align

    files = find_pcd_files(args.files)
    base = args.add_to
    if base:
        files = [f for f in files if os.path.abspath(f) != os.path.abspath(base)]
        dup = already_merged(read_manifest(base), files)
        for f in dup:
            print(f"note: {f} is already in {base}; skipping it")
        files = [f for f in files if f not in dup]
        if not files:
            print("error: no new scans to add", file=sys.stderr)
            return 2
        files = [base] + files
    elif len(files) < 2:
        print("error: give at least two PCD files (or a folder containing them)", file=sys.stderr)
        return 2
    if args.output:
        try:
            format_of(args.output)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    if args.min_spacing is not None and args.min_spacing <= 0:
        print("error: --min-spacing must be positive", file=sys.stderr)
        return 2

    clouds = []
    for path in files:
        c = read_pcd(path)
        print(f"Loaded {c.name}: {len(c):,} points, fields: {', '.join(c.field_names)}")
        clouds.append(c)

    def progress(msg, frac):
        print(f"[{frac:4.0%}] {msg}")

    # bound methods: each cloud's coordinates are made only when needed
    xyz = [c.finite_xyz for c in clouds]
    settings = AlignSettings(voxel_size=args.voxel, seed=args.seed, anchor=bool(base))
    result = align(xyz, settings, progress)

    print(f"\nVoxel size used for alignment: {result.voxel_size:.4g}")
    for p in result.pairs:
        print(f"  {clouds[p.source].name} <-> {clouds[p.target].name}: "
              f"overlap {p.fitness:.0%}, rmse {p.rmse:.4g}")
    for i, c in enumerate(clouds):
        print(f"  {c.name}: {result.cloud_fitness[i]:.0%} of points overlap the others")
    for w in result.warnings():
        print("warning:", w)

    from .quality import assess

    quality = assess(result.fine or xyz, result.transforms, result.voxel_size)
    print()
    for line in quality.lines([c.name for c in clouds]):
        print(line)

    if args.transforms:
        with open(args.transforms, "w") as f:
            json.dump({c.path: np.asarray(t).tolist() for c, t in zip(clouds, result.transforms)},
                      f, indent=2)
        print(f"Wrote transforms to {args.transforms}")

    if not args.output:
        return 0
    include = [True] * len(clouds)
    if not result.ok:
        if args.skip_unaligned:
            include = list(result.connected)
            for c, inc in zip(clouds, include):
                if not inc:
                    print(f"note: leaving out {c.name} (not aligned)")
            if sum(include) < 2:
                print("error: fewer than two clouds aligned; nothing to merge", file=sys.stderr)
                return 1
        elif not args.force:
            print("error: not all clouds could be aligned; not merging (use --skip-unaligned "
                  "to leave them out, or --force to merge anyway)", file=sys.stderr)
            return 1
    cleanup = CleanupOptions(dedup=args.dedup is not None, dedup_cell=args.dedup or None,
                             denoise=args.denoise, remove_moving=args.remove_moving)
    n, partial, notes, stats = save_merged(
        clouds, result.transforms, args.output, include, add_source=args.source_field,
        compressed=args.compressed, base_path=base, voxel_size=result.voxel_size,
        min_spacing=args.min_spacing, cleanup=cleanup, progress=progress)
    if stats is not None:
        kept = [c.name for c, inc in zip(clouds, include) if inc]
        for line in stats.lines(kept):
            print(line)
    if partial:
        print("note: fields missing from some files were filled with 0:", ", ".join(partial))
    for note in notes:
        print("note:", note)
    print(f"Wrote {n:,} points to {args.output} (scan list in {args.output}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
