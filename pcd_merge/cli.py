"""Command-line interface: ``pcd-merge a.pcd b.pcd [...] -o merged.pcd``.

With no file arguments the desktop app is started instead.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _parse(argv):
    p = argparse.ArgumentParser(
        prog="pcd-merge",
        description="Automatically align two or more PCD point clouds and merge them. "
        "Run without arguments to open the desktop app.",
    )
    p.add_argument("files", nargs="*", help="PCD files; the first one is the reference frame")
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
                   help="merge even if some clouds could not be aligned")
    p.add_argument("--selftest", action="store_true",
                   help="run a quick end-to-end check of this installation and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        from .selftest import run as selftest

        return selftest()
    if not args.files:
        from .gui import run

        return run()

    from .cleanup import CleanupOptions
    from .exporters import format_of, write_cloud
    from .merge import merge_clouds_with_stats
    from .pcd_io import read_pcd
    from .registration import AlignSettings, align

    if len(args.files) < 2:
        print("error: give at least two PCD files", file=sys.stderr)
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
    for path in args.files:
        c = read_pcd(path)
        print(f"Loaded {c.name}: {len(c):,} points, fields: {', '.join(c.field_names)}")
        clouds.append(c)

    def progress(msg, frac):
        print(f"[{frac:4.0%}] {msg}")

    xyz = [c.xyz[c.finite_mask()] for c in clouds]
    result = align(xyz, AlignSettings(voxel_size=args.voxel, seed=args.seed), progress)

    print(f"\nVoxel size used for alignment: {result.voxel_size:.4g}")
    for p in result.pairs:
        print(f"  {clouds[p.source].name} <-> {clouds[p.target].name}: "
              f"overlap {p.fitness:.0%}, rmse {p.rmse:.4g}")
    for i, c in enumerate(clouds):
        print(f"  {c.name}: {result.cloud_fitness[i]:.0%} of points overlap the others")
    for w in result.warnings():
        print("warning:", w)

    from .quality import assess

    quality = assess(xyz, result.transforms, result.voxel_size)
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
    if not result.ok and not args.force:
        print("error: not all clouds could be aligned; not merging (use --force to merge anyway)",
              file=sys.stderr)
        return 1
    cleanup = CleanupOptions(dedup=args.dedup is not None, dedup_cell=args.dedup or None,
                             denoise=args.denoise, remove_moving=args.remove_moving)
    merged, partial, stats = merge_clouds_with_stats(
        clouds, result.transforms, add_source_field=args.source_field, cleanup=cleanup,
        progress=progress)
    if stats is not None:
        for line in stats.lines([c.name for c in clouds]):
            print(line)
    if partial:
        print("note: fields missing from some files were filled with 0:", ", ".join(partial))
    if args.min_spacing:
        from .thinning import thin

        before = len(merged)
        merged = thin(merged, args.min_spacing)
        print(f"Thinned to {args.min_spacing:g} spacing: kept {len(merged):,} of {before:,} points")
    for note in write_cloud(args.output, merged, compressed=args.compressed):
        print("note:", note)
    print(f"Wrote {len(merged):,} points to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
