"""Finding scans in folders, and the record of what went into a merged cloud."""

from __future__ import annotations

import json
import os
import re

import numpy as np


def natural_key(path: str):
    """Sort key so that ``scan_2`` comes before ``scan_10``."""
    parts = re.split(r"(\d+)", path.replace("\\", "/").lower())
    return [int(p) if p.isdigit() else p for p in parts]


def find_pcd_files(paths: list[str]) -> list[str]:
    """Expand folders (including subfolders) into the .pcd files inside them.

    Files inside each folder come in natural order (``scan_2`` before
    ``scan_10``): batch alignment starts by matching neighbours in this order,
    which for most scanners is the order the scans were captured in.
    Files given directly keep the order they were given in. Duplicates are
    dropped.
    """
    return find_files(paths, (".pcd",))


def find_files(paths: list[str], exts: tuple[str, ...]) -> list[str]:
    """Like :func:`find_pcd_files` for any set of extensions (lower case, with dot)."""
    out: list[str] = []
    seen: set[str] = set()

    def add(p):
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)

    def wanted(name: str) -> bool:
        return os.path.splitext(name)[1].lower() in exts

    for p in paths:
        if os.path.isdir(p):
            if p.lower().rstrip("/\\").endswith(".pcdproj"):
                continue  # a project folder, not a folder of scans
            found = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and not d.lower().endswith(".pcdproj")]
                found += [os.path.join(root, f) for f in files
                          if wanted(f) and not f.endswith((".part", ".json"))]
            for f in sorted(found, key=natural_key):
                add(f)
        elif wanted(p):
            add(p)
    return out


# --------------------------------------------------------------------------
# Manifest: ``merged.pcd.json`` next to a merged cloud, listing the scans in it.


def manifest_path(merged_path: str) -> str:
    return merged_path + ".json"


def read_manifest(merged_path: str) -> list[dict]:
    """Scans recorded for an earlier merge, or ``[]`` if there is no record."""
    try:
        with open(manifest_path(merged_path), encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("scans", []))
    except (OSError, ValueError, AttributeError):
        return []


def write_manifest(merged_path: str, scans: list[dict], voxel_size: float | None = None) -> None:
    from . import __version__

    data = {"program": f"PCD Merge {__version__}", "scans": scans}
    if voxel_size:
        data["voxel_size"] = voxel_size
    tmp = manifest_path(merged_path) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, manifest_path(merged_path))


def already_merged(manifest: list[dict], paths: list[str]) -> list[str]:
    """Which of ``paths`` the manifest says are already in the merged cloud."""
    names = {os.path.normcase(os.path.abspath(e.get("file", ""))) for e in manifest}
    return [p for p in paths if os.path.normcase(os.path.abspath(p)) in names]


def save_merged(clouds, transforms, path: str, include=None, add_source: bool = False,
                compressed: bool = False, base_path: str | None = None,
                voxel_size: float | None = None, min_spacing: float | None = None,
                cleanup=None, progress=None):
    """Merge the included clouds, write the cloud (format from the extension)
    and its manifest.

    ``clouds`` are :class:`~pcd_merge.pcd_io.PointCloud` objects (in memory)
    or :class:`~pcd_merge.readers.ScanSource` objects (read from disk as
    needed). Without clean-up, thinning or PCD compression the merge is
    streamed: each scan is transformed and written a chunk at a time, so the
    merged cloud is never held in memory and there is no size limit beyond
    the disk. The other options need every point at once and load the scans.

    ``base_path`` is set when cloud 0 is an existing merged cloud being added
    to: its manifest (if any) is carried over and the new scans are appended.
    ``cleanup`` is an optional :class:`~pcd_merge.cleanup.CleanupOptions`.
    Returns ``(number of points, partially-present fields, notes, cleanup stats)``.
    """
    from .exporters import format_of, write_chunks, write_cloud
    from .merge import StreamedMerge, merge_clouds_with_stats, source_ids
    from .pcd_io import PointCloud
    from .readers import MemorySource

    progress = progress or (lambda _m, _f: None)

    include = list(include) if include is not None else [True] * len(clouds)
    keep = [k for k, inc in enumerate(include) if inc]
    sel = [clouds[k] for k in keep]
    tfs = [transforms[k] for k in keep]
    streamed = (not (cleanup is not None and cleanup.any) and not min_spacing
                and not (compressed and format_of(path) == ".pcd"))
    notes = []
    stats = None
    if streamed:
        sources = [MemorySource(c) if isinstance(c, PointCloud) else c for c in sel]
        merge = StreamedMerge(sources, tfs, add_source_field=add_source)
        partial, numbered, ids = merge.missing, merge.numbered, list(merge.ids)
        msg = f"Merging and writing {merge.total:,} points to {os.path.basename(path)}"
        progress(msg, 0.0)
        notes += write_chunks(path, merge.dtype, lambda: merge, total=merge.total,
                              progress=lambda f: progress(msg, f))
        counts = list(merge.written)
        n_points = sum(counts)
    else:
        sel = [c if isinstance(c, PointCloud) else c.read() for c in sel]
        merged, partial, stats = merge_clouds_with_stats(
            sel, tfs, add_source_field=add_source, cleanup=cleanup,
            progress=lambda m, f: progress(m, 0.5 * f))
        numbered = add_source or "source" in merged.dtype.names
        ids = source_ids(sel)
        counts = [len(c) for c in sel]
        if min_spacing:
            from .thinning import thin

            progress(f"Thinning {len(merged):,} points to {min_spacing:g} spacing", 0.5)
            before = len(merged)
            merged = thin(merged, min_spacing)
            notes.append(f"thinned to a minimum spacing of {min_spacing:g}: kept {len(merged):,} "
                         f"of {before:,} points ({len(merged) / max(before, 1):.0%})")
        msg = f"Writing {len(merged):,} points to {os.path.basename(path)}"
        progress(msg, 0.6)
        notes += write_cloud(path, merged, compressed=compressed,
                             progress=lambda f: progress(msg, 0.6 + 0.4 * f))
        n_points = len(merged)
    entries = []
    if base_path is not None and include[0]:
        entries = read_manifest(base_path) or [{"file": os.path.abspath(base_path),
                                                "note": "existing merged cloud"}]
        ids = [None] + ids[1:]  # the base's own scans are already listed
    for c, t, sid, n in zip(sel, tfs, ids, counts):
        if sid is None:
            continue
        entry = {"file": os.path.abspath(c.path) if c.path else c.name,
                 "points": int(n),
                 "transform": np.asarray(t).tolist()}
        if isinstance(c, PointCloud):
            scan_index = c.extra.get("scan_index")
        else:
            scan_index = c.index if c.scans_in_file > 1 else None
        if scan_index is not None:
            entry["scan"] = int(scan_index)
        if numbered:
            entry["source"] = sid
        entries.append(entry)
    write_manifest(path, entries, voxel_size)
    return n_points, partial, notes, stats
