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
    out: list[str] = []
    seen: set[str] = set()

    def add(p):
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)

    for p in paths:
        if os.path.isdir(p):
            found = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                found += [os.path.join(root, f) for f in files if f.lower().endswith(".pcd")]
            for f in sorted(found, key=natural_key):
                add(f)
        elif p.lower().endswith(".pcd"):
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
                progress=None):
    """Merge the included clouds, write the cloud (format from the extension)
    and its manifest.

    ``base_path`` is set when cloud 0 is an existing merged cloud being added
    to: its manifest (if any) is carried over and the new scans are appended.
    Returns ``(number of points, partially-present fields, notes)``.
    """
    from .exporters import write_cloud
    from .merge import merge_clouds, source_ids

    progress = progress or (lambda _m, _f: None)

    include = list(include) if include is not None else [True] * len(clouds)
    keep = [k for k, inc in enumerate(include) if inc]
    sel = [clouds[k] for k in keep]
    tfs = [transforms[k] for k in keep]
    progress("Applying transforms to full-resolution clouds", 0.1)
    merged, partial = merge_clouds(sel, tfs, add_source_field=add_source)
    numbered = add_source or "source" in merged.dtype.names
    ids = source_ids(sel)
    entries = []
    if base_path is not None and include[0]:
        entries = read_manifest(base_path) or [{"file": os.path.abspath(base_path),
                                                "note": "existing merged cloud"}]
        ids = [None] + ids[1:]  # the base's own scans are already listed
    for c, t, sid in zip(sel, tfs, ids):
        if sid is None:
            continue
        entry = {"file": os.path.abspath(c.path) if c.path else c.name,
                 "points": int(len(c)),
                 "transform": np.asarray(t).tolist()}
        if numbered:
            entry["source"] = sid
        entries.append(entry)
    notes = []
    if min_spacing:
        from .thinning import thin

        progress(f"Thinning {len(merged):,} points to {min_spacing:g} spacing", 0.4)
        before = len(merged)
        merged = thin(merged, min_spacing)
        notes.append(f"thinned to a minimum spacing of {min_spacing:g}: kept {len(merged):,} "
                     f"of {before:,} points ({len(merged) / max(before, 1):.0%})")
    progress(f"Writing {len(merged):,} points", 0.6)
    notes += write_cloud(path, merged, compressed=compressed)
    write_manifest(path, entries, voxel_size)
    return len(merged), partial, notes
