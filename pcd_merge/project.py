"""Projects: one folder per site that remembers everything done to its scans.

A project is a folder (so it is easy to copy, zip or put on a network drive)::

    Office Building.pcdproj/
      project.json   scans, transforms, quality, settings, exports
      edits/         per-scan delete masks and split regions (Edit mode, 0.10)
      cache/         preview samples and heat maps (safe to delete; rebuilt)
      view/          prepared viewing file for Inspect (0.11, rebuildable)
      photos/        pictures attached to notes
      exports/       default place for anything exported

Rules (see docs/plan-1.0.md):

* **Scans are referenced, never changed.** Each scan's path is stored both
  relative to the project folder and absolute, with its size, modification
  time and a quick hash, so a moved or replaced file is noticed and the user
  can point to it (:meth:`Project.check_scans`, :meth:`Project.relocate`).
* **Work is stored as instructions**, not as new point files: transforms,
  settings, later delete masks. The full cloud is only produced on export.
* **The cache makes reopening fast**: each scan's preview sample is kept in
  ``cache/`` so the 3D view comes back without re-reading the scans.
* **Versioned**: ``project.json`` has ``format_version``. Newer files are
  refused with a clear message instead of being half read.

This module has no Qt in it; the app autosaves through :meth:`Project.save`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field

import numpy as np

FORMAT_VERSION = 1
SUFFIX = ".pcdproj"
PROJECT_FILE = "project.json"
SUBFOLDERS = ("edits", "cache", "view", "photos", "exports")
_HASH_BLOCK = 1 << 20  # bytes read from the start, middle and end for the quick hash
MAX_RECENT = 10


class ProjectError(ValueError):
    """A project that cannot be opened (missing, damaged, or from a newer version)."""


def quick_hash(path: str) -> str:
    """Fast fingerprint of a big file: its size plus 1 MB from the start,
    middle and end. Enough to notice a replaced or edited scan without
    reading gigabytes."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        for pos in sorted({0, max(0, size // 2 - _HASH_BLOCK // 2), max(0, size - _HASH_BLOCK)}):
            f.seek(pos)
            h.update(f.read(_HASH_BLOCK))
    return h.hexdigest()[:20]


def project_folder(path: str) -> str:
    """The project folder for a path given by the user: the folder itself,
    its ``project.json``, or a name without the ``.pcdproj`` suffix."""
    path = os.path.abspath(path)
    if os.path.basename(path) == PROJECT_FILE:
        return os.path.dirname(path)
    if not path.lower().endswith(SUFFIX) and not os.path.isdir(path):
        path += SUFFIX
    return path


def is_project(path: str) -> bool:
    return os.path.isfile(os.path.join(project_folder(path), PROJECT_FILE))


# --------------------------------------------------------------------------
# Scans


@dataclass
class ScanEntry:
    """One scan in a project. ``path`` is absolute while the project is open."""

    path: str
    index: int = 0  # scan inside the file (multi-scan E57)
    scans_in_file: int = 1
    name: str = ""
    points: int = 0
    fields: list = field(default_factory=list)
    size: int = 0
    mtime: float = 0.0
    hash: str = ""
    transform: list | None = None  # 4x4, cloud -> project frame; None = not aligned
    visible: bool = True
    locked: bool = False  # transform checked by the user; Align keeps it
    base: bool = False  # an earlier merged cloud that new scans are added to
    viewpoint: list = field(default_factory=lambda: [0, 0, 0, 1, 0, 0, 0])  # scanner pose

    @property
    def key(self) -> str:
        p = os.path.normcase(os.path.abspath(self.path))
        return f"{p}#{self.index}" if self.scans_in_file > 1 else p

    @property
    def matrix(self) -> np.ndarray:
        return np.eye(4) if self.transform is None else np.asarray(self.transform, float)

    def status(self) -> str:
        """``ok``, ``missing`` or ``changed`` (the file is not the one added)."""
        if not os.path.isfile(self.path):
            return "missing"
        try:
            st = os.stat(self.path)
        except OSError:
            return "missing"
        if st.st_size != self.size:
            return "changed"
        if abs(st.st_mtime - self.mtime) < 1e-3 or not self.hash:
            return "ok"
        return "ok" if quick_hash(self.path) == self.hash else "changed"

    def stamp(self) -> None:
        """Record the file's current size, time and hash."""
        st = os.stat(self.path)
        self.size, self.mtime = st.st_size, st.st_mtime
        self.hash = quick_hash(self.path)

    @classmethod
    def from_source(cls, source, base: bool = False) -> "ScanEntry":
        """Entry for a :class:`~pcd_merge.readers.ScanSource` or loaded PointCloud."""
        from .pcd_io import PointCloud

        if isinstance(source, PointCloud):
            e = cls(path=os.path.abspath(source.path), name=source.name, points=len(source),
                    fields=list(source.field_names), base=base,
                    viewpoint=[float(v) for v in source.viewpoint],
                    index=int(source.extra.get("scan_index", 0)),
                    scans_in_file=int(source.extra.get("scans_in_file", 1)))
        else:
            e = cls(path=os.path.abspath(source.path), index=source.index,
                    scans_in_file=source.scans_in_file, name=source.name, points=source.count,
                    fields=list(source.dtype.names), base=base,
                    viewpoint=[float(v) for v in source.viewpoint])
        e.stamp()
        return e

    def cache_name(self) -> str:
        return f"{self.hash or 'nohash'}-{self.index}"


# --------------------------------------------------------------------------
# Project


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class Project:
    """``folder`` is empty for a project that has not been saved yet (it then
    lives in memory only, and the cache is not used)."""

    folder: str = ""
    scans: list[ScanEntry] = field(default_factory=list)
    # Align settings (voxel size), clean-up options, export options, view...
    settings: dict = field(default_factory=dict)
    alignment: dict | None = None  # see alignment_to_dict()
    quality: dict | None = None  # QualityReport.to_dict() plus the report text
    exports: list = field(default_factory=list)  # {"path", "format", "points", "time"}
    notes: str = ""
    created: str = field(default_factory=_now)
    modified: str = field(default_factory=_now)

    # ---- locations

    @property
    def saved(self) -> bool:
        return bool(self.folder)

    @property
    def name(self) -> str:
        if not self.folder:
            return "Untitled"
        base = os.path.basename(self.folder.rstrip("/\\"))
        return base[:-len(SUFFIX)] if base.lower().endswith(SUFFIX) else base

    @property
    def file(self) -> str:
        return os.path.join(self.folder, PROJECT_FILE)

    def sub(self, name: str) -> str:
        p = os.path.join(self.folder, name)
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> str:
        return self.sub("cache")

    @property
    def exports_dir(self) -> str:
        return self.sub("exports")

    # ---- create / open / save

    @classmethod
    def create(cls, path: str) -> "Project":
        """New empty project at ``path`` (``.pcdproj`` is added if missing).
        Refuses to overwrite an existing project."""
        folder = project_folder(path)
        if os.path.exists(os.path.join(folder, PROJECT_FILE)):
            raise ProjectError(f"There is already a project at {folder}")
        os.makedirs(folder, exist_ok=True)
        p = cls(folder=folder)
        for s in SUBFOLDERS:
            os.makedirs(os.path.join(folder, s), exist_ok=True)
        p.save()
        return p

    @classmethod
    def open(cls, path: str) -> "Project":
        folder = project_folder(path)
        try:
            with open(os.path.join(folder, PROJECT_FILE), encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ProjectError(f"No project found at {folder}") from None
        except (OSError, ValueError) as e:
            raise ProjectError(f"The project file in {folder} could not be read ({e})") from None
        return cls.from_dict(data, folder)

    def save(self) -> None:
        """Write ``project.json`` atomically (a crash never leaves half a file)."""
        if not self.folder:
            raise ProjectError("The project has no folder yet; use save_as()")
        os.makedirs(self.folder, exist_ok=True)
        self.modified = _now()
        tmp = self.file + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, self.file)

    def save_as(self, path: str) -> "Project":
        """Copy this project (settings, cache, edits; not the exports) to a new
        folder and return the copy. Scans stay where they are."""
        folder = project_folder(path)
        if os.path.exists(os.path.join(folder, PROJECT_FILE)):
            raise ProjectError(f"There is already a project at {folder}")
        os.makedirs(folder, exist_ok=True)
        for s in SUBFOLDERS:
            src = os.path.join(self.folder, s) if self.folder else ""
            dst = os.path.join(folder, s)
            if s != "exports" and os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
            os.makedirs(dst, exist_ok=True)
        copy = Project.from_dict(self.to_dict(), folder)
        copy.save()
        return copy

    # ---- (de)serialising

    def to_dict(self) -> dict:
        from . import __version__

        scans = []
        for s in self.scans:
            d = asdict(s)
            d["path"] = os.path.abspath(s.path)
            try:
                d["relative_path"] = (os.path.relpath(s.path, self.folder).replace(os.sep, "/")
                                      if self.folder else None)
            except ValueError:  # another drive on Windows
                d["relative_path"] = None
            scans.append(d)
        return {
            "format_version": FORMAT_VERSION,
            "program": f"PCD Merge {__version__}",
            "created": self.created,
            "modified": self.modified,
            "scans": scans,
            "settings": self.settings,
            "alignment": self.alignment,
            "quality": self.quality,
            "exports": self.exports,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict, folder: str) -> "Project":
        if not isinstance(data, dict) or "scans" not in data:
            raise ProjectError("This is not a PCD Merge project file")
        version = int(data.get("format_version", 1))
        if version > FORMAT_VERSION:
            raise ProjectError(f"This project was saved by a newer PCD Merge ({data.get('program')}); "
                               "update the app to open it")
        known = set(ScanEntry.__dataclass_fields__)
        scans = []
        for d in data.get("scans", []):
            entry = ScanEntry(**{k: v for k, v in d.items() if k in known})
            rel = d.get("relative_path")
            if rel:
                # A project moved together with its scans: the relative path wins.
                cand = os.path.normpath(os.path.join(folder, rel))
                if os.path.isfile(cand):
                    entry.path = cand
            scans.append(entry)
        return cls(folder=folder, scans=scans, settings=dict(data.get("settings") or {}),
                   alignment=data.get("alignment"), quality=data.get("quality"),
                   exports=list(data.get("exports") or []), notes=data.get("notes", ""),
                   created=data.get("created", _now()), modified=data.get("modified", _now()))

    # ---- scans

    def find(self, key: str) -> ScanEntry | None:
        return next((s for s in self.scans if s.key == key), None)

    def add(self, source, base: bool = False) -> ScanEntry:
        """Add a scan (a ScanSource or loaded PointCloud). An already-present
        scan is returned as it is."""
        entry = ScanEntry.from_source(source, base=base)
        old = self.find(entry.key)
        if old is not None:
            return old
        if base:
            self.scans = [s for s in self.scans if not s.base]
            self.scans.insert(0, entry)
        else:
            self.scans.append(entry)
        return entry

    def remove(self, keys) -> None:
        keys = set(keys)
        self.scans = [s for s in self.scans if s.key not in keys]

    def check_scans(self) -> list[tuple[ScanEntry, str]]:
        """Scans whose file is ``missing`` or ``changed``."""
        return [(s, st) for s in self.scans if (st := s.status()) != "ok"]

    def relocate(self, entry: ScanEntry, new_path: str) -> None:
        """Point a scan at its file's new location (and accept it as it is now).
        Other missing scans from the same folder are looked for next to it."""
        old_dir = os.path.dirname(entry.path)
        new_dir = os.path.dirname(os.path.abspath(new_path))
        entry.path = os.path.abspath(new_path)
        entry.stamp()
        for s in self.scans:
            if s is not entry and os.path.dirname(s.path) == old_dir and s.status() == "missing":
                cand = os.path.join(new_dir, os.path.basename(s.path))
                if os.path.isfile(cand) and os.path.getsize(cand) == s.size:
                    s.path = cand

    def copy_scans_in(self, progress=None) -> None:
        """Copy every scan file into ``scans/`` inside the project, so the
        project can be handed over on its own."""
        dest = self.sub("scans")
        files = sorted({s.path for s in self.scans if os.path.isfile(s.path)})
        mapping = {}
        for k, src in enumerate(files):
            if os.path.dirname(os.path.abspath(src)) == os.path.abspath(dest):
                mapping[src] = src
                continue
            target = os.path.join(dest, os.path.basename(src))
            n = 1
            while os.path.exists(target) and quick_hash(target) != quick_hash(src):
                stem, ext = os.path.splitext(os.path.basename(src))
                target = os.path.join(dest, f"{stem}_{n}{ext}")
                n += 1
            if not os.path.exists(target):
                shutil.copy2(src, target)
            mapping[src] = target
            if progress:
                progress(f"Copying {os.path.basename(src)}", (k + 1) / len(files))
        for s in self.scans:
            if s.path in mapping:
                s.path = mapping[s.path]

    # ---- transforms and results

    def transforms(self) -> list[np.ndarray]:
        return [s.matrix for s in self.scans]

    def set_alignment(self, result, quality=None, report_lines=None) -> None:
        """Store an :class:`~pcd_merge.registration.AlignResult` (and its
        quality report) against the current scans, in order."""
        if len(result.transforms) != len(self.scans):
            raise ValueError("one transform per scan is required")
        for s, t in zip(self.scans, result.transforms):
            s.transform = np.asarray(t, float).tolist()
        self.alignment = alignment_to_dict(result, [s.key for s in self.scans])
        self.quality = None
        if quality is not None:
            self.quality = quality.to_dict()
            self.quality["keys"] = [s.key for s in self.scans]
            if report_lines is not None:
                self.quality["report"] = list(report_lines)

    def clear_alignment(self) -> None:
        for s in self.scans:
            if not s.locked:
                s.transform = None
        self.alignment = None
        self.quality = None

    def alignment_matches_scans(self) -> bool:
        """Whether the stored alignment was made for exactly these scans."""
        return bool(self.alignment) and self.alignment.get("keys") == [s.key for s in self.scans]

    def record_export(self, path: str, points: int, notes=None, visible: bool = False) -> dict:
        """Remember an exported cloud (it is listed in the project and can be
        shown in the view). Exporting to the same file again replaces the entry."""
        path = os.path.abspath(path)
        self.exports = [e for e in self.exports
                        if os.path.normcase(e.get("path", "")) != os.path.normcase(path)]
        entry = {"path": path, "format": os.path.splitext(path)[1], "points": int(points),
                 "time": _now(), "scans": [s.key for s in self.scans], "visible": visible}
        if os.path.isfile(path):
            entry["hash"] = quick_hash(path)
        if notes:
            entry["notes"] = list(notes)
        self.exports.append(entry)
        return entry

    def remove_export(self, path: str) -> None:
        self.exports = [e for e in self.exports
                        if os.path.normcase(e.get("path", "")) != os.path.normcase(path)]

    def save_export_preview(self, export: dict, xyz: np.ndarray) -> None:
        if not self.folder or not export.get("hash"):
            return
        path = os.path.join(self.cache_dir, f"export-{export['hash']}.npy")
        np.save(path + ".part.npy", np.asarray(xyz, dtype=np.float64))
        os.replace(path + ".part.npy", path)

    def load_export_preview(self, export: dict) -> np.ndarray | None:
        if not self.folder or not export.get("hash"):
            return None
        try:
            xyz = np.load(os.path.join(self.cache_dir, f"export-{export['hash']}.npy"))
        except (OSError, ValueError):
            return None
        return xyz if xyz.ndim == 2 and xyz.shape[1] == 3 else None

    # ---- cache

    def _cache_file(self, entry: ScanEntry, what: str) -> str:
        return os.path.join(self.cache_dir, f"{what}-{entry.cache_name()}.npy")

    def save_preview(self, entry: ScanEntry, xyz: np.ndarray) -> None:
        """Keep a scan's preview sample (random order) for fast reopening."""
        if not self.folder:
            return
        path = self._cache_file(entry, "preview")
        tmp = path + ".part.npy"
        np.save(tmp, np.asarray(xyz, dtype=np.float64))
        os.replace(tmp, path)

    def load_preview(self, entry: ScanEntry) -> np.ndarray | None:
        """The cached preview sample, or None if missing or the scan changed."""
        if not entry.hash or not self.folder:
            return None
        try:
            xyz = np.load(self._cache_file(entry, "preview"))
        except (OSError, ValueError):
            return None
        return xyz if xyz.ndim == 2 and xyz.shape[1] == 3 else None

    def save_heatmap(self, point_error: list[np.ndarray]) -> None:
        if len(point_error) != len(self.scans) or not self.folder:
            return
        np.savez(os.path.join(self.cache_dir, "heatmap.part.npz"),
                 keys=np.array([s.key for s in self.scans]),
                 **{f"e{i}": np.asarray(e, np.float32) for i, e in enumerate(point_error)})
        os.replace(os.path.join(self.cache_dir, "heatmap.part.npz"),
                   os.path.join(self.cache_dir, "heatmap.npz"))

    def load_heatmap(self) -> list[np.ndarray] | None:
        if not self.folder:
            return None
        try:
            with np.load(os.path.join(self.cache_dir, "heatmap.npz")) as z:
                if list(z["keys"]) != [s.key for s in self.scans]:
                    return None
                return [z[f"e{i}"].astype(np.float64) for i in range(len(self.scans))]
        except (OSError, KeyError, ValueError):
            return None

    def clear_cache(self) -> None:
        shutil.rmtree(os.path.join(self.folder, "cache"), ignore_errors=True)
        os.makedirs(os.path.join(self.folder, "cache"), exist_ok=True)


# --------------------------------------------------------------------------
# Alignment results


def alignment_to_dict(result, keys: list[str]) -> dict:
    return {
        "keys": list(keys),
        "voxel_size": float(result.voxel_size),
        "connected": [bool(c) for c in result.connected],
        "cloud_fitness": [float(f) for f in result.cloud_fitness],
        "reference": int(result.reference),
        "anchored": bool(result.anchored),
        "pairs": [{"source": p.source, "target": p.target, "fitness": float(p.fitness),
                   "rmse": float(p.rmse), "structure": float(p.structure),
                   "transformation": np.asarray(p.transformation).tolist()}
                  for p in result.pairs],
    }


def alignment_from_dict(d: dict, transforms: list[np.ndarray]):
    """Rebuild an AlignResult (without the downsampled clouds) from a project."""
    from .registration import AlignResult, PairResult

    pairs = [PairResult(source=p["source"], target=p["target"],
                        transformation=np.asarray(p["transformation"], float),
                        fitness=p["fitness"], rmse=p["rmse"], structure=p.get("structure", 0.0))
             for p in d.get("pairs", [])]
    return AlignResult(transforms=[np.asarray(t, float) for t in transforms], pairs=pairs,
                       voxel_size=d["voxel_size"], connected=list(d["connected"]),
                       cloud_fitness=list(d["cloud_fitness"]), reference=d.get("reference", 0),
                       anchored=d.get("anchored", False))


# --------------------------------------------------------------------------
# Old merged clouds


def from_manifest(merged_path: str, folder: str) -> Project:
    """New project from a merged cloud and its ``merged.pcd.json`` manifest:
    the merged cloud becomes the fixed base, so new scans can be added to it
    (the old "Add new scans to an existing merged cloud")."""
    from .files import read_manifest
    from .readers import open_scan

    project = Project.create(folder)
    project.add(open_scan(merged_path), base=True)
    project.scans[0].transform = np.eye(4).tolist()
    project.scans[0].locked = True
    project.settings["merged_from"] = read_manifest(merged_path)
    project.save()
    return project


# --------------------------------------------------------------------------
# Recent projects (stored in the user's settings folder, not in the project)


def config_dir() -> str:
    d = os.environ.get("PCD_MERGE_CONFIG_DIR")
    if not d:
        if os.name == "nt":
            d = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "PCD Merge")
        else:
            d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                             "pcd-merge")
    return d


def _recent_file() -> str:
    return os.path.join(config_dir(), "recent_projects.json")


def recent_projects() -> list[str]:
    """Recently used project folders that still exist, newest first."""
    try:
        with open(_recent_file(), encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, ValueError):
        return []
    return [p for p in items if isinstance(p, str) and is_project(p)][:MAX_RECENT]


def add_recent(folder: str) -> None:
    folder = os.path.abspath(folder)
    items = [p for p in recent_projects()
             if os.path.normcase(os.path.abspath(p)) != os.path.normcase(folder)]
    items.insert(0, folder)
    try:
        os.makedirs(config_dir(), exist_ok=True)
        tmp = _recent_file() + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items[:MAX_RECENT], f, indent=2)
        os.replace(tmp, _recent_file())
    except OSError:
        pass  # not being able to remember is no reason to fail
