"""The open project as the window sees it: the saved :class:`Project` plus
what only lives in memory (loaded clouds, preview samples, the last
alignment result), and the background jobs that fill it in.

Panels read from and write to one :class:`Document`; it emits ``changed``
so every panel and the 3D view can refresh, and autosaves the project a
moment after each change (once the project has a folder).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ..merge import sensor_origin
from ..pcd_io import PointCloud
from ..project import Project, ProjectError, ScanEntry, add_recent, alignment_from_dict
from ..quality import QualityReport
from ..readers import open_scan, sample_points
from .common import PREVIEW_POINTS
from .workers import Cancelled


class Scan:
    """A scan of the project at run time."""

    def __init__(self, entry: ScanEntry, sample: np.ndarray | None, cloud: PointCloud | None):
        self.entry = entry
        self.sample = sample if sample is not None else np.zeros((0, 3))  # random order
        self.cloud = cloud  # full resolution, once loaded

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def points(self) -> int:
        return len(self.cloud) if self.cloud is not None else self.entry.points

    @property
    def viewpoint(self):
        return self.cloud.viewpoint if self.cloud is not None else tuple(self.entry.viewpoint)

    def source(self):
        """Something :func:`files.save_merged` can read: the cloud, or the file."""
        return self.cloud if self.cloud is not None else open_scan(self.entry.path,
                                                                    self.entry.index)

    def sensor_origin(self, t) -> np.ndarray:
        return sensor_origin(SimpleNamespace(viewpoint=self.viewpoint), t)


def preview_sample(cloud: PointCloud, n: int = PREVIEW_POINTS) -> np.ndarray:
    """A random subset of up to ``n`` finite points, in random order.

    Taking the first k rows of it gives a random k-point preview, so the 3D
    view never has to touch the full cloud again.
    """
    d = cloud.data
    rng = np.random.default_rng(0)
    if len(d) > n:
        idx = np.unique(rng.integers(0, len(d), n))
        d = d[idx]
    xyz = np.empty((len(d), 3), dtype=np.float64)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = d["x"], d["y"], d["z"]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz[rng.permutation(len(xyz))]


# --------------------------------------------------------------------------
# background jobs (run through TaskBar.run, so they get ``progress``)


def load_files(paths, base_only=False, *, progress):
    """Read scans fully. Returns ([(cloud, sample)], errors)."""
    from ..readers import read_cloud, read_clouds

    out, errors = [], []
    sizes = []
    for p in paths:
        try:
            sizes.append(max(1, os.path.getsize(p)))
        except OSError:
            sizes.append(1)
    total, done = float(sum(sizes)), 0.0
    for p, size in zip(paths, sizes):
        name = os.path.basename(p)
        msg = f"Loading {name} ({size / 1e6:,.0f} MB)"
        base = done / total
        span = size / total
        progress(msg, base)
        try:
            def report(f, m=msg, b=base, s=span):
                progress(m, b + s * f * 0.95)
            clouds = [read_cloud(p, progress=report)] if base_only else read_clouds(p, report)
            progress(f"Preparing preview of {name}", base + span * 0.95)
            out += [(c, preview_sample(c)) for c in clouds]
        except Cancelled:
            raise
        except Exception as e:
            errors.append(f"{name}: {e}")
        done += size
    return out, errors


def load_clouds(entries, *, progress):
    """Read the full clouds of existing project scans (for Align, walk-through,
    clean-up). Returns ({index: cloud}, errors)."""
    out, errors = {}, []
    total = max(1, sum(e.points for _, e in entries))
    done = 0
    for i, e in entries:
        msg = f"Loading {e.name}"
        try:
            out[i] = open_scan(e.path, e.index).read(
                progress=lambda f, d=done, n=e.points: progress(msg, (d + f * n) / total))
        except Cancelled:
            raise
        except Exception as ex:
            errors.append(f"{e.name}: {ex}")
        done += e.points
    return out, errors


def sample_missing(entries, *, progress):
    """Preview samples for scans that are not cached, streamed from disk
    (the scans are not loaded). Returns {index: sample}."""
    out = {}
    for k, (i, e) in enumerate(entries):
        progress(f"Reading a preview of {e.name}", k / max(1, len(entries)))
        try:
            out[i] = sample_points(open_scan(e.path, e.index), PREVIEW_POINTS)
        except Cancelled:
            raise
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------


class Document(QObject):
    """Emits ``changed(what)``: "scans", "alignment", "visibility", "project"."""

    changed = Signal(str)
    saved = Signal()

    AUTOSAVE_MS = 1500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.scans: list[Scan] = []
        self.result = None  # AlignResult of the last Align (or rebuilt from the project)
        self.quality: QualityReport | None = None
        self.report_lines: list[str] = []
        self.export_samples: dict[str, np.ndarray] = {}  # merged clouds shown in the view
        self.dirty = False
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(self.AUTOSAVE_MS)
        self._autosave.timeout.connect(self.save)

    # ---- state

    @property
    def base(self) -> Scan | None:
        return self.scans[0] if self.scans and self.scans[0].entry.base else None

    @property
    def aligned(self) -> bool:
        return self.result is not None

    def transforms(self) -> list[np.ndarray]:
        if self.result is not None:
            return [np.asarray(t, float) for t in self.result.transforms]
        return [np.eye(4) for _ in self.scans]

    def previews(self) -> list[np.ndarray]:
        """Per scan, the points drawn in the orbit view (a random prefix of its sample)."""
        if not self.scans:
            return []
        per = PREVIEW_POINTS // len(self.scans)
        return [s.sample[:per] for s in self.scans]

    def origin(self) -> np.ndarray:
        """Draw relative to the reference cloud so large (e.g. UTM) coordinates
        don't lose precision in float32 OpenGL buffers."""
        for s in self.scans:
            if len(s.sample):
                return s.sample[:2000].mean(axis=0)
        for xyz in self.export_samples.values():
            if len(xyz):
                return xyz[:2000].mean(axis=0)
        return np.zeros(3)

    def visible(self) -> list[bool]:
        return [s.entry.visible for s in self.scans]

    def keys(self) -> set[str]:
        return {s.entry.key for s in self.scans}

    def missing_clouds(self, indices=None) -> list[int]:
        idx = range(len(self.scans)) if indices is None else indices
        return [i for i in idx if self.scans[i].cloud is None]

    # ---- changes

    def touch(self, what: str = "project") -> None:
        self.dirty = True
        self.changed.emit(what)
        if self.project.saved:
            self._autosave.start()

    def add_loaded(self, loaded, base: bool = False) -> None:
        """Add ``[(cloud, sample)]`` from :func:`load_files`."""
        if base:
            # the new base replaces any old one, and is not also a plain scan
            keys = {ScanEntry.from_source(c).key for c, _ in loaded}
            self.scans = [s for s in self.scans if not s.entry.base and s.entry.key not in keys]
            self.project.scans = [s.entry for s in self.scans]
        for c, sample in loaded:
            entry = self.project.add(c, base=base)
            if any(s.entry is entry for s in self.scans):
                continue
            scan = Scan(entry, sample, c)
            if base:
                entry.transform, entry.locked = np.eye(4).tolist(), True
                self.scans.insert(0, scan)
            else:
                self.scans.append(scan)
            self.project.save_preview(entry, sample)
        self.project.scans = [s.entry for s in self.scans]
        self.forget_alignment()
        self.touch("scans")

    def remove(self, indices) -> None:
        drop = set(indices)
        self.scans = [s for i, s in enumerate(self.scans) if i not in drop]
        self.project.scans = [s.entry for s in self.scans]
        self.forget_alignment()
        self.touch("scans")

    def clear(self) -> None:
        self.scans = []
        self.project.scans = []
        self.forget_alignment()
        self.touch("scans")

    def set_visible(self, i: int, on: bool) -> None:
        if self.scans[i].entry.visible != on:
            self.scans[i].entry.visible = on
            self.touch("visibility")

    def forget_alignment(self) -> None:
        self.result = None
        self.quality = None
        self.report_lines = []
        self.project.alignment = None
        self.project.quality = None

    def set_alignment(self, result, quality, report_lines) -> None:
        self.result, self.quality, self.report_lines = result, quality, list(report_lines)
        self.project.set_alignment(result, quality, report_lines)
        if quality is not None:
            self.project.save_heatmap(quality.point_error)
        self.touch("alignment")

    def set_setting(self, key: str, value) -> None:
        if self.project.settings.get(key) != value:
            self.project.settings[key] = value
            self.touch("settings")

    # ---- merged / exported clouds

    def export(self, path: str) -> dict | None:
        key = os.path.normcase(os.path.abspath(path))
        return next((e for e in self.project.exports
                     if os.path.normcase(e.get("path", "")) == key), None)

    def visible_exports(self) -> list[np.ndarray]:
        """Preview samples of the merged clouds that are ticked (already in the
        project frame, so they are drawn as they are)."""
        return [self.export_samples[e["path"]] for e in self.project.exports
                if e.get("visible") and e["path"] in self.export_samples]

    def set_export_visible(self, path: str, on: bool) -> None:
        e = self.export(path)
        if e is not None and bool(e.get("visible")) != on:
            e["visible"] = on
            self.touch("visibility")

    def set_export_sample(self, path: str, xyz: np.ndarray) -> None:
        self.export_samples[path] = xyz
        e = self.export(path)
        if e is not None:
            self.project.save_export_preview(e, xyz)
        self.changed.emit("visibility")

    def cached_export_sample(self, path: str) -> np.ndarray | None:
        if path in self.export_samples:
            return self.export_samples[path]
        e = self.export(path)
        xyz = self.project.load_export_preview(e) if e is not None else None
        if xyz is not None:
            self.export_samples[path] = xyz
        return xyz

    def remove_export(self, path: str) -> None:
        self.project.remove_export(path)
        self.export_samples.pop(path, None)
        self.touch("exports")

    # ---- files

    def save(self) -> bool:
        self._autosave.stop()
        if not self.project.saved:
            return False
        try:
            self.project.save()
        except (OSError, ProjectError):
            return False
        self.dirty = False
        self.saved.emit()
        return True

    def save_as(self, path: str) -> None:
        self.project = self.project.save_as(path)
        # re-point the run-time scans at the copy's entries
        for s, e in zip(self.scans, self.project.scans):
            s.entry = e
            if len(s.sample):
                self.project.save_preview(e, s.sample)
        if self.quality is not None:
            self.project.save_heatmap(self.quality.point_error)
        add_recent(self.project.folder)
        self.save()
        self.changed.emit("project")

    def new(self) -> None:
        self._autosave.stop()
        self.project = Project()
        self.scans = []
        self.export_samples = {}
        self.forget_alignment()
        self.dirty = False
        self.changed.emit("project")

    def open(self, path: str) -> list[ScanEntry]:
        """Open a project: the view comes back from the cache straight away;
        clouds are loaded later, when something needs them. Returns the
        entries whose cached preview is missing (to sample in the background)."""
        project = Project.open(path)
        self._autosave.stop()
        self.project = project
        self.scans = [Scan(e, project.load_preview(e), None) for e in project.scans]
        self.export_samples = {}
        for e in project.exports:
            if e.get("visible") and (xyz := project.load_export_preview(e)) is not None:
                self.export_samples[e["path"]] = xyz
        self.result, self.quality, self.report_lines = None, None, []
        if project.alignment and project.alignment_matches_scans():
            self.result = alignment_from_dict(project.alignment, project.transforms())
            q = project.quality
            if q and q.get("keys") == [s.entry.key for s in self.scans]:
                try:
                    self.quality = QualityReport.from_dict(q, project.load_heatmap())
                except (KeyError, TypeError, ValueError):
                    self.quality = None
                self.report_lines = list(q.get("report", []))
        else:
            project.alignment = project.quality = None
        self.dirty = False
        add_recent(project.folder)
        self.changed.emit("project")
        return [s.entry for s in self.scans if not len(s.sample)]

    def set_clouds(self, clouds: dict) -> None:
        for i, c in clouds.items():
            if i < len(self.scans):
                self.scans[i].cloud = c

    def set_samples(self, samples: dict) -> None:
        for i, xyz in samples.items():
            if i < len(self.scans):
                self.scans[i].sample = xyz
                self.project.save_preview(self.scans[i].entry, xyz)
        self.changed.emit("scans")

    def relocate(self, i: int, path: str) -> None:
        """Point a missing scan at its new file (neighbours are found too)."""
        self.project.relocate(self.scans[i].entry, path)
        self.touch("scans")



def sample_file(path, *, progress):
    """Preview sample of a merged cloud, streamed (it can be huge)."""
    progress(f"Reading a preview of {os.path.basename(path)}", 0.0)
    return path, sample_points(open_scan(path), PREVIEW_POINTS)
