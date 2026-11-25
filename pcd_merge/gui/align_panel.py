"""Align mode, right-hand panel: everything for getting scans into one frame
and exporting the result (today's features, moved out of the old left panel).

Align, Retry, the quality report, clean-up with its live preview (moves to
Edit mode in 0.10), merge / export, save transforms, and adding new scans to
an earlier merged cloud.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .. import retry
from ..cleanup import CleanupOptions, clean_masks
from ..exporters import FORMATS, file_filter
from ..files import save_merged
from ..quality import assess, fmt_distance
from ..readers import input_filter
from ..registration import AlignSettings, align
from .common import GRADE_COLOURS, HINT_STYLE
from .save_dialog import ask_save_path
from .workers import QuietWorker


# --------------------------------------------------------------------------
# background jobs


def _ext(path: str) -> str:
    """The file's format extension (".copc.laz" counts as one)."""
    return ".copc.laz" if path.lower().endswith(".copc.laz") else \
        os.path.splitext(path)[1].lower()


def _stem(path: str) -> str:
    name = os.path.basename(path)
    return name[:-len(_ext(path))] if _ext(path) else name


def align_clouds(clouds, settings, previews, quality_ref=None, *, progress):
    """Align, then grade the result.

    ``quality_ref`` is ``(clouds, voxel)`` from the first Align. Retries use
    coarser voxel sizes, and the grade limits scale with the voxel size, so
    retries are graded against the first Align's clouds and voxel size to
    keep grades comparable (and Retry from looking better just by being
    coarser).
    """
    # Bound methods: each full-size coordinate array is made only when needed,
    # one at a time.
    xyz = [c.finite_xyz for c in clouds]
    result = align(xyz, settings, lambda msg, frac: progress(msg, frac * 0.9))
    progress("Measuring alignment quality", 0.92)
    # Reuse the downsampled clouds from alignment instead of the full ones.
    q_xyz, q_voxel = quality_ref or (result.fine or xyz, result.voxel_size)
    quality = assess(q_xyz, result.transforms, q_voxel, preview_xyz=previews)
    return result, quality


def merge_and_save(sources, names, transforms, path, include, add_source, compressed, spacing,
                   cleanup, base_path, voxel_size, edits=None, keys=None, part=None, *,
                   progress):
    n, partial, notes, stats = save_merged(
        sources, transforms, path, include, add_source=add_source, compressed=compressed,
        base_path=base_path, voxel_size=voxel_size, min_spacing=spacing, cleanup=cleanup,
        progress=progress, edits=edits, keys=keys, part=part)
    if stats is not None:
        notes = stats.lines([nm for nm, inc in zip(names, include) if inc]) + notes
    if edits:
        notes.append("the points deleted in Edit mode were left out")
    if part is not None:
        notes.append(f"only the part {part.name!r} was saved")
    return path, n, partial, notes, include


def cleanup_preview(origins, counts, transforms, previews, opts):
    """Run the chosen clean-up on the preview points (a random subset of each
    scan, already in memory), so ticking an option updates the view quickly.

    Returns per scan a mask of the preview points that would be kept, the
    stats and the counts, scaled up to an estimate for the full scans."""
    xyz, weights = [], []
    for n, t, p in zip(counts, transforms, previews):
        t = np.asarray(t, dtype=np.float64)
        xyz.append(p @ t[:3, :3].T + t[:3, 3])
        weights.append(n / max(len(p), 1))
    keep, stats = clean_masks(xyz, origins, opts, weights=weights)
    removed = [int(round((~k).sum() * w)) for k, w in zip(keep, weights)]
    return keep, stats, removed, weights


# --------------------------------------------------------------------------


def _spin(decimals, hi, step, special, tip):
    s = QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(0.0, hi)
    s.setSingleStep(step)
    s.setSpecialValueText(special)
    s.setToolTip(tip)
    return s


class AlignPanel(QWidget):
    status = Signal(str)

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.doc = win.doc
        self._reset_retries()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)
        body = self.body = QWidget()
        scroll.setWidget(body)
        lv = QVBoxLayout(body)

        # ---- alignment
        box = QGroupBox("Alignment")
        form = QFormLayout(box)
        self.voxel = _spin(3, 100.0, 0.05, "Automatic",
                           "Grid size used to thin the scans for alignment only (same units as "
                           "the scans, usually metres). Larger is faster and more robust, smaller "
                           "is more precise. Merging always uses every original point.")
        self.voxel.valueChanged.connect(lambda v: self.doc.set_setting("voxel", v or None))
        form.addRow("Voxel size:", self.voxel)
        lv.addWidget(box)

        self.b_align = QPushButton("Align")
        self.b_align.setProperty("primary", True)
        self.b_align.clicked.connect(lambda: self.start_align(new_seed=False))
        lv.addWidget(self.b_align)

        self.grade_label = QLabel()
        self.grade_label.setToolTip(
            "Good: the scans agree closely where they overlap. Fair: usable, but check the "
            "heat map for red areas. Poor: likely misaligned or barely overlapping.")
        self.grade_label.hide()
        lv.addWidget(self.grade_label)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Alignment results will appear here.")
        self.report.setMinimumHeight(160)
        font = self.report.font()
        font.setFamily("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.report.setFont(font)
        lv.addWidget(self.report)

        row = QHBoxLayout()
        self.b_retry = QPushButton("Retry")
        self.b_retry.setToolTip(
            "Try again with a coarser voxel size and more search attempts, which helps with "
            "repetitive or low-detail scenes. Each press goes coarser (1.5×, 2×, 3× the first "
            "try). The best result so far is kept.")
        self.b_retry.clicked.connect(self.retry)
        self.b_save_tf = QPushButton("Save transforms…")
        self.b_save_tf.setToolTip("Write the 4x4 matrices and the quality report as JSON.")
        self.b_save_tf.clicked.connect(self.save_transforms)
        row.addWidget(self.b_retry)
        row.addWidget(self.b_save_tf)
        lv.addLayout(row)

        self.b_base = QPushButton("Add to an existing merged cloud…")
        self.b_base.setToolTip(
            "Pick a merged cloud you saved earlier. It stays exactly where it is and the "
            "other scans in the project are aligned to it and appended when you merge.")
        self.b_base.clicked.connect(self._base_dialog)
        lv.addWidget(self.b_base)

        from .manual_align import ManualAlignBox

        self.manual = ManualAlignBox(self)
        lv.addWidget(self.manual)

        # ---- clean-up: built here (the merge uses it) but shown in Edit mode
        box = self.cleanup_box = QGroupBox("Automatic clean-up (applied when saving)")
        cl = QVBoxLayout(box)
        self.cb_dedup = QCheckBox("Remove duplicates where scans overlap")
        self.cb_dedup.setToolTip(
            "Where two scans cover the same spot, keep only the points of the scan that "
            "measured it most densely (usually the closest, most accurate one). Nothing is "
            "averaged or moved, and areas seen by one scan are left alone.")
        self.dedup_cell = _spin(3, 10.0, 0.05, "Automatic",
                                "Size of the cubes in which one scan is chosen (scan units, "
                                "usually metres). Automatic is a few times the point spacing.")
        self.dedup_cell.setEnabled(False)
        self.cb_dedup.toggled.connect(self.dedup_cell.setEnabled)
        cl.addWidget(self.cb_dedup)
        drow = QHBoxLayout()
        drow.addSpacing(22)
        drow.addWidget(QLabel("Cell size:"))
        drow.addWidget(self.dedup_cell)
        drow.addStretch(1)
        cl.addLayout(drow)
        self.cb_denoise = QCheckBox("Remove isolated noise (dust, rain, strays)")
        self.cb_denoise.setToolTip(
            "Drops points that are far from all of their neighbours compared with the rest "
            "of the scan. Conservative: real surfaces are kept.")
        cl.addWidget(self.cb_denoise)
        self.cb_moving = QCheckBox("Remove moving objects (people, cars)")
        self.cb_moving.setToolTip(
            "Drops points from one scan where another scan looked straight through that spot "
            "and saw something further away. Objects hidden from the other scan are kept. "
            "Works on scans saved in their scanner's own frame (raw scans).")
        cl.addWidget(self.cb_moving)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Points to remove:"))
        self.removed_mode = QComboBox()
        self.removed_mode.addItems(["Highlight in white", "Hide"])
        self.removed_mode.setToolTip(
            "How the aligned view shows the points the ticked clean-up would remove. "
            "The preview uses a sample of each scan, so it is a close guide; the merge "
            "itself works on every point.")
        self.removed_mode.currentIndexChanged.connect(self._removed_mode_changed)
        prow.addWidget(self.removed_mode)
        prow.addStretch(1)
        cl.addLayout(prow)
        self.cleanup_info = QLabel("")
        self.cleanup_info.setWordWrap(True)
        self.cleanup_info.setStyleSheet(HINT_STYLE)
        cl.addWidget(self.cleanup_info)

        # clean-up preview: recomputed shortly after any option changes
        self._cleanup_gen = 0
        self._cleanup_thread = None
        self._cleanup_pending = False
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.setInterval(250)
        self._cleanup_timer.timeout.connect(self._start_cleanup_preview)
        for cb in (self.cb_dedup, self.cb_denoise, self.cb_moving):
            cb.toggled.connect(self.cleanup_changed)
        self.dedup_cell.valueChanged.connect(self.cleanup_changed)

        # ---- export
        box = QGroupBox("Merge and export")
        ev = QVBoxLayout(box)
        self.cb_source = QCheckBox("Add a 'source' field (scan of each point)")
        self.cb_compress = QCheckBox("Compress PCD output")
        self.cb_compress.setToolTip("Needs the whole merged cloud in memory; leave off for "
                                    "very large sites.")
        ev.addWidget(self.cb_source)
        ev.addWidget(self.cb_compress)
        self.spacing = _spin(3, 10.0, 0.01, "Full resolution",
                             "Thin the merged cloud so no two points are closer than this "
                             "distance (in the scans' units, e.g. 0.02 = 2 cm). Original points "
                             "are kept, not averaged, so intensity and other fields stay exact. "
                             "Leave at Full resolution to save every point.")
        f2 = QFormLayout()
        f2.addRow("Min. point spacing:", self.spacing)
        ev.addLayout(f2)
        self.cleanup_summary = QLabel()
        self.cleanup_summary.setWordWrap(True)
        self.cleanup_summary.setStyleSheet(HINT_STYLE)
        ev.addWidget(self.cleanup_summary)
        self.b_merge = QPushButton("Accept and merge…")
        self.b_merge.setProperty("primary", True)
        self.b_merge.clicked.connect(self.merge)
        ev.addWidget(self.b_merge)
        lv.addWidget(box)
        lv.addStretch(1)

        for w in (self.cb_dedup, self.cb_denoise, self.cb_moving, self.cb_source,
                  self.cb_compress):
            w.toggled.connect(self._store_settings)
        for w in (self.dedup_cell, self.spacing):
            w.valueChanged.connect(self._store_settings)
        self._loading_settings = False

    # ---------------------------------------------------------------- settings

    def load_settings(self):
        """Fill the controls from the project (after New / Open)."""
        s = self.doc.project.settings
        self._loading_settings = True
        try:
            self.voxel.setValue(s.get("voxel") or 0.0)
            c = s.get("cleanup", {})
            self.cb_dedup.setChecked(bool(c.get("dedup")))
            self.dedup_cell.setValue(c.get("dedup_cell") or 0.0)
            self.cb_denoise.setChecked(bool(c.get("denoise")))
            self.cb_moving.setChecked(bool(c.get("remove_moving")))
            e = s.get("export", {})
            self.cb_source.setChecked(bool(e.get("source_field")))
            self.cb_compress.setChecked(bool(e.get("compressed")))
            self.spacing.setValue(e.get("min_spacing") or 0.0)
        finally:
            self._loading_settings = False
        self._reset_retries()
        self.show_result()

    def _store_settings(self, *_):
        if self._loading_settings:
            return
        self.doc.set_setting("cleanup", {"dedup": self.cb_dedup.isChecked(),
                                         "dedup_cell": self.dedup_cell.value() or None,
                                         "denoise": self.cb_denoise.isChecked(),
                                         "remove_moving": self.cb_moving.isChecked()})
        export = dict(self.doc.project.settings.get("export", {}))
        export.update(source_field=self.cb_source.isChecked(),
                      compressed=self.cb_compress.isChecked(),
                      min_spacing=self.spacing.value() or None)
        self.doc.set_setting("export", export)

    # ---------------------------------------------------------------- state

    def update_buttons(self, busy: bool):
        n = len(self.doc.scans)
        aligned = self.doc.result is not None
        self.b_align.setEnabled(not busy and n >= 2)
        self.b_retry.setEnabled(not busy and n >= 2 and aligned)
        self.b_merge.setEnabled(not busy and aligned)
        self.b_save_tf.setEnabled(not busy and aligned)
        self.b_base.setEnabled(not busy)
        self.manual.update_buttons(busy)

    def scans_changed(self):
        """Scans added or removed: earlier results no longer apply."""
        self._reset_retries()
        self.manual.scans_changed()
        self.show_result()
        self.cleanup_changed()

    def _reset_retries(self):
        self._retry_press = 0  # how many times Retry was pressed since Align
        self._retry_base = None  # (settings, voxel, quality clouds) of the first Align
        self._best = None  # (result, quality, label) best so far
        self._last_label = ""
        if hasattr(self, "b_retry"):
            self.b_retry.setText("Retry")

    def show_result(self):
        """Show the document's alignment (or nothing) in the grade and report."""
        q = self.doc.quality
        if self.doc.result is None:
            self.grade_label.hide()
            self.report.clear()
            return
        if q is not None:
            grade = q.grade
            self.grade_label.setText(f"Alignment quality: {grade}")
            self.grade_label.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {GRADE_COLOURS[grade]};")
            self.grade_label.show()
        else:
            self.grade_label.hide()
        self.report.setPlainText("\n".join(self.doc.report_lines))

    # ---------------------------------------------------------------- align

    def _base_settings(self, seed: int) -> AlignSettings:
        return AlignSettings(voxel_size=self.voxel.value() or None, seed=seed,
                             anchor=self.doc.base is not None)

    def start_align(self, new_seed: bool = False):
        """Align (a fresh start: forgets earlier retries)."""
        if len(self.doc.scans) < 2:
            return
        self.win.with_clouds(lambda: self._align_now(new_seed))

    def _align_now(self, new_seed):
        self._reset_retries()
        seed = random.randrange(1 << 30) if new_seed else 0
        self._last_label = "first try"
        clouds = [s.cloud for s in self.doc.scans]
        self.win.tasks.run(align_clouds, (clouds, self._base_settings(seed), self.doc.previews()),
                           self._aligned, "Aligning…")

    def retry(self):
        if len(self.doc.scans) < 2:
            return
        if self._retry_base is None:  # no Align yet in this session of retries
            self.start_align(new_seed=True)
            return
        self.win.with_clouds(self._retry_now)

    def _retry_now(self):
        base, base_voxel, q_ref = self._retry_base
        self._retry_press += 1
        settings = retry.settings_for(base, base_voxel, self._retry_press,
                                      random.randrange(1 << 30))
        lv = retry.level(self._retry_press)
        self._last_label = f"retry {self._retry_press}: {lv.describe(settings.voxel_size)}"
        clouds = [s.cloud for s in self.doc.scans]
        self.win.tasks.run(align_clouds, (clouds, settings, self.doc.previews(), q_ref),
                           self._aligned,
                           f"Retrying with a coarser voxel size ({settings.voxel_size:.3g})…")

    def _aligned(self, payload):
        result, quality = payload
        label = self._last_label
        if self._retry_base is None:  # first Align: retries escalate from here
            self._retry_base = (self._base_settings(0), result.voxel_size,
                                (result.fine, result.voxel_size) if result.fine else None)
        note = ""
        if retry.is_better((result, quality), self._best and self._best[:2]):
            self._best = (result, quality, label)
            if self._retry_press:
                note = f"Retry {self._retry_press} improved the result."
        else:
            kept = self._best[2]
            note = (f"Retry {self._retry_press} gave {quality.grade} "
                    f"({label}); keeping the better result from the {kept}.")
            result, quality, label = self._best
        nxt = retry.level(self._retry_press + 1)
        self.b_retry.setText(f"Retry ({nxt.voxel_factor:g}× coarser)")
        grade = quality.grade
        lines = quality.lines([s.name for s in self.doc.scans])[2:]  # grade is shown above
        lines += ["", f"Voxel size used: {result.voxel_size:.3g}. Distances are to the other "
                  "scan's surface; points more than "
                  f"{fmt_distance(quality.threshold)} away count as not overlapping."]
        warns = result.warnings()
        if warns:
            lines += ["", "Warnings:"] + ["  " + w for w in warns]
        lines += ["", f"Shown: {label}."]
        if note:
            lines += [note]
        if warns or grade != "Good":
            lines += ["", "Try Retry: each press uses a coarser voxel size and more search "
                      "attempts, and the best result is kept."]
        self.doc.set_alignment(result, quality, lines)
        self.show_result()
        self.win.view.show_aligned(heat=grade != "Good")
        self.cleanup_changed()
        if warns or grade != "Good":
            self.status.emit(f"Aligned, quality {grade}. Check the error heat map before merging.")
        else:
            self.status.emit("Aligned. Check the view, then Accept and merge (or Retry).")

    # ---------------------------------------------------------------- merge

    def cleanup_options(self) -> CleanupOptions:
        return CleanupOptions(dedup=self.cb_dedup.isChecked(),
                              dedup_cell=self.dedup_cell.value() or None,
                              denoise=self.cb_denoise.isChecked(),
                              remove_moving=self.cb_moving.isChecked())

    def merge(self, part=None, fmt: str | None = None, revit: bool = False):
        """Merge and save. Points deleted in Edit mode are left out; ``part``
        saves only that part of a split site. One scan on its own can be
        saved too (cleaned up in Edit), with no alignment. ``fmt`` (such as
        ".copc.laz") is the format offered first instead of the last one used.
        ``revit`` saves an E57 for ReCap / Revit and then shows how to bring
        it into Revit."""
        if revit:
            fmt = ".e57"
        doc = self.doc
        result = doc.result
        if result is None and len(doc.scans) != 1:
            return
        include = [True] * len(doc.scans)
        if result is None:
            pass
        elif not result.ok:
            bad = [s.name for s, ok in zip(doc.scans, result.connected) if not ok]
            box = QMessageBox(self)
            box.setWindowTitle("PCD Merge")
            box.setIcon(QMessageBox.Question)
            shown = ", ".join(bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
            box.setText(f"{len(bad)} scan(s) could not be aligned: {shown}.")
            box.setInformativeText("Leave them out of the merged cloud, or merge them in "
                                   "their original position?")
            b_skip = box.addButton("Leave them out", QMessageBox.AcceptRole)
            b_all = box.addButton("Merge anyway", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_skip)
            box.exec()
            if box.clickedButton() is b_skip:
                include = list(result.connected)
                if sum(include) < 2:
                    QMessageBox.warning(self, "PCD Merge", "Fewer than two scans aligned; "
                                        "there is nothing to merge.")
                    return
            elif box.clickedButton() is not b_all:
                return
        elif doc.quality is not None and doc.quality.grade == "Poor":
            ans = QMessageBox.question(
                self, "PCD Merge",
                "The alignment quality is Poor: the scans disagree or barely overlap. "
                "Merge anyway?")
            if ans != QMessageBox.Yes:
                return
        export = doc.project.settings.get("export", {})
        ext = fmt or export.get("format", ".pcd")
        base = doc.base
        suggested = None
        if base is not None and fmt is None:
            ext = _ext(base.entry.path)  # keep adding to the same file
            suggested = base.entry.path
        stem = doc.project.name if doc.project.saved else (
            "merged" if len(doc.scans) > 1 else _stem(doc.scans[0].entry.path) + " cleaned")
        if part is not None:
            stem += " - " + part.name
        filters = "E57 for ReCap / Revit (*.e57)" if revit else file_filter()
        chosen = next((f for f in filters.split(";;") if f.endswith(f"(*{ext})")), "")
        title = ("Export for Revit" if revit else
                 "Save viewing file" if fmt == ".copc.laz" else
                 f"Save part: {part.name}" if part is not None else
                 "Save merged cloud" if len(doc.scans) > 1 else "Save cleaned cloud")
        path, selected = ask_save_path(self, title, stem + ext, filters, chosen, doc, suggested)
        if not path:
            return
        typed = _ext(path)
        if revit and typed != ".e57":
            path += ".e57"
        elif typed in FORMATS:
            ext = typed
        else:  # no (known) extension typed: use the format picked in the dialog
            ext = next((e for e in FORMATS if selected.endswith(f"(*{e})")), ".pcd")
            path += ext
        if base is None and fmt is None:
            export["format"] = ext
            doc.set_setting("export", dict(export))
        compressed = self.cb_compress.isChecked() and ext == ".pcd"
        if compressed:
            try:
                import lzf  # noqa: F401
            except ImportError:
                QMessageBox.warning(self, "PCD Merge", "Compressed output needs python-lzf; "
                                    "writing uncompressed binary instead.")
                compressed = False
        opts = self.cleanup_options()
        spacing = self.spacing.value() or None

        edits = doc.edit_snapshot()
        keys = [s.entry.key for s in doc.scans]

        self._for_revit = revit

        def go():
            sources = [s.source() for s in doc.scans]
            self.win.tasks.run(
                merge_and_save,
                (sources, [s.name for s in doc.scans], doc.transforms(), path, include,
                 self.cb_source.isChecked(), compressed, spacing, opts,
                 base.entry.path if base is not None else None,
                 result.voxel_size if result is not None else None, edits, keys, part),
                self._merged, "Merging…")

        if opts.any or spacing or compressed:
            self.win.with_clouds(go)  # these need every point in memory anyway
        else:
            go()  # streamed: scans that are not loaded are read straight from disk

    def _merged(self, payload):
        path, n, partial, notes, include = payload
        doc = self.doc
        doc.project.record_export(path, n, notes)
        doc.export_samples.pop(os.path.abspath(path), None)  # the file is new
        # Show the result instead of the scans it was made from.
        for s in doc.scans:
            s.entry.visible = False
        doc.touch("exports")
        doc.touch("visibility")
        msg = f"Saved {n:,} points to {path}"
        left_out = len(include) - sum(include)
        if left_out:
            msg += f"\n\n{left_out} scan(s) that could not be aligned were left out."
        if partial:
            msg += ("\n\nThese fields were missing from some scans and were filled with 0: "
                    + ", ".join(partial))
        for note in notes:
            msg += "\n\n" + note[0].upper() + note[1:] + "."
        msg += ("\n\nThe list of scans in it is saved next to it as "
                f"{os.path.basename(path)}.json, so scans already in it are skipped if "
                "you add new scans to it later.")
        msg += ("\n\nThe merged cloud is now listed under Merged clouds and shown in the "
                "view; the scans are hidden. Tick or untick any of them to compare.")
        if getattr(self, "_for_revit", False):
            from .revit_help import show_revit_steps

            self._for_revit = False
            show_revit_steps(self, path, [n for n in notes if not n.startswith("E57 has no")])
        else:
            QMessageBox.information(self, "PCD Merge", msg)
        self.win.show_export(os.path.abspath(path), True)

    def save_transforms(self):
        doc = self.doc
        if doc.result is None:
            return
        path, _ = ask_save_path(self, "Save transforms", "transforms.json", "JSON (*.json)",
                                doc=doc)
        if not path:
            return
        keys = [f"{s.entry.path}#{s.entry.index}" if s.entry.scans_in_file > 1 else s.entry.path
                for s in doc.scans]
        data = {
            "reference": keys[0],
            "voxel_size": doc.result.voxel_size,
            "transforms": {k: np.asarray(t).tolist() for k, t in zip(keys, doc.transforms())},
        }
        if doc.quality is not None:
            data["quality"] = doc.quality.to_dict()
            data["quality"]["files"] = keys
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status.emit(f"Transforms saved to {os.path.basename(path)}.")

    def _base_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Existing merged cloud", "", input_filter())
        if path:
            self.win.load_paths([path], base=True)

    # ---------------------------------------------------------------- clean-up preview

    def _removed_mode_changed(self, i):
        self.win.view.removed_hidden = i == 1
        self.win.view.redraw()

    def update_cleanup_summary(self):
        """The line under Merge saying what the merge will leave out."""
        opts = self.cleanup_options()
        chosen = [name for name, on in (("duplicates", opts.dedup), ("noise", opts.denoise),
                                        ("moving objects", opts.remove_moving)) if on]
        n_edits = len(self.doc.edits.point_edits)
        self.cleanup_summary.setText(
            ("Clean-up when merging: removes " + ", ".join(chosen) if chosen else
             "No automatic clean-up")
            + (f"; {n_edits} edit(s) leave points out" if n_edits else "")
            + ". Both are set in Edit mode.")

    def cleanup_changed(self, *_):
        """Options, scans or alignment changed: drop the old preview and
        schedule a new one (debounced, so dragging the cell size stays smooth)."""
        self._cleanup_gen += 1
        self.win.view.set_removed(None)
        opts = self.cleanup_options()
        self.update_cleanup_summary()
        if not opts.any:
            self.cleanup_info.setText("")
        elif self.doc.result is None:
            self.cleanup_info.setText("Align first to see the effect in the view.")
        else:
            self.cleanup_info.setText("Updating preview…")
            self._cleanup_timer.start()

    def _start_cleanup_preview(self):
        if self._cleanup_thread is not None:
            self._cleanup_pending = True  # run again when the current one ends
            return
        opts = self.cleanup_options()
        doc = self.doc
        if not opts.any or doc.result is None or not doc.scans:
            return
        self._cleanup_pending = False
        tfs = doc.transforms()
        origins = [s.sensor_origin(t) for s, t in zip(doc.scans, tfs)]
        counts = [s.points for s in doc.scans]
        thread = QThread()
        worker = QuietWorker(self._cleanup_gen, cleanup_preview, origins, counts, tfs,
                             doc.previews(), opts)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._cleanup_done)
        worker.failed.connect(self._cleanup_failed)
        self._cleanup_thread, self._cleanup_worker = thread, worker
        thread.start()

    def _cleanup_thread_ended(self):
        self._cleanup_thread.quit()
        self._cleanup_thread.wait()
        self._cleanup_thread = self._cleanup_worker = None
        if self._cleanup_pending:
            self._start_cleanup_preview()

    def _cleanup_done(self, gen, payload):
        self._cleanup_thread_ended()
        if gen != self._cleanup_gen:
            return  # options changed meanwhile; a newer preview is on its way
        keep, stats, removed, weights = payload
        total = sum(s.points for s in self.doc.scans)
        lines = []
        if sum(removed):
            parts = [(label, sum(c * w for c, w in zip(counts, weights)))
                     for label, counts in (("noise", stats.noise),
                                           ("moving objects", stats.moving),
                                           ("duplicates", stats.duplicate)) if counts]
            detail = ", ".join(f"{label} {int(n):,}" for label, n in parts)
            lines.append(f"Preview: about {sum(removed):,} of {total:,} points "
                         f"({sum(removed) / max(total, 1):.0%}) would be removed ({detail}).")
        else:
            lines.append("Preview: no points would be removed.")
        if stats.no_sensor_position:
            lines.append("Moving objects: no scanner position for "
                         + ", ".join(self.doc.scans[i].name for i in stats.no_sensor_position)
                         + ", so it is not used as evidence.")
        self.cleanup_info.setText("\n".join(lines))
        self.win.view.set_removed(keep)

    def _cleanup_failed(self, gen, message):
        self._cleanup_thread_ended()
        if gen == self._cleanup_gen:
            self.cleanup_info.setText("Preview failed: " + message)

    def shutdown(self):
        if self._cleanup_thread is not None:  # let a running preview finish first
            self._cleanup_thread.quit()
            self._cleanup_thread.wait()
