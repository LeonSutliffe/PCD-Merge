"""Desktop app: drop PCD files in, align them, preview, then merge and save."""

from __future__ import annotations

import json
import os
import random
import sys
import traceback

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .cleanup import CleanupOptions
from .exporters import FORMATS, file_filter, write_cloud
from .merge import merge_clouds_with_stats
from .pcd_io import PointCloud, read_pcd
from .registration import AlignResult, AlignSettings, align
from .thinning import thin

try:
    import pyqtgraph.opengl as gl

    HAVE_GL = True
except Exception:  # pragma: no cover - depends on the machine's OpenGL
    HAVE_GL = False

PREVIEW_POINTS = 400_000  # total points drawn in the 3D view
PALETTE = [
    (0.95, 0.35, 0.25), (0.25, 0.65, 0.95), (0.35, 0.85, 0.35), (0.95, 0.80, 0.20),
    (0.75, 0.40, 0.90), (0.20, 0.85, 0.80), (0.95, 0.55, 0.75), (0.65, 0.65, 0.65),
]


# --------------------------------------------------------------------------
# Background work


class Cancelled(Exception):
    """Raised inside a worker when the user presses Cancel."""


class _Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.cancel_requested = False
        self._last = ("", -1.0)

    def _progress(self, msg: str, frac: float):
        # Called from the work itself; this is where a Cancel takes effect.
        if self.cancel_requested:
            raise Cancelled()
        # Skip updates the bar would not show, so huge files don't flood the UI.
        if msg != self._last[0] or frac - self._last[1] >= 0.005:
            self._last = (msg, frac)
            self.progress.emit(msg, frac)

    def run(self):
        try:
            result = self.fn(*self.args, progress=self._progress)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # report every failure to the UI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(result)


def _preview_sample(cloud: PointCloud, n: int = PREVIEW_POINTS) -> np.ndarray:
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


def _load_files(paths, progress):
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
            c = read_pcd(p, progress=lambda f, m=msg, b=base, s=span: progress(m, b + s * f * 0.95))
            progress(f"Preparing preview of {name}", base + span * 0.95)
            out.append((c, _preview_sample(c)))
        except Cancelled:
            raise
        except Exception as e:
            errors.append(f"{name}: {e}")
        done += size
    return out, errors


def _align_clouds(clouds, settings, progress):
    # Pass bound methods so only one full-size coordinate array exists at a time.
    return align([c.finite_xyz for c in clouds], settings, progress)


def _merge_and_save(clouds, transforms, path, add_source, compressed, spacing, cleanup,
                    progress):
    def merge_progress(msg, frac):
        progress(msg, 0.5 * frac)

    merged, partial, stats = merge_clouds_with_stats(
        clouds, transforms, add_source_field=add_source, cleanup=cleanup,
        progress=merge_progress)
    notes = stats.lines([c.name for c in clouds]) if stats is not None else []
    if spacing:
        progress(f"Thinning {len(merged):,} points to {spacing:g} spacing", 0.5)
        before = len(merged)
        merged = thin(merged, spacing)
        notes.append(f"thinned to a minimum spacing of {spacing:g}: kept {len(merged):,} "
                     f"of {before:,} points ({len(merged) / max(before, 1):.0%})")
    msg = f"Writing {len(merged):,} points to {os.path.basename(path)}"
    progress(msg, 0.6)
    notes += write_cloud(path, merged, compressed=compressed,
                         progress=lambda f: progress(msg, 0.6 + 0.4 * f))
    return path, len(merged), partial, notes


# --------------------------------------------------------------------------
# Main window


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PCD Merge {__version__}")
        self.resize(1300, 820)
        self.setAcceptDrops(True)

        self.clouds: list[PointCloud] = []
        self.previews: list[np.ndarray] = []  # subsampled xyz per cloud, as drawn
        self._samples: list[np.ndarray] = []  # random preview sample per cloud
        self.result: AlignResult | None = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._items = []
        self._origin = np.zeros(3)

        # ---- left panel
        left = QWidget()
        lv = QVBoxLayout(left)

        self.drop_hint = QLabel("Drag and drop two or more .pcd files here\n(or use Add files…)")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; padding: 18px; color: #666; }"
        )
        lv.addWidget(self.drop_hint)

        self.file_list = QListWidget()
        self.file_list.itemChanged.connect(self._redraw)
        lv.addWidget(self.file_list, 1)

        row = QHBoxLayout()
        b_add = QPushButton("Add files…")
        b_add.clicked.connect(self._add_dialog)
        self.b_remove = QPushButton("Remove")
        self.b_remove.clicked.connect(self._remove_selected)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear)
        for b in (b_add, self.b_remove, b_clear):
            row.addWidget(b)
        lv.addLayout(row)

        box = QGroupBox("Alignment")
        form = QFormLayout(box)
        self.voxel = QDoubleSpinBox()
        self.voxel.setDecimals(3)
        self.voxel.setRange(0.0, 100.0)
        self.voxel.setSingleStep(0.05)
        self.voxel.setSpecialValueText("Automatic")
        self.voxel.setToolTip(
            "Grid size used to thin the clouds for alignment only (same units as the "
            "scans, usually metres). Larger is faster and more robust, smaller is more precise. "
            "Merging always uses every original point."
        )
        form.addRow("Voxel size:", self.voxel)
        lv.addWidget(box)

        self.b_align = QPushButton("Align")
        self.b_align.setStyleSheet("font-weight: bold; padding: 8px;")
        self.b_align.clicked.connect(lambda: self._start_align(new_seed=False))
        lv.addWidget(self.b_align)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Alignment results will appear here.")
        self.report.setMaximumHeight(190)
        lv.addWidget(self.report)

        row = QHBoxLayout()
        self.b_retry = QPushButton("Retry")
        self.b_retry.setToolTip("Run the alignment again with a different random start.")
        self.b_retry.clicked.connect(lambda: self._start_align(new_seed=True))
        self.b_merge = QPushButton("Accept and merge…")
        self.b_merge.setStyleSheet("font-weight: bold; padding: 6px;")
        self.b_merge.clicked.connect(self._merge)
        row.addWidget(self.b_retry)
        row.addWidget(self.b_merge, 1)
        lv.addLayout(row)

        box = QGroupBox("Clean-up while merging (optional)")
        cl = QVBoxLayout(box)
        self.cb_dedup = QCheckBox("Remove duplicate points where scans overlap")
        self.cb_dedup.setToolTip(
            "Where two scans cover the same spot, keep only the points of the scan that "
            "measured it most densely (usually the closest, most accurate one). Nothing is "
            "averaged or moved, and areas seen by one scan are left alone.")
        self.dedup_cell = QDoubleSpinBox()
        self.dedup_cell.setDecimals(3)
        self.dedup_cell.setRange(0.0, 10.0)
        self.dedup_cell.setSingleStep(0.05)
        self.dedup_cell.setSpecialValueText("Automatic")
        self.dedup_cell.setToolTip(
            "Size of the cubes in which one scan is chosen (scan units, usually metres). "
            "Automatic is a few times the point spacing.")
        self.dedup_cell.setEnabled(False)
        self.cb_dedup.toggled.connect(self.dedup_cell.setEnabled)
        drow = QHBoxLayout()
        drow.addWidget(self.cb_dedup, 1)
        drow.addWidget(QLabel("cell:"))
        drow.addWidget(self.dedup_cell)
        cl.addLayout(drow)
        self.cb_denoise = QCheckBox("Remove isolated noise points (dust, rain, stray returns)")
        self.cb_denoise.setToolTip(
            "Drops points that are far from all of their neighbours compared with the rest "
            "of the scan. Conservative: real surfaces are kept.")
        cl.addWidget(self.cb_denoise)
        self.cb_moving = QCheckBox("Remove moving objects (people, cars) seen in only one scan")
        self.cb_moving.setToolTip(
            "Drops points from one scan where another scan looked straight through that spot "
            "and saw something further away. Objects hidden from the other scan are kept. "
            "Works on scans saved in their scanner's own frame (raw scans).")
        cl.addWidget(self.cb_moving)
        lv.addWidget(box)

        self.cb_source = QCheckBox("Add a 'source' field (which file each point came from)")
        self.cb_compress = QCheckBox("Compress PCD output (binary_compressed)")
        lv.addWidget(self.cb_source)
        lv.addWidget(self.cb_compress)
        self.spacing = QDoubleSpinBox()
        self.spacing.setDecimals(3)
        self.spacing.setRange(0.0, 10.0)
        self.spacing.setSingleStep(0.01)
        self.spacing.setSpecialValueText("Full resolution")
        self.spacing.setToolTip(
            "Thin the merged cloud so no two points are closer than this distance "
            "(in the scans' units, e.g. 0.02 = 2 cm). Original points are kept, not "
            "averaged, so intensity and other fields stay exact. Leave at Full "
            "resolution to save every point.")
        spacing_row = QFormLayout()
        spacing_row.addRow("Min. point spacing:", self.spacing)
        lv.addLayout(spacing_row)

        self.b_save_tf = QPushButton("Save transforms…")
        self.b_save_tf.clicked.connect(self._save_transforms)
        lv.addWidget(self.b_save_tf)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setToolTip("Stop the current step. Nothing already loaded is lost.")
        self.b_cancel.clicked.connect(self._cancel)
        self.b_cancel.setVisible(False)
        prow = QHBoxLayout()
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.b_cancel)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        lv.addLayout(prow)
        lv.addWidget(self.status)
        # Ticks while work runs, so the elapsed time keeps moving even during
        # a single long step.
        self._clock = QElapsedTimer()
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._update_progress_text)
        self._frac = 0.0

        # ---- right panel: 3D view
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        self.rb_orig = QRadioButton("Original positions")
        self.rb_aligned = QRadioButton("Aligned")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_orig)
        grp.addButton(self.rb_aligned)
        self.rb_orig.setChecked(True)
        self.rb_orig.toggled.connect(self._redraw)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.rb_orig)
        top.addWidget(self.rb_aligned)
        top.addStretch(1)
        top.addWidget(QLabel("Point size:"))
        self.pt_size = QSlider(Qt.Horizontal)
        self.pt_size.setRange(1, 8)
        self.pt_size.setValue(2)
        self.pt_size.setFixedWidth(120)
        self.pt_size.valueChanged.connect(self._redraw)
        top.addWidget(self.pt_size)
        b_fit = QPushButton("Reset view")
        b_fit.clicked.connect(self._fit_view)
        top.addWidget(b_fit)
        rv.addLayout(top)

        if HAVE_GL:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor((25, 25, 30))
            rv.addWidget(self.view, 1)
        else:
            self.view = None
            msg = QLabel("3D preview unavailable (OpenGL could not be initialised).")
            msg.setAlignment(Qt.AlignCenter)
            rv.addWidget(msg, 1)
        hint = QLabel("Left-drag: rotate   Right/middle-drag: pan   Wheel: zoom")
        hint.setStyleSheet("color: #888;")
        rv.addWidget(hint)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 920])
        self.setCentralWidget(split)

        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)
        self._update_buttons()

    # ---------------------------------------------------------------- state

    def _busy(self) -> bool:
        return self._thread is not None

    def _update_buttons(self):
        busy = self._busy()
        n = len(self.clouds)
        self.b_align.setEnabled(not busy and n >= 2)
        self.b_retry.setEnabled(not busy and self.result is not None)
        self.b_merge.setEnabled(not busy and self.result is not None)
        self.b_save_tf.setEnabled(not busy and self.result is not None)
        self.b_remove.setEnabled(not busy and n > 0)
        self.rb_aligned.setEnabled(self.result is not None)
        self.drop_hint.setVisible(n == 0)

    def _set_status(self, msg: str, frac: float | None = None):
        self.status.setText(msg)
        if frac is not None:
            self._frac = max(0.0, min(1.0, frac))
            self.progress.setValue(int(self._frac * 1000))
            self._update_progress_text()

    def _update_progress_text(self):
        if not self._busy():
            self.progress.setFormat("")
            return
        secs = self._clock.elapsed() / 1000.0
        text = f"{self._frac:.0%}   {_fmt_secs(secs)}"
        # A remaining-time estimate is only shown once it can be meaningful.
        if 0.1 <= self._frac < 1.0 and secs > 3:
            text += f" (about {_fmt_secs(secs * (1 - self._frac) / self._frac)} left)"
        self.progress.setFormat(text)

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel_requested = True
            self.b_cancel.setEnabled(False)
            self.status.setText("Cancelling after the current step…")

    def _run(self, fn, args, on_done, label):
        if self._busy():
            return
        self._thread = QThread()
        self._worker = _Worker(fn, *args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._set_status)
        # Connect to bound methods (not lambdas) so Qt delivers the results
        # on the GUI thread.
        self._on_done = on_done
        self._worker.finished.connect(self._worker_done)
        self._worker.failed.connect(self._worker_failed)
        self._worker.cancelled.connect(self._worker_cancelled)
        self._clock.start()
        self._tick.start()
        self.b_cancel.setEnabled(True)
        self.b_cancel.setVisible(True)
        self._set_status(label, 0.0)
        self._thread.start()
        self._update_buttons()

    def _worker_done(self, payload):
        self._finish(self._on_done, payload)

    def _worker_failed(self, message):
        self._finish(None, message, failed=True)

    def _worker_cancelled(self):
        self._finish(None, None, cancelled=True)

    def _finish(self, on_done, payload, failed=False, cancelled=False):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self._tick.stop()
        self.b_cancel.setVisible(False)
        secs = self._clock.elapsed() / 1000.0
        self.progress.setValue(0 if failed or cancelled else 1000)
        self.progress.setFormat("" if failed or cancelled else f"Done in {_fmt_secs(secs)}")
        if cancelled:
            self._set_status("Cancelled.")
        elif failed:
            self._set_status("Failed: " + payload)
            QMessageBox.critical(self, "PCD Merge", payload)
        else:
            on_done(payload)
        self._update_buttons()

    # ---------------------------------------------------------------- files

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                paths += sorted(os.path.join(p, f) for f in os.listdir(p)
                                if f.lower().endswith(".pcd"))
            elif p.lower().endswith(".pcd"):
                paths.append(p)
        if paths:
            self.add_files(paths)
        else:
            self._set_status("Only .pcd files (or folders containing them) can be dropped.")

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add point clouds", "",
                                                "Point clouds (*.pcd);;All files (*)")
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        known = {c.path for c in self.clouds}
        paths = [p for p in paths if p not in known]
        if paths and not self._busy():
            self._run(_load_files, (paths,), self._loaded, "Loading…")

    def _loaded(self, payload):
        loaded, errors = payload
        self.clouds.extend(c for c, _ in loaded)
        self._samples.extend(p for _, p in loaded)
        self._rebuild_previews()
        self._invalidate()
        self._refresh_list()
        self._fit_view()
        if errors:
            QMessageBox.warning(self, "Some files could not be read", "\n".join(errors))
        self._set_status(f"{len(self.clouds)} cloud(s) loaded. "
                         + ("Press Align." if len(self.clouds) >= 2 else "Add at least one more."))

    def _rebuild_previews(self):
        if not self.clouds:
            return
        # Samples are random and in random order, so a prefix is a random subset.
        per = PREVIEW_POINTS // len(self.clouds)
        out = [s[:per] for s in self._samples]
        self.previews = out
        # Draw relative to the reference cloud so large (e.g. UTM) coordinates
        # don't lose precision in float32 OpenGL buffers.
        self._origin = out[0].mean(axis=0) if len(out[0]) else np.zeros(3)

    def _refresh_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for i, c in enumerate(self.clouds):
            r, g, b = PALETTE[i % len(PALETTE)]
            pm = QPixmap(14, 14)
            pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
            text = f"{c.name}  ({len(c):,} pts)"
            if i == 0:
                text += "  [reference]"
            item = QListWidgetItem(QIcon(pm), text)
            item.setToolTip(f"{c.path}\nFields: {', '.join(c.field_names)}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        self._update_buttons()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            del self.clouds[r]
            del self._samples[r]
        self._rebuild_previews()
        self._invalidate()
        self._refresh_list()
        self._redraw()

    def _clear(self):
        if self._busy():
            return
        self.clouds.clear()
        self.previews.clear()
        self._samples.clear()
        self._invalidate()
        self._refresh_list()
        self._redraw()
        self._set_status("Ready.", 0)

    def _invalidate(self):
        self.result = None
        self.report.clear()
        self.rb_orig.setChecked(True)

    # ---------------------------------------------------------------- align

    def _start_align(self, new_seed: bool):
        if len(self.clouds) < 2:
            return
        seed = random.randrange(1 << 30) if new_seed else 0
        settings = AlignSettings(voxel_size=self.voxel.value() or None, seed=seed)
        self._run(_align_clouds, (list(self.clouds), settings), self._aligned, "Aligning…")

    def _aligned(self, result: AlignResult):
        self.result = result
        lines = [f"Voxel size used: {result.voxel_size:.3g}", "", "Pair overlap:"]
        for p in result.pairs:
            lines.append(f"  {p.source + 1} ↔ {p.target + 1}: {p.fitness:.0%}"
                         f"  (rmse {p.rmse:.3g})")
        lines += ["", "Each cloud vs. the rest:"]
        for i, f in enumerate(result.cloud_fitness):
            lines.append(f"  {i + 1} {self.clouds[i].name}: {f:.0%}")
        warns = result.warnings()
        if warns:
            lines += ["", "Warnings:"] + ["  " + w for w in warns]
            lines += ["", "Try Retry, or a larger voxel size."]
        self.report.setPlainText("\n".join(lines))
        self.rb_aligned.setChecked(True)
        self._redraw()
        if warns:
            self._set_status("Aligned with warnings. Check the preview before merging.")
        else:
            self._set_status("Aligned. Check the preview, then Accept and merge (or Retry).")

    # ---------------------------------------------------------------- merge

    def _merge(self):
        if not self.result:
            return
        if not self.result.ok:
            ans = QMessageBox.question(
                self, "PCD Merge",
                "Some clouds could not be aligned and would be merged in their original "
                "position. Merge anyway?")
            if ans != QMessageBox.Yes:
                return
        base = os.path.dirname(self.clouds[0].path or "")
        ext = getattr(self, "_save_ext", ".pcd")
        filters = file_filter()
        chosen = next(f for f in filters.split(";;") if f.endswith(f"(*{ext})"))
        path, selected = QFileDialog.getSaveFileName(self, "Save merged cloud",
                                                     os.path.join(base, "merged" + ext),
                                                     filters, chosen)
        if not path:
            return
        typed = os.path.splitext(path)[1].lower()
        if typed in FORMATS:
            ext = typed
        else:  # no (known) extension typed: use the format picked in the dialog
            ext = next((e for e in FORMATS if selected.endswith(f"(*{e})")), ".pcd")
            path += ext
        self._save_ext = ext
        compressed = self.cb_compress.isChecked() and ext == ".pcd"
        if compressed:
            try:
                import lzf  # noqa: F401
            except ImportError:
                QMessageBox.warning(self, "PCD Merge", "Compressed output needs python-lzf; "
                                    "writing uncompressed binary instead.")
                compressed = False
        self._run(_merge_and_save,
                  (list(self.clouds), list(self.result.transforms), path,
                   self.cb_source.isChecked(), compressed, self.spacing.value() or None,
                   self._cleanup_options()),
                  self._merged, "Merging…")

    def _cleanup_options(self) -> CleanupOptions:
        return CleanupOptions(dedup=self.cb_dedup.isChecked(),
                              dedup_cell=self.dedup_cell.value() or None,
                              denoise=self.cb_denoise.isChecked(),
                              remove_moving=self.cb_moving.isChecked())

    def _merged(self, payload):
        path, n, partial, notes = payload
        msg = f"Saved {n:,} points to {path}"
        if partial:
            msg += ("\n\nThese fields were missing from some files and were filled with 0: "
                    + ", ".join(partial))
        for note in notes:
            msg += "\n\n" + note[0].upper() + note[1:] + "."
        self._set_status(f"Saved {n:,} points to {os.path.basename(path)}.")
        QMessageBox.information(self, "PCD Merge", msg)

    def _save_transforms(self):
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save transforms", "transforms.json",
                                              "JSON (*.json)")
        if not path:
            return
        data = {
            "reference": self.clouds[0].path,
            "voxel_size": self.result.voxel_size,
            "transforms": {c.path: np.asarray(t).tolist()
                           for c, t in zip(self.clouds, self.result.transforms)},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Transforms saved to {os.path.basename(path)}.")

    # ---------------------------------------------------------------- view

    def _redraw(self, *_):
        if self.view is None:
            return
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        aligned = self.rb_aligned.isChecked() and self.result is not None
        size = float(self.pt_size.value())
        for i, xyz in enumerate(self.previews):
            item = self.file_list.item(i)
            if item is not None and item.checkState() != Qt.Checked:
                continue
            pts = xyz
            if aligned:
                t = self.result.transforms[i]
                pts = xyz @ t[:3, :3].T + t[:3, 3]
            pts = (pts - self._origin).astype(np.float32)
            r, g, b = PALETTE[i % len(PALETTE)]
            sc = gl.GLScatterPlotItem(pos=pts, color=(r, g, b, 1.0), size=size, pxMode=True)
            sc.setGLOptions("opaque")
            self.view.addItem(sc)
            self._items.append(sc)

    def _fit_view(self):
        self._redraw()
        if self.view is None or not self.previews:
            return
        allp = np.concatenate(self.previews) - self._origin
        lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
        centre = (lo + hi) / 2
        from pyqtgraph import Vector

        self.view.setCameraPosition(pos=Vector(*centre), distance=float(np.linalg.norm(hi - lo)) * 1.1,
                                    elevation=35, azimuth=-60)


def _fmt_secs(secs: float) -> str:
    secs = int(round(secs))
    return f"{secs // 60}:{secs % 60:02d}" if secs >= 60 else f"{secs} s"


def run(argv=None) -> int:
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("PCD Merge")
    from .startup import before_main_window

    if before_main_window(app):
        return 0  # installing or updating; the new copy starts by itself
    win = MainWindow()
    win.show()
    files = [a for a in (sys.argv[1:] if argv is None else argv[1:]) if a.lower().endswith(".pcd")]
    if files:
        win.add_files(files)
    return app.exec()
