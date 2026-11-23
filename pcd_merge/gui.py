"""Desktop app: drop PCD files in, align them, preview, then merge and save."""

from __future__ import annotations

import json
import os
import random
import sys
import traceback

import numpy as np
from PySide6.QtCore import QElapsedTimer, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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

from . import __version__, retry
from .cleanup import CleanupOptions, clean_masks
from .cli import _cloud_key
from .exporters import FORMATS, file_filter
from .merge import sensor_origin
from .files import already_merged, read_manifest, save_merged
from .flyscene import build_scene
from .pcd_io import PointCloud
from .readers import find_scan_files, input_filter, read_cloud, read_clouds
from .quality import (CHANGED, NO_OVERLAP, QualityReport, assess, colour_stops, error_colours,
                      fmt_distance)
from .registration import AlignResult, AlignSettings, align

try:
    import pyqtgraph.opengl as gl

    HAVE_GL = True
except Exception:  # pragma: no cover - depends on the machine's OpenGL
    HAVE_GL = False

PREVIEW_POINTS = 400_000
REMOVED_COLOUR = (1.0, 1.0, 1.0, 1.0)  # points the clean-up would remove (white: unlike any cloud colour)  # total points drawn in the 3D view
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


def _load_files(paths, base_only=False, *, progress):
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
            report = lambda f, m=msg, b=base, s=span: progress(m, b + s * f * 0.95)  # noqa: E731
            clouds = read_clouds(p, progress=report) if not base_only else [read_cloud(p, progress=report)]
            progress(f"Preparing preview of {name}", base + span * 0.95)
            out += [(c, _preview_sample(c)) for c in clouds]
        except Cancelled:
            raise
        except Exception as e:
            errors.append(f"{name}: {e}")
        done += size
    return out, errors


def _align_clouds(clouds, settings, previews, quality_ref=None, *, progress):
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

    def align_progress(msg, frac):
        progress(msg, frac * 0.9)

    result = align(xyz, settings, align_progress)
    progress("Measuring alignment quality", 0.92)
    # Reuse the downsampled clouds from alignment instead of the full ones.
    q_xyz, q_voxel = quality_ref or (result.fine or xyz, result.voxel_size)
    quality = assess(q_xyz, result.transforms, q_voxel, preview_xyz=previews)
    return result, quality


GRADE_COLOURS = {"Good": "#2e9e44", "Fair": "#d08a00", "Poor": "#d0342c"}


def _legend_pixmap(width=160, height=12) -> QPixmap:
    pm = QPixmap(width, height)
    grad = QLinearGradient(0, 0, width, 0)
    for pos, (r, g, b) in colour_stops():
        grad.setColorAt(pos, QColor.fromRgbF(r, g, b))
    p = QPainter(pm)
    p.fillRect(0, 0, width, height, grad)
    p.end()
    return pm


def _merge_and_save(clouds, transforms, path, include, add_source, compressed, spacing,
                    cleanup, base_path, voxel_size, progress):
    n, partial, notes, stats = save_merged(
        clouds, transforms, path, include, add_source=add_source, compressed=compressed,
        base_path=base_path, voxel_size=voxel_size, min_spacing=spacing, cleanup=cleanup,
        progress=progress)
    if stats is not None:
        notes = stats.lines([c.name for c, inc in zip(clouds, include) if inc]) + notes
    return path, n, partial, notes, include


def _cleanup_preview(clouds, transforms, previews, opts):
    """Run the chosen clean-up on the preview points (a random subset of each
    cloud, already in memory), so ticking an option updates the view quickly.

    Returns per cloud a mask of the preview points that would be kept, and the
    counts. Counts are scaled up to an estimate for the full clouds."""
    xyz, origins, weights = [], [], []
    for c, t, p in zip(clouds, transforms, previews):
        t = np.asarray(t, dtype=np.float64)
        xyz.append(p @ t[:3, :3].T + t[:3, 3])
        origins.append(sensor_origin(c, t))
        weights.append(len(c) / max(len(p), 1))
    keep, stats = clean_masks(xyz, origins, opts, weights=weights)
    removed = [int(round((~k).sum() * w)) for k, w in zip(keep, weights)]
    return keep, stats, removed


class _PreviewWorker(QObject):
    """Runs one clean-up preview on its own thread; no progress, no cancel."""

    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, gen, *args):
        super().__init__()
        self.gen, self.args = gen, args

    def run(self):
        try:
            result = _cleanup_preview(*self.args)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(self.gen, f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(self.gen, result)


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
        self._reset_retries()
        self.quality: QualityReport | None = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._items = []
        self._origin = np.zeros(3)
        # Set when cloud 0 is an existing merged cloud that new scans are added to.
        self.base_path: str | None = None
        self._loading_base = False

        # ---- left panel
        left = QWidget()
        lv = QVBoxLayout(left)

        self.drop_hint = QLabel("Drag and drop two or more scans (PCD, PLY, LAS, LAZ, E57),\n"
                                "or a whole folder, here (or use Add files…)")
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
        b_folder = QPushButton("Add folder…")
        b_folder.setToolTip("Add every scan (PCD, PLY, LAS, LAZ, E57) in a folder and its subfolders.")
        b_folder.clicked.connect(self._add_folder_dialog)
        self.b_remove = QPushButton("Remove")
        self.b_remove.clicked.connect(self._remove_selected)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self._clear)
        for b in (b_add, b_folder, self.b_remove, b_clear):
            row.addWidget(b)
        lv.addLayout(row)

        self.b_base = QPushButton("Add new scans to an existing merged cloud…")
        self.b_base.setToolTip(
            "Pick a merged .pcd you saved earlier. It stays exactly where it is and the "
            "other files in the list are aligned to it and appended when you merge.")
        self.b_base.clicked.connect(self._base_dialog)
        lv.addWidget(self.b_base)

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

        self.grade_label = QLabel()
        self.grade_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.grade_label.setToolTip(
            "Good: the clouds agree closely where they overlap. Fair: usable, but check the "
            "heatmap for red areas. Poor: likely misaligned or barely overlapping.")
        self.grade_label.hide()
        lv.addWidget(self.grade_label)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Alignment results will appear here.")
        self.report.setMaximumHeight(230)
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
        self.b_retry.clicked.connect(self._retry)
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
        prow = QHBoxLayout()
        prow.addWidget(QLabel("In the preview, points to remove:"))
        self.removed_mode = QComboBox()
        self.removed_mode.addItems(["Highlight in white", "Hide"])
        self.removed_mode.setToolTip(
            "How the aligned preview shows the points the ticked clean-up would remove. "
            "The preview uses a sample of each cloud, so it is a close guide; the merge "
            "itself works on every point.")
        prow.addWidget(self.removed_mode)
        prow.addStretch(1)
        cl.addLayout(prow)
        self.cleanup_info = QLabel("")
        self.cleanup_info.setWordWrap(True)
        self.cleanup_info.setStyleSheet("color: #888;")
        cl.addWidget(self.cleanup_info)
        lv.addWidget(box)

        # Clean-up preview: recomputed shortly after any option changes.
        self._cleanup_keep = None  # per-cloud mask of preview points kept
        self._cleanup_gen = 0
        self._cleanup_thread = None
        self._cleanup_pending = False
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.setInterval(250)
        self._cleanup_timer.timeout.connect(self._start_cleanup_preview)
        for cb in (self.cb_dedup, self.cb_denoise, self.cb_moving):
            cb.toggled.connect(self._cleanup_changed)
        self.dedup_cell.valueChanged.connect(self._cleanup_changed)
        self.removed_mode.currentIndexChanged.connect(self._redraw)

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
        top.addSpacing(16)
        top.addWidget(QLabel("Colour:"))
        self.colour_mode = QComboBox()
        self.colour_mode.addItems(["Per cloud", "Alignment error"])
        self.colour_mode.setToolTip(
            "Alignment error colours each point by its distance to the surface of the other "
            "clouds: blue fits, red is off. Grey points are seen by only one scan.")
        self.colour_mode.currentIndexChanged.connect(self._redraw)
        top.addWidget(self.colour_mode)
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
        self.b_fly = QPushButton("Walk through…")
        self.b_fly.setToolTip(
            "Open a first-person view and walk or fly around inside the scan (W A S D and "
            "the mouse). Shows the clouds as in this view: aligned, or in their original "
            "positions. To walk through a merged file, load just that file.")
        self.b_fly.clicked.connect(self._fly_through)
        top.addWidget(self.b_fly)
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
        bottom = QHBoxLayout()
        hint = QLabel("Left-drag: rotate   Right/middle-drag: pan   Wheel: zoom")
        hint.setStyleSheet("color: #888;")
        bottom.addWidget(hint)
        bottom.addStretch(1)
        self.legend = QWidget()
        lg = QHBoxLayout(self.legend)
        lg.setContentsMargins(0, 0, 0, 0)
        self.legend_lo = QLabel("0")
        bar = QLabel()
        bar.setPixmap(_legend_pixmap())
        self.legend_hi = QLabel()
        grey = QPixmap(12, 12)
        grey.fill(QColor.fromRgbF(*NO_OVERLAP))
        swatch = QLabel()
        swatch.setPixmap(grey)
        magenta = QPixmap(12, 12)
        magenta.fill(QColor.fromRgbF(*CHANGED))
        swatch_changed = QLabel()
        swatch_changed.setPixmap(magenta)
        swatch_changed.setToolTip(
            "Seen differently by the scans: moved objects, people, opened doors. "
            "These don't lower the grade unless most of the overlap is like this, "
            "which points to a misalignment instead.")
        for w in (QLabel("Error:"), self.legend_lo, bar, self.legend_hi, QLabel("  "),
                  swatch_changed, QLabel("changed"), QLabel("  "), swatch, QLabel("no overlap")):
            lg.addWidget(w)
        self.legend.hide()
        bottom.addWidget(self.legend)
        rv.addLayout(bottom)

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
        self.b_base.setEnabled(not busy)
        self.rb_aligned.setEnabled(self.result is not None)
        self.colour_mode.setEnabled(self.quality is not None)
        self.drop_hint.setVisible(n == 0)
        self.b_fly.setEnabled(not busy and n > 0 and HAVE_GL)

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
        paths = find_scan_files([url.toLocalFile() for url in e.mimeData().urls()])
        if paths:
            self.add_files(paths)
        else:
            self._set_status("Only scans (PCD, PLY, LAS, LAZ, E57) or folders containing them can be dropped.")

    def _add_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add point clouds", "", input_filter())
        if paths:
            self.add_files(paths)

    def _add_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Add every scan in a folder")
        if path:
            paths = find_scan_files([path])
            if paths:
                self.add_files(paths)
            else:
                self._set_status("No scans found in that folder.")

    def _base_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Existing merged cloud", "", input_filter())
        if not path or self._busy():
            return
        if self.base_path and self.clouds:
            del self.clouds[0]  # replace the previous base
            del self._samples[0]
        self.base_path = None
        keep = [k for k, c in enumerate(self.clouds)
                if os.path.abspath(c.path or "") != os.path.abspath(path)]
        self.clouds = [self.clouds[k] for k in keep]
        self._samples = [self._samples[k] for k in keep]
        self._loading_base = True
        self._run(_load_files, ([path], True), self._loaded, "Loading merged cloud…")

    def add_files(self, paths):
        known = {os.path.abspath(c.path) for c in self.clouds if c.path}
        paths = [p for p in paths if os.path.abspath(p) not in known]
        skipped = []
        if self.base_path:
            skipped = already_merged(read_manifest(self.base_path), paths)
            paths = [p for p in paths if p not in skipped]
        if skipped:
            self._set_status(f"Skipped {len(skipped)} file(s) already in the merged cloud.")
        if paths and not self._busy():
            self._loading_base = False
            self._run(_load_files, (paths,), self._loaded, "Loading…")

    def _loaded(self, payload):
        loaded, errors = payload
        if self._loading_base:
            self._loading_base = False
            if loaded:
                base, sample = loaded[0]
                self.base_path = base.path
                self.clouds.insert(0, base)
                self._samples.insert(0, sample)
                done = set(already_merged(read_manifest(self.base_path),
                                          [c.path for c in self.clouds[1:]]))
                keep = [0] + [k for k in range(1, len(self.clouds))
                              if self.clouds[k].path not in done]
                self.clouds = [self.clouds[k] for k in keep]
                self._samples = [self._samples[k] for k in keep]
        else:
            self.clouds.extend(c for c, _ in loaded)
            self._samples.extend(p for _, p in loaded)
        self._rebuild_previews()
        self._invalidate()
        self._refresh_list()
        self._fit_view()
        if errors:
            QMessageBox.warning(self, "Some files could not be read", "\n".join(errors))
        if self.base_path and len(self.clouds) < 2:
            self._set_status("Merged cloud loaded. Now add the new scans to append to it.")
        else:
            self._set_status(f"{len(self.clouds)} cloud(s) loaded. "
                             + ("Press Align." if len(self.clouds) >= 2
                                else "Add at least one more."))

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
            if i == 0 and self.base_path:
                text += "  [existing merged cloud, stays fixed]"
            elif i == 0:
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
        if 0 in rows:
            self.base_path = None
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
        self.base_path = None
        self._invalidate()
        self._refresh_list()
        self._redraw()
        self._set_status("Ready.", 0)

    def _invalidate(self):
        self.result = None
        self._cleanup_changed()
        self.quality = None
        self._reset_retries()
        self.report.clear()
        self.grade_label.hide()
        self.colour_mode.setCurrentIndex(0)
        self.rb_orig.setChecked(True)

    # ---------------------------------------------------------------- align

    def _reset_retries(self):
        self._retry_press = 0  # how many times Retry was pressed since Align
        self._retry_base = None  # (settings, voxel, quality clouds) of the first Align
        self._best = None  # (result, quality, label) best so far
        self._last_label = ""
        if hasattr(self, "b_retry"):
            self.b_retry.setText("Retry")

    def _base_settings(self, seed: int) -> AlignSettings:
        return AlignSettings(voxel_size=self.voxel.value() or None, seed=seed,
                             anchor=self.base_path is not None)

    def _start_align(self, new_seed: bool = False):
        """Align (a fresh start: forgets earlier retries)."""
        if len(self.clouds) < 2:
            return
        self._reset_retries()
        seed = random.randrange(1 << 30) if new_seed else 0
        self._last_label = "first try"
        self._run(_align_clouds, (list(self.clouds), self._base_settings(seed),
                                  list(self.previews)),
                  self._aligned, "Aligning…")

    def _retry(self):
        if len(self.clouds) < 2:
            return
        if self._retry_base is None:  # no Align yet in this session of retries
            self._start_align(new_seed=True)
            return
        base, base_voxel, q_ref = self._retry_base
        self._retry_press += 1
        settings = retry.settings_for(base, base_voxel, self._retry_press,
                                      random.randrange(1 << 30))
        lv = retry.level(self._retry_press)
        self._last_label = f"retry {self._retry_press}: {lv.describe(settings.voxel_size)}"
        self._run(_align_clouds, (list(self.clouds), settings, list(self.previews), q_ref),
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
        if self._retry_press:
            nxt = retry.level(self._retry_press + 1)
            self.b_retry.setText(f"Retry ({nxt.voxel_factor:g}× coarser)")
        else:
            self.b_retry.setText(f"Retry ({retry.level(1).voxel_factor:g}× coarser)")
        self.result, self.quality = result, quality
        grade = quality.grade
        self.grade_label.setText(f"Alignment quality: {grade}")
        self.grade_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {GRADE_COLOURS[grade]};")
        self.grade_label.show()
        lines = quality.lines([c.name for c in self.clouds])[2:]  # grade is shown above
        lines += ["", f"Voxel size used: {result.voxel_size:.3g}. Distances are to the other "
                  "cloud's surface; points more than "
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
        self.report.setPlainText("\n".join(lines))
        self.legend_hi.setText(fmt_distance(quality.scale_max))
        self.rb_aligned.setChecked(True)
        if grade != "Good":
            self.colour_mode.setCurrentIndex(1)  # show where the clouds disagree
        self._update_buttons()
        self._cleanup_changed()
        self._redraw()
        if warns or grade != "Good":
            self._set_status(f"Aligned, quality {grade}. Check the error heatmap before merging.")
        else:
            self._set_status("Aligned. Check the preview, then Accept and merge (or Retry).")

    # ---------------------------------------------------------------- merge

    def _merge(self):
        if not self.result:
            return
        include = [True] * len(self.clouds)
        if not self.result.ok:
            bad = [c.name for c, ok in zip(self.clouds, self.result.connected) if not ok]
            box = QMessageBox(self)
            box.setWindowTitle("PCD Merge")
            box.setIcon(QMessageBox.Question)
            shown = ", ".join(bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
            box.setText(f"{len(bad)} cloud(s) could not be aligned: {shown}.")
            box.setInformativeText("Leave them out of the merged cloud, or merge them in "
                                   "their original position?")
            b_skip = box.addButton("Leave them out", QMessageBox.AcceptRole)
            b_all = box.addButton("Merge anyway", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_skip)
            box.exec()
            if box.clickedButton() is b_skip:
                include = list(self.result.connected)
                if sum(include) < 2:
                    QMessageBox.warning(self, "PCD Merge", "Fewer than two clouds aligned; "
                                        "there is nothing to merge.")
                    return
            elif box.clickedButton() is not b_all:
                return
        elif self.quality is not None and self.quality.grade == "Poor":
            ans = QMessageBox.question(
                self, "PCD Merge",
                "The alignment quality is Poor: the clouds disagree or barely overlap. "
                "Merge anyway?")
            if ans != QMessageBox.Yes:
                return
        ext = getattr(self, "_save_ext", ".pcd")
        if self.base_path:
            ext = ".pcd"  # keep adding to the same file by default
            suggested = self.base_path
        else:
            suggested = os.path.join(os.path.dirname(self.clouds[0].path or ""), "merged" + ext)
        filters = file_filter()
        chosen = next(f for f in filters.split(";;") if f.endswith(f"(*{ext})"))
        path, selected = QFileDialog.getSaveFileName(self, "Save merged cloud", suggested,
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
                  (list(self.clouds), list(self.result.transforms), path, include,
                   self.cb_source.isChecked(), compressed, self.spacing.value() or None,
                   self._cleanup_options(), self.base_path, self.result.voxel_size),
                  self._merged, "Merging…")

    def _cleanup_options(self) -> CleanupOptions:
        return CleanupOptions(dedup=self.cb_dedup.isChecked(),
                              dedup_cell=self.dedup_cell.value() or None,
                              denoise=self.cb_denoise.isChecked(),
                              remove_moving=self.cb_moving.isChecked())

    def _merged(self, payload):
        path, n, partial, notes, include = payload
        msg = f"Saved {n:,} points to {path}"
        left_out = len(include) - sum(include)
        if left_out:
            msg += f"\n\n{left_out} cloud(s) that could not be aligned were left out."
        if partial:
            msg += ("\n\nThese fields were missing from some files and were filled with 0: "
                    + ", ".join(partial))
        for note in notes:
            msg += "\n\n" + note[0].upper() + note[1:] + "."
        if path.lower().endswith(".pcd"):
            msg += ("\n\nThe list of scans in it is saved next to it as "
                    f"{os.path.basename(path)}.json, so scans already in it are skipped if "
                    "you add new scans to it later.")
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
            "transforms": {_cloud_key(c): np.asarray(t).tolist()
                           for c, t in zip(self.clouds, self.result.transforms)},
        }
        if self.quality is not None:
            data["quality"] = self.quality.to_dict()
            data["quality"]["files"] = [c.path for c in self.clouds]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"Transforms saved to {os.path.basename(path)}.")

    def closeEvent(self, e):
        if self._cleanup_thread is not None:  # let a running preview finish first
            self._cleanup_thread.quit()
            self._cleanup_thread.wait()
        super().closeEvent(e)

    # ---------------------------------------------------------------- clean-up preview

    def _cleanup_changed(self, *_):
        """Options, clouds or alignment changed: drop the old preview and
        schedule a new one (debounced, so dragging the cell size stays smooth)."""
        self._cleanup_gen += 1
        had = self._cleanup_keep is not None
        self._cleanup_keep = None
        opts = self._cleanup_options()
        if not opts.any:
            self.cleanup_info.setText("")
        elif self.result is None:
            self.cleanup_info.setText("Align first to see the effect in the preview.")
        else:
            self.cleanup_info.setText("Updating preview…")
            self._cleanup_timer.start()
        if had:
            self._redraw()

    def _start_cleanup_preview(self):
        if self._cleanup_thread is not None:
            self._cleanup_pending = True  # run again when the current one ends
            return
        opts = self._cleanup_options()
        if not opts.any or self.result is None or len(self.previews) != len(self.clouds):
            return
        self._cleanup_pending = False
        thread = QThread()
        worker = _PreviewWorker(self._cleanup_gen, list(self.clouds),
                                list(self.result.transforms), list(self.previews), opts)
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
        keep, stats, removed = payload
        self._cleanup_keep = keep
        total = sum(len(c) for c in self.clouds)
        lines = []
        if sum(removed):
            parts = [(label, sum(c * w for c, w in zip(counts, self._weights())))
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
                         + ", ".join(self.clouds[i].name for i in stats.no_sensor_position)
                         + ", so it is not used as evidence.")
        self.cleanup_info.setText("\n".join(lines))
        self._redraw()

    def _cleanup_failed(self, gen, message):
        self._cleanup_thread_ended()
        if gen == self._cleanup_gen:
            self.cleanup_info.setText("Preview failed: " + message)

    def _weights(self):
        return [len(c) / max(len(p), 1) for c, p in zip(self.clouds, self.previews)]

    # ---------------------------------------------------------------- view

    def _redraw(self, *_):
        if self.view is None:
            return
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        aligned = self.rb_aligned.isChecked() and self.result is not None
        heat = (aligned and self.quality is not None and self.colour_mode.currentIndex() == 1
                and len(self.quality.point_error) == len(self.previews))
        self.legend.setVisible(heat)
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
            if heat:
                color = error_colours(self.quality.point_error[i], self.quality.scale_max)
            else:
                color = (*PALETTE[i % len(PALETTE)], 1.0)
            keep = None
            if (aligned and self._cleanup_keep is not None
                    and len(self._cleanup_keep[i]) == len(pts)):
                keep = self._cleanup_keep[i]
            if keep is not None and not keep.all():
                gone = pts[~keep]
                pts = pts[keep]
                if isinstance(color, np.ndarray) and color.ndim == 2:
                    color = color[keep]
                if self.removed_mode.currentIndex() == 0:
                    self._add_points(gone, REMOVED_COLOUR, size + 1)
            self._add_points(pts, color, size)

    def _add_points(self, pts, color, size):
        if not len(pts):
            return
        sc = gl.GLScatterPlotItem(pos=pts, color=color, size=size, pxMode=True)
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


    # ---------------------------------------------------------------- fly-through

    def _fly_through(self):
        shown = [i for i in range(len(self.clouds))
                 if (it := self.file_list.item(i)) is None or it.checkState() == Qt.Checked]
        if not shown:
            return
        aligned = self.rb_aligned.isChecked() and self.result is not None
        transforms = [self.result.transforms[i] if aligned else np.eye(4) for i in shown]
        clouds = [self.clouds[i] for i in shown]
        self._fly_title = ("Walk through: " + ", ".join(c.name for c in clouds[:3])
                           + (f" and {len(clouds) - 3} more" if len(clouds) > 3 else "")
                           + (" (aligned)" if aligned and len(clouds) > 1 else ""))
        self._run(build_scene, (clouds, transforms), self._fly_ready, "Preparing walk-through…")

    def _fly_ready(self, scene):
        from .flythrough import FlyThroughWindow

        old = getattr(self, "_fly_win", None)
        if old is not None:
            try:
                old.close()
            except RuntimeError:  # already deleted
                pass
        self._fly_win = FlyThroughWindow(scene, self._fly_title, parent=self)
        self._fly_win.show()
        self._set_status("Walk-through open. Click into it, then use W A S D and the mouse.")


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
    files = find_scan_files([a for a in (sys.argv[1:] if argv is None else argv[1:])
                             if os.path.isfile(a)])
    if files:
        win.add_files(files)
    return app.exec()
