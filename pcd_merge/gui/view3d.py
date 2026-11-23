"""The one 3D view in the middle of the window.

Orbit (rotate / pan / zoom around the site, with top, front and side views
and an orthographic option) and walk-through (first person, inside the scan)
are two modes of the same view; the walk-through used to be a separate
window. A tool bar along the top of the view changes with the app's mode.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QPushButton, QRadioButton, QSlider, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..quality import CHANGED, NO_OVERLAP, error_colours, fmt_distance
from .common import EXPORT_COLOUR, HINT_STYLE, REMOVED_COLOUR, cloud_colour, legend_pixmap

try:
    import pyqtgraph.opengl as gl

    HAVE_GL = True
except Exception:  # pragma: no cover - depends on the machine's OpenGL
    HAVE_GL = False

# camera presets: name -> (elevation, azimuth) in degrees
CAMERAS = {"Orbit": None, "Top": (90, -90), "Front": (0, -90), "Side": (0, 0)}


class View3D(QWidget):
    walk_requested = Signal()

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self._items = []
        self._removed = None  # per scan: mask of preview points kept by the clean-up
        self._walk = None
        self._ortho = False
        self.removed_hidden = False  # clean-up preview: hide removed points instead of white

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # ---- tool bar
        self.toolbar = QWidget()
        rows = QVBoxLayout(self.toolbar)
        rows.setContentsMargins(4, 4, 4, 2)
        rows.setSpacing(2)
        top = QHBoxLayout()
        rows.addLayout(top)
        self.rb_orig = QRadioButton("Original positions")
        self.rb_aligned = QRadioButton("Aligned")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_orig)
        grp.addButton(self.rb_aligned)
        self.rb_orig.setChecked(True)
        self.rb_orig.toggled.connect(self.redraw)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.rb_orig)
        top.addWidget(self.rb_aligned)
        top.addSpacing(12)
        top.addWidget(QLabel("Colour:"))
        self.colour_mode = QComboBox()
        self.colour_mode.addItems(["Per scan", "Alignment error"])
        self.colour_mode.setToolTip(
            "Alignment error colours each point by its distance to the surface of the other "
            "scans: blue fits, red is off. Grey points are seen by only one scan.")
        self.colour_mode.currentIndexChanged.connect(self.redraw)
        top.addWidget(self.colour_mode)
        top.addStretch(1)
        top = QHBoxLayout()
        rows.addLayout(top)
        top.addWidget(QLabel("View:"))
        self.camera = QComboBox()
        self.camera.addItems(list(CAMERAS))
        self.camera.setToolTip("Orbit freely, or look straight down (Top), from the front "
                               "or from the side.")
        self.camera.activated.connect(self._camera_preset)
        top.addWidget(self.camera)
        self.cb_ortho = QCheckBox("Orthographic")
        self.cb_ortho.setToolTip("No perspective: sizes are the same near and far, "
                                 "like a plan or an elevation.")
        self.cb_ortho.toggled.connect(self._set_ortho)
        top.addWidget(self.cb_ortho)
        top.addStretch(1)
        top.addWidget(QLabel("Point size:"))
        self.pt_size = QSlider(Qt.Horizontal)
        self.pt_size.setRange(1, 8)
        self.pt_size.setValue(2)
        self.pt_size.setFixedWidth(70)
        self.pt_size.valueChanged.connect(self.redraw)
        top.addWidget(self.pt_size)
        b_fit = QPushButton("Reset view")
        b_fit.clicked.connect(self.fit)
        top.addWidget(b_fit)
        self.b_walk = QPushButton("Walk through")
        self.b_walk.setToolTip(
            "Walk or fly around inside the scan in this view (W A S D and the mouse; Esc "
            "to come back). Shows the scans as here: aligned or in their original positions.")
        self.b_walk.clicked.connect(self.walk_requested)
        top.addWidget(self.b_walk)
        lay.addWidget(self.toolbar)

        # ---- the view itself: orbit, or the walk-through
        self.stack = QStackedWidget()
        if HAVE_GL:
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor((25, 25, 30))
            self.stack.addWidget(self.view)
        else:
            self.view = None
            msg = QLabel("3D preview unavailable (OpenGL could not be initialised).")
            msg.setAlignment(Qt.AlignCenter)
            self.stack.addWidget(msg)
        lay.addWidget(self.stack, 1)

        # ---- bottom line: mouse help and the heat-map legend
        self.bottom = QWidget()
        bottom = QHBoxLayout(self.bottom)
        bottom.setContentsMargins(4, 0, 4, 2)
        hint = QLabel("Left-drag: rotate   Right/middle-drag: pan   Wheel: zoom")
        hint.setStyleSheet(HINT_STYLE)
        bottom.addWidget(hint)
        bottom.addStretch(1)
        self.legend = QWidget()
        lg = QHBoxLayout(self.legend)
        lg.setContentsMargins(0, 0, 0, 0)
        self.legend_hi = QLabel()
        bar = QLabel()
        bar.setPixmap(legend_pixmap())
        swatch_changed = QLabel()
        swatch_changed.setPixmap(_swatch(CHANGED))
        swatch_changed.setToolTip(
            "Seen differently by the scans: moved objects, people, opened doors. "
            "These don't lower the grade unless most of the overlap is like this, "
            "which points to a misalignment instead.")
        swatch_none = QLabel()
        swatch_none.setPixmap(_swatch(NO_OVERLAP))
        for w in (QLabel("Error:"), QLabel("0"), bar, self.legend_hi, QLabel("  "),
                  swatch_changed, QLabel("changed"), QLabel("  "), swatch_none,
                  QLabel("no overlap")):
            lg.addWidget(w)
        self.legend.hide()
        bottom.addWidget(self.legend)
        lay.addWidget(self.bottom)

    # ---------------------------------------------------------------- state

    @property
    def walking(self) -> bool:
        return self._walk is not None

    def update_controls(self, busy: bool):
        doc = self.doc
        self.rb_aligned.setEnabled(doc.aligned)
        if not doc.aligned and self.rb_aligned.isChecked():
            self.rb_orig.setChecked(True)
        self.colour_mode.setEnabled(doc.quality is not None)
        if doc.quality is None:
            self.colour_mode.setCurrentIndex(0)
        self.b_walk.setEnabled(not busy and bool(doc.scans) and HAVE_GL)

    def show_aligned(self, heat: bool | None = None):
        if self.doc.aligned:
            self.rb_aligned.setChecked(True)
            if heat is not None and self.doc.quality is not None:
                self.colour_mode.setCurrentIndex(1 if heat else 0)
        self.redraw()

    def aligned_shown(self) -> bool:
        return self.rb_aligned.isChecked() and self.doc.aligned

    def set_removed(self, keep):
        """Show which preview points the clean-up would remove (None: none)."""
        self._removed = keep
        self.redraw()

    # ---------------------------------------------------------------- drawing

    def redraw(self, *_):
        if self.view is None:
            return
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        doc = self.doc
        previews = doc.previews()
        origin = doc.origin()
        aligned = self.aligned_shown()
        q = doc.quality
        heat = (aligned and q is not None and self.colour_mode.currentIndex() == 1
                and len(q.point_error) == len(previews))
        self.legend.setVisible(heat)
        if q is not None:
            self.legend_hi.setText(fmt_distance(q.scale_max))
        size = float(self.pt_size.value())
        tfs = doc.transforms()
        for i, (xyz, scan) in enumerate(zip(previews, doc.scans)):
            if not scan.entry.visible or not len(xyz):
                continue
            pts = xyz
            if aligned:
                t = tfs[i]
                pts = xyz @ t[:3, :3].T + t[:3, 3]
            pts = (pts - origin).astype(np.float32)
            if heat and len(q.point_error[i]) >= len(pts):
                color = error_colours(q.point_error[i][:len(pts)], q.scale_max)
            else:
                color = (*cloud_colour(i), 1.0)
            keep = None
            if aligned and self._removed is not None and i < len(self._removed) \
                    and len(self._removed[i]) == len(pts):
                keep = self._removed[i]
            if keep is not None and not keep.all():
                gone = pts[~keep]
                pts = pts[keep]
                if isinstance(color, np.ndarray) and color.ndim == 2:
                    color = color[keep]
                if not self.removed_hidden:
                    self._add_points(gone, REMOVED_COLOUR, size + 1)
            self._add_points(pts, color, size)
        for xyz in doc.visible_exports():
            self._add_points((xyz - origin).astype(np.float32), (*EXPORT_COLOUR, 1.0), size)

    def _add_points(self, pts, color, size):
        if not len(pts):
            return
        sc = gl.GLScatterPlotItem(pos=pts, color=color, size=size, pxMode=True)
        sc.setGLOptions("opaque")
        self.view.addItem(sc)
        self._items.append(sc)

    def _extent(self):
        previews = [p for p in self.doc.previews() if len(p)] + self.doc.visible_exports()
        if not previews:
            return None
        allp = np.concatenate(previews) - self.doc.origin()
        lo, hi = np.percentile(allp, 2, axis=0), np.percentile(allp, 98, axis=0)
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo)) or 1.0

    def fit(self):
        self.redraw()
        self.camera.setCurrentIndex(0)
        self._apply_camera(35, -60)

    def _camera_preset(self, *_):
        preset = CAMERAS[self.camera.currentText()]
        if preset is None:
            self._apply_camera(35, -60)
        else:
            self._apply_camera(*preset)

    def _set_ortho(self, on):
        self._ortho = on
        self._camera_preset()

    def _apply_camera(self, elevation, azimuth):
        if self.view is None:
            return
        ext = self._extent()
        if ext is None:
            return
        centre, size = ext
        from pyqtgraph import Vector

        # pyqtgraph has no true orthographic camera; a very narrow field of
        # view from far away looks the same.
        fov = 1.0 if self._ortho else 60.0
        dist = size * 1.1 * (60.0 / fov if self._ortho else 1.0)
        self.view.opts["fov"] = fov
        self.view.setCameraPosition(pos=Vector(*centre), distance=dist,
                                    elevation=elevation, azimuth=azimuth)

    # ---------------------------------------------------------------- walk-through

    def show_walk(self, scene, title: str):
        from ..flythrough import FlyThroughWindow

        self.leave_walk()
        self._walk = FlyThroughWindow(scene, title, parent=self, on_exit=self.leave_walk)
        self.stack.addWidget(self._walk)
        self.stack.setCurrentWidget(self._walk)
        self.toolbar.hide()
        self.bottom.hide()

    def leave_walk(self):
        if self._walk is None:
            return
        w, self._walk = self._walk, None
        w.view.release_gl()
        self.stack.setCurrentIndex(0)
        self.stack.removeWidget(w)
        w.deleteLater()
        self.toolbar.show()
        self.bottom.show()


def _swatch(rgb) -> QPixmap:
    pm = QPixmap(12, 12)
    pm.fill(QColor.fromRgbF(*rgb))
    return pm
