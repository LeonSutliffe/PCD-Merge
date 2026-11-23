"""The one 3D view in the middle of the window.

Orbit (rotate / pan / zoom around the site, with top, front and side views
and an orthographic option) and walk-through (first person, inside the scan)
are two modes of the same view; the walk-through used to be a separate
window. A tool bar along the top of the view changes with the app's mode.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
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

SELECT_COLOUR = (1.0, 0.85, 0.1, 1.0)  # the Edit selection: yellow
DELETED_COLOUR = (0.45, 0.12, 0.12, 1.0)  # deleted points, when shown: dark red

# camera presets: name -> (elevation, azimuth) in degrees
CAMERAS = {"Orbit": None, "Top": (90, -90), "Front": (0, -90), "Side": (0, 0)}


class View3D(QWidget):
    walk_requested = Signal()
    walk_fallback = Signal(str)  # the walk-through opened in its own window (the error)
    selection_changed = Signal()

    NAV_HINT = "Left-drag: rotate   Right/middle-drag: pan   Wheel: zoom"

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self._items = []
        self._removed = None  # per scan: mask of preview points kept by the clean-up
        self._walk = None
        self._ortho = False
        self.removed_hidden = False  # clean-up preview: hide removed points instead of white
        # Edit mode
        self.edit_mode = False
        self.tool = None  # an edit_tools.Tool while one is chosen
        self.selection = []  # regions selected in Edit mode (not yet deleted)
        self.show_deleted = False  # draw deleted points (dark red) instead of hiding them
        self._sel_masks = {}  # scan index -> mask of its drawn points that are selected
        self._edit_cache = {}  # scan index -> (key, project-frame points, hidden mask)
        self._redraw_soon = QTimer(self)
        self._redraw_soon.setSingleShot(True)
        self._redraw_soon.setInterval(40)
        self._redraw_soon.timeout.connect(self.redraw)

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
        self.colour_mode.addItems(["Per scan", "Alignment error", "Height", "Intensity",
                                   "RGB colour"])
        self.colour_mode.setToolTip(
            "Alignment error colours each point by its distance to the surface of the other "
            "scans: blue fits, red is off. Grey points are seen by only one scan. "
            "Height goes from blue (low) to red (high), which makes floors easy to tell apart. "
            "Intensity shows how strongly each point reflected the laser (dark to light). "
            "RGB colour uses the scans' own colours, where they have them.")
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
            self.view.setMouseTracking(True)
            self.overlay = _Overlay(self)
            self._filter = _ToolFilter(self)
            self.view.installEventFilter(self._filter)
            self.stack.addWidget(self.view)
        else:
            self.view = None
            self.overlay = None
            msg = QLabel("3D preview unavailable (OpenGL could not be initialised).")
            msg.setAlignment(Qt.AlignCenter)
            self.stack.addWidget(msg)
        lay.addWidget(self.stack, 1)

        # ---- bottom line: mouse help and the heat-map legend
        self.bottom = QWidget()
        bottom = QHBoxLayout(self.bottom)
        bottom.setContentsMargins(4, 0, 4, 2)
        self.hint = QLabel(self.NAV_HINT)
        self.hint.setStyleSheet(HINT_STYLE)
        bottom.addWidget(self.hint)
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
        if self.edit_mode and doc.aligned and not self.rb_aligned.isChecked():
            self.rb_aligned.setChecked(True)
        # edits are made where the scans are aligned, so Edit always shows that
        self.rb_orig.setEnabled(not (self.edit_mode and doc.aligned))
        model = self.colour_mode.model()
        model.item(1).setEnabled(doc.quality is not None)
        have = {k: any(k in (s.attrs or {}) for s in doc.scans)
                or any(k in (a or {}) for a in doc.export_attrs.values())
                for k in ("intensity", "rgb")}
        model.item(3).setEnabled(have["intensity"])
        model.item(4).setEnabled(have["rgb"])
        k = self.colour_mode.currentIndex()
        if (k == 1 and doc.quality is None) or (k == 3 and not have["intensity"]) \
                or (k == 4 and not have["rgb"]):
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
        self._redraw_soon.stop()
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
        mode = self.colour_mode.currentIndex()
        height = mode == 2 and self._height_range(aligned)
        attr = {3: "intensity", 4: "rgb"}.get(mode)
        irange = self._intensity_range() if attr == "intensity" else None
        for i, (xyz, scan) in enumerate(zip(previews, doc.scans)):
            if not scan.entry.visible or not len(xyz):
                continue
            proj, hidden = self.project_points(i)
            pts = proj if aligned else xyz
            if height:
                lo, span = height
                color = error_colours(np.clip(pts[:, 2] - lo, 0.0, span), span)
            pts = (pts - origin).astype(np.float32)
            vals = (scan.attrs or {}).get(attr) if attr else None
            if height:
                pass  # coloured above, from the real heights
            elif vals is not None and len(vals) >= len(pts):
                color = _attr_colours(attr, vals[:len(pts)], irange)
            elif heat and len(q.point_error[i]) >= len(pts):
                color = error_colours(q.point_error[i][:len(pts)], q.scale_max)
            else:
                color = (*cloud_colour(i), 1.0)
            keep = ~hidden
            if hidden.any() and self.edit_mode and self.show_deleted:
                self._add_points(pts[hidden], DELETED_COLOUR, size)
            if aligned and self._removed is not None and i < len(self._removed) \
                    and len(self._removed[i]) == len(pts):
                gone = keep & ~self._removed[i]
                if gone.any() and not self.removed_hidden:
                    self._add_points(pts[gone], REMOVED_COLOUR, size + 1)
                keep &= self._removed[i]
            sel = self._sel_masks.get(i) if self.edit_mode else None
            if sel is not None and len(sel) == len(pts) and sel.any():
                self._add_points(pts[sel], SELECT_COLOUR, size + 1)
                keep &= ~sel
            if not keep.all():
                pts = pts[keep]
                if isinstance(color, np.ndarray) and color.ndim == 2:
                    color = color[keep]
            self._add_points(pts, color, size)
        for xyz, attrs in zip(doc.visible_exports(), doc.visible_export_attrs()):
            vals = (attrs or {}).get(attr) if attr else None
            if height:
                lo, span = height
                color = error_colours(np.clip(xyz[:, 2] - lo, 0.0, span), span)
            elif vals is not None and len(vals) == len(xyz):
                color = _attr_colours(attr, vals, irange)
            else:
                color = (*EXPORT_COLOUR, 1.0)
            self._add_points((xyz - origin).astype(np.float32), color, size)

    def _intensity_range(self):
        """(low, high) intensity over what is shown, ignoring the odd outlier."""
        vals = [s.attrs["intensity"] for s in self.doc.scans
                if s.entry.visible and s.attrs and "intensity" in s.attrs]
        vals += [a["intensity"] for a in self.doc.visible_export_attrs()
                 if a and "intensity" in a]
        v = np.concatenate([x[:200_000] for x in vals]) if vals else np.zeros(0)
        v = v[np.isfinite(v)]
        if not len(v):
            return 0.0, 1.0
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        return float(lo), float(hi if hi > lo else lo + 1.0)

    def _height_range(self, aligned):
        """(lowest, span) of the shown points' heights, ignoring the odd stray."""
        zs = []
        for i, scan in enumerate(self.doc.scans):
            if scan.entry.visible and len(self.doc.previews()[i]):
                proj, hidden = self.project_points(i)
                pts = proj if aligned else self.doc.previews()[i]
                zs.append(pts[~hidden, 2] if aligned else pts[:, 2])
        z = np.concatenate(zs) if zs else np.zeros(0)
        if not len(z):
            return None
        lo, hi = np.percentile(z, 1), np.percentile(z, 99)
        return float(lo), float(max(hi - lo, 1e-6))

    def redraw_soon(self):
        """Redraw after a moment (several changes in a row draw once)."""
        if not self._redraw_soon.isActive():
            self._redraw_soon.start()

    # ---------------------------------------------------------------- edits

    def project_points(self, i: int):
        """Scan ``i``'s drawn points in the project frame (where the edits
        are), and which of them the edits hide (deleted, or in a hidden part)."""
        doc = self.doc
        xyz = doc.previews()[i]
        t = doc.transforms()[i]
        key = (doc.edit_version, id(doc.scans[i].sample), len(xyz), t.tobytes())
        hit = self._edit_cache.get(i)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        proj = xyz @ t[:3, :3].T + t[:3, 3]
        hidden = doc.hidden_by_edits(proj, i)
        self._edit_cache[i] = (key, proj, hidden)
        return proj, hidden

    def set_edit_mode(self, on: bool):
        self.edit_mode = on
        if not on:
            self.set_tool(None)
            self.set_selection([])
        self.update_controls(False)
        self.redraw()

    def set_tool(self, tool):
        if self.tool is not None:
            self.tool.cancel()
        self.tool = tool
        if self.view is not None:
            self.view.setCursor(tool.cursor if tool is not None else Qt.ArrowCursor)
            self.overlay.update()
        self.hint.setText(tool.hint if tool is not None else self.NAV_HINT)

    def set_selection(self, regions):
        self.selection = list(regions)
        self._sel_masks = {}
        for r in self.selection:
            self._or_selection(r)
        self.redraw()
        self.selection_changed.emit()

    def extend_selection(self, region):
        """Add one region to the selection (a brush dab, a lasso)."""
        self.selection.append(region)
        self._or_selection(region)
        self.redraw_soon()
        self.selection_changed.emit()

    def _or_selection(self, region):
        for i, scan in enumerate(self.doc.scans):
            if not scan.entry.visible:
                continue
            proj, hidden = self.project_points(i)
            if not len(proj):
                continue
            m = region.mask(proj)
            if not self.show_deleted:
                m &= ~hidden
            old = self._sel_masks.get(i)
            self._sel_masks[i] = m if old is None or len(old) != len(m) else old | m

    def selected_counts(self):
        """Per scan index: (drawn points selected, drawn points of the scan)."""
        return {i: (int(m.sum()), len(m)) for i, m in self._sel_masks.items()}

    def camera_mvp(self) -> np.ndarray:
        """Maps project-frame points to clip space with the current camera."""
        v = self.view
        vp = v.getViewport()
        m = v.projectionMatrix(vp, vp) * v.viewMatrix()
        a = np.array(m.data(), dtype=float).reshape(4, 4).T  # Qt stores columns
        shift = np.eye(4)
        shift[:3, 3] = -self.doc.origin()  # points are drawn relative to the origin
        return a @ shift

    def ndc(self, pos) -> np.ndarray:
        """A position in the view widget (logical pixels) in device coordinates."""
        v = self.view
        return np.array([2.0 * pos.x() / max(1, v.width()) - 1.0,
                         1.0 - 2.0 * pos.y() / max(1, v.height())])

    def pick(self, pos, radius_px: float = 7.0):
        """The nearest drawn point under the mouse (project frame), or None."""
        if self.view is None:
            return None
        mvp = self.camera_mvp()
        target = self.ndc(pos)
        half = np.array([self.view.width() / 2.0, self.view.height() / 2.0])
        best, best_w = None, np.inf
        for i, scan in enumerate(self.doc.scans):
            if not scan.entry.visible:
                continue
            proj, hidden = self.project_points(i)
            if not len(proj):
                continue
            pts = proj if self.show_deleted else proj[~hidden]
            clip = pts @ mvp[:3, :3].T + mvp[:3, 3]
            w = pts @ mvp[3, :3] + mvp[3, 3]
            front = w > 1e-9
            if not front.any():
                continue
            d = (clip[front, :2] / w[front, None] - target) * half
            near = np.einsum("ij,ij->i", d, d) <= radius_px * radius_px
            if near.any():
                ws = w[front][near]
                k = int(np.argmin(ws))
                if ws[k] < best_w:
                    best_w, best = ws[k], pts[front][near][k]
        return best

    def pixel_size(self) -> float:
        """Roughly how much one pixel spans in the scene, at the view's centre."""
        if self.view is None:
            return 0.01
        return float(self.view.pixelSize(self.view.opts["center"]))

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
        try:
            self._walk = FlyThroughWindow(scene, title, parent=self, on_exit=self.leave_walk)
            self._walk.on_gl_failed = lambda err: self._walk_in_window(scene, title, err)
            self.stack.addWidget(self._walk)
            self.stack.setCurrentWidget(self._walk)
        except Exception as e:  # pragma: no cover - depends on the graphics driver
            self._walk = None
            self._walk_in_window(scene, title, f"{type(e).__name__}: {e}")
            return
        self.toolbar.hide()
        self.bottom.hide()

    def _walk_in_window(self, scene, title, error: str):
        """Fallback: the walk-through in a window of its own (as before 0.9)."""
        from ..flythrough import FlyThroughWindow

        self._walk_window = FlyThroughWindow(scene, title)  # kept so it is not collected
        self._walk_window.show()
        self._walk_window.raise_()
        self.walk_fallback.emit(error)

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


class _Overlay(QWidget):
    """Draws the tool's outline (lasso, rectangle, brush circle) over the view."""

    def __init__(self, owner: View3D):
        super().__init__(owner.view)
        self.owner = owner
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.resize(owner.view.size())
        self.show()

    def paintEvent(self, _e):
        tool = self.owner.tool
        if tool is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tool.paint(p)
        p.end()


class _ToolFilter(QObject):
    """Sends the 3D view's mouse to the chosen Edit tool. Alt + drag (and the
    right and middle buttons, and the wheel) still move the camera."""

    def __init__(self, owner: View3D):
        super().__init__(owner)
        self.owner = owner

    def eventFilter(self, obj, ev):
        owner = self.owner
        t = ev.type()
        if t == QEvent.Resize and owner.overlay is not None:
            owner.overlay.resize(obj.size())
            return False
        tool = owner.tool
        if tool is None:
            return False
        if t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            tool.cancel()
            owner.overlay.update()
            return False
        if t in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            if ev.modifiers() & Qt.AltModifier:
                return False
            if t == QEvent.MouseButtonPress:
                if ev.button() != Qt.LeftButton:
                    return False
                done = tool.press(ev)
            elif t == QEvent.MouseMove:
                if ev.buttons() & (Qt.RightButton | Qt.MiddleButton):
                    return False
                done = tool.move(ev)
            else:
                if ev.button() != Qt.LeftButton:
                    return False
                done = tool.release(ev)
            owner.overlay.update()
            return bool(done)
        if t == QEvent.Leave:
            tool.leave()
            owner.overlay.update()
        return False


def _attr_colours(attr: str, vals: np.ndarray, irange) -> np.ndarray:
    """RGBA colours from intensity (grey, dark to light) or 8-bit RGB."""
    out = np.ones((len(vals), 4), np.float32)
    if attr == "rgb":
        out[:, :3] = np.asarray(vals, np.float32) / 255.0
    else:
        lo, hi = irange
        g = np.clip((np.asarray(vals, np.float64) - lo) / (hi - lo), 0.0, 1.0)
        g = np.nan_to_num(g, nan=0.0) ** 0.8 * 0.92 + 0.08  # never quite black
        out[:, 0] = out[:, 1] = out[:, 2] = g
    return out


def _swatch(rgb) -> QPixmap:
    pm = QPixmap(12, 12)
    pm.fill(QColor.fromRgbF(*rgb))
    return pm
