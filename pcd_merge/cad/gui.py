"""The Model mode: a fourth tab next to Align, Edit and Inspect.

:func:`install` adds it to the main window. Nothing else in the app knows
about it, so deleting this package removes it (see ``docs/model-mode.md``).

The panel has three parts:

* **Levels**: find the building's floors and ceilings, and adjust them.
* **Plans and sections**: slice the site, trace the walls into lines, and
  save them as DXF for any CAD program.
* **3D model**: fit walls, slabs, roofs, columns and pipes, check them in
  the 3D view, and save them as IFC (BIM), STEP (mechanical CAD) or 3D DXF.

Measurements snap to the model's corners, edges and faces while it is
shown.
"""

from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .levels import Level
from .model import KINDS, Model
from .plan import Drawing, TraceSettings
from .snap import Snapper

HINT_STYLE = "color: #888;"
ROLE_ID = Qt.UserRole + 1
KIND_COLOURS = {
    "wall": (0.95, 0.95, 0.95, 1.0),
    "slab": (0.45, 0.85, 0.45, 1.0),
    "roof": (0.95, 0.45, 0.35, 1.0),
    "column": (0.35, 0.80, 1.00, 1.0),
    "pipe": (0.95, 0.45, 0.95, 1.0),
    "plane": (0.75, 0.75, 0.40, 1.0),
}
PLAN_COLOUR = (1.0, 0.60, 0.10, 1.0)
PICKED_COLOUR = (1.0, 0.85, 0.10, 1.0)
DETAILS = [("Fine (3 cm)", 0.03), ("Normal (5 cm)", 0.05), ("Coarse (10 cm)", 0.10)]
SNAP_PX = 12  # snap within this many pixels


def _spin(lo, hi, value, step=0.05, decimals=2, suffix=" m", tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    return s


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(HINT_STYLE)
    return lab


# ---------------------------------------------------------------- background jobs
# (plain functions: they run on a worker thread with a progress callback)


def _levels_job(clouds, sensors, *, progress):
    from .levels import find_levels

    progress("Looking for floors and ceilings", 0.2)
    xyz = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    sz = np.concatenate([np.full(len(c), np.nan if s is None else float(s[2]))
                         for c, s in zip(clouds, sensors)]) if clouds else np.zeros(0)
    return find_levels(xyz, sz)


def _plan_job(sources, edits, z, thickness, settings, name, *, progress):
    from .plan import make_plan

    return make_plan(sources, edits, z, thickness, settings, name, progress=progress)


def _section_job(sources, edits, a, b, thickness, settings, name, *, progress):
    from .plan import make_section

    return make_section(sources, edits, a, b, thickness, settings, name, progress=progress)


def _all_plans_job(sources, edits, levels, above, thickness, settings, base, points, *,
                   progress):
    """A plan of every level, each saved as its own DXF next to ``base``."""
    from .dxf import write_drawings
    from .plan import plan_band, plan_from_points
    from .points import gather

    cuts = [lv.cut_height(above) for lv in levels]
    lo = min(plan_band(z, thickness)[0] for z in cuts)
    hi = max(plan_band(z, thickness)[1] for z in cuts)

    def within(xyz):  # only the bands, not the floors in between
        keep = np.zeros(len(xyz), dtype=bool)
        for z in cuts:
            a, b = plan_band(z, thickness)
            keep |= (xyz[:, 2] >= a) & (xyz[:, 2] <= b)
        return keep

    pts = gather(sources, edits, zmin=lo, zmax=hi, within=within,
                 progress=lambda m, f: progress(m, 0.6 * f))
    stem, _ext = os.path.splitext(base)
    out = []
    for k, (lv, z) in enumerate(zip(levels, cuts)):
        progress(f"Tracing {lv.name}", 0.6 + 0.4 * k / max(len(levels), 1))
        d = plan_from_points(pts.xyz, z, thickness, settings, lv.name)
        path = f"{stem} - {lv.name}.dxf"
        write_drawings(path, [d], points=points)
        out.append((path, d))
    return out


def _fit_job(sources, edits, settings, levels=None, *, progress):
    from .fit import fit_model
    from .points import TooManyPoints, gather

    try:
        site = gather(sources, edits, cell=settings.detail, per_source=True,
                      limit=settings.max_points, progress=lambda m, f: progress(m, 0.4 * f))
    except TooManyPoints as e:
        raise ValueError(
            f"That is over {e.limit:,} points at this detail, more than the fitting can "
            "take. Choose a coarser detail, or split the site into parts in Edit mode and "
            "fit one part at a time (only ticked parts are used).") from None
    elements, _faces = fit_model(site, settings, levels=levels or None,
                                 progress=lambda m, f: progress(m, 0.4 + 0.6 * f))
    return elements


# ---------------------------------------------------------------- the panel


class ModelPanel(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.doc = win.doc
        self.view = win.view
        self.model = Model()
        self.drawing: Drawing | None = None
        self._items = []  # what this panel draws in the 3D view
        self._active = False
        self.snapper = Snapper()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)
        lay.addWidget(_hint("Editable CAD geometry from the scans, not a mesh: floor plans and "
                            "sections as lines, and walls, floors, roofs, columns and pipes as "
                            "simple solids. Uses the ticked scans, aligned, without deleted "
                            "points or hidden parts."))

        # ---- levels
        g = QGroupBox("Levels")
        gl_ = QVBoxLayout(g)
        self.b_levels = QPushButton("Find levels")
        self.b_levels.setToolTip("Find the floors and ceilings of the building: large flat "
                                 "horizontal surfaces. A floor is one the scanners stood above.")
        self.b_levels.clicked.connect(self.find_levels)
        gl_.addWidget(self.b_levels)
        self.level_list = QListWidget()
        self.level_list.setMaximumHeight(110)
        self.level_list.setToolTip("Double-click to rename.")
        self.level_list.currentRowChanged.connect(self._level_selected)
        self.level_list.itemChanged.connect(self._level_renamed)
        gl_.addWidget(self.level_list)
        form = QFormLayout()
        self.level_floor = _spin(-1e5, 1e5, 0.0, 0.01, tip="Height of this level's floor.")
        self.level_ceiling = _spin(-1e5, 1e5, 0.0, 0.01,
                                   tip="Height of this level's ceiling (0: none found).")
        self.level_floor.editingFinished.connect(self._level_edited)
        self.level_ceiling.editingFinished.connect(self._level_edited)
        form.addRow("Floor at", self.level_floor)
        form.addRow("Ceiling at", self.level_ceiling)
        gl_.addLayout(form)
        row = QHBoxLayout()
        self.b_add_level = QPushButton("Add")
        self.b_add_level.setToolTip("Add a level by hand (set its floor height above).")
        self.b_add_level.clicked.connect(self.add_level)
        self.b_del_level = QPushButton("Delete")
        self.b_del_level.clicked.connect(self.delete_level)
        row.addWidget(self.b_add_level)
        row.addWidget(self.b_del_level)
        row.addStretch(1)
        gl_.addLayout(row)
        lay.addWidget(g)

        # ---- plans and sections
        g = QGroupBox("Floor plans and sections (DXF)")
        gl_ = QVBoxLayout(g)
        form = QFormLayout()
        self.plan_level = QComboBox()
        self.plan_level.setToolTip("The level to cut, or a height of your own.")
        self.plan_level.currentIndexChanged.connect(self._plan_level_changed)
        form.addRow("Level", self.plan_level)
        self.cut_above = _spin(0.05, 20, 1.2, 0.05, tip="How far above the floor the plan "
                               "cuts: about 1.2 m goes through walls, doors and windows.")
        form.addRow("Cut above floor", self.cut_above)
        self.cut_height = _spin(-1e5, 1e5, 1.2, 0.05, tip="The height the plan cuts at.")
        form.addRow("Cut at height", self.cut_height)
        self.thickness = _spin(0.02, 1.0, 0.10, 0.01, tip="How thick a slice is taken. Thicker "
                               "catches more points on sparse scans, thinner is crisper.")
        form.addRow("Slice thickness", self.thickness)
        self.close_gaps = _spin(0.0, 2.0, 0.30, 0.05, tip="Pieces of one wall closer than this "
                                "are joined into one line. Openings wider than this (doors) "
                                "stay open.")
        form.addRow("Close gaps up to", self.close_gaps)
        gl_.addLayout(form)
        self.cb_square = QCheckBox("Square up to the building")
        self.cb_square.setChecked(True)
        self.cb_square.setToolTip("Lines within 3° of the building's main directions are "
                                  "turned onto them, so corners are exactly square.")
        gl_.addWidget(self.cb_square)
        self.cb_dxf_points = QCheckBox("Put the slice's points in the DXF too")
        self.cb_dxf_points.setChecked(True)
        self.cb_dxf_points.setToolTip("On a layer of their own (SCAN), to trace what the "
                                      "automatic lines missed. Turn the layer off to hide them.")
        gl_.addWidget(self.cb_dxf_points)
        row = QHBoxLayout()
        self.b_plan = QPushButton("Make plan")
        self.b_plan.clicked.connect(self.make_plan)
        self.b_all_plans = QPushButton("Plans of every level…")
        self.b_all_plans.setToolTip("Make a plan of each level and save each as its own DXF.")
        self.b_all_plans.clicked.connect(self._all_plans_dialog)
        row.addWidget(self.b_plan)
        row.addWidget(self.b_all_plans)
        gl_.addLayout(row)
        gl_.addWidget(_hint("Section: measure a distance across the building (Measure, M), "
                            "then cut along it."))
        row = QHBoxLayout()
        self.section_line = QComboBox()
        self.section_line.setToolTip("The measurement the section runs along, from its first "
                                     "point to its second.")
        row.addWidget(self.section_line, 1)
        self.b_section = QPushButton("Make section")
        self.b_section.clicked.connect(self.make_section)
        row.addWidget(self.b_section)
        gl_.addLayout(row)
        self.preview = self._make_preview()
        if self.preview is not None:
            gl_.addWidget(self.preview)
        self.drawing_info = QLabel()
        self.drawing_info.setWordWrap(True)
        gl_.addWidget(self.drawing_info)
        self.b_dxf = QPushButton("Save DXF…")
        self.b_dxf.clicked.connect(self._dxf_dialog)
        gl_.addWidget(self.b_dxf)
        lay.addWidget(g)

        # ---- 3D model
        g = QGroupBox("3D model (IFC, STEP)")
        gl_ = QVBoxLayout(g)
        form = QFormLayout()
        self.detail = QComboBox()
        for label, _v in DETAILS:
            self.detail.addItem(label)
        self.detail.setCurrentIndex(1)
        self.detail.setToolTip("The scan is thinned to one point per cube of this size first. "
                               "Finer finds smaller things but takes longer; a big site may "
                               "need Coarse, or fitting one part at a time.")
        form.addRow("Detail", self.detail)
        self.wall_thick = _spin(0.02, 2.0, 0.15, 0.01, tip="Walls seen from one side only "
                                "(outside walls scanned from inside) get this thickness. Walls "
                                "seen from both sides get their real thickness.")
        form.addRow("One-sided walls", self.wall_thick)
        self.slab_thick = _spin(0.02, 2.0, 0.20, 0.01, tip="Floors, ceilings and roofs seen "
                                "from one side get this thickness. A ceiling with a floor just "
                                "above it gets the real slab thickness.")
        form.addRow("One-sided slabs", self.slab_thick)
        gl_.addLayout(form)
        self.cb_planes = QCheckBox("Walls, floors and roofs")
        self.cb_planes.setChecked(True)
        self.cb_cyl = QCheckBox("Columns and pipes")
        self.cb_cyl.setChecked(True)
        self.cb_clutter = QCheckBox("Leave out shelves and furniture")
        self.cb_clutter.setChecked(True)
        self.cb_clutter.setToolTip(
            "Only keep floors and ceilings at the building's levels, and walls that reach the "
            "ceiling or stand a storey high, so shelving, cupboards and tables are not "
            "modelled. A wall hidden lower down by shelving is carried down to the floor. "
            "Uses the levels above, or finds them if there are none.")
        gl_.addWidget(self.cb_planes)
        gl_.addWidget(self.cb_clutter)
        gl_.addWidget(self.cb_cyl)
        self.b_fit = QPushButton("Fit shapes")
        self.b_fit.setToolTip("Fit walls, floors, roofs, columns and pipes to the scan. "
                              "Replaces the shapes fitted before.")
        self.b_fit.clicked.connect(self.fit)
        gl_.addWidget(self.b_fit)
        self.element_list = QListWidget()
        self.element_list.setMinimumHeight(140)
        self.element_list.setToolTip("Tick to show; select one to see it in yellow. Delete "
                                     "removes a wrong one; it is not exported.")
        self.element_list.itemChanged.connect(self._element_ticked)
        self.element_list.currentItemChanged.connect(self._element_selected)
        gl_.addWidget(self.element_list)
        self.element_info = QLabel()
        self.element_info.setWordWrap(True)
        gl_.addWidget(self.element_info)
        row = QHBoxLayout()
        self.b_del_el = QPushButton("Delete")
        self.b_del_el.clicked.connect(self.delete_element)
        self.b_clear = QPushButton("Clear all")
        self.b_clear.clicked.connect(self.clear_elements)
        row.addWidget(self.b_del_el)
        row.addWidget(self.b_clear)
        row.addStretch(1)
        gl_.addLayout(row)
        row = QHBoxLayout()
        self.b_ifc = QPushButton("Save IFC…")
        self.b_ifc.setToolTip("For BIM programs: Revit, ArchiCAD, FreeCAD, BlenderBIM, "
                              "SketchUp. Real walls, slabs, columns and pipes on their levels.")
        self.b_ifc.clicked.connect(lambda: self._export_dialog("ifc"))
        self.b_step = QPushButton("Save STEP…")
        self.b_step.setToolTip("For mechanical CAD: Fusion, SolidWorks, Inventor, Onshape, "
                               "FreeCAD, Rhino. Exact solids.")
        self.b_step.clicked.connect(lambda: self._export_dialog("step"))
        self.b_dxf3d = QPushButton("Save 3D DXF…")
        self.b_dxf3d.setToolTip("The solids as 3D faces, one layer per kind, for programs "
                                "that read neither IFC nor STEP.")
        self.b_dxf3d.clicked.connect(lambda: self._export_dialog("dxf"))
        row.addWidget(self.b_ifc)
        row.addWidget(self.b_step)
        row.addWidget(self.b_dxf3d)
        gl_.addLayout(row)
        self.cb_show = QCheckBox("Show the model in the 3D view")
        self.cb_show.setChecked(True)
        self.cb_show.toggled.connect(self.draw)
        gl_.addWidget(self.cb_show)
        self.cb_snap = QCheckBox("Measurements snap to the model")
        self.cb_snap.setChecked(True)
        self.cb_snap.setToolTip("While the model or a plan is shown, a measurement click near "
                                "a corner or edge snaps to it, and a click on a wall or floor "
                                "lands on its fitted plane instead of one rough scan point.")
        gl_.addWidget(self.cb_snap)
        lay.addWidget(g)
        lay.addStretch(1)

        self.refresh()

    # ---------------------------------------------------------------- helpers

    def _make_preview(self):
        try:
            import pyqtgraph as pg
        except Exception:  # pragma: no cover
            return None
        w = pg.PlotWidget()
        w.setMinimumHeight(220)
        w.setAspectLocked(True)
        w.showGrid(x=True, y=True, alpha=0.15)
        w.setBackground((25, 25, 28))
        w.setToolTip("The traced lines (orange) over the slice (grey). Wheel to zoom, drag "
                     "to pan.")
        self._pts_item = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(150, 150, 150, 160))
        self._line_item = pg.PlotDataItem(pen=pg.mkPen((255, 150, 30), width=2), connect="pairs")
        self._circle_item = pg.PlotDataItem(pen=pg.mkPen((80, 200, 255), width=2),
                                            connect="pairs")
        w.addItem(self._pts_item)
        w.addItem(self._line_item)
        w.addItem(self._circle_item)
        return w

    def _start_dir(self, name: str) -> str:
        p = self.doc.project
        return os.path.join(p.folder, name) if p.saved else name

    def _ready(self) -> bool:
        """Scans to work on, in one frame."""
        doc = self.doc
        if not doc.scans and not any(e.get("visible") for e in doc.project.exports):
            QMessageBox.information(self, "PCD Merge", "Add scans first.")
            return False
        movable = [s for s in doc.scans if not s.entry.base]
        if len(movable) + (1 if doc.base else 0) > 1 and not doc.aligned:
            QMessageBox.information(self, "PCD Merge", "Align the scans first (Align mode): "
                                    "the model is made from the scans in their aligned "
                                    "positions.")
            return False
        return True

    def sources(self):
        """The ticked scans (and ticked merged clouds), to read from."""
        from ..readers import MemorySource, open_scan
        from .points import Source

        doc = self.doc
        tfs = doc.transforms()
        out = []
        for i, s in enumerate(doc.scans):
            if not s.entry.visible:
                continue
            src = s.source()
            if not hasattr(src, "chunks"):  # a loaded cloud
                src = MemorySource(src)
            out.append(Source(src, tfs[i], s.entry.key, s.sensor_origin(tfs[i]), s.name))
        for e in doc.project.exports:
            if e.get("visible") and os.path.isfile(e.get("path", "")):
                out.append(Source(open_scan(e["path"]), np.eye(4), None, None,
                                  os.path.basename(e["path"]), use_edits=False))
        return out

    def trace_settings(self) -> TraceSettings:
        return TraceSettings(close_gaps=self.close_gaps.value(), square=self.cb_square.isChecked())

    def changed(self) -> None:
        """The model changed: keep it, redraw, refresh the lists."""
        self.model.settings.update(self._settings())
        if self.doc.project.saved:
            try:
                self.model.save(self.doc.project.folder)
            except OSError as e:
                self.win.tasks.set_status(f"The model could not be saved: {e}")
        self.refresh()
        self.draw()

    def _settings(self) -> dict:
        return {"cut_above": self.cut_above.value(), "thickness": self.thickness.value(),
                "close_gaps": self.close_gaps.value(), "square": self.cb_square.isChecked(),
                "detail": self.detail.currentIndex(), "wall_thickness": self.wall_thick.value(),
                "slab_thickness": self.slab_thick.value(),
                "skip_clutter": self.cb_clutter.isChecked()}

    def load_settings(self) -> None:
        s = self.model.settings
        self.cut_above.setValue(float(s.get("cut_above", 1.2)))
        self.thickness.setValue(float(s.get("thickness", 0.10)))
        self.close_gaps.setValue(float(s.get("close_gaps", 0.30)))
        self.cb_square.setChecked(bool(s.get("square", True)))
        self.detail.setCurrentIndex(int(s.get("detail", 1)))
        self.wall_thick.setValue(float(s.get("wall_thickness", 0.15)))
        self.slab_thick.setValue(float(s.get("slab_thickness", 0.20)))
        self.cb_clutter.setChecked(bool(s.get("skip_clutter", True)))

    # ---------------------------------------------------------------- the project

    def project_changed(self) -> None:
        """Another project was opened (or a new one started)."""
        p = self.doc.project
        self.model = Model.load(p.folder) if p.saved else Model()
        self.drawing = None
        self.load_settings()
        self._show_drawing()
        self.refresh()
        self.draw()

    def project_saved(self) -> None:
        p = self.doc.project
        if p.saved and (self.model.levels or self.model.elements) \
                and not os.path.isfile(Model.path(p.folder)):
            self.model.save(p.folder)  # first save of a new project

    def doc_changed(self, what: str) -> None:
        if what == "project":
            self.project_changed()
            return
        if what == "measurements":
            self._fill_sections()
        if what in ("scans", "alignment", "visibility", "exports", "project"):
            self.draw()
        self.update_buttons(self.win.tasks.busy)

    # ---------------------------------------------------------------- levels

    def find_levels(self) -> None:
        if not self._ready():
            return
        doc = self.doc
        tfs = doc.transforms()
        clouds, sensors = [], []
        for i, s in enumerate(doc.scans):
            if not s.entry.visible or not len(s.sample):
                continue
            t = tfs[i]
            xyz = s.sample @ t[:3, :3].T + t[:3, 3]
            xyz = xyz[~doc.hidden_by_edits(xyz, i)]
            clouds.append(xyz)
            sensors.append(s.sensor_origin(t))
        for xyz in doc.visible_exports():
            clouds.append(np.asarray(xyz, float))
            sensors.append(None)
        self.win.tasks.run(_levels_job, (clouds, sensors), self._levels_found,
                           "Finding levels…")

    def _levels_found(self, levels) -> None:
        if not levels:
            self.win.tasks.set_status("No floors found: there is no large level surface. Add "
                                      "a level by hand.")
            return
        self.model.levels = levels
        self.model.assign_levels()
        self.changed()
        self.level_list.setCurrentRow(0)
        self.win.tasks.set_status(f"Found {len(levels)} level(s): " + ", ".join(
            f"{lv.name} at {lv.floor:.2f} m" for lv in levels) + ".")

    def _level_selected(self, row: int) -> None:
        ok = 0 <= row < len(self.model.levels)
        for w in (self.level_floor, self.level_ceiling, self.b_del_level):
            w.setEnabled(ok)
        if ok:
            lv = self.model.levels[row]
            for w, v in ((self.level_floor, lv.floor), (self.level_ceiling, lv.ceiling or 0.0)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    def _level_edited(self) -> None:
        row = self.level_list.currentRow()
        if not 0 <= row < len(self.model.levels):
            return
        lv = self.model.levels[row]
        f, c = self.level_floor.value(), self.level_ceiling.value()
        new_c = c if c > f + 0.5 else None
        if (lv.floor, lv.ceiling) != (f, new_c):
            lv.floor, lv.ceiling = f, new_c
            self._sort_levels(lv)

    def _level_renamed(self, item) -> None:
        row = self.level_list.row(item)
        if 0 <= row < len(self.model.levels):
            name = item.text().split("  ")[0].strip()
            if name and name != self.model.levels[row].name:
                self.model.levels[row].name = name
                self.changed()

    def add_level(self) -> None:
        top = max((lv.ceiling or lv.floor for lv in self.model.levels), default=None)
        z = 0.0 if top is None else top + 0.3
        lv = Level(f"Level {len(self.model.levels)}", z)
        self.model.levels.append(lv)
        self._sort_levels(lv)

    def _sort_levels(self, keep) -> None:
        self.model.levels.sort(key=lambda lv: lv.floor)
        self.model.assign_levels()
        self.changed()
        self.level_list.setCurrentRow(self.model.levels.index(keep))

    def delete_level(self) -> None:
        row = self.level_list.currentRow()
        if 0 <= row < len(self.model.levels):
            del self.model.levels[row]
            self.model.assign_levels()
            self.changed()

    # ---------------------------------------------------------------- plans and sections

    def _plan_level_changed(self, *_):
        custom = self.plan_level.currentIndex() >= len(self.model.levels)
        self.cut_height.setEnabled(custom)
        self.cut_above.setEnabled(not custom)
        if not custom:
            lv = self.model.levels[self.plan_level.currentIndex()]
            self.cut_height.setValue(lv.cut_height(self.cut_above.value()))

    def plan_height(self) -> tuple[float, str]:
        k = self.plan_level.currentIndex()
        if 0 <= k < len(self.model.levels):
            lv = self.model.levels[k]
            return lv.cut_height(self.cut_above.value()), lv.name
        z = self.cut_height.value()
        return z, f"Plan at {z:.2f} m"

    def make_plan(self) -> None:
        if not self._ready():
            return
        z, name = self.plan_height()
        self.win.tasks.run(_plan_job, (self.sources(), self.doc.edit_snapshot(), z,
                                       self.thickness.value(), self.trace_settings(), name),
                           self._drawing_done, f"Making the plan of {name}…")

    def _sections(self):
        return [m for m in self.doc.measurements if m.kind == "distance"]

    def _fill_sections(self) -> None:
        cur = self.section_line.currentData()
        self.section_line.blockSignals(True)
        self.section_line.clear()
        for m in self._sections():
            self.section_line.addItem(m.label(), m.id)
        k = self.section_line.findData(cur)
        if k >= 0:
            self.section_line.setCurrentIndex(k)
        self.section_line.blockSignals(False)

    def make_section(self) -> None:
        if not self._ready():
            return
        m = self.doc.measurement(self.section_line.currentData()) \
            if self.section_line.currentData() else None
        if m is None:
            return
        a, b = np.asarray(m.points[0], float), np.asarray(m.points[1], float)
        if np.linalg.norm((b - a)[:2]) < 0.1:
            QMessageBox.information(self, "PCD Merge", "That measurement is (nearly) vertical: "
                                    "a section needs a line across the building.")
            return
        name = f"Section {m.name or m.label()}"
        self.win.tasks.run(_section_job, (self.sources(), self.doc.edit_snapshot(), a[:2], b[:2],
                                          self.thickness.value(), self.trace_settings(), name),
                           self._drawing_done, "Making the section…")

    def _drawing_done(self, d: Drawing) -> None:
        self.drawing = d
        self._show_drawing()
        self.draw()
        self.update_buttons(False)
        if not len(d.points):
            self.win.tasks.set_status("The slice is empty: no points at that height. Check "
                                      "the level, or the scans that are ticked.")
        else:
            self.win.tasks.set_status(f"{d.name}: {len(d.segments)} line(s), "
                                      f"{len(d.circles)} round thing(s). Save DXF… to keep it.")

    def _show_drawing(self) -> None:
        d = self.drawing
        if d is None:
            self.drawing_info.setText("")
            if self.preview is not None:
                self._pts_item.setData([], [])
                self._line_item.setData([], [])
                self._circle_item.setData([], [])
            return
        self.drawing_info.setText(
            f"<b>{d.name}</b>: {len(d.segments)} lines ({d.total_length():.1f} m), "
            f"{len(d.circles)} circles" + (f", cut at {d.z:.2f} m" if d.kind == "plan" else ""))
        if self.preview is None:
            return
        pts = d.points
        if len(pts) > 150_000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 150_000, replace=False)]
        self._pts_item.setData(pts[:, 0], pts[:, 1]) if len(pts) else self._pts_item.setData([], [])
        segs = np.asarray(d.segments, float).reshape(-1, 2)
        self._line_item.setData(segs[:, 0], segs[:, 1]) if len(segs) else \
            self._line_item.setData([], [])
        ring = []
        th = np.linspace(0, 2 * np.pi, 33)
        for x, y, r in np.asarray(d.circles, float).reshape(-1, 3):
            c = np.column_stack([x + r * np.cos(th), y + r * np.sin(th)])
            ring += [c[:-1], c[1:]]
        if ring:
            pairs = np.stack([np.concatenate(ring[0::2]), np.concatenate(ring[1::2])], 1)
            pairs = pairs.reshape(-1, 2)
            self._circle_item.setData(pairs[:, 0], pairs[:, 1])
        else:
            self._circle_item.setData([], [])
        self.preview.autoRange()

    def _dxf_dialog(self) -> None:
        if self.drawing is None:
            return
        safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in self.drawing.name)
        path, _ = QFileDialog.getSaveFileName(self, "Save DXF", self._start_dir(f"{safe}.dxf"),
                                              "DXF drawing (*.dxf)")
        if path:
            self.save_dxf(path)

    def save_dxf(self, path: str) -> None:
        from .dxf import write_drawings

        if not path.lower().endswith(".dxf"):
            path += ".dxf"
        n = write_drawings(path, [self.drawing], points=self.cb_dxf_points.isChecked())
        self.win.tasks.set_status(f"Saved {n['lines']} lines and {n['circles']} circles to "
                                  f"{path} (units: metres).")

    def _all_plans_dialog(self) -> None:
        if not self.model.levels:
            QMessageBox.information(self, "PCD Merge", "Find the levels first.")
            return
        if not self._ready():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the plans (one DXF per level, named after it)",
            self._start_dir("Plan.dxf"), "DXF drawing (*.dxf)")
        if path:
            self.save_all_plans(path)

    def save_all_plans(self, path: str) -> None:
        self.win.tasks.run(_all_plans_job, (self.sources(), self.doc.edit_snapshot(),
                                            list(self.model.levels), self.cut_above.value(),
                                            self.thickness.value(), self.trace_settings(), path,
                                            self.cb_dxf_points.isChecked()),
                           self._all_plans_done, "Making the plans…")

    def _all_plans_done(self, out) -> None:
        if out:
            self.drawing = out[0][1]
            self._show_drawing()
            self.draw()
        self.win.tasks.set_status(f"Saved {len(out)} plan(s): " + ", ".join(
            os.path.basename(p) for p, _ in out) + ".")

    # ---------------------------------------------------------------- 3D model

    def fit_settings(self):
        from .fit import FitSettings

        return FitSettings(detail=DETAILS[self.detail.currentIndex()][1],
                           wall_thickness=self.wall_thick.value(),
                           slab_thickness=self.slab_thick.value(),
                           planes=self.cb_planes.isChecked(), cylinders=self.cb_cyl.isChecked(),
                           skip_clutter=self.cb_clutter.isChecked())

    def fit(self) -> None:
        if not self._ready():
            return
        self.win.tasks.run(_fit_job, (self.sources(), self.doc.edit_snapshot(),
                                      self.fit_settings(), list(self.model.levels)),
                           self._fitted, "Fitting shapes…")

    def _fitted(self, elements) -> None:
        self.model.elements = list(elements)
        self.model.assign_levels()
        self.changed()
        c = self.model.counts()
        if not c:
            self.win.tasks.set_status("No shapes found. Try a finer detail, or check that the "
                                      "scans are ticked and aligned.")
            return
        self.win.tasks.set_status("Fitted " + ", ".join(
            f"{n} {KINDS.get(k, k).lower()}{'s' if n != 1 else ''}" for k, n in c.items()) + ".")

    def _element_ticked(self, item) -> None:
        e = self.model.element(item.data(ROLE_ID))
        if e is not None and e.visible != (item.checkState() == Qt.Checked):
            e.visible = item.checkState() == Qt.Checked
            self.changed()

    def _element_selected(self, *_):
        e = self.current_element()
        if e is None:
            self.element_info.setText("")
        else:
            lv = self.model.levels[e.level].name if e.level is not None and \
                e.level < len(self.model.levels) else "no level"
            self.element_info.setText(
                f"{e.label()}: {e.summary()}, on {lv}. Fitted to {e.points:,} points, "
                f"{e.rms * 1000:.1f} mm from them on average (RMS).")
        self.draw()
        self.update_buttons(self.win.tasks.busy)

    def current_element(self):
        it = self.element_list.currentItem()
        return self.model.element(it.data(ROLE_ID)) if it is not None else None

    def delete_element(self) -> None:
        e = self.current_element()
        if e is None:
            return
        row = self.element_list.currentRow()
        self.model.elements.remove(e)
        self.changed()
        if self.element_list.count():
            self.element_list.setCurrentRow(min(row, self.element_list.count() - 1))

    def clear_elements(self) -> None:
        if self.model.elements and QMessageBox.question(
                self, "PCD Merge", f"Remove all {len(self.model.elements)} fitted shapes?") \
                == QMessageBox.Yes:
            self.model.elements = []
            self.changed()

    def _export_dialog(self, fmt: str) -> None:
        if not self.model.elements:
            return
        name, filt = {"ifc": ("model.ifc", "IFC building model (*.ifc)"),
                      "step": ("model.step", "STEP solids (*.step *.stp)"),
                      "dxf": ("model-3d.dxf", "DXF 3D faces (*.dxf)")}[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "Save the model", self._start_dir(name), filt)
        if path:
            self.export(path, fmt)

    def export(self, path: str, fmt: str) -> None:
        from .. import __version__

        els = [e for e in self.model.elements if e.visible]
        ext = {"ifc": ".ifc", "step": ".step", "dxf": ".dxf"}[fmt]
        if not os.path.splitext(path)[1]:
            path += ext
        title = self.doc.project.name if self.doc.project.saved else "Scanned building"
        if fmt == "ifc":
            from .ifc import write_ifc

            write_ifc(path, els, self.model.levels, project=title, app_version=__version__)
        elif fmt == "step":
            from .step import write_step

            write_step(path, els, name=title)
        else:
            from .dxf import write_model

            write_model(path, els)
        self.win.tasks.set_status(f"Saved {len(els)} shape(s) to {path}.")

    # ---------------------------------------------------------------- lists

    def refresh(self) -> None:
        # levels
        row = self.level_list.currentRow()
        self.level_list.blockSignals(True)
        self.level_list.clear()
        for lv in self.model.levels:
            ceil = f", ceiling {lv.ceiling:.2f} m" if lv.ceiling is not None else ""
            it = QListWidgetItem(f"{lv.name}  (floor {lv.floor:.2f} m{ceil})")
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.level_list.addItem(it)
        self.level_list.blockSignals(False)
        if self.model.levels:
            self.level_list.setCurrentRow(min(max(row, 0), len(self.model.levels) - 1))
        self._level_selected(self.level_list.currentRow())
        # plan levels
        cur = self.plan_level.currentIndex()
        self.plan_level.blockSignals(True)
        self.plan_level.clear()
        for lv in self.model.levels:
            self.plan_level.addItem(lv.name)
        self.plan_level.addItem("Custom height")
        self.plan_level.setCurrentIndex(cur if 0 <= cur < self.plan_level.count() else 0)
        self.plan_level.blockSignals(False)
        self._plan_level_changed()
        self._fill_sections()
        # elements
        sel = self.current_element()
        self.element_list.blockSignals(True)
        self.element_list.clear()
        numbers: dict[str, int] = {}
        for e in self.model.elements:
            numbers[e.kind] = numbers.get(e.kind, 0) + 1
            name = e.name or f"{KINDS.get(e.kind, e.kind)} {numbers[e.kind]}"
            it = QListWidgetItem(f"{name}  ({e.summary()})")
            it.setData(ROLE_ID, e.id)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if e.visible else Qt.Unchecked)
            r, g, b, _ = KIND_COLOURS.get(e.kind, (1, 1, 1, 1))
            it.setForeground(QColor.fromRgbF(r * 0.8, g * 0.8, b * 0.8))
            self.element_list.addItem(it)
            if sel is not None and e.id == sel.id:
                self.element_list.setCurrentItem(it)
        self.element_list.blockSignals(False)
        self._element_selected()
        self.update_buttons(self.win.tasks.busy)

    def update_buttons(self, busy: bool) -> None:
        has = bool(self.doc.scans) or any(e.get("visible") for e in self.doc.project.exports)
        for b in (self.b_levels, self.b_plan, self.b_fit):
            b.setEnabled(not busy and has)
        self.b_all_plans.setEnabled(not busy and has and bool(self.model.levels))
        self.b_section.setEnabled(not busy and has and self.section_line.count() > 0)
        self.b_dxf.setEnabled(self.drawing is not None)
        els = bool(self.model.elements)
        for b in (self.b_ifc, self.b_step, self.b_dxf3d, self.b_clear):
            b.setEnabled(els and not busy)
        self.b_del_el.setEnabled(self.current_element() is not None)
        self.b_add_level.setEnabled(not busy)

    # ---------------------------------------------------------------- the 3D view

    def activate(self, on: bool) -> None:
        self._active = on

    def draw(self, *_):
        """Draw the model's wireframe and the traced lines over the scan."""
        view = self.view
        if view.view is None:
            return
        for it in self._items:
            try:
                view.view.removeItem(it)
            except Exception:
                pass
        self._items = []
        shown = self.cb_show.isChecked() and view.measurements_shown()
        self.snapper = Snapper()
        if not shown:
            return
        import pyqtgraph.opengl as gl
        from OpenGL.GL import GL_DEPTH_TEST

        origin = self.doc.origin()

        def add(segs, colour, width):
            segs = np.asarray(segs, float).reshape(-1, 3)
            if not len(segs):
                return
            item = gl.GLLinePlotItem(pos=(segs - origin).astype(np.float32), color=colour,
                                     width=width, antialias=True, mode="lines")
            item.setGLOptions("translucent")
            item.updateGLOptions({GL_DEPTH_TEST: False})
            item.setDepthValue(9)
            view.view.addItem(item)
            self._items.append(item)

        picked = self.current_element()
        by_kind: dict[str, list] = {}
        for e in self.model.elements:
            if e.visible and (picked is None or e.id != picked.id):
                by_kind.setdefault(e.kind, []).append(e.edges(16).reshape(-1, 3))
        for kind, parts in by_kind.items():
            add(np.concatenate(parts), KIND_COLOURS.get(kind, (1, 1, 1, 1)), 1.5)
        if picked is not None:
            add(picked.edges(24).reshape(-1, 3), PICKED_COLOUR, 3.0)
        drawings = []
        if self.drawing is not None and len(self.drawing.segments):
            d = self.drawing
            add(d.to_3d(np.asarray(d.segments).reshape(-1, 2)), PLAN_COLOUR, 2.5)
            th = np.linspace(0, 2 * np.pi, 25)
            for x, y, r in np.asarray(d.circles, float).reshape(-1, 3):
                ring = d.to_3d(np.column_stack([x + r * np.cos(th), y + r * np.sin(th)]))
                add(np.stack([ring[:-1], ring[1:]], 1), PLAN_COLOUR, 2.5)
            drawings.append(d)
        self.snapper = Snapper(self.model.elements, drawings)

    def snap(self, hit):
        """A measurement click: snapped to the model if it is shown."""
        if hit is None or not self.cb_snap.isChecked() or not self.snapper:
            return hit
        p, what = self.snapper.snap(hit, self.view.pixel_size() * SNAP_PX)
        if what:
            self.win.tasks.set_status(f"Snapped to the model's {what}.")
        return p


def install(win) -> ModelPanel:
    """Add Model mode to the main window: its tab, its panel, its menu entry,
    and snapping of measurements. Called once, from the window."""
    from ..gui.edit_tools import MeasureTool

    panel = ModelPanel(win)
    win.right.addWidget(panel)
    k = win.modes.addTab("Model")
    win.modes.setTabToolTip(k, "Make CAD geometry from the scans: floor plans and sections "
                               "as DXF lines, walls, floors, roofs, columns and pipes as IFC "
                               "or STEP solids.")
    win.modes.currentChanged.connect(lambda i: panel.activate(i == k))
    win.doc.changed.connect(panel.doc_changed)
    win.doc.saved.connect(panel.project_saved)
    win.tasks.busy_changed.connect(panel.update_buttons)
    # the View menu: Model mode, next to the other modes
    a = QAction("Model mode", win)
    a.setShortcut(QKeySequence(f"F{k + 2}"))
    a.triggered.connect(lambda: win.modes.setCurrentIndex(k))
    win.addAction(a)
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == "View" and act.menu() is not None:
            act.menu().addAction(a)
            break

    # measurements snap to the model: wrap the view's pick while measuring
    pick = win.view.pick

    def snapped_pick(pos, *args, **kw):
        hit = pick(pos, *args, **kw)
        if isinstance(win.view.tool, MeasureTool):
            hit = panel.snap(hit)
        return hit

    win.view.pick = snapped_pick
    win.model_panel = panel
    panel.project_changed()
    return panel
