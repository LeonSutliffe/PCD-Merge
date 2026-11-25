"""Align by hand, in the Align panel: when Align cannot match a scan, click
the same spots on it and on a scan that is already placed.

Picks are made where the scans are drawn now (the project frame), so the
correction found from them is applied on top of the scan's current
transform. Only the two scans are drawn while picking.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)

from ..manual import (MIN_PAIRS, ManualAlignError, manual_result, overlap, refine,
                      rigid_from_pairs)
from ..quality import assess, fmt_distance
from ..registration import auto_voxel_size
from .common import HINT_STYLE
from .edit_tools import PairTool

FIXED_MARK = (1.0, 0.85, 0.1, 1.0)  # picks on the fixed scan: yellow
MOVING_MARK = (1.0, 0.3, 0.85, 1.0)  # picks on the scan to move: magenta


def manual_align_job(samples, transforms, moved, fixed, correction, voxel, tighten, previews,
                     *, progress):
    """Apply the picked correction, tighten it, and grade the alignment.
    Returns (transforms, fitness, how far tightening moved it, quality)."""
    tfs = [np.asarray(t, float) for t in transforms]
    rough = correction @ tfs[moved]
    mov = samples[moved]
    fix = samples[fixed] @ tfs[fixed][:3, :3].T + tfs[fixed][:3, 3]
    if tighten:
        progress("Tightening the alignment", 0.1)
        t, fitness = refine(mov, fix, rough, voxel)
        shift = float(np.linalg.norm(
            (mov @ t[:3, :3].T + t[:3, 3]) - (mov @ rough[:3, :3].T + rough[:3, 3]),
            axis=1).mean())
    else:
        t, fitness, shift = rough, overlap(mov, fix, rough, voxel), 0.0
    tfs[moved] = t
    progress("Measuring alignment quality", 0.5)
    quality = assess(samples, tfs, voxel, preview_xyz=previews)
    return tfs, fitness, shift, quality


class ManualAlignBox(QGroupBox):
    def __init__(self, panel):
        super().__init__("Align by hand")
        self.panel = panel
        self.win = panel.win
        self.doc = panel.doc
        self.fixed_pts: list[np.ndarray] = []
        self.moving_pts: list[np.ndarray] = []
        self._stopping = False

        lay = QVBoxLayout(self)
        intro = QLabel("If Align can't place a scan, click the same spots (corners, poles, "
                       "door frames) on it and on a scan that is already right.")
        intro.setWordWrap(True)
        intro.setStyleSheet(HINT_STYLE)
        lay.addWidget(intro)
        grid = QGridLayout()
        self.moving = QComboBox()
        self.moving.setToolTip("The scan that is in the wrong place.")
        self.fixed = QComboBox()
        self.fixed.setToolTip("A scan that is already where it should be. It does not move.")
        grid.addWidget(QLabel("Move:"), 0, 0)
        grid.addWidget(self.moving, 0, 1)
        grid.addWidget(QLabel("Onto:"), 1, 0)
        grid.addWidget(self.fixed, 1, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        self.cb_level = QCheckBox("Scans are level (turn about the vertical only)")
        self.cb_level.setToolTip(
            "For scans from a levelled tripod: only the heading is found from the points, "
            "so two pairs are enough and a careless click can't tilt the scan.")
        self.cb_tighten = QCheckBox("Tighten automatically afterwards")
        self.cb_tighten.setChecked(True)
        self.cb_tighten.setToolTip(
            "After placing the scan from your points, fine-tune it on the scans themselves "
            "(ICP), so the clicks only need to be roughly right.")
        lay.addWidget(self.cb_level)
        lay.addWidget(self.cb_tighten)
        row = QHBoxLayout()
        self.b_pick = QPushButton("Pick points")
        self.b_pick.setCheckable(True)
        self.b_pick.setToolTip("Shows only the two scans. Click a spot on the fixed scan "
                               "(yellow marks), then the same spot on the scan to move "
                               "(magenta), and so on.")
        self.b_pick.toggled.connect(self._pick_toggled)
        self.b_undo = QPushButton("Undo point")
        self.b_undo.clicked.connect(self.undo_point)
        self.b_apply = QPushButton("Apply")
        self.b_apply.setProperty("primary", True)
        self.b_apply.clicked.connect(self.apply)
        row.addWidget(self.b_pick)
        row.addWidget(self.b_undo)
        row.addWidget(self.b_apply)
        lay.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet(HINT_STYLE)
        lay.addWidget(self.status)

        self.moving.currentIndexChanged.connect(self._scans_chosen)
        self.fixed.currentIndexChanged.connect(self._scans_chosen)
        self.win.view.tool_changed.connect(self._tool_changed)
        self.scans_changed()

    # ---------------------------------------------------------------- state

    @property
    def view(self):
        return self.win.view

    def scans_changed(self) -> None:
        """Scans added or removed: refill the lists, drop the picks."""
        self.stop_picking()
        names = [s.name for s in self.doc.scans]
        for combo, default in ((self.moving, 1), (self.fixed, 0)):
            keep = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            k = keep if 0 <= keep < len(names) else min(default, len(names) - 1)
            combo.setCurrentIndex(k)
            combo.blockSignals(False)
        if len(names) >= 2 and self.moving.currentIndex() == self.fixed.currentIndex():
            self.moving.setCurrentIndex(1 if self.fixed.currentIndex() == 0 else 0)
        self._update()

    def _scans_chosen(self, *_):
        picking = self.picking()
        self._clear_picks()
        if picking:
            self.view.set_only([self.fixed.currentIndex(), self.moving.currentIndex()])
        self._update()

    def picking(self) -> bool:
        return isinstance(self.view.tool, PairTool)

    def pairs(self) -> int:
        return min(len(self.fixed_pts), len(self.moving_pts))

    def need(self) -> int:
        return 2 if self.cb_level.isChecked() else MIN_PAIRS

    def update_buttons(self, busy: bool) -> None:
        self._busy = busy
        self._update()

    def _update(self):
        busy = getattr(self, "_busy", False)
        ok = (len(self.doc.scans) >= 2 and self.moving.currentIndex() >= 0
              and self.moving.currentIndex() != self.fixed.currentIndex())
        self.setEnabled(len(self.doc.scans) >= 2)
        self.b_pick.setEnabled(not busy and ok and self.view.view is not None)
        self.b_undo.setEnabled(bool(self.fixed_pts))
        self.b_apply.setEnabled(not busy and ok and self.pairs() >= self.need())
        if not ok and len(self.doc.scans) >= 2:
            self.status.setText("Choose two different scans.")
        elif self.picking():
            self.status.setText(self._progress_text())
        elif self.pairs() >= self.need():
            self.status.setText(self._progress_text())
        elif not self.fixed_pts:
            self.status.setText("")

    def _progress_text(self) -> str:
        n = self.pairs()
        text = f"{n} pair{'s' if n != 1 else ''} picked"
        if n >= self.need():
            try:
                fit = rigid_from_pairs(self.moving_pts[:n], self.fixed_pts[:n],
                                       level=self.cb_level.isChecked())
                text += f"; they agree within {fmt_distance(fit.rms)}"
                if n > self.need() and fit.residuals.max() > 3 * max(np.median(fit.residuals),
                                                                    1e-3):
                    text += f" (pair {fit.worst + 1} fits worst: a mis-click?)"
            except ManualAlignError as e:
                text += f" ({e})"
        if n < self.need():
            text += f". {self.need() - n} more needed."
        else:
            text += ". Pick more to be safe, or Apply."
        return text + "\n" + self.pick_hint()

    # ---------------------------------------------------------------- picking

    def _pick_toggled(self, on):
        if on:
            self.start_picking()
        else:
            self.stop_picking()

    def start_picking(self) -> None:
        if self.doc.aligned and not self.view.aligned_shown():
            self.view.show_aligned()
        self.view.set_only([self.fixed.currentIndex(), self.moving.currentIndex()])
        if not self.picking():
            self.view.set_tool(PairTool(self.view, self))
        self._sync_button()
        self._update()

    def stop_picking(self) -> None:
        """Stop picking and forget the picks (Esc, leaving Align, another tool)."""
        if self._stopping:
            return
        self._stopping = True
        try:
            if self.picking():
                self.view.set_tool(None)
            self._clear_picks()
            if self.view.only is not None:
                self.view.set_only(None)
            self._sync_button()
            self._update()
        finally:
            self._stopping = False

    def _tool_changed(self):
        if not self.picking() and (self.view.only is not None or self.fixed_pts):
            self.stop_picking()
        self._sync_button()

    def _sync_button(self):
        on = self.picking()
        if self.b_pick.isChecked() != on:
            self.b_pick.blockSignals(True)
            self.b_pick.setChecked(on)
            self.b_pick.blockSignals(False)

    def pick_scan(self) -> int:
        """The scan the next click is on."""
        if len(self.fixed_pts) > len(self.moving_pts):
            return self.moving.currentIndex()
        return self.fixed.currentIndex()

    def pick_hint(self) -> str:
        k = self.pairs() + 1
        if self.pick_scan() == self.fixed.currentIndex():
            return f"Click spot {k} on {self.fixed.currentText()} (the fixed scan)."
        return f"Now click the same spot {k} on {self.moving.currentText()}."

    def pick_missed(self) -> None:
        name = self.doc.scans[self.pick_scan()].name
        self.status.setText(f"No point of {name} there: click on that scan itself.\n"
                            + self.pick_hint())

    def add_pick(self, pt) -> None:
        pt = np.asarray(pt, float)
        if self.pick_scan() == self.fixed.currentIndex():
            self.fixed_pts.append(pt)
        else:
            self.moving_pts.append(pt)
        self._show_markers()
        self._update()
        if self.picking():
            self.view.hint.setText(self.view.tool.hint)

    def undo_point(self) -> None:
        if len(self.fixed_pts) > len(self.moving_pts):
            self.fixed_pts.pop()
        elif self.moving_pts:
            self.moving_pts.pop()
        self._show_markers()
        self._update()
        if self.picking():
            self.view.hint.setText(self.view.tool.hint)

    def _clear_picks(self):
        self.fixed_pts, self.moving_pts = [], []
        self._show_markers()

    def _show_markers(self):
        marks = [(p, FIXED_MARK, str(k + 1)) for k, p in enumerate(self.fixed_pts)]
        marks += [(p, MOVING_MARK, str(k + 1)) for k, p in enumerate(self.moving_pts)]
        self.view.set_markers(marks)

    # ---------------------------------------------------------------- apply

    def apply(self) -> None:
        n = self.pairs()
        moved, fixed = self.moving.currentIndex(), self.fixed.currentIndex()
        try:
            fit = rigid_from_pairs(self.moving_pts[:n], self.fixed_pts[:n],
                                   level=self.cb_level.isChecked())
        except ManualAlignError as e:
            self.status.setText(f"Can't place the scan: {e}.")
            return
        doc = self.doc
        samples = [s.sample for s in doc.scans]
        voxel = doc.result.voxel_size if doc.result is not None else auto_voxel_size(samples)
        self._applying = dict(moved=moved, fixed=fixed, pairs=n, rms=fit.rms, voxel=voxel,
                              names=(self.moving.currentText(), self.fixed.currentText()))
        self.stop_picking()
        self.win.tasks.run(manual_align_job,
                           (samples, doc.transforms(), moved, fixed, fit.transform, voxel,
                            self.cb_tighten.isChecked(), doc.previews()),
                           self._applied, "Aligning by hand…")

    def _applied(self, payload):
        tfs, fitness, shift, quality = payload
        a = self._applying
        doc = self.doc
        result = manual_result(doc.result, len(doc.scans), tfs, a["moved"], a["fixed"],
                               fitness, a["voxel"])
        moving, fixed = a["names"]
        lines = [f"{moving} was placed by hand onto {fixed} from {a['pairs']} pairs of points "
                 f"(they agreed within {fmt_distance(a['rms'])})."]
        if self.cb_tighten.isChecked():
            lines.append(f"Tightened automatically: it moved {fmt_distance(shift)} more; "
                         f"{fitness:.0%} of its points now lie on {fixed}.")
        else:
            lines.append(f"{fitness:.0%} of its points lie on {fixed}.")
        lines += [""] + quality.lines([s.name for s in doc.scans])[2:]
        warns = result.warnings()
        if warns:
            lines += ["", "Warnings:"] + ["  " + w for w in warns]
        doc.set_alignment(result, quality, lines)
        self.panel._reset_retries()
        self.panel.show_result()
        self.view.show_aligned(heat=quality.grade != "Good")
        self.panel.cleanup_changed()
        self.panel.status.emit(f"{moving} placed by hand, quality {quality.grade}. Check the "
                               "view, then merge, or pick again to improve it.")
        self.status.setText(lines[0] + (" " + lines[1] if len(lines) > 1 else ""))
