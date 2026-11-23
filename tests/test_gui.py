"""The app window end to end (headless): add scans, align, save the project,
reopen it from the cache, merge from disk."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from pcd_merge.exporters import write_cloud  # noqa: E402
from pcd_merge.readers import read_cloud  # noqa: E402
from pcd_merge.synthetic import make_scan, make_scene, make_transform  # noqa: E402


@pytest.fixture
def scans(tmp_path):
    scene = make_scene(seed=0, density=60)
    poses = [make_transform(0.0, 0.0, (-4, 0, 1.5)), make_transform(6.0, 0.02, (3, 2, 1.5))]
    paths = []
    for i, (p, ext) in enumerate(zip(poses, (".pcd", ".laz"))):
        path = str(tmp_path / f"scan{i}{ext}")
        write_cloud(path, make_scan(scene, p[:2, 3], 16, p, seed=i))
        paths.append(path)
    return paths


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("PCD_MERGE_CONFIG_DIR", str(tmp_path / "cfg"))
    from pcd_merge.gui import MainWindow
    from pcd_merge.gui import main_window as mw

    # never ask "save first?" about an untitled project (a modal box would hang the test)
    monkeypatch.setattr(mw.QMessageBox, "question",
                        lambda *a, **k: mw.QMessageBox.Discard)
    wins = []

    def make():
        w = MainWindow()
        qtbot.addWidget(w)
        wins.append(w)
        return w

    yield make
    for w in wins:
        w.doc.dirty = False  # no "save first?" question when the test closes it


def _wait(qtbot, win, timeout=120_000):
    qtbot.waitUntil(lambda: not win.tasks.busy, timeout=timeout)


def test_layout_has_modes_tree_view_and_panels(window):
    win = window()
    assert [win.modes.tabText(i) for i in range(win.modes.count())] == ["Align", "Edit",
                                                                         "Inspect"]
    win.modes.setCurrentIndex(1)
    assert win.right.currentWidget() is win.edit_panel
    win.modes.setCurrentIndex(0)
    assert win.right.currentWidget() is win.align_panel
    assert not win.align_panel.b_align.isEnabled()
    assert "Untitled" in win.windowTitle()


def test_align_save_reopen_and_merge(window, qtbot, scans, tmp_path, monkeypatch):
    win = window()
    win.load_paths(scans)
    _wait(qtbot, win)
    assert len(win.doc.scans) == 2 and win.align_panel.b_align.isEnabled()
    win.align_panel.voxel.setValue(0.3)
    win.align_panel.start_align()
    _wait(qtbot, win)
    assert win.doc.result is not None and win.doc.quality is not None
    assert win.align_panel.grade_label.text().startswith("Alignment quality")

    win.doc.save_as(str(tmp_path / "Site"))
    folder = win.doc.project.folder
    assert os.path.isfile(os.path.join(folder, "project.json"))
    win.doc.set_visible(1, False)  # autosaved shortly after
    qtbot.waitUntil(lambda: not win.doc.dirty, timeout=5000)

    other = window()
    other.open_project(folder)
    assert not other.tasks.busy  # previews came from the cache: nothing to read
    doc = other.doc
    assert [s.cloud for s in doc.scans] == [None, None]
    assert all(len(s.sample) for s in doc.scans)
    assert doc.result is not None and doc.quality.grade == win.doc.quality.grade
    assert doc.scans[1].entry.visible is False
    assert np.allclose(doc.transforms()[1], win.doc.transforms()[1])
    assert other.align_panel.voxel.value() == pytest.approx(0.3)
    assert other.view.aligned_shown()

    # Merge without loading the scans: streamed from the files.
    from pcd_merge.gui import align_panel as ap

    out = str(tmp_path / "merged.ply")
    monkeypatch.setattr(ap.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    monkeypatch.setattr(ap.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(ap.QMessageBox, "question", lambda *a, **k: ap.QMessageBox.Yes)
    other.align_panel.merge()
    _wait(qtbot, other)
    assert [s.cloud for s in doc.scans] == [None, None]
    n = sum(len(read_cloud(p)) for p in scans)
    assert len(read_cloud(out)) == n
    assert doc.project.exports[-1]["points"] == n
    # The merged cloud is listed, shown instead of the scans, and can be hidden.
    _wait(qtbot, other)
    ex = doc.project.exports[-1]
    assert ex["visible"] and len(doc.visible_exports()) == 1
    assert not any(s.entry.visible for s in doc.scans)
    item = other.project_panel.g_exports.child(0)
    assert "merged.ply" in item.text(0)
    item.setCheckState(0, Qt.Unchecked)
    assert not ex["visible"] and doc.visible_exports() == []
    item = other.project_panel.g_exports.child(0)
    item.setCheckState(0, Qt.Checked)
    assert len(doc.visible_exports()) == 1  # from memory / the cache, no re-read
    assert not other.tasks.busy

    # Align again after reopening loads the clouds first.
    other.align_panel.start_align()
    _wait(qtbot, other)
    assert all(s.cloud is not None for s in doc.scans)


def test_missing_scan_is_flagged_and_relocated(window, qtbot, scans, tmp_path, monkeypatch):
    win = window()
    win.load_paths(scans)
    _wait(qtbot, win)
    win.doc.save_as(str(tmp_path / "Site"))
    folder = win.doc.project.folder
    moved = tmp_path / "moved"
    moved.mkdir()
    for p in scans:
        os.replace(p, moved / os.path.basename(p))
    from pcd_merge.gui import main_window as mw

    warned = []
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    other = window()
    other.open_project(folder)
    assert warned and "missing" in warned[0]
    assert "file missing" in other.project_panel.g_scans.child(0).text(0)
    monkeypatch.setattr(mw.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(moved / os.path.basename(scans[0])), ""))
    other._locate(0)
    assert other.doc.project.check_scans() == []  # the neighbour was found too


def test_close_project_saves_and_empties_the_window(window, qtbot, scans, tmp_path):
    win = window()
    win.load_paths(scans)
    _wait(qtbot, win)
    win.doc.save_as(str(tmp_path / "Site"))
    folder = win.doc.project.folder
    win.doc.set_visible(0, False)
    win.close_project()
    assert win.doc.scans == [] and not win.doc.project.saved
    assert "Untitled" in win.windowTitle()
    from pcd_merge.project import Project

    assert Project.open(folder).scans[0].visible is False  # saved on close


def test_edit_mode_select_delete_undo_and_save(window, qtbot, scans, tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint

    from pcd_merge.edits import Box
    from pcd_merge.gui import align_panel as ap
    from pcd_merge.gui.view3d import HAVE_GL

    win = window()
    win.resize(1200, 800)
    win.load_paths([scans[0]])
    _wait(qtbot, win)
    doc, panel, view = win.doc, win.edit_panel, win.view
    win.modes.setCurrentIndex(1)
    assert win.right.currentWidget() is panel and view.edit_mode
    assert not panel.b_delete.isEnabled() and not win.a_undo.isEnabled()

    # select one half of the scan (as a tool would: regions in space)
    full = read_cloud(scans[0]).data
    cut = float(np.median(full["x"]))
    half = Box([-1e3, -1e3, -1e3], [cut, 1e3, 1e3])
    panel.begin_selection(add=False, kind="box")
    panel.extend_selection(half)
    shown, est = panel.selected_estimate()
    assert shown > 0 and est == pytest.approx(len(full) / 2, rel=0.1)
    assert panel.b_delete.isEnabled() and win.a_delete.isEnabled()
    panel.delete_selection()
    assert len(doc.edits.point_edits) == 1 and panel.history.count() == 1
    assert "Deleted about" in panel.history.item(0).text()
    assert not panel.regions and not panel.b_delete.isEnabled()
    proj, hidden = view.project_points(0)
    assert np.array_equal(hidden, proj[:, 0] <= cut)
    assert win.a_undo.isEnabled()
    panel.undo()
    assert not doc.edits and not view.project_points(0)[1].any() and win.a_redo.isEnabled()
    panel.redo()
    assert len(doc.edits.point_edits) == 1

    if HAVE_GL:
        # the brush picks the drawn point under the mouse
        win.show()
        qtbot.waitExposed(win)
        panel.choose_tool("brush")
        assert view.tool is not None and view.tool.name == "brush"
        gv = view.view
        mvp = view.camera_mvp()
        keep = proj[~hidden]
        target = keep[len(keep) // 2]
        c = mvp @ np.r_[target, 1.0]
        x = (c[0] / c[3] + 1) / 2 * gv.width()
        y = (1 - c[1] / c[3]) / 2 * gv.height()
        hit = view.pick(QPoint(int(round(x)), int(round(y))))
        assert hit is not None and hit[0] > cut  # deleted points are not picked
        qtbot.mouseClick(gv, Qt.LeftButton, pos=QPoint(int(round(x)), int(round(y))))
        assert panel.regions and panel.kinds == ["brush"]
        panel.clear_selection()
        panel.choose_tool(None)
        assert view.tool is None

    # save the cleaned scan: every full-resolution point of the deleted half is left out
    out = str(tmp_path / "clean.pcd")
    monkeypatch.setattr(ap.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    monkeypatch.setattr(ap.QMessageBox, "information", lambda *a, **k: None)
    assert panel.b_save.isEnabled()
    panel._save()
    _wait(qtbot, win)
    got = read_cloud(out).data
    assert len(got) == int((full["x"] > cut).sum()) and got["x"].min() > cut

    # leaving Edit drops the tool and the selection, and the deleted points stay hidden
    win.modes.setCurrentIndex(0)
    assert not view.edit_mode and view.tool is None
    assert view.project_points(0)[1].any()
    win.doc.dirty = False  # untitled project with work in it: no "save first?" on close


def test_edit_mode_split_into_parts_and_save_one(window, qtbot, scans, tmp_path, monkeypatch):
    from pcd_merge.gui import align_panel as ap
    from pcd_merge.gui.edit_tools import cut_plane

    win = window()
    win.load_paths([scans[0]])
    _wait(qtbot, win)
    doc, panel, view = win.doc, win.edit_panel, win.view
    win.modes.setCurrentIndex(1)
    # clean-up options now live in Edit
    assert win.align_panel.cleanup_box.parent() is not None
    assert panel.isAncestorOf(win.align_panel.cleanup_box)

    full = read_cloud(scans[0]).data
    panel.split_height.setValue(float(np.median(full["z"])))
    h = panel.split_height.value()  # rounded to centimetres
    panel.split_at_height()
    parts = doc.edits.parts
    assert [p.name for p in parts] == [f"above {h:.2f} m", f"below {h:.2f} m"]
    tree = win.project_panel.g_parts
    assert tree.childCount() == 2 and not tree.isDisabled()
    # hide the lower part: its points are not drawn
    tree.child(1).setCheckState(0, Qt.Unchecked)
    assert not parts[1].visible
    proj, hidden = view.project_points(0)
    assert np.array_equal(hidden, proj[:, 2] < h)
    # the split target is now a part; split the upper part again with a cut line
    assert panel.split_target.currentData() == parts[0].id
    mvp = np.diag([0.1, 0.1, -0.1, 1.0])  # looking straight down, orthographic
    plane = cut_plane(np.array([0.0, -1.0]), np.array([0.0, 1.0]), mvp)  # the line x = 0
    assert abs(abs(plane.normal[0]) - 1) < 1e-9 and abs(plane.offset) < 1e-9
    panel.split_with(plane, ("Side A", "Side B"))
    names = [p.name for p in doc.edits.parts]
    assert names[0] == f"below {h:.2f} m" and names[1].endswith("Side A")
    panel.undo()
    assert len(doc.edits.parts) == 2
    # rename in the tree
    win.project_panel.g_parts.child(0).setText(0, "Upstairs")
    assert doc.edits.parts[0].name == "Upstairs"

    # save one part on its own
    out = str(tmp_path / "upstairs.pcd")
    monkeypatch.setattr(ap.QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    monkeypatch.setattr(ap.QMessageBox, "information", lambda *a, **k: None)
    win.save_part(doc.edits.parts[0].id)
    _wait(qtbot, win)
    got = read_cloud(out).data
    assert len(got) == int((full["z"] >= h).sum())

    # colour by height draws without the alignment error option
    view.colour_mode.setCurrentIndex(2)
    view.redraw()
    win.doc.dirty = False


def test_walk_through_opens_reports_and_falls_back(window, qtbot, scans, tmp_path, monkeypatch):
    from pcd_merge import flythrough
    from pcd_merge.gui import main_window as mw

    shown = []
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    win = window()
    win.load_paths([scans[0]])
    win.walk_through()  # still loading: says so instead of doing nothing
    assert shown and "still running" in shown[0]
    _wait(qtbot, win)
    win.walk_through()
    _wait(qtbot, win)
    assert win.view.walking
    # the changing help and point-count texts never make the window wider
    walk = win.view._walk
    win.show()
    qtbot.waitExposed(win)
    width = win.width()
    walk.help.setText("W A S D: move " * 40)
    walk.info.setText("123.4 M of 567.8 M points drawn   60 fps " * 5)
    qtbot.wait(50)
    assert win.width() == width and walk.minimumSizeHint().width() < 700
    win.view.leave_walk()
    assert not win.view.walking

    # after a merge the scans are hidden and the merged cloud shown: walk through that
    merged = str(tmp_path / "merged.pcd")
    write_cloud(merged, read_cloud(scans[1]).data)
    win.doc.project.record_export(merged, 10, visible=True)
    win.doc.set_visible(0, False)
    win.walk_through()
    _wait(qtbot, win)
    assert win.view.walking  # before, this did nothing at all
    win.view.leave_walk()
    win.doc.project.exports.clear()
    win.doc.set_visible(0, True)

    # if the walk-through cannot be built inside the window, it opens in its own
    real = flythrough.FlyThroughWindow

    def embedded_fails(scene, title="", parent=None, on_exit=None):
        if on_exit is not None:
            raise RuntimeError("no second OpenGL view here")
        return real(scene, title)

    monkeypatch.setattr(flythrough, "FlyThroughWindow", embedded_fails)
    win.walk_through()
    _wait(qtbot, win)
    assert not win.view.walking and win.view._walk_window.isVisible()
    assert "window of its own" in win.tasks.status.text()
    win.view._walk_window.close()
    win.doc.dirty = False


def test_unexpected_errors_are_shown_and_logged(tmp_path, monkeypatch, qtbot):
    import sys

    from pcd_merge.gui import main_window as mw

    monkeypatch.setenv("PCD_MERGE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored afterwards
    boxes = []
    monkeypatch.setattr(mw.QMessageBox, "exec", lambda self: boxes.append(self.text()))
    mw.install_error_handler()
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert boxes and "ValueError: boom" in boxes[0]
    log = (tmp_path / "cfg" / "error.log").read_text()
    assert "ValueError: boom" in log and "Traceback" in log


def test_colour_by_intensity_and_rgb(window, qtbot, scans, tmp_path):
    from pcd_merge.pcd_io import write_pcd
    from pcd_merge.readers import pack_rgb

    # a scan with colour as well as intensity
    d = read_cloud(scans[0]).data
    rgb = np.zeros(len(d), dtype=[(n, d.dtype[n]) for n in d.dtype.names] + [("rgb", "<f4")])
    for n in d.dtype.names:
        rgb[n] = d[n]
    rgb["rgb"] = pack_rgb(np.full(len(d), 200), np.full(len(d), 10), np.full(len(d), 30))
    coloured = str(tmp_path / "coloured.pcd")
    write_pcd(coloured, rgb)

    win = window()
    win.load_paths([coloured])
    _wait(qtbot, win)
    view, doc = win.view, win.doc
    scan = doc.scans[0]
    assert set(scan.attrs) == {"intensity", "rgb"} and len(scan.attrs["rgb"]) == len(scan.sample)
    assert scan.attrs["rgb"][0].tolist() == [200, 10, 30]
    model = view.colour_mode.model()
    assert model.item(3).isEnabled() and model.item(4).isEnabled()
    for k in (3, 4):
        view.colour_mode.setCurrentIndex(k)
        view.redraw()
        colours = view._items[-1].color
        assert colours.ndim == 2 and len(colours) > 0
    assert np.allclose(colours[0, :3], [200 / 255, 10 / 255, 30 / 255], atol=1e-3)

    # the colours come back from the cache when the project is reopened
    doc.save_as(str(tmp_path / "Coloured"))
    other = window()
    other.open_project(doc.project.folder)
    assert not other.tasks.busy
    assert np.array_equal(other.doc.scans[0].attrs["intensity"], scan.attrs["intensity"])


def test_measure_in_every_mode_list_export_and_reopen(window, qtbot, scans, tmp_path):
    import csv

    from PySide6.QtCore import QPoint

    from pcd_merge.gui.view3d import HAVE_GL

    win = window()
    win.resize(1200, 800)
    win.load_paths([scans[0]])
    _wait(qtbot, win)
    doc, view, panel = win.doc, win.view, win.inspect_panel
    if not HAVE_GL:
        pytest.skip("no OpenGL")
    win.show()
    qtbot.waitExposed(win)

    # the Measure button above the view works in Align mode too
    assert view.b_measure.isEnabled()
    view.b_measure.click()
    assert view.tool is not None and view.tool.name == "measure" and view.b_measure.isChecked()
    gv = view.view
    proj, _ = view.project_points(0)
    mvp = view.camera_mvp()

    def screen(p):
        c = mvp @ np.r_[p, 1.0]
        return QPoint(int(round((c[0] / c[3] + 1) / 2 * gv.width())),
                      int(round((1 - c[1] / c[3]) / 2 * gv.height())))

    a, b = proj[10], proj[len(proj) // 2]
    qtbot.mouseClick(gv, Qt.LeftButton, pos=screen(a))
    assert view._pending is not None and not doc.measurements
    qtbot.mouseClick(gv, Qt.LeftButton, pos=screen(b))
    assert len(doc.measurements) == 1 and view._pending is None
    m = doc.measurements[0]
    assert m.kind == "distance"
    assert m.length == pytest.approx(float(np.linalg.norm(m.b - m.a)))
    assert "Measured: Distance" in win.tasks.status.text()
    assert len(view._meas_items) == 3  # line, end points, label
    assert win.project_panel.g_meas.childCount() == 1

    # Inspect opens with measuring on; a point measurement
    win.modes.setCurrentIndex(2)
    assert win.right.currentWidget() is panel and panel.measuring()
    panel.rb_point.setChecked(True)
    qtbot.mouseClick(gv, Qt.LeftButton, pos=screen(b))
    assert [x.kind for x in doc.measurements] == ["distance", "point"]
    assert panel.list.count() == 2 and panel.current_id() == doc.measurements[1].id
    assert panel.details.text().startswith("X ")

    # rename and hide from the tree, delete from the panel
    it = win.project_panel.g_meas.child(0)
    it.setText(0, "Door width")
    assert doc.measurements[0].name == "Door width"
    it = win.project_panel.g_meas.child(0)
    it.setCheckState(0, Qt.Unchecked)
    assert not doc.measurements[0].visible
    assert len(view._meas_items) == 2  # only the point: its marker and label

    out = str(tmp_path / "m.csv")
    panel.export_csv(out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["name"] for r in rows][0] == "Door width" and rows[1]["kind"] == "point"

    # a screenshot, as shown and twice as large (where OpenGL draws: not headless)
    w, h = panel.save_screenshot(str(tmp_path / "shot.png"))
    if gv.isValid():
        assert (w, h) != (0, 0) and os.path.isfile(tmp_path / "shot.png")
        w2, h2 = panel.save_screenshot(str(tmp_path / "big.png"), 2)
        assert w2 == 2 * gv.width() and h2 == 2 * gv.height()
    else:
        assert (w, h) == (0, 0) and "could not be saved" in win.tasks.status.text()

    # measurements are saved in the project
    doc.save_as(str(tmp_path / "Measured"))
    other = window()
    other.open_project(doc.project.folder)
    assert [x.name for x in other.doc.measurements] == ["Door width", ""]
    assert other.inspect_panel.list.count() == 2

    # in Edit mode Measure takes over from the selection tools, and back
    win.modes.setCurrentIndex(1)
    win.edit_panel.choose_tool("brush")
    view.b_measure.click()
    assert view.tool.name == "measure" and win.edit_panel.tool_buttons[None].isChecked()
    win.edit_panel.choose_tool("lasso")
    assert view.tool.name == "lasso" and not view.b_measure.isChecked()
    win.modes.setCurrentIndex(2)

    panel.list.setCurrentRow(1)
    panel.delete_current()
    assert len(doc.measurements) == 1
    # leaving Inspect stops measuring
    win.modes.setCurrentIndex(0)
    assert view.tool is None and not view.b_measure.isChecked()


def test_align_by_hand_from_picked_points(window, qtbot, scans):
    from PySide6.QtCore import QPoint

    from pcd_merge.gui.view3d import HAVE_GL

    win = window()
    win.resize(1200, 800)
    win.load_paths(scans)
    _wait(qtbot, win)
    doc, view, box = win.doc, win.view, win.align_panel.manual
    assert not doc.aligned and box.isEnabled()
    assert (box.moving.currentIndex(), box.fixed.currentIndex()) == (1, 0)
    assert not box.b_apply.isEnabled()
    pose0 = make_transform(0.0, 0.0, (-4, 0, 1.5))
    pose1 = make_transform(6.0, 0.02, (3, 2, 1.5))
    truth = np.linalg.inv(pose0) @ pose1  # scan 1 -> scan 0's frame

    if HAVE_GL:
        win.show()
        qtbot.waitExposed(win)
        box.b_pick.click()
        assert box.picking() and view.only == {0, 1} and box.pick_scan() == 0
        # a click snaps to the fixed scan only
        proj, _ = view.project_points(0)
        mvp = view.camera_mvp()
        c = mvp @ np.r_[proj[5], 1.0]
        pos = QPoint(int(round((c[0] / c[3] + 1) / 2 * view.view.width())),
                     int(round((1 - c[1] / c[3]) / 2 * view.view.height())))
        qtbot.mouseClick(view.view, Qt.LeftButton, pos=pos)
        assert len(box.fixed_pts) == 1 and box.pick_scan() == 1
        box.undo_point()
        assert not box.fixed_pts

    # spots on scan 0, and the same spots in scan 1's frame, clicked a few cm off
    s0 = doc.scans[0].sample
    rng = np.random.default_rng(3)
    spots = s0[np.argsort(np.linalg.norm(s0[:, :2] - [7, 1], axis=1))[::97][:4]]
    inv = np.linalg.inv(truth)
    for p in spots:
        box.add_pick(p)
        box.add_pick(p @ inv[:3, :3].T + inv[:3, 3] + rng.normal(0, 0.03, 3))
    assert box.pairs() == 4 and box.b_apply.isEnabled()
    assert "agree within" in box.status.text()
    box.apply()
    _wait(qtbot, win)
    assert doc.aligned and doc.quality is not None
    t1 = doc.transforms()[1]
    assert np.abs(t1[:3, 3] - truth[:3, 3]).max() < 0.05
    assert np.abs(t1[:3, :3] - truth[:3, :3]).max() < 0.01
    assert "placed by hand" in win.align_panel.report.toPlainText()
    assert not box.picking() and view.only is None and not box.fixed_pts
    assert win.align_panel.b_merge.isEnabled()
