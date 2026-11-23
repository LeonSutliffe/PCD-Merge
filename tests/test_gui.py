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
