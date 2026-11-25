"""Model mode (pcd_merge/cad): levels, plans and sections, fitted solids,
DXF / IFC / STEP files, snapping, and the panel in the window.

Delete this file together with pcd_merge/cad to remove Model mode."""

import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pcd_merge.cad import synthetic as syn  # noqa: E402
from pcd_merge.cad.dxf import write_drawings, write_model  # noqa: E402
from pcd_merge.cad.fit import FitSettings, fit_model  # noqa: E402
from pcd_merge.cad.ifc import ifc_guid, step_real, step_str, write_ifc  # noqa: E402
from pcd_merge.cad.levels import find_levels  # noqa: E402
from pcd_merge.cad.model import Model, box, cylinder  # noqa: E402
from pcd_merge.cad.plan import (TraceSettings, plan_from_points,  # noqa: E402
                                section_from_points, trace)
from pcd_merge.cad.points import ArraySource, Source, from_arrays, gather  # noqa: E402
from pcd_merge.cad.snap import Snapper  # noqa: E402
from pcd_merge.cad.step import write_step  # noqa: E402


@pytest.fixture(scope="module")
def building():
    return syn.make_building(density=1200, seed=1)


@pytest.fixture(scope="module")
def all_points(building):
    return np.concatenate([p for p, _ in building])


@pytest.fixture(scope="module")
def fitted(building):
    srcs = [Source(ArraySource(p), np.eye(4), None, s, f"s{i}") for i, (p, s) in
            enumerate(building)]
    site = gather(srcs, cell=0.05, per_source=True)
    elements, faces = fit_model(site, FitSettings())
    return site, elements, faces


def _length(seg):
    return float(np.linalg.norm(seg[1] - seg[0]))


def _find(segs, a, b, tol=0.03):
    """The segment from a to b (either way round), or None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    for s in segs:
        if (np.abs(s[0] - a).max() < tol and np.abs(s[1] - b).max() < tol) or \
                (np.abs(s[0] - b).max() < tol and np.abs(s[1] - a).max() < tol):
            return s
    return None


# ---------------------------------------------------------------- plans and sections


def test_plan_traces_walls_doors_and_corners(all_points):
    d = plan_from_points(all_points, 1.2, 0.10, name="Ground")
    segs = d.segments
    lo = syn.WALL
    hi_x, hi_y = syn.OUTER[1] - syn.WALL, syn.OUTER[3] - syn.WALL
    a, b = syn.INNER
    dy0, dy1, _h = syn.DOOR
    # the room's outline: exact, square corners
    for p, q in [((lo, lo), (hi_x, lo)), ((lo, hi_y), (hi_x, hi_y)),
                 ((lo, lo), (lo, hi_y)), ((hi_x, lo), (hi_x, hi_y))]:
        assert _find(segs, p, q) is not None, (p, q)
    # the inner wall's two faces, with the door left open, and the door's reveals closed
    for x in (a, b):
        assert _find(segs, (x, lo), (x, dy0)) is not None
        assert _find(segs, (x, dy1), (x, hi_y)) is not None
    assert _find(segs, (a, dy0), (b, dy0)) is not None
    assert _find(segs, (a, dy1), (b, dy1)) is not None
    assert len(segs) == 10
    # the column is a circle
    (cx, cy), r = syn.COLUMN
    assert len(d.circles) == 1
    assert np.allclose(d.circles[0], [cx, cy, r], atol=0.01)
    assert d.to_3d([[1.0, 2.0]]).tolist() == [[1.0, 2.0, 1.2]]


def test_section_traces_floors_ceilings_and_walls(all_points):
    d = section_from_points(all_points, (0.0, 6.5), (12.0, 6.5), 0.10)
    segs = d.segments
    # x along the line (from x = 0), y = height
    assert _find(segs, (syn.WALL, 0.0), (syn.WALL, 2.7)) is not None
    assert _find(segs, (syn.WALL, 3.0), (syn.WALL, 5.7)) is not None
    floors = [s for s in segs if abs(s[0, 1] - s[1, 1]) < 0.01]
    heights = sorted({round(float(s[0, 1]), 1) for s in floors})
    assert heights == [0.0, 2.7, 3.0, 5.7]
    p3 = d.to_3d([[2.0, 1.5]])[0]
    assert np.allclose(p3, [2.0, 6.5, 1.5])


def test_trace_on_a_rotated_room():
    """Square-up works on the building's own axes, not only on X and Y."""
    rng = np.random.default_rng(3)
    ang = np.radians(23.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = []
    for (x0, y0), (x1, y1) in [((0, 0), (6, 0)), ((6, 0), (6, 4)), ((6, 4), (0, 4)),
                               ((0, 4), (0, 0))]:
        t = rng.random(3000)[:, None]
        pts.append(np.array([x0, y0]) + t * (np.array([x1, y1]) - np.array([x0, y0])))
    pts = np.concatenate(pts) + rng.normal(0, 0.003, (12000, 2))
    segs = trace(pts @ rot.T, TraceSettings())
    assert len(segs) == 4
    ang_out = np.degrees(np.arctan2(*(segs[:, 1] - segs[:, 0]).T[::-1])) % 90
    assert np.allclose(ang_out, 23.0, atol=0.05)
    corners = (np.array([[0, 0], [6, 0], [6, 4], [0, 4]]) @ rot.T)
    ends = segs.reshape(-1, 2)
    for c in corners:
        assert np.linalg.norm(ends - c, axis=1).min() < 0.02


def test_empty_slice_gives_an_empty_drawing():
    d = plan_from_points(np.zeros((0, 3)), 1.0)
    assert len(d.segments) == 0 and len(d.points) == 0


# ---------------------------------------------------------------- levels


def test_levels_from_scanner_positions(building):
    sp = from_arrays([p for p, _ in building], [s for _, s in building])
    levels = find_levels(sp.xyz, sp.sensor_z())
    assert [lv.name for lv in levels] == ["Level 0", "Level 1"]
    for lv, (f, c) in zip(levels, syn.LEVELS):
        assert lv.floor == pytest.approx(f, abs=0.02)
        assert lv.ceiling == pytest.approx(c, abs=0.02)
    assert levels[0].cut_height() == pytest.approx(1.2, abs=0.02)


def test_levels_without_scanner_positions_from_a_sparse_sample(all_points):
    idx = np.random.default_rng(0).choice(len(all_points), 40_000, replace=False)
    levels = find_levels(all_points[idx])
    assert [(round(lv.floor, 1), round(lv.ceiling, 1)) for lv in levels] == [(0.0, 2.7),
                                                                            (3.0, 5.7)]


# ---------------------------------------------------------------- fitting


def test_fit_walls_slabs_column_and_pipe(fitted):
    _site, els, _faces = fitted
    kinds = {}
    for e in els:
        kinds.setdefault(e.kind, []).append(e)
    # inner wall seen from both rooms: its real thickness, on both storeys
    both = [e for e in kinds["wall"] if not e.name]
    assert len(both) == 2
    for e in both:
        assert e.size[1] == pytest.approx(syn.INNER[1] - syn.INNER[0], abs=0.01)
        assert e.size[0] == pytest.approx(syn.OUTER[3] - 2 * syn.WALL, abs=0.03)
    # outer walls (seen from inside only): 4 faces per room side x 2 storeys
    one = [e for e in kinds["wall"] if e.name]
    assert len(one) == 12
    for e in one:
        assert e.size[2] == pytest.approx(2.7, abs=0.03)
    # the slab between the storeys: real thickness 0.3 m
    mid = [e for e in kinds["slab"] if not e.name]
    assert len(mid) == 2
    for e in mid:
        assert e.size[2] == pytest.approx(0.3, abs=0.01)
        assert e.bottom() == pytest.approx(2.7, abs=0.01)
    (col,) = kinds["column"]
    (cx, cy), r = syn.COLUMN
    assert col.radius == pytest.approx(r, abs=0.01)
    assert np.allclose(col.origin[:2], [cx, cy], atol=0.01)
    assert col.w == [0.0, 0.0, 1.0]
    (pipe,) = kinds["pipe"]
    py, pz, pr = syn.PIPE
    assert pipe.radius == pytest.approx(pr, abs=0.01)
    assert abs(pipe.w[0]) == pytest.approx(1.0)
    assert pipe.origin[1] == pytest.approx(py, abs=0.01)
    assert pipe.origin[2] == pytest.approx(pz, abs=0.01)


def test_frames_are_right_handed(fitted):
    for e in fitted[1]:
        u, v, w = e.axes()
        assert np.cross(u, v) @ w == pytest.approx(1.0, abs=1e-6), e.kind


def test_gather_applies_transform_slice_cell_and_edits():
    from pcd_merge.edits import Box, Edit, EditStack

    pts = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [2, 0, 5.0], [3, 0, 0]], float)
    t = np.eye(4)
    t[:3, 3] = (10, 0, 0)
    st = EditStack()
    st.add(Edit("delete", Box([12.5, -1, -1], [13.5, 1, 1])))
    src = Source(ArraySource(pts, chunk=2), t, "k")
    got = gather([src], st, zmax=1.0, cell=0.5)
    assert sorted(got.xyz[:, 0].tolist()) == [10.0, 11.0]


def _thin_reference(pts, cell, ref):
    """One point per cube, the first one read: the slow obvious way."""
    seen, out = set(), []
    for q in pts:
        k = tuple(np.floor((q - ref) / cell).astype(int))
        if k not in seen:
            seen.add(k)
            out.append(q)
    return np.array(out)


def test_gather_thins_across_chunks_like_one_pass():

    rng = np.random.default_rng(3)
    a = rng.random((30_000, 3)) * 2
    b = rng.random((20_000, 3)) * 2 + 0.5
    srcs = [Source(ArraySource(a, chunk=997), np.eye(4), None, None, "a"),
            Source(ArraySource(b, chunk=1500), np.eye(4), None, None, "b")]
    ref = np.floor(a[0])
    # one cube count across both scans
    got = gather(srcs, cell=0.1)
    want = _thin_reference(np.concatenate([a, b]), 0.1, ref)
    assert len(got) == len(want)
    assert np.allclose(np.sort(got.xyz, axis=0), np.sort(want, axis=0))
    # each scan on its own
    got = gather(srcs, cell=0.1, per_source=True)
    wa, wb = _thin_reference(a, 0.1, ref), _thin_reference(b, 0.1, ref)
    assert (got.scan == 0).sum() == len(wa) and (got.scan == 1).sum() == len(wb)
    assert np.allclose(np.sort(got.xyz[got.scan == 0], axis=0), np.sort(wa, axis=0))
    assert np.allclose(np.sort(got.xyz[got.scan == 1], axis=0), np.sort(wb, axis=0))


def test_gather_reports_progress_within_a_scan_and_can_stop():
    pts = np.random.default_rng(0).random((10_000, 3))
    src = Source(ArraySource(pts, chunk=1000), np.eye(4), None, None, "one big scan")
    seen = []
    gather([src], cell=0.05, progress=lambda m, f: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert len({round(f, 2) for f in seen}) >= 10  # moves with every chunk, not 0 then 100 %

    class Stop(Exception):
        pass

    calls = []

    def stop(m, f):
        calls.append(f)
        if f >= 0.3:
            raise Stop()

    with pytest.raises(Stop):
        gather([src], cell=0.05, progress=stop)
    assert max(calls) < 0.5  # stopped part of the way through


def test_gather_stops_early_when_over_the_limit():
    from pcd_merge.cad.points import TooManyPoints

    pts = np.random.default_rng(0).random((50_000, 3)) * 10
    src = Source(ArraySource(pts, chunk=1000), np.eye(4), None, None, "a")
    seen = []
    with pytest.raises(TooManyPoints):
        gather([src], cell=0.01, limit=5_000, progress=lambda m, f: seen.append(f))
    assert max(seen) < 0.5  # without reading the whole scan


def test_covariances_in_slabs_match_one_pass(monkeypatch):
    import pcd_merge.cad.geom as G

    pts = np.random.default_rng(2).random((20_000, 3)) * [10, 2, 2]
    whole = G.local_covariances(pts, 0.2, 30)
    monkeypatch.setattr(G, "TILE_POINTS", 3_000)
    seen = []
    tiled = G.local_covariances(pts, 0.2, 30, progress=seen.append)
    assert len(seen) == 7 and seen[-1] == 1.0
    assert np.allclose(tiled, whole, atol=1e-9)


@pytest.fixture(scope="module")
def shelving():
    b = syn.make_building(density=1200, seed=1, shelving=True)
    srcs = [Source(ArraySource(p), np.eye(4), None, s, f"s{i}") for i, (p, s) in enumerate(b)]
    return gather(srcs, cell=0.05, per_source=True)


def test_fit_leaves_out_shelving(shelving):
    els, _faces = fit_model(shelving, FitSettings())
    walls = [e for e in els if e.kind == "wall"]
    slabs = [e for e in els if e.kind == "slab"]
    # the same building as without shelving: 7 walls a storey, 4 slabs/floors/ceilings
    assert len(walls) == 14
    for w in walls:
        assert w.size[2] == pytest.approx(2.7, abs=0.03)  # floor to ceiling, even behind shelves
        assert w.bottom() == pytest.approx(0.0, abs=0.03) or \
            w.bottom() == pytest.approx(3.0, abs=0.03)
    assert len(slabs) == 6
    for e in slabs:  # no shelf boards among the floors
        assert min(abs(e.bottom() - z) for z in (-0.2, 2.7, 5.7)) < 0.03
    # the floor hidden under the shelves reaches the walls of the room
    left = [e for e in slabs if e.name == "Floor (top seen)" and e.origin[0] < 1]
    assert len(left) == 1 and left[0].size[1] == pytest.approx(7.4, abs=0.15)


def test_fit_keeps_shelving_when_asked(shelving):
    els, _faces = fit_model(shelving, FitSettings(skip_clutter=False))
    walls = [e for e in els if e.kind == "wall"]
    assert len(walls) > 14  # the shelving's backs and ends


# ---------------------------------------------------------------- files


def test_dxf_plan(tmp_path, all_points):
    d = plan_from_points(all_points, 1.2, name="Ground floor")
    path = tmp_path / "plan.dxf"
    n = write_drawings(str(path), [d])
    text = path.read_text()
    assert n["lines"] == len(d.segments) and n["circles"] == 1
    lines = text.splitlines()
    assert lines[:4] == ["0", "SECTION", "2", "HEADER"] and lines[-2:] == ["0", "EOF"]
    assert text.count("\nLINE\n") == len(d.segments)
    assert text.count("\nCIRCLE\n") == 1
    assert "\nWALLS\n" in text and "\nSCAN\n" in text
    assert "AC1009" in text
    # group codes and values alternate: every code is an integer
    assert all(re.fullmatch(r"-?\d+", lines[k].strip()) for k in range(0, len(lines), 2))
    # without points
    write_drawings(str(path), [d], points=False)
    assert "\nPOINT\n" not in path.read_text()


def test_dxf_model(tmp_path, fitted):
    path = tmp_path / "m.dxf"
    assert write_model(str(path), fitted[1]) == len(fitted[1])
    text = path.read_text()
    boxes = sum(e.shape == "box" for e in fitted[1])
    cyl = sum(e.shape == "cylinder" for e in fitted[1])
    assert text.count("\n3DFACE\n") == 6 * boxes + 3 * 24 * cyl


def test_ifc_strings_and_guids():
    g = ifc_guid()
    assert len(g) == 22 and g[0] in "0123"
    assert step_str("Bob's café") == "'Bob''s caf\\X2\\00E9\\X0\\'"
    assert step_real(0) == "0." and step_real(2) == "2." and step_real(1e-7) == "1.E-7"


def test_ifc_file(tmp_path, fitted, building):
    sp = from_arrays([p for p, _ in building], [s for _, s in building])
    levels = find_levels(sp.xyz, sp.sensor_z())
    path = tmp_path / "m.ifc"
    counts = write_ifc(str(path), fitted[1], levels, project="Test", app_version="1.0")
    assert counts == {"IFCWALL": 14, "IFCSLAB": 6, "IFCCOLUMN": 1, "IFCPIPESEGMENT": 1}
    text = path.read_text()
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;")
    assert "FILE_SCHEMA(('IFC4'));" in text
    assert text.count("IFCBUILDINGSTOREY(") == 2
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.geom
    import ifcopenshell.validate

    f = ifcopenshell.open(str(path))
    log = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, log)
    assert log.statements == []
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    col = f.by_type("IfcColumn")[0]
    assert col.ContainedInStructure[0].RelatingStructure.Name == "Level 0"
    v = np.array(ifcopenshell.geom.create_shape(st, col).geometry.verts).reshape(-1, 3)
    (cx, cy), r = syn.COLUMN
    assert np.allclose(v.min(axis=0)[:2], [cx - r, cy - r], atol=0.02)
    slab = [s for s in f.by_type("IfcSlab") if s.Name.startswith("Floor slab")][0]
    assert slab.ContainedInStructure[0].RelatingStructure.Name == "Level 1"


def test_step_file(tmp_path, fitted):
    path = tmp_path / "m.step"
    n = write_step(str(path), fitted[1])
    text = path.read_text()
    assert n == len(fitted[1]) == text.count("MANIFOLD_SOLID_BREP(")
    assert "AUTOMOTIVE_DESIGN" in text and text.count("CYLINDRICAL_SURFACE(") == 2
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        assert len(vols) == n
        total = sum(gmsh.model.occ.getMass(d, t) for d, t in vols) / 1e9  # mm³ -> m³
        want = sum(np.prod(e.size) if e.shape == "box" else np.pi * e.radius ** 2 * e.size[2]
                   for e in fitted[1])
        assert total == pytest.approx(want, rel=1e-6)
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------- model, snapping


def test_model_saved_and_loaded(tmp_path, fitted):
    from pcd_merge.cad.levels import Level

    m = Model([Level("Ground", 0.0, 2.7, 80.0)], list(fitted[1]), {"cut_above": 1.1})
    m.assign_levels()
    m.save(str(tmp_path))
    again = Model.load(str(tmp_path))
    assert again.to_dict() == m.to_dict()
    assert Model.load(str(tmp_path / "nothing")).elements == []


def test_snap_to_corner_edge_and_face():
    wall = box("wall", (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (4, 0.2, 3))
    col = cylinder("column", (10, 0, 0), (0, 0, 1), 0.2, 3)
    s = Snapper([wall, col])
    p, what = s.snap([0.03, -0.02, 2.97], radius=0.1)
    assert what == "corner" and np.allclose(p, [0, 0, 3])
    p, what = s.snap([2.0, 0.03, 3.02], radius=0.1)
    assert what == "edge" and np.allclose(p, [2.0, 0.0, 3.0])
    p, what = s.snap([2.0, 0.012, 1.5], radius=0.1)
    assert what == "face" and np.allclose(p, [2.0, 0.0, 1.5])
    p, what = s.snap([2.0, -1.0, 1.5], radius=0.1)
    assert what is None
    p, what = s.snap([10.02, 0.0, 3.03], radius=0.1)
    assert what == "corner" and np.allclose(p, [10, 0, 3])


# ---------------------------------------------------------------- the panel


def test_model_mode_in_the_window(qtbot, tmp_path, monkeypatch, building):
    from pcd_merge.gui import MainWindow
    from pcd_merge.gui import main_window as mw
    from pcd_merge.pcd_io import write_pcd

    monkeypatch.setenv("PCD_MERGE_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(mw.QMessageBox, "question", lambda *a, **k: mw.QMessageBox.Discard)
    from pcd_merge.gui import workers

    def failed(_parent, _title, text):  # a failed job must not stop the test with a dialog
        raise AssertionError(text)

    monkeypatch.setattr(workers.QMessageBox, "critical", failed)
    # one scan: the ground floor's second room, with the column
    pts, sensor = building[1]
    data = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    data["x"], data["y"], data["z"] = pts.T
    path = str(tmp_path / "room.pcd")
    write_pcd(path, data)
    raw = open(path, "rb").read().replace(
        b"VIEWPOINT 0 0 0 1 0 0 0", ("VIEWPOINT %g %g %g 1 0 0 0" % tuple(sensor)).encode())
    open(path, "wb").write(raw)  # where the scanner stood

    win = MainWindow()
    qtbot.addWidget(win)
    panel = win.model_panel
    assert win.modes.tabText(3) == "Model"
    win.modes.setCurrentIndex(3)
    assert win.right.currentWidget() is panel

    def wait():
        qtbot.waitUntil(lambda: not win.tasks.busy, timeout=120_000)

    win.load_paths([path])
    wait()
    panel.find_levels()
    wait()
    assert panel.model.levels, win.tasks.status.text()
    assert [round(lv.floor, 1) for lv in panel.model.levels] == [0.0]
    assert panel.model.levels[0].ceiling == pytest.approx(2.7, abs=0.03)
    panel.make_plan()
    wait()
    d = panel.drawing
    assert d is not None and d.z == pytest.approx(1.2, abs=0.03)
    assert len(d.segments) >= 4 and len(d.circles) == 1
    out = tmp_path / "plan.dxf"
    panel.save_dxf(str(out))
    assert out.read_text().count("\nLINE\n") == len(d.segments)
    panel.fit()
    wait()
    kinds = panel.model.counts()
    assert kinds.get("wall", 0) >= 4 and kinds.get("column") == 1
    assert panel.element_list.count() == len(panel.model.elements)
    for fmt in ("ifc", "step", "dxf"):
        panel.export(str(tmp_path / f"m.{fmt}"), fmt)
        assert (tmp_path / f"m.{fmt}").stat().st_size > 1000
    # snapping: a click next to the column's top centre lands on it
    col = next(e for e in panel.model.elements if e.kind == "column")
    top = np.asarray(col.origin) + col.size[2] * np.asarray(col.w)
    assert panel.snapper
    monkeypatch.setattr(win.view, "pixel_size", lambda: 0.01)
    assert np.allclose(panel.snap(top + [0.03, 0.0, 0.0]), top)
    # the model is kept in a saved project, and comes back when it is opened
    win.save_project_as = None  # (not used: save through the document)
    win.doc.save_as(str(tmp_path / "Site"))
    panel.changed()
    folder = win.doc.project.folder
    assert os.path.isfile(Model.path(folder))
    n = len(panel.model.elements)
    win.doc.dirty = False
    win.open_project(folder)
    wait()
    assert len(win.model_panel.model.elements) == n
    win.doc.dirty = False
