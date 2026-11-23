"""Edit mode's data: regions, delete/restore order, undo, parts, grow to
object, and edits honoured by every merge and export path."""

import json

import numpy as np
import pytest

from pcd_merge.edits import (Box, Edit, EditStack, HalfSpace, ScreenShape, Sphere, Voxels,
                             grow_object, pack_keys, points_in_polygon, region_from_dict,
                             unpack_keys)
from pcd_merge.files import save_merged
from pcd_merge.pcd_io import PointCloud, read_pcd, write_pcd
from pcd_merge.project import Project
from pcd_merge.readers import open_scans


def _grid(n=11, step=1.0):
    g = np.arange(n) * step
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def _cloud(xyz, path=None):
    d = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("intensity", "<f4")])
    d["x"], d["y"], d["z"] = xyz.T
    d["intensity"] = np.arange(len(xyz))
    if path:
        write_pcd(str(path), d)
    return PointCloud(d, str(path) if path else None)


def test_regions_contain_what_they_should():
    xyz = _grid()
    s = Sphere([5, 5, 5], 1.0)
    assert s.mask(xyz).sum() == 7  # centre and its six neighbours
    b = Box([2, 2, 2], [0, 0, 0])  # corners in any order
    assert b.mask(xyz).sum() == 27
    h = HalfSpace([0, 0, 2], 5.0)  # z >= 5 (normal is made unit length)
    assert h.mask(xyz).sum() == 11 * 11 * 6
    assert (h.mask(xyz) ^ h.flipped().mask(xyz)).sum() >= 11 * 11 * 10
    for r in (s, b, h):
        again = region_from_dict(json.loads(json.dumps(r.to_dict())))
        assert np.array_equal(again.mask(xyz), r.mask(xyz))


def test_polygon_and_screen_shape():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.2], [0.99, 0.01]])
    assert points_in_polygon(pts, sq).tolist() == [True, False, False, True]

    # An orthographic camera looking down -z: NDC x, y are world x, y / 10.
    mvp = np.diag([0.1, 0.1, -0.1, 1.0])
    lasso = ScreenShape([[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]], mvp)
    xyz = _grid() - 5  # -5..5 on each axis
    m = lasso.mask(xyz)
    # every depth is selected, only |x|, |y| <= 2
    assert m.sum() == 5 * 5 * 11
    assert np.abs(xyz[m][:, :2]).max() <= 2
    again = region_from_dict(json.loads(json.dumps(lasso.to_dict())))
    assert np.array_equal(again.mask(xyz), m)


def test_perspective_screen_shape_ignores_points_behind_the_camera():
    # camera at the origin looking along -z (w = -z)
    mvp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -0.2], [0, 0, -1, 0]], float)
    lasso = ScreenShape([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]], mvp)
    xyz = np.array([[0, 0, -5.0], [0, 0, 5.0], [3, 0, -5.0]])
    assert lasso.mask(xyz).tolist() == [True, False, False]


def test_voxels_pack_and_save_compactly():
    ijk = np.array([[0, 0, 0], [-3, 7, 1_000_000], [-1_000_000, -1, 2]])
    assert np.array_equal(unpack_keys(pack_keys(ijk)), ijk)
    v = Voxels(0.5, [0, 0, 0], pack_keys(np.array([[0, 0, 0], [1, 0, 0]])))
    assert v.mask(np.array([[0.1, 0.1, 0.1], [0.9, 0.2, 0.3], [1.1, 0, 0]])).tolist() == \
        [True, True, False]
    d = v.to_dict()
    assert isinstance(d["keys"], str)
    again = region_from_dict(json.loads(json.dumps(d)))
    assert np.array_equal(again.keys, v.keys)
    with pytest.raises(ValueError):
        region_from_dict({"kind": "teapot"})


def test_last_edit_wins_and_limits_to_scans():
    xyz = _grid()
    st = EditStack()
    st.add(Edit("delete", Box([0, 0, 0], [10, 10, 4.5])))  # bottom five layers
    st.add(Edit("restore", Sphere([5, 5, 0], 1.0)))
    st.add(Edit("delete", Sphere([5, 5, 0], 0.1), scans=["b"]))
    d = st.deleted(xyz, "a")
    assert d.sum() == 11 * 11 * 5 - 6  # sphere at the floor's edge holds 6 points
    assert st.deleted(xyz, "b").sum() == d.sum() + 1
    assert np.array_equal(st.keep(xyz, "a"), ~d)


def test_undo_redo_and_splits():
    xyz = _grid()
    st = EditStack()
    assert not st and not st.can_undo()
    st.add(Edit("delete", Sphere([0, 0, 0], 0.5)))
    a, b = st.split(HalfSpace([1, 0, 0], 4.5), ("West", "East"))
    assert [p.name for p in st.parts] == ["West", "East"]
    assert a.contains(xyz).sum() + b.contains(xyz).sum() == len(xyz)
    # split one part again
    c, d = st.split(HalfSpace([0, 0, 1], 5), ("Upper", "Lower"), within=b)
    assert [p.name for p in st.parts] == ["West", "Upper", "Lower"]
    assert (c.contains(xyz) & a.contains(xyz)).sum() == 0
    assert st.undo() == "split" and [p.name for p in st.parts] == ["West", "East"]
    assert st.redo() == "split" and len(st.parts) == 3
    assert st.undo() == "split" and st.undo() == "split" and st.parts == []
    assert isinstance(st.undo(), Edit) and not st
    assert st.undo() is None
    assert isinstance(st.redo(), Edit) and st
    st.add(Edit("delete", Sphere([9, 9, 9], 0.5)))
    assert not st.can_redo()  # a new edit drops what was undone
    # saving keeps the edits and parts (not the undo markers)
    st.split(HalfSpace([1, 0, 0], 4.5))
    again = EditStack.from_dict(json.loads(json.dumps(st.to_dict())))
    assert len(again.point_edits) == 2 and len(again.parts) == 2
    assert np.array_equal(again.keep(xyz, part=again.parts[0]), st.keep(xyz, part=st.parts[0]))


def test_grow_object_takes_the_chair_not_the_floor():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-5, 5, (8000, 2)), rng.normal(0, 0.003, 8000)]
    # a "chair": a 0.5 m cube of points standing on the floor at (1, 1)
    chair = rng.uniform(0, 0.5, (3000, 3)) + [1, 1, 0.02]
    # a second object 2 m away that must not be taken
    other = rng.uniform(0, 0.5, (3000, 3)) + [-2, 1, 0.02]
    xyz = np.vstack([floor, chair, other])
    v = grow_object(xyz, seed=[1.25, 1.25, 0.3], cell=0.1)
    took = v.mask(xyz)
    assert took[len(floor):len(floor) + len(chair)].mean() > 0.9  # the bottom few cm go with the floor
    assert took[len(floor) + len(chair):].sum() == 0
    assert took[:len(floor)].mean() < 0.02
    # without a cell size one is picked from the spacing
    assert grow_object(xyz, seed=[1.25, 1.25, 0.3]).mask(chair).mean() > 0.8
    # nowhere near anything: nothing
    assert not grow_object(xyz, seed=[50, 50, 50]).mask(xyz).any()


@pytest.fixture
def two_scans(tmp_path):
    a = _cloud(_grid(), tmp_path / "a.pcd")
    b = _cloud(_grid() + [20, 0, 0], tmp_path / "b.pcd")
    t = [np.eye(4), np.eye(4)]
    t[1][:3, 3] = [-20, 0, 0]  # b sits on top of a once aligned
    st = EditStack()
    st.add(Edit("delete", Box([-0.5, -0.5, -0.5], [10.5, 10.5, 2.5])))  # z <= 2
    st.add(Edit("delete", Sphere([5, 5, 8], 0.1), scans=["kb"]))
    return tmp_path, [a, b], t, st


@pytest.mark.parametrize("ext", [".pcd", ".ply", ".las"])
def test_streamed_merge_leaves_out_deleted_points(two_scans, ext):
    tmp, clouds, t, st = two_scans
    sources = [open_scans(c.path)[0] for c in clouds]
    out = str(tmp / f"m{ext}")
    n, *_ = save_merged(sources, t, out, edits=st, keys=["ka", "kb"])
    per = 11 * 11 * 8
    assert n == 2 * per - 1
    got = open_scans(out)[0].read().data
    assert len(got) == n and got["z"].min() > 2.5


def test_in_memory_merge_and_parts_leave_out_points_too(two_scans):
    tmp, clouds, t, st = two_scans
    from pcd_merge.cleanup import CleanupOptions

    a, _ = st.split(HalfSpace([1, 0, 0], 4.5), ("East", "West"))
    out = str(tmp / "east.pcd")
    n, *_ = save_merged(clouds, t, out, edits=st, keys=["ka", "kb"], part=a,
                        cleanup=CleanupOptions(dedup=True, dedup_cell=0.01))
    got = read_pcd(out).data
    assert len(got) == n and got["x"].min() >= 5 and got["z"].min() > 2.5
    assert n == 6 * 11 * 8  # the two scans are on top of each other: dedup halves them


def test_edits_are_saved_in_the_project(tmp_path):
    proj = Project.create(str(tmp_path / "Site"))
    proj.edits.add(Edit("delete", Sphere([1, 2, 3], 0.5), label="Brush"))
    proj.edits.split(HalfSpace([0, 0, 1], 3.0), ("Ground floor", "First floor"))
    proj.save()
    again = Project.open(proj.folder)
    assert again.edits.point_edits[0].label == "Brush"
    assert [p.name for p in again.edits.parts] == ["Ground floor", "First floor"]


def test_union_of_regions():
    from pcd_merge.edits import Union, union

    xyz = _grid()
    u = Union([Sphere([0, 0, 0], 0.5), Sphere([10, 10, 10], 0.5), Box([5, 5, 5], [5, 5, 6])])
    assert u.mask(xyz).sum() == 4
    again = region_from_dict(json.loads(json.dumps(u.to_dict())))
    assert np.array_equal(again.mask(xyz), u.mask(xyz))
    s = Sphere([1, 1, 1], 1)
    assert union([s]) is s
    assert not Union([]).mask(xyz).any()
