"""Project folders: saving, reopening, moved and changed scans, cache, recent list."""

import json
import os
import shutil

import numpy as np
import pytest

from pcd_merge.files import write_manifest
from pcd_merge.pcd_io import write_pcd
from pcd_merge.project import (FORMAT_VERSION, Project, ProjectError, add_recent,
                               alignment_from_dict, from_manifest, is_project, project_folder,
                               quick_hash, recent_projects)
from pcd_merge.quality import PairQuality, QualityReport
from pcd_merge.readers import open_scans
from pcd_merge.registration import AlignResult, PairResult


def _scan(path, n=500, seed=0):
    rng = np.random.default_rng(seed)
    d = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    for k in "xyz":
        d[k] = rng.uniform(-5, 5, n)
    write_pcd(str(path), d)
    return str(path)


@pytest.fixture
def site(tmp_path):
    scans = tmp_path / "scans"
    scans.mkdir()
    paths = [_scan(scans / f"s{k}.pcd", seed=k) for k in range(3)]
    proj = Project.create(str(tmp_path / "Office"))
    for p in paths:
        proj.add(open_scans(p)[0])
    return tmp_path, proj, paths


def _result(n):
    tfs = [np.eye(4)] + [np.diag([1, 1, 1, 1.0]) for _ in range(n - 1)]
    tfs[1][:3, 3] = [1, 2, 3]
    return AlignResult(transforms=tfs, voxel_size=0.1, connected=[True] * n,
                       cloud_fitness=[0.9] * n,
                       pairs=[PairResult(1, 0, tfs[1], 0.8, 0.01, structure=0.7)])


def test_create_makes_the_folder_layout(site):
    tmp, proj, _ = site
    assert proj.folder.endswith("Office.pcdproj") and proj.name == "Office"
    for sub in ("edits", "cache", "view", "photos", "exports"):
        assert os.path.isdir(os.path.join(proj.folder, sub))
    assert is_project(str(tmp / "Office")) and is_project(proj.file)
    with pytest.raises(ProjectError):
        Project.create(proj.folder)


def test_save_and_reopen_keeps_everything(site):
    tmp, proj, paths = site
    proj.scans[2].visible = False
    proj.scans[1].locked = True
    proj.settings = {"voxel": 0.2, "cleanup": {"dedup": True}}
    q = QualityReport(voxel_size=0.1, threshold=0.3,
                      pairs=[PairQuality(0, 1, 0.6, 0.01, 0.005, 0.02, "Good", 0.004, 0.97)],
                      cloud_overlap=[0.6, 0.6, 0.0], cloud_rmse=[0.01, 0.01, np.nan],
                      change_threshold=0.2, cloud_changed=[0.03, 0.03, 0.0])
    proj.set_alignment(_result(3), q, ["Alignment quality: Good"])
    proj.record_export(str(tmp / "m.laz"), 1500)
    proj.save()

    again = Project.open(proj.folder)
    assert [s.path for s in again.scans] == [os.path.abspath(p) for p in paths]
    assert [s.visible for s in again.scans] == [True, True, False]
    assert again.scans[1].locked and again.settings["voxel"] == 0.2
    assert np.allclose(again.transforms()[1][:3, 3], [1, 2, 3])
    assert again.alignment_matches_scans()
    r = alignment_from_dict(again.alignment, again.transforms())
    assert r.voxel_size == 0.1 and r.pairs[0].structure == 0.7 and r.ok
    q2 = QualityReport.from_dict(again.quality)
    assert q2.grade == "Good" and q2.to_dict() == q.to_dict()
    assert again.quality["report"] == ["Alignment quality: Good"]
    assert again.exports[0]["points"] == 1500
    data = json.load(open(proj.file))
    assert data["format_version"] == FORMAT_VERSION
    assert data["scans"][0]["relative_path"] == "../scans/s0.pcd"


def test_project_moved_with_its_scans_finds_them(site):
    tmp, proj, _ = site
    proj.save()
    moved = tmp / "elsewhere"
    moved.mkdir()
    shutil.move(proj.folder, moved / "Office.pcdproj")
    shutil.move(str(tmp / "scans"), moved / "scans")
    again = Project.open(str(moved / "Office.pcdproj"))
    assert again.check_scans() == []
    assert again.scans[0].path == str(moved / "scans" / "s0.pcd")


def test_missing_and_replaced_scans_are_noticed_and_relocated(site):
    tmp, proj, paths = site
    os.remove(paths[0])
    _scan(paths[1], n=600, seed=9)  # replaced by a different scan
    problems = {os.path.basename(s.path): st for s, st in proj.check_scans()}
    assert problems == {"s0.pcd": "missing", "s1.pcd": "changed"}

    new_dir = tmp / "new"
    new_dir.mkdir()
    shutil.move(paths[2], new_dir / "s2.pcd")
    _scan(new_dir / "s0.pcd", seed=0)
    proj.relocate(proj.scans[2], str(new_dir / "s2.pcd"))
    proj.relocate(proj.scans[0], str(new_dir / "s0.pcd"))
    assert proj.scans[0].status() == "ok" and proj.scans[2].status() == "ok"


def test_relocating_one_scan_finds_its_neighbours(site):
    tmp, proj, paths = site
    new_dir = tmp / "moved"
    shutil.move(str(tmp / "scans"), new_dir)
    assert len(proj.check_scans()) == 3
    proj.relocate(proj.scans[0], str(new_dir / "s0.pcd"))
    assert proj.check_scans() == []


def test_newer_or_damaged_project_files_are_refused(tmp_path):
    proj = Project.create(str(tmp_path / "a"))
    data = json.load(open(proj.file))
    data["format_version"] = FORMAT_VERSION + 1
    json.dump(data, open(proj.file, "w"))
    with pytest.raises(ProjectError, match="newer"):
        Project.open(proj.folder)
    open(proj.file, "w").write("{broken")
    with pytest.raises(ProjectError):
        Project.open(proj.folder)
    with pytest.raises(ProjectError):
        Project.open(str(tmp_path / "nothing"))


def test_preview_and_heatmap_cache(site):
    _, proj, paths = site
    xyz = np.random.default_rng(0).normal(size=(100, 3))
    proj.save_preview(proj.scans[0], xyz)
    assert np.array_equal(proj.load_preview(proj.scans[0]), xyz)
    assert proj.load_preview(proj.scans[1]) is None
    errs = [np.arange(5.0), np.full(3, np.nan), np.zeros(0)]
    proj.save_heatmap(errs)
    got = proj.load_heatmap()
    assert np.array_equal(got[0], errs[0]) and np.isnan(got[1]).all()
    proj.remove([proj.scans[2].key])
    assert proj.load_heatmap() is None  # made for other scans
    # a changed scan gets a new hash, so its old preview is not used
    _scan(paths[0], n=700, seed=5)
    proj.scans[0].stamp()
    assert proj.load_preview(proj.scans[0]) is None


def test_adding_the_same_scan_twice_and_multi_scan_keys(site, tmp_path):
    _, proj, paths = site
    assert proj.add(open_scans(paths[0])[0]) is proj.scans[0]
    assert len(proj.scans) == 3


def test_save_as_copies_cache_but_not_exports(site, tmp_path):
    _, proj, _ = site
    proj.save_preview(proj.scans[0], np.zeros((4, 3)))
    open(os.path.join(proj.exports_dir, "m.pcd"), "w").write("x")
    copy = proj.save_as(str(tmp_path / "Copy"))
    assert copy.folder.endswith("Copy.pcdproj") and len(copy.scans) == 3
    assert copy.load_preview(copy.scans[0]) is not None
    assert os.listdir(copy.exports_dir) == []


def test_copy_scans_in_makes_a_self_contained_project(site):
    tmp, proj, _ = site
    proj.copy_scans_in()
    proj.save()
    shutil.rmtree(tmp / "scans")
    again = Project.open(proj.folder)
    assert again.check_scans() == []
    assert all(s.path.startswith(proj.folder) for s in again.scans)


def test_from_manifest_makes_the_merged_cloud_a_fixed_base(tmp_path):
    merged = _scan(tmp_path / "merged.pcd", n=2000)
    write_manifest(merged, [{"file": "/old/a.pcd", "points": 1000}])
    proj = from_manifest(merged, str(tmp_path / "Site"))
    assert proj.scans[0].base and proj.scans[0].locked
    assert proj.settings["merged_from"][0]["file"] == "/old/a.pcd"


def test_recent_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("PCD_MERGE_CONFIG_DIR", str(tmp_path / "cfg"))
    a = Project.create(str(tmp_path / "a")).folder
    b = Project.create(str(tmp_path / "b")).folder
    add_recent(a)
    add_recent(b)
    add_recent(a)
    assert recent_projects() == [a, b]
    shutil.rmtree(b)
    assert recent_projects() == [a]


def test_quick_hash_and_folder_names(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(os.urandom(3 << 20))
    h = quick_hash(str(p))
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 1
    p.write_bytes(bytes(data))
    assert quick_hash(str(p)) != h
    assert project_folder(str(tmp_path / "x")) == str(tmp_path / "x.pcdproj")
    assert project_folder(str(tmp_path / "x.pcdproj" / "project.json")) == str(tmp_path / "x.pcdproj")


def test_preview_colours_are_cached_with_the_preview(site):
    _, proj, _ = site
    xyz = np.zeros((5, 3))
    attrs = {"intensity": np.arange(5, dtype=np.float32),
             "rgb": np.full((5, 3), 7, np.uint8)}
    assert proj.load_attrs(proj.scans[0]) is None
    proj.save_preview(proj.scans[0], xyz, attrs)
    got = proj.load_attrs(proj.scans[0])
    assert np.array_equal(got["intensity"], attrs["intensity"]) and got["rgb"].dtype == np.uint8
    proj.save_preview(proj.scans[1], xyz, {})  # no intensity or colour: cached as such
    assert proj.load_attrs(proj.scans[1]) == {}
