"""Batch (many scans / whole folder) and incremental (add to a merged cloud) modes."""

import json
import os

import numpy as np
import pytest

from pcd_merge.cli import main
from pcd_merge.files import find_pcd_files, read_manifest
from pcd_merge.merge import merge_clouds, source_ids
from pcd_merge.pcd_io import PointCloud, read_pcd, write_pcd
from pcd_merge.registration import AlignSettings, align
from pcd_merge.synthetic import make_scan, make_scene, make_transform, transform_error

# Sensor positions around a ~40 m site. In file order, scans 2 -> 3 and
# 3 -> 4 are far apart and do not overlap, so batch mode has to bridge.
SITE = [(-14, -14), (-6, -14), (2, -14), (6, 12), (-4, 12), (10, -12), (14, -4)]


@pytest.fixture(scope="module")
def site():
    rng = np.random.default_rng(1)
    poses = [make_transform(rng.uniform(-3, 3), 0.02, (*p, 1.5)) for p in SITE]
    scene = make_scene(seed=0, density=60)
    scans = [make_scan(scene, p[:2, 3], 16, p, seed=i) for i, p in enumerate(poses)]
    return poses, scans


def _xyz(d):
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def test_find_pcd_files_recurses_in_natural_order(tmp_path):
    for rel in ["b/scan_10.pcd", "b/scan_2.pcd", "a/scan_1.PCD", "a/notes.txt", ".hidden/x.pcd"]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    found = [os.path.relpath(f, tmp_path).replace(os.sep, "/") for f in find_pcd_files([str(tmp_path)])]
    assert found == ["a/scan_1.PCD", "b/scan_2.pcd", "b/scan_10.pcd"]
    # files given directly keep their order, duplicates dropped
    one = str(tmp_path / "b/scan_10.pcd")
    assert find_pcd_files([one, str(tmp_path / "b")])[:1] == [one]
    assert len(find_pcd_files([one, str(tmp_path / "b")])) == 2


def test_batch_bridges_gaps_and_flags_junk(site):
    poses, scans = site
    rng = np.random.default_rng(5)
    junk = rng.uniform(-50, 50, (20000, 3)) + 500
    clouds = [_xyz(s) for s in scans] + [junk]
    r = align(clouds, AlignSettings(seed=1, exhaustive_max=3))
    for i in range(len(scans)):
        rot, trans = transform_error(r.transforms[i], np.linalg.inv(poses[0]) @ poses[i])
        assert rot < 0.2 and trans < 0.03, f"scan {i}: {rot:.3f} deg, {trans:.3f}"
    assert r.connected == [True] * len(scans) + [False]


def test_reference_moves_if_first_cloud_matches_nothing(site):
    poses, scans = site
    junk = np.random.default_rng(6).uniform(-50, 50, (20000, 3)) + 500
    r = align([junk] + [_xyz(s) for s in scans[:3]], AlignSettings(seed=2))
    assert r.reference == 1 and not r.connected[0]
    rot, trans = transform_error(r.transforms[3], np.linalg.inv(poses[0]) @ poses[2])
    assert rot < 0.2 and trans < 0.03
    assert any("reference" in w for w in r.warnings())


def test_source_numbering_continues_from_existing_merge():
    a = np.zeros(4, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("source", "<u2")])
    a["source"] = [0, 0, 1, 2]
    b = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    clouds = [PointCloud(a), PointCloud(b), PointCloud(b.copy())]
    assert source_ids(clouds) == [None, 3, 4]
    merged, partial = merge_clouds(clouds, [np.eye(4)] * 3)  # numbered even without the flag
    assert partial == []
    assert merged["source"].tolist() == [0, 0, 1, 2, 3, 3, 4, 4]


def test_cli_folder_then_add_new_scans(tmp_path, site):
    poses, scans = site
    day1, day2 = tmp_path / "site" / "day1", tmp_path / "site" / "day2"
    day1.mkdir(parents=True)
    day2.mkdir()
    for i in range(3):
        write_pcd(str(day1 / f"scan_{i + 1}.pcd"), scans[i])
    for i in (5, 6):
        write_pcd(str(day2 / f"scan_{i + 1}.pcd"), scans[i])
    out = str(tmp_path / "merged.pcd")
    assert main([str(day1), "-o", out, "--seed", "1", "--source-field"]) == 0
    assert len(read_manifest(out)) == 3

    # Whole site folder again: day-1 scans are recognised and skipped, the new
    # ones are aligned to the merged cloud and appended in its frame.
    assert main(["--add-to", out, str(tmp_path / "site"), "-o", out, "--seed", "2"]) == 0
    m = read_pcd(out)
    assert len(m) == sum(len(scans[i]) for i in (0, 1, 2, 5, 6))
    assert sorted(np.unique(m.data["source"]).tolist()) == [0, 1, 2, 3, 4]
    entries = json.load(open(out + ".json"))["scans"]
    assert [os.path.basename(e["file"]) for e in entries] == [
        "scan_1.pcd", "scan_2.pcd", "scan_3.pcd", "scan_6.pcd", "scan_7.pcd"]
    for e in entries:
        k = int(os.path.basename(e["file"])[5]) - 1
        truth = np.linalg.inv(poses[0]) @ poses[k]
        rot, trans = transform_error(np.array(e["transform"]), truth)
        assert rot < 0.2 and trans < 0.03, e["file"]
