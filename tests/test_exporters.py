import numpy as np
import pytest

from pcd_merge.exporters import FORMATS, file_filter, format_of, write_cloud
from pcd_merge.pcd_io import read_pcd

laspy = pytest.importorskip("laspy")


def _sample(n=2000, intensity_float=True, offset=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(3)
    dt = [("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
          ("intensity", "<f4" if intensity_float else "<u2"), ("ring", "<u2"),
          ("timestamp", "<f8"), ("rgb", "<f4"),
          ("normal_x", "<f4"), ("normal_y", "<f4"), ("normal_z", "<f4"),
          ("hist", "<f4", (2,)), ("source", "<u2")]
    d = np.zeros(n, dtype=dt)
    for k, o in zip("xyz", offset):
        d[k] = o + rng.uniform(-80, 80, n)
    d["intensity"] = rng.uniform(0, 1, n) if intensity_float else rng.integers(0, 256, n)
    d["ring"] = rng.integers(0, 64, n)
    d["timestamp"] = 1.7e9 + rng.uniform(0, 10, n)
    packed = (rng.integers(0, 256, n) << 16 | rng.integers(0, 256, n) << 8
              | rng.integers(0, 256, n)).astype(np.uint32)
    d["rgb"] = packed.view(np.float32)
    d["normal_z"] = 1
    d["hist"] = rng.normal(size=(n, 2))
    d["source"] = rng.integers(0, 3, n)
    return d


def _rgb(d):
    v = d["rgb"].view(np.uint32)
    return (v >> 16) & 255, (v >> 8) & 255, v & 255


def _read_ply(path):
    with open(path, "rb") as f:
        assert f.readline() == b"ply\n"
        assert f.readline() == b"format binary_little_endian 1.0\n"
        types = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
                 "uint": "u4", "float": "f4", "double": "f8"}
        fields, n = [], 0
        while (line := f.readline().decode().split()) != ["end_header"]:
            if line[0] == "element":
                n = int(line[2])
            elif line[0] == "property":
                fields.append((line[2], "<" + types[line[1]]))
        return np.frombuffer(f.read(), dtype=fields, count=n)


def test_format_from_extension():
    assert format_of("a/b/Merged.LAZ") == ".laz"
    with pytest.raises(ValueError):
        format_of("merged.xyz")
    assert all(ext in file_filter() for ext in FORMATS)


def test_pcd_unchanged(tmp_path):
    d = _sample(200)
    p = str(tmp_path / "m.pcd")
    assert write_cloud(p, d) == []
    c = read_pcd(p)
    for name in d.dtype.names:
        np.testing.assert_array_equal(c.data[name], d[name])


def test_ply_keeps_every_field(tmp_path):
    d = _sample()
    p = str(tmp_path / "m.ply")
    assert write_cloud(p, d) == []
    out = _read_ply(p)
    for k in ("x", "y", "z", "intensity", "ring", "timestamp", "source"):
        np.testing.assert_array_equal(out[k], d[k])
    for k, v in zip(("red", "green", "blue"), _rgb(d)):
        np.testing.assert_array_equal(out[k], v)
    np.testing.assert_array_equal(out["nz"], d["normal_z"])
    np.testing.assert_array_equal(out["hist_1"], d["hist"][:, 1])


@pytest.mark.parametrize("ext", [".las", ".laz"])
def test_las_keeps_intensity_colour_time_and_extras(tmp_path, ext):
    d = _sample()
    p = str(tmp_path / f"m{ext}")
    notes = write_cloud(p, d)
    assert any("intensity_raw" in n for n in notes)
    las = laspy.read(p)
    assert las.header.version.minor == 4 and las.point_format.id == 7
    for k in "xyz":
        assert np.abs(np.asarray(las[k]) - d[k]).max() <= 0.0001
    # float intensity: rescaled to 16 bits, exact values in an extra field
    assert las.intensity.max() == 65535
    np.testing.assert_array_equal(las.intensity_raw, d["intensity"])
    order = np.argsort(d["intensity"])
    assert np.all(np.diff(las.intensity[order].astype(int)) >= 0)
    for k, v in zip(("red", "green", "blue"), _rgb(d)):
        np.testing.assert_array_equal(las[k], v * 257)
    np.testing.assert_array_equal(las.gps_time, d["timestamp"])
    np.testing.assert_array_equal(las.point_source_id, d["source"])
    np.testing.assert_array_equal(las.ring, d["ring"])
    np.testing.assert_array_equal(las.normal_z, d["normal_z"])
    np.testing.assert_array_equal(las.hist_0, d["hist"][:, 0])


def test_las_integer_intensity_exact_and_large_coordinates(tmp_path):
    d = _sample(intensity_float=False, offset=(512_000.0, 5_400_000.0, 300.0))
    p = str(tmp_path / "m.las")
    notes = write_cloud(p, d)
    assert not any("intensity" in n for n in notes)
    las = laspy.read(p)
    np.testing.assert_array_equal(las.intensity, d["intensity"])
    assert "intensity_raw" not in las.point_format.extra_dimension_names
    for k in "xyz":
        assert np.abs(np.asarray(las[k]) - d[k]).max() <= 0.0001


def test_e57_keeps_coordinates_intensity_colour_time(tmp_path):
    pye57 = pytest.importorskip("pye57")
    d = _sample(offset=(512_000.0, 5_400_000.0, 300.0))
    p = str(tmp_path / "m.e57")
    notes = write_cloud(p, d)
    assert notes and "ring" in notes[0]
    e = pye57.E57(p)
    assert e.scan_count == 1
    assert "timeStamp" in e.get_header(0).point_fields
    r = e.read_scan_raw(0, ignore_unsupported_fields=True)
    for k, e57n in zip("xyz", ("cartesianX", "cartesianY", "cartesianZ")):
        np.testing.assert_array_equal(r[e57n], d[k])  # double precision
    np.testing.assert_array_equal(r["intensity"], d["intensity"])
    for k, v in zip(("colorRed", "colorGreen", "colorBlue"), _rgb(d)):
        np.testing.assert_array_equal(r[k], v)


def test_failed_write_leaves_no_file(tmp_path):
    d = _sample(0)
    p = tmp_path / "m.e57"
    with pytest.raises(ValueError):
        write_cloud(str(p), d)
    assert list(tmp_path.iterdir()) == []
