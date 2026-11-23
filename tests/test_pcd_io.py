import numpy as np
import pytest

from pcd_merge.pcd_io import PCDError, _lzf_decompress_py, read_pcd, write_pcd

lzf = pytest.importorskip("lzf")


def _sample(n=500):
    rng = np.random.default_rng(1)
    d = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                           ("ring", "<u2"), ("t", "<f8"), ("hist", "<f4", (3,))])
    for k in ("x", "y", "z", "intensity"):
        d[k] = rng.normal(size=n)
    d["ring"] = rng.integers(0, 64, n)
    d["t"] = rng.uniform(1e9, 2e9, n)
    d["hist"] = rng.normal(size=(n, 3))
    return d


@pytest.mark.parametrize("mode", ["binary", "compressed", "ascii"])
def test_roundtrip_keeps_all_fields(tmp_path, mode):
    d = _sample()
    p = str(tmp_path / f"{mode}.pcd")
    write_pcd(p, d, binary=mode != "ascii", compressed=mode == "compressed")
    c = read_pcd(p)
    assert c.field_names == d.dtype.names
    for name in d.dtype.names:
        np.testing.assert_array_equal(c.data[name], d[name])


def test_reads_pcl_style_ascii_header(tmp_path):
    p = tmp_path / "a.pcd"
    p.write_text(
        "# .PCD v.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z intensity\n"
        "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA ascii\n"
        "1 2 3 10\n4 5 6 20\nnan nan nan 0\n"
    )
    c = read_pcd(str(p))
    assert len(c) == 3
    assert c.finite_mask().tolist() == [True, True, False]
    np.testing.assert_allclose(c.data["intensity"], [10, 20, 0])


def test_padding_fields_are_kept_unique(tmp_path):
    p = tmp_path / "pad.pcd"
    header = ("VERSION .7\nFIELDS x y z _ intensity _\nSIZE 4 4 4 1 4 1\nTYPE F F F U F U\n"
              "COUNT 1 1 1 4 1 12\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n")
    rec = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("_", "u1", (4,)),
                             ("intensity", "<f4"), ("_1", "u1", (12,))])
    rec["x"] = [1, 2]
    rec["intensity"] = [7, 8]
    p.write_bytes(header.encode() + rec.tobytes())
    c = read_pcd(str(p))
    np.testing.assert_array_equal(c.data["intensity"], [7, 8])
    np.testing.assert_array_equal(c.data["x"], [1, 2])


def test_pure_python_lzf_matches_c():
    raw = (b"hello hello hello world " * 200) + bytes(range(256)) * 4
    comp = lzf.compress(raw)
    assert _lzf_decompress_py(comp, len(raw)) == raw


def test_not_a_pcd(tmp_path):
    p = tmp_path / "x.pcd"
    p.write_text("hello\nDATA ascii\n")
    with pytest.raises(PCDError):
        read_pcd(str(p))


def test_missing_xyz(tmp_path):
    p = tmp_path / "x.pcd"
    p.write_text("FIELDS a b\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n1 2\n")
    with pytest.raises(PCDError):
        read_pcd(str(p))
