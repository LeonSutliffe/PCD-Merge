"""Save a merged cloud as PCD, PLY, LAS, LAZ or E57.

``write_cloud(path, data)`` picks the format from the file extension and
returns a list of notes about anything the format could not store exactly
(for example LiDAR fields that E57 has no slot for), so the caller can tell
the user.

What each format keeps:

* **PCD**: everything (see ``pcd_io``).
* **PLY**: everything. Every field becomes a PLY property; packed ``rgb``
  becomes ``red``/``green``/``blue`` and normals become ``nx``/``ny``/``nz``
  so CloudCompare, MeshLab and Blender recognise them.
* **LAS / LAZ** (LAS 1.4, point format 6, or 7 with colour): intensity,
  colour, GPS time (from a ``timestamp``/``time``/``t``/``gps_time`` field),
  classification and the ``source`` field (as point source ID) go into the
  standard LAS slots; every other field is kept as an "extra bytes"
  dimension. Coordinates are stored at 0.1 mm resolution (coarser only for
  huge extents).
* **E57**: coordinates (double precision), intensity, colour and time stamp.
  E57 has no place for other per-point fields, so those are left out and
  listed in the notes.
"""

from __future__ import annotations

import os

import numpy as np

from .pcd_io import write_pcd

# extension -> label shown in file dialogs
FORMATS = {
    ".pcd": "PCD point cloud",
    ".ply": "PLY point cloud",
    ".las": "LAS (survey / GIS)",
    ".laz": "LAZ (compressed LAS)",
    ".e57": "E57 (scanner exchange)",
}

_TIME_FIELDS = ("gps_time", "timestamp", "time", "t")
_NORMAL_SETS = (("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz"))


def file_filter() -> str:
    """Qt file-dialog filter string listing every supported format."""
    return ";;".join(f"{label} (*{ext})" for ext, label in FORMATS.items())


def format_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        raise ValueError(f"unsupported output format '{ext or path}'; "
                         f"use one of {', '.join(FORMATS)}")
    return ext


def write_cloud(path: str, data: np.ndarray, compressed: bool = False) -> list[str]:
    """Write ``data`` (a structured array from ``merge_clouds``) to ``path``.

    ``compressed`` only affects PCD (binary_compressed). The file is written
    to ``path + '.part'`` first and renamed when complete.
    """
    ext = format_of(path)
    if ext == ".pcd":
        write_pcd(path, data, compressed=compressed)
        return []
    tmp = path + ".part" + ext  # libraries pick the codec from the extension
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        if ext == ".ply":
            notes = _write_ply(tmp, data)
        elif ext in (".las", ".laz"):
            notes = _write_las(tmp, data, ext == ".laz")
        else:
            notes = _write_e57(tmp, data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return notes


# --------------------------------------------------------------------------
# helpers


def _columns(data: np.ndarray):
    """Yield (name, 1-D array) for every field, expanding COUNT>1 fields."""
    for name in data.dtype.names:
        col = data[name]
        if col.ndim == 1:
            yield name, col
        else:
            flat = col.reshape(len(data), -1)
            for k in range(flat.shape[1]):
                yield f"{name}_{k}", flat[:, k]


def _unpack_rgb(col: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a PCL packed colour (float or uint32 bit pattern) into r, g, b."""
    if col.dtype.itemsize != 4:
        col = col.astype(np.uint32)
    v = np.ascontiguousarray(col).view(np.uint32)
    return ((v >> 16) & 0xFF).astype(np.uint8), ((v >> 8) & 0xFF).astype(np.uint8), \
        (v & 0xFF).astype(np.uint8)


def _time_field(names) -> str | None:
    for n in _TIME_FIELDS:
        if n in names:
            return n
    return None


# --------------------------------------------------------------------------
# PLY

_PLY_TYPES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
              "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


def _write_ply(path: str, data: np.ndarray) -> list[str]:
    notes = []
    rename = {}
    names = set(data.dtype.names)
    if all(n in names for n in _NORMAL_SETS[0]) and not any(n in names for n in _NORMAL_SETS[1]):
        rename = dict(zip(_NORMAL_SETS[0], _NORMAL_SETS[1]))

    cols: list[tuple[str, np.ndarray]] = []
    for name, col in _columns(data):
        if name in ("rgb", "rgba") and not {"red", "green", "blue"} & names:
            r, g, b = _unpack_rgb(col)
            cols += [("red", r), ("green", g), ("blue", b)]
            continue
        key = col.dtype.base.str[1:]
        if key not in _PLY_TYPES:  # 64-bit integers: PLY has none
            col = col.astype(np.float64)
            notes.append(f"PLY has no 64-bit integers; '{name}' was saved as double")
        cols.append((rename.get(name, name), col))

    out_dtype = np.dtype([(n, c.dtype.newbyteorder("<")) for n, c in cols])
    out = np.empty(len(data), dtype=out_dtype)
    for n, c in cols:
        out[n] = c
    header = ["ply", "format binary_little_endian 1.0", "comment written by PCD Merge",
              f"element vertex {len(out)}"]
    header += [f"property {_PLY_TYPES[c.dtype.base.str[1:]]} {n}" for n, c in cols]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(out.tobytes())
    return notes


# --------------------------------------------------------------------------
# LAS / LAZ


def _las_scales(xyz_min: np.ndarray, xyz_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0.1 mm resolution unless the extent is too large for 32-bit integers."""
    offset = np.floor(np.where(np.isfinite(xyz_min), xyz_min, 0.0))
    extent = float(np.max(xyz_max - offset, initial=0.0))
    scale = 0.0001
    while extent / scale > 2**31 - 2:
        scale *= 10
    return offset, np.full(3, scale)


def _intensity_u16(col: np.ndarray) -> tuple[np.ndarray, bool]:
    """Map an intensity field to LAS's uint16. Returns (values, exact)."""
    vals = col.astype(np.float64)
    if vals.size == 0:
        return vals.astype(np.uint16), True
    finite = vals[np.isfinite(vals)]
    lo = float(finite.min(initial=0.0))
    hi = float(finite.max(initial=0.0))
    if lo >= 0 and hi <= 65535 and np.all(finite == np.round(finite)):
        return np.nan_to_num(vals).astype(np.uint16), True
    # Rescale to use the full 16-bit range, as the LAS spec recommends.
    span = hi - min(lo, 0.0)
    scaled = (np.nan_to_num(vals) - min(lo, 0.0)) * (65535.0 / span if span > 0 else 0.0)
    return np.clip(np.round(scaled), 0, 65535).astype(np.uint16), False


def _write_las(path: str, data: np.ndarray, compress: bool) -> list[str]:
    import laspy

    notes: list[str] = []
    names = list(data.dtype.names)
    has_rgb = any(n in names for n in ("rgb", "rgba")) or all(
        n in names for n in ("red", "green", "blue"))
    header = laspy.LasHeader(point_format=7 if has_rgb else 6, version="1.4")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    if len(xyz):
        header.offsets, header.scales = _las_scales(xyz.min(axis=0), xyz.max(axis=0))
    else:
        header.offsets, header.scales = np.zeros(3), np.full(3, 0.0001)

    # Decide which PCD field fills which standard LAS dimension.
    tf = _time_field(names)
    cls_field = next((c for c in ("classification", "label")
                      if c in names and data[c].dtype.kind in "iu" and data[c].ndim == 1), None)
    standard = {d.lower() for d in
                laspy.PointFormat(header.point_format.id).standard_dimension_names}
    consumed = {"x", "y", "z"}
    if "intensity" in names and data["intensity"].ndim == 1:
        consumed.add("intensity")
    if has_rgb:
        consumed |= {"rgb", "rgba", "red", "green", "blue"} & set(names)
    if tf and data[tf].ndim == 1:
        consumed.add(tf)
    else:
        tf = None
    if "source" in names and data["source"].ndim == 1:
        consumed.add("source")
    if cls_field:
        consumed.add(cls_field)
    taken = {"x", "y", "z", "intensity", "red", "green", "blue", "gps_time",
             "point_source_id", "classification"}

    same_name: list[tuple[str, np.ndarray]] = []  # e.g. return_number, user_data
    extra: list[tuple[str, np.ndarray]] = []
    for name, col in _columns(data):
        if name in consumed or col.dtype.kind not in "iuf":
            continue
        low = name.lower()
        if low in standard and low not in taken:
            same_name.append((low, col))
            taken.add(low)
        elif low in standard or low in taken:
            extra.append((f"{name}_pcd", col))  # avoid clashing with a LAS dimension
        else:
            extra.append((name, col))

    intensity_exact = True
    if "intensity" in consumed:
        intensity_vals, intensity_exact = _intensity_u16(data["intensity"])
        if not intensity_exact:
            extra.append(("intensity_raw", data["intensity"].astype(np.float32)))
            notes.append("LAS stores intensity as whole numbers 0-65535, so it was rescaled "
                         "to that range; the exact values are kept in the extra field "
                         "'intensity_raw'")

    if extra:
        header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=c.dtype.base.newbyteorder("<"))
                               for n, c in extra])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if "intensity" in consumed:
        las.intensity = intensity_vals
    if has_rgb:
        if all(n in names for n in ("red", "green", "blue")):
            r, g, b = (data[n] for n in ("red", "green", "blue"))
            mult = 257 if max(np.max(r, initial=0), np.max(g, initial=0),
                              np.max(b, initial=0)) <= 255 else 1
        else:
            r, g, b = _unpack_rgb(data["rgb" if "rgb" in names else "rgba"])
            mult = 257
        las.red, las.green, las.blue = (
            np.clip(np.asarray(c, np.int64) * mult, 0, 65535).astype(np.uint16) for c in (r, g, b))
    if tf:
        las.gps_time = data[tf].astype(np.float64)
        if tf != "gps_time":
            notes.append(f"'{tf}' was saved as LAS GPS time")
    if "source" in consumed:
        las.point_source_id = data["source"].astype(np.uint16)
    if cls_field:
        las.classification = np.clip(data[cls_field], 0, 255).astype(np.uint8)
    for low, col in same_name:
        try:
            las[low] = col
        except (ValueError, OverflowError, TypeError):
            notes.append(f"field '{low}' did not fit LAS's '{low}' and was left out")
    for n, c in extra:
        las[n] = c
    kept = [n for n, _ in extra if n != "intensity_raw"]
    if kept:
        notes.append("kept as LAS extra fields: " + ", ".join(kept))

    if compress:
        las.write(path, laz_backend=laspy.LazBackend.Lazrs)
    else:
        las.write(path)
    return notes


# --------------------------------------------------------------------------
# E57


def _write_e57(path: str, data: np.ndarray) -> list[str]:
    import uuid

    import pye57
    from pye57 import libe57

    names = list(data.dtype.names)
    n = len(data)
    if n == 0:
        raise ValueError("E57 files need at least one point")
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    z = data["z"].astype(np.float64)
    cols: dict[str, np.ndarray] = {"cartesianX": x, "cartesianY": y, "cartesianZ": z}
    stored = {"x", "y", "z"}
    if "intensity" in names and data["intensity"].ndim == 1:
        cols["intensity"] = data["intensity"].astype(np.float64)
        stored.add("intensity")
    if any(c in names for c in ("rgb", "rgba")):
        key = "rgb" if "rgb" in names else "rgba"
        cols["colorRed"], cols["colorGreen"], cols["colorBlue"] = _unpack_rgb(data[key])
        stored.add(key)
    elif all(c in names for c in ("red", "green", "blue")):
        for e57n, n_ in zip(("colorRed", "colorGreen", "colorBlue"), ("red", "green", "blue")):
            cols[e57n] = np.clip(data[n_], 0, 255).astype(np.uint8)
            stored.add(n_)
    tf = _time_field(names)
    if tf and data[tf].ndim == 1:
        cols["timeStamp"] = data[tf].astype(np.float64)
        stored.add(tf)

    e57 = pye57.E57(path, mode="w")
    img = e57.image_file
    try:
        scan = libe57.StructureNode(img)
        scan.set("guid", libe57.StringNode(img, "{%s}" % uuid.uuid4()))
        scan.set("name", libe57.StringNode(img, "Merged"))
        scan.set("description", libe57.StringNode(img, "Merged by PCD Merge"))

        lo = np.array([x.min(), y.min(), z.min()])
        hi = np.array([x.max(), y.max(), z.max()])
        bounds = libe57.StructureNode(img)
        for k, ax in enumerate("xyz"):
            bounds.set(f"{ax}Minimum", libe57.FloatNode(img, float(lo[k])))
            bounds.set(f"{ax}Maximum", libe57.FloatNode(img, float(hi[k])))
        scan.set("cartesianBounds", bounds)

        proto = libe57.StructureNode(img)
        for k, ax in enumerate("XYZ"):
            proto.set(f"cartesian{ax}", libe57.FloatNode(
                img, float(lo[k]), libe57.E57_DOUBLE, float(lo[k]), float(hi[k])))
        if "intensity" in cols:
            v = cols["intensity"]
            imin, imax = float(np.nanmin(v)), float(np.nanmax(v))
            lim = libe57.StructureNode(img)
            lim.set("intensityMinimum", libe57.FloatNode(img, imin))
            lim.set("intensityMaximum", libe57.FloatNode(img, imax))
            scan.set("intensityLimits", lim)
            proto.set("intensity", libe57.FloatNode(img, imin, libe57.E57_SINGLE, imin, imax))
        if "colorRed" in cols:
            lim = libe57.StructureNode(img)
            for c in ("Red", "Green", "Blue"):
                lim.set(f"color{c}Minimum", libe57.IntegerNode(img, 0))
                lim.set(f"color{c}Maximum", libe57.IntegerNode(img, 255))
                proto.set(f"color{c}", libe57.IntegerNode(img, 0, 0, 255))
            scan.set("colorLimits", lim)
        if "timeStamp" in cols:
            v = cols["timeStamp"]
            proto.set("timeStamp", libe57.FloatNode(
                img, float(v.min()), libe57.E57_DOUBLE, float(v.min()), float(v.max())))

        points = libe57.CompressedVectorNode(img, proto, libe57.VectorNode(img, True))
        scan.set("points", points)
        e57.data3d.append(scan)

        chunk = min(n, 1_000_000)
        bufs = libe57.VectorSourceDestBuffer()
        arrays = {}
        for name, v in cols.items():
            arr = np.empty(chunk, dtype=v.dtype)
            arrays[name] = arr
            bufs.append(libe57.SourceDestBuffer(img, name, arr, chunk, True, True))
        writer = points.writer(bufs)
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            for name, arr in arrays.items():
                arr[:m] = cols[name][start:start + m]
            writer.write(m)
        writer.close()
    finally:
        e57.close()

    left_out = [f for f in names if f not in stored]
    if left_out:
        return ["E57 has no place for these fields, so they were left out: "
                + ", ".join(left_out) + " (save as PCD, PLY or LAS to keep them)"]
    return []
