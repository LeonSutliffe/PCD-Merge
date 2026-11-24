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

Every format is written in chunks by a :class:`CloudWriter`, so a big merged
cloud can be streamed to disk piece by piece (:func:`write_chunks`) without
ever being held in memory as a whole. LAS and E57 need value ranges (bounds,
intensity range) in their headers before the first point, so for those the
chunks are first measured (:func:`survey`), then written.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable, Iterator

import numpy as np

from .pcd_io import _KIND_TO_PCD, write_pcd

# extension -> label shown in file dialogs
FORMATS = {
    ".pcd": "PCD point cloud",
    ".ply": "PLY point cloud",
    ".las": "LAS (survey / GIS)",
    ".laz": "LAZ (compressed LAS)",
    ".e57": "E57 (scanner exchange)",
    ".copc.laz": "COPC (cloud-optimised LAZ, for viewers)",
}

_TIME_FIELDS = ("gps_time", "timestamp", "time", "t")
_NORMAL_SETS = (("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz"))
_COUNT_WIDTH = 20  # header point counts are padded so they can be filled in at the end


def file_filter() -> str:
    """Qt file-dialog filter string listing every supported format."""
    return ";;".join(f"{label} (*{ext})" for ext, label in FORMATS.items())


def format_of(path: str) -> str:
    if path.lower().endswith(".copc.laz"):
        return ".copc.laz"
    ext = os.path.splitext(path)[1].lower()
    if ext not in FORMATS:
        raise ValueError(f"unsupported output format '{ext or path}'; "
                         f"use one of {', '.join(FORMATS)}")
    return ext


def needs_survey(path: str) -> bool:
    """Whether writing this format needs value ranges before the first point."""
    return format_of(path) in (".las", ".laz", ".e57", ".copc.laz")


def write_cloud(path: str, data: np.ndarray, compressed: bool = False,
                progress=None) -> list[str]:
    """Write ``data`` (a structured array from ``merge_clouds``) to ``path``.

    ``compressed`` only affects PCD (binary_compressed). The file is written
    to a ``.part`` file first and renamed when complete. ``progress(fraction)``
    is reported while writing.
    """
    if format_of(path) == ".pcd":
        write_pcd(path, data, compressed=compressed, progress=progress)
        return []
    return write_chunks(path, data.dtype, lambda: _slices(data), total=len(data),
                        progress=progress)


def _slices(data: np.ndarray, size: int = 1_000_000) -> Iterator[np.ndarray]:
    if not len(data):
        yield data
    for s in range(0, len(data), size):
        yield data[s:s + size]


def write_chunks(path: str, dtype: np.dtype, chunks: Callable[[], Iterable[np.ndarray]],
                 total: int | None = None, progress=None) -> list[str]:
    """Write a cloud that arrives in pieces.

    ``chunks()`` must return a fresh iterable of structured arrays (all of
    ``dtype``) each time it is called: formats whose header needs value
    ranges call it twice. ``total`` (points, an upper bound is fine) is only
    used for progress. Returns the notes, as :func:`write_cloud` does.
    """
    report = progress or (lambda _f: None)
    two_pass = needs_survey(path)
    stats = None
    if two_pass:
        stats = survey(chunks(), dtype,
                       progress=(lambda n: report(0.3 * n / total)) if total else None)
    writer = open_writer(path, dtype, stats)
    done = 0
    try:
        for c in chunks():
            if len(c):
                writer.write(c)
            done += len(c)
            if total:
                f = done / total
                report(0.3 + 0.7 * f if two_pass else f)
        notes = writer.close()
    except BaseException:
        writer.abort()
        raise
    report(1.0)
    return notes


# --------------------------------------------------------------------------
# value ranges


class FieldStats:
    """Range of one 1-D field across all chunks."""

    def __init__(self):
        self.lo = np.inf
        self.hi = -np.inf
        self.integral = True

    def add(self, col: np.ndarray):
        v = col[np.isfinite(col)] if col.dtype.kind == "f" else col
        if not len(v):
            return
        self.lo = min(self.lo, float(v.min()))
        self.hi = max(self.hi, float(v.max()))
        if self.integral and col.dtype.kind == "f":
            self.integral = bool(np.all(v == np.round(v)))


def survey(chunks: Iterable[np.ndarray], dtype: np.dtype, progress=None) -> dict:
    """Ranges of every 1-D numeric field: ``{name: FieldStats}`` plus ``"_count"``."""
    names = [n for n in dtype.names if dtype[n].ndim == 0 and dtype[n].kind in "iuf"
             and n not in ("rgb", "rgba")]
    stats = {n: FieldStats() for n in names}
    count = 0
    for c in chunks:
        for n in names:
            stats[n].add(c[n])
        count += len(c)
        if progress:
            progress(count)
    stats["_count"] = count
    return stats


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


def _column_types(dtype: np.dtype):
    """(name, scalar dtype) for every column :func:`_columns` would yield."""
    for name in dtype.names:
        sub = dtype[name]
        if not sub.shape:
            yield name, sub
        else:
            for k in range(int(np.prod(sub.shape))):
                yield f"{name}_{k}", sub.base


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


class CloudWriter:
    """Writes one point file chunk by chunk. ``close()`` returns the notes."""

    def __init__(self, path: str, tmp: str):
        self.path, self.tmp = path, tmp
        if os.path.exists(tmp):
            os.remove(tmp)

    def write(self, chunk: np.ndarray) -> None:
        raise NotImplementedError

    def _finish(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> list[str]:
        notes = self._finish()
        os.replace(self.tmp, self.path)
        return notes

    def abort(self) -> None:
        try:
            self._release()
        except Exception:
            pass
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _release(self) -> None:
        pass


def open_writer(path: str, dtype: np.dtype, stats: dict | None = None) -> CloudWriter:
    """A chunk writer for ``path`` (format from the extension). LAS/LAZ/E57
    need ``stats`` from :func:`survey`."""
    ext = format_of(path)
    tmp = path + ".part" + ext  # libraries pick the codec from the extension
    if ext == ".pcd":
        return _PCDWriter(path, path + ".part", dtype)
    if ext == ".ply":
        return _PLYWriter(path, tmp, dtype)
    if stats is None:
        raise ValueError(f"{ext} output needs the value ranges first (see survey())")
    if ext in (".las", ".laz"):
        return _LASWriter(path, tmp, dtype, stats, ext == ".laz")
    if ext == ".copc.laz":
        return _COPCWriter(path, path + ".part.laz", dtype, stats)
    return _E57Writer(path, tmp, dtype, stats)


def _padded_count_line(prefix: str, n: int) -> bytes:
    return f"{prefix} {n:<{_COUNT_WIDTH}d}".rstrip().ljust(len(prefix) + 1 + _COUNT_WIDTH).encode()


# --------------------------------------------------------------------------
# PCD (binary), streamed


class _PCDWriter(CloudWriter):
    def __init__(self, path, tmp, dtype):
        super().__init__(path, tmp)
        self.dtype = dtype.newbyteorder("<")
        names, sizes, types, counts = [], [], [], []
        for name in dtype.names:
            base, shape = dtype.fields[name][0].base, dtype.fields[name][0].shape
            names.append(name)
            sizes.append(str(base.itemsize))
            types.append(_KIND_TO_PCD[base.kind])
            counts.append(str(int(np.prod(shape)) if shape else 1))
        self.f = open(tmp, "wb")
        self.f.write(("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                      f"FIELDS {' '.join(names)}\nSIZE {' '.join(sizes)}\n"
                      f"TYPE {' '.join(types)}\nCOUNT {' '.join(counts)}\n").encode())
        self._width_at = self.f.tell()
        self.f.write(_padded_count_line("WIDTH", 0) + b"\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        self._points_at = self.f.tell()
        self.f.write(_padded_count_line("POINTS", 0) + b"\nDATA binary\n")
        self.n = 0

    def write(self, chunk):
        self.f.write(np.ascontiguousarray(chunk.astype(self.dtype, copy=False)).tobytes())
        self.n += len(chunk)

    def _finish(self):
        self.f.seek(self._width_at)
        self.f.write(_padded_count_line("WIDTH", self.n))
        self.f.seek(self._points_at)
        self.f.write(_padded_count_line("POINTS", self.n))
        self.f.close()
        return []

    def _release(self):
        self.f.close()


# --------------------------------------------------------------------------
# PLY

_PLY_TYPES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
              "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}


class _PLYWriter(CloudWriter):
    def __init__(self, path, tmp, dtype):
        super().__init__(path, tmp)
        self.notes = []
        names = set(dtype.names)
        rename = {}
        if all(n in names for n in _NORMAL_SETS[0]) and not any(n in names for n in _NORMAL_SETS[1]):
            rename = dict(zip(_NORMAL_SETS[0], _NORMAL_SETS[1]))
        self.unpack = None
        cols = []  # (output name, output type, source column name)
        for name, t in _column_types(dtype):
            if name in ("rgb", "rgba") and not {"red", "green", "blue"} & names and self.unpack is None:
                self.unpack = name
                cols += [("red", np.dtype("u1"), None), ("green", np.dtype("u1"), None),
                         ("blue", np.dtype("u1"), None)]
                continue
            key = t.str[1:]
            if key not in _PLY_TYPES:  # 64-bit integers: PLY has none
                t = np.dtype("f8")
                self.notes.append(f"PLY has no 64-bit integers; '{name}' was saved as double")
            cols.append((rename.get(name, name), t.newbyteorder("<"), name))
        self.cols = cols
        self.out_dtype = np.dtype([(n, t) for n, t, _ in cols])
        self.f = open(tmp, "wb")
        self.f.write(b"ply\nformat binary_little_endian 1.0\ncomment written by PCD Merge\n")
        self._count_at = self.f.tell()
        self.f.write(_padded_count_line("element vertex", 0) + b"\n")
        props = [f"property {_PLY_TYPES[t.str[1:]]} {n}" for n, t, _ in cols]
        self.f.write(("\n".join(props + ["end_header"]) + "\n").encode())
        self.n = 0

    def write(self, chunk):
        out = np.empty(len(chunk), dtype=self.out_dtype)
        src = dict(_columns(chunk))
        if self.unpack:
            out["red"], out["green"], out["blue"] = _unpack_rgb(chunk[self.unpack])
        for n, _t, s in self.cols:
            if s is not None:
                out[n] = src[s]
        self.f.write(out.tobytes())
        self.n += len(chunk)

    def _finish(self):
        self.f.seek(self._count_at)
        self.f.write(_padded_count_line("element vertex", self.n))
        self.f.close()
        return self.notes

    def _release(self):
        self.f.close()


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


def _intensity_map(st: FieldStats, kind: str):
    """How to fit an intensity field into LAS's uint16: returns (exact, lo, factor)."""
    if st.lo == np.inf:  # no values
        return True, 0.0, 0.0
    if st.lo >= 0 and st.hi <= 65535 and (kind in "iu" or st.integral):
        return True, 0.0, 1.0
    # Rescale to use the full 16-bit range, as the LAS spec recommends.
    lo = min(st.lo, 0.0)
    span = st.hi - lo
    return False, lo, (65535.0 / span if span > 0 else 0.0)


def _intensity_u16(col: np.ndarray) -> tuple[np.ndarray, bool]:
    """Map an intensity field to LAS's uint16 on its own. Returns (values, exact)."""
    st = FieldStats()
    st.add(col)
    exact, lo, factor = _intensity_map(st, col.dtype.kind)
    return _apply_intensity(col, exact, lo, factor), exact


def _apply_intensity(col, exact, lo, factor):
    vals = np.nan_to_num(col.astype(np.float64))
    if exact:
        return vals.astype(np.uint16)
    return np.clip(np.round((vals - lo) * factor), 0, 65535).astype(np.uint16)


class _LASWriter(CloudWriter):
    def __init__(self, path, tmp, dtype, stats, compress):
        import laspy

        super().__init__(path, tmp)
        notes: list[str] = []
        names = list(dtype.names)
        self.has_rgb = any(n in names for n in ("rgb", "rgba")) or all(
            n in names for n in ("red", "green", "blue"))
        header = laspy.LasHeader(point_format=7 if self.has_rgb else 6, version="1.4")
        lo = np.array([stats[k].lo for k in "xyz"])
        hi = np.array([stats[k].hi for k in "xyz"])
        if stats["_count"] and np.all(np.isfinite(lo)):
            header.offsets, header.scales = _las_scales(lo, hi)
        else:
            header.offsets, header.scales = np.zeros(3), np.full(3, 0.0001)

        def one_d(n):
            return n in names and dtype[n].ndim == 0

        tf = _time_field(names)
        self.cls_field = next((c for c in ("classification", "label")
                               if one_d(c) and dtype[c].kind in "iu"), None)
        standard = {d.lower() for d in
                    laspy.PointFormat(header.point_format.id).standard_dimension_names}
        consumed = {"x", "y", "z"}
        self.intensity = one_d("intensity")
        if self.intensity:
            consumed.add("intensity")
        if self.has_rgb:
            consumed |= {"rgb", "rgba", "red", "green", "blue"} & set(names)
        self.tf = tf if tf and one_d(tf) else None
        if self.tf:
            consumed.add(self.tf)
        self.source = one_d("source")
        if self.source:
            consumed.add("source")
        if self.cls_field:
            consumed.add(self.cls_field)
        taken = {"x", "y", "z", "intensity", "red", "green", "blue", "gps_time",
                 "point_source_id", "classification"}

        self.same_name: list[tuple[str, str]] = []  # (LAS dimension, column), e.g. user_data
        self.extra: list[tuple[str, str]] = []  # (extra dimension, column)
        extra_types = []
        for name, t in _column_types(dtype):
            if name in consumed or t.kind not in "iuf":
                continue
            low = name.lower()
            if low in standard and low not in taken:
                self.same_name.append((low, name))
                taken.add(low)
            elif low in standard or low in taken:
                self.extra.append((f"{name}_pcd", name))  # avoid clashing with a LAS dimension
                extra_types.append(t)
            else:
                self.extra.append((name, name))
                extra_types.append(t)

        self.int_map = None
        self.intensity_raw = False
        if self.intensity:
            self.int_map = _intensity_map(stats["intensity"], dtype["intensity"].kind)
            if not self.int_map[0]:
                self.intensity_raw = True
                self.extra.append(("intensity_raw", "intensity"))
                extra_types.append(np.dtype("f4"))
                notes.append("LAS stores intensity as whole numbers 0-65535, so it was rescaled "
                             "to that range; the exact values are kept in the extra field "
                             "'intensity_raw'")
        self.rgb_mult = 257
        if self.has_rgb and all(n in names for n in ("red", "green", "blue")):
            top = max(stats[c].hi if c in stats else 0 for c in ("red", "green", "blue"))
            self.rgb_mult = 257 if top <= 255 else 1
        if self.tf and self.tf != "gps_time":
            notes.append(f"'{self.tf}' was saved as LAS GPS time")
        if self.extra:
            header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=t.newbyteorder("<"))
                                   for (n, _), t in zip(self.extra, extra_types)])
        kept = [n for n, _ in self.extra if n != "intensity_raw"]
        if kept:
            notes.append("kept as LAS extra fields: " + ", ".join(kept))
        self.notes = notes
        self.header = header
        self._skipped: set[str] = set()
        self._open(tmp, compress)

    def _open(self, tmp, compress):
        import laspy

        kw = {"laz_backend": laspy.LazBackend.Lazrs} if compress else {}
        self.w = laspy.open(tmp, mode="w", header=self.header, do_compress=compress, **kw)

    def write(self, data):
        self.w.write_points(self.records(data))

    def records(self, data):
        """``data`` as LAS point records (laspy), ready to write."""
        import laspy

        las = laspy.ScaleAwarePointRecord.zeros(len(data), header=self.header)
        las.x, las.y, las.z = data["x"], data["y"], data["z"]
        if self.intensity:
            las.intensity = _apply_intensity(data["intensity"], *self.int_map)
        names = data.dtype.names
        if self.has_rgb:
            if all(n in names for n in ("red", "green", "blue")):
                r, g, b = (data[n] for n in ("red", "green", "blue"))
            else:
                r, g, b = _unpack_rgb(data["rgb" if "rgb" in names else "rgba"])
            las.red, las.green, las.blue = (
                np.clip(np.asarray(c, np.int64) * self.rgb_mult, 0, 65535).astype(np.uint16)
                for c in (r, g, b))
        if self.tf:
            las.gps_time = data[self.tf].astype(np.float64)
        if self.source:
            las.point_source_id = data["source"].astype(np.uint16)
        if self.cls_field:
            las.classification = np.clip(data[self.cls_field], 0, 255).astype(np.uint8)
        cols = dict(_columns(data)) if self.same_name or self.extra else {}
        for low, name in self.same_name:
            if low in self._skipped:
                continue
            try:
                las[low] = cols[name]
            except (ValueError, OverflowError, TypeError):
                self._skipped.add(low)
        for n, name in self.extra:
            las[n] = data["intensity"].astype(np.float32) if n == "intensity_raw" else cols[name]
        return las

    def _finish(self):
        self.w.close()
        notes = list(self.notes)
        for low in sorted(self._skipped):
            notes.insert(0, f"field '{low}' did not fit LAS's '{low}' and was left out")
        return notes

    def _release(self):
        self.w.close()


class _COPCWriter(_LASWriter):
    """LAZ with the points in octree order and an index: see :mod:`.copc`."""

    def __init__(self, path, tmp, dtype, stats):
        self._stats = stats
        super().__init__(path, tmp, dtype, stats, compress=True)

    def _open(self, tmp, compress):
        from .copc import CopcSpool

        st = self._stats
        lo = np.array([st[k].lo for k in "xyz"])
        hi = np.array([st[k].hi for k in "xyz"])
        self.w = None
        self.spool = CopcSpool(tmp, self.header, st["_count"], lo, hi)

    def write(self, data):
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
        self.spool.add(self.records(data), xyz)

    def _finish(self):
        self.spool.finish(self.tmp)
        notes = list(self.notes)
        for low in sorted(self._skipped):
            notes.insert(0, f"field '{low}' did not fit LAS's '{low}' and was left out")
        return notes

    def _release(self):
        self.spool.remove_spool()


# --------------------------------------------------------------------------
# E57


class _E57Writer(CloudWriter):
    CHUNK = 1_000_000

    def __init__(self, path, tmp, dtype, stats):
        import uuid

        import pye57
        from pye57 import libe57

        super().__init__(path, tmp)
        if not stats["_count"]:
            raise ValueError("E57 files need at least one point")
        names = list(dtype.names)
        one_d = [n for n in names if dtype[n].ndim == 0]
        self.cols: dict[str, tuple] = {}  # e57 name -> (numpy type, how to get it)
        stored = {"x", "y", "z"}
        for ax, n in zip("XYZ", "xyz"):
            self.cols[f"cartesian{ax}"] = ("f8", n)
        if "intensity" in one_d:
            self.cols["intensity"] = ("f8", "intensity")
            stored.add("intensity")
        rgb = next((c for c in ("rgb", "rgba") if c in names), None)
        if rgb:
            for k, c in enumerate(("Red", "Green", "Blue")):
                self.cols[f"color{c}"] = ("u1", (rgb, k))
            stored.add(rgb)
        elif all(c in names for c in ("red", "green", "blue")):
            for e57n, n_ in zip(("colorRed", "colorGreen", "colorBlue"), ("red", "green", "blue")):
                self.cols[e57n] = ("u1", n_)
                stored.add(n_)
        tf = _time_field(names)
        if tf and tf in one_d:
            self.cols["timeStamp"] = ("f8", tf)
            stored.add(tf)
        left_out = [f for f in names if f not in stored]
        self.notes = (["E57 has no place for these fields, so they were left out: "
                       + ", ".join(left_out) + " (save as PCD, PLY or LAS to keep them)"]
                      if left_out else [])

        self.e57 = pye57.E57(tmp, mode="w")
        img = self.e57.image_file
        scan = libe57.StructureNode(img)
        scan.set("guid", libe57.StringNode(img, "{%s}" % uuid.uuid4()))
        scan.set("name", libe57.StringNode(img, "Merged"))
        scan.set("description", libe57.StringNode(img, "Merged by PCD Merge"))

        lo = np.array([stats[k].lo for k in "xyz"])
        hi = np.array([stats[k].hi for k in "xyz"])
        bounds = libe57.StructureNode(img)
        for k, ax in enumerate("xyz"):
            bounds.set(f"{ax}Minimum", libe57.FloatNode(img, float(lo[k])))
            bounds.set(f"{ax}Maximum", libe57.FloatNode(img, float(hi[k])))
        scan.set("cartesianBounds", bounds)

        proto = libe57.StructureNode(img)
        for k, ax in enumerate("XYZ"):
            proto.set(f"cartesian{ax}", libe57.FloatNode(
                img, float(lo[k]), libe57.E57_DOUBLE, float(lo[k]), float(hi[k])))
        if "intensity" in self.cols:
            st = stats["intensity"]
            imin, imax = (st.lo, st.hi) if st.lo <= st.hi else (0.0, 0.0)
            lim = libe57.StructureNode(img)
            lim.set("intensityMinimum", libe57.FloatNode(img, imin))
            lim.set("intensityMaximum", libe57.FloatNode(img, imax))
            scan.set("intensityLimits", lim)
            proto.set("intensity", libe57.FloatNode(img, imin, libe57.E57_SINGLE, imin, imax))
        if "colorRed" in self.cols:
            lim = libe57.StructureNode(img)
            for c in ("Red", "Green", "Blue"):
                lim.set(f"color{c}Minimum", libe57.IntegerNode(img, 0))
                lim.set(f"color{c}Maximum", libe57.IntegerNode(img, 255))
                proto.set(f"color{c}", libe57.IntegerNode(img, 0, 0, 255))
            scan.set("colorLimits", lim)
        if "timeStamp" in self.cols:
            st = stats[tf]
            proto.set("timeStamp", libe57.FloatNode(
                img, float(st.lo), libe57.E57_DOUBLE, float(st.lo), float(st.hi)))

        points = libe57.CompressedVectorNode(img, proto, libe57.VectorNode(img, True))
        scan.set("points", points)
        self.e57.data3d.append(scan)

        bufs = libe57.VectorSourceDestBuffer()
        self.arrays = {}
        for name, (t, _) in self.cols.items():
            arr = np.empty(self.CHUNK, dtype=t)
            self.arrays[name] = arr
            bufs.append(libe57.SourceDestBuffer(img, name, arr, self.CHUNK, True, True))
        self.writer = points.writer(bufs)

    def write(self, data):
        rgb_cache = {}
        for start in range(0, len(data), self.CHUNK):
            part = data[start:start + self.CHUNK]
            m = len(part)
            for name, (t, how) in self.cols.items():
                if isinstance(how, tuple):  # one channel of packed colour
                    if how[0] not in rgb_cache:
                        rgb_cache[how[0]] = _unpack_rgb(part[how[0]])
                    v = rgb_cache[how[0]][how[1]]
                elif t == "u1":
                    v = np.clip(part[how], 0, 255)
                else:
                    v = part[how]
                self.arrays[name][:m] = v
            rgb_cache.clear()
            self.writer.write(m)

    def _finish(self):
        self.writer.close()
        self.e57.close()
        return self.notes

    def _release(self):
        try:
            self.writer.close()
        finally:
            self.e57.close()
