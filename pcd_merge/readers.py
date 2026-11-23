"""Reading scans in every input format: PCD, PLY, LAS, LAZ and E57.

Every format is read into the same shape as :mod:`pcd_io` produces for PCD:
a numpy structured array with ``x``, ``y``, ``z`` and whatever else the file
has (``intensity``, packed ``rgb``, ``gps_time``, ...), so alignment, merging,
clean-up and export work the same whatever the scanner software wrote.

Big sites do not fit in memory, so every scan is a :class:`ScanSource`:

* ``source.count``, ``source.dtype``, ``source.bounds`` and
  ``source.viewpoint`` come from the file header where the format has them,
  without reading the points;
* ``source.chunks()`` yields the points a few million at a time, so a scan can
  be streamed through (merged, exported, sampled) without ever holding it all;
* ``source.read()`` loads the whole scan as a :class:`~pcd_io.PointCloud`, as
  alignment needs today.

One E57 file can hold several scans (one per scanner position). Each becomes
its own source, told apart by ``index``.

Colour is always returned as a PCL-style packed ``rgb`` float field (8 bits per
channel), whatever the file stored, so the rest of the program only deals with
one colour layout.
"""

from __future__ import annotations

import os
from typing import Callable, Iterator

import numpy as np

from .pcd_io import (PCDError, PointCloud, _build_dtype, _parse_header, read_pcd)

# file extension -> label, for "open" dialogs and folder scanning
INPUT_FORMATS = {
    ".pcd": "PCD point cloud",
    ".ply": "PLY point cloud",
    ".las": "LAS",
    ".laz": "LAZ (compressed LAS)",
    ".e57": "E57 (scanner exchange)",
}

CHUNK_POINTS = 2_000_000  # points per chunk when streaming

FracFn = Callable[[float], None] | None


class ScanReadError(ValueError):
    """A file that is not a scan this program can read."""


def is_scan_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in INPUT_FORMATS


def input_filter() -> str:
    """Qt open-dialog filter: all supported scans first, then each format."""
    pattern = " ".join(f"*{e}" for e in INPUT_FORMATS)
    each = ";;".join(f"{label} (*{ext})" for ext, label in INPUT_FORMATS.items())
    return f"Point clouds ({pattern});;{each};;All files (*)"


def pack_rgb(r, g, b) -> np.ndarray:
    """8-bit channels -> PCL packed colour (float32 holding the bit pattern)."""
    v = (np.asarray(r, np.uint32) << 16) | (np.asarray(g, np.uint32) << 8) | np.asarray(b, np.uint32)
    return v.astype(np.uint32).view(np.float32)


def _to_u8(values: np.ndarray, sixteen_bit: bool) -> np.ndarray:
    v = np.asarray(values)
    if sixteen_bit:
        return (v.astype(np.uint32) >> 8).astype(np.uint8)
    return np.clip(v, 0, 255).astype(np.uint8)


def _quat_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------
# Sources


class ScanSource:
    """One scan in a file: header facts now, points on demand."""

    path: str
    index: int = 0  # which scan inside the file (E57 files can hold several)
    label: str | None = None  # name inside the file, if it has one
    count: int = 0  # points in the file (may include invalid ones)
    dtype: np.dtype
    viewpoint: tuple = (0, 0, 0, 1, 0, 0, 0)
    bounds: tuple[np.ndarray, np.ndarray] | None = None  # (lo, hi) if the header has it
    scans_in_file: int = 1

    @property
    def name(self) -> str:
        base = os.path.basename(self.path)
        if self.scans_in_file > 1:
            inner = f": {self.label}" if self.label else ""
            return f"{base} [{self.index + 1}/{self.scans_in_file}{inner}]"
        return base

    @property
    def key(self) -> str:
        """Identifies this scan: absolute path, plus ``#index`` inside multi-scan files."""
        p = os.path.normcase(os.path.abspath(self.path))
        return f"{p}#{self.index}" if self.scans_in_file > 1 else p

    def chunks(self, size: int = CHUNK_POINTS) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def read(self, progress: FracFn = None) -> PointCloud:
        """Load every point. ``progress(fraction)`` while reading."""
        report = progress or (lambda _f: None)
        out = np.empty(self.count, dtype=self.dtype)
        pos = 0
        for chunk in self.chunks():
            n = len(chunk)
            if pos + n > len(out):  # header undercounted; grow
                out = np.concatenate([out[:pos], np.empty(pos + n - len(out) + n, self.dtype)])
            out[pos:pos + n] = chunk
            pos += n
            report(pos / max(self.count, 1))
        report(1.0)
        return self._cloud(out[:pos] if pos != len(out) else out)

    def _cloud(self, data: np.ndarray) -> PointCloud:
        extra = {"format": os.path.splitext(self.path)[1].lower()}
        if self.scans_in_file > 1:
            extra.update(scan_index=self.index, scans_in_file=self.scans_in_file, name=self.name)
        return PointCloud(data=data, path=self.path, viewpoint=tuple(self.viewpoint), extra=extra)


class MemorySource(ScanSource):
    """A cloud already in memory, streamed like a file (for merging/export)."""

    def __init__(self, cloud: PointCloud):
        self.cloud = cloud
        self.path = cloud.path or cloud.name
        self.index = int(cloud.extra.get("scan_index", 0))
        self.scans_in_file = int(cloud.extra.get("scans_in_file", 1))
        self.label = None
        self.count = len(cloud)
        self.dtype = cloud.data.dtype
        self.viewpoint = cloud.viewpoint
        self.bounds = None

    @property
    def name(self) -> str:
        return self.cloud.name

    def chunks(self, size: int = CHUNK_POINTS):
        d = self.cloud.data
        for s in range(0, len(d), size):
            yield d[s:s + size]

    def read(self, progress: FracFn = None) -> PointCloud:
        return self.cloud


class PCDSource(ScanSource):
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            header, self._offset = _parse_header(f)
        if "FIELDS" not in header:
            raise ScanReadError(f"{os.path.basename(path)} does not look like a PCD file")
        self.dtype, _ = _build_dtype(header)
        for axis in "xyz":
            if axis not in self.dtype.names:
                raise ScanReadError(f"{os.path.basename(path)} has no '{axis}' field")
        if "POINTS" in header:
            self.count = int(header["POINTS"][0])
        else:
            self.count = int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0])
        self._mode = header["DATA"][0].lower()
        self.viewpoint = tuple(float(v) for v in header.get("VIEWPOINT", [0, 0, 0, 1, 0, 0, 0]))

    def chunks(self, size: int = CHUNK_POINTS):
        if self._mode != "binary":  # ascii / compressed: decode all, then slice
            data = read_pcd(self.path).data
            for s in range(0, len(data), size):
                yield data[s:s + size]
            return
        rec = self.dtype.itemsize
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            left = self.count
            while left > 0:
                n = min(size, left)
                raw = f.read(n * rec)
                if len(raw) < n * rec:
                    raise PCDError(f"{os.path.basename(self.path)} is truncated")
                yield np.frombuffer(raw, dtype=self.dtype, count=n).copy()
                left -= n

    def read(self, progress: FracFn = None) -> PointCloud:
        return read_pcd(self.path, progress=progress)


# ---- PLY

_PLY_NP = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
           "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
           "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
           "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_PLY_RENAME = {"scalar_intensity": "intensity", "scalar_Intensity": "intensity",
               "Intensity": "intensity", "reflectance": "intensity"}


class PLYSource(ScanSource):
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            if f.readline().strip() != b"ply":
                raise ScanReadError(f"{os.path.basename(path)} is not a PLY file")
            fmt = None
            elements: list[tuple[str, int, list]] = []
            while True:
                raw = f.readline()
                if not raw:
                    raise ScanReadError("unexpected end of PLY header")
                parts = raw.decode("ascii", errors="replace").split()
                if not parts or parts[0] in ("comment", "obj_info"):
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    elements.append((parts[1], int(parts[2]), []))
                elif parts[0] == "property" and elements:
                    elements[-1][2].append(parts[1:])
                elif parts[0] == "end_header":
                    break
            self._offset = f.tell()
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ScanReadError(f"unsupported PLY format '{fmt}'")
        self._fmt = fmt
        end = ">" if fmt == "binary_big_endian" else "<"
        skip = 0  # bytes of elements stored before the vertices
        self._skip_rows = 0  # ascii: lines before the vertices
        vertex = None
        for name, n, props in elements:
            if name == "vertex":
                vertex = (n, props)
                break
            if any(p[0] == "list" for p in props):
                raise ScanReadError("PLY with list properties before the vertices is not supported")
            skip += n * sum(np.dtype(_PLY_NP[p[0]]).itemsize for p in props)
            self._skip_rows += n
        if vertex is None:
            raise ScanReadError(f"{os.path.basename(path)} has no vertices")
        n, props = vertex
        if any(p[0] == "list" for p in props):
            raise ScanReadError("PLY vertices with list properties are not supported")
        self._offset += skip if fmt != "ascii" else 0
        self.count = n
        self._file_dtype = np.dtype([(p[1], end + _PLY_NP[p[0]]) for p in props])
        names = [p[1] for p in props]
        for axis in "xyz":
            if axis not in names:
                raise ScanReadError(f"{os.path.basename(path)} has no '{axis}' property")
        self._has_rgb = all(c in names for c in ("red", "green", "blue"))
        self._rgb16 = self._has_rgb and self._file_dtype["red"].itemsize > 1
        seen: set[str] = set()
        fields, self._map = [], []  # output fields; (file name, our name) copied as is
        for p in props:
            src = p[1]
            if self._has_rgb and src in ("red", "green", "blue", "alpha"):
                if src == "red":
                    fields.append(("rgb", "<f4"))
                    seen.add("rgb")
                continue
            name = base = _PLY_RENAME.get(src, src)
            k = 1
            while name in seen:  # renames must not collide
                name, k = f"{base}_{k}", k + 1
            seen.add(name)
            fields.append((name, "<" + _PLY_NP[p[0]]))
            self._map.append((src, name))
        self.dtype = np.dtype(fields)

    def _convert(self, raw: np.ndarray) -> np.ndarray:
        out = np.empty(len(raw), dtype=self.dtype)
        for src, dst in self._map:
            out[dst] = raw[src]
        if self._has_rgb:
            out["rgb"] = pack_rgb(*(_to_u8(raw[c], self._rgb16) for c in ("red", "green", "blue")))
        return out

    def chunks(self, size: int = CHUNK_POINTS):
        if self._fmt == "ascii":
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                for _ in range(self._skip_rows):
                    f.readline()
                ncol = len(self._file_dtype.names)
                left = self.count
                while left > 0:
                    n = min(size, left)
                    lines = [f.readline() for _ in range(n)]
                    vals = np.array(b" ".join(lines).split(), dtype=np.float64)
                    if vals.size < n * ncol:
                        raise ScanReadError(f"{os.path.basename(self.path)} is truncated")
                    vals = vals[:n * ncol].reshape(n, ncol)
                    raw = np.empty(n, dtype=self._file_dtype)
                    for k, name in enumerate(self._file_dtype.names):
                        raw[name] = vals[:, k]
                    yield self._convert(raw)
                    left -= n
            return
        rec = self._file_dtype.itemsize
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            left = self.count
            while left > 0:
                n = min(size, left)
                buf = f.read(n * rec)
                if len(buf) < n * rec:
                    raise ScanReadError(f"{os.path.basename(self.path)} is truncated")
                yield self._convert(np.frombuffer(buf, dtype=self._file_dtype, count=n))
                left -= n


# ---- LAS / LAZ

class LASSource(ScanSource):
    """LAS 1.0-1.4 and LAZ (via laspy + lazrs)."""

    def __init__(self, path: str):
        import laspy

        self.path = path
        try:
            with laspy.open(path) as r:
                h = r.header
                self.count = int(h.point_count)
                fmt = h.point_format
                lo, hi = np.asarray(h.mins, float), np.asarray(h.maxs, float)
                self._dims = list(fmt.dimension_names)
                self._extra = list(fmt.extra_dimension_names)
                self._extra_types = {d.name: d.dtype for d in fmt.dimensions
                                     if d.name in self._extra}
        except (laspy.errors.LaspyException, OSError, ValueError) as e:
            raise ScanReadError(f"{os.path.basename(path)}: {e}") from None
        if self.count and np.all(np.isfinite(lo)) and np.all(hi >= lo):
            self.bounds = (lo, hi)
        dims = set(self._dims)
        fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
        self._plain = []  # (laspy name, our name)
        self._restore_intensity = "intensity_raw" in self._extra
        if "intensity" in dims:
            fields.append(("intensity", "<f4" if self._restore_intensity else "<u2"))
        self._has_rgb = {"red", "green", "blue"} <= dims
        if self._has_rgb:
            fields.append(("rgb", "<f4"))
        for d, t in (("gps_time", "<f8"), ("return_number", "u1"), ("number_of_returns", "u1"),
                     ("classification", "u1"), ("user_data", "u1"), ("point_source_id", "<u2")):
            if d in dims:
                fields.append((d, t))
                self._plain.append((d, d))
        for e in self._extra:
            if e == "intensity_raw":
                continue
            ours = e[:-4] if e.endswith("_pcd") else e
            if ours in {f[0] for f in fields}:
                ours = e
            t = np.dtype(self._extra_types.get(e, np.float64))
            if t.shape:
                fields.append((ours, t.base.newbyteorder("<"), t.shape))
            else:
                fields.append((ours, t.newbyteorder("<")))
            self._plain.append((e, ours))
        self.dtype = np.dtype(fields)

    def chunks(self, size: int = CHUNK_POINTS):
        import laspy

        with laspy.open(self.path) as r:
            rgb16 = None
            for pts in r.chunk_iterator(size):
                n = len(pts)
                out = np.empty(n, dtype=self.dtype)
                out["x"], out["y"], out["z"] = pts.x, pts.y, pts.z
                if "intensity" in self.dtype.names:
                    out["intensity"] = pts["intensity_raw"] if self._restore_intensity \
                        else pts.intensity
                if self._has_rgb:
                    r_, g_, b_ = (np.asarray(pts[c]) for c in ("red", "green", "blue"))
                    if rgb16 is None and n:
                        # LAS says 16-bit, but some writers store 0-255.
                        rgb16 = max(int(r_.max()), int(g_.max()), int(b_.max())) > 255
                    out["rgb"] = pack_rgb(*(_to_u8(c, bool(rgb16)) for c in (r_, g_, b_)))
                for src, dst in self._plain:
                    out[dst] = np.asarray(pts[src])
                yield out


# ---- E57

_E57_FIELDS = {  # E57 name -> (our name, numpy type)
    "intensity": ("intensity", "<f4"),
    "timeStamp": ("timestamp", "<f8"),
    "rowIndex": ("row", "<u4"),
    "columnIndex": ("column", "<u4"),
    "returnIndex": ("return_number", "u1"),
    "returnCount": ("number_of_returns", "u1"),
}


class E57Source(ScanSource):
    """One scan (``data3D`` entry) of an E57 file, with its pose applied."""

    def __init__(self, path: str, index: int = 0, _e57=None):
        import pye57

        self.path, self.index = path, index
        own = _e57 is None
        try:
            e57 = _e57 or pye57.E57(path)
        except Exception as e:  # pye57 raises plain exceptions for bad files
            raise ScanReadError(f"{os.path.basename(path)}: {e}") from None
        try:
            self.scans_in_file = e57.scan_count
            if not 0 <= index < self.scans_in_file:
                raise ScanReadError(f"{os.path.basename(path)} has no scan {index + 1}")
            h = e57.get_header(index)
            fields = set(h.point_fields)
            self.count = int(h.point_count)
            try:
                self.label = str(h.node["name"].value()) or None
            except Exception:
                self.label = None
            if {"cartesianX", "cartesianY", "cartesianZ"} <= fields:
                self._coords = ("cartesianX", "cartesianY", "cartesianZ")
                self._invalid = "cartesianInvalidState" if "cartesianInvalidState" in fields else None
            elif {"sphericalRange", "sphericalAzimuth", "sphericalElevation"} <= fields:
                self._coords = ("sphericalRange", "sphericalAzimuth", "sphericalElevation")
                self._invalid = "sphericalInvalidState" if "sphericalInvalidState" in fields else None
            else:
                raise ScanReadError(f"{self.name} has no coordinates")
            self._rot, self._tr = np.eye(3), np.zeros(3)
            if h.has_pose():
                try:
                    q = [float(e.value()) for e in h.node["pose"]["rotation"]]
                    self._rot = _quat_matrix(*q)
                except Exception:
                    pass
                try:
                    self._tr = np.array([float(e.value()) for e in h.node["pose"]["translation"]])
                except Exception:
                    pass
            # The scanner stood at the pose's origin: that is the PCD VIEWPOINT.
            self.viewpoint = (*self._tr.tolist(), 1, 0, 0, 0)
            self._rgb = [c for c in ("colorRed", "colorGreen", "colorBlue") if c in fields]
            self._rgb_max = 255
            if len(self._rgb) == 3:
                try:
                    lim = h.node["colorLimits"]
                    self._rgb_max = max(float(lim[f"color{c}Maximum"].value())
                                        for c in ("Red", "Green", "Blue"))
                except Exception:
                    pass
            else:
                self._rgb = []
            self._other = [f for f in _E57_FIELDS if f in fields]
            out = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
            out += [_E57_FIELDS[f] for f in self._other if f == "intensity"]
            if self._rgb:
                out.append(("rgb", "<f4"))
            out += [_E57_FIELDS[f] for f in self._other if f != "intensity"]
            self.dtype = np.dtype(out)
            if "cartesianBounds" in h.scan_fields and not h.has_pose():
                try:
                    b = h.node["cartesianBounds"]
                    lo = np.array([float(b[f"{a}Minimum"].value()) for a in "xyz"])
                    hi = np.array([float(b[f"{a}Maximum"].value()) for a in "xyz"])
                    if np.all(np.isfinite(lo)) and np.all(hi >= lo):
                        self.bounds = (lo, hi)
                except Exception:
                    pass
        finally:
            if own:
                e57.close()

    def chunks(self, size: int = CHUNK_POINTS):
        import pye57
        from pye57 import libe57

        e57 = pye57.E57(self.path)
        try:
            h = e57.get_header(self.index)
            img = e57.image_file
            names = list(self._coords) + self._rgb + self._other
            if self._invalid:
                names.append(self._invalid)
            size = max(1, min(size, self.count or 1))
            bufs = libe57.VectorSourceDestBuffer()
            arrays = {}
            for n in names:
                kind = "u2" if n in self._rgb else ("i1" if n == self._invalid else
                                                     ("f8" if n not in ("rowIndex", "columnIndex",
                                                                         "returnIndex", "returnCount")
                                                      else "i4"))
                arr = np.empty(size, dtype=kind)
                arrays[n] = arr
                bufs.append(libe57.SourceDestBuffer(img, n, arr, size, True, True))
            reader = h.points.reader(bufs)
            try:
                while True:
                    got = reader.read()
                    if not got:
                        break
                    yield self._convert({k: v[:got] for k, v in arrays.items()})
            finally:
                reader.close()
        finally:
            e57.close()

    def _convert(self, cols: dict) -> np.ndarray:
        ok = None
        if self._invalid:
            ok = cols[self._invalid] == 0
            if ok.all():
                ok = None
        a, b, c = (cols[n] if ok is None else cols[n][ok] for n in self._coords)
        if self._coords[0] == "sphericalRange":
            rc = a * np.cos(c)
            xyz = np.stack([rc * np.cos(b), rc * np.sin(b), a * np.sin(c)], 1)
        else:
            xyz = np.stack([a, b, c], 1)
        xyz = xyz @ self._rot.T + self._tr
        out = np.empty(len(xyz), dtype=self.dtype)
        out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        if self._rgb:
            r, g, b_ = (cols[n] if ok is None else cols[n][ok] for n in self._rgb)
            if self._rgb_max > 255:
                r, g, b_ = (v.astype(np.float64) * (255.0 / self._rgb_max) for v in (r, g, b_))
            out["rgb"] = pack_rgb(*(np.clip(v, 0, 255).astype(np.uint8) for v in (r, g, b_)))
        for f in self._other:
            v = cols[f] if ok is None else cols[f][ok]
            out[_E57_FIELDS[f][0]] = v
        return out


# --------------------------------------------------------------------------
# Opening files


def open_scans(path: str) -> list[ScanSource]:
    """Every scan in a file (several for multi-scan E57 files, else one)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        try:
            return [PCDSource(path)]
        except PCDError as e:
            raise ScanReadError(str(e)) from None
    if ext == ".ply":
        return [PLYSource(path)]
    if ext in (".las", ".laz"):
        return [LASSource(path)]
    if ext == ".e57":
        import pye57

        try:
            e57 = pye57.E57(path)
        except Exception as e:
            raise ScanReadError(f"{os.path.basename(path)}: {e}") from None
        try:
            return [E57Source(path, i, _e57=e57) for i in range(e57.scan_count)]
        finally:
            e57.close()
    raise ScanReadError(f"{os.path.basename(path)}: unsupported file type '{ext}' "
                        f"(use {', '.join(INPUT_FORMATS)})")


def open_scan(path: str, index: int = 0) -> ScanSource:
    if os.path.splitext(path)[1].lower() == ".e57":
        return E57Source(path, index)
    return open_scans(path)[0]


def read_cloud(path: str, index: int = 0, progress: FracFn = None) -> PointCloud:
    """Read one scan of any supported format into memory."""
    return open_scan(path, index).read(progress=progress)


def read_clouds(path: str, progress: FracFn = None) -> list[PointCloud]:
    """Every scan in a file, each as its own cloud."""
    sources = open_scans(path)
    total = max(1, sum(s.count for s in sources))
    out, done = [], 0
    report = progress or (lambda _f: None)
    for s in sources:
        out.append(s.read(progress=lambda f, d=done, n=s.count: report((d + f * n) / total)))
        done += s.count
    report(1.0)
    return out


def sample_points(source: ScanSource, n: int, seed: int = 0) -> np.ndarray:
    """A random subset of up to ``n`` finite points as (k, 3) float64, in random
    order, read in one streaming pass (the scan is never held in memory)."""
    rng = np.random.default_rng(seed)
    total = max(source.count, 1)
    keep_frac = min(1.0, n / total)
    parts = []
    for chunk in source.chunks():
        if keep_frac < 1.0:
            chunk = chunk[rng.random(len(chunk)) < keep_frac * 1.02]
        xyz = np.empty((len(chunk), 3), dtype=np.float64)
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = chunk["x"], chunk["y"], chunk["z"]
        parts.append(xyz[np.isfinite(xyz).all(axis=1)])
    xyz = np.concatenate(parts) if parts else np.zeros((0, 3))
    if len(xyz) > n:
        xyz = xyz[rng.choice(len(xyz), n, replace=False)]
    return xyz[rng.permutation(len(xyz))]


def find_scan_files(paths: list[str], exts=None) -> list[str]:
    """Like :func:`files.find_pcd_files`, for every input format."""
    from .files import find_files

    return find_files(paths, tuple(exts or INPUT_FORMATS))
