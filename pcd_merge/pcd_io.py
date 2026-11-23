"""Reading and writing PCD files without losing any per-point fields.

Open3D's own PCD reader keeps only xyz, colours and normals, which would drop
LiDAR attributes such as ``intensity``, ``ring`` or ``timestamp``. This module
parses PCD files (ascii, binary and binary_compressed) into a numpy structured
array so that every field survives the round trip into the merged file.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

try:  # C implementation of LZF, much faster for binary_compressed files.
    import lzf as _lzf
except ImportError:  # pragma: no cover - exercised only without python-lzf
    _lzf = None


_TYPE_MAP = {
    ("F", 4): "f4",
    ("F", 8): "f8",
    ("I", 1): "i1",
    ("I", 2): "i2",
    ("I", 4): "i4",
    ("I", 8): "i8",
    ("U", 1): "u1",
    ("U", 2): "u2",
    ("U", 4): "u4",
    ("U", 8): "u8",
}
_KIND_TO_PCD = {"f": "F", "i": "I", "u": "U"}


# Optional progress callback: ``progress(fraction)`` with fraction in 0..1.
FracFn = Callable[[float], None] | None

# Files are read and written in pieces of this size so that progress can be
# reported and no second full-size copy of the data is needed.
_CHUNK_BYTES = 64 * 1024 * 1024


class PCDError(ValueError):
    """Raised when a file is not a PCD file this module can read."""


@dataclass
class PointCloud:
    """A point cloud with all of its PCD fields.

    ``data`` is a structured numpy array with one record per point. Fields
    with COUNT > 1 are stored as sub-arrays.
    """

    data: np.ndarray
    path: str | None = None
    viewpoint: tuple[float, ...] = (0, 0, 0, 1, 0, 0, 0)
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path) if self.path else "cloud"

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.data.dtype.names

    def __len__(self) -> int:
        return len(self.data)

    @property
    def xyz(self) -> np.ndarray:
        """Point coordinates as an (N, 3) float64 array."""
        d = self.data
        out = np.empty((len(d), 3), dtype=np.float64)
        out[:, 0], out[:, 1], out[:, 2] = d["x"], d["y"], d["z"]
        return out

    def finite_xyz(self) -> np.ndarray:
        """Like ``xyz`` but without points that have NaN or infinite coordinates."""
        xyz = self.xyz
        ok = np.isfinite(xyz).all(axis=1)
        return xyz if ok.all() else xyz[ok]

    def has_field(self, name: str) -> bool:
        return name in self.data.dtype.names

    def finite_mask(self) -> np.ndarray:
        d = self.data
        return np.isfinite(d["x"]) & np.isfinite(d["y"]) & np.isfinite(d["z"])


# --------------------------------------------------------------------------
# LZF


def lzf_decompress(data: bytes, out_len: int) -> bytes:
    """Decompress LZF data (as written by PCL for binary_compressed)."""
    if _lzf is not None:
        out = _lzf.decompress(data, out_len + 1)
        if out is None or len(out) != out_len:
            raise PCDError("LZF decompression failed (corrupt file?)")
        return out
    return _lzf_decompress_py(data, out_len)


def _lzf_decompress_py(data: bytes, out_len: int) -> bytes:
    """Pure-python LZF decoder. Slow, used only if python-lzf is missing."""
    out = bytearray(out_len)
    ip = op = 0
    n = len(data)
    while ip < n:
        ctrl = data[ip]
        ip += 1
        if ctrl < 32:  # literal run
            ctrl += 1
            out[op : op + ctrl] = data[ip : ip + ctrl]
            ip += ctrl
            op += ctrl
        else:  # back reference
            length = ctrl >> 5
            ref = op - ((ctrl & 0x1F) << 8) - 1
            if length == 7:
                length += data[ip]
                ip += 1
            ref -= data[ip]
            ip += 1
            length += 2
            if ref < 0:
                raise PCDError("LZF decompression failed (corrupt file?)")
            if ref + length <= op:
                out[op : op + length] = out[ref : ref + length]
            else:  # overlapping copy must go byte by byte
                for k in range(length):
                    out[op + k] = out[ref + k]
            op += length
    if op != out_len:
        raise PCDError("LZF decompression failed (size mismatch)")
    return bytes(out)


# --------------------------------------------------------------------------
# Reading


def _parse_header(f) -> tuple[dict, int]:
    header: dict[str, list[str]] = {}
    while True:
        raw = f.readline()
        if not raw:
            raise PCDError("unexpected end of file in PCD header")
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            return header, f.tell()


def _build_dtype(header: dict) -> tuple[np.dtype, list[tuple[str, str, int, int]]]:
    names = header.get("FIELDS")
    if not names:
        raise PCDError("PCD header has no FIELDS line")
    n = len(names)
    sizes = [int(s) for s in header.get("SIZE", ["4"] * n)]
    types = [t.upper() for t in header.get("TYPE", ["F"] * n)]
    counts = [int(c) for c in header.get("COUNT", ["1"] * n)]
    if not (len(sizes) == len(types) == len(counts) == n):
        raise PCDError("PCD header FIELDS/SIZE/TYPE/COUNT lengths differ")

    specs = []
    dtype_fields = []
    seen: dict[str, int] = {}
    for name, size, typ, count in zip(names, sizes, types, counts):
        key = (typ, size)
        if key not in _TYPE_MAP:
            raise PCDError(f"unsupported PCD field type {typ}{size} for '{name}'")
        # PCL pads with fields named "_"; numpy needs unique names.
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        np_type = "<" + _TYPE_MAP[key]
        dtype_fields.append((name, np_type) if count == 1 else (name, np_type, (count,)))
        specs.append((name, np_type, count, size))
    return np.dtype(dtype_fields), specs


def read_pcd(path: str, progress: FracFn = None) -> PointCloud:
    """Read a PCD file, keeping every field.

    ``progress`` is called with the fraction read so far (0..1).
    """
    report = progress or (lambda _f: None)
    with open(path, "rb") as f:
        header, offset = _parse_header(f)

        if "FIELDS" not in header:
            raise PCDError(f"{path} does not look like a PCD file")
        dtype, specs = _build_dtype(header)
        for axis in ("x", "y", "z"):
            if axis not in dtype.names:
                raise PCDError(f"{os.path.basename(path)} has no '{axis}' field")

        if "POINTS" in header:
            npoints = int(header["POINTS"][0])
        else:
            npoints = int(header.get("WIDTH", ["0"])[0]) * int(header.get("HEIGHT", ["1"])[0])
        mode = header["DATA"][0].lower()

        if mode == "binary":
            # Read straight into the final array, in chunks for progress.
            data = np.empty(npoints, dtype=dtype)
            buf = memoryview(data.view(np.uint8).reshape(-1))
            need, pos = len(buf), 0
            while pos < need:
                got = f.readinto(buf[pos : pos + _CHUNK_BYTES])
                if not got:
                    raise PCDError(f"{os.path.basename(path)} is truncated")
                pos += got
                report(pos / need)
        elif mode == "ascii":
            data = _read_ascii(f.read(), dtype, specs, npoints, report)
        elif mode == "binary_compressed":
            data = _read_compressed(f.read(), dtype, specs, npoints)
        else:
            raise PCDError(f"unsupported PCD DATA type '{mode}'")
    report(1.0)

    viewpoint = tuple(float(v) for v in header.get("VIEWPOINT", [0, 0, 0, 1, 0, 0, 0]))
    return PointCloud(data=data, path=path, viewpoint=viewpoint)


def _parse_ascii_numbers(body: bytes, limit: int, report) -> np.ndarray:
    """Parse whitespace-separated numbers in chunks (fast, low memory)."""
    parts, total, start, n = [], 0, 0, len(body)
    while start < n and total < limit:
        end = min(n, start + _CHUNK_BYTES)
        if end < n:  # cut at a line break so no number is split
            nl = body.rfind(b"\n", start, end)
            end = nl + 1 if nl > start else n
        text = body[start:end].decode("ascii", errors="replace")
        try:
            vals = np.fromstring(text, dtype=np.float64, sep=" ")
        except (ValueError, DeprecationWarning):  # odd tokens; fall back to split
            vals = np.array(text.split(), dtype=np.float64)
        parts.append(vals)
        total += vals.size
        start = end
        report(min(1.0, start / n))
    values = np.concatenate(parts) if parts else np.zeros(0)
    return values[:limit]


def _read_ascii(body: bytes, dtype: np.dtype, specs, npoints: int, report=None) -> np.ndarray:
    ncols = sum(s[2] for s in specs)
    if npoints == 0:
        return np.zeros(0, dtype=dtype)
    try:
        values = _parse_ascii_numbers(body, npoints * ncols, report or (lambda _f: None))
    except ValueError as e:
        raise PCDError(f"ASCII PCD has a value that is not a number ({e})") from None
    if values.size < npoints * ncols:
        raise PCDError("ASCII PCD has fewer values than the header says")
    values = values.reshape(npoints, ncols)
    out = np.zeros(npoints, dtype=dtype)
    col = 0
    for name, np_type, count, _size in specs:
        chunk = values[:, col : col + count]
        if name in ("rgb", "rgba") and count == 1:
            # packed colours are written as integers or floats; keep bit pattern
            out[name] = _ascii_packed_colour(chunk[:, 0], np_type)
        else:
            out[name] = chunk[:, 0] if count == 1 else chunk
        col += count
    return out


def _ascii_packed_colour(values: np.ndarray, np_type: str) -> np.ndarray:
    if np_type[1] != "f":
        return values
    if np.all(np.isfinite(values)) and np.all(values == np.round(values)) and values.max(initial=0) > 1:
        # stored as the integer value of the packed colour
        return values.astype(np.uint32).view(np.float32)
    return values.astype(np.float32)


def _read_compressed(body: bytes, dtype: np.dtype, specs, npoints: int) -> np.ndarray:
    if len(body) < 8:
        raise PCDError("binary_compressed PCD is truncated")
    comp_size, raw_size = struct.unpack("<II", body[:8])
    raw = lzf_decompress(body[8 : 8 + comp_size], raw_size)
    out = np.zeros(npoints, dtype=dtype)
    # Data is stored field by field (column-major).
    pos = 0
    for name, np_type, count, size in specs:
        nbytes = npoints * count * size
        col = np.frombuffer(raw, dtype=np_type, count=npoints * count, offset=pos)
        out[name] = col if count == 1 else col.reshape(npoints, count)
        pos += nbytes
    return out


# --------------------------------------------------------------------------
# Writing


def write_pcd(path: str, cloud: PointCloud | np.ndarray, binary: bool = True,
              compressed: bool = False, progress: FracFn = None) -> None:
    """Write a structured array (or PointCloud) as an unorganised PCD file.

    ``compressed=True`` writes binary_compressed (needs python-lzf).
    ``progress`` is called with the fraction written so far (0..1).
    """
    report = progress or (lambda _f: None)
    data = cloud.data if isinstance(cloud, PointCloud) else cloud
    names, sizes, types, counts = [], [], [], []
    for name in data.dtype.names:
        base, shape = data.dtype.fields[name][0].base, data.dtype.fields[name][0].shape
        names.append(name)
        sizes.append(str(base.itemsize))
        types.append(_KIND_TO_PCD[base.kind])
        counts.append(str(int(np.prod(shape)) if shape else 1))
    n = len(data)
    if compressed and _lzf is None:
        raise RuntimeError("binary_compressed output needs the python-lzf package")
    mode = "binary_compressed" if compressed else ("binary" if binary else "ascii")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(names)}\n"
        f"SIZE {' '.join(sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(counts)}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        f"DATA {mode}\n"
    )
    # make sure we write little-endian, packed records
    le_dtype = data.dtype.newbyteorder("<")
    packed = np.ascontiguousarray(data.astype(le_dtype, copy=False))
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        if mode == "binary":
            # write in chunks from the array's own memory (no full copy)
            raw = memoryview(packed.view(np.uint8).reshape(-1)) if n else memoryview(b"")
            for pos in range(0, len(raw), _CHUNK_BYTES):
                f.write(raw[pos : pos + _CHUNK_BYTES])
                report(min(1.0, (pos + _CHUNK_BYTES) / len(raw)))
        elif mode == "binary_compressed":
            cols = b"".join(np.ascontiguousarray(packed[name]).tobytes() for name in names)
            comp = _lzf.compress(cols) if cols else b""
            if comp is None:  # incompressible; lzf refuses to grow the data
                comp = _lzf.compress(cols, len(cols) * 2 + 16)
            f.write(struct.pack("<II", len(comp), len(cols)))
            f.write(comp)
        else:
            _write_ascii(f, packed, names)
    os.replace(tmp, path)
    report(1.0)


def _write_ascii(f, data: np.ndarray, names) -> None:
    cols = []
    for name in names:
        col = data[name]
        col = col.reshape(len(data), -1)
        cols.append(col)
    for i in range(len(data)):
        parts = []
        for col in cols:
            for v in col[i]:
                parts.append(repr(float(v)) if col.dtype.kind == "f" else str(int(v)))
        f.write((" ".join(parts) + "\n").encode("ascii"))
