"""Write COPC files: Cloud Optimized Point Clouds (https://copc.io).

A COPC file is an ordinary LAZ 1.4 file whose points are stored in the
cubes of an octree, coarsest first, with an index of the cubes at the end.
A viewer reads the index and then only the cubes it needs for what is on
screen, at the level of detail it needs, so a building-sized site opens at
once and stays sharp close up. CloudCompare, QGIS, PDAL, Potree and web
viewers such as https://viewer.copc.io open it; any LAS software still
reads it as a normal LAZ file.

laspy reads COPC but does not write it, so the octree and the index are
built here, on top of laspy (point records) and lazrs (compression):

1. Every point is given a level of detail at random, so that each cube of
   the octree ends up with about :data:`NODE_POINTS` points (a random
   subset of a cube's points is an even sample of it). The level and the
   point's cube at that level make one sort key.
2. The points are converted to LAS records and spooled to a temporary file
   next to the output, with their keys, so the whole cloud is never held in
   memory as Python objects.
3. The keys are sorted; each cube's points are compressed as one LAZ chunk
   (LAZ "variable size chunks"), and the index (the COPC hierarchy) records
   where each cube's chunk starts.
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np

NODE_POINTS = 100_000  # aim for about this many points per cube
_COORD_BITS = 19  # per axis in the sort key: depth up to 19
_SEED = 20251123


def level_weights(count: int, node_points: int = NODE_POINTS) -> np.ndarray:
    """Share of the points to put at each level of detail.

    Scans are surfaces, so the cubes that hold points grow about four-fold
    per level; giving level ``l`` a share proportional to 4^l puts about the
    same number of points in every cube."""
    depth = 0
    while count / sum(4 ** k for k in range(depth + 1)) > node_points and depth < 16:
        depth += 1
    w = 4.0 ** np.arange(depth + 1)
    return w / w.sum()


def node_keys(xyz: np.ndarray, levels: np.ndarray, cube_min: np.ndarray,
              size: float) -> np.ndarray:
    """One uint64 per point: its level, then its cube's x, y, z at that level.
    Sorting by it puts the points cube by cube, coarsest level first."""
    lv = levels.astype(np.int64)
    cells = (1 << lv)[:, None]
    ijk = np.floor((xyz - cube_min) / size * cells).astype(np.int64)
    ijk = np.clip(ijk, 0, cells - 1)
    b = _COORD_BITS
    return ((lv.astype(np.uint64) << np.uint64(3 * b))
            | (ijk[:, 0].astype(np.uint64) << np.uint64(2 * b))
            | (ijk[:, 1].astype(np.uint64) << np.uint64(b))
            | ijk[:, 2].astype(np.uint64))


def unpack_key(key: int) -> tuple[int, int, int, int]:
    b, m = _COORD_BITS, (1 << _COORD_BITS) - 1
    return key >> (3 * b), (key >> (2 * b)) & m, (key >> b) & m, key & m


def copc_info_bytes(center, halfsize, spacing, root_offset, root_size, gps_min, gps_max):
    return (struct.pack("<3d", *map(float, center))
            + struct.pack("<dd", halfsize, spacing)
            + struct.pack("<QQ", root_offset, root_size)
            + struct.pack("<dd", gps_min, gps_max)
            + bytes(8 * 11))


def hierarchy_bytes(entries) -> bytes:
    """``entries``: (level, x, y, z, offset, byte size, point count)."""
    return b"".join(struct.pack("<4iQii", *e) for e in entries)


def _evlr_header(user_id: str, record_id: int, length: int, description: str) -> bytes:
    return (struct.pack("<H", 0) + user_id.encode().ljust(16, b"\0")
            + struct.pack("<HQ", record_id, length) + description.encode().ljust(32, b"\0"))


class CopcSpool:
    """Collects LAS point records with their octree keys, then writes the
    COPC file in :meth:`finish`. Used by the exporter's COPC writer."""

    def __init__(self, tmp_base: str, header, count: int, lo, hi, seed: int = _SEED):
        self.header = header
        self.tmp_base = tmp_base
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if not np.all(np.isfinite(lo)):
            lo = hi = np.zeros(3)
        self.lo, self.hi = lo, hi
        self.center = (lo + hi) / 2
        self.halfsize = float(max(np.max(hi - lo) / 2, 1e-3)) * 1.000001
        self.cube_min = self.center - self.halfsize
        self.weights = np.cumsum(level_weights(max(count, 1), NODE_POINTS))
        self.rng = np.random.default_rng(seed)
        self._rec = open(tmp_base + ".records", "wb")
        self._keys = open(tmp_base + ".keys", "wb")
        self.count = 0
        self.dtype = None
        self.gps = [np.inf, -np.inf]

    def add(self, records, xyz: np.ndarray) -> None:
        arr = records.array
        if self.dtype is None:
            self.dtype = arr.dtype
        n = len(arr)
        levels = np.minimum(np.searchsorted(self.weights, self.rng.random(n)),
                            len(self.weights) - 1)
        keys = node_keys(np.asarray(xyz, float), levels, self.cube_min, 2 * self.halfsize)
        self._rec.write(arr.tobytes())
        self._keys.write(keys.tobytes())
        if "gps_time" in arr.dtype.names and n:
            self.gps = [min(self.gps[0], float(arr["gps_time"].min())),
                        max(self.gps[1], float(arr["gps_time"].max()))]
        self.count += n

    def close_spool(self) -> None:
        for f in (self._rec, self._keys):
            if not f.closed:
                f.close()

    def remove_spool(self) -> None:
        self.close_spool()
        for ext in (".records", ".keys"):
            try:
                os.remove(self.tmp_base + ext)
            except OSError:
                pass

    def finish(self, path: str, progress=None) -> None:
        """Sort the spooled points into the octree and write ``path``."""
        import laspy
        import lazrs

        self.close_spool()
        h = self.header
        n = self.count
        if n:
            keys = np.memmap(self.tmp_base + ".keys", dtype=np.uint64, mode="r")
            recs = np.memmap(self.tmp_base + ".records", dtype=self.dtype, mode="r")
            order = np.argsort(keys, kind="stable")
            skeys = keys[order]
            starts = np.flatnonzero(np.r_[True, skeys[1:] != skeys[:-1]])
            ends = np.r_[starts[1:], n]
            node_ids = [int(k) for k in skeys[starts]]
            del skeys
        else:
            starts = ends = np.zeros(0, int)
            node_ids = []

        pf = h.point_format
        laz = lazrs.LazVlr.new_for_compression(pf.id, pf.num_extra_bytes, True)
        vlrs = list(h.vlrs)
        h.vlrs.clear()
        info = laspy.VLR("copc", 1, "copc info", copc_info_bytes(
            self.center, self.halfsize, 2 * self.halfsize / 128, 0, 0, 0.0, 0.0))
        h.vlrs.append(info)
        h.vlrs.append(laspy.VLR("laszip encoded", 22204, "lazrs", laz.record_data()))
        for v in vlrs:
            h.vlrs.append(v)
        h.set_compressed(True)
        h.global_encoding.wkt = True
        h.point_count = n
        h.mins, h.maxs = self.lo, self.hi
        h.start_of_first_evlr = 0
        h.number_of_evlrs = 1

        with open(path, "w+b") as f:
            h.write_to(f)
            data_start = f.tell()
            comp = lazrs.LasZipCompressor(f, laz)
            for k, (a, b) in enumerate(zip(starts, ends)):
                comp.compress_many(np.ascontiguousarray(recs[order[a:b]]).tobytes())
                comp.finish_current_chunk()
                if progress and k % 64 == 0:
                    progress(k / max(len(starts), 1))
            comp.done()
            f.flush()
            # where each cube's chunk landed, from the chunk table just written
            f.seek(data_start)
            table = lazrs.read_chunk_table(f, laz) if n else []
            offset = data_start + 8  # after the pointer to the chunk table
            found = {}
            for key, (a, b), (pts, nbytes) in zip(node_ids, zip(starts, ends), table):
                if pts != b - a:
                    raise RuntimeError("COPC: a cube's chunk does not match its points")
                found[unpack_key(key)] = (offset, int(nbytes), int(pts))
                offset += nbytes
            # every ancestor of a cube needs an entry, even with no points
            for (d, x, y, z) in list(found):
                while d > 0:
                    d, x, y, z = d - 1, x >> 1, y >> 1, z >> 1
                    found.setdefault((d, x, y, z), (0, 0, 0))
            if not found:
                found[(0, 0, 0, 0)] = (0, 0, 0)
            entries = [(*key, *found[key]) for key in sorted(found)]
            page = hierarchy_bytes(entries)
            f.seek(0, io.SEEK_END)
            evlr_start = f.tell()
            f.write(_evlr_header("copc", 1000, len(page), "EPT hierarchy"))
            f.write(page)
            # now the header knows where the index is
            gps = self.gps if np.isfinite(self.gps[0]) else [0.0, 0.0]
            info.record_data = copc_info_bytes(self.center, self.halfsize,
                                               2 * self.halfsize / 128, evlr_start + 60,
                                               len(page), *gps)
            h.start_of_first_evlr = evlr_start
            f.seek(0)
            h.write_to(f, ensure_same_size=True)
        if n:
            del keys, recs, order
        self.remove_spool()
