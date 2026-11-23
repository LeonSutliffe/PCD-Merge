"""Apply alignment transforms to full-resolution clouds and join them."""

from __future__ import annotations

import numpy as np

from .pcd_io import PointCloud

_NORMAL_SETS = (("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz"))


def transform_cloud(cloud: PointCloud, t: np.ndarray, drop_invalid: bool = True) -> np.ndarray:
    """Return a copy of ``cloud.data`` with points (and normals) transformed.

    Maths is done in float64 and written back in each field's original type.
    """
    data = cloud.data
    if drop_invalid:
        data = data[cloud.finite_mask()]
    out = data.copy()
    t = np.asarray(t, dtype=np.float64)
    r, tr = t[:3, :3], t[:3, 3]

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    xyz = xyz @ r.T + tr
    for k, name in enumerate(("x", "y", "z")):
        out[name] = xyz[:, k]

    names = set(data.dtype.names)
    for trio in _NORMAL_SETS:
        if all(n in names for n in trio):
            nrm = np.stack([data[n] for n in trio], axis=1).astype(np.float64)
            nrm = nrm @ r.T
            for k, n in enumerate(trio):
                out[n] = nrm[:, k]
    return out


def _merged_dtype(dtypes: list[np.dtype]) -> tuple[np.dtype, list[str]]:
    """Union of all fields, in first-seen order, with promoted types."""
    order: list[str] = []
    types: dict[str, list] = {}
    for dt in dtypes:
        for name in dt.names:
            sub = dt.fields[name][0]
            if name not in types:
                order.append(name)
                types[name] = []
            types[name].append(sub)
    fields = []
    for name in order:
        subs = types[name]
        shapes = {s.shape for s in subs}
        base = subs[0].base
        if len(shapes) > 1:
            raise ValueError(f"field '{name}' has a different COUNT in different files")
        if name in ("rgb", "rgba"):
            base = subs[0].base  # packed colour: keep bit pattern, don't promote
        else:
            base = np.result_type(*[s.base for s in subs])
        shape = subs[0].shape
        fields.append((name, base.newbyteorder("<")) if not shape else (name, base.newbyteorder("<"), shape))
    missing = [n for n in order if any(n not in dt.names for dt in dtypes)]
    return np.dtype(fields), missing


def merge_clouds(clouds: list[PointCloud], transforms: list[np.ndarray],
                 add_source_field: bool = False,
                 progress=None) -> tuple[np.ndarray, list[str]]:
    """Transform every cloud and concatenate all of their points.

    Fields present in only some files are kept; points from files without that
    field get 0. Returns the merged structured array and the names of those
    partially-present fields so the caller can warn about them.

    ``add_source_field`` adds a ``source`` field (U2) holding the index of the
    file each point came from. ``progress(fraction)`` is called as clouds are
    processed.

    Points are transformed in chunks straight into the output array, so the
    only full-size allocation is the merged result itself.
    """
    if len(clouds) != len(transforms):
        raise ValueError("one transform per cloud is required")
    report = progress or (lambda _f: None)
    dtype, missing = _merged_dtype([c.data.dtype for c in clouds])
    if add_source_field:
        if "source" in dtype.names:
            raise ValueError("clouds already have a 'source' field")
        dtype = np.dtype(dtype.descr + [("source", "<u2")])

    masks = [c.finite_mask() for c in clouds]
    counts = [int(m.sum()) for m in masks]
    total = sum(counts)
    out = np.zeros(total, dtype=dtype)
    pos = done = 0
    for k, (c, t, m) in enumerate(zip(clouds, transforms, masks)):
        data = c.data
        all_ok = counts[k] == len(data)
        for start in range(0, len(data), _CHUNK):
            src = data[start : start + _CHUNK]
            if not all_ok:
                src = src[m[start : start + _CHUNK]]
            sl = slice(pos, pos + len(src))
            _copy_transformed(src, out, sl, t, dtype)
            if add_source_field:
                out["source"][sl] = k
            pos += len(src)
            done += len(src)
            report(done / max(1, total))
    report(1.0)
    return out, missing


_CHUNK = 2_000_000  # points transformed at a time (bounds float64 scratch memory)


def _copy_transformed(src: np.ndarray, out: np.ndarray, sl: slice, t: np.ndarray,
                      dtype: np.dtype) -> None:
    """Write ``src`` (one chunk of one cloud) into ``out[sl]`` with ``t`` applied."""
    t = np.asarray(t, dtype=np.float64)
    r, tr = t[:3, :3], t[:3, 3]
    names = src.dtype.names
    xyz = np.empty((len(src), 3), dtype=np.float64)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = src["x"], src["y"], src["z"]
    xyz = xyz @ r.T + tr
    moved = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    for trio in _NORMAL_SETS:
        if all(n in names for n in trio):
            nrm = np.stack([src[n] for n in trio], axis=1).astype(np.float64) @ r.T
            for j, n in enumerate(trio):
                moved[n] = nrm[:, j]
    for name in names:
        dst = out[name]
        if name in moved:
            # cast back to the file's own type first, as transform_cloud does
            dst[sl] = moved[name].astype(src.dtype[name], copy=False)
        elif name in ("rgb", "rgba") and src.dtype[name] != dtype[name]:
            dst[sl] = src[name].view(dtype[name])
        else:
            dst[sl] = src[name]
