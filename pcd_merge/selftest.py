"""Quick end-to-end check that a build works: ``PCD Merge.exe --selftest``.

Writes a log and exits 0 on success, 1 on failure. Used by CI on the bundled
Windows program, and handy for checking an install by hand.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback


def run(log_path: str | None = None) -> int:
    log_path = log_path or os.environ.get("PCD_MERGE_SELFTEST_LOG") or os.path.join(
        tempfile.gettempdir(), "pcd-merge-selftest.log")
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        if sys.stdout is not None:
            print(msg, flush=True)

    ok = False
    t0 = time.time()
    try:
        import numpy as np

        from . import __version__
        from .merge import merge_clouds
        from .pcd_io import read_pcd, write_pcd
        from .registration import AlignSettings, align
        from .synthetic import make_scan, make_scene, make_transform, transform_error

        log(f"PCD Merge {__version__} self-test, Python {sys.version.split()[0]}")
        poses = [make_transform(0.3, 0.02, (-5, 0, 1.5)), make_transform(2.0, 0.03, (4, 3, 1.6))]
        scene = make_scene(seed=0, density=100)
        tmp = tempfile.mkdtemp(prefix="pcd-merge-selftest-")
        paths = []
        for i, p in enumerate(poses):
            path = os.path.join(tmp, f"scan{i}.pcd")
            write_pcd(path, make_scan(scene, p[:2, 3], 18, p, seed=i), compressed=(i == 1))
            paths.append(path)
        clouds = [read_pcd(p) for p in paths]
        log(f"read/write ok: {[len(c) for c in clouds]} points")

        r = align([c.xyz for c in clouds], AlignSettings(seed=1))
        rot, trans = transform_error(r.transforms[1], np.linalg.inv(poses[0]) @ poses[1])
        log(f"alignment error: {rot:.4f} deg, {trans:.4f} m")
        if rot > 0.5 or trans > 0.05:
            raise RuntimeError("alignment is wrong")

        from .quality import assess

        q = assess([c.xyz for c in clouds], r.transforms, r.voxel_size,
                   preview_xyz=[c.xyz[::10] for c in clouds])
        log(f"quality ok: {q.grade}, rmse {q.pairs[0].rmse:.4f}")
        if q.grade == "Poor":
            raise RuntimeError("quality report says the correct alignment is poor")

        merged, _ = merge_clouds(clouds, r.transforms)
        out = os.path.join(tmp, "merged.pcd")
        write_pcd(out, merged)
        assert len(read_pcd(out)) == sum(len(c) for c in clouds)
        log(f"merge ok: {len(merged)} points")

        # Every export format, read back with its own library, so a build
        # that is missing laspy/lazrs/pye57 (or their DLLs) fails here.
        import laspy
        import pye57

        from .exporters import FORMATS, write_cloud

        for ext in FORMATS:
            write_cloud(os.path.join(tmp, "merged" + ext), merged)
        for ext in (".las", ".laz"):
            assert len(laspy.read(os.path.join(tmp, "merged" + ext)).points) == len(merged)
        with laspy.CopcReader.open(os.path.join(tmp, "merged.copc.laz")) as copc:
            assert len(copc.query()) == len(merged)  # the octree index finds every point
        e57 = pye57.E57(os.path.join(tmp, "merged.e57"))
        assert e57.get_header(0).point_count == len(merged)
        e57.close()
        log(f"export ok: {', '.join(FORMATS)}")

        from .project import Project
        from .readers import read_cloud

        for ext in FORMATS:
            back = read_cloud(os.path.join(tmp, "merged" + ext))
            assert len(back) == len(merged), (ext, len(back))
        log(f"read back ok: {', '.join(FORMATS)}")
        proj = Project.create(os.path.join(tmp, "Self-test"))
        proj.add(read_cloud(os.path.join(tmp, "merged.pcd")))
        proj.save()
        assert len(Project.open(proj.folder).scans) == 1
        log("project ok")

        from PySide6.QtWidgets import QApplication

        from .gui import HAVE_GL, MainWindow

        app = QApplication.instance() or QApplication([sys.argv[0]])
        w = MainWindow()
        w.close()
        log(f"GUI ok (3D preview available: {HAVE_GL})")

        from .flyscene import build_scene
        from .flythrough import FlyThroughWindow

        fly = FlyThroughWindow(build_scene(clouds, r.transforms), "self-test")
        log(f"walk-through ok: {len(fly.scene):,} points, "
            f"{len(fly.scene.scan_positions)} scan positions")
        fly.close()
        del app
        ok = True
    except Exception:
        log("FAILED\n" + traceback.format_exc())
    log(f"{'PASSED' if ok else 'FAILED'} in {time.time() - t0:.1f}s")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    return 0 if ok else 1
