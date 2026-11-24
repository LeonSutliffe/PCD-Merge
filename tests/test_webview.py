"""The browser walk-through: one .html file with the viewer and the points."""

import base64
import gzip
import json
import re

import numpy as np
import pytest

from pcd_merge.edits import Edit, EditStack, Sphere
from pcd_merge.flyscene import build_scene, even_subset
from pcd_merge.measure import Measurement
from pcd_merge.pcd_io import PointCloud
from pcd_merge.synthetic import make_scan, make_scene, make_transform
from pcd_merge.webview import page_html, pack, quantise, web_scene, write_page

POSES = [make_transform(0.0, 0.0, (-5, 0, 1.5)), make_transform(1.0, 0.0, (4, 3, 1.6))]


@pytest.fixture(scope="module")
def clouds():
    scene = make_scene(seed=0, density=60)
    return [PointCloud(make_scan(scene, p[:2, 3], 18, p, seed=i), path=f"scan{i}.pcd")
            for i, p in enumerate(POSES)]


def _unpack(text):
    meta = json.loads(re.search(r'<script type="application/json" id="meta">(.*?)</script>',
                                text, re.S).group(1))
    blocks = re.findall(r'<script type="application/octet-stream" class="pts">(.*?)</script>',
                        text, re.S)
    data = gzip.decompress(b"".join(base64.b64decode(b) for b in blocks))
    return meta, data


def test_page_holds_the_points_to_the_millimetre(clouds, tmp_path):
    sc = build_scene(clouds, POSES)
    m = Measurement("distance", [sc.origin + [0, 0, 0], sc.origin + [3, 4, 0]], name="Door")
    path = tmp_path / "walk.html"
    size = write_page(str(path), sc, "My <site>", [m], app_version="9.9")
    text = path.read_text(encoding="utf-8")
    assert size == len(text.encode())
    assert "<title>My &lt;site&gt;</title>" in text
    assert "http://" not in text.replace("http://www.w3.org", "")  # nothing from the internet
    meta, data = _unpack(text)
    assert meta["points"] == len(sc) and meta["chunks"] == len(sc.chunk_count)
    assert meta["has_intensity"] and not meta["has_rgb"]
    assert meta["measurements"][0]["name"] == "Door"
    assert np.allclose(meta["measurements"][0]["points"][1], [3, 4, 0])
    assert meta["start"]["pos"] and meta["speed"] > 0

    c, n = meta["chunks"], meta["points"]
    table = np.frombuffer(data, "<f4", c * 11).reshape(c, 11)
    counts = np.frombuffer(data, "<u4", c, c * 44)
    q = np.frombuffer(data, "<u2", n * 3, c * 48).reshape(n, 3).astype(np.float64)
    assert np.array_equal(counts, sc.chunk_count)
    chunk = np.repeat(np.arange(c), counts)
    back = table[chunk, 0:3] + q * table[chunk, 3:6]
    err = np.abs(back - sc.vertices["pos"]).max()
    assert err < 1e-3  # under a millimetre
    assert np.allclose(table[:, 6:9], sc.chunk_centre) and np.allclose(table[:, 10],
                                                                        sc.chunk_spacing)


def test_even_thinning_keeps_far_areas():
    rng = np.random.default_rng(1)
    # dense next to a scanner, sparse far away
    near = rng.random((400_000, 3)) * [10, 10, 0.01]
    far = rng.random((20_000, 3)) * [10, 10, 0.01] + [100, 0, 0]
    xyz = np.concatenate([near, far])
    keep = even_subset(xyz, 40_000, rng)
    assert 30_000 <= len(keep) <= 40_000
    kept_far = int((keep >= len(near)).sum())
    # at random only ~1/21 of the kept points would be far ones; evenly about half
    assert kept_far > 0.3 * len(keep)
    assert len(np.unique(keep)) == len(keep)


def test_web_scene_leaves_out_deleted_points_and_is_small(clouds, tmp_path):
    full = build_scene(clouds, POSES)
    centre = full.vertices["pos"][100].astype(float) + full.origin
    st = EditStack()
    st.add(Edit("delete", Sphere(centre, 2.0)))
    sc = web_scene(clouds, POSES, st, ["a", "b"], max_points=30_000)
    assert len(sc) <= 30_000
    xyz = sc.vertices["pos"].astype(float) + sc.origin
    assert not (np.linalg.norm(xyz - centre, axis=1) < 1.99).any()
    meta, data = pack(sc)
    assert len(data) < 12 * len(sc) + 64 * meta["chunks"]
    lo, step, q = quantise(sc)
    assert q.dtype == np.dtype("<u2") and lo.shape == (len(sc.chunk_count), 3)


def test_page_runs_in_a_browser(clouds, tmp_path):
    """Open the page in a real (headless) browser where one is available."""
    pw = pytest.importorskip("playwright.sync_api")
    sc = build_scene(clouds, POSES, max_points=50_000)
    path = tmp_path / "walk.html"
    write_page(str(path), sc, "Test", [])
    try:
        with pw.sync_playwright() as p:
            b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                        "--ignore-gpu-blocklist"])
            page = b.new_page(viewport={"width": 900, "height": 600})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(path.as_uri())
            page.wait_for_function("window.__walk", timeout=60_000)
            page.wait_for_timeout(300)
            assert page.evaluate("window.__walk.drawn") > 0
            assert page.evaluate("window.__walk.mode") == "walk"  # opens walking
            before = page.evaluate("window.__walk.cam.pos")
            page.keyboard.down("KeyW")
            page.wait_for_timeout(400)
            page.keyboard.up("KeyW")
            assert page.evaluate("window.__walk.cam.pos") != before
            page.keyboard.press("KeyM")
            page.mouse.click(450, 400)
            page.mouse.click(520, 420)
            assert page.evaluate("window.__walk.measurements.length") in (0, 1)
            assert not errors, errors
            b.close()
    except Exception as e:  # no browser on this machine
        if "Executable doesn't exist" in str(e) or "BrowserType.launch" in str(e):
            pytest.skip("no browser")
        raise
