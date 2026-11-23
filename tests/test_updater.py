import os
import zipfile

import pytest

from pcd_merge import __version__, updater


def test_version_compare():
    assert updater.is_newer("0.10.0", "0.9.9")
    assert updater.is_newer("v1.0.0", "0.2.0")
    assert not updater.is_newer("0.2.0", "0.2.0")
    assert not updater.is_newer("0.1.5", "0.2.0")
    assert not updater.is_newer("garbage", "0.2.0")


def test_release_from_json_picks_windows_zip():
    data = {
        "tag_name": "v9.9.9",
        "html_url": "https://example/release",
        "body": "notes",
        "assets": [
            {"name": "source.tar.gz", "browser_download_url": "x", "size": 1},
            {"name": "PCD-Merge-9.9.9-windows-x64.zip", "browser_download_url": "https://dl/zip",
             "size": 123},
        ],
    }
    rel = updater.release_from_json(data)
    assert rel.version == "9.9.9" and rel.url == "https://dl/zip" and rel.size == 123
    assert updater.release_from_json({"tag_name": "v1", "assets": []}) is None


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)


@pytest.mark.parametrize("prefix", ["", "PCD Merge/"])
def test_extract_app_handles_both_layouts(tmp_path, prefix):
    zp = str(tmp_path / "r.zip")
    _zip(zp, {prefix + updater.EXE_NAME: "exe", prefix + "_internal/lib.dll": "dll"})
    dest = str(tmp_path / "app-new")
    os.makedirs(dest)  # a stale staging folder is replaced
    updater.extract_app(zp, dest)
    assert os.path.isfile(os.path.join(dest, updater.EXE_NAME))
    assert os.path.isfile(os.path.join(dest, "_internal", "lib.dll"))
    assert not os.path.exists(dest + ".unzip")


def test_extract_app_rejects_wrong_zip(tmp_path):
    zp = str(tmp_path / "r.zip")
    _zip(zp, {"readme.txt": "hi"})
    with pytest.raises(RuntimeError):
        updater.extract_app(zp, str(tmp_path / "app-new"))


def test_swap_script(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    os.makedirs(updater.install_root())
    path = updater.write_swap_script(4242, ["C:\\scans\\a.pcd"])
    text = open(path).read()
    assert "PID eq 4242" in text
    assert updater.app_dir() in text and "app-new" in text
    assert '"C:\\scans\\a.pcd"' in text
    assert "{" not in text  # every placeholder was filled


def test_disabled_when_running_from_source():
    assert not updater.enabled()
    assert __version__


# --------------------------------------------------------------------------
# Download robustness: a local HTTP server that drops the connection part-way


import hashlib
import http.server
import threading

PAYLOAD_FILES = {updater.EXE_NAME: os.urandom(300_000), "_internal/a.dll": os.urandom(200_000)}


def _payload(tmp_path):
    zp = tmp_path / "src.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for n, c in PAYLOAD_FILES.items():
            z.writestr(n, c)
    return zp.read_bytes()


def _serve(data, cut_after=None, honour_range=True, max_cuts=1):
    """Serve ``data``; the first ``max_cuts`` responses stop after ``cut_after`` bytes."""
    state = {"cuts": 0, "requests": []}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            start = 0
            rng = self.headers.get("Range")
            state["requests"].append(rng)
            if rng and honour_range:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                self.send_response(200)
            body = data[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if cut_after is not None and state["cuts"] < max_cuts:
                state["cuts"] += 1
                self.wfile.write(body[:cut_after])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/r.zip", state


def test_download_resumes_after_dropped_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.time, "sleep", lambda s: None)
    data = _payload(tmp_path)
    srv, url, state = _serve(data, cut_after=100_000, max_cuts=2)
    try:
        dest = str(tmp_path / "out.zip")
        updater.download(url, dest, expected_size=len(data),
                         sha256=hashlib.sha256(data).hexdigest())
    finally:
        srv.shutdown()
    assert open(dest, "rb").read() == data
    assert state["requests"][1] == "bytes=100000-"  # resumed, not restarted


def test_download_restarts_when_range_not_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.time, "sleep", lambda s: None)
    data = _payload(tmp_path)
    srv, url, _ = _serve(data, cut_after=100_000, honour_range=False)
    try:
        dest = str(tmp_path / "out.zip")
        updater.download(url, dest, expected_size=len(data))
    finally:
        srv.shutdown()
    assert open(dest, "rb").read() == data


def test_download_gives_clear_error_when_always_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(updater.time, "sleep", lambda s: None)
    data = _payload(tmp_path)
    srv, url, _ = _serve(data, cut_after=0, max_cuts=99)
    try:
        with pytest.raises(updater.DownloadError, match="stopped at"):
            updater.download(url, str(tmp_path / "out.zip"), expected_size=len(data),
                             attempts=3)
    finally:
        srv.shutdown()


def test_download_checks_checksum(tmp_path, monkeypatch):
    data = _payload(tmp_path)
    srv, url, _ = _serve(data)
    try:
        with pytest.raises(updater.DownloadError, match="checksum"):
            updater.download(url, str(tmp_path / "out.zip"), sha256="0" * 64)
    finally:
        srv.shutdown()


def test_release_digest_is_read():
    rel = updater.release_from_json({"tag_name": "v1.0.0", "assets": [{
        "name": "PCD-Merge-1.0.0-windows-x64.zip", "browser_download_url": "u", "size": 5,
        "digest": "sha256:" + "ab" * 32}]})
    assert rel.sha256 == "ab" * 32


def test_gpu_addon_asset_is_never_taken_for_the_program():
    from pcd_merge.updater import release_from_json

    data = {"tag_name": "v0.9.0", "assets": [
        {"name": "PCD-Merge-GPU-addon-x64.zip", "browser_download_url": "https://x/addon"},
        {"name": "PCD-Merge-GPU-addon-windows-x64.zip", "browser_download_url": "https://x/a2"},
        {"name": "PCD-Merge-0.9.0-windows-x64.zip", "browser_download_url": "https://x/app"},
    ]}
    assert release_from_json(data).url == "https://x/app"
