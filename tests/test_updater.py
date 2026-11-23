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
