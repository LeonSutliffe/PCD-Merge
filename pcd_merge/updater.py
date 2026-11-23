"""Self-install, desktop shortcut and automatic updates for the bundled app.

Only active in the bundled (PyInstaller) Windows build. When running from
source nothing here runs.

Layout on disk::

    %LOCALAPPDATA%\\PCD Merge\\
        app\\                 the bundled program (PCD Merge.exe + libraries)
        app-new\\             a new version being staged (temporary)
        swap.cmd              helper that swaps app-new into place (temporary)

* First run from anywhere else (e.g. the unzipped download): the program is
  copied into ``app``, Desktop and Start Menu shortcuts pointing at
  ``app\\PCD Merge.exe`` are created, and the installed copy is started.
* Every start from the shortcut: the latest GitHub release is checked. If it
  is newer, the user is asked; on yes the zip is downloaded into ``app-new``
  and a small script waits for the program to exit, swaps the folders and
  starts the new version. The shortcut never changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable

from . import __version__

REPO = "LeonSutliffe/PCD-Merge"
APP_NAME = "PCD Merge"
EXE_NAME = "PCD Merge.exe"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
ASSET_PATTERN = re.compile(r"^(?!.*addon).*windows.*\.zip$", re.IGNORECASE)

ProgressFn = Callable[[int, int], None]  # (bytes done, bytes total)


# --------------------------------------------------------------------------
# Where things are


def is_bundled() -> bool:
    return bool(getattr(sys, "frozen", False))


def enabled() -> bool:
    """Install/update only in the bundled Windows build, unless switched off."""
    return is_bundled() and sys.platform == "win32" and not os.environ.get("PCD_MERGE_NO_UPDATE")


def install_root() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_NAME)


def app_dir() -> str:
    return os.path.join(install_root(), "app")


def running_dir() -> str:
    return os.path.dirname(os.path.abspath(sys.executable))


def is_installed_copy() -> bool:
    return os.path.normcase(os.path.abspath(running_dir())) == os.path.normcase(
        os.path.abspath(app_dir())
    )


# --------------------------------------------------------------------------
# Versions and releases


def parse_version(v: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(n) for n in nums[:3]) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


@dataclass
class Release:
    version: str
    url: str  # download URL of the Windows zip
    size: int
    notes: str
    page: str
    sha256: str = ""  # from the release asset's digest, when GitHub provides it


def latest_release(timeout: float = 5.0) -> Release | None:
    """Ask GitHub for the newest release. Returns None on any problem."""
    req = urllib.request.Request(
        LATEST_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "pcd-merge-updater"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return None
    return release_from_json(data)


def release_from_json(data: dict) -> Release | None:
    for asset in data.get("assets", []):
        if ASSET_PATTERN.search(asset.get("name", "")):
            return Release(
                version=data.get("tag_name", "").lstrip("v"),
                url=asset["browser_download_url"],
                size=int(asset.get("size", 0)),
                notes=data.get("body") or "",
                page=data.get("html_url", ""),
                sha256=_sha256_of(asset.get("digest") or ""),
            )
    return None


def _sha256_of(digest: str) -> str:
    algo, _, value = digest.partition(":")
    return value.lower() if algo.lower() == "sha256" and len(value) == 64 else ""


def check_for_update() -> Release | None:
    """The latest release if it is newer than this program, else None."""
    rel = latest_release()
    if rel and is_newer(rel.version):
        return rel
    return None


# --------------------------------------------------------------------------
# Staging a new version


class DownloadError(RuntimeError):
    pass


def download(url: str, dest: str, progress: ProgressFn | None = None,
             expected_size: int = 0, sha256: str = "", attempts: int = 6) -> None:
    """Download ``url`` to ``dest``, resuming if the connection drops.

    Python's HTTP client ends a download silently when the connection is cut
    part-way, which used to leave a truncated zip ("File is not a zip file").
    So the size is checked against the server's (or ``expected_size``), short
    downloads are resumed with an HTTP Range request, and the finished file is
    checked against ``sha256`` when given.
    """
    total = expected_size
    done = 0
    open(dest, "wb").close()
    last_error: Exception | None = None
    for attempt in range(attempts):
        if total and done >= total:
            break
        headers = {"User-Agent": "pcd-merge-updater"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60) as r:
                if done and r.status != 206:
                    done = 0  # server ignored the Range request: start again
                    open(dest, "wb").close()
                if not total:
                    length = int(r.headers.get("Content-Length") or 0)
                    total = length + done if length else 0
                with open(dest, "r+b" if done else "wb") as f:
                    f.seek(done)
                    f.truncate()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            if not total:
                break  # size unknown: trust the end of the stream
        except Exception as e:  # network trouble: wait a moment and resume
            last_error = e
        if not (total and done >= total) and attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 10))
    if total and done != total:
        raise DownloadError(
            f"the download stopped at {done / 1e6:.0f} of {total / 1e6:.0f} MB "
            f"({last_error or 'connection closed early'})")
    if sha256:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != sha256:
            raise DownloadError("the downloaded file is damaged (checksum mismatch)")
    if not zipfile.is_zipfile(dest):
        raise DownloadError("the downloaded file is not a valid zip")


def extract_app(zip_path: str, dest: str) -> None:
    """Unzip a release into ``dest`` so that ``dest/PCD Merge.exe`` exists.

    The zip may hold the files directly or inside one top-level folder.
    """
    if os.path.exists(dest):
        shutil.rmtree(dest)
    tmp = dest + ".unzip"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    root = tmp
    entries = os.listdir(tmp)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
        root = os.path.join(tmp, entries[0])
    if not os.path.isfile(os.path.join(root, EXE_NAME)):
        shutil.rmtree(tmp)
        raise RuntimeError(f"the download does not contain {EXE_NAME}")
    shutil.move(root, dest)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)


def stage_update(rel: Release, progress: ProgressFn | None = None) -> str:
    """Download and unpack ``rel`` into the staging folder. Returns its path."""
    os.makedirs(install_root(), exist_ok=True)
    staged = os.path.join(install_root(), "app-new")
    fd, zpath = tempfile.mkstemp(suffix=".zip", prefix="pcd-merge-")
    os.close(fd)
    try:
        download(rel.url, zpath, progress, expected_size=rel.size, sha256=rel.sha256)
        extract_app(zpath, staged)
    finally:
        try:
            os.remove(zpath)
        except OSError:
            pass
    return staged


def stage_self() -> str:
    """Copy the running program into the staging folder (first-run install)."""
    os.makedirs(install_root(), exist_ok=True)
    staged = os.path.join(install_root(), "app-new")
    if os.path.exists(staged):
        shutil.rmtree(staged)
    shutil.copytree(running_dir(), staged)
    return staged


# --------------------------------------------------------------------------
# Swapping it in after we exit

SWAP_SCRIPT = r"""@echo off
setlocal
set "APP={app}"
set "NEW={new}"
set "OLD={old}"
rem wait for the running program to close
:wait
tasklist /FI "PID eq {pid}" 2>NUL | find "{pid}" >NUL
if not errorlevel 1 (
  ping -n 2 127.0.0.1 >NUL
  goto wait
)
if exist "%OLD%" rmdir /s /q "%OLD%"
set /a tries=0
:swap
if exist "%APP%" move "%APP%" "%OLD%" >NUL 2>&1
if not exist "%APP%" goto moved
set /a tries+=1
if %tries% GEQ 30 goto giveup
ping -n 2 127.0.0.1 >NUL
goto swap
:giveup
rem could not replace the old version (still open?); start it as it is
start "" "%APP%\{exe}" {args}
goto end
:moved
move "%NEW%" "%APP%" >NUL 2>&1
if not exist "%APP%\{exe}" (
  rem something went wrong: put the old version back
  if exist "%OLD%" move "%OLD%" "%APP%" >NUL 2>&1
)
start "" "%APP%\{exe}" {args}
if exist "%OLD%" rmdir /s /q "%OLD%"
:end
del "%~f0"
"""


def _quote_args(args: list[str]) -> str:
    return " ".join('"' + a.replace('"', "") + '"' for a in args)


def write_swap_script(pid: int, args: list[str] | None = None) -> str:
    root = install_root()
    script = SWAP_SCRIPT.format(
        app=app_dir(),
        new=os.path.join(root, "app-new"),
        old=os.path.join(root, "app-old"),
        pid=pid,
        exe=EXE_NAME,
        args=_quote_args(args or []),
    )
    path = os.path.join(root, "swap.cmd")
    with open(path, "w", newline="\r\n") as f:
        f.write(script)
    return path


def launch_swap(args: list[str] | None = None) -> None:
    """Start the swap script detached; the caller must then exit promptly."""
    path = write_swap_script(os.getpid(), args)
    flags = 0
    if sys.platform == "win32":
        # hidden console (ping needs one), own process group so it outlives us
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen(["cmd", "/c", path], creationflags=flags, close_fds=True,
                     cwd=install_root())


# --------------------------------------------------------------------------
# Shortcuts

_SHORTCUT_PS = r"""
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
foreach ($folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
  if (-not $folder) {{ continue }}
  $s = $shell.CreateShortcut((Join-Path $folder '{name}.lnk'))
  $s.TargetPath = '{target}'
  $s.WorkingDirectory = '{workdir}'
  $s.IconLocation = '{target},0'
  $s.Description = 'Align and merge LiDAR point clouds'
  $s.Save()
}}
"""


def create_shortcuts() -> bool:
    """Desktop and Start Menu shortcuts to the installed program."""
    if sys.platform != "win32":
        return False
    target = os.path.join(app_dir(), EXE_NAME)
    ps = _SHORTCUT_PS.format(
        name=APP_NAME.replace("'", "''"),
        target=target.replace("'", "''"),
        workdir=os.path.expanduser("~").replace("'", "''"),
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            check=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
        )
        return True
    except Exception:
        return False
