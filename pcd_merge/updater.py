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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable

from . import __version__

REPO = "LeonSutliffe/PCD-Merge"
APP_NAME = "PCD Merge"
EXE_NAME = "PCD Merge.exe"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
ASSET_PATTERN = re.compile(r"windows.*\.zip$", re.IGNORECASE)

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
            )
    return None


def check_for_update() -> Release | None:
    """The latest release if it is newer than this program, else None."""
    rel = latest_release()
    if rel and is_newer(rel.version):
        return rel
    return None


# --------------------------------------------------------------------------
# Staging a new version


def download(url: str, dest: str, progress: ProgressFn | None = None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "pcd-merge-updater"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)


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
        download(rel.url, zpath, progress)
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
