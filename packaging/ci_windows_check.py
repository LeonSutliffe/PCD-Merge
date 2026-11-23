"""CI check on Windows: the update swap and shortcut creation really work.

Usage: python packaging/ci_windows_check.py "dist/PCD Merge"

Uses a throw-away LOCALAPPDATA. Installs a fake "old version", stages the real
build as the new version, runs the same swap script an update uses (waiting on
a process that exits after a few seconds), and checks that the new build ends
up in place and starts (it is started with --selftest, which writes a log).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

build = os.path.abspath(sys.argv[1])
root = tempfile.mkdtemp(prefix="pcdm-localappdata-")
os.environ["LOCALAPPDATA"] = root
log = os.path.join(tempfile.gettempdir(), "pcd-merge-selftest.log")
if os.path.exists(log):
    os.remove(log)

from pcd_merge import updater  # noqa: E402  (after LOCALAPPDATA is set)

# "old" installed version: a folder with a dummy exe
os.makedirs(updater.app_dir())
with open(os.path.join(updater.app_dir(), updater.EXE_NAME), "w") as f:
    f.write("old")
# new version staged next to it
shutil.copytree(build, os.path.join(updater.install_root(), "app-new"))

# a process standing in for the running program; the script must wait for it
sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"])
script = updater.write_swap_script(sleeper.pid, ["--selftest"])
t0 = time.time()
subprocess.Popen(["cmd", "/c", script], creationflags=subprocess.CREATE_NO_WINDOW
                 | subprocess.CREATE_NEW_PROCESS_GROUP, cwd=updater.install_root())

deadline = time.time() + 300
while time.time() < deadline and not os.path.exists(log):
    time.sleep(1)
waited = time.time() - t0
assert os.path.exists(log), "the swapped-in program never ran its self-test"
time.sleep(2)
text = open(log, encoding="utf-8").read()
print(text)
assert "PASSED" in text, "self-test of the swapped-in program failed"
assert waited >= 3, "swap did not wait for the running program to exit"
exe = os.path.join(updater.app_dir(), updater.EXE_NAME)
assert os.path.getsize(exe) > 1000, "old exe is still in place"
assert not os.path.exists(os.path.join(updater.install_root(), "app-new"))
print("update swap OK")

assert updater.create_shortcuts(), "creating shortcuts failed"
desktop = subprocess.run(
    ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
    capture_output=True, text=True).stdout.strip()
lnk = os.path.join(desktop, "PCD Merge.lnk")
assert os.path.exists(lnk), f"no shortcut at {lnk}"
print("shortcut OK:", lnk)
