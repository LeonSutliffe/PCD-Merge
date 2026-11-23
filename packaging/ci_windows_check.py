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
# Leftovers from an earlier failed update that Windows won't let us delete:
# an "app-new" and an "app-old" each holding a read-only file and a file kept
# open by another process. They must not block this update.
locks = []
for name in ("app-new", "app-old"):
    d = os.path.join(updater.install_root(), name, "_internal", "x.dist-info", "licenses")
    os.makedirs(d)
    ro = os.path.join(d, "LICENSE")
    with open(ro, "w") as f:
        f.write("x")
    os.chmod(ro, 0o444)
    held = os.path.join(d, "held.txt")
    with open(held, "w") as f:
        f.write("x")
    locks.append(subprocess.Popen(
        [sys.executable, "-c", f"import time; f=open({held!r}); time.sleep(120)"]))
time.sleep(2)
updater.clean_leftovers()  # must not raise even though it can't delete everything
assert os.path.exists(os.path.join(updater.install_root(), "app-new")), "lock not held?"

# new version staged in a fresh folder next to them, as stage_update does
staged = updater._fresh_dir("app-new")
shutil.copytree(build, staged)

# a process standing in for the running program; the script must wait for it
sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"])
script = updater.write_swap_script(sleeper.pid, ["--selftest"], staged)
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
assert not os.path.exists(staged)
print("update swap OK (with undeletable leftovers present)")
for p in locks:
    p.kill()
    p.wait()
time.sleep(1)
updater.clean_leftovers()
left = [n for n in os.listdir(updater.install_root()) if n.startswith("app-")]
print("left in install folder after clean-up:", left)
assert left == [], "leftovers were not cleaned up once released"
assert os.path.isfile(exe), "the installed program went missing during clean-up"

assert updater.create_shortcuts(), "creating shortcuts failed"
desktop = subprocess.run(
    ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
    capture_output=True, text=True).stdout.strip()
lnk = os.path.join(desktop, "PCD Merge.lnk")
assert os.path.exists(lnk), f"no shortcut at {lnk}"
print("shortcut OK:", lnk)
