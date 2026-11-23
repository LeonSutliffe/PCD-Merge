"""What the bundled app does before its main window opens.

* Started from outside the install folder (first run): offer to install for
  this user and put a shortcut on the Desktop, then start the installed copy.
* Started from the install folder (the shortcut): check GitHub for a newer
  release and offer to update, then start (the new version, if updated).

Any failure here is reported and the program simply carries on running.
"""

from __future__ import annotations

import sys
import threading

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from . import __version__, updater


def _in_background(app: QApplication, label: str, fn, *args, determinate: bool = False):
    """Run ``fn`` in a thread while showing a small progress window.

    ``fn`` receives a progress(done, total) callback as its last argument
    when ``determinate`` is true. Returns fn's result or raises its error.
    """
    dlg = QProgressDialog(label, None, 0, 0)
    dlg.setWindowTitle("PCD Merge")
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setMinimumDuration(400)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    state = {"done": 0, "total": 0, "result": None, "error": None}

    def progress(done, total):
        state["done"], state["total"] = done, total

    def work():
        try:
            state["result"] = fn(*args, progress) if determinate else fn(*args)
        except BaseException as e:  # noqa: BLE001 - reported to the user below
            state["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        if determinate and state["total"]:
            dlg.setMaximum(100)
            dlg.setValue(int(100 * state["done"] / state["total"]))
            mb = state["done"] / 1e6, state["total"] / 1e6
            dlg.setLabelText(f"{label}\n{mb[0]:.0f} of {mb[1]:.0f} MB")
        app.processEvents()
        t.join(0.05)
    dlg.close()
    if state["error"] is not None:
        raise state["error"]
    return state["result"]


def _first_run_install(app: QApplication) -> bool:
    """Returns True if the app should exit (the installed copy is starting)."""
    ans = QMessageBox.question(
        None, "Install PCD Merge",
        "Install PCD Merge for your user account?\n\n"
        "It will be copied to your AppData folder and a 'PCD Merge' shortcut will be put on "
        "your Desktop and in the Start menu. Starting it from that shortcut checks for "
        "updates each time.\n\nChoose No to just run it from here this once.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if ans != QMessageBox.Yes:
        return False
    try:
        staged = _in_background(app, "Installing PCD Merge…", updater.stage_self)
        if not updater.create_shortcuts():
            QMessageBox.warning(None, "PCD Merge",
                                "PCD Merge was installed, but the Desktop shortcut could not be "
                                f"created. You can start it from:\n{updater.app_dir()}")
        updater.launch_swap(sys.argv[1:], staged)
        return True
    except Exception as e:
        QMessageBox.warning(None, "PCD Merge", f"Installing failed:\n{e}\n\n"
                            "PCD Merge will run from its current folder instead.")
        return False


def _check_and_update(app: QApplication) -> bool:
    """Returns True if the app should exit (the updated version is starting)."""
    try:
        rel = _in_background(app, "Checking for updates…", updater.check_for_update)
    except Exception:
        return False
    if rel is None:
        return False
    settings = QSettings("PCD Merge", "PCD Merge")
    if settings.value("skip_version", "") == rel.version:
        return False

    box = QMessageBox()
    box.setWindowTitle("Update available")
    box.setText(f"PCD Merge {rel.version} is available (you have {__version__}).\n\n"
                "Update now? It takes a minute, then PCD Merge starts again.")
    if rel.notes.strip():
        box.setDetailedText(rel.notes.strip())
    b_now = box.addButton("Update now", QMessageBox.AcceptRole)
    box.addButton("Not now", QMessageBox.RejectRole)
    b_skip = box.addButton("Skip this version", QMessageBox.DestructiveRole)
    box.setDefaultButton(b_now)
    box.exec()
    clicked = box.clickedButton()
    if clicked is b_skip:
        settings.setValue("skip_version", rel.version)
        return False
    if clicked is not b_now:
        return False
    try:
        staged = _in_background(app, f"Downloading PCD Merge {rel.version}…",
                                updater.stage_update, rel, determinate=True)
        updater.launch_swap(sys.argv[1:], staged)
        return True
    except Exception as e:
        _update_failed(rel, e)
        return False


def _update_failed(rel: "updater.Release", error: Exception) -> None:
    """Explain the failure and offer the manual route: download the zip and
    run the exe inside, which installs over the current copy."""
    box = QMessageBox()
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("PCD Merge")
    box.setText(f"The update to {rel.version} failed:\n{error}\n\n"
                "You can update by hand instead: download the zip, unzip it and run "
                "PCD Merge.exe inside. It will install over this version and keep your "
                "shortcut.\n\nFor now, PCD Merge carries on with the current version.")
    b_open = box.addButton("Open download page", QMessageBox.AcceptRole)
    box.addButton("Close", QMessageBox.RejectRole)
    box.setDefaultButton(b_open)
    box.exec()
    if box.clickedButton() is b_open:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(rel.page or updater.RELEASES_PAGE))


def before_main_window(app: QApplication) -> bool:
    """Run install / update steps. True means: exit now, don't open the window."""
    if not updater.enabled():
        return False
    if not updater.is_installed_copy():
        return _first_run_install(app)
    return _check_and_update(app)
