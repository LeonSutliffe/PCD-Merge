"""Entry point for the bundled Windows program (PyInstaller).

Double-click / shortcut: opens the app (after the install / update check).
Files dropped on the icon are loaded into the app.
``--selftest`` runs a quick end-to-end check and exits with 0 or 1.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv[1:]:
        from pcd_merge.selftest import run as selftest

        sys.exit(selftest())
    from pcd_merge.gui import run

    sys.exit(run())
