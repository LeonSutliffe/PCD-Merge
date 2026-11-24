"""One Save As window for everything the app writes (merged clouds, cleaned
clouds, parts, viewing files, browser walk-throughs, measurements,
screenshots, transforms), so each one lets you choose where it goes and
starts in the folder you last saved to."""

from __future__ import annotations

import os
import re

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog

from ..project import last_save_folder, remember_save_folder


def safe_name(name: str) -> str:
    """A file name from free text (part names can hold ':' and the like)."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name)
    return re.sub(r"\s+", " ", name).strip(" .") or "untitled"


def start_folder(doc=None) -> str:
    """Where a Save As window opens: the folder last saved to, else the
    folder holding the scans, else Documents."""
    d = last_save_folder()
    if d:
        return d
    if doc is not None:
        for s in getattr(doc, "scans", []):
            folder = os.path.dirname(s.entry.path)
            if os.path.isdir(folder):
                return folder
    return QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) \
        or os.path.expanduser("~")


def ask_save_path(parent, title: str, name: str, filters: str, chosen: str = "",
                  doc=None, suggested: str | None = None) -> tuple[str, str]:
    """Show the Save As window. ``name`` is the file name offered in the
    start folder; ``suggested`` (a full path) wins over that, for a file
    that is usually saved in place. Returns (path, chosen filter); the path
    is "" if cancelled. The folder chosen is remembered for next time."""
    start = suggested or os.path.join(start_folder(doc), safe_name(name))
    path, selected = QFileDialog.getSaveFileName(parent, title, start, filters, chosen)
    if path:
        remember_save_folder(os.path.dirname(os.path.abspath(path)))
    return path, selected
