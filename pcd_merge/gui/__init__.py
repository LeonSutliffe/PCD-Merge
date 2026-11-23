"""Desktop app. One module per part of the window (see main_window.py)."""

from .main_window import MainWindow, run
from .view3d import HAVE_GL

__all__ = ["HAVE_GL", "MainWindow", "run"]
