"""The app's look: one dark theme for every panel, so the 3D view and the
controls around it read as one window instead of grey forms around a black
box. Applied once to the whole application (``apply``).

Buttons that are the next thing to press carry ``primary="true"`` (a blue
button); ``danger="true"`` is for Delete.
"""

from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QStyleFactory

BG = "#15171c"  # behind everything, and inside lists and inputs
PANEL = "#1d2027"  # the side panels, menus, the top and bottom bars
RAISED = "#262a33"  # buttons, hovered rows
LINE = "#323744"  # borders
TEXT = "#e6e8ee"
DIM = "#9aa1af"  # secondary text
FAINT = "#6b7280"  # hints, disabled
ACCENT = "#4c9dff"
ACCENT_DARK = "#2f6fc0"
DANGER = "#ff8a8a"

STYLE = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {PANEL}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE}; padding: 5px; }}
QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {RAISED}; border-radius: 4px; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_DARK}; }}
QMenu::item:disabled {{ color: {FAINT}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 8px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {LINE}; color: {DIM}; }}
QSplitter::handle {{ background: {LINE}; }}
QSplitter::handle:horizontal {{ width: 1px; }}

QGroupBox {{ border: 1px solid {LINE}; border-radius: 10px; margin-top: 22px;
            padding: 10px 8px 8px 8px; background: {BG}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; top: 2px; padding: 0 2px;
                   color: {DIM}; font-weight: bold; }}

QPushButton, QToolButton {{ background: {RAISED}; border: 1px solid {LINE}; border-radius: 6px;
                            padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ border-color: #4a5160; background: #2c313b; }}
QPushButton:pressed, QToolButton:pressed {{ background: {BG}; }}
QPushButton:checked, QToolButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {FAINT}; background: {PANEL};
                                              border-color: {RAISED}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; border-color: {ACCENT}; color: white;
                              font-weight: bold; padding: 7px 14px; }}
QPushButton[primary="true"]:hover {{ background: #63aaff; }}
QPushButton[primary="true"]:disabled {{ background: #26364d; border-color: #26364d;
                                       color: #7f93ad; }}
QPushButton[danger="true"] {{ background: #3a2226; border-color: #5b2d33; color: {DANGER}; }}
QPushButton[danger="true"]:hover {{ background: #4a2a30; }}

QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {BG}; border: 1px solid {LINE}; border-radius: 6px; padding: 3px 6px;
    selection-background-color: {ACCENT_DARK}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {FAINT}; }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border;
    width: 16px; border: none; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-arrow {{ image: url(@ICONS@/up.png); width: 8px; height: 8px; }}
QAbstractSpinBox::down-arrow {{ image: url(@ICONS@/down.png); width: 8px; height: 8px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url(@ICONS@/up-off.png); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url(@ICONS@/down-off.png); }}
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: url(@ICONS@/down.png); width: 8px; height: 8px; }}
QComboBox::down-arrow:disabled {{ image: url(@ICONS@/down-off.png); }}
QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QTreeView::indicator, QListView::indicator {{ width: 14px; height: 14px;
    border: 1px solid #4a5160; border-radius: 4px; background: {BG}; }}
QCheckBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url(@ICONS@/tick.png); }}
QCheckBox::indicator:disabled {{ border-color: {RAISED}; }}
QCheckBox::indicator:checked:disabled {{ background: #26364d; border-color: #26364d; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid #4a5160;
    border-radius: 8px; background: {BG}; }}
QRadioButton::indicator:checked {{ border-color: {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 white, stop:0.35 white, stop:0.45 {ACCENT},
                                stop:1 {ACCENT}); }}
QRadioButton::indicator:disabled {{ border-color: {RAISED}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE};
                               selection-background-color: {ACCENT_DARK}; }}

QTreeView, QListView, QTableView {{ background: {BG}; border: 1px solid {LINE};
    border-radius: 8px; alternate-background-color: {PANEL}; outline: 0; }}
QTreeView::item, QListView::item {{ padding: 3px 2px; border-radius: 4px; }}
QTreeView::item:hover, QListView::item:hover {{ background: {RAISED}; }}
QTreeView::item:selected, QListView::item:selected {{ background: #23395a; color: {TEXT}; }}
QHeaderView::section {{ background: {PANEL}; border: none; padding: 4px; color: {DIM}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {LINE}; border-radius: 3px; min-height: 24px;
                     min-width: 24px; }}
QScrollBar::handle:hover {{ background: #4a5160; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QProgressBar {{ background: {RAISED}; border: none; border-radius: 4px; text-align: center;
               color: {TEXT}; max-height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QSlider::groove:horizontal {{ height: 4px; background: {RAISED}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; width: 12px; margin: -5px 0;
                             border-radius: 6px; }}

/* the mode switch at the top: a segmented control */
QTabBar#modes {{ background: {BG}; border: 1px solid {LINE}; border-radius: 9px; }}
QTabBar#modes::tab {{ background: transparent; color: {DIM}; border: none; margin: 3px;
                     padding: 5px 22px; border-radius: 6px; font-weight: bold; }}
QTabBar#modes::tab:hover {{ color: {TEXT}; }}
QTabBar#modes::tab:selected {{ background: {ACCENT}; color: white; }}

/* the bar along the top of the window, and the strip above the 3D view */
QToolBar#topbar {{ background: {PANEL}; border: none; border-bottom: 1px solid {LINE};
                  padding: 4px 10px; spacing: 6px; }}
QWidget#viewstrip {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QLabel#projectname {{ color: {DIM}; }}
QLabel#appname {{ font-weight: bold; }}
"""


def palette() -> QPalette:
    p = QPalette()
    c = {
        QPalette.Window: PANEL, QPalette.WindowText: TEXT, QPalette.Base: BG,
        QPalette.AlternateBase: PANEL, QPalette.ToolTipBase: RAISED, QPalette.ToolTipText: TEXT,
        QPalette.Text: TEXT, QPalette.Button: RAISED, QPalette.ButtonText: TEXT,
        QPalette.BrightText: "#ffffff", QPalette.Highlight: ACCENT_DARK,
        QPalette.HighlightedText: "#ffffff", QPalette.Link: ACCENT,
        QPalette.PlaceholderText: FAINT, QPalette.Mid: LINE, QPalette.Dark: BG,
        QPalette.Light: RAISED, QPalette.Midlight: LINE, QPalette.Shadow: "#000000",
    }
    for role, col in c.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(FAINT))
    return p


def _icons() -> str:
    """Small arrows and the tick, drawn once into a folder the style sheet
    can point at (style sheets take images from files only)."""
    folder = os.path.join(tempfile.gettempdir(), "pcd-merge-theme")
    os.makedirs(folder, exist_ok=True)

    def save(name, colour, draw):
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(colour), 4.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        draw(p)
        p.end()
        pm.save(os.path.join(folder, name))

    def chevron(up):
        def draw(p):
            path = QPainterPath(QPointF(6, 21 if up else 11))
            path.lineTo(16, 11 if up else 21)
            path.lineTo(26, 21 if up else 11)
            p.drawPath(path)
        return draw

    def tick(p):
        path = QPainterPath(QPointF(7, 17))
        path.lineTo(13, 23)
        path.lineTo(25, 9)
        p.drawPath(path)

    save("up.png", DIM, chevron(True))
    save("down.png", DIM, chevron(False))
    save("up-off.png", LINE, chevron(True))
    save("down-off.png", LINE, chevron(False))
    save("tick.png", "#ffffff", tick)
    return folder.replace("\\", "/")


def apply(app: QApplication | None = None) -> None:
    """Dark theme for the whole application (Fusion style, so it looks the
    same on every Windows version)."""
    app = app or QApplication.instance()
    if app is None:
        return
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(palette())
    try:
        icons = _icons()
    except OSError:  # no temp folder: no arrows or ticks drawn, the rest still works
        icons = "."
    app.setStyleSheet(STYLE.replace("@ICONS@", icons))
