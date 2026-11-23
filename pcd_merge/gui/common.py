"""Small things shared by the panels: colours, formatting, the legend."""

from __future__ import annotations

from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap

from ..quality import colour_stops

PREVIEW_POINTS = 400_000  # total points drawn in the orbit view
# points the clean-up would remove (white: unlike any cloud colour)
REMOVED_COLOUR = (1.0, 1.0, 1.0, 1.0)
PALETTE = [
    (0.95, 0.35, 0.25), (0.25, 0.65, 0.95), (0.35, 0.85, 0.35), (0.95, 0.80, 0.20),
    (0.75, 0.40, 0.90), (0.20, 0.85, 0.80), (0.95, 0.55, 0.75), (0.65, 0.65, 0.65),
]
GRADE_COLOURS = {"Good": "#2e9e44", "Fair": "#d08a00", "Poor": "#d0342c"}
HINT_STYLE = "color: #888;"


def cloud_colour(i: int) -> tuple[float, float, float]:
    return PALETTE[i % len(PALETTE)]


def swatch_icon(i: int, size: int = 14) -> QIcon:
    r, g, b = cloud_colour(i)
    pm = QPixmap(size, size)
    pm.fill(QColor(int(r * 255), int(g * 255), int(b * 255)))
    return QIcon(pm)


def legend_pixmap(width=160, height=12) -> QPixmap:
    pm = QPixmap(width, height)
    grad = QLinearGradient(0, 0, width, 0)
    for pos, (r, g, b) in colour_stops():
        grad.setColorAt(pos, QColor.fromRgbF(r, g, b))
    p = QPainter(pm)
    p.fillRect(0, 0, width, height, grad)
    p.end()
    return pm


def fmt_secs(secs: float) -> str:
    secs = int(round(secs))
    return f"{secs // 60}:{secs % 60:02d}" if secs >= 60 else f"{secs} s"
