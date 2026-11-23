"""First-person fly-through window: walk around inside a scan.

Controls: W A S D (or arrow keys) to move, mouse to look around, Space / E to
go up, C / Q to go down, Shift to go faster, Ctrl to go slower, mouse wheel to
change speed. Click into the view to look around without holding a button,
Esc to get the mouse back.

Points are drawn with a small OpenGL shader. Which points are drawn is decided
every frame by :func:`flyscene.plan_frame` (level of detail), so even very
large clouds stay smooth. See ``flyscene.py`` for how that works.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QCursor, QGuiApplication, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .flyscene import (VERTEX_DTYPE, FlyCamera, Scene, default_speed, plan_frame,
                       start_camera)

BACKGROUND = (0.07, 0.07, 0.09)
LOOK_DEG_PER_PX = 0.15
# name: (pixels between drawn points, most points per frame)
DETAIL = {
    "Low": (3.0, 1_500_000),
    "Medium": (2.0, 3_000_000),
    "High": (1.4, 6_000_000),
    "Maximum": (1.0, 12_000_000),
}
DEFAULT_DETAIL = "High"
MIN_BUDGET = 250_000

_VERT = """
#version 120
attribute vec3 a_pos;
attribute vec4 a_col;
attribute float a_spacing;
uniform mat4 u_mvp;
uniform mat4 u_mv;
uniform float u_focal;
uniform float u_level;
uniform float u_scale;
uniform float u_max_size;
uniform float u_fog;
uniform vec3 u_bg;
varying vec3 v_col;
void main() {
    vec4 p = vec4(a_pos, 1.0);
    gl_Position = u_mvp * p;
    float d = max(-(u_mv * p).z, 1e-3);
    gl_PointSize = clamp(a_spacing * u_level * u_scale * u_focal / d, 1.0, u_max_size);
    float f = clamp(d / u_fog, 0.0, 1.0);
    v_col = mix(a_col.rgb, u_bg, 0.8 * f * f);
}
"""

_FRAG = """
#version 120
varying vec3 v_col;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    if (dot(q, q) > 1.0) discard;
    gl_FragColor = vec4(v_col, 1.0);
}
"""


class FlyView(QOpenGLWidget):
    """OpenGL view with a first-person camera."""

    def __init__(self, scene: Scene, parent=None, on_status=None):
        super().__init__(parent)
        self.scene = scene
        self.cam: FlyCamera = start_camera(scene)
        self.start = (self.cam.pos.copy(), self.cam.yaw, self.cam.pitch)
        self.speed = default_speed(scene)
        self.follow_look = False
        self.point_scale = 1.6
        self.px_target, self.max_budget = DETAIL[DEFAULT_DETAIL]
        self.budget = self.max_budget
        self.on_status = on_status
        self.on_escape = None  # set when embedded: Esc leaves the walk-through
        self.error: str | None = None

        self._keys: set[int] = set()
        self._locked = False
        self._drag: QPoint | None = None
        self._press_pos: QPoint | None = None
        self._moving = False
        self._last_input = 0.0
        self._last_tick = time.perf_counter()
        self._last_paint = 0.0
        self._fps = 0.0
        self._drawn = 0
        self._prog = None
        self._vbo = None
        self._dirty_colours = True

        fmt = QSurfaceFormat.defaultFormat()
        fmt.setDepthBufferSize(24)
        self.setFormat(fmt)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------- GL

    def initializeGL(self):
        try:
            self._init_gl()
        except Exception as e:  # pragma: no cover - depends on the graphics driver
            self.error = f"{type(e).__name__}: {e}"

    def _init_gl(self):
        from OpenGL import GL

        def compile_shader(src, kind):
            sh = GL.glCreateShader(kind)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sh).decode(errors="replace"))
            return sh

        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, compile_shader(_VERT, GL.GL_VERTEX_SHADER))
        GL.glAttachShader(prog, compile_shader(_FRAG, GL.GL_FRAGMENT_SHADER))
        for i, name in enumerate(("a_pos", "a_col", "a_spacing")):
            GL.glBindAttribLocation(prog, i, name)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode(errors="replace"))
        self._prog = prog
        self._u = {n: GL.glGetUniformLocation(prog, n) for n in (
            "u_mvp", "u_mv", "u_focal", "u_level", "u_scale", "u_max_size", "u_fog", "u_bg")}
        self._vbo = GL.glGenBuffers(1)
        self._upload()

    def _upload(self):
        from OpenGL import GL

        v = self.scene.vertices
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, None, GL.GL_STATIC_DRAW)
        raw = v.view(np.uint8).reshape(-1)
        step = 64 << 20  # upload in pieces so no huge temporary copies are made
        for off in range(0, len(raw), step):
            part = raw[off:off + step]
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, off, len(part), part)
        if GL.glGetError() == GL.GL_OUT_OF_MEMORY:
            raise RuntimeError("the graphics card ran out of memory; try fewer points")
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self._dirty_colours = False

    def recolour(self, mode: str):
        self.scene.set_colours(mode)
        if self._vbo is not None:
            self.makeCurrent()
            try:
                self._upload()
            finally:
                self.doneCurrent()
        self.update()

    def paintGL(self):
        from OpenGL import GL

        now = time.perf_counter()
        dt = now - self._last_paint
        self._last_paint = now
        if 0 < dt < 1:
            self._fps = 0.9 * self._fps + 0.1 / dt if self._fps else 1 / dt
            if self._moving:  # adapt the point budget so moving stays smooth
                if dt > 1 / 28:
                    self.budget = max(MIN_BUDGET, int(self.budget * 0.8))
                elif dt < 1 / 50:
                    self.budget = min(self.max_budget, int(self.budget * 1.15))

        GL.glClearColor(*BACKGROUND, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._prog is None or self.error:
            return
        try:
            self._draw(GL)
        except Exception as e:  # pragma: no cover - depends on the graphics driver
            self.error = f"{type(e).__name__}: {e}"
        self._report()

    def _draw(self, GL):
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        GL.glViewport(0, 0, w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        try:
            GL.glEnable(GL.GL_POINT_SPRITE)  # needed for gl_PointCoord on some drivers
        except Exception:
            pass
        GL.glGetError()  # GL_POINT_SPRITE is invalid on core profiles; ignore

        view = self.cam.view_matrix()
        proj = self.cam.projection_matrix(w / h)
        mvp = proj @ view
        focal = self.cam.focal_px(h)
        budget = self.budget if self._moving else self.max_budget
        plan = plan_frame(self.scene, self.cam.pos, mvp, focal, budget, self.px_target)
        self._drawn = plan.points

        GL.glUseProgram(self._prog)
        u = self._u
        GL.glUniformMatrix4fv(u["u_mvp"], 1, GL.GL_TRUE, mvp.astype(np.float32))
        GL.glUniformMatrix4fv(u["u_mv"], 1, GL.GL_TRUE, view.astype(np.float32))
        GL.glUniform1f(u["u_focal"], float(focal))
        GL.glUniform1f(u["u_scale"], float(self.point_scale))
        GL.glUniform1f(u["u_max_size"], float(max(8.0, 40.0 * dpr)))
        diag = float(np.linalg.norm(self.scene.hi - self.scene.lo)) or 1.0
        GL.glUniform1f(u["u_fog"], float(diag * 1.2))
        GL.glUniform3f(u["u_bg"], *BACKGROUND)

        stride = VERTEX_DTYPE.itemsize
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(VERTEX_DTYPE.fields["pos"][1]))
        GL.glVertexAttribPointer(1, 4, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, stride,
                                 ctypes.c_void_p(VERTEX_DTYPE.fields["rgba"][1]))
        GL.glVertexAttribPointer(2, 1, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 ctypes.c_void_p(VERTEX_DTYPE.fields["spacing"][1]))
        for size_factor, firsts, counts in plan.levels:
            GL.glUniform1f(u["u_level"], size_factor)
            GL.glMultiDrawArrays(GL.GL_POINTS, firsts, counts, len(firsts))
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glDisableVertexAttribArray(2)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)

    def release_gl(self):
        if self._vbo is None:
            return
        from OpenGL import GL

        self.makeCurrent()
        try:
            GL.glDeleteBuffers(1, [self._vbo])
            if self._prog:
                GL.glDeleteProgram(self._prog)
        except Exception:
            pass
        self._vbo = self._prog = None
        self.doneCurrent()

    def _report(self):
        if self.on_status:
            self.on_status(self)

    # ------------------------------------------------------------- movement

    def set_speed(self, v: float):
        self.speed = float(max(1e-3, v))
        self._report()

    def go_to(self, pos, yaw=None, pitch=None):
        self.cam.pos = np.asarray(pos, dtype=np.float64).copy()
        if yaw is not None:
            self.cam.yaw = yaw
        if pitch is not None:
            self.cam.pitch = pitch
        self._poke()

    def go_to_start(self):
        self.go_to(*self.start)

    def _poke(self):
        """Something changed: draw at the moving budget until things settle."""
        self._last_input = time.perf_counter()
        self._moving = True
        if not self._timer.isActive():
            self._last_tick = time.perf_counter()
            self._timer.start()
        self.update()

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        k = self._keys
        fwd = (Qt.Key_W in k or Qt.Key_Up in k) - (Qt.Key_S in k or Qt.Key_Down in k)
        right = (Qt.Key_D in k or Qt.Key_Right in k) - (Qt.Key_A in k or Qt.Key_Left in k)
        up = ((Qt.Key_Space in k or Qt.Key_E in k or Qt.Key_PageUp in k)
              - (Qt.Key_C in k or Qt.Key_Q in k or Qt.Key_PageDown in k))
        if fwd or right or up:
            mods = QGuiApplication.queryKeyboardModifiers()
            v = self.speed
            if mods & Qt.ShiftModifier:
                v *= 4
            if mods & Qt.ControlModifier:
                v *= 0.25
            step = v * dt
            if fwd and right:
                step /= np.sqrt(2)
            self.cam.move(fwd * step, right * step, up * step, self.follow_look)
            self._last_input = now
            self.update()
        elif now - self._last_input > 0.25:
            # Settled: one last frame at full detail, then sleep.
            self._timer.stop()
            self._moving = False
            self.update()

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Escape:
            if self._locked:
                self._unlock()
            elif self.on_escape is not None:
                self.on_escape()
            else:
                self.window().close()
            return
        if e.isAutoRepeat():
            return
        if key == Qt.Key_Home:
            self.go_to_start()
            return
        if Qt.Key_1 <= key <= Qt.Key_9:
            i = key - Qt.Key_1
            if i < len(self.scene.scan_positions):
                self.go_to(self.scene.scan_positions[i][1])
            return
        self._keys.add(key)
        self._poke()

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self._keys.discard(e.key())

    def focusOutEvent(self, e):
        self._keys.clear()
        self._unlock()
        super().focusOutEvent(e)

    def wheelEvent(self, e):
        notches = e.angleDelta().y() / 120.0
        if notches:
            self.set_speed(self.speed * 1.25 ** notches)
            if hasattr(self.window(), "sync_speed"):
                self.window().sync_speed()

    def mousePressEvent(self, e):
        self.setFocus()
        if not self._locked:
            self._drag = e.position().toPoint()
            self._press_pos = self._drag

    def mouseReleaseEvent(self, e):
        if self._locked:
            return
        moved = (self._press_pos is not None
                 and (e.position().toPoint() - self._press_pos).manhattanLength() > 4)
        self._drag = self._press_pos = None
        if not moved and e.button() == Qt.LeftButton:
            self._lock()

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        if self._locked:
            centre = self.rect().center()
            d = p - centre
            if d.isNull():
                return
            QCursor.setPos(self.mapToGlobal(centre))
        elif self._drag is not None:
            d = p - self._drag
            self._drag = p
        else:
            return
        self.cam.look(d.x() * LOOK_DEG_PER_PX, d.y() * LOOK_DEG_PER_PX)
        self._poke()

    def _lock(self):
        self._locked = True
        self.setCursor(Qt.BlankCursor)
        self.grabMouse()
        QCursor.setPos(self.mapToGlobal(self.rect().center()))
        self._report()

    def _unlock(self):
        if not self._locked:
            return
        self._locked = False
        self.releaseMouse()
        self.unsetCursor()
        self._report()

    @property
    def locked(self) -> bool:
        return self._locked


class FlyThroughWindow(QWidget):
    """A :class:`FlyView` with its settings: a window of its own, or, with
    ``on_exit``, a panel embedded in the main window's 3D view (Esc and the
    "Leave" button call ``on_exit``)."""

    def __init__(self, scene: Scene, title: str = "Fly-through", parent=None, on_exit=None):
        super().__init__(parent, Qt.Widget if on_exit else Qt.Window)
        self.embedded = on_exit is not None
        if not self.embedded:
            self.setAttribute(Qt.WA_DeleteOnClose)
            self.setWindowTitle(title)
            self.resize(1280, 800)
        self.scene = scene

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Colour:"))
        self.colour = QComboBox()
        self.colour.addItems(scene.colour_modes)
        self.colour.currentTextChanged.connect(self._recolour)
        top.addWidget(self.colour)
        top.addSpacing(12)
        top.addWidget(QLabel("Detail:"))
        self.detail = QComboBox()
        self.detail.addItems(list(DETAIL))
        self.detail.setCurrentText(DEFAULT_DETAIL)
        self.detail.setToolTip("How many points are drawn. Lower it if moving feels jerky; "
                               "the view also lowers it by itself while you move.")
        self.detail.currentTextChanged.connect(self._set_detail)
        top.addWidget(self.detail)
        top.addSpacing(12)
        top.addWidget(QLabel("Point size:"))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(5, 30)
        self.size.setValue(16)
        self.size.setFixedWidth(110)
        self.size.valueChanged.connect(self._set_size)
        top.addWidget(self.size)
        top.addSpacing(12)
        top.addWidget(QLabel("Speed:"))
        self.speed = QDoubleSpinBox()
        self.speed.setDecimals(2)
        self.speed.setRange(0.01, 10000.0)
        self.speed.setSuffix(" /s")
        self.speed.setToolTip("Walking speed in scan units (usually metres) per second. "
                              "Mouse wheel changes it too; hold Shift to go 4x faster, "
                              "Ctrl for 4x slower.")
        self.speed.valueChanged.connect(lambda v: self.view.set_speed(v))
        top.addWidget(self.speed)
        self.follow = QCheckBox("Fly where I look")
        self.follow.setToolTip("Off: W/S move level, like walking, and Space/C go up and "
                               "down. On: W flies in the direction you are looking.")
        self.follow.toggled.connect(self._set_follow)
        top.addWidget(self.follow)
        top.addStretch(1)
        self.jump = QComboBox()
        self.jump.setToolTip("Jump to where a scan was taken (keys 1-9 do the same).")
        self.jump.addItem("Jump to scan position…")
        for i, (name, _) in enumerate(scene.scan_positions):
            self.jump.addItem(f"{i + 1}: {name}" if i < 9 else name)
        self.jump.setEnabled(bool(scene.scan_positions))
        self.jump.activated.connect(self._jump)
        top.addWidget(self.jump)
        b_start = QPushButton("Back to start")
        b_start.setToolTip("Home key")
        b_start.clicked.connect(lambda: (self.view.go_to_start(), self.view.setFocus()))
        top.addWidget(b_start)
        if self.embedded:
            b_leave = QPushButton("Leave walk-through")
            b_leave.setToolTip("Back to the orbit view (Esc)")
            b_leave.clicked.connect(on_exit)
            top.addWidget(b_leave)
        lay.addLayout(top)

        self.view = FlyView(scene, self, on_status=self._status)
        if self.embedded:
            self.view.on_escape = on_exit
            lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)

        bottom = QHBoxLayout()
        self.help = QLabel()
        self.help.setStyleSheet("color: #888;")
        bottom.addWidget(self.help, 1)
        self.info = QLabel()
        self.info.setStyleSheet("color: #888;")
        bottom.addWidget(self.info)
        lay.addLayout(bottom)

        self.speed.setValue(self.view.speed)
        self._status(self.view)
        QTimer.singleShot(0, self.view.setFocus)
        QTimer.singleShot(500, self._check_error)

    def sync_speed(self):
        self.speed.blockSignals(True)
        self.speed.setValue(self.view.speed)
        self.speed.blockSignals(False)
        self.view.setFocus()

    def _status(self, view: FlyView):
        if view.locked:
            self.help.setText("W A S D: move   Mouse: look   Space/E: up   C/Q: down   "
                              "Shift: faster   Wheel: speed   Esc: release the mouse")
        else:
            self.help.setText("Click the view to look around with the mouse (or drag)   "
                              "W A S D: move   Space/C: up/down   Home: start   Esc: "
                              + ("leave" if self.embedded else "close"))
        shown = len(self.scene)
        txt = f"{view._drawn / 1e6:.1f} M of {shown / 1e6:.1f} M points drawn"
        if self.scene.total_points > shown:
            txt += f" ({self.scene.total_points / 1e6:.0f} M in the scans)"
        if view._moving and view._fps:
            txt += f"   {view._fps:.0f} fps"
        self.info.setText(txt)

    def _recolour(self, mode):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.view.recolour(mode)
        finally:
            QApplication.restoreOverrideCursor()
        self.view.setFocus()

    def _set_detail(self, name):
        self.view.px_target, self.view.max_budget = DETAIL[name]
        self.view.budget = self.view.max_budget
        self.view.update()
        self.view.setFocus()

    def _set_size(self, v):
        self.view.point_scale = v / 10.0
        self.view.update()

    def _set_follow(self, on):
        self.view.follow_look = on
        self.view.setFocus()

    def _jump(self, i):
        if i > 0:
            self.view.go_to(self.scene.scan_positions[i - 1][1])
        self.jump.setCurrentIndex(0)
        self.view.setFocus()

    def _check_error(self):
        if self.view.error:
            QMessageBox.critical(self, "Fly-through",
                                 "The 3D view could not be drawn on this computer's graphics "
                                 f"card.\n\n{self.view.error}")
            if self.embedded:
                self.view.on_escape()
            else:
                self.close()

    def closeEvent(self, e):
        self.view.release_gl()
        super().closeEvent(e)


def main(argv=None) -> int:  # pragma: no cover - manual use
    """``python -m pcd_merge.flythrough scan.pcd [more.pcd ...]``"""
    import sys

    from .flyscene import build_scene
    from .pcd_io import read_pcd

    args = sys.argv[1:] if argv is None else argv
    app = QApplication.instance() or QApplication(sys.argv)
    clouds = [read_pcd(p) for p in args]
    scene = build_scene(clouds, progress=lambda m, f: print(f"{f:4.0%} {m}"))
    win = FlyThroughWindow(scene, "Fly-through: " + ", ".join(c.name for c in clouds))
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
