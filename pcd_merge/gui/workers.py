"""Background work: one long task at a time, shown in the status bar along
the bottom of the window with a progress bar, time left and Cancel."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QElapsedTimer, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QWidget

from .common import fmt_secs


class Cancelled(Exception):
    """Raised inside a worker when the user presses Cancel."""


class Worker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.cancel_requested = False
        self._last = ("", -1.0)

    def _progress(self, msg: str, frac: float):
        # Called from the work itself; this is where a Cancel takes effect.
        if self.cancel_requested:
            raise Cancelled()
        # Skip updates the bar would not show, so huge files don't flood the UI.
        if msg != self._last[0] or frac - self._last[1] >= 0.005:
            self._last = (msg, frac)
            self.progress.emit(msg, frac)

    def run(self):
        try:
            result = self.fn(*self.args, progress=self._progress)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # report every failure to the UI
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(result)


class QuietWorker(QObject):
    """Runs ``fn(*args)`` on its own thread; no progress, no cancel. Used for
    previews that are recomputed while the user changes options."""

    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, gen, fn, *args):
        super().__init__()
        self.gen, self.fn, self.args = gen, fn, args

    def run(self):
        try:
            result = self.fn(*self.args)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(self.gen, f"{type(e).__name__}: {e}")
        else:
            self.finished.emit(self.gen, result)


class TaskBar(QWidget):
    """Status text, progress bar and Cancel; runs one task at a time."""

    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._on_done = None
        self._on_fail = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        # The bar comes first, on the left, so it stays in sight even when the
        # window is wider than the screen (the right-hand end is then cut off).
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.progress.setMinimumWidth(160)
        self.progress.setMaximumWidth(320)
        lay.addWidget(self.progress, 1)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setToolTip("Stop the current step. Nothing already loaded is lost.")
        self.b_cancel.clicked.connect(self.cancel)
        self.b_cancel.setVisible(False)
        lay.addWidget(self.b_cancel)
        self.status = QLabel("Ready.")
        self.status.setMinimumWidth(120)
        lay.addWidget(self.status, 2)
        # Ticks while work runs, so the elapsed time keeps moving even during
        # a single long step.
        self._clock = QElapsedTimer()
        self._tick = QTimer(self)
        self._tick.setInterval(500)
        self._tick.timeout.connect(self._update_text)
        self._frac = 0.0

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def set_status(self, msg: str, frac: float | None = None):
        self.status.setText(msg)
        self.status.setToolTip(msg)
        if frac is not None:
            self._frac = max(0.0, min(1.0, frac))
            self.progress.setValue(int(self._frac * 1000))
            self._update_text()

    def _update_text(self):
        if not self.busy:
            self.progress.setFormat("")
            return
        secs = self._clock.elapsed() / 1000.0
        text = f"{self._frac:.0%}   {fmt_secs(secs)}"
        # A remaining-time estimate is only shown once it can be meaningful.
        if 0.1 <= self._frac < 1.0 and secs > 3:
            text += f" (about {fmt_secs(secs * (1 - self._frac) / self._frac)} left)"
        self.progress.setFormat(text)

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel_requested = True
            self.b_cancel.setEnabled(False)
            self.status.setText("Cancelling after the current step…")

    def run(self, fn, args, on_done, label, on_fail=None) -> bool:
        """Start ``fn(*args, progress=...)`` in the background. ``on_done(result)``
        runs on the GUI thread. Returns False if another task is running."""
        if self.busy:
            return False
        self._thread = QThread()
        self._worker = Worker(fn, *args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.set_status)
        # Connect to bound methods (not lambdas) so Qt delivers the results
        # on the GUI thread.
        self._on_done, self._on_fail = on_done, on_fail
        self._worker.finished.connect(self._worker_done)
        self._worker.failed.connect(self._worker_failed)
        self._worker.cancelled.connect(self._worker_cancelled)
        self._clock.start()
        self._tick.start()
        self.b_cancel.setEnabled(True)
        self.b_cancel.setVisible(True)
        self.set_status(label, 0.0)
        self._thread.start()
        self.busy_changed.emit(True)
        return True

    def _worker_done(self, payload):
        self._finish(self._on_done, payload)

    def _worker_failed(self, message):
        self._finish(None, message, failed=True)

    def _worker_cancelled(self):
        self._finish(None, None, cancelled=True)

    def _finish(self, on_done, payload, failed=False, cancelled=False):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self._tick.stop()
        self.b_cancel.setVisible(False)
        secs = self._clock.elapsed() / 1000.0
        self.progress.setValue(0 if failed or cancelled else 1000)
        self.progress.setFormat("" if failed or cancelled else f"Done in {fmt_secs(secs)}")
        on_fail = self._on_fail
        self._on_done = self._on_fail = None
        try:
            if cancelled:
                self.set_status("Cancelled.")
                if on_fail:
                    on_fail(None)
            elif failed:
                self.set_status("Failed: " + payload)
                if on_fail:
                    on_fail(payload)
                else:
                    QMessageBox.critical(self.window(), "PCD Merge", payload)
            else:
                on_done(payload)
        finally:
            self.busy_changed.emit(False)

    def wait(self):
        """Block until the running task ends (used when the window closes)."""
        if self._worker is not None:
            self._worker.cancel_requested = True
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
