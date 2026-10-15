"""Latest-request-wins numerical preparation; publication stays on Qt's thread."""
from __future__ import annotations

import threading
from PySide6.QtCore import QObject, QThreadPool, Signal, Slot

from GRIM_Backend.plotting.requests import PlotCancelled


def dispose_result(result):
    close = getattr(result, "close", None)
    if callable(close):
        close()


class PlotJobRunner(QObject):
    completed = Signal(object, object, object)
    publication_failed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = None
        self._pending = None
        self._closed = False
        self.completed.connect(self._completed)

    @property
    def busy(self):
        return self._running is not None

    def cancel(self):
        if self._running is not None:
            self._running[0].set()
        self._pending = None

    def close(self):
        self._closed = True
        self.cancel()

    def submit(self, compute, publish):
        if self._closed:
            raise RuntimeError("This plot session has closed.")
        self.cancel()
        job = (threading.Event(), compute, publish)
        if self._running is not None:
            self._pending = job
        else:
            self._launch(job)

    def _launch(self, job):
        self._running = job
        event, compute, _publish = job

        def work():
            result, error = None, None
            try:
                result = compute(event.is_set)
            except PlotCancelled:
                pass
            except Exception as exc:
                error = exc
            try:
                self.completed.emit(job, result, error)
            except RuntimeError:
                # The owning window may have closed during a NumPy call.
                dispose_result(result)

        QThreadPool.globalInstance().start(work)

    @Slot(object, object, object)
    def _completed(self, job, result, error):
        self._running = None
        event, _compute, publish = job
        pending, self._pending = self._pending, None
        try:
            if event.is_set() or self._closed:
                dispose_result(result)
            else:
                publish(result, error)
        except Exception as exc:
            dispose_result(result)
            self.publication_failed.emit(exc)
        finally:
            # A publication callback can synchronously request another plot.
            # Never overwrite that newer running request with a saved pending one.
            if pending is not None and not self._closed and self._running is None and self._pending is None:
                self._launch(pending)
