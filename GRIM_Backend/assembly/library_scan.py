"""Cancellable library discovery; no directory traversal on the Qt thread."""
import threading
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _Signals(QObject):
    done = Signal(object)
    progress = Signal(int)


class LibraryScan(QRunnable):
    def __init__(self, identifiers, folder, mapping):
        super().__init__()
        self.signals = _Signals()
        self.cancel = threading.Event()
        self.identifiers, self.folder, self.mapping = tuple(identifiers), folder, dict(mapping)

    @Slot()
    def run(self):
        from .workflow import suggest_response_mapping
        try:
            result = suggest_response_mapping(self.identifiers, self.folder, self.mapping,
                cancel_check=self.cancel.is_set, progress_callback=self.signals.progress.emit)
            self.signals.done.emit((self, result, ""))
        except Exception as exc:
            self.signals.done.emit((self, None, str(exc)))

    def start(self):
        QThreadPool.globalInstance().start(self)
