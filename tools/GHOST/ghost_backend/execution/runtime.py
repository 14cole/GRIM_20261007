"""Small runtime adapters for the Python 3.6+ headless GHOST backend."""
from contextlib import contextmanager
import sys
import threading

if sys.version_info < (3, 7):
    from ghost_backend.execution._dataclasses import asdict, dataclass, field, replace
else:
    from dataclasses import asdict, dataclass, field, replace

try:
    from contextvars import ContextVar
except ImportError:
    ContextVar = None


class ScopedValue:
    """A solver setting restored after nested calls, exceptions, and threads."""

    def __init__(self, name, default):
        self._default = default
        self._context = ContextVar(name, default=default) if ContextVar else None
        self._local = threading.local() if self._context is None else None

    def get(self):
        if self._context is not None:
            return self._context.get()
        return getattr(self._local, 'value', self._default)

    @contextmanager
    def override(self, value):
        if self._context is not None:
            token = self._context.set(value)
            try:
                yield
            finally:
                self._context.reset(token)
        else:
            previous = self.get()
            self._local.value = value
            try:
                yield
            finally:
                self._local.value = previous


# Spawned numerical workers that run one BLAS thread each (the BoR near-pair and
# compressed-tile process pools). OpenBLAS reserves its per-thread buffers when
# numpy/scipy are imported -- in the worker's bootstrap, before any initializer
# can limit the pool -- so an unpinned worker inherits the host thread count:
# NumPy's and SciPy's OpenBLAS pools committed about 1.05 GB per worker on an
# 8-core/16-thread host while the worker touched ~50 MB (15 near workers held
# ~16 GB of commit). With this environment a worker commits ~40 MB. A spawned
# child copies the parent's environment when it is created, so the values stay
# set while a pool can start processes; the count lets pools overlap.
SINGLE_THREAD_WORKER_ENVIRONMENT = (
    ('OPENBLAS_NUM_THREADS', '1'),
    ('OMP_NUM_THREADS', '1'),
    ('MKL_NUM_THREADS', '1'),
)
_WORKER_ENVIRONMENT_LOCK = threading.Lock()
_WORKER_ENVIRONMENT_STATE = {'count': 0, 'saved': None}


def pin_worker_environment():
    """Set the single-thread worker environment (reference counted).

    Pair every call with :func:`release_worker_environment`. The parent's own
    BLAS and OpenMP runtimes are already initialized and do not re-read these
    variables; only processes spawned while pinned see them.
    """
    import os
    with _WORKER_ENVIRONMENT_LOCK:
        state = _WORKER_ENVIRONMENT_STATE
        if state['count'] == 0:
            state['saved'] = {name: os.environ.get(name)
                              for name, _ in SINGLE_THREAD_WORKER_ENVIRONMENT}
            for name, value in SINGLE_THREAD_WORKER_ENVIRONMENT:
                os.environ[name] = value
        state['count'] += 1


def release_worker_environment():
    """Undo one :func:`pin_worker_environment`; the last release restores the values."""
    import os
    with _WORKER_ENVIRONMENT_LOCK:
        state = _WORKER_ENVIRONMENT_STATE
        if state['count'] <= 0:
            return
        state['count'] -= 1
        if state['count'] == 0:
            for name, value in (state['saved'] or {}).items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            state['saved'] = None


@contextmanager
def single_thread_worker_environment():
    """Scope in which spawned workers start with one BLAS/OpenMP thread."""
    pin_worker_environment()
    try:
        yield
    finally:
        release_worker_environment()


def unlink_if_exists(path):
    """Remove a file, ignoring only absence (Path.unlink on Python 3.6)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_text_lf(path, text):
    """Write UTF-8 text with LF newlines on every supported interpreter."""
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        return stream.write(text)
