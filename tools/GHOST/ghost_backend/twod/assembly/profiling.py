"""Optional assembly-component elapsed sums, including worker-local queries.

Concurrent component times are elapsed sums, not additive whole-solve wall time.
Far integration includes geometric masks, rule selection and kernel contraction;
its deferred scatter is measured separately.
"""
from contextlib import contextmanager
from functools import wraps
import threading
import time
from ghost_backend.execution.runtime import ScopedValue

_PROFILE = ScopedValue('ghost_assembly_components', None)


class AssemblyProfile:
    def __init__(self):
        self.seconds = {}
        self.calls = {}
        self.lock = threading.Lock()

    def record(self, name, elapsed):
        with self.lock:
            self.seconds[name] = self.seconds.get(name, 0.) + elapsed
            self.calls[name] = self.calls.get(name, 0) + 1

    def evidence(self):
        return dict(seconds=dict(self.seconds), calls=dict(self.calls),
                    semantics='elapsed component sums; concurrent or nested work may overlap')


@contextmanager
def profile_scope():
    current = _PROFILE.get()
    if current is not None:
        yield current
    else:
        profile = AssemblyProfile()
        with _PROFILE.override(profile):
            yield profile


def assembly_component(name):
    def decorate(function):
        # Nested tile functions are created in the parent assembly scope, then
        # invoked on ordinary worker threads without inherited ContextVars.
        # Bind that profile explicitly; module-level decorators bind None.
        bound_profile = _PROFILE.get()
        @wraps(function)
        def measured(*args, **kwargs):
            profile = _PROFILE.get() or bound_profile
            if profile is None:
                return function(*args, **kwargs)
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                profile.record(name, time.perf_counter()-started)
        return measured
    return decorate
