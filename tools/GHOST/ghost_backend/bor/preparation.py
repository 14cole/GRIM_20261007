"""Numerical surface contexts owned by one public call and its cap retries.

No global cache retains geometry, material state or frequency-dependent
coefficients. The context is released on success, cancellation and failure.
"""
from contextlib import contextmanager
import hashlib
import pickle

from ghost_backend.execution.runtime import ScopedValue

_CURRENT = ScopedValue('ghost_bor_numerical_preparation', default=None)


class NumericalPreparation:
    def __init__(self):
        self.surfaces = {}
        self.crosses = {}
        self.cursor = 0
        self.reused = 0

    def begin_attempt(self):
        self.cursor = 0
        # A larger modal cap changes the far angular sampling rule. Discard
        # obsolete dense tables before admission/build; checked near bands
        # remain valid for their own Fourier orders and can be extended.
        for surface in self.surfaces.values():
            surface.close_streaming()
            for name in ('_G_table', '_K_tables', '_KI_tables'):
                if hasattr(surface, name):
                    setattr(surface, name, None)
            surface._near_cache.clear()
            surface._angular_shared.clear()
        for cross in self.crosses.values():
            cross.close_streaming()
            cross._G = cross._B = None

    def surface(self, points, frequency, kwargs):
        from ghost_backend.bor.solver import BorPecSolver
        import numpy as np
        key = (self.cursor, hashlib.sha256(pickle.dumps(
            (np.asarray(points, dtype=float), float(frequency), kwargs), protocol=5)).digest())
        self.cursor += 1
        if key in self.surfaces:
            self.reused += 1
            return self.surfaces[key]
        surface = BorPecSolver(points, frequency, **kwargs)
        self.surfaces[key] = surface
        return surface

    def cross(self, observer, source, kwargs):
        from ghost_backend.bor.solver import BorCrossOperators
        key = (id(observer), id(source), pickle.dumps(kwargs, protocol=5))
        if key not in self.crosses:
            self.crosses[key] = BorCrossOperators(observer, source, **kwargs)
        return self.crosses[key]

    def close(self):
        for cross in self.crosses.values():
            cross.close_streaming()
        self.crosses.clear()
        for surface in self.surfaces.values():
            surface.close_streaming()
        self.surfaces.clear()

    def evidence(self):
        return dict(surface_contexts=len(self.surfaces), reused_surface_contexts=self.reused,
                    cross_contexts=len(self.crosses))


@contextmanager
def numerical_preparation():
    owner = NumericalPreparation()
    with _CURRENT.override(owner):
        try:
            yield owner
        finally:
            owner.close()


def prepared_surface(points, frequency, **kwargs):
    owner = _CURRENT.get()
    if owner is None:
        from ghost_backend.bor.solver import BorPecSolver
        return BorPecSolver(points, frequency, **kwargs)
    return owner.surface(points, frequency, kwargs)


def prepared_cross(observer, source, **kwargs):
    owner = _CURRENT.get()
    if owner is None:
        from ghost_backend.bor.solver import BorCrossOperators
        return BorCrossOperators(observer, source, **kwargs)
    return owner.cross(observer, source, kwargs)
