"""Run-scoped, measured dense-factor choices supplied by execution planning."""
from contextlib import contextmanager
from ghost_backend.execution.runtime import ScopedValue

_CHOICES=ScopedValue('ghost_measured_factor_choices',None)


def _validated(choices):
    values={}
    for size,variant in (choices or {}).items():
        n=int(size)
        if n<=0 or variant not in ('lu','hodlr'):
            raise ValueError('Measured factor choices require positive sizes and lu/hodlr variants.')
        values[n]=variant
    return values


@contextmanager
def scope(choices):
    values=_validated(choices)
    with _CHOICES.override(values):yield


def install(choices):
    current=_CHOICES.get()
    if current is None:raise RuntimeError('Measured factor choices require an execution scope.')
    values=_validated(choices)
    current.clear();current.update(values)


def chosen(n):
    return (_CHOICES.get() or {}).get(int(n))
