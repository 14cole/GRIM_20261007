"""Per-load memory policy shared by file readers and background workers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import operator

_IMPORT_MEMORY_BUDGET = ContextVar('grim_import_memory_budget', default=None)


def checked_import_budget(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('The load memory budget must be a positive byte count')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError('The load memory budget must be a positive byte count') from exc
    if result <= 0:
        raise ValueError('The load memory budget must be a positive byte count')
    return int(result)


def current_import_memory_budget():
    return _IMPORT_MEMORY_BUDGET.get()


@contextmanager
def import_memory_budget(value):
    """Apply a captured budget in this thread only, including nested readers."""
    value = checked_import_budget(value)
    token = _IMPORT_MEMORY_BUDGET.set(value if value is not None else current_import_memory_budget())
    try:
        yield
    finally:
        _IMPORT_MEMORY_BUDGET.reset(token)
