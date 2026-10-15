"""Captured, engineer-controlled working-memory policy for ISAR workers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import operator
import os

from GRIM_Backend.datasets.memory import _available_import_memory_bytes

_POLICY = ContextVar("grim_isar_compute_memory", default=None)
FALLBACK_BYTES = 2 * 1024**3
BATCH_BYTES = 8 * 1024**2


def resolve_memory_limit(limit_bytes=None, *, use_environment=True):
    """Use half currently available RAM unless explicitly configured.

    An older environment setting remains supported for headless callers. The
    GUI passes its resolved budget, so it is stable for the entire worker job.
    """
    if limit_bytes is None:
        configured = os.environ.get("GRIM_ISAR_WORKING_SET_MB") if use_environment else None
        if configured:
            try:
                limit_bytes = int(configured) * 1024**2
            except (TypeError, ValueError):
                pass
        if limit_bytes is None:
            available = _available_import_memory_bytes()
            return max(1, int(available * .5)) if available else FALLBACK_BYTES
    if isinstance(limit_bytes, bool):
        raise ValueError("ISAR memory limit must be a positive integer byte count")
    try:
        limit = operator.index(limit_bytes)
    except TypeError as exc:
        raise ValueError("ISAR memory limit must be a positive integer byte count") from exc
    if limit <= 0:
        raise ValueError("ISAR memory limit must be a positive integer byte count")
    return limit


@contextmanager
def compute_memory_policy(limit_bytes=None, *, allow_exceed=False):
    if not isinstance(allow_exceed, bool):
        raise ValueError("ISAR allow-above-budget setting must be True or False")
    token = _POLICY.set((resolve_memory_limit(limit_bytes), allow_exceed))
    try:
        yield
    finally:
        _POLICY.reset(token)


def current_memory_policy():
    return _POLICY.get()
