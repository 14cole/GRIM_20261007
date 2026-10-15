"""Bounded binary-reader destinations using the shared managed allocator."""
from contextlib import contextmanager
import sys

import numpy as np

from GRIM_Backend.datasets.load_policy import import_memory_budget
from GRIM_Backend.datasets.memory import _checked_dense_import_allocation
from GRIM_Backend.datasets.result_storage import operation_result_storage

BLOCK_CELLS = 32_768


@contextmanager
def binary_import_storage(storage, *, cache_dir=None, max_output_bytes=None):
    if storage not in {"memory", "disk"}:
        raise ValueError("storage must be memory or disk")
    with operation_result_storage(storage, cache_dir=cache_dir):
        with import_memory_budget(max_output_bytes):
            yield


def binary_import_allocation(shape, dtype, *, storage, source,
                             scratch_bytes=0, retained_metadata_bytes=0, max_output_bytes=None):
    """Keep address/shape validation while budgeting mapped outputs as disk."""
    scratch_bytes = int(scratch_bytes)
    if storage == "memory":
        report = _checked_dense_import_allocation(
            shape, (dtype,), source=source, resident_bytes=scratch_bytes,
            max_output_bytes=max_output_bytes)
    else:
        report = _checked_dense_import_allocation(
            shape, (dtype,), source=source, max_output_bytes=sys.maxsize)
        # The mapped address range is not a retained heap allocation. The binary
        # source, decoded temporary samples, and validation masks are block-sized.
        working = _checked_dense_import_allocation(
            (max(1, scratch_bytes),), (np.uint8,), source=source,
            max_output_bytes=max_output_bytes)
        report.update(peak_bytes=working["peak_bytes"], resident_bytes=scratch_bytes,
                      limit_bytes=working["limit_bytes"])
    report["retained_bytes"] = int(retained_metadata_bytes) + (report["dense_bytes"] if storage == "memory" else 0)
    return report


def loaded_binary_grid(grid, storage):
    grid.load_storage = storage
    # Loading has its own recorder path, separate from derived operations.
    grid.__dict__.pop("_operation_storage_options", None)
    return grid


def estimate_binary_import(path, *, storage="memory"):
    """Discover PIO/SS layout and bounded working memory without decoding cells."""
    from GRIM_Backend.datasets.grid import RcsGrid
    lower = str(path).lower()
    reader = (RcsGrid._load_pio_bounded if lower.endswith((".pio", ".cmplx_di")) else
              RcsGrid._load_ss_bounded if lower.endswith(".ss") else None)
    if reader is None:
        return None
    if storage not in {"memory", "disk"}:
        raise ValueError("storage must be memory or disk")
    report = reader(path, storage=storage, max_output_bytes=sys.maxsize, _estimate_only=True)
    return dict(report, estimated_peak_bytes=report["peak_bytes"],
                source_format="PIO" if lower.endswith((".pio", ".cmplx_di")) else "SS")
