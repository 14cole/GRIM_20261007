"""Private numeric-member caches for bounded native archive loading.

Mapped files are never the input archive. A cache owns no arrays: array base
chains retain its owner, so closing a dataset cannot invalidate surviving views.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import shutil
import tempfile
import weakref
import zipfile

import numpy as np

from GRIM_Backend.datasets.memory import _bounded_grid_selections
from GRIM_Backend.datasets.storage_registry import register_cache, forget_cache

AXES = ("azimuths", "elevations", "frequencies", "polarizations")
KEYS = ("azimuth", "elevation", "frequency", "polarization")
BLOCK_CELLS = 262_144
_pending_cleanup = set()


def _remove_cache(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        _pending_cleanup.discard(path)
        forget_cache(path)
    except OSError:
        # Windows may release the final mapping just after its ndarray owner.
        # Retry at the next cache boundary and on clean interpreter shutdown.
        _pending_cleanup.add(path)
    else:
        _pending_cleanup.discard(path)
        forget_cache(path)


def _retry_cleanup():
    for path in tuple(_pending_cleanup):
        _remove_cache(path)


def _close_cache(path, mappings):
    for mapping in mappings:
        mapping.close()
    _remove_cache(path)


atexit.register(_retry_cleanup)


class _CacheOwner:
    def __init__(self, directory=None):
        _retry_cleanup()
        self.path = tempfile.mkdtemp(prefix="grim-native-", dir=directory)
        self.mappings = []
        self.finalizer = weakref.finalize(self, _close_cache, self.path, self.mappings)
        register_cache(self.path, kind="native", owner=self)


class _CachedArray(np.ndarray):
    """An ndarray whose views keep its private cache directory alive."""
    def __array_finalize__(self, source):
        self._cache_owner = getattr(source, "_cache_owner", None)


class NativeArrays(dict):
    """Internal, privately decoded samples eligible for constructor adoption."""
    preserve_samples = True


def selection_indices(path, report, selection):
    """Read axis coordinates only and return inclusive physical selections."""
    selection = dict(selection or {})
    unknown = set(selection) - set(KEYS)
    if unknown:
        raise ValueError("unknown native selection axis: " + ", ".join(sorted(unknown)))
    with np.load(path, allow_pickle=report.get("allow_legacy_pickle", False)) as data:
        axes = [np.asarray(data[name]) for name in AXES]
        raw_units = data["units"].item() if "units" in data.files else ""
    if isinstance(raw_units, bytes):
        raw_units = raw_units.decode("utf-8")
    try:
        units = json.loads(raw_units) if isinstance(raw_units, str) and raw_units else (raw_units or {})
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} contains corrupt units metadata; refusing to "
            "guess frequency, RCS, or angular conventions") from exc
    if not isinstance(units, dict):
        raise ValueError("native units metadata must be a JSON object")
    indices = []
    for name, axis in zip(KEYS, axes):
        if axis.ndim != 1 or not axis.size:
            raise ValueError(f"invalid {name} axis in native archive")
        requested = selection.get(name)
        if requested is None:
            keep = np.arange(axis.size, dtype=np.intp)
        elif name == "polarization":
            requested = [requested] if isinstance(requested, str) else list(requested)
            wanted = {str(value).strip().casefold() for value in requested}
            labels = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in axis]
            available = {label.strip().casefold() for label in labels}
            if wanted - available:
                raise ValueError("unknown native polarization: " + ", ".join(sorted(wanted - available)))
            keep = np.asarray([i for i, label in enumerate(labels) if label.strip().casefold() in wanted], dtype=np.intp)
        else:
            bounds = np.asarray(requested, dtype=float)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} selection must be a finite inclusive (start, stop) range")
            keep = np.flatnonzero((axis >= bounds[0]) & (axis <= bounds[1]))
        if not keep.size:
            raise ValueError(f"native {name} selection contains no samples")
        indices.append(keep)
    return axes, tuple(indices), units


def _mapped(path, owner, *, mode="r+"):
    source = np.load(path, mmap_mode=mode, allow_pickle=False)
    # The subclass remains in every plain ndarray base chain after asarray.
    # Its owner keeps all private files alive while the mapping is referenced.
    result = source.view(_CachedArray)
    result._cache_owner = owner
    owner.mappings.append(source._mmap)
    return result


def _output_array(owner, index, shape, dtype, storage):
    if storage == "memory":
        return np.empty(shape, dtype=dtype)
    path = Path(owner.path) / f"selected-{index}.npy"
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    del mapped
    return _mapped(path, owner)


def native_component_dtype(layouts):
    """Match the eager constructor's shared real precision for power/phase.

    Promotion must happen before phase wrapping or integer conversion. The
    other component can require float64 even when this member alone would
    normally use float32.
    """
    dtypes = [layouts[key][2] for key in ("rcs_power", "rcs_phase") if key in layouts]
    return np.dtype(np.float64 if any(dtype.kind == "f" and dtype.itemsize > 4 for dtype in dtypes) else np.float32)


def load_native_arrays(path, report, *, storage="disk", selection=None, cache_dir=None):
    """Extract numeric members in bounded blocks and optionally subset them."""
    owner = _CacheOwner(cache_dir)
    arrays = NativeArrays()
    axes, indices, units = selection_indices(path, report, selection)
    source_shape = tuple(axis.size for axis in axes)
    selected_shape = tuple(index.size for index in indices)
    component_dtype = native_component_dtype(report["member_layouts"])
    try:
        with zipfile.ZipFile(path) as archive:
            for number, info in enumerate(archive.infolist()):
                if info.is_dir():
                    continue
                key = info.filename[:-4]
                if key in AXES:
                    # Legacy polarization axes can be explicitly opted-in
                    # object arrays. Apply their coordinates before the
                    # generic object path, exactly like string-array axes.
                    arrays[key] = np.array(axes[AXES.index(key)][indices[AXES.index(key)]], copy=True)
                    continue
                dtype = report["member_layouts"].get(key, ((), False, np.dtype(object)))[2]
                if dtype.hasobject:
                    # Explicit legacy-pickle opt-in remains available, but
                    # objects cannot be mapped and are budgeted as resident.
                    with archive.open(info) as member:
                        value = np.lib.format.read_array(member, allow_pickle=True)
                    if selection is not None and value.ndim >= 4 and value.shape[:4] == source_shape:
                        shape = selected_shape + value.shape[4:]
                        target = np.empty(shape, dtype=value.dtype)
                        if target.size:
                            for tile in _bounded_grid_selections(shape, BLOCK_CELLS):
                                index = tuple(indices[axis][part] for axis, part in enumerate(tile[:4]))
                                index += tuple(np.arange(value.shape[axis])[part] for axis, part in enumerate(tile[4:], 4))
                                target[tile] = value[np.ix_(*index)]
                        value = target
                    arrays[key] = value
                    continue
                member_path = Path(owner.path) / f"member-{number}.npy"
                with archive.open(info) as member, member_path.open("wb") as output:
                    shutil.copyfileobj(member, output, length=1024 * 1024)
                value = _mapped(member_path, owner)
                grid_shaped = value.ndim >= 4 and value.shape[:4] == source_shape
                # Preserve all ancillary cell arrays with the same exact
                # coordinate selection; lower-dimensional metadata is unchanged.
                target_dtype = value.dtype
                if key == "rcs_complex" and value.dtype.kind == "c":
                    target_dtype = np.dtype(np.complex128 if value.dtype.itemsize > 8 else np.complex64)
                elif key in {"rcs_power", "rcs_phase"} and value.dtype.kind in "iuf":
                    target_dtype = component_dtype
                must_copy = storage == "memory" or (selection is not None and grid_shaped) or target_dtype != value.dtype
                if not must_copy:
                    arrays[key] = value
                    continue
                shape = selected_shape + value.shape[4:] if grid_shaped else value.shape
                target = _output_array(owner, number, shape, target_dtype, storage)
                if not shape:
                    target[()] = value[()]
                elif target.size:
                    for tile in _bounded_grid_selections(shape, BLOCK_CELLS):
                        if grid_shaped:
                            index = tuple(indices[axis][part] for axis, part in enumerate(tile[:4]))
                            index += tuple(np.arange(value.shape[axis])[part] for axis, part in enumerate(tile[4:], 4))
                            target[tile] = value[np.ix_(*index)]
                        else:
                            target[tile] = value[tile]
                arrays[key] = target
                # The selected/converted destination is independent. Release
                # the complete extracted member now instead of retaining it
                # for the lifetime of a tiny selected region.
                mapping = value.base._mmap
                del value
                mapping.close()
                owner.mappings.remove(mapping)
                member_path.unlink()
        arrays.cache_path = owner.path
        arrays.axis_units = units
        arrays.selected = selection is not None
        return arrays
    except BaseException:
        arrays.clear()
        raise
