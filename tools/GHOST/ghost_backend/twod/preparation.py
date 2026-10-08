"""Immutable-input preparation shared for the lifetime of a 2-D run.

Material tables are captured once per run. Small resource forecasts are also
shared. Run-owned workers and the optional bounded inverse cache are closed at
scope exit; nothing is reused across independent runs.

The shared ``MaterialLibrary`` also accumulates the run-level union of
warnings and information.  Each solve reports (and quality-gates) only the
notices it raised itself (``twod.solver._SolveNotices``), so the library's
lists must not be copied into per-solve metadata.
"""
from contextlib import contextmanager
from collections import OrderedDict
from functools import wraps
import json
import hashlib
import sys
from pathlib import Path

from ghost_backend.execution.runtime import ScopedValue

_ACTIVE = ScopedValue('ghost_2d_preparation', None)
_SWEEP_FREQUENCIES = ScopedValue('ghost_2d_mesh_frequencies', None)


class ForecastCache(OrderedDict):
    """Small run forecasts, bounded by retained bytes instead of frequency count."""
    def __init__(self, max_bytes=16 * 1024 * 1024):
        super().__init__()
        self.max_bytes = max_bytes
        self.retained_bytes = 0
        self._sizes = {}

    @staticmethod
    def _size(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, dict):
            result += sum(ForecastCache._size(k, seen) + ForecastCache._size(v, seen)
                          for k, v in value.items())
        elif isinstance(value, (tuple, list)):
            result += sum(ForecastCache._size(v, seen) for v in value)
        return result

    def __setitem__(self, key, value):
        size = self._size((key, value)) + 160  # ordered-map and accounting slots
        if key in self:
            self.pop(key)
        if size > self.max_bytes:
            return
        while self and self.retained_bytes + size > self.max_bytes:
            self.pop(next(iter(self)))
        super().__setitem__(key, value)
        self._sizes[key] = size
        self.retained_bytes += size

    def pop(self, key, *default):
        if key in self:
            self.retained_bytes -= self._sizes.pop(key)
        return super().pop(key, *default)


@contextmanager
def sweep_mesh_scope(frequencies_ghz):
    """Keep the original request's sizing frequencies across frequency-local solves."""
    if _SWEEP_FREQUENCIES.get() is not None:
        yield
    else:
        with _SWEEP_FREQUENCIES.override(tuple(float(f) for f in frequencies_ghz)):
            yield


def mesh_frequencies(frequencies_ghz):
    return _SWEEP_FREQUENCIES.get() or tuple(frequencies_ghz)


@contextmanager
def preparation_scope():
    if _ACTIVE.get() is not None:
        yield _ACTIVE.get()
        return
    with _ACTIVE.override({'materials': {}, 'geometry': {}, 'fingerprints': {}, 'forecasts': ForecastCache(),
                           'hits': 0, 'resources': {}}):
        try:
            yield _ACTIVE.get()
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            # Closing one resource must not strand the other workers/factors.
            # Preserve an active solve/cancellation error; otherwise report the
            # first cleanup failure after every resource has had its turn.
            for resource in list(_ACTIVE.get()['resources'].values()):
                try:
                    resource.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None and primary_error is None:
                raise cleanup_error


def run_resources():
    state = _ACTIVE.get()
    return state.get('resources') if state is not None else None


def prepared_execution(function):
    @wraps(function)
    def call(*args, **kwargs):
        frequencies = kwargs.get('frequencies_ghz', args[1] if len(args) > 1 else ())
        with preparation_scope(), sweep_mesh_scope(frequencies):
            return function(*args, **kwargs)
    return call


def forecast_cache():
    """Run-owned forecast records; admission against free RAM is never cached."""
    state = _ACTIVE.get()
    return state['forecasts'] if state is not None else None


def material_fingerprints(snapshot, base_dir):
    """Verify checkpoint inputs against the material tables captured in this run."""
    from ghost_backend.geometry.io import material_filename_from_row
    from ghost_backend.twod.geometry import _resolve_material_file
    entries = [snapshot.get('ibcs', []) or [], snapshot.get('dielectrics', []) or []]
    fingerprints = {}
    for row in entries[0] + entries[1]:
        name = material_filename_from_row(row)
        if name:
            path = Path(_resolve_material_file(base_dir, name))
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            digest = digest.hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Material file changed while being captured: {}'.format(path))
            fingerprints[str(path)] = digest
    state = _ACTIVE.get()
    key = (base_dir, json.dumps(entries, sort_keys=True))
    if state is not None and key in state['fingerprints'] and state['fingerprints'][key] != fingerprints:
        raise ValueError('Material files changed after run preparation. Start the run again with the updated files.')
    return fingerprints


def prepare_geometry(snapshot, material_base_dir=None, units='inches'):
    # Import lazily to preserve the public solver's validation hooks.
    from ghost_backend.twod import solver
    state = _ACTIVE.get()
    base_dir = solver._material_base_dir_for_snapshot(snapshot, material_base_dir)
    scale = solver._unit_scale_to_meters(units)
    entries = [snapshot.get('ibcs', []) or [], snapshot.get('dielectrics', []) or []]
    material_key = (base_dir, json.dumps(entries, sort_keys=True))
    geometry_key = (material_key, scale, hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode('utf-8')).hexdigest())
    if state is not None and geometry_key in state['geometry']:
        state['hits'] += 1
        return state['geometry'][geometry_key]
    library = state['materials'].get(material_key) if state is not None else None
    if library is None:
        fingerprints = material_fingerprints(snapshot, base_dir)
        library = solver.MaterialLibrary.from_entries(entries[0], entries[1], base_dir=base_dir)
        if material_fingerprints(snapshot, base_dir) != fingerprints:
            raise ValueError('Material files changed during run preparation.')
        if state is not None:
            state['materials'][material_key] = library
            state['fingerprints'][material_key] = fingerprints
    report = solver.validate_geometry_snapshot_for_solver(
        snapshot, base_dir=base_dir, meters_scale=scale, material_library=library)
    result = (base_dir, report, library, float(scale))
    if state is not None:
        state['geometry'][geometry_key] = result
    return result
