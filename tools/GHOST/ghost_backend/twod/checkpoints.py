"""Atomic, input-verified frequency checkpoints for desktop 2-D sweeps.

Archives contain column arrays and JSON only (never executable pickle). A
completed frequency is reusable only with matching inputs, material contents,
backend source, precision, quality settings, and an intact content digest.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import time
import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=lambda v: v.item())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(arguments, options, precision, certified, solver_kind='2d'):
    from ghost_backend.twod.geometry import _material_base_dir_for_snapshot
    from ghost_backend.twod.preparation import material_fingerprints
    snapshot = arguments['geometry_snapshot']
    base = _material_base_dir_for_snapshot(snapshot, arguments.get('material_base_dir'))
    files = material_fingerprints(snapshot, base)
    source = hashlib.sha256()
    backend = Path(__file__).resolve().parents[1]
    folders = ('twod', 'compressed', 'linalg', 'execution', 'runs', 'geometry')
    if solver_kind == 'bor':
        folders += ('bor',)
    for folder in folders:
        for path in sorted(p for p in (backend / folder).rglob('*')
                           if p.is_file() and p.suffix in ('.py','.f','.f90','.c','.dll','.so','.dylib')):
            source.update(str(path.relative_to(backend)).replace('\\', '/').encode('utf-8'))
            source.update(path.read_bytes())
    inputs = {key: value for key,value in arguments.items()
              if key not in ('progress_callback', 'abort_event', 'frequencies_ghz')}
    # With a shared mesh reference, the sweep controls its conservative scale.
    if arguments.get('mesh_reference_ghz') is not None:
        inputs['mesh_frequencies_ghz'] = arguments['frequencies_ghz']
    return hashlib.sha256(_json(dict(inputs=inputs, options=options, precision=precision,
        certified=certified, solver_kind=solver_kind, materials=files,
        source=source.hexdigest(), schema=2)).encode('utf-8')).hexdigest()


def missing_frequencies(arguments, directory, options, precision, certified,
                        *, solver_kind='2d', checkpoint=None):
    """Probe matching checkpoints before spending work on solve forecasts.

    This read-only hint never authorizes reuse: the execution path still
    verifies availability and the full reader checks integrity before export.
    A lost or changed checkpoint therefore returns to normal solve admission.
    """
    if checkpoint is not None:
        checkpoint()
    identity = input_identity(arguments, options, precision, certified, solver_kind)
    store = FrequencyCheckpoints(directory, identity, certified, create=False)
    missing = []
    for frequency in arguments['frequencies_ghz']:
        if checkpoint is not None:
            checkpoint()
        if not store.available(frequency):
            missing.append(frequency)
    return missing


class FrequencyCheckpoints:
    def __init__(self, directory, identity, certified, *, create=True):
        self.directory = Path(directory) / identity
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.certified = bool(certified)

    def _path(self, frequency):
        key = hashlib.sha256(float(frequency).hex().encode('ascii')).hexdigest()[:24]
        return self.directory / (key + '.npz')

    def save(self, frequency, result):
        if self.certified and result.get('metadata', {}).get('mesh_convergence_certified') is not True:
            raise ValueError('An uncertified frequency cannot enter a certified checkpoint.')
        rows = result['samples']
        from ghost_backend.twod.samples import sample_columns
        packed = sample_columns(rows)
        columns = sorted(packed) if packed is not None else sorted(set(key for row in rows for key in row))
        arrays, encodings = {}, []
        for index,key in enumerate(columns):
            if packed is not None:
                arrays['c' + str(index)] = packed[key]
                arrays['p' + str(index)] = np.ones(len(rows), dtype=bool)
                encodings.append('scalar')
                continue
            values = [row.get(key) for row in rows]
            # Store numeric solver fields directly; JSON is only a fallback for
            # heterogeneous extension fields. No object-dtype arrays are used.
            if all(type(value) in (int, float, bool, str) for value in values) and len({type(value) for value in values}) == 1:
                arrays['c' + str(index)] = np.asarray(values)
                encodings.append('scalar')
            else:
                arrays['c' + str(index)] = np.asarray([_json(value) for value in values])
                encodings.append('json')
            arrays['p' + str(index)] = np.asarray([key in row for row in rows], dtype=bool)
        header = {key: value for key,value in result.items() if key not in ('samples', 'co_solved_samples')}
        record = dict(identity=self.identity, frequency=float(frequency), certified=self.certified,
                      columns=columns, encodings=encodings, result=header, co_solved='co_solved_samples' in result)
        if result.get('solver') == 'bor_mom_rcs' and record['co_solved']:
            # BoR public per-channel rows omit the combined view's labels.
            record['channel_labels'] = {pol: [key for key in ('polarization', 'polarization_internal')
                                               if len(values) and key in values[0]]
                                        for pol, values in result['co_solved_samples'].items()}
        arrays['header'] = np.frombuffer(_json(record).encode('utf-8'), dtype=np.uint8)
        path = self._path(frequency)
        handle, temporary = tempfile.mkstemp(prefix='frequency-', suffix='.npz', dir=str(self.directory))
        try:
            with os.fdopen(handle, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            digest = _sha(temporary)
            os.replace(temporary, str(path))
            marker = dict(identity=self.identity, frequency=float(frequency), sha256=digest)
            handle, manifest = tempfile.mkstemp(prefix='digest-', suffix='.json', dir=str(self.directory))
            try:
                with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                    stream.write(_json(marker))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest, str(path.with_suffix('.json')))
            finally:
                if os.path.exists(manifest):
                    os.unlink(manifest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, frequency):
        path = self._path(frequency)
        try:
            marker = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if marker != dict(identity=self.identity, frequency=float(frequency), sha256=_sha(path)):
                return None
            with np.load(str(path), allow_pickle=False) as data:
                record = json.loads(data['header'].tobytes().decode('utf-8'))
                if (record['identity'] != self.identity or record['frequency'] != float(frequency)
                        or record['certified'] != self.certified):
                    return None
                from ghost_backend.twod.samples import checkpoint_samples, SampleTable, SampleSelection
                rows = checkpoint_samples(data, record)
                for index,key in (() if rows is not None else enumerate(record['columns'])):
                    values, present = data['c'+str(index)], data['p'+str(index)]
                    if rows is None:
                        rows = [{} for _ in values]
                    if len(values) != len(rows) or len(present) != len(rows):
                        return None
                    for row,value,exists in zip(rows, values, present):
                        if exists:
                            row[key] = value.item() if record['encodings'][index] == 'scalar' else json.loads(str(value))
            result = record['result']
            result['samples'] = rows or []
            if self.certified and result.get('metadata', {}).get('mesh_convergence_certified') is not True:
                return None
            if (np.any(rows.data[:len(rows), 0] != float(frequency)) if isinstance(rows, SampleTable)
                    else any(float(row['frequency_ghz']) != float(frequency) for row in result['samples'])):
                return None
            if record['co_solved']:
                if isinstance(rows, SampleTable):
                    labels = rows.labels.get('polarization')
                    if labels is None:
                        return None
                    channel_rows = {}
                    for pol in ('VV', 'HH'):
                        table = rows
                        if 'channel_labels' in record:
                            allowed = record['channel_labels'][pol]
                            table = SampleTable(rows.data, len(rows), fields=rows.fields,
                                labels={key: value for key, value in rows.labels.items() if key in allowed})
                        channel_rows[pol] = SampleSelection(table, np.flatnonzero(labels == pol))
                    result['co_solved_samples'] = channel_rows
                else:
                    result['co_solved_samples'] = {pol: [row for row in result['samples'] if row['polarization'] == pol]
                                                   for pol in ('VV', 'HH')}
                    if 'channel_labels' in record:
                        for pol, values in result['co_solved_samples'].items():
                            result['co_solved_samples'][pol] = [
                                {key: value for key, value in row.items()
                                 if key not in ('polarization', 'polarization_internal')
                                 or key in record['channel_labels'][pol]} for row in values]
            return result
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            return None

    def available(self, frequency):
        """Verify bytes and the result header without decoding sample columns.

        The full reader still validates the columns immediately before export.
        This keeps solve-time memory bounded to one frequency and eliminates
        the former second decompression of every reused result.
        """
        path = self._path(frequency)
        try:
            marker = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if marker != dict(identity=self.identity, frequency=float(frequency), sha256=_sha(path)):
                return False
            with np.load(str(path), allow_pickle=False) as data:
                record = json.loads(data['header'].tobytes().decode('utf-8'))
                required = {'header'} | {'{}{}'.format(prefix, index)
                    for index in range(len(record['columns'])) for prefix in ('c', 'p')}
                return (record['identity'] == self.identity
                    and record['frequency'] == float(frequency)
                    and record['certified'] == self.certified
                    and len(record['encodings']) == len(record['columns'])
                    and required.issubset(data.files)
                    and (not self.certified or record['result'].get('metadata', {}).get(
                        'mesh_convergence_certified') is True))
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            return False


def _retained_bytes(value, seen=None):
    """Conservative owned object/array size for exceptional unsaved results."""
    import sys
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return total + (_retained_bytes(value.base, seen) if value.base is not None else 0)
    if isinstance(value, dict):
        return total + sum(_retained_bytes(k, seen) + _retained_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return total + sum(_retained_bytes(v, seen) for v in value)
    if type(value).__module__ == 'ghost_backend.twod.samples':
        return total + _retained_bytes(vars(value), seen)
    return total


def run_checkpointed(solve, arguments, directory, options, precision, certified,
                     *, solver_kind='2d', merge=None, frequency_workers=1):
    from ghost_backend.twod.preparation import preparation_scope, sweep_mesh_scope
    # Direct API callers need the same run-owned workers, material snapshot,
    # and bounded inverse cache as the desktop's outer preparation scope.
    # Nested scopes share ownership; the outermost scope alone closes resources.
    with preparation_scope(), sweep_mesh_scope(arguments['frequencies_ghz']):
        return _run_checkpointed(solve, arguments, directory, options, precision,
                                 certified, solver_kind=solver_kind, merge=merge,
                                 frequency_workers=frequency_workers)


def _run_checkpointed(solve, arguments, directory, options, precision, certified,
                      *, solver_kind, merge, frequency_workers=1):
    from ghost_backend.twod.solver import _merge_frequency_results
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    from ghost_backend.execution.options import _MEMORY_ALLOCATION
    from contextlib import ExitStack
    started = time.perf_counter()
    profiles = []
    frequencies = list(arguments['frequencies_ghz'])
    if solver_kind != 'bor' and len(set(frequencies)) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported in a co-polarized result grid.')
    identity = input_identity(arguments, options, precision, certified, solver_kind=solver_kind)
    try:
        store = FrequencyCheckpoints(directory, identity, certified)
    except OSError as exc:
        result = solve(**arguments)
        warning = 'Frequency checkpoints unavailable; completed result remains available: ' + str(exc)
        metadata = result.setdefault('metadata', {})
        metadata.setdefault('warnings', []).append(warning)
        metadata['warning_count'] = len(metadata['warnings'])
        metadata['frequency_checkpoints'] = dict(directory=str(directory), completed=len(frequencies),
            persisted=0, reused=0, input_sha256=identity, write_warnings=[warning])
        return result
    progress = arguments.get('progress_callback')
    abort = arguments.get('abort_event')
    reused = 0
    unsaved, checkpoint_warnings, completed_frequencies = {}, [], []
    retained = 0
    # Exceptional fallback storage is explicitly subtracted from subsequent
    # solve admission. Large failures return a clearly identified partial
    # result instead of silently exceeding the solve's RAM reservation.
    initial_limit = _solve_memory_limit_gb()
    fallback_limit = min(64 * 1024**2, max(0., initial_limit) * 1024**3 * .02)
    parallel = None
    if solver_kind in ('2d', 'bor') and frequency_workers != 1:
        from ghost_backend.execution.frequency_sweep import compute_parallel
        parallel = compute_parallel(solve, arguments, directory, store, options, precision,
                                    certified, frequency_workers, initial_limit)
        if parallel is not None:
            completed_frequencies = parallel['completed']
            unsaved, checkpoint_warnings = parallel['unsaved'], parallel['warnings']
            profiles, reused = parallel['profiles'], parallel['reused']
            retained = _retained_bytes(unsaved)
    for index,frequency in enumerate(frequencies):
        if abort is not None and abort.is_set():
            raise InterruptedError('Solve canceled; completed frequency checkpoints were retained.')
        if parallel is not None and frequency in completed_frequencies:
            continue
        if retained > fallback_limit:
            break
        if frequency in unsaved:
            pass  # Preserve BoR's duplicate-frequency request semantics.
        elif store.available(frequency):
            reused += 1
        else:
            progress_index = len(completed_frequencies) if parallel is not None else index
            def report(done, total, message):
                if progress:
                    progress(progress_index*1000 + int(1000*done/max(total,1)), len(frequencies)*1000, message)
            scope = (_MEMORY_ALLOCATION.override(max(0., _solve_memory_limit_gb() - retained / 1024**3))
                     if retained else ExitStack())
            memory_failure = None
            try:
                with scope:
                    result = solve(**dict(arguments, frequencies_ghz=[frequency], progress_callback=report))
            except MemoryError as exc:
                if not unsaved:
                    raise
                memory_failure = str(exc)
            if memory_failure is not None:
                checkpoint_warnings.append('Further solving stopped while preserving unsaved completed '
                    'frequencies: ' + memory_failure)
                break
            if result.get('metadata', {}).get('runtime_profile'):
                profiles.append(result['metadata']['runtime_profile'])
            try:
                store.save(frequency, result)
            except OSError as exc:
                warning = 'Frequency {:g} GHz computed but checkpoint could not be saved: {}'.format(frequency, exc)
                checkpoint_warnings.append(warning)
                unsaved[frequency] = result
                retained = _retained_bytes(unsaved)
            del result
        completed_frequencies.append(frequency)
        if progress:
            completed_count = len(completed_frequencies)
            progress(completed_count*1000, len(frequencies)*1000,
                      'Frequency {:g} GHz {}; {} of {} complete ({} reused).'.format(
                          frequency, 'computed; checkpoint unavailable' if frequency in unsaved else 'saved',
                          completed_count,len(frequencies),reused))
        if retained > fallback_limit:
            break
    if abort is not None and abort.is_set():
        raise InterruptedError('Solve canceled; completed frequency checkpoints were retained.')
    unsaved_count = sum(frequency in unsaved for frequency in completed_frequencies)
    def completed():
        for frequency in completed_frequencies:
            value = unsaved.get(frequency)
            if value is None:
                value = store.load(frequency)
            if value is None:
                raise IOError('A completed frequency checkpoint changed before result export. Rerun to recompute it.')
            yield value
    # Parallel completion order is independent of the requested output order.
    if parallel is not None:
        completed_frequencies = [f for f in frequencies if f in completed_frequencies]
    result = (merge or _merge_frequency_results)(completed(), completed_frequencies)
    remaining = [f for f in frequencies if f not in completed_frequencies]
    if remaining:
        checkpoint_warnings.append('PARTIAL RESULT: stopped after {} of {} frequencies because continuing '
            'with unsaved results could exceed the available RAM allowance. Remaining frequencies: {} GHz. '
            'Completed samples remain available for manual export.'.format(
                len(completed_frequencies), len(frequencies), ', '.join(map(str, remaining))))
        result['metadata']['partial_result'] = True
        result['metadata']['remaining_frequencies_ghz'] = remaining
        result['metadata']['requested_frequency_count'] = len(frequencies)
    result['metadata'].setdefault('warnings', []).extend(checkpoint_warnings)
    result['metadata']['warning_count'] = len(result['metadata']['warnings'])
    result['metadata']['frequency_checkpoints'] = dict(directory=str(store.directory),
        completed=len(completed_frequencies), persisted=len(completed_frequencies) - unsaved_count,
            reused=reused, input_sha256=identity, write_warnings=checkpoint_warnings)
    if parallel is not None:
        result['metadata']['frequency_execution'] = parallel['details']
    peaks = [p['sampled_peak_process_rss_bytes'] for p in profiles if p.get('sampled_peak_process_rss_bytes') is not None]
    result['metadata']['runtime_profile'] = dict(
        wall_seconds=time.perf_counter()-started,
        stage_seconds={key: sum(p.get('stage_seconds',{}).get(key,0.) for p in profiles)
                       for key in set(key for p in profiles for key in p.get('stage_seconds',{}))},
        stage_calls={key: sum(p.get('stage_calls',{}).get(key,0) for p in profiles)
                     for key in set(key for p in profiles for key in p.get('stage_calls',{}))},
        sampled_peak_process_rss_bytes=max(peaks) if peaks else None,
        stage_semantics='Current execution only; cached stages excluded; nested stages may overlap.',
        memory_semantics='Process samples from frequencies computed in this execution; cached samples excluded.')
    for key in ('sampled_peak_process_tree_rss_bytes', 'sampled_peak_process_tree_private_bytes'):
        values = [p[key] for p in profiles if p.get(key) is not None]
        result['metadata']['runtime_profile'][key] = max(values) if values else None
    result['metadata']['runtime_profile']['process_tree_incomplete_samples'] = sum(
        p.get('process_tree_incomplete_samples', 0) for p in profiles)
    result['metadata']['runtime_profile']['process_tree_memory_semantics'] = (
        'Parent plus descendant RSS sums from this execution only; cached samples excluded; '
        'shared pages may be counted more than once. Private bytes are Windows private commit.')
    if parallel is not None:
        profile = result['metadata']['runtime_profile']
        for key, value in parallel['worker_peaks'].items():
            profile[key.replace('sampled_peak_', 'sampled_peak_frequency_worker_')] = value
            # Independent worker maxima are not a simultaneous process-tree
            # measurement. Do not present a worker's footprint as total RAM.
            profile[key] = None
        profile['memory_semantics'] = (
            'Total concurrent RSS was not sampled. frequency_worker fields are maxima '
            'of individual worker solve samples, not aggregate sweep memory.')
        profile['process_tree_memory_semantics'] = profile['memory_semantics']
        profile['stage_semantics'] = (
            'Current execution only; cached stages excluded. Worker stage times are summed; '
            'concurrent and nested work may overlap and their sum may exceed wall time.')
    return result
