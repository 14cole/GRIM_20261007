"""Run-owned recovery files. A new solve never opens a previous run."""
import bisect
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
import weakref
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from ghost_backend.twod.checkpoints import FrequencyCheckpoints

SCHEMA = 'ghost.run-recovery.v1'
_READERS = {}
_READER_LOCK = threading.RLock()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.', suffix='.writing', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2,
                      default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()).encode('utf-8')


class RunStore(FrequencyCheckpoints):
    def __init__(self, run):
        self.directory = run.directory / 'frequencies'
        self.identity = run.manifest['run_id'] + ':' + run.manifest['input_sha256']
        self.certified = bool(run.manifest['certified'])
        self.frequencies = run.manifest['frequencies_ghz']

    def _path(self, frequency):
        index = self.frequencies.index(float(frequency))
        return self.directory / ('frequency_{:06d}_{:.12g}GHz.npz'.format(index+1, frequency))


class RecoveryRun:
    @classmethod
    def create(cls, root, arguments, options, precision, certified, *, solver_kind='2d',
               source_path='', context=None):
        from ghost_backend.twod.geometry import _material_base_dir_for_snapshot, _resolve_material_file
        from ghost_backend.geometry.io import material_filename_from_row, Segment, build_geometry_text
        run_id = uuid.uuid4().hex
        directory = Path(root).expanduser().resolve() / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + run_id)
        directory.mkdir(parents=True, exist_ok=False)
        inputs = directory / 'inputs'
        inputs.mkdir()
        (directory / 'frequencies').mkdir()
        snapshot = copy.deepcopy(arguments['geometry_snapshot'])
        base = _material_base_dir_for_snapshot(snapshot, arguments.get('material_base_dir'))
        files = {}
        try:
            for row in list(snapshot.get('ibcs', [])) + list(snapshot.get('dielectrics', [])):
                name = material_filename_from_row(row)
                if not name or name in files:
                    continue
                path = Path(_resolve_material_file(base, name))
                before = path.stat()
                raw = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Material changed while capturing this run: ' + str(path))
                atomic_bytes(inputs/name, raw)
                files[name] = hashlib.sha256(raw).hexdigest()
            atomic_bytes(inputs/'geometry_snapshot.json', json_bytes(snapshot))
            segments = []
            for segment in snapshot['segments']:
                pairs = segment['point_pairs']
                segments.append(Segment(segment['name'], str(segment['seg_type']),
                    list(segment['properties']), [p[k] for p in pairs for k in ('x1','x2')],
                    [p[k] for p in pairs for k in ('y1','y2')]))
            text = build_geometry_text(snapshot.get('title','Recovery geometry'), segments,
                                       snapshot.get('ibcs',[]), snapshot.get('dielectrics',[]))
            atomic_bytes(inputs/'geometry.geo', text.encode('utf-8'))
            files.update({name:sha(inputs/name) for name in ('geometry_snapshot.json','geometry.geo')})
            clean = {k:v for k,v in arguments.items() if k not in ('progress_callback','abort_event')}
            clean = copy.deepcopy(clean)
            clean['geometry_snapshot'] = snapshot
            clean['material_base_dir'] = str(inputs)
            request = dict(arguments=clean, options=options, precision=precision, solver_kind=solver_kind,
                           certified=bool(certified), context=copy.deepcopy(context or {}))
            atomic_bytes(inputs/'request.json', json_bytes(request))
            files['request.json'] = sha(inputs/'request.json')
            manifest = dict(schema=SCHEMA, run_id=run_id, state='running',
                created_utc=datetime.now(timezone.utc).isoformat(), source_path=str(source_path),
                solver_kind=solver_kind, certified=bool(certified),
                frequencies_ghz=list(map(float, arguments['frequencies_ghz'])), input_files=files,
                input_sha256=hashlib.sha256(json_bytes(files)).hexdigest(), exports=[])
            atomic_bytes(directory/'run.json', json_bytes(manifest))
            return cls(directory, manifest)
        except BaseException as exc:
            atomic_bytes(directory/'capture_failed.txt', str(exc).encode('utf-8'))
            raise

    @classmethod
    def open(cls, directory):
        directory = Path(directory).expanduser().resolve()
        manifest = json.loads((directory/'run.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA or uuid.UUID(hex=manifest['run_id']).hex != manifest['run_id']:
            raise ValueError('Not a supported run recovery folder.')
        return cls(directory, manifest)

    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest
        self.store = RunStore(self)

    def verify_inputs(self):
        if hashlib.sha256(json_bytes(self.manifest['input_files'])).hexdigest() != self.manifest['input_sha256']:
            raise ValueError('Recovery input manifest has changed.')
        for name, expected in self.manifest['input_files'].items():
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Invalid recovery input filename.')
            if sha(self.directory/'inputs'/name) != expected:
                raise ValueError('Captured input changed: ' + name)

    def request(self):
        self.verify_inputs()
        return json.loads((self.directory/'inputs/request.json').read_text(encoding='utf-8'))

    def arguments(self, callbacks):
        values = self.request()['arguments']
        for key in ('progress_callback','abort_event'):
            if key in callbacks:
                values[key] = callbacks[key]
        return values

    def save(self, frequency, result):
        self.verify_inputs()
        value = dict(result)
        value['_recovery_row_counts'] = dict(samples=len(result['samples']),
            **{p:len(result['co_solved_samples'][p]) for p in ('VV','HH')})
        self.store.save(frequency, value)
        self.load(frequency)

    def completed(self):
        return [f for f in dict.fromkeys(self.manifest['frequencies_ghz']) if self.store.available(f)]

    def load(self, frequency):
        from ghost_backend.twod.samples import compact_samples
        with compact_samples():
            result = self.store.load(frequency)
        if result is None:
            raise IOError('Frequency output is missing or corrupt: {:g} GHz'.format(frequency))
        result.pop('_recovery_row_counts', None)
        return result

    def header(self, frequency):
        if not self.store.available(frequency):
            raise IOError('Frequency output is missing or corrupt: {:g} GHz'.format(frequency))
        with np.load(self.store._path(frequency), allow_pickle=False) as archive:
            return json.loads(archive['header'].tobytes().decode('utf-8'))['result']

    def status(self, state, error=None, **extra):
        manifest = dict(self.manifest, state=state, completed_frequencies_ghz=self.completed(), **extra)
        if error is not None:
            manifest['message'] = str(error)
        atomic_bytes(self.directory/'run.json', json_bytes(manifest))
        self.manifest = manifest

    def result(self):
        from ghost_backend.twod.solver import _merge_frequency_results
        from ghost_backend.twod.samples import compact_samples
        self.verify_inputs()
        available = self.completed()
        if not available:
            raise ValueError('This run has no verified completed frequencies to recover.')
        frequencies = [f for f in self.manifest['frequencies_ghz'] if f in available]
        counts = {}
        def headers():
            for f in frequencies:
                h = self.header(f)
                counts[f] = h.pop('_recovery_row_counts')
                h.update(samples=[], co_solved_samples={'VV':[], 'HH':[]})
                yield h
        merge = _merge_frequency_results
        if self.manifest['solver_kind'] == 'bor':
            from ghost_backend.bor.checkpoints import merge_frequency_results as merge
        with compact_samples():
            result = merge(headers(), frequencies)
        reader = FrequencyReader(self)
        ordered = sorted(frequencies) if self.manifest['solver_kind']=='bor' else frequencies
        result['samples'] = DiskSamples(reader, ordered, counts)
        for pol in ('VV','HH'):
            channel_order = ordered if result['metadata'].get('expanded_to_360') else frequencies
            result['co_solved_samples'][pol] = DiskSamples(reader, channel_order, counts, pol)
        result['_recovery_run'] = str(self.directory)
        result['metadata'].pop('frequency_checkpoints', None)
        result['metadata'].setdefault('geometry_units_in', self.request()['arguments'].get('geometry_units', 'inches'))
        missing = [f for f in self.manifest['frequencies_ghz'] if f not in available]
        result['metadata']['run_recovery'] = dict(directory=str(self.directory), run_id=self.manifest['run_id'],
            input_sha256=self.manifest['input_sha256'], completed=len(frequencies),
            requested=len(self.manifest['frequencies_ghz']), remaining_frequencies_ghz=missing)
        if missing:
            result['metadata'].update(partial_result=True, remaining_frequencies_ghz=missing,
                requested_frequency_count=len(self.manifest['frequencies_ghz']))
        return result


class FrequencyReader:
    def __init__(self, run):
        self.run, self.frequency, self.value = run, None, None
        # Keep successful run files while their disk-backed result is displayed.
        # Interrupted and partially exported runs are never removed here.
        directory = str(run.directory.resolve())
        with _READER_LOCK:
            _READERS[directory] = _READERS.get(directory, 0) + 1
        weakref.finalize(self, _release_reader, directory)

    def read(self, frequency):
        if self.frequency != frequency:
            self.value = None
            self.value = self.run.load(frequency)
            self.frequency = frequency
        return self.value


class DiskSamples(Sequence):
    def __init__(self, reader, frequencies, counts, channel=None):
        self.reader, self.frequencies, self.channel = reader, list(frequencies), channel
        self.ends = np.cumsum([counts[f][channel or 'samples'] for f in frequencies]).tolist()

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _rows(self, frequency):
        value = self.reader.read(frequency)
        return value['samples'] if self.channel is None else value['co_solved_samples'][self.channel]

    def __iter__(self):
        for frequency in self.frequencies:
            yield from self._rows(frequency)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chunk = bisect.bisect_right(self.ends, index)
        return self._rows(self.frequencies[chunk])[index-(self.ends[chunk-1] if chunk else 0)]


def _release_reader(directory):
    with _READER_LOCK:
        count = _READERS.get(directory, 1) - 1
        if count:
            _READERS[directory] = count
        else:
            _READERS.pop(directory, None)
            cleanup_exported(directory)


def cleanup_exported(directory):
    with _READER_LOCK:
        if not _READERS.get(str(Path(directory).resolve())):
            _cleanup_exported(directory)


def _cleanup_exported(directory):
    """Delete only a released, completely exported run whose final files verify."""
    try:
        run = RecoveryRun.open(directory)
        if run.manifest['state'] != 'exported' or not run.manifest.get('exports'):
            return
        if set(run.completed()) != set(run.manifest['frequencies_ghz']):
            return
        for item in run.manifest['exports']:
            if sha(item['path']) != item['sha256']:
                return
        # Refuse links/junctions and unexpected roots before recursive cleanup.
        target = run.directory.resolve()
        if not target.name.endswith('_'+run.manifest['run_id']) or target.parent == target:
            return
        for path in [target, *target.rglob('*')]:
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                return
        shutil.rmtree(target)
    except (OSError, ValueError, KeyError):
        pass  # Preserve recovery data whenever verification or cleanup fails.
