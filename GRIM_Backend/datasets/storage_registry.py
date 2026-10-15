"""Inventory and conservative cleanup of GRIM-owned temporary sample files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid
import weakref

import numpy as np

_MARKER = '.grim-cache-owner.json'
_PREFIXES = ('grim-result-', 'grim-native-')
_SESSION = uuid.uuid4().hex
_LOCK = threading.RLock()
_CACHES = {}


def _process_start(pid):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def register_cache(path, *, kind='result', owner=None):
    path = Path(path).resolve()
    record = {'schema': 1, 'pid': os.getpid(), 'process_start': _process_start(os.getpid()),
              'session': _SESSION, 'kind': str(kind), 'created': time.time(), 'path': str(path)}
    # Tracking must not prevent an otherwise valid operation on a restricted folder.
    try:
        (path / _MARKER).write_text(json.dumps(record), encoding='utf-8')
    except OSError:
        pass
    with _LOCK:
        _CACHES[str(path)] = (record, weakref.ref(owner) if owner is not None else None)


def forget_cache(path):
    with _LOCK:
        _CACHES.pop(str(Path(path).resolve()), None)


def _plain_directory(path):
    try:
        status = path.lstat()
        return path.is_dir() and not path.is_symlink() and not (getattr(status, 'st_file_attributes', 0) & 0x400)
    except OSError:
        return False


def _read_record(path):
    try:
        marker = path / _MARKER
        if marker.is_symlink() or marker.stat().st_size > 4096:
            return None
        record = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            return None
        if (record.get('schema') != 1 or Path(record['path']).resolve() != path.resolve()
                or not isinstance(record['pid'], int) or isinstance(record['pid'], bool)
                or record['pid'] <= 0 or not isinstance(record.get('session'), str)):
            return None
        return record
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _abandoned(record):
    """Unknown/access-denied owners are never treated as abandoned."""
    if record['pid'] == os.getpid() and record.get('session') == _SESSION:
        return False
    try:
        import psutil
        process = psutil.Process(record['pid'])
        started = record.get('process_start')
        return isinstance(started, (int, float)) and abs(process.create_time() - started) > .01
    except ImportError:
        return False
    except Exception as error:
        import psutil
        return isinstance(error, psutil.NoSuchProcess)


def _cache_bytes(path):
    total = 0
    try:
        # Current caches have only direct files. Do not traverse reparse points.
        for item in path.iterdir():
            if item.name != _MARKER and not item.is_symlink() and item.is_file():
                total += item.stat().st_size
    except OSError:
        pass
    return total


def cache_inventory(directories=()):
    with _LOCK:
        known = dict(_CACHES)
    roots = {Path(tempfile.gettempdir()).resolve()}
    roots.update(Path(value).resolve() for value in directories if value)
    roots.update(Path(value).parent for value in known)
    paths = {Path(value) for value in known}
    for root in roots:
        try:
            paths.update(path for path in root.iterdir() if path.name.startswith(_PREFIXES))
        except OSError:
            continue
    result = []
    for path in sorted(paths):
        if not _plain_directory(path):
            continue
        record = _read_record(path)
        if record is None:
            continue
        result.append({**record, 'path': str(path.resolve()), 'bytes': _cache_bytes(path),
                       'abandoned': _abandoned(record)})
    return result


def cleanup_abandoned(directories=()):
    """Remove only marked, direct GRIM cache folders with a proven dead owner."""
    removed, errors = [], []
    for entry in cache_inventory(directories):
        if not entry['abandoned']:
            continue
        path = Path(entry['path'])
        parent = path.parent.resolve()
        # Revalidate identity and containment immediately before recursive removal.
        record = _read_record(path)
        if (not path.name.startswith(_PREFIXES) or not _plain_directory(path)
                or path.resolve().parent != parent or record is None
                or record.get('session') != entry['session'] or not _abandoned(record)):
            continue
        try:
            shutil.rmtree(path)
            forget_cache(path)
            removed.append(entry)
        except OSError as error:
            errors.append(str(error))
    return {'removed': len(removed), 'bytes': sum(item['bytes'] for item in removed), 'errors': errors}


def dataset_storage_bytes(datasets, *, seen=None):
    """Count already materialized sample buffers without deriving power/phase."""
    from GRIM_Backend.datasets.result_storage import array_storage
    seen = set() if seen is None else seen
    totals = {'memory': 0, 'disk': 0}
    for dataset in datasets:
        values = [getattr(dataset, name, None) for name in ('_complex', '_power', '_phase')]
        values.extend(value for value in getattr(dataset, 'extra', {}).values() if isinstance(value, np.ndarray))
        for value in values:
            if not isinstance(value, np.ndarray):
                continue
            base = value
            while isinstance(getattr(base, 'base', None), np.ndarray):
                base = base.base
            if id(base) in seen:
                continue
            seen.add(id(base))
            totals[array_storage(base)] += base.nbytes
    return totals
