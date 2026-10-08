"""Freeze an HPC run's backend independently of its editable source checkout."""
import os
from pathlib import Path
import shutil
import tempfile

from ghost_backend.execution.provenance import (
    backend_source_inventory, backend_source_records,
    describe_source_mismatch, sha256_file,
)

_RUNTIME_DATA_GLOBS = (
    "geometry/geometries/*.geo", "geometry/geometries/BOR/*.geo",
    "geometry/templates/*.csv", "execution/DATACLASSES_LICENSE.txt",
    "execution/thread_control/LICENSE*.txt",
)
_NAMESPACE_MARKER = '"""Frozen GHOST backend for this HPC run."""\n'


def _snapshot_records(backend_dir):
    root = Path(backend_dir).resolve()
    records = {str(Path(path).relative_to(root)): Path(path)
               for path in backend_source_records(str(root)).values()}
    for pattern in _RUNTIME_DATA_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                records[str(path.relative_to(root))] = path
    for path in records.values():
        if os.path.commonpath([str(root), str(path.resolve())]) != str(root):
            raise ValueError("Backend snapshot source escapes its package: " + str(path))
    return records


def _file_inventory(records):
    return {name: sha256_file(str(path)) for name, path in sorted(records.items())}


def snapshot_backend_runtime(backend_dir, run_dir):
    """Atomically publish a verified backend at run_dir/runtime; return its import root.

    Existing snapshots are never overwritten. Source/native bytes are verified
    before publication, including a check that the source stayed unchanged.
    Geometry and material inputs are frozen separately by the driver.
    """
    source, run = Path(backend_dir).resolve(), Path(run_dir).resolve()
    if not source.is_dir():
        raise ValueError("Backend snapshot source is not a directory: " + str(source))
    run.mkdir(parents=True, exist_ok=True)
    destination, lock = run / "runtime", run / ".runtime-snapshot.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Another backend snapshot is being created for " + str(run)) from exc
    staging = None
    try:
        os.close(descriptor)
        if os.path.lexists(str(destination)):
            raise FileExistsError("Run backend snapshot already exists: " + str(destination))
        records = _snapshot_records(source)
        before = _file_inventory(records)
        if not before:
            raise ValueError("Backend snapshot source contains no runtime files.")
        source_inventory = backend_source_inventory(str(source))
        staging = Path(tempfile.mkdtemp(prefix=".runtime-staging-", dir=str(run))).resolve()
        copied_backend = staging / "ghost_backend"
        for relative, path in sorted(records.items()):
            target = copied_backend / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
        copied = _file_inventory(_snapshot_records(copied_backend))
        after = _file_inventory(_snapshot_records(source))
        if before != copied or before != after:
            raise RuntimeError("Backend source changed while its run snapshot was being copied; retry submission.")
        copied_inventory = backend_source_inventory(str(copied_backend))
        if source_inventory != copied_inventory:
            raise RuntimeError("Backend snapshot inventory differs from source: " +
                               describe_source_mismatch(source_inventory, copied_inventory))
        # Isolate a namespace checkout from other installed regular packages.
        # This marker is included in the published snapshot's source fingerprint.
        marker = copied_backend / "__init__.py"
        if not marker.exists():
            marker.write_text(_NAMESPACE_MARKER, encoding="utf-8")
        os.rename(str(staging), str(destination))
        staging = None
        return destination
    finally:
        if staging is not None:
            if staging.parent != run or not staging.name.startswith(".runtime-staging-"):
                raise RuntimeError("Unsafe backend snapshot staging cleanup path.")
            shutil.rmtree(str(staging))
        lock.unlink()
