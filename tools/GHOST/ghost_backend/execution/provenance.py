"""Source inventories, runtime fingerprints, and output attestations."""

import hashlib
import json
import math
import ntpath
import os
import platform
import posixpath
import sys
import tempfile
import threading
from typing import Any, Dict, List, Sequence


_BACKEND_SOURCE_SUFFIXES = (
    ".py",
    ".pyx",
    ".pxd",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".f",
    ".f90",
    ".so",
    ".dll",
    ".dylib",
    ".pyd",
)


def backend_source_paths(backend_dir: 'str') -> 'List[str]':
    """Return Python and native artifacts from the backend and its packages."""

    paths: 'List[str]' = []
    packages = {"assembly", "bor", "compressed", "execution", "geometry",
                "hpc", "io", "linalg", "runs", "twod", "ui", "validation"}
    root = os.path.abspath(backend_dir)
    for directory, folders, filenames in os.walk(root):
        folders[:] = [name for name in folders if (
            name in packages if directory == root else
            os.path.isfile(os.path.join(directory, name, "__init__.py"))
        )]
        paths.extend(os.path.join(directory, name) for name in filenames
                     if name.endswith(_BACKEND_SOURCE_SUFFIXES))
    return sorted(paths)


def sha256_file(path: 'str') -> 'str':
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SOURCE_DIGESTS = {}
_SOURCE_DIGEST_LOCK = threading.Lock()


def source_file_digest(path: 'str') -> 'str':
    """``sha256_file`` of a backend source file, memoized on (path, size, mtime_ns).

    The source bundle (178 files, 4.4 MB) is hashed by the provenance
    manifest, the timing-history key and the checkpoint identity of every
    public call; one process re-reads it only when a file's stat changes.
    Output artifacts are never memoized (``sha256_file``).
    """

    path = os.path.abspath(path)
    try:
        status = os.stat(path)
    except OSError:
        return sha256_file(path)
    key = (path, int(status.st_size), int(status.st_mtime_ns))
    with _SOURCE_DIGEST_LOCK:
        cached = _SOURCE_DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = sha256_file(path)
    with _SOURCE_DIGEST_LOCK:
        if len(_SOURCE_DIGESTS) >= 4096:
            _SOURCE_DIGESTS.clear()
        _SOURCE_DIGESTS[key] = digest
    return digest


def source_bundle_fingerprint(records: 'Dict[str, str]') -> 'str':
    """Hash logical source names and exact bytes, independent of location."""

    payload = [
        {
            "path": str(logical_name),
            "sha256": source_file_digest(path),
        }
        for logical_name, path in sorted(records.items())
    ]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def backend_source_records(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'Dict[str, str]':
    """Logical name -> path for everything a solve's identity depends on."""

    records = {
        "ghost_backend/" + os.path.relpath(path, backend_dir).replace(os.sep, "/"): path
        for path in backend_source_paths(backend_dir)
    }
    records.update(extra_records or {})
    return records


def backend_source_inventory(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'Dict[str, str]':
    """Per-file hashes behind `backend_source_fingerprint`.

    The fingerprint alone can only say that *something* under ghost_backend/ differs
    from what a run recorded, which is not enough to act on -- the usual cause
    is a partially updated tree, and the useful question is which file.
    Recording the inventory beside the fingerprint lets the mismatch name the
    files that were added, removed, or edited.
    """

    return {
        name: source_file_digest(path)
        for name, path in sorted(backend_source_records(
            backend_dir, extra_records
        ).items())
    }


def compare_source_inventories(
    expected: 'Dict[str, str]',
    actual: 'Dict[str, str]',
) -> 'Dict[str, List[str]]':
    """(changed, added, removed) logical names between two inventories."""

    expected = dict(expected or {})
    actual = dict(actual or {})
    return {
        "changed": sorted(
            name for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        ),
        "added": sorted(set(actual) - set(expected)),
        "removed": sorted(set(expected) - set(actual)),
    }


def describe_source_mismatch(
    expected: 'Dict[str, str]',
    actual: 'Dict[str, str]',
) -> 'str':
    """One-line summary of an inventory difference, or '' when identical."""

    diff = compare_source_inventories(expected, actual)
    parts = []
    for label in ("changed", "added", "removed"):
        names = diff[label]
        if names:
            shown = ", ".join(names[:6])
            if len(names) > 6:
                shown += f", ... (+{len(names) - 6} more)"
            parts.append(f"{label}: {shown}")
    return "; ".join(parts)


def backend_source_fingerprint(
    backend_dir: 'str',
    extra_records: 'Dict[str, str]' = None,
) -> 'str':
    """Hash a backend tree plus logically named runner/config files."""

    return source_bundle_fingerprint(
        backend_source_records(backend_dir, extra_records)
    )


def embed_output_attestation(
    result: 'Dict[str, Any]',
    provenance: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    """Embed source, runtime, geometry, and solve identity in a result before export."""

    payload = dict(provenance)
    payload["schema"] = "ghost.workflow.embedded-attestation.v1"
    result.setdefault("metadata", {})["output_attestation"] = payload
    return result


def read_embedded_attestation(output_path: 'str') -> 'Dict[str, Any]':
    """Attestation carried inside a .grim, or {} when there is none."""

    import numpy as np

    try:
        with np.load(output_path, allow_pickle=False) as payload:
            raw = np.asarray(
                payload["solver_metadata_json"]
            ).reshape(()).item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        audit = json.loads(str(raw))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{os.path.basename(output_path)} has no readable solver audit; "
            "it may be truncated or from an older run."
        ) from exc
    metadata = audit.get("metadata", audit)
    found = metadata.get("output_attestation")
    return dict(found) if isinstance(found, dict) else {}


def verify_embedded_attestation(
    output_path: 'str',
    expected: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    """Verify every caller-specified provenance field carried in the artifact.

    Reading the file at all is the integrity check: numpy validates the zip
    CRC-32 of each member, so a corrupted result raises rather than verifying.
    """

    payload = read_embedded_attestation(output_path)
    name = os.path.basename(output_path)
    if payload.get("schema") != "ghost.workflow.embedded-attestation.v1":
        raise ValueError(
            f"{name} carries no embedded run attestation; it predates this "
            "format or was written by a different tool."
        )
    for field, want in expected.items():
        got = payload.get(field)
        if _stable_json_value(got) != _stable_json_value(want):
            raise ValueError(
                f"{name} was produced under a different {field}: "
                f"recorded {got!r}, expected {want!r}."
            )
    return payload


def _stable_json_value(value: 'Any') -> 'Any':
    """Convert build-configuration objects to deterministic JSON values."""

    if isinstance(value, dict):
        return {
            str(key): _stable_json_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    return str(value)


def stable_json_fingerprint(value: 'Any') -> 'str':
    """Hash a JSON-like solve specification deterministically."""

    raw = json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def manifest_solve_spec_fingerprint(manifest: 'Dict[str, Any]') -> 'str':
    """Hash every immutable field that defines the underlying unit solves.

    Completion status and byte inventories are written after individual unit
    attestations, so they are deliberately excluded. Editing a grid, unit,
    solver control, or source/runtime hash changes this fingerprint and
    invalidates every stale per-unit output. Legacy derived-grid configuration
    has a separate attestation because it does not change the underlying VV/HH
    solve.
    """

    mutable_fields = {
        "status",
        "output_sha256",
        "collection",
        "collection_manifest",


        "azel_config",
    }
    solve_spec = {
        str(key): value
        for key, value in manifest.items()
        if str(key) not in mutable_fields
    }
    return stable_json_fingerprint(solve_spec)


def unit_solve_spec_fingerprint(unit: 'Dict[str, Any]') -> 'str':
    """Hash the complete per-unit record, including its angular grid."""

    return stable_json_fingerprint(unit)


def _package_configuration(package: 'Any') -> 'Dict[str, Any]':
    config_module = getattr(package, "__config__", None)
    config = getattr(config_module, "CONFIG", None)
    return _stable_json_value(config) if isinstance(config, dict) else {}


def runtime_environment_payload() -> 'Dict[str, Any]':
    """Numerically relevant interpreter/platform/library configuration."""

    import numpy as np
    try:
        import scipy
    except Exception:
        scipy = None
    implementation = getattr(sys, "implementation", None)
    from ghost_backend.linalg.hierarchical import factor_mode
    from ghost_backend.linalg.sweep import mode as compression_mode
    from ghost_backend.compressed.runtime import automatic_storage, storage_budget
    from ghost_backend.execution.options import current_options
    profile = current_options()
    factorization = profile['factorization'] if profile is not None else factor_mode()
    return {
        "execution_options": profile,
        "cpu_factorization": factorization,
        # Automatic storage follows free memory at solve time, so record the setting, not bytes.
        "compressed_storage_budget_bytes": (('automatic' if automatic_storage() else storage_budget())
                                            if factorization in ('compressed', 'adaptive') else None),
        "cpu_rhs_compression": compression_mode(),
        "python_version": sys.version,
        "python_implementation": getattr(implementation, "name", ""),
        "python_cache_tag": getattr(implementation, "cache_tag", ""),
        "byteorder": sys.byteorder,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "numpy_version": np.__version__,
        "numpy_config": _package_configuration(np),
        "scipy_version": getattr(scipy, "__version__", "unavailable"),
        "scipy_config": (
            _package_configuration(scipy) if scipy is not None else {}
        ),
    }


# Recorded for provenance, never compared: the kernel release and CPU
# description of the host, and the sections of NumPy's and SciPy's build
# configuration that describe hardware rather than the build -- the CPU
# features found at import ("SIMD Extensions", which differ between a login
# node and a compute node of another generation) and the build host
# ("Machine Information").  The build dependencies (BLAS/LAPACK name and
# version), compilers and library versions stay strict.
_RUNTIME_INFORMATIONAL_FIELDS = frozenset({"platform_release", "platform_processor"})
_RUNTIME_INFORMATIONAL_CONFIG_SECTIONS = frozenset({"SIMD Extensions", "Machine Information"})
_RUNTIME_CHECK_SWITCH = "GHOST_RUNTIME_ENVIRONMENT_CHECK"
_runtime_mismatches_warned = set()


def runtime_compatibility_payload(payload: 'Dict[str, Any]' = None) -> 'Dict[str, Any]':
    """Keep numerical dependencies strict without binding a run to one host.

    Kernel releases, CPU descriptions and the CPU features the numerical
    libraries detect at import remain informational provenance.  OS family,
    architecture, interpreter and numerical builds stay strict.
    """
    environment = runtime_environment_payload() if payload is None else payload
    compatible = {}
    for key, value in environment.items():
        if key in _RUNTIME_INFORMATIONAL_FIELDS:
            continue
        if key in ("numpy_config", "scipy_config") and isinstance(value, dict):
            value = {section: entry for section, entry in value.items()
                     if section not in _RUNTIME_INFORMATIONAL_CONFIG_SECTIONS}
        compatible[key] = value
    return compatible


def _runtime_value_summary(value: 'Any') -> 'str':
    text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 160 else text[:157] + "..."


def describe_runtime_mismatch(expected_payload: 'Any', actual_payload: 'Dict[str, Any]' = None) -> 'str':
    """Name the compatibility fields of ``expected_payload`` (a recorded
    ``runtime_environment_payload``) that differ from the current environment;
    '' when they agree or when no payload was recorded."""
    if not isinstance(expected_payload, dict) or not expected_payload:
        return ""
    expected = runtime_compatibility_payload(expected_payload)
    actual = runtime_compatibility_payload(actual_payload)
    parts = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) == actual.get(key):
            continue
        before, now = expected.get(key), actual.get(key)
        if isinstance(before, dict) and isinstance(now, dict):
            sections = [section for section in sorted(set(before) | set(now)) if before.get(section) != now.get(section)]
            parts.append("%s: %s" % (key, ", ".join(
                "%s recorded %s, now %s" % (section, _runtime_value_summary(before.get(section)),
                                             _runtime_value_summary(now.get(section))) for section in sections)))
        else:
            parts.append("%s: recorded %s, now %s" % (key, _runtime_value_summary(before), _runtime_value_summary(now)))
    return "; ".join(parts)


def runtime_environment_check_mode() -> 'str':
    """'strict' (a mismatch refuses the unit) or 'warn' (``GHOST_RUNTIME_ENVIRONMENT_CHECK=warn``)."""
    raw = os.environ.get(_RUNTIME_CHECK_SWITCH, "").strip().lower()
    return "warn" if raw in ("warn", "warning", "0", "off", "false", "no") else "strict"


def verify_runtime_environment(expected_fingerprint: 'str', expected_payload: 'Any' = None,
                               origin: 'str' = "the run manifest") -> 'None':
    """Refuse (strict) or warn once (warn mode) when the current numerical
    runtime differs from the one a run recorded; name the differing fields
    when the run recorded its submission environment."""
    actual = runtime_environment_fingerprint()
    if actual == str(expected_fingerprint):
        return
    detail = describe_runtime_mismatch(expected_payload)
    if not detail:
        detail = ("the run does not record its submission environment, so the differing "
                  "field cannot be named; compare Python, NumPy, SciPy and BLAS versions "
                  "against the submit host")
    message = (
        "Python/NumPy/SciPy/BLAS runtime differs from %s (%s). Start a new run in this "
        "numerical environment, or set %s=warn to solve here under the recorded runtime "
        "attestation and accept that difference." % (origin, detail, _RUNTIME_CHECK_SWITCH))
    if runtime_environment_check_mode() == "warn":
        if actual not in _runtime_mismatches_warned:
            _runtime_mismatches_warned.add(actual)
            print("  [warn] " + message, flush=True)
        return
    raise RuntimeError(message)


def runtime_environment_fingerprint() -> 'str':
    raw = json.dumps(
        runtime_compatibility_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
