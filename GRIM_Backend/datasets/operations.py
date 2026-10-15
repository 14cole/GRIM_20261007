"""Operation provenance shared by button actions and the Python dataset API.

Records describe operation lineage, not a checksum of the numerical samples.
Existing format-specific metadata and the Python replay recorder remain intact.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
from threading import Lock
import uuid
import weakref

from datetime import datetime, timezone

import numpy as np


OPERATION_PROVENANCE_KEY = "operation_provenance_json"
_SCHEMA = "grim.operation-provenance.v1"
_ACTIVE_OPERATION = ContextVar("grim_dataset_operation", default=False)
_SOURCE_IDENTITIES = weakref.WeakKeyDictionary()
_IDENTITY_LOCK = Lock()


def _is_grid(value):
    return (hasattr(value, "native_sample_payload") and hasattr(value, "extra")
            and hasattr(value, "azimuths") and not isinstance(value, type))


def operation_events(grid):
    """Read valid common events; unfamiliar legacy metadata stays untouched."""
    raw = (grid.extra or {}).get(OPERATION_PROVENANCE_KEY)
    if raw is None:
        return []
    try:
        text = np.asarray(raw).reshape(()).item()
        record = json.loads(str(text))
        if record.get("schema") != _SCHEMA or not isinstance(record.get("events"), list):
            return []
        return [event for event in record["events"]
                if isinstance(event, dict) and isinstance(event.get("id"), str)]
    except (TypeError, ValueError, AttributeError):
        return []


def operation_recorded_here(grid):
    """Whether this particular result already owns its core history entry."""
    return getattr(grid, "_operation_recorded_output", None) == id(grid)


def _parameter(value):
    if isinstance(value, np.generic):
        return _parameter(value.item())
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return _parameter(value.tolist())
        # Large requested coordinate arrays should not become another full
        # sample cache. The existing script recorder retains replay values.
        array = np.ascontiguousarray(value)
        payload = (repr(array.tolist()).encode("utf-8") if array.dtype.hasobject
                   else memoryview(array).cast("B"))
        return {"array_shape": list(array.shape), "dtype": array.dtype.str,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "first": _parameter(array.flat[0]), "last": _parameter(array.flat[-1])}
    if isinstance(value, dict):
        return {str(key): _parameter(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 256:
            digest = hashlib.sha256()
            for item in value:
                digest.update(json.dumps(_parameter(item), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
                digest.update(b"\n")
            return {"sequence_length": len(value), "sha256": digest.hexdigest(),
                    "first": _parameter(value[0]), "last": _parameter(value[-1])}
        return [_parameter(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [_parameter(value.start), _parameter(value.stop), _parameter(value.step)]}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _grids(value):
    if _is_grid(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return [grid for item in value for grid in _grids(item)]
    return []


def _reference(grid):
    events = operation_events(grid)
    with _IDENTITY_LOCK:
        identity = _SOURCE_IDENTITIES.get(grid)
        if identity is None:
            identity = events[-1].get("output_id") if events else None
            identity = identity or uuid.uuid4().hex
            _SOURCE_IDENTITIES[grid] = identity
    return {
        "source_id": identity,
        "source_path": str(grid.source_path) if grid.source_path is not None else None,
        "shape": list(grid.shape),
        "linear_quantity": grid.linear_quantity(),
        "axis_units": {key: _parameter(grid.units.get(key))
                       for key in ("azimuth", "elevation", "frequency")},
        "previous_operation_id": events[-1]["id"] if events else None,
    }


def _metadata_value(value):
    """Describe metadata without deriving samples or hashing large arrays."""
    if isinstance(value, np.ndarray):
        if value.size > 256:
            return {"array_shape": list(value.shape), "dtype": value.dtype.str,
                    "array_values_compared": False}
        return _parameter(value)
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) > 256:
            return {"sequence_length": len(value), "sequence_values_compared": False}
        return [_metadata_value(item) for item in value]
    return _parameter(value)


def _metadata_snapshot(grid):
    return {
        "units": {str(key): _metadata_value(value) for key, value in (grid.units or {}).items()},
        "extra": {str(key): _metadata_value(value) for key, value in (grid.extra or {}).items()
                  if key != OPERATION_PROVENANCE_KEY},
        "rcs_domain": str(grid.rcs_domain),
    }


def _metadata_changes(before, after):
    changes = {}
    for section in ("units", "extra"):
        old, new = before[section], after[section]
        added = {key: value for key, value in new.items() if key not in old}
        removed = {key: value for key, value in old.items() if key not in new}
        changed = {key: {"before": old[key], "after": new[key]}
                   for key in old.keys() & new.keys() if old[key] != new[key]}
        if added or removed or changed:
            changes[section] = {"added": added, "removed": removed, "changed": changed}
    if before["rcs_domain"] != after["rcs_domain"]:
        changes["rcs_domain"] = {"before": before["rcs_domain"], "after": after["rcs_domain"]}
    return changes


_RESULT_DETAILS = {
    "statistics_dataset": "statistics_reduction_json",
    "stitch_many": "stitch_provenance_json",
    "range_calibrate": "range_calibration_json",
    "time_gate": "time_gate_json",
    "translate_phase_center": "phase_center_translation_json",
    "flip_phase": "phase_sign_correction_json",
    "decimate_axis": "decimation_json",
    "regrid_axis": "regrid_support_json",
    "interpolate_axis": "regrid_support_json",
    "combine_elevation_pair_to_azimuth_360": "elevation_pair_to_azimuth_json",
}


def _json_detail(extra, key):
    try:
        value = np.asarray(extra[key]).reshape(()).item()
        return _parameter(json.loads(str(value)))
    except (KeyError, TypeError, ValueError):
        return None


def _effective_details(result, operation, parameters):
    # Bound defaults describe the public call; method-specific reports carry
    # its normalized values and computed choices (for example compensation
    # is false for a remove gate even if the caller left compensate=True).
    effective = dict(parameters)
    detail = _json_detail(result.extra or {}, _RESULT_DETAILS.get(operation))
    if isinstance(detail, dict):
        for key in parameters.keys() & detail.keys():
            effective[key] = detail[key]
    return effective, detail


def _assumptions(result, sources, operation, parameters):
    assumptions = {}
    extra = result.extra or {}
    for key in extra:
        if "assumption" in key or "attestation" in key:
            value = _json_detail(extra, key) if key.endswith("_json") else _metadata_value(extra[key])
            if value is not None:
                assumptions[key] = value
    if operation.startswith("coherent_") or (operation == "stitch_many" and parameters.get("policy") == "coherent-mean"):
        assumptions["phase_relationship"] = "Input phases are used as supplied; comparable phase references are the engineer's responsibility."
    if operation in {"incoherent_add", "incoherent_add_many", "incoherent_subtract"}:
        assumptions["combination_domain"] = "Linear power; no phase relationship is retained."
    if operation == "arithmetic_db_subtract":
        assumptions["combination_domain"] = "Dimensionless linear power ratio; zero or missing denominators yield missing output."
    if operation == "time_gate" and sources:
        # The actual convention is recorded even when it is a default.
        details = _json_detail(extra, "time_gate_json") or {}
        assumptions["time_convention_used"] = details.get("time_convention")
        assumptions["missing_samples"] = "Only complete frequency sweeps are gated."
    return assumptions


def operation_effective_parameters(operation, **parameters):
    """Let a numerical method report resolved defaults without owning logging.

    Nested implementation calls cannot overwrite the public operation's
    arguments. Failed calls release this context without publishing a record.
    """
    active = _ACTIVE_OPERATION.get()
    if active and active["operation"] == operation:
        active["effective_parameters"].update({
            key: _parameter(value) for key, value in parameters.items()
        })


def _record(result, sources, operation, parameters, source_metadata, resolved_parameters):
    # A legitimate no-op may return its input. Never annotate or otherwise
    # mutate the engineer's source dataset just to record that no-op.
    if any(result is source for source in sources):
        return
    events, seen = [], set()
    for source in sources:
        for event in operation_events(source):
            if event["id"] not in seen:
                seen.add(event["id"])
                events.append(event)
    effective, details = _effective_details(result, operation, parameters)
    effective.update(resolved_parameters)
    output_metadata = _metadata_snapshot(result)
    event = {"operation": operation, "parameters": parameters,
             "effective_parameters": effective,
             "inputs": [_reference(source) for source in sources],
             "output_shape": list(result.shape),
             "assumptions": _assumptions(result, sources, operation, effective),
             "metadata_changes": [_metadata_changes(before, output_metadata) for before in source_metadata],
             "recorded_utc": datetime.now(timezone.utc).isoformat()}
    if details is not None:
        event["details"] = details
    # An event identifies an execution, not equivalent-looking arguments.
    # Different unnamed grids and repeated edits must never collapse into
    # one ancestor merely because their path, dimensions, and units match.
    event["id"] = uuid.uuid4().hex
    event["output_id"] = uuid.uuid4().hex
    with _IDENTITY_LOCK:
        _SOURCE_IDENTITIES[result] = event["output_id"]
    events.append(event)
    extra = dict(result.extra or {})
    extra[OPERATION_PROVENANCE_KEY] = json.dumps(
        {"schema": _SCHEMA, "events": events}, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result.extra = extra
    description = ", ".join(f"{key}={value}" for key, value in effective.items())
    line = f"GRIM operation: {operation}" + (f" ({description})" if description else "")
    if operation == "time_gate":
        # Keep the engineer-facing range and taper legible in both GUI and
        # scripted history; machine-readable effective values remain above.
        line = (
            f"GRIM operation: {operation}; Time gate ({effective['mode']} "
            f"{effective['start_m']:g} to {effective['stop_m']:g} m, "
            f"taper {100.0 * effective['taper']:g}%); coherent gain "
            f"compensation {'on' if effective['compensate'] else 'off'}"
        )
    history = str(result.history or "").strip()
    result.history = f"{history}\n{line}" if history else line
    result._operation_recorded_output = id(result)


def dataset_operation(name):
    """Record one public operation, suppressing its nested implementation calls."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _ACTIVE_OPERATION.get():
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            sources, parameters = [], {}
            for key, value in bound.arguments.items():
                inputs = _grids(value)
                if inputs:
                    sources.extend(inputs)
                elif key not in {"self", "cls"}:
                    parameters[key] = _parameter(value)
            source_metadata = [_metadata_snapshot(source) for source in sources]
            active = {"operation": name, "effective_parameters": {}}
            token = _ACTIVE_OPERATION.set(active)
            try:
                result = function(*args, **kwargs)
            finally:
                _ACTIVE_OPERATION.reset(token)
            for output in _grids(result):
                _record(output, sources, name, parameters, source_metadata, active["effective_parameters"])
            return result
        return wrapped
    return decorate
