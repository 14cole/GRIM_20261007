"""Expand current native sample storage for standalone GHOST consumers.

GHOST's numerical services still consume power/phase and raw field arrays.
This boundary adapter accepts compact GRIM version 2 without importing the
viewer application, and drops storage framing before a service writes legacy
output arrays. It never assigns phase to nonzero power-only samples.
"""
from __future__ import annotations

import json

import numpy as np

C0 = 299_792_458.0


def decode_native_samples(payload, *, label="GRIM dataset"):
    if "grim_format_version" not in payload:
        return payload
    result = dict(payload)

    def scalar(key):
        if key not in result:
            raise ValueError(f"{label}: missing native metadata {key!r}.")
        value = np.asarray(result[key])
        if value.size != 1:
            raise ValueError(f"{label}: {key} must be scalar.")
        value = value.reshape(-1)[0]
        return value.item() if isinstance(value, np.generic) else value

    version = scalar("grim_format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
        raise ValueError(f"{label}: unsupported grim_format_version {version!r}.")
    if version == 1:
        result.pop("grim_format_version")
        return result
    kind, domain = scalar("storage_kind"), scalar("storage_domain")
    if kind not in {"complex", "power", "power_phase"}:
        raise ValueError(f"{label}: unsupported native storage_kind {kind!r}.")
    if domain not in {"physical", "raw_field"} or (domain == "raw_field" and kind != "complex"):
        raise ValueError(f"{label}: unsupported native storage_domain {domain!r}.")
    expected_keys = ({"rcs_complex"} if kind == "complex" else
                     {"rcs_power"} if kind == "power" else {"rcs_power", "rcs_phase"})
    if (set(result) & {"rcs_complex", "rcs_power", "rcs_phase"}) != expected_keys:
        raise ValueError(f"{label}: conflicting native sample arrays.")
    shape = tuple(np.asarray(result[key]).size for key in
                  ("azimuths", "elevations", "frequencies", "polarizations"))
    if kind == "complex":
        field = np.asarray(result.pop("rcs_complex"))
        if field.dtype.kind != "c" or field.shape != shape:
            raise ValueError(f"{label}: invalid native complex shape or dtype.")
        finite = np.isfinite(field.real) & np.isfinite(field.imag)
        missing = np.isnan(field.real) & np.isnan(field.imag)
        if np.any(~(finite | missing)):
            raise ValueError(f"{label}: native complex samples contain infinity or one-sided NaN.")
        try:
            raw_units = np.asarray(result.get("units", "{}")).item()
            if isinstance(raw_units, bytes):
                raw_units = raw_units.decode("utf-8")
            units = json.loads(raw_units) if raw_units else {}
            if not isinstance(units, dict):
                raise ValueError("units must be an object")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{label}: invalid native units metadata.") from exc
        quantity = str(units.get("rcs_linear_quantity", "")).strip().lower()
        if not quantity:
            quantity = "sigma_2d" if str(units.get("rcs_log_unit", "dBsm")).lower() == "dbke" else "sigma_3d"
        scale = None
        if str(result.get("power_domain", "")) == "delta_amp_sq" and str(result.get("rcs_domain", "")) == "delta":
            scale = 1.
        elif quantity == "sigma_3d":
            scale = np.sqrt(4. * np.pi)
        elif quantity == "sigma_2d":
            unit = str(units.get("frequency", "GHz")).strip().lower()
            frequency_scale = {"hz": 1., "khz": 1.e3, "mhz": 1.e6, "ghz": 1.e9}.get(unit)
            if frequency_scale is None:
                raise ValueError(f"{label}: unsupported frequency units {unit!r}.")
            frequency = np.asarray(result["frequencies"], dtype=float)
            if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.):
                raise ValueError(f"{label}: frequencies must be finite and positive.")
            scale = (1. / (2. * np.sqrt(2. * np.pi * frequency * frequency_scale / C0)))[None, None, :, None]
        if domain == "raw_field":
            if scale is None:
                raise ValueError(f"{label}: unsupported raw field normalization {quantity!r}.")
            if "rcs_amp_real" in result or "rcs_amp_imag" in result:
                raise ValueError(f"{label}: duplicate native raw complex samples.")
            raw, physical = field, field * scale
        else:
            physical = field
            raw = field / scale if scale is not None else None
        with np.errstate(over="ignore", invalid="ignore"):
            result["rcs_power"] = np.abs(physical) ** 2
        result["rcs_phase"] = np.angle(physical)
        if units.get("phase_wrap") == "0_360":
            result["rcs_phase"] = np.mod(result["rcs_phase"], 2. * np.pi)
        if raw is not None and not (domain == "physical" and
                                   ("rcs_amp_real" in result or "rcs_amp_imag" in result)):
            # Keep tiny quadrature components; never reconstruct from phase.
            result["rcs_amp_real"], result["rcs_amp_imag"] = raw.real, raw.imag
            result["raw_complex_amplitude_preserved"] = np.asarray(True)
        if str(result.get("rcs_domain", "")) == "complex_amplitude":
            result["rcs_domain"] = np.asarray("power_phase")
    elif kind == "power":
        power = np.asarray(result["rcs_power"])
        if power.shape != shape or power.dtype.kind not in "iuf":
            raise ValueError(f"{label}: invalid native power shape or dtype.")
        result["rcs_phase"] = np.full(shape, np.nan, dtype=power.dtype if power.dtype.kind == "f" else np.float64)
    for key in ("grim_format_version", "storage_kind", "storage_domain"):
        result.pop(key, None)
    return result
