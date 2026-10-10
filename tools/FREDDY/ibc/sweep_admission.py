"""Count-first desktop sweep admission, shared by GUI and calculation services.

These are deterministic per-study budgets, not a promise about available system
RAM. Estimates include boxed scalar/list storage for the non-NumPy path and
working/captured result copies. Existing streamed candidate/trial loops remain
streamed; their *retained* caches and result products must fit the same budget.
"""
from __future__ import annotations

import math
import operator

# Match the scales already used by IBC Batch and Sensitivity respectively.
MAX_SWEEP_POINTS = 1_000_000
MAX_RESPONSE_POINTS = 2_000_000
MAX_STUDY_BYTES = 512 * 1024 * 1024
_SCALAR_BYTES = 32  # Python float plus its list slot; also bounds float64 arrays.


def sweep_count(start: float, stop: float, step: float) -> int:
    """Match make_sweep's anchored endpoints without constructing the values."""
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("Sweep start, stop, and step must be finite.")
    if step <= 0:
        raise ValueError("Sweep step must be > 0.")
    if stop < start:
        raise ValueError("Sweep stop must be >= start.")
    intervals = (stop - start) / step
    if not math.isfinite(intervals):
        raise ValueError(
            "Sweep point count exceeds the supported desktop range. "
            "Reduce the range or increase the step."
        )
    return math.floor(intervals + 1e-12) + 1


def frequency_sweep_count(start: float, stop: float, step: float) -> int:
    if not math.isfinite(start) or start <= 0:
        raise ValueError("Frequency start must be finite and > 0 GHz.")
    return sweep_count(start, stop, step)


def _positive_count(value: int, label: str) -> int:
    count = _nonnegative_count(value, label)
    if count == 0:
        raise ValueError(f"{label} count must be a positive integer.")
    return count


def _nonnegative_count(value: int, label: str) -> int:
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{label} count must be a nonnegative integer.") from None
    if isinstance(value, bool) or count < 0:
        raise ValueError(f"{label} count must be a nonnegative integer.")
    return count


def validate_axis_count(count: int, label: str = "Sweep") -> int:
    count = _positive_count(count, label)
    if count > MAX_SWEEP_POINTS:
        raise ValueError(
            f"{label} requests {count:,} points; the desktop limit is "
            f"{MAX_SWEEP_POINTS:,} per axis. Reduce the range or increase the step."
        )
    return count


def validate_grid(frequency_count: int, other_count: int = 1, *,
                  label: str = "Analysis", metric_count: int = 8,
                  retained_grids: int = 2, layer_count: int = 1,
                  extra_bytes: int = 0) -> int:
    """Reject axes/products/storage before interpolation or result allocation.

    Two nominal grids allow for conversion into captured display results.
    Working space includes vector solver temporaries and the four complex wave
    arrays per layer (including the finite cutoff couplings), plus material
    lists. Extra storage covers workflow-specific retained caches.
    """
    nf = validate_axis_count(frequency_count, f"{label} frequency sweep")
    nx = validate_axis_count(other_count, f"{label} secondary sweep")
    points = nf * nx  # Python integers cannot overflow during this check.
    if points > MAX_RESPONSE_POINTS:
        raise ValueError(
            f"{label} requests {points:,} response points ({nf:,} x {nx:,}); "
            f"the desktop limit is {MAX_RESPONSE_POINTS:,}. "
            "Reduce a range or increase a frequency/angle/thickness step."
        )
    metrics = _positive_count(metric_count, "Metric")
    grids = _positive_count(retained_grids, "Retained grid")
    layer_count = _nonnegative_count(layer_count, "Layer")
    extra_bytes = _nonnegative_count(extra_bytes, "Extra storage")
    estimate = (points * metrics * grids * _SCALAR_BYTES
                + nf * (512 + layer_count * 192)
                + (nf + nx) * _SCALAR_BYTES + extra_bytes)
    if estimate > MAX_STUDY_BYTES:
        whole_mib, decimal_mib = divmod(estimate * 10 // 1024**2, 10)
        raise ValueError(
            f"{label} needs an estimated {whole_mib:,}.{decimal_mib} MiB for "
            f"{nf:,} x {nx:,} points and retained results; the desktop study "
            f"limit is {MAX_STUDY_BYTES / 1024**2:g} MiB. Increase the step, "
            "reduce the range/layers, or retain fewer comparison results."
        )
    return estimate


def validate_analysis_grid(frequency_count: int, other_count: int = 1, *,
                           layer_count: int = 1, uncertainty: bool = False,
                           polarizations: int = 1, label: str = "Analysis") -> int:
    # Nominal + min/max + current corner, and a copy of each retained result.
    grids = (8 if uncertainty else 2) * _positive_count(polarizations, "Polarization")
    return validate_grid(frequency_count, other_count, label=label,
                         retained_grids=grids, layer_count=layer_count)


def validate_inverse_grid(frequency_count: int, angle_count: int, *,
                          layer_count: int, case_count: int, design_count: int,
                          top_n: int) -> int:
    frequency_count = validate_axis_count(frequency_count, "Inverse frequency sweep")
    angle_count = validate_axis_count(angle_count, "Inverse angle sweep")
    layer_count = _nonnegative_count(layer_count, "Layer")
    cases = _positive_count(case_count, "Tolerance case")
    designs = _positive_count(design_count, "Design")
    kept = min(designs, _positive_count(top_n, "Keep best"))
    # Each cached layer has Zc, kz and two cutoff couplings (complex128).
    # Scores have five doubles; checkpoints may copy them. Comparison samples
    # are boxed floats per angle/case/candidate, with one plotting copy.
    extra = (frequency_count * angle_count * layer_count * cases * 64
             + designs * 5 * 8 * 2
             + frequency_count * angle_count * cases * kept * _SCALAR_BYTES * 2)
    return validate_grid(frequency_count, angle_count, label="Inverse design",
                         layer_count=layer_count, extra_bytes=extra)


def validate_mix_grid(frequency_count: int, angle_count: int = 1, *,
                      component_count: int = 1, retained_results: int = 1) -> int:
    frequency_count = validate_axis_count(frequency_count, "Material Mix frequency sweep")
    angle_count = validate_axis_count(angle_count, "Material Mix angle sweep")
    component_count = _nonnegative_count(component_count, "Component")
    kept = _positive_count(retained_results, "Keep best")
    # Property curves/targets plus retained recipe performance plots; component
    # complex property lists need 48 bytes per boxed complex plus list slot.
    extra = (frequency_count * component_count * 2 * 48
             + frequency_count * kept * 20 * _SCALAR_BYTES)
    return validate_grid(frequency_count, angle_count, label="Material Mix",
                         metric_count=2, retained_grids=2 * kept,
                         extra_bytes=extra)
