"""Power and phase comparison for duplicate imported samples."""
from __future__ import annotations

import numpy as np


def _cst_samples_equivalent(left_power, left_phase, right_power, right_phase):
    """Return True when two seam/duplicate rows encode the same field."""

    if not np.isclose(
        float(left_power), float(right_power), rtol=1.0e-8, atol=1.0e-12
    ):
        return False


    if float(left_power) == 0.0 and float(right_power) == 0.0:
        return True
    left_phase = float(left_phase)
    right_phase = float(right_phase)
    if np.isnan(left_phase) and np.isnan(right_phase):
        return True
    if not (np.isfinite(left_phase) and np.isfinite(right_phase)):
        return False
    phase_error = np.angle(np.exp(1j * (left_phase - right_phase)))
    return abs(float(phase_error)) <= 1.0e-8


_PARSED_SAMPLE_BLOCK_CELLS = 32768


def parsed_sample_memory(shape, *, sample_dtype=np.float64,
                         duplicate_bytes_per_cell=8, extra_working_bytes=0):
    """Plan typed parser arrays, axis maps, and bounded canonicalization scratch."""
    cells = int(np.prod(shape, dtype=object))
    component_bytes = np.dtype(sample_dtype).itemsize
    dense_bytes = cells * 2 * component_bytes
    # Axis arrays/maps and CSV row/header buffers. No per-cell Python records
    # survive discovery. Typed duplicate state is released before conversion.
    bookkeeping = 128 * sum(shape) + 128 * 1024
    # Conversion holds a complex output plus indexed power/phase, square root,
    # complex exponential, product, and masks for at most one small block.
    scratch = min(cells, _PARSED_SAMPLE_BLOCK_CELLS) * 8 * component_bytes
    resident = max(cells * duplicate_bytes_per_cell + extra_working_bytes,
                   dense_bytes + scratch) + bookkeeping
    return dense_bytes, dense_bytes + resident


def construct_parsed_grid(cls, *axes, **kwargs):
    """Adopt validated parser buffers and canonicalize recoverable samples in blocks.

    Phase wrapping follows RcsGrid exactly. Missing magnitude clears phase;
    unknown phase, zero-field phase, and unwrapped turns stay lossless.
    The arrays must be fresh parser-owned arrays with nonnegative power or NaN.
    """
    from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN

    power, phase = kwargs["rcs_power"], kwargs["rcs_phase"]
    kwargs["_adopt_clean_arrays"] = _ADOPT_CLEAN_ARRAYS_TOKEN
    kwargs["rcs_domain"] = kwargs.get("rcs_domain") or "power_phase"
    phase_wrap = str((kwargs.get("units") or {}).get("phase_wrap", "")).strip()
    if phase_wrap and phase_wrap not in {"0_360", "-180_180"}:
        raise ValueError("phase_wrap must be '0_360' or '-180_180' when declared")
    has_phase, recoverable = False, True
    block_cells = _PARSED_SAMPLE_BLOCK_CELLS
    for start in range(0, power.size, block_cells):
        p = power.reshape(-1)[start:start + block_cells]
        q = phase.reshape(-1)[start:start + block_cells]
        finite = np.isfinite(p)
        # Match RcsGrid cleaning, including the sign of an explicit -0 power.
        np.maximum(p, 0.0, out=p)
        q[~finite] = np.nan
        finite_phase = np.isfinite(q)
        if phase_wrap:
            if phase_wrap == "0_360":
                outside = finite_phase & ((q < 0) | (q >= 2 * np.pi))
                wrapped = np.remainder(q[outside], 2 * np.pi)
                wrapped[wrapped >= 2 * np.pi] = 0
            else:
                outside = finite_phase & ((q < -np.pi) | (q >= np.pi))
                wrapped = (q[outside] + np.pi) % (2 * np.pi) - np.pi
                wrapped[wrapped >= np.pi] = -np.pi
            q[outside] = wrapped
        has_phase = has_phase or bool(np.any(finite & finite_phase))
        out_of_range = ((q < 0) | (q >= 2 * np.pi)) if phase_wrap == "0_360" else (
            (q < -np.pi) | (q > np.pi))
        if (np.any(finite & ~finite_phase) or np.any((p == 0) & (q != 0))
                or np.any(finite_phase & out_of_range)):
            recoverable = False
    # Loop temporaries must not coexist with the conversion workspace.
    if power.size:
        del p, q, finite, finite_phase, out_of_range
        if phase_wrap:
            del outside, wrapped
    if not has_phase:
        kwargs.pop("rcs_phase")
    elif recoverable:
        field = np.empty(power.shape, dtype=(
            np.complex128 if power.dtype.itemsize > 4 else np.complex64
        ))
        for start in range(0, power.size, block_cells):
            end = start + block_cells
            field.reshape(-1)[start:end] = cls._complex_from_power_phase(
                power.reshape(-1)[start:end], phase.reshape(-1)[start:end])
        kwargs.pop("rcs_power")
        kwargs.pop("rcs_phase")
        kwargs["rcs"] = field
    return cls(*axes, **kwargs)
