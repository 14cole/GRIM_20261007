"""Dataset copying, regridding, decimation, offsets, and complex division."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid


_MAX_EXPLICIT_AXIS_POINTS = 1_000_000


def _derived_response_extra(dataset: RcsGrid) -> dict:
    """Keep response-independent ancillary models, dropping stale grid claims."""

    stale = {
        "amplitude_convention",
        "power_domain",
        "rcs_domain",
        "solver_metadata_json",
        "solver_certification",
        "production_mesh_certification_json",
        "source_body_mesh_certification_json",
        "requested_radar_grid_json",
        "rcs_amp_real",
        "rcs_amp_imag",
    }
    original_shape = tuple(dataset.rcs_power.shape)
    extra = {}
    for key, value in (dataset.extra or {}).items():
        if key in stale:
            continue
        array = np.asarray(value)
        if array.ndim >= 4 and tuple(array.shape[:4]) == original_shape:
            continue
        if isinstance(value, np.ndarray):


            shared = value.view()
            shared.setflags(write=False)
            extra[key] = shared
        else:
            extra[key] = copy.deepcopy(value)
    return extra


def _copy_grid(
    dataset: RcsGrid,
    *,
    rcs=None,
    rcs_power,
    rcs_phase=None,
    rcs_domain=None,
    units=None,
) -> RcsGrid:
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs=rcs,
        rcs_power=rcs_power,
        rcs_phase=(
            None
            if rcs is not None
            else dataset.rcs_phase if rcs_phase is None else rcs_phase
        ),
        rcs_domain=rcs_domain or dataset.rcs_domain,
        source_path=dataset.source_path,
        history=dataset.history,
        units=dict(dataset.units or {}) if units is None else dict(units),
        extra=_derived_response_extra(dataset),
    )


def _authoritative_response_phase(dataset: RcsGrid) -> np.ndarray:
    """Return the phase from validated raw real/imaginary arrays or stored phase.

    Raw arrays use ``atan2``; a power/phase grid returns its stored phase as a view.
    """

    raw_pair = dataset._complete_authoritative_raw_arrays()
    if raw_pair is None:
        return dataset.rcs_phase
    real, imag = raw_pair
    phase = np.empty(dataset.rcs_power.shape, dtype=np.float64)
    np.arctan2(
        np.asarray(imag, dtype=np.float64),
        np.asarray(real, dtype=np.float64),
        out=phase,
    )
    return phase


def _phase_shifted_response(dataset: RcsGrid, phase_degrees: float) -> np.ndarray:
    """Return authoritative phase shifted into the canonical signed interval."""

    phase = np.array(_authoritative_response_phase(dataset), copy=True)
    delta = np.deg2rad(float(phase_degrees))
    with np.errstate(invalid="ignore"):
        np.add(phase, delta + np.pi, out=phase)
        np.remainder(phase, 2.0 * np.pi, out=phase)
        np.subtract(phase, np.pi, out=phase)
    return phase


def duplicate_dataset(dataset: RcsGrid) -> RcsGrid:
    """Return a detached copy with the same numerical samples and metadata."""

    return RcsGrid(
        dataset.azimuths.copy(),
        dataset.elevations.copy(),
        dataset.frequencies.copy(),
        dataset.polarizations.copy(),
        rcs_power=dataset.rcs_power.copy(),
        rcs_phase=dataset.rcs_phase.copy(),
        rcs_domain=dataset.rcs_domain,
        source_path=dataset.source_path,
        history=dataset.history,
        units=copy.deepcopy(dataset.units or {}),
        extra=copy.deepcopy(dataset.extra or {}),
    )


def _positive_stride(value, name: str) -> int:
    """Validate a replayed physical-axis stride without silently rounding it."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    stride = int(value)
    if stride <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return stride


def crop_dataset(
    dataset: RcsGrid,
    *,
    azimuths=None,
    elevations=None,
    frequencies=None,
    azimuth_range=None,
    elevation_range=None,
    frequency_range=None,
    azimuth_stride: int = 1,
    elevation_stride: int = 1,
    frequency_stride: int = 1,
    polarizations=None,
) -> RcsGrid:
    """Select axis values or numeric ranges, then retain every Nth sample.

    Ranges use the dataset axis units. ``RcsGrid.axis_crop`` applies the metadata
    and sample-selection rules. Striding selects samples without filtering.
    """

    strides = {
        "azimuth": _positive_stride(azimuth_stride, "azimuth_stride"),
        "elevation": _positive_stride(elevation_stride, "elevation_stride"),
        "frequency": _positive_stride(frequency_stride, "frequency_stride"),
    }
    ranged = dataset.axis_crop(
        azimuths=azimuths,
        elevations=elevations,
        frequencies=frequencies,
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        frequency_range=frequency_range,
        polarizations=polarizations,
    )
    if all(value == 1 for value in strides.values()):
        return ranged
    return ranged.axis_crop(
        azimuths=np.asarray(ranged.azimuths)[:: strides["azimuth"]],
        elevations=np.asarray(ranged.elevations)[:: strides["elevation"]],
        frequencies=np.asarray(ranged.frequencies)[:: strides["frequency"]],
        polarizations=np.asarray(ranged.polarizations),
    )


def regrid_axis(
    dataset: RcsGrid,
    axis: str,
    *,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
    values=None,
) -> RcsGrid:
    """Interpolate one numeric axis onto the requested coordinates.

    Supply either ``values`` or all of ``start``, ``stop``, and ``step`` in the
    dataset axis unit. ``RcsGrid.interpolate_axis`` applies complex/linear
    interpolation and rejects extrapolation.
    """

    aliases = {
        "az": "azimuth",
        "azimuths": "azimuth",
        "el": "elevation",
        "elevations": "elevation",
        "freq": "frequency",
        "frequencies": "frequency",
    }
    axis_name = str(axis).strip().lower()
    axis_name = aliases.get(axis_name, axis_name)
    if axis_name not in {"azimuth", "elevation", "frequency"}:
        raise ValueError("axis must be 'azimuth', 'elevation', or 'frequency'")

    bounds_supplied = any(item is not None for item in (start, stop, step))
    if values is not None and bounds_supplied:
        raise ValueError("provide either values or start/stop/step, not both")
    if values is None:
        if any(item is None for item in (start, stop, step)):
            raise ValueError("start, stop, and step are required when values is omitted")
        start_value = float(start)
        stop_value = float(stop)
        step_value = float(step)
        if not all(math.isfinite(item) for item in (start_value, stop_value, step_value)):
            raise ValueError("start, stop, and step must be finite")
        if step_value <= 0.0:
            raise ValueError("step must be positive")
        if stop_value < start_value:
            raise ValueError("stop must be greater than or equal to start")
        count_float = math.floor(
            (stop_value - start_value) / step_value + 1.0e-12
        ) + 1
        if count_float > _MAX_EXPLICIT_AXIS_POINTS:
            raise ValueError(
                f"requested grid has {count_float:,} points; the safety limit is "
                f"{_MAX_EXPLICIT_AXIS_POINTS:,}"
            )
        target = start_value + step_value * np.arange(count_float, dtype=float)
    else:
        target = np.asarray(values, dtype=float).ravel()
        if target.size > _MAX_EXPLICIT_AXIS_POINTS:
            raise ValueError(
                f"requested grid has {target.size:,} points; the safety limit is "
                f"{_MAX_EXPLICIT_AXIS_POINTS:,}"
            )

    if target.size == 0:
        raise ValueError("target grid must contain at least one value")
    if not np.all(np.isfinite(target)):
        raise ValueError("target grid values must be finite")
    if target.size > 1 and np.any(np.diff(target) <= 0.0):
        raise ValueError("target grid values must be strictly increasing")
    return dataset.interpolate_axis(axis_name, target)


def join_datasets(*datasets: RcsGrid, tol: float = 1.0e-6) -> RcsGrid:
    """Join grids with the GUI's conflict and available-memory policies."""

    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        available = None
    maximum = int(available * 0.5) if available is not None else None
    return RcsGrid.join_many(
        *datasets,
        tol=float(tol),
        overlap="error",
        max_output_bytes=maximum,
    )


def stitch_datasets(
    *datasets: RcsGrid,
    policy: str = "priority-first",
    tol: float = 1.0e-6,
    metadata_attested: bool = False,
    max_output_bytes=None,
    return_report: bool = False,
):
    """Stitch compatible datasets using the core method and its provenance.

    The return value is an :class:`RcsGrid` unless ``return_report`` is true,
    in which case it is the exact ``RcsGrid.stitch_many`` result.
    """

    return RcsGrid.stitch_many(
        *datasets,
        policy=policy,
        tol=float(tol),
        metadata_attested=metadata_attested,
        max_output_bytes=max_output_bytes,
        return_report=return_report,
    )


def shift_dataset(
    dataset: RcsGrid,
    *,
    azimuth_degrees: float | None = None,
    elevation_degrees: float | None = None,
    phase_degrees: float | None = None,
) -> RcsGrid:
    """Apply the GUI Shift operation with explicit enabled values."""

    result = dataset
    if azimuth_degrees is not None:
        result = result.shift_azimuth(float(azimuth_degrees))
    if elevation_degrees is not None:
        result = result.shift_elevation(float(elevation_degrees))
    if phase_degrees is not None:
        result = _copy_grid(
            result,
            rcs_power=result.rcs_power,
            rcs_phase=_phase_shifted_response(result, float(phase_degrees)),
            rcs_domain="complex_amplitude",
        )
    return result


def wrap_phase(dataset: RcsGrid, mode: str = "-180_180") -> RcsGrid:
    """Wrap stored phase using the user-facing degree interval names."""

    mode_name = str(mode).strip()
    if mode_name not in {"-180_180", "0_360"}:
        raise ValueError("mode must be '-180_180' or '0_360'")
    return dataset.wrap_phase(mode=mode_name)


def offset_db(dataset: RcsGrid, value_db: float) -> RcsGrid:
    """Shift all displayed RCS values by a constant number of decibels."""

    scale = 10.0 ** (float(value_db) / 10.0)
    return _copy_grid(
        dataset,
        rcs_power=dataset.rcs_power * scale,
        rcs_phase=_authoritative_response_phase(dataset),
        rcs_domain=dataset.rcs_domain,
    )


def coherent_divide(
    numerator: RcsGrid,
    denominator: RcsGrid,
    *,
    metadata_attested: bool = False,
) -> RcsGrid:
    """Complex element-wise division using the GUI's finite/nonzero policy."""

    if not isinstance(metadata_attested, (bool, np.bool_)):
        raise TypeError("metadata_attested must be True or False")
    numerator._assert_compatible(
        denominator,
        coherent=True,
        coherent_metadata_attested=metadata_attested,
        _scan_phase_samples=False,
    )
    shape = tuple(int(value) for value in numerator.rcs_power.shape)
    raw_response = (
        numerator._complete_authoritative_raw_arrays() is not None
        or denominator._complete_authoritative_raw_arrays() is not None
    )
    real_dtype = (
        np.dtype(np.float64)
        if raw_response
        else np.result_type(
            numerator.rcs_power.dtype,
            numerator.rcs_phase.dtype,
            denominator.rcs_power.dtype,
            denominator.rcs_phase.dtype,
        )
    )
    complex_dtype = (
        np.dtype(np.complex128)
        if real_dtype.itemsize > np.dtype(np.float32).itemsize
        else np.dtype(np.complex64)
    )
    result_power = np.full(shape, np.nan, dtype=real_dtype)
    result_phase = np.full(shape, np.nan, dtype=real_dtype)


    cells_per_azimuth = max(1, int(np.prod(shape[1:], dtype=np.int64)))
    working_bytes_per_cell = (
        3 * complex_dtype.itemsize + real_dtype.itemsize + 3
    )
    rows_per_block = max(
        1,
        (16 * 1024**2) // (cells_per_azimuth * working_bytes_per_cell),
    )
    for start in range(0, shape[0], rows_per_block):
        stop = min(shape[0], start + rows_per_block)
        selection = (slice(start, stop), slice(None), slice(None), slice(None))
        left = np.asarray(numerator.rcs_slice(selection), dtype=complex_dtype)
        right = np.asarray(denominator.rcs_slice(selection), dtype=complex_dtype)
        quotient = np.full(left.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        valid = np.isfinite(left) & np.isfinite(right) & (right != 0)
        np.divide(left, right, out=quotient, where=valid)
        power_block = result_power[selection]
        np.multiply(quotient.real, quotient.real, out=power_block)
        power_block += quotient.imag * quotient.imag
        np.arctan2(
            quotient.imag,
            quotient.real,
            out=result_phase[selection],
        )
        finite_result = np.isfinite(power_block) & np.isfinite(
            result_phase[selection]
        )
        power_block[~finite_result] = np.nan
        result_phase[selection][~finite_result] = np.nan
    usable_count = int(np.count_nonzero(np.isfinite(result_power)))
    if usable_count == 0:
        raise ValueError(
            "coherent division has no common usable complex samples with a "
            "nonzero denominator"
        )
    units = dict(numerator.units or {})
    units["rcs_log_unit"] = "dB"
    units["rcs_linear_quantity"] = "power_ratio"


    extra = {}
    for key in ("time_convention", "polarization_basis"):
        left_value = numerator._declared_scalar_metadata(key)
        right_value = denominator._declared_scalar_metadata(key)
        if left_value and right_value:
            extra[key] = left_value
    history, attestation_extra = numerator._coherent_attestation_provenance(
        (denominator,),
        operation="coherent-divide",
        metadata_attested=metadata_attested,
    )
    if history is None:
        history = numerator.history
    if attestation_extra:


        attestation_extra.pop("phase_reference", None)
        extra.update(attestation_extra)
    extra["coherent_sample_qa_json"] = json.dumps(
        {
            "schema": "grim.coherent-sample-qa.v1",
            "operation": "coherent-divide",
            "total_sample_count": int(result_power.size),
            "usable_sample_count": usable_count,
            "masked_sample_count": int(result_power.size - usable_count),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    extra["amplitude_convention"] = "complex field ratio"
    return RcsGrid(
        numerator.azimuths,
        numerator.elevations,
        numerator.frequencies,
        numerator.polarizations,
        rcs_power=result_power,
        rcs_phase=result_phase,
        rcs_domain="complex_amplitude",
        source_path=numerator.source_path,
        history=history,
        units=units,
        extra=extra,
    )


def convert_extrusion(dataset: RcsGrid, *, to: str, length_m: float) -> RcsGrid:
    """Convert between sigma_3d/dBsm and sigma_2d/dBke for an extrusion."""

    destination = str(to).strip().lower()
    if destination not in {"dbke", "dbsm"}:
        raise ValueError("to must be 'dbke' or 'dbsm'")
    length = float(length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length_m must be positive and finite")
    source_quantity = str(dataset.linear_quantity()).strip().lower()
    source_log_unit = str(dataset.default_log_unit()).strip().lower()
    if destination == "dbke":
        if source_quantity != "sigma_3d" or source_log_unit != "dbsm":
            raise ValueError(
                "dBsm→dBke requires a sigma_3d/dBsm dataset; "
                f"received {source_quantity or 'unknown'}/{source_log_unit or 'unknown'}"
            )
    elif source_quantity != "sigma_2d" or source_log_unit != "dbke":
        raise ValueError(
            "dBke→dBsm requires a sigma_2d/dBke dataset; "
            f"received {source_quantity or 'unknown'}/{source_log_unit or 'unknown'}"
        )
    frequency_hz = np.asarray(dataset._frequency_value_to_hz(dataset.frequencies), dtype=float)
    c0 = 299_792_458.0
    if destination == "dbke":
        scale = c0 / (2.0 * length * length * frequency_hz)
        quantity = "sigma_2d"
    else:
        scale = 2.0 * length * length * frequency_hz / c0
        quantity = "sigma_3d"
    scale = np.where(np.isfinite(frequency_hz) & (frequency_hz > 0.0), scale, np.nan)
    scale_4d = scale.reshape(1, 1, -1, 1)
    power = dataset.rcs_power * scale_4d
    units = dict(dataset.units or {})
    units["rcs_log_unit"] = "dBke" if destination == "dbke" else "dBsm"
    units["rcs_linear_quantity"] = quantity
    if dataset.rcs_domain == "complex_amplitude":
        rcs = dataset.rcs * np.sqrt(np.maximum(scale_4d, 0.0))
        return RcsGrid(
            dataset.azimuths,
            dataset.elevations,
            dataset.frequencies,
            dataset.polarizations,
            rcs,
            rcs_power=power,
            rcs_domain=dataset.rcs_domain,
            source_path=dataset.source_path,
            history=dataset.history,
            units=units,
            extra=_derived_response_extra(dataset),
        )
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs=None,
        rcs_power=power,
        rcs_phase=dataset.rcs_phase,
        rcs_domain=dataset.rcs_domain,
        source_path=dataset.source_path,
        history=dataset.history,
        units=units,
        extra=_derived_response_extra(dataset),
    )


def decimate_axis(
    dataset: RcsGrid,
    *,
    axis: str,
    factor: int,
    mode: str = "power",
    metadata_attested: bool = False,
) -> RcsGrid:
    """Boxcar-prefilter and decimate one numeric axis by an integer factor.

    ``mode='power'`` averages finite linear power and marks phase unknown.
    ``mode='coherent'`` averages the finite complex field and retains the
    resulting phase. The final partial bin is retained with its actual count.
    """

    axis_key = str(axis).strip().lower()
    axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2}
    if axis_key not in axis_map:
        raise ValueError("axis must be azimuth, elevation, or frequency")
    if isinstance(factor, (bool, np.bool_)) or not isinstance(
        factor, (int, np.integer)
    ):
        raise TypeError("factor must be an integer of at least 2")
    factor = int(factor)
    if factor < 2:
        raise ValueError("factor must be an integer of at least 2")
    mode_key = str(mode).strip().lower().replace("_", "-")
    if mode_key not in {"power", "coherent"}:
        raise ValueError("mode must be 'power' or 'coherent'")
    if not isinstance(metadata_attested, (bool, np.bool_)):
        raise TypeError("metadata_attested must be True or False")

    axis_index = axis_map[axis_key]
    source_axis = np.asarray(dataset.get_axis(axis_key), dtype=float)
    if source_axis.size < 2 or np.any(~np.isfinite(source_axis)):
        raise ValueError("decimation requires at least two finite axis coordinates")
    source_steps = np.diff(source_axis)
    if np.any(source_steps <= 0.0):
        raise ValueError("decimation requires a strictly increasing source axis")
    nominal_step = float(np.median(source_steps))
    spacing_atol = max(
        abs(nominal_step) * 1.0e-9,
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(source_axis))))
        * 32.0,
    )
    if not np.allclose(
        source_steps,
        nominal_step,
        rtol=1.0e-6,
        atol=spacing_atol,
    ):
        raise ValueError(
            "decimation requires uniformly spaced source coordinates; regrid "
            "to a uniform fine grid before applying a boxcar prefilter"
        )

    source_count = int(source_axis.size)
    full_bin_count, remainder = divmod(source_count, factor)
    output_axis_parts = []
    if full_bin_count:
        output_axis_parts.append(
            np.mean(
                source_axis[: full_bin_count * factor].reshape(
                    full_bin_count, factor
                ),
                axis=1,
            )
        )
    if remainder:
        output_axis_parts.append(
            np.asarray(
                [float(np.mean(source_axis[full_bin_count * factor :]))],
                dtype=float,
            )
        )
    output_axis = np.concatenate(output_axis_parts)
    output_shape = list(dataset.rcs_power.shape)
    output_shape[axis_index] = int(output_axis.size)
    output_shape = tuple(output_shape)
    output_power = np.full(output_shape, np.nan, dtype=np.float64)
    output_phase = np.full(output_shape, np.nan, dtype=np.float64)
    output_power_by_bin = np.moveaxis(output_power, axis_index, 0)
    output_phase_by_bin = np.moveaxis(output_phase, axis_index, 0)
    other_cell_count = max(
        1,
        int(np.prod(dataset.rcs_power.shape, dtype=np.int64)) // source_count,
    )
    transient_bytes_per_cell = 64 if mode_key == "coherent" else 24
    bins_per_block = max(
        1,
        (32 * 1024**2)
        // max(1, factor * other_cell_count * transient_bytes_per_cell),
    )

    def aggregate_bins(start: int, bin_count: int, samples_per_bin: int):
        stop = start + bin_count * samples_per_bin
        source_selection = [slice(None)] * 4
        source_selection[axis_index] = slice(start, stop)
        source_selection = tuple(source_selection)
        if mode_key == "coherent":
            field = np.asarray(dataset.rcs_slice(source_selection), dtype=np.complex128)
            field = np.moveaxis(field, axis_index, 0)
            field = field.reshape(
                (bin_count, samples_per_bin) + field.shape[1:]
            )
            valid = np.isfinite(field.real) & np.isfinite(field.imag)
            field_sum = np.sum(np.where(valid, field, 0.0 + 0.0j), axis=1)
            magnitude_sum = np.sum(
                np.where(valid, np.abs(field), 0.0), axis=1
            )
            count = np.sum(valid, axis=1)
            mean_field = np.full(field_sum.shape, np.nan + 1j * np.nan)
            np.divide(field_sum, count, out=mean_field, where=count > 0)
            mean_magnitude = np.zeros(magnitude_sum.shape, dtype=np.float64)
            np.divide(
                magnitude_sum,
                count,
                out=mean_magnitude,
                where=count > 0,
            )
            cancellation_floor = (
                32.0 * np.finfo(np.float64).eps * mean_magnitude
            )
            near_zero = (count > 0) & (
                np.abs(mean_field) <= cancellation_floor
            )
            mean_field[near_zero] = 0.0 + 0.0j
            power = mean_field.real * mean_field.real + mean_field.imag * mean_field.imag
            phase = np.angle(mean_field)
            phase[(count == 0) | (power == 0.0)] = np.nan
        else:
            power_block = np.asarray(dataset.rcs_power[source_selection], dtype=np.float64)
            power_block = np.moveaxis(power_block, axis_index, 0)
            power_block = power_block.reshape(
                (bin_count, samples_per_bin) + power_block.shape[1:]
            )
            valid = np.isfinite(power_block)
            power_sum = np.sum(
                np.where(valid, power_block, 0.0), axis=1
            )
            count = np.sum(valid, axis=1)
            power = np.full(power_sum.shape, np.nan, dtype=np.float64)
            np.divide(power_sum, count, out=power, where=count > 0)
            phase = np.full(power.shape, np.nan, dtype=np.float64)
        return power, phase

    for output_start in range(0, full_bin_count, bins_per_block):
        output_stop = min(full_bin_count, output_start + bins_per_block)
        power, phase = aggregate_bins(
            output_start * factor,
            output_stop - output_start,
            factor,
        )
        output_power_by_bin[output_start:output_stop] = power
        output_phase_by_bin[output_start:output_stop] = phase
    if remainder:
        power, phase = aggregate_bins(
            full_bin_count * factor,
            1,
            remainder,
        )
        output_power_by_bin[full_bin_count:] = power
        output_phase_by_bin[full_bin_count:] = phase

    axes = [
        np.array(dataset.azimuths, copy=True),
        np.array(dataset.elevations, copy=True),
        np.array(dataset.frequencies, copy=True),
        np.array(dataset.polarizations, copy=True),
    ]
    axes[axis_index] = output_axis
    attestation_history = None
    attestation_extra = None
    if mode_key == "coherent":
        attestation_history, attestation_extra = (
            dataset._coherent_attestation_provenance(
                (),
                operation=f"decimate-{axis_key}",
                metadata_attested=metadata_attested,
            )
        )
    extra = dataset._derived_response_extra(
        operation=f"decimate-{axis_key}-{mode_key}",
        coherent=(mode_key == "coherent"),
        attestation_extra=attestation_extra,
    )
    dataset._invalidate_assembly_sampling_hash(
        extra, f"decimate-{axis_key}-{mode_key}"
    )
    extra["decimation_json"] = json.dumps(
        {
            "schema": "grim.decimation.v1",
            "axis": axis_key,
            "factor": factor,
            "filter": "finite boxcar mean",
            "mode": mode_key,
            "partial_final_bin_retained": bool(source_axis.size % factor),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    history_entry = (
        f"Decimate {axis_key} by {factor} with finite boxcar {mode_key} mean"
    )
    if source_axis.size % factor:
        history_entry += "; retained final partial bin"
    prior_history = (
        attestation_history
        if attestation_history is not None
        else dataset.history
    )
    history = (
        f"{prior_history}\n{history_entry}" if prior_history else history_entry
    )
    return RcsGrid(
        axes[0],
        axes[1],
        axes[2],
        axes[3],
        rcs_power=output_power,
        rcs_phase=output_phase,
        rcs_domain="power_phase",
        source_path=dataset.source_path,
        history=history,
        units=dict(dataset.units or {}),
        extra=extra,
    )


def medianize_azimuth(
    dataset: RcsGrid,
    *,
    window_degrees: float,
    slide_degrees: float,
    periodic: bool | None = None,
    seam_conflict: str = "first",
    return_report: bool = False,
):
    """Medianize linear power over sliding physical-azimuth windows.

    The public parameters are always degrees. The returned coordinate retains
    the source dataset's native angular unit. A nearly complete 360-degree cut
    is treated as periodic by default, allowing windows to cross its seam.
    A closed cut (endpoints 360 degrees apart) counts that direction once; where
    the closing repeat disagrees with the opening sample, ``seam_conflict``
    ``"first"`` (default) keeps the opening sample and records the discarded
    cells in history, while ``"error"`` rejects the cut instead.
    ``return_report=True`` returns ``(grid, report)``.
    """

    policy = str(seam_conflict).strip().lower()
    if policy not in {"first", "error"}:
        raise ValueError("seam_conflict must be 'first' or 'error'")
    window = float(window_degrees)
    slide = float(slide_degrees)
    if not math.isfinite(window) or not math.isfinite(slide) or window <= 0.0 or slide <= 0.0:
        raise ValueError("window_degrees and slide_degrees must be positive and finite")
    unit_raw = str((dataset.units or {}).get("azimuth", "deg")).strip().lower()
    if unit_raw in {"deg", "degree", "degrees"}:
        angle_unit = "deg"
    elif unit_raw in {"rad", "radian", "radians"}:
        angle_unit = "rad"
    else:
        raise ValueError(f"unsupported azimuth unit {unit_raw!r}; use deg or rad")
    native_azimuth = np.asarray(dataset.azimuths, dtype=float)
    azimuth = np.rad2deg(native_azimuth) if angle_unit == "rad" else native_azimuth.copy()
    if azimuth.size < 2:
        raise ValueError("medianize requires at least two azimuth samples")
    if not np.all(np.isfinite(azimuth)) or np.any(np.diff(azimuth) <= 0.0):
        raise ValueError("medianize requires a finite, strictly increasing azimuth axis")

    typical_step = float(np.median(np.diff(azimuth)))
    span = float(azimuth[-1] - azimuth[0])
    if periodic is None:
        periodic = span >= 360.0 - max(1.5 * typical_step, 1.0e-6)
    periodic = bool(periodic)
    if periodic and window > 360.0 + 1.0e-9:
        raise ValueError("a periodic median window cannot exceed 360 degrees")

    half = window * 0.5
    sample_azimuth = azimuth
    merged_seam_power = None
    discarded_seam_cells = 0
    if periodic and np.isclose(span, 360.0, rtol=0.0, atol=1.0e-7):


        merged_seam_power = np.array(dataset.rcs_power[0], copy=True)
        merged_seam_phase = np.array(dataset.rcs_phase[0], copy=True)
        # The closing endpoint re-measures the opening direction; it only
        # fills cells the opening sample left missing.
        seam_conflict_mask = RcsGrid._merge_equivalent_sample_blocks(
            merged_seam_power,
            merged_seam_phase,
            dataset.rcs_power[-1],
            dataset.rcs_phase[-1],
            context="medianize periodic endpoint",
            keep_existing_on_conflict=policy == "first",
        )
        discarded_seam_cells = int(np.count_nonzero(seam_conflict_mask))
        sample_azimuth = azimuth[:-1]
    if periodic:
        count_float = np.ceil(360.0 / slide - 1.0e-12)
        if count_float > _MAX_EXPLICIT_AXIS_POINTS:
            raise ValueError(
                f"medianize would create {int(count_float):,} azimuths; "
                f"the safety limit is {_MAX_EXPLICIT_AXIS_POINTS:,}"
            )
        centers = float(azimuth[0]) + np.arange(int(count_float), dtype=float) * slide
        centers = centers[centers < float(azimuth[0]) + 360.0 - 1.0e-10]
        source_indices = np.arange(sample_azimuth.size, dtype=np.int64)
        augmented_azimuth = np.concatenate(
            (sample_azimuth - 360.0, sample_azimuth, sample_azimuth + 360.0)
        )
        augmented_indices = np.tile(source_indices, 3)
    else:
        first = float(azimuth[0]) + half
        last = float(azimuth[-1]) - half
        if last < first:
            centers = np.asarray([(float(azimuth[0]) + float(azimuth[-1])) * 0.5])
        else:
            count_float = np.floor((last - first) / slide + 1e-9) + 1.0
            if count_float > _MAX_EXPLICIT_AXIS_POINTS:
                raise ValueError(
                    f"medianize would create {int(count_float):,} azimuths; "
                    f"the safety limit is {_MAX_EXPLICIT_AXIS_POINTS:,}"
                )
            centers = first + np.arange(int(count_float), dtype=float) * slide
    shape = (centers.size,) + dataset.rcs_power.shape[1:]
    power = np.empty(shape, dtype=dataset.rcs_power.dtype)
    for index, center in enumerate(centers):
        if periodic:
            left = int(np.searchsorted(augmented_azimuth, center - half, side="left"))
            right = int(np.searchsorted(augmented_azimuth, center + half, side="right"))
            samples = np.unique(augmented_indices[left:right])
        else:
            left = int(np.searchsorted(azimuth, center - half, side="left"))
            right = int(np.searchsorted(azimuth, center + half, side="right"))
            samples = np.arange(left, right, dtype=np.int64)
        if samples.size == 0:
            if periodic:
                distance = np.abs(
                    (sample_azimuth - center + 180.0) % 360.0 - 180.0
                )
            else:
                distance = np.abs(azimuth - center)
            samples = np.asarray([int(np.argmin(distance))])
        sample_power = dataset.rcs_power[samples]
        if merged_seam_power is not None and np.any(samples == 0):


            sample_power[np.flatnonzero(samples == 0)] = merged_seam_power
        power[index] = np.nanmedian(sample_power, axis=0)
    output_centers = np.deg2rad(centers) if angle_unit == "rad" else centers
    extra = {}
    for key in ("time_convention", "polarization_basis"):
        value = dataset._declared_scalar_metadata(key)
        if value:
            extra[key] = value
    extra["amplitude_convention"] = (
        "sliding median of linear power; coherent phase is undefined"
    )
    history = dataset.history
    if discarded_seam_cells:
        history_entry = (
            f"Medianize periodic seam: kept the opening sample at "
            f"{azimuth[0]:.12g} deg where the closing repeat at "
            f"{azimuth[-1]:.12g} deg disagreed; discarded "
            f"{discarded_seam_cells} conflicting cell(s)"
        )
        prior_history = str(history or "").strip()
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )
    result = RcsGrid(
        output_centers,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs=None,
        rcs_power=power,
        rcs_phase=np.full_like(power, np.nan),
        rcs_domain=dataset.rcs_domain,
        source_path=dataset.source_path,
        history=history,
        units=dict(dataset.units or {}),
        extra=extra,
    )
    if not return_report:
        return result
    report = {
        "schema": "grim.medianize-report.v1",
        "periodic": periodic,
        "seam_merged": merged_seam_power is not None,
        "seam_conflict": policy,
        "discarded_seam_cell_count": discarded_seam_cells,
    }
    return result, report


def wedge_to_conic(
    dataset: RcsGrid,
    *,
    mode: str = "regrid",
    assume_missing_cross_pol_zero: bool = False,
) -> RcsGrid:
    """Replay the GUI's physical wedge-to-normal-conic conversion."""

    mode_key = str(mode).strip().lower()
    if mode_key == "relabel":
        raise ValueError(
            "Wedge relabel is not physically representable as a rectangular "
            "RcsGrid; use mode='regrid'."
        )
    if mode_key != "regrid":
        raise ValueError("mode must be 'regrid'")
    return dataset.convert_wedge_to_conic(
        attest_wedge_axes=False,
        assume_missing_cross_pol_zero=assume_missing_cross_pol_zero,
    )


# --------------------------------------------------------------------------
# Down-range time gating
# --------------------------------------------------------------------------

_GATE_PAD_FACTOR = 4
_GATE_WORK_BYTES = 32 * 1024**2


def _declared_time_sign(dataset: RcsGrid) -> str | None:
    """Return "+jwt", "-jwt", or None when no time convention is declared."""

    from GRIM_Backend.datasets.metadata import canonical_time_convention

    signs = set()
    for container in (dataset.units or {}, dataset.extra or {}):
        for key in ("time_convention", "phase_reference", "amplitude_convention"):
            raw = container.get(key)
            if raw is None:
                continue
            array = np.asarray(raw)
            if array.size != 1:
                continue
            sign = canonical_time_convention(str(array.reshape(-1)[0].item()))
            if sign in {"+jwt", "-jwt"}:
                signs.add(sign)
    if len(signs) > 1:
        raise ValueError("dataset declares contradictory time conventions")
    return next(iter(signs)) if signs else None


def _uniform_frequencies_hz(dataset: RcsGrid) -> np.ndarray:
    frequencies = np.asarray(
        dataset._frequency_value_to_hz(np.asarray(dataset.frequencies, dtype=float)),
        dtype=float,
    )
    if frequencies.size < 8:
        raise ValueError("time gating needs at least 8 frequencies")
    steps = np.diff(frequencies)
    step = float(np.mean(steps))
    if not np.all(np.isfinite(frequencies)) or step <= 0.0 or np.max(
        np.abs(steps - step)
    ) > 1.0e-6 * step:
        raise ValueError(
            "time gating needs increasing, uniformly spaced frequencies; "
            "Regrid the frequency axis first"
        )
    return frequencies


def _gate_fft_size(count: int) -> int:
    return 1 << int(np.ceil(np.log2(_GATE_PAD_FACTOR * count)))


def gate_geometry(dataset: RcsGrid) -> dict[str, float]:
    """Down-range resolution and unambiguous range (metres) of a sweep."""

    from GRIM_Backend.datasets.constants import C0

    frequencies = _uniform_frequencies_hz(dataset)
    step = float(np.mean(np.diff(frequencies)))
    return {
        "resolution_m": C0 / (2.0 * step * frequencies.size),
        "unambiguous_m": C0 / (2.0 * step),
    }


def _gate_window(ranges, start_m, stop_m, taper, mode):
    """1 inside [start, stop] with raised-cosine edges; inverted for "remove"."""

    width = stop_m - start_m
    edge = 0.5 * float(taper) * width
    gate = ((ranges >= start_m) & (ranges <= stop_m)).astype(float)
    if edge > 0.0:
        rising = (ranges >= start_m) & (ranges < start_m + edge)
        falling = (ranges > stop_m - edge) & (ranges <= stop_m)
        gate[rising] = 0.5 - 0.5 * np.cos(np.pi * (ranges[rising] - start_m) / edge)
        gate[falling] = 0.5 - 0.5 * np.cos(np.pi * (stop_m - ranges[falling]) / edge)
    return 1.0 - gate if mode == "remove" else gate


def _gate_sweeps(sweeps, frequencies_hz, *, start_m, stop_m, taper, mode, compensate):
    """Gate complex frequency sweeps along the last axis (GRIM exp(+jwt) law)."""

    from GRIM_Backend.datasets.constants import C0

    count = frequencies_hz.size
    step = float(np.mean(np.diff(frequencies_hz)))
    size = _gate_fft_size(count)
    ranges = np.fft.fftfreq(size, d=step) * (C0 / 2.0)
    gate = _gate_window(ranges, start_m, stop_m, taper, mode)
    gated = np.fft.fft(np.fft.ifft(sweeps, n=size, axis=-1) * gate, axis=-1)[..., :count]
    if compensate and mode == "keep":
        center = 0.5 * (start_m + stop_m)
        point = np.exp(-4j * np.pi * frequencies_hz * center / C0)
        response = np.fft.fft(np.fft.ifft(point, n=size) * gate)[:count] / point
        response = np.where(np.abs(response) > 1.0e-3, response, 1.0)
        gated = gated / response
    return gated


def _validate_gate(dataset, start_m, stop_m, taper, mode):
    start_m, stop_m, taper = float(start_m), float(stop_m), float(taper)
    if mode not in {"keep", "remove"}:
        raise ValueError("mode must be 'keep' or 'remove'")
    if not all(math.isfinite(value) for value in (start_m, stop_m, taper)):
        raise ValueError("gate start, stop, and taper must be finite")
    if not 0.0 <= taper <= 1.0:
        raise ValueError("taper must be between 0 and 1")
    if stop_m <= start_m:
        raise ValueError("gate stop must be beyond gate start")
    geometry = gate_geometry(dataset)
    half = 0.5 * geometry["unambiguous_m"]
    if start_m < -half or stop_m > half:
        raise ValueError(
            f"the gate must lie within -{half:.4g} to +{half:.4g} m, the "
            "unambiguous down range of this frequency step"
        )
    if stop_m - start_m < geometry["resolution_m"]:
        raise ValueError(
            f"the gate is narrower than the {geometry['resolution_m']:.4g} m "
            "down-range resolution"
        )
    return start_m, stop_m, taper


def time_gate(
    dataset: RcsGrid,
    *,
    start_m: float,
    stop_m: float,
    taper: float = 0.2,
    mode: str = "keep",
    compensate: bool = True,
) -> RcsGrid:
    """Gate the down-range response of every frequency sweep.

    Each azimuth/elevation/polarization sweep is transformed to down range
    (inverse FFT over frequency, zero-padded), multiplied by a gate over
    ``[start_m, stop_m]`` metres from the phase reference (positive away from
    the radar), and transformed back to the original frequencies. ``taper``
    is the fraction of the gate width given to raised-cosine edges.
    ``mode="remove"`` keeps everything except the gate. ``compensate``
    (keep mode) divides out the gate's band-edge droop for a point at the
    gate center. Data declaring exp(-jwt) are conjugated around the gate;
    undeclared data follow GRIM's exp(+jwt) law. Sweeps with any missing
    sample are left missing.
    """

    mode = str(mode).strip().lower()
    start_m, stop_m, taper = _validate_gate(dataset, start_m, stop_m, taper, mode)
    frequencies = _uniform_frequencies_hz(dataset)
    conjugate = _declared_time_sign(dataset) == "-jwt"
    shape = tuple(int(value) for value in dataset.rcs_power.shape)
    power = np.full(shape, np.nan, dtype=np.float64)
    phase = np.full(shape, np.nan, dtype=np.float64)
    sweeps_per_azimuth = max(1, shape[1] * shape[3])
    bytes_per_azimuth = sweeps_per_azimuth * _gate_fft_size(shape[2]) * 16 * 3
    rows = max(1, _GATE_WORK_BYTES // bytes_per_azimuth)
    usable = 0
    for first in range(0, shape[0], rows):
        block = (slice(first, min(shape[0], first + rows)),)
        field = np.asarray(dataset.rcs_slice(block + (slice(None),) * 3), dtype=np.complex128)
        sweeps = np.moveaxis(field, 2, -1)
        if conjugate:
            sweeps = np.conj(sweeps)
        complete = np.all(np.isfinite(sweeps), axis=-1)
        gated = _gate_sweeps(
            np.where(complete[..., None], sweeps, 0.0), frequencies,
            start_m=start_m, stop_m=stop_m, taper=taper, mode=mode,
            compensate=compensate,
        )
        if conjugate:
            gated = np.conj(gated)
        gated = np.moveaxis(np.where(complete[..., None], gated, np.nan), -1, 2)
        power[block] = np.abs(gated) ** 2
        phase[block] = np.angle(gated)
        usable += int(np.count_nonzero(complete))
    if usable == 0:
        raise ValueError(
            "time gating needs complex (phase) data with every frequency present "
            "in at least one sweep"
        )
    extra = _derived_response_extra(dataset)
    extra["time_gate_json"] = json.dumps(
        {
            "schema": "grim.time-gate.v1",
            "start_m": start_m,
            "stop_m": stop_m,
            "taper": taper,
            "mode": mode,
            "compensate": bool(compensate and mode == "keep"),
            "time_convention": "exp(-jwt)" if conjugate else "exp(+jwt)",
            "complete_sweeps": usable,
            "total_sweeps": int(shape[0] * shape[1] * shape[3]),
        },
        sort_keys=True,
    )
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs_power=power,
        rcs_phase=phase,
        rcs_domain="complex_amplitude",
        source_path=dataset.source_path,
        history=dataset.history,
        units=dict(dataset.units or {}),
        extra=extra,
    )


def down_range_profile(
    dataset: RcsGrid,
    *,
    elevation_index: int,
    polarization_index: int,
    gate: dict | None = None,
    max_sweeps: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Median Hann-windowed down-range power over azimuth, for gate previews.

    Returns ``(ranges_m, linear_power)`` sorted by range. ``gate`` (the
    ``time_gate`` keyword arguments) previews the gated result on the same
    scale as the ungated profile.
    """

    from GRIM_Backend.datasets.constants import C0

    frequencies = _uniform_frequencies_hz(dataset)
    count = frequencies.size
    azimuths = np.unique(np.linspace(
        0, dataset.rcs_power.shape[0] - 1, min(max_sweeps, dataset.rcs_power.shape[0])
    ).round().astype(int))
    field = np.asarray(
        dataset.rcs_slice((azimuths, int(elevation_index), slice(None), int(polarization_index))),
        dtype=np.complex128,
    )
    field = field[np.all(np.isfinite(field), axis=-1)]
    if field.size == 0:
        raise ValueError("no complete complex sweep for this elevation and polarization")
    if _declared_time_sign(dataset) == "-jwt":
        field = np.conj(field)
    if gate is not None:
        field = _gate_sweeps(field, frequencies, **gate)
    size = _gate_fft_size(count)
    window = np.hanning(count)
    profile = np.abs(np.fft.ifft(field * window, n=size, axis=-1)) ** 2
    profile = np.fft.fftshift(np.median(profile, axis=0))
    step = float(np.mean(np.diff(frequencies)))
    ranges = np.fft.fftshift(np.fft.fftfreq(size, d=step)) * (C0 / 2.0)
    return ranges, profile


# --------------------------------------------------------------------------
# Phase-center translation
# --------------------------------------------------------------------------


def _angle_axis_radians(dataset: RcsGrid, axis: str) -> np.ndarray:
    values = np.asarray(dataset.get_axis(axis), dtype=float)
    unit = str((dataset.units or {}).get(axis, "deg")).strip().lower()
    if unit in {"rad", "radian", "radians"}:
        return values
    if unit in {"deg", "degree", "degrees"}:
        return np.deg2rad(values)
    raise ValueError(f"unsupported {axis} unit {unit!r}; use deg or rad")


def translate_phase_center(
    dataset: RcsGrid,
    *,
    x_m: float,
    y_m: float,
    z_m: float,
) -> RcsGrid:
    """Move the phase reference to body point ``(x, y, z)`` metres.

    GRIM angles are coming-from radar directions
    ``u = (cos el cos az, cos el sin az, sin el)`` in body axes, and a point
    scatterer at ``p`` has phase ``exp(+j 2k u.p)`` under exp(+jwt). Referencing
    the field to ``t`` therefore multiplies every sample by ``exp(-j 2k u.t)``;
    data declaring exp(-jwt) use the conjugate ramp. Levels are unchanged, and
    samples without phase stay without phase.
    """

    from GRIM_Backend.datasets.constants import C0

    offset = np.asarray([x_m, y_m, z_m], dtype=float)
    if not np.all(np.isfinite(offset)):
        raise ValueError("the phase-center offset must be finite")
    if not np.any(offset):
        raise ValueError("the phase-center offset is zero")
    power = dataset.rcs_power
    phase_in = _authoritative_response_phase(dataset)
    if not np.any(np.isfinite(power) & np.isfinite(phase_in)):
        raise ValueError("moving the phase center needs complex (phase) data")
    azimuth = _angle_axis_radians(dataset, "azimuth")
    elevation = _angle_axis_radians(dataset, "elevation")
    declarations = set()
    for key in ("elevation_coordinate_convention", "sentri_elevation_convention"):
        metadata = dataset.inspect_scalar_metadata(key)
        if metadata.malformed_sources:
            raise ValueError(f"{key} must be scalar")
        declarations.update(value.strip().lower() for value in metadata.declarations)
    if len(declarations) > 1:
        raise ValueError("dataset declares contradictory elevation conventions")
    convention = next(iter(declarations), "")
    if convention == "sentri_theta_top_zero":
        elevation = np.pi / 2.0 - elevation
    elif convention not in {"", "grim_elevation_waterline_zero_top_positive"}:
        raise ValueError(f"unsupported elevation convention {convention!r}")
    wavenumber = 2.0 * np.pi * np.asarray(
        dataset._frequency_value_to_hz(np.asarray(dataset.frequencies, dtype=float)),
        dtype=float,
    ) / C0
    projection = (
        np.cos(elevation)[None, :] * np.cos(azimuth)[:, None] * offset[0]
        + np.cos(elevation)[None, :] * np.sin(azimuth)[:, None] * offset[1]
        + np.sin(elevation)[None, :] * offset[2]
    )
    sign = 1.0 if _declared_time_sign(dataset) == "-jwt" else -1.0
    phase = np.empty(power.shape, dtype=np.float64)
    rows = max(1, _GATE_WORK_BYTES // max(1, 8 * int(np.prod(power.shape[1:]))))
    for first in range(0, power.shape[0], rows):
        block = slice(first, min(power.shape[0], first + rows))
        ramp = sign * 2.0 * projection[block, :, None] * wavenumber[None, None, :]
        shifted = np.asarray(phase_in[block], dtype=np.float64) + ramp[..., None] + np.pi
        phase[block] = np.remainder(shifted, 2.0 * np.pi) - np.pi

    units = dict(dataset.units or {})
    extra = _derived_response_extra(dataset)
    previous = np.zeros(3)
    if "phase_center_translation_json" in extra:
        try:
            previous = np.asarray(
                json.loads(str(extra["phase_center_translation_json"]))["total_offset_m"],
                dtype=float,
            )
        except (ValueError, KeyError, TypeError):
            previous = np.zeros(3)
    total = previous + offset
    extra["phase_center_translation_json"] = json.dumps(
        {
            "schema": "grim.phase-center-translation.v1",
            "last_offset_m": offset.tolist(),
            "total_offset_m": total.tolist(),
            "direction_convention": "coming-from radar u=(cos el cos az, cos el sin az, sin el)",
            "time_convention": "exp(-jwt)" if sign > 0 else "exp(+jwt)",
            "source_elevation_convention": convention or "GRIM signed elevation (assumed)",
        },
        sort_keys=True,
    )
    note = f"moved by ({offset[0]:g}, {offset[1]:g}, {offset[2]:g}) m in body axes"
    for container in (units, extra):
        declared = container.get("phase_reference")
        if declared is None or np.asarray(declared).size != 1:
            continue
        text = str(np.asarray(declared).reshape(-1)[0].item()).strip()
        if text:
            container["phase_reference"] = f"{text}; {note}"
    # The same translation applies to every contributing field. Carry its
    # reference declarations forward so later coherent checks do not compare
    # the new reference against unshifted lineage from an earlier combination.
    if "coherent_source_conventions_json" in extra:
        try:
            record = json.loads(str(np.asarray(
                extra["coherent_source_conventions_json"]
            ).reshape(()).item()))
            references = record.get("declared_values", {}).get("phase_reference")
            if isinstance(references, list):
                record["declared_values"]["phase_reference"] = [
                    f"{reference}; {note}" for reference in references
                ]
                extra["coherent_source_conventions_json"] = json.dumps(record, sort_keys=True)
        except (ValueError, TypeError, AttributeError):
            pass
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs_power=power,
        rcs_phase=phase,
        rcs_domain="complex_amplitude",
        source_path=dataset.source_path,
        history=dataset.history,
        units=units,
        extra=extra,
    )
