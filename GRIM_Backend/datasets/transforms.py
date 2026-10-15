"""Dataset copying, regridding, decimation, offsets, and complex division."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from GRIM_Backend.datasets.result_storage import allocate_result, copy_result

from GRIM_Backend.datasets.operations import dataset_operation, operation_effective_parameters

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.constants import C0, _ADOPT_CLEAN_ARRAYS_TOKEN
from GRIM_Backend.datasets.memory import _bounded_grid_selections


_MAX_EXPLICIT_AXIS_POINTS = 1_000_000


def _derived_response_extra(dataset: RcsGrid) -> dict:
    """Keep response-independent ancillary models, dropping stale grid claims."""

    return dataset._response_independent_extra()


def _copy_grid(
    dataset: RcsGrid,
    *,
    rcs=None,
    rcs_power=None,
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
            else dataset.phase if rcs_phase is None else rcs_phase
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
        return dataset.phase
    real, imag = raw_pair
    phase = allocate_result(dataset.power.shape, np.float64)
    for selection in _bounded_grid_selections(dataset.shape, 262_144):
        np.arctan2(
            np.asarray(imag[selection], dtype=np.float64),
            np.asarray(real[selection], dtype=np.float64),
            out=phase[selection],
        )
    return phase


def _phase_shifted_response(dataset: RcsGrid, phase_degrees: float) -> np.ndarray:
    """Return authoritative phase shifted into the canonical signed interval."""

    phase = copy_result(_authoritative_response_phase(dataset))
    delta = np.deg2rad(float(phase_degrees))
    with np.errstate(invalid="ignore"):
        np.add(phase, delta + np.pi, out=phase)
        np.remainder(phase, 2.0 * np.pi, out=phase)
        np.subtract(phase, np.pi, out=phase)
    return phase


@dataset_operation('duplicate_dataset')
def duplicate_dataset(dataset: RcsGrid) -> RcsGrid:
    """Return a detached copy with the same numerical samples and metadata."""

    samples = dataset._sample_kwargs()
    # A raw solver field travels in ``samples``; its real/imaginary metadata
    # views are re-derived by the constructor rather than deep-copied twice.
    raw_in_samples = samples.get("rcs_storage_domain") == "raw_field"
    extra = {
        key: (copy_result(value) if isinstance(value, np.ndarray)
              and value.dtype.kind in "biufc" else copy.deepcopy(value))
        for key, value in (dataset.extra or {}).items()
        if not (raw_in_samples and key in ("rcs_amp_real", "rcs_amp_imag"))
    }
    return RcsGrid(
        dataset.azimuths.copy(),
        dataset.elevations.copy(),
        dataset.frequencies.copy(),
        dataset.polarizations.copy(),
        **samples,
        rcs_domain=dataset.rcs_domain,
        source_path=dataset.source_path,
        history=dataset.history,
        units=copy.deepcopy(dataset.units or {}),
        extra=extra,
        _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
    )


@dataset_operation('flip_phase')
def flip_phase(dataset: RcsGrid) -> RcsGrid:
    """Correct the stored phase sign: ``I + jQ`` becomes ``I - jQ``.

    Magnitude, coordinates, and the declared interpretation of the samples
    stay unchanged. This is a phase-sign correction, not an automatic change
    of time convention, and not a 180-degree phase offset. Unknown phases
    remain unknown. A second flip restores the original response.
    """

    dataset._ensure_legacy_source_current()
    extra = _derived_response_extra(dataset)
    dataset._invalidate_assembly_sampling_hash(extra, 'flip-phase')
    # Conjugation changes no amplitude normalization. Keep that declaration
    # while the shared derived-response policy removes stale solver reports.
    if 'amplitude_convention' in (dataset.extra or {}):
        extra['amplitude_convention'] = copy.deepcopy(dataset.extra['amplitude_convention'])
    extra.pop('raw_complex_amplitude_preserved', None)
    units = copy.deepcopy(dataset.units or {})
    raw_pair = (dataset._complete_authoritative_raw_arrays()
                if dataset._raw_normalization_defined() else None)
    if dataset._complex is not None and dataset._storage_domain == 'raw_field':
        raw_pair = (dataset._complex.real, dataset._complex.imag)
    blocks = lambda: _bounded_grid_selections(dataset.shape, 262_144)

    if dataset._complex is not None:
        # Retain raw solver quadratures without normalizing/reconstructing
        # them, including a validated authoritative pair attached later.
        if raw_pair is not None:
            real, imag = raw_pair
            dtype = np.result_type(real.dtype, imag.dtype, np.complex64)
            field = allocate_result(dataset.shape, dtype)
            for block in blocks():
                field[block].real = real[block]
                np.negative(imag[block], out=field[block].imag, dtype=field.real.dtype)
            samples = {'rcs': field, 'rcs_storage_domain': 'raw_field'}
        else:
            field = allocate_result(dataset.shape, dataset._complex.dtype)
            for block in blocks():
                np.conjugate(dataset._complex[block], out=field[block])
            samples = {'rcs': field}
    else:
        # Polar storage can encode unknown phase, zero-amplitude phase, and
        # unwrapped turns that a complex field cannot. Preserve that evidence.
        samples = {'rcs_power': copy_result(dataset._power)}
        if dataset._phase is not None:
            phase = allocate_result(dataset.shape, dataset._phase.dtype)
            wrap = str(units.get('phase_wrap', '')).strip()
            for block in blocks():
                part = phase[block]
                np.negative(dataset._phase[block], out=part)
                if wrap == '0_360':
                    finite = np.isfinite(part)
                    part[finite] = np.remainder(part[finite], 2.0 * np.pi)
                    part[part >= 2.0 * np.pi] = 0.0
                elif wrap == '-180_180':
                    outside = np.isfinite(part) & ((part < -np.pi) | (part >= np.pi))
                    part[outside] = np.remainder(part[outside] + np.pi, 2.0 * np.pi) - np.pi
            samples['rcs_phase'] = phase
        if raw_pair is not None:
            # An exceptional polar grid may retain raw authority beside
            # unknown/zero phase. Conjugate both representations consistently
            # without filling in any phase that the source did not store.
            real, imag = raw_pair
            extra['rcs_amp_real'] = copy_result(real)
            imag_dtype = np.result_type(imag.dtype, np.float32)
            extra['rcs_amp_imag'] = allocate_result(dataset.shape, imag_dtype)
            for block in blocks():
                np.negative(imag[block], out=extra['rcs_amp_imag'][block], dtype=imag_dtype)
            extra['raw_complex_amplitude_preserved'] = True
    extra['phase_sign_correction_json'] = json.dumps({
        'schema': 'grim.phase-sign-correction.v1',
        'operation': 'complex_conjugate',
        'interpretation': 'Correct stored phase sign; declared time and phase conventions retained.',
        'quadratures': 'I unchanged; Q negated',
        'unknown_phase': 'preserved',
    }, sort_keys=True)
    return RcsGrid(
        dataset.azimuths, dataset.elevations, dataset.frequencies,
        dataset.polarizations, **samples,
        rcs_domain=dataset.rcs_domain, source_path=dataset.source_path,
        history=dataset.history, units=units, extra=extra,
        _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
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


@dataset_operation('crop_dataset')
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


@dataset_operation('regrid_axis')
def regrid_axis(
    dataset: RcsGrid,
    axis: str,
    *,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
    values=None,
    max_gap: float | None = None,
    support: str = "interpolate",
) -> RcsGrid:
    """Interpolate one numeric axis onto the requested coordinates.

    Supply either ``values`` or all of ``start``, ``stop``, and ``step`` in the
    dataset axis unit. ``RcsGrid.interpolate_axis`` applies complex/linear
    interpolation and rejects extrapolation. Optional ``max_gap`` uses the
    same native axis unit. ``support='measured-only'`` masks new coordinates;
    the default keeps the existing permissive interpolation behavior.
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
    return dataset.interpolate_axis(axis_name, target, max_gap=max_gap, support=support)


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


@dataset_operation('shift_dataset')
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
        delta = float(phase_degrees)
        if np.isfinite(delta) and delta % 360.0 == 0.0:
            return result
        if result.storage_kind == "complex":
            rotation = np.exp(1j * np.deg2rad(delta))
            field = allocate_result(result.shape, (
                np.complex128 if result.real_dtype.itemsize > 4 else np.complex64
            ))
            read_field, _dtype = result._bounded_complex_slice_reader()
            for selection in _bounded_grid_selections(result.shape, (16 * 1024**2) // 48):
                field[selection] = read_field(selection) * rotation
            return _copy_grid(
                result, rcs=field,
                rcs_domain="complex_amplitude",
            )
        raw_pair = result._complete_authoritative_raw_arrays()
        if raw_pair is not None and result.linear_quantity() in {"sigma_2d", "sigma_3d"}:
            # Retain authoritative precision without materializing a complex
            # whole-grid response from the rounded display power.
            real, imag = raw_pair
            power = allocate_result(result.shape, np.float64)
            if result.linear_quantity() == "sigma_2d":
                k = 2.0 * np.pi * result._frequency_value_to_hz(result.frequencies) / C0
            for selection in _bounded_grid_selections(result.shape, 262_144):
                block = np.square(np.asarray(real[selection], dtype=np.float64))
                block += np.square(np.asarray(imag[selection], dtype=np.float64))
                if result.linear_quantity() == "sigma_2d":
                    block /= (4.0 * k[selection[2]])[None, None, :, None]
                else:
                    block *= 4.0 * np.pi
                power[selection] = block
            return _copy_grid(
                result, rcs_power=power,
                rcs_phase=_phase_shifted_response(result, delta),
                rcs_domain="complex_amplitude",
            )
        result = _copy_grid(
            result,
            rcs_power=result.power,
            rcs_phase=_phase_shifted_response(result, float(phase_degrees)),
            rcs_domain="complex_amplitude",
        )
    return result


@dataset_operation('wrap_phase')
def wrap_phase(dataset: RcsGrid, mode: str = "-180_180") -> RcsGrid:
    """Wrap stored phase using the user-facing degree interval names."""

    mode_name = str(mode).strip()
    if mode_name not in {"-180_180", "0_360"}:
        raise ValueError("mode must be '-180_180' or '0_360'")
    return dataset.wrap_phase(mode=mode_name)


@dataset_operation('offset_db')
def offset_db(dataset: RcsGrid, value_db: float) -> RcsGrid:
    """Shift all displayed RCS values by a constant number of decibels."""

    scale = 10.0 ** (float(value_db) / 10.0)
    if dataset.storage_kind == "complex":
        field = allocate_result(dataset.shape, (
            np.complex128 if dataset.real_dtype.itemsize > 4 else np.complex64
        ))
        read_field, _dtype = dataset._bounded_complex_slice_reader()
        for selection in _bounded_grid_selections(dataset.shape, (16 * 1024**2) // 48):
            field[selection] = read_field(selection) * np.sqrt(scale)
        return _copy_grid(dataset, rcs=field, rcs_domain=dataset.rcs_domain)
    raw_pair = dataset._complete_authoritative_raw_arrays()
    quantity = dataset.linear_quantity()
    use_raw = (
        raw_pair is not None and quantity in {"sigma_2d", "sigma_3d"}
        and dataset.power.size > 0
    )
    if use_raw and quantity == "sigma_2d":
        frequency_hz = dataset._frequency_value_to_hz(dataset.frequencies)
        use_raw = bool(
            np.all(np.isfinite(frequency_hz)) and np.all(frequency_hz > 0.0)
        )
    if use_raw:
        # The display cache may be float32 even when solver quadratures are
        # float64. Normalize those authoritative fields before scaling, keeping
        # the derived response in float64 for later coherent residuals.
        shape = dataset.power.shape
        power = allocate_result(shape, np.float64)
        phase = allocate_result(shape, np.float64)
        for selection in _bounded_grid_selections(shape, (16 * 1024**2) // 48):
            field = dataset._authoritative_raw_amplitude_from_pair(
                raw_pair, selection
            )
            power_block = power[selection]
            np.multiply(field.real, field.real, out=power_block)
            power_block += field.imag * field.imag
            power_block *= scale
            np.arctan2(field.imag, field.real, out=phase[selection])
        return _copy_grid(
            dataset, rcs_power=power, rcs_phase=phase,
            rcs_domain=dataset.rcs_domain,
        )
    power = allocate_result(dataset.shape, np.result_type(dataset.power.dtype, scale))
    for selection in _bounded_grid_selections(dataset.shape, 262_144):
        power[selection] = dataset.power[selection] * scale
    return _copy_grid(
        dataset,
        rcs_power=power,
        rcs_phase=_authoritative_response_phase(dataset),
        rcs_domain=dataset.rcs_domain,
    )


@dataset_operation('coherent_divide')
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
    shape = tuple(int(value) for value in numerator.power.shape)
    raw_response = (
        numerator._complete_authoritative_raw_arrays() is not None
        or denominator._complete_authoritative_raw_arrays() is not None
    )
    real_dtype = (
        np.dtype(np.float64)
        if raw_response
        else np.result_type(
            numerator.power.dtype,
            numerator.phase.dtype,
            denominator.power.dtype,
            denominator.phase.dtype,
        )
    )
    complex_dtype = (
        np.dtype(np.complex128)
        if real_dtype.itemsize > np.dtype(np.float32).itemsize
        else np.dtype(np.complex64)
    )
    result_field = allocate_result(shape, complex_dtype, fill=np.nan + 1j * np.nan)


    working_bytes_per_cell = (
        3 * complex_dtype.itemsize + real_dtype.itemsize + 3
    )
    usable_count = 0
    for selection in _bounded_grid_selections(
        shape, (16 * 1024**2) // working_bytes_per_cell
    ):
        left = np.asarray(numerator.rcs_slice(selection), dtype=complex_dtype)
        right = np.asarray(denominator.rcs_slice(selection), dtype=complex_dtype)
        quotient = np.full(left.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        valid = np.isfinite(left) & np.isfinite(right) & (right != 0)
        np.divide(left, right, out=quotient, where=valid)
        power_block = np.empty(quotient.shape, dtype=real_dtype)
        np.multiply(quotient.real, quotient.real, out=power_block)
        power_block += quotient.imag * quotient.imag
        finite_result = np.isfinite(power_block) & np.isfinite(quotient)
        quotient[~finite_result] = np.nan + 1j * np.nan
        result_field[selection] = quotient
        usable_count += int(np.count_nonzero(finite_result))
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
            "total_sample_count": int(result_field.size),
            "usable_sample_count": usable_count,
            "masked_sample_count": int(result_field.size - usable_count),
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
        rcs=result_field,
        rcs_domain="complex_amplitude",
        source_path=numerator.source_path,
        history=history,
        units=units,
        extra=extra,
    )


@dataset_operation('convert_extrusion')
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
    units = dict(dataset.units or {})
    units["rcs_log_unit"] = "dBke" if destination == "dbke" else "dBsm"
    units["rcs_linear_quantity"] = quantity
    if dataset.storage_kind == "complex":
        rcs = allocate_result(dataset.shape, np.complex128)
        read_field, _dtype = dataset._bounded_complex_slice_reader()
        for selection in _bounded_grid_selections(dataset.shape, 65_536):
            block = rcs[selection]
            np.multiply(read_field(selection), np.sqrt(
                np.maximum(scale[selection[2]], 0.0))[None, None, :, None], out=block)
            block[~np.isfinite(block.real) | ~np.isfinite(block.imag)] = np.nan + 1j * np.nan
        return RcsGrid(
            dataset.azimuths,
            dataset.elevations,
            dataset.frequencies,
            dataset.polarizations,
            rcs,
            rcs_power=None,
            rcs_domain=dataset.rcs_domain,
            source_path=dataset.source_path,
            history=dataset.history,
            units=units,
            extra=_derived_response_extra(dataset),
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )
    power = allocate_result(dataset.shape, np.result_type(dataset.power.dtype, scale.dtype))
    for selection in _bounded_grid_selections(dataset.shape, 262_144):
        power[selection] = dataset.power[selection] * scale[selection[2]][None, None, :, None]
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs=None,
        rcs_power=power,
        rcs_phase=dataset.phase,
        rcs_domain=dataset.rcs_domain,
        source_path=dataset.source_path,
        history=dataset.history,
        units=units,
        extra=_derived_response_extra(dataset),
    )


def decimation_axis_plan(source_axis, factor: int, partial: str = "keep"):
    """Resolve block centers and sampling facts for processing and UI preview."""
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("factor must be an integer of at least 2")
    factor = int(factor)
    if factor < 2:
        raise ValueError("factor must be an integer of at least 2")
    if partial not in {"keep", "drop"}:
        raise ValueError("partial must be 'keep' or 'drop'")
    source_axis = np.asarray(source_axis, dtype=float)
    if source_axis.size < 2 or np.any(~np.isfinite(source_axis)):
        raise ValueError("decimation requires at least two finite axis coordinates")
    steps = np.diff(source_axis)
    if np.any(steps <= 0.0):
        raise ValueError("decimation requires a strictly increasing source axis")
    nominal_step = float(np.median(steps))
    atol = max(abs(nominal_step) * 1e-9,
               np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(source_axis)))) * 32)
    if not np.allclose(steps, nominal_step, rtol=1e-6, atol=atol):
        raise ValueError("decimation requires uniformly spaced source coordinates; regrid to a uniform fine grid first")
    full_bins, remainder = divmod(int(source_axis.size), factor)
    centers = source_axis[:full_bins * factor].reshape(full_bins, factor).mean(axis=1)
    if remainder and partial == "keep":
        centers = np.append(centers, np.mean(source_axis[full_bins * factor:]))
    if not centers.size:
        raise ValueError("dropping the partial bin leaves no output; keep it or choose a smaller factor")
    output_steps = np.diff(centers)
    uniform = output_steps.size < 2 or bool(np.allclose(
        output_steps, output_steps[0], rtol=1e-6, atol=atol))
    return centers, {
        "partial_bin_policy": partial,
        "source_coordinate_count": int(source_axis.size),
        "output_coordinate_count": int(centers.size),
        "full_bin_count": full_bins,
        "partial_bin_sample_count": remainder,
        "partial_final_bin_retained": bool(remainder and partial == "keep"),
        "discarded_source_coordinate_count": remainder if partial == "drop" else 0,
        "output_uniform": uniform,
        "output_step_min": float(np.min(output_steps)) if output_steps.size else None,
        "output_step_max": float(np.max(output_steps)) if output_steps.size else None,
    }


@dataset_operation('decimate_axis')
def decimate_axis(
    dataset: RcsGrid,
    *,
    axis: str,
    factor: int,
    mode: str = "power",
    metadata_attested: bool = False,
    partial: str = "keep",
) -> RcsGrid:
    """Boxcar-prefilter and decimate one numeric axis by an integer factor.

    ``mode='power'`` averages finite linear power and marks phase unknown.
    ``mode='coherent'`` averages the finite complex field and retains the
    resulting phase. ``partial='keep'`` retains the final partial bin at its
    actual center (which may break uniform output spacing). ``partial='drop'``
    explicitly discards those trailing samples and preserves uniform spacing.
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
    output_axis, sampling_report = decimation_axis_plan(source_axis, factor, partial)
    source_count = int(source_axis.size)
    full_bin_count, remainder = divmod(source_count, factor)
    output_shape = list(dataset.power.shape)
    output_shape[axis_index] = int(output_axis.size)
    output_shape = tuple(output_shape)
    coherent = mode_key == "coherent"
    output = allocate_result(
        output_shape, np.complex128 if coherent else np.float64,
        fill=np.nan + 1j * np.nan if coherent else np.nan,
    )
    output_by_bin = np.moveaxis(output, axis_index, 0)
    other_cell_count = max(
        1,
        int(np.prod(dataset.power.shape, dtype=np.int64)) // source_count,
    )
    transient_bytes_per_cell = 64 if mode_key == "coherent" else 24
    bins_per_block = max(
        1,
        (32 * 1024**2)
        // max(1, factor * other_cell_count * transient_bytes_per_cell),
    )

    def aggregate_bins(start: int, bin_count: int, samples_per_bin: int, other_selection):
        stop = start + bin_count * samples_per_bin
        source_selection = list(other_selection)
        source_selection.insert(axis_index, slice(None))
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
            return mean_field
        else:
            power_block = np.asarray(dataset.power[source_selection], dtype=np.float64)
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
            return power

    for output_start in range(0, full_bin_count, bins_per_block):
        output_stop = min(full_bin_count, output_start + bins_per_block)
        bin_count = output_stop - output_start
        other_shape = output_by_bin.shape[1:]
        block_cells = max(1, (32 * 1024**2) // (factor * bin_count * transient_bytes_per_cell))
        for other_selection in _bounded_grid_selections(other_shape, block_cells):
            output_by_bin[(slice(output_start, output_stop),) + other_selection] = aggregate_bins(
                output_start * factor, bin_count, factor, other_selection,
            )
    if remainder and partial == "keep":
        block_cells = max(1, (32 * 1024**2) // (remainder * transient_bytes_per_cell))
        for other_selection in _bounded_grid_selections(output_by_bin.shape[1:], block_cells):
            output_by_bin[(slice(full_bin_count, None),) + other_selection] = aggregate_bins(
                full_bin_count * factor, 1, remainder, other_selection,
            )

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
            **sampling_report,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    history_entry = (
        f"Decimate {axis_key} by {factor} with finite boxcar {mode_key} mean"
    )
    if remainder:
        history_entry += (
            f"; retained final partial bin ({remainder} source coordinates)"
            if partial == "keep" else f"; explicitly discarded {remainder} trailing source coordinates"
        )
    if not sampling_report["output_uniform"]:
        history_entry += "; output spacing is nonuniform"
    prior_history = (
        attestation_history
        if attestation_history is not None
        else dataset.history
    )
    history = (
        f"{prior_history}\n{history_entry}" if prior_history else history_entry
    )
    samples = {"rcs": output} if coherent else {"rcs_power": output}
    if coherent and any(np.any(output[selection] == 0.0)
                        for selection in _bounded_grid_selections(output.shape, 262_144)):
        # Exact cancellation has undefined phase. Preserve that distinction
        # instead of exposing atan2(0, 0) as an observed phase.
        output_power = allocate_result(output.shape, np.float64)
        output_phase = allocate_result(output.shape, np.float64)
        for selection in _bounded_grid_selections(output.shape, 262_144):
            block = output[selection]
            power_block = block.real * block.real + block.imag * block.imag
            phase_block = np.angle(block)
            phase_block[power_block == 0.0] = np.nan
            output_power[selection] = power_block
            output_phase[selection] = phase_block
        samples = {"rcs_power": output_power, "rcs_phase": output_phase}
    return RcsGrid(
        axes[0],
        axes[1],
        axes[2],
        axes[3],
        **samples,
        rcs_domain="power_phase",
        source_path=dataset.source_path,
        history=history,
        units=dict(dataset.units or {}),
        extra=extra,
    )


def _median_window_bounds(coordinates, coordinate_roundoff, center, half, origin):
    """Include a boundary sample despite angular conversion roundoff.

    The allowance follows the float64 spacing of each coordinate and the
    arithmetic used to form the window, rather than a fixed degree tolerance.
    Keep the original-coordinate allowance when a periodic copy crosses zero:
    subtracting 360 degrees does not undo roundoff in that source coordinate.
    """

    lower, upper = center - half, center + half
    arithmetic_roundoff = 4.0 * max(
        np.spacing(abs(center)), np.spacing(abs(half)), np.spacing(abs(origin))
    )
    lower_roundoff = arithmetic_roundoff + 2.0 * np.spacing(abs(lower))
    upper_roundoff = arithmetic_roundoff + 2.0 * np.spacing(abs(upper))
    left = int(np.searchsorted(coordinates, lower, side="left"))
    right = int(np.searchsorted(coordinates, upper, side="right"))
    while left > 0 and lower - coordinates[left - 1] <= (
        lower_roundoff + coordinate_roundoff[left - 1]
    ):
        left -= 1
    while right < coordinates.size and coordinates[right] - upper <= (
        upper_roundoff + coordinate_roundoff[right]
    ):
        right += 1
    return left, right


@dataset_operation('medianize_azimuth')
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
    Windows containing no source coordinates remain missing (NaN).
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
    operation_effective_parameters(
        "medianize_azimuth", periodic=periodic, seam_conflict=policy,
        window_degrees=window, slide_degrees=slide,
    )
    if periodic and window > 360.0 + 1.0e-9:
        raise ValueError("a periodic median window cannot exceed 360 degrees")

    half = window * 0.5
    sample_azimuth = azimuth
    sample_roundoff = 2.0 * np.spacing(np.abs(azimuth))
    merged_seam_power = None
    discarded_seam_cells = 0
    if periodic and np.isclose(span, 360.0, rtol=0.0, atol=1.0e-7):


        merged_seam_power = allocate_result(dataset.shape[1:], dataset.power.dtype)
        # The closing endpoint re-measures the opening direction; it only
        # fills cells the opening sample left missing.
        for selection in _bounded_grid_selections(dataset.shape[1:], 262_144):
            first, last = (0,) + selection, (-1,) + selection
            power_block = np.array(dataset.power[first], copy=True)
            phase_block = np.array(dataset.phase[first], copy=True)
            seam_conflict_mask = RcsGrid._merge_equivalent_sample_blocks(
                power_block, phase_block, dataset.power[last], dataset.phase[last],
                context="medianize periodic endpoint",
                keep_existing_on_conflict=policy == "first",
            )
            discarded_seam_cells += int(np.count_nonzero(seam_conflict_mask))
            merged_seam_power[selection] = power_block
        sample_azimuth = azimuth[:-1]
        sample_roundoff = sample_roundoff[:-1]
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
        augmented_roundoff = (
            np.tile(sample_roundoff, 3)
            + 2.0 * np.spacing(np.abs(augmented_azimuth))
        )
    else:
        first = float(azimuth[0]) + half
        last = float(azimuth[-1]) - half
        if last < first:
            centers = np.asarray([(float(azimuth[0]) + float(azimuth[-1])) * 0.5])
        else:
            count_float = np.floor((last - first) / slide) + 1.0
            # Dividing an endpoint error by a narrow slide can amplify it.
            # Check the possible final center in coordinate units instead of
            # adding a fixed tolerance to the dimensionless center count.
            endpoint_roundoff = 4.0 * max(
                np.spacing(abs(azimuth[0])), np.spacing(abs(azimuth[-1])),
                np.spacing(abs(first)), np.spacing(abs(last)), np.spacing(half),
            )
            next_center = first + count_float * slide
            final_center = first + (count_float - 1.0) * slide
            if (
                last > first and next_center > final_center
                and next_center - last <= endpoint_roundoff
            ):
                count_float += 1.0
            if count_float > _MAX_EXPLICIT_AXIS_POINTS:
                raise ValueError(
                    f"medianize would create {int(count_float):,} azimuths; "
                    f"the safety limit is {_MAX_EXPLICIT_AXIS_POINTS:,}"
                )
            centers = first + np.arange(int(count_float), dtype=float) * slide
    shape = (centers.size,) + dataset.power.shape[1:]
    power = allocate_result(shape, dataset.power.dtype, fill=np.nan)
    for index, center in enumerate(centers):
        if periodic:
            left, right = _median_window_bounds(
                augmented_azimuth, augmented_roundoff, center, half, azimuth[0]
            )
            samples = np.unique(augmented_indices[left:right])
        else:
            left, right = _median_window_bounds(
                azimuth, sample_roundoff, center, half, azimuth[0]
            )
            samples = np.arange(left, right, dtype=np.int64)
        if samples.size == 0:
            continue
        for selection in _bounded_grid_selections(
            dataset.shape[1:], max(1, 262_144 // samples.size)
        ):
            sample_power = dataset.power[(samples,) + selection]
            if merged_seam_power is not None and np.any(samples == 0):
                sample_power = np.array(sample_power, copy=True)
                sample_power[np.flatnonzero(samples == 0)] = merged_seam_power[selection]
            power[(index,) + selection] = np.nanmedian(sample_power, axis=0)
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
        rcs_phase=None,
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


@dataset_operation('wedge_to_conic')
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
    """Down-range resolution and unambiguous range (meters) of a sweep."""

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


@dataset_operation('time_gate')
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
    ``[start_m, stop_m]`` meters from the phase reference (positive away from
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
    shape = tuple(int(value) for value in dataset.power.shape)
    result_field = allocate_result(shape, np.complex128, fill=np.nan + 1j * np.nan)
    bytes_per_sweep = _gate_fft_size(shape[2]) * 16 * 3
    sweep_block = max(1, _GATE_WORK_BYTES // bytes_per_sweep)
    usable = 0
    for sweep_selection in _bounded_grid_selections((shape[0], shape[1], shape[3]), sweep_block):
        block = sweep_selection[:2] + (slice(None), sweep_selection[2])
        field = np.asarray(dataset.rcs_slice(block), dtype=np.complex128)
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
        result_field[block] = gated
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
        rcs=result_field,
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
    maximum_work_bytes: int = 32 * 1024**2,
    cancel_check=None,
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
        0, dataset.power.shape[0] - 1, min(max_sweeps, dataset.power.shape[0])
    ).round().astype(int))
    from .reduction_workspace import workspace_array, BLOCK_CELLS
    from GRIM_Backend.plotting.requests import check_cancel
    size = _gate_fft_size(count)
    bytes_per_sweep = 48 * count + 80 * size
    sweeps_per_block = max(1, int(maximum_work_bytes) // max(1, bytes_per_sweep))
    profiles = workspace_array((len(azimuths), size), np.float64)
    window = np.hanning(count)
    conjugate = _declared_time_sign(dataset) == "-jwt"
    read_field, _dtype = dataset._bounded_complex_slice_reader()
    complete_count = 0
    for first in range(0, len(azimuths), sweeps_per_block):
        check_cancel(cancel_check)
        field = np.asarray(read_field((azimuths[first:first + sweeps_per_block],
            int(elevation_index), slice(None), int(polarization_index))), dtype=np.complex128)
        field = field[np.all(np.isfinite(field), axis=-1)]
        if not field.size:
            continue
        if conjugate:
            field = np.conj(field)
        if gate is not None:
            field = _gate_sweeps(field, frequencies, **gate)
        power = np.abs(np.fft.ifft(field * window, n=size, axis=-1)) ** 2
        profiles[complete_count:complete_count + len(field)] = power
        complete_count += len(field)
    if not complete_count:
        raise ValueError("no complete complex sweep for this elevation and polarization")
    profile = np.empty(size, dtype=float)
    columns = max(1, BLOCK_CELLS // complete_count)
    for first in range(0, size, columns):
        check_cancel(cancel_check)
        stop = min(size, first + columns)
        profile[first:stop] = np.median(profiles[:complete_count, first:stop], axis=0)
    profile = np.fft.fftshift(profile)
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


@dataset_operation('translate_phase_center')
def translate_phase_center(
    dataset: RcsGrid,
    *,
    x_m: float,
    y_m: float,
    z_m: float,
) -> RcsGrid:
    """Move the phase reference to body point ``(x, y, z)`` meters.

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
    power = dataset.power
    complex_storage = dataset.storage_kind == "complex"
    # Complex storage carries a phase for every finite sample and is shifted
    # below straight from its field, so only power/phase storage needs the
    # authoritative phase array, and only a grid without any phase scans it.
    phase_in = None if complex_storage else _authoritative_response_phase(dataset)
    if dataset.phase_coverage() == "none" and not any(np.any(
        np.isfinite(power[selection]) & np.isfinite(phase_in[selection]))
        for selection in _bounded_grid_selections(dataset.shape, 262_144)
    ):
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
    sign = 1.0 if _declared_time_sign(dataset) == "-jwt" else -1.0
    shifted_response = allocate_result(
        power.shape, np.complex128 if complex_storage else np.float64
    )
    for block in _bounded_grid_selections(dataset.shape, _GATE_WORK_BYTES // 64):
        azimuth_block, elevation_block = azimuth[block[0]], elevation[block[1]]
        projection = (
            np.cos(elevation_block)[None, :] * np.cos(azimuth_block)[:, None] * offset[0]
            + np.cos(elevation_block)[None, :] * np.sin(azimuth_block)[:, None] * offset[1]
            + np.sin(elevation_block)[None, :] * offset[2]
        )
        ramp = sign * 2.0 * projection[:, :, None] * wavenumber[block[2]][None, None, :]
        if complex_storage:
            shifted_response[block] = dataset.rcs_slice(block) * np.exp(1j * ramp[..., None])
        else:
            shifted = np.asarray(phase_in[block], dtype=np.float64) + ramp[..., None] + np.pi
            shifted_response[block] = np.remainder(shifted, 2.0 * np.pi) - np.pi

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
        **({"rcs": shifted_response} if complex_storage else
           {"rcs_power": power, "rcs_phase": shifted_response}),
        rcs_domain="complex_amplitude",
        source_path=dataset.source_path,
        history=dataset.history,
        units=units,
        extra=extra,
    )
