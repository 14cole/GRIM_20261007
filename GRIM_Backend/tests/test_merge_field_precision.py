"""Known-answer merge cases for authoritative fields, zeros, and precision."""

from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid


def _normalization(quantity, frequencies):
    if quantity == "sigma_2d":
        k0 = 2.0 * np.pi * np.asarray(frequencies) * 1.0e9 / C0
        return 1.0 / (2.0 * np.sqrt(k0))[None, None, :, None]
    return np.sqrt(4.0 * np.pi)


def _solver(raw, *, quantity="sigma_3d", azimuths=(0.0,), frequencies=(1.0,), blank_phase=False):
    shape = (len(azimuths), 1, len(frequencies), 1)
    raw = np.asarray(raw, dtype=np.complex128).reshape(shape)
    field = raw * _normalization(quantity, frequencies)
    phase = np.angle(field).astype(np.float32)
    if blank_phase:
        phase[:] = np.nan
    return RcsGrid(
        azimuths, [0.0], frequencies, ["VV"],
        rcs_power=(np.abs(field) ** 2).astype(np.float32),
        rcs_phase=phase,
        units={
            "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
            "rcs_linear_quantity": quantity,
            "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm",
        },
        extra={"rcs_amp_real": raw.real.copy(), "rcs_amp_imag": raw.imag.copy()},
    )


def _ordinary(power, phase, *, azimuths=(0.0,), dtype=np.float64):
    shape = (len(azimuths), 1, 1, 1)
    return RcsGrid(
        azimuths, [0.0], [1.0], ["VV"],
        rcs_power=np.asarray(power, dtype=dtype).reshape(shape),
        rcs_phase=np.asarray(phase, dtype=dtype).reshape(shape),
        units={"frequency": "GHz", "rcs_linear_quantity": "sigma_3d"},
    )


@pytest.mark.parametrize("quantity", ["sigma_3d", "sigma_2d"])
def test_coherent_mean_uses_raw_field_with_blank_cached_phase(quantity):
    left = _solver([1j, 2j], quantity=quantity, frequencies=(1.0, 2.0), blank_phase=True)
    right = _solver([1.0, 2.0], quantity=quantity, frequencies=(1.0, 2.0))
    expected = np.array([0.5 + 0.5j, 1.0 + 1.0j]).reshape(1, 1, 2, 1)
    expected *= _normalization(quantity, (1.0, 2.0))

    result, report = RcsGrid.stitch_many(left, right, policy="coherent-mean", return_report=True)

    np.testing.assert_allclose(result.rcs, expected, rtol=1.0e-14)
    assert report["contributing_count"] == 4
    assert report["overlap_count"] == 2
    assert report["masked_missing_phase_sample_count"] == 0
    assert np.isnan(left.rcs_phase).all()


def test_coherent_mean_counts_zero_and_masks_only_unknown_nonzero_fields():
    left = _ordinary([0.0, 4.0, np.nan, 0.0], [np.nan] * 4, azimuths=(0, 1, 2, 3))
    right = _ordinary([4.0, 4.0, 4.0, 0.0], [0.0, 0.0, 0.0, np.nan], azimuths=(0, 1, 2, 3))

    with np.errstate(all="raise"):
        result, report = RcsGrid.stitch_many(left, right, policy="coherent-mean", return_report=True)

    np.testing.assert_allclose(result.rcs_power.ravel(), [1.0, 4.0, 4.0, 0.0])
    assert np.isnan(result.rcs_phase.ravel()[-1])
    assert result.rcs.ravel()[-1] == 0.0j
    assert report["contributing_count"] == 6
    assert report["overlap_count"] == 2
    assert report["equal_count"] == 1
    assert report["masked_missing_phase_sample_count"] == 1


@pytest.mark.parametrize("quantity", ["sigma_3d", "sigma_2d"])
@pytest.mark.parametrize("policy", ["priority-first", "priority-last", "power-mean", "coherent-mean"])
def test_disjoint_merge_preserves_weak_difference_between_strong_fields(quantity, policy):
    left = _solver([1.0 + 1.0e-8], quantity=quantity)
    right = _solver([1.0], quantity=quantity)
    extension = _solver([2.0], quantity=quantity, frequencies=(2.0,))
    np.testing.assert_array_equal(left.rcs_power, right.rcs_power)
    expected = abs((left.rcs - right.rcs).item()) ** 2

    merged_left = RcsGrid.stitch_many(left, extension, policy=policy)
    merged_right = RcsGrid.stitch_many(right, extension, policy=policy)
    residual = merged_left.coherent_subtract(merged_right).rcs_power[0, 0, 0, 0]

    assert residual > 0.0
    np.testing.assert_allclose(residual, expected, rtol=2.0e-7, atol=0.0)
    assert merged_left.rcs_power.dtype == np.float64


@pytest.mark.parametrize("quantity", ["sigma_3d", "sigma_2d"])
@pytest.mark.parametrize("policy", ["priority-first", "priority-last"])
def test_priority_overlap_preserves_selected_complex_field_precision(quantity, policy):
    phase = 0.7
    left = _solver([np.exp(1j * phase) * (1.0 + 1.0e-8)], quantity=quantity)
    right = _solver([np.exp(1j * phase)], quantity=quantity)
    other = _solver([5j], quantity=quantity)
    expected = abs((left.rcs - right.rcs).item()) ** 2
    inputs_left = (left, other) if policy == "priority-first" else (other, left)
    inputs_right = (right, other) if policy == "priority-first" else (other, right)

    merged_left = RcsGrid.stitch_many(*inputs_left, policy=policy)
    merged_right = RcsGrid.stitch_many(*inputs_right, policy=policy)
    actual = merged_left.coherent_subtract(merged_right).rcs_power.item()

    np.testing.assert_allclose(actual, expected, rtol=2.0e-7, atol=0.0)
    np.testing.assert_allclose(merged_left.rcs, left.rcs, rtol=5.0e-16)


@pytest.mark.parametrize("quantity", ["sigma_3d", "sigma_2d"])
def test_power_mean_uses_precise_power_but_masks_phase_only_at_overlaps(quantity):
    left = _solver([1 + 1.0e-8j, 1 + 1.0e-8], quantity=quantity, azimuths=(0, 1), blank_phase=True)
    right = _solver([2j, 3j], quantity=quantity, azimuths=(1, 2), blank_phase=True)
    left_before = left.rcs.copy()
    right_before = right.rcs.copy()

    result = RcsGrid.stitch_many(left, right, policy="power-mean")

    np.testing.assert_allclose(result.rcs_power.ravel(), [
        abs(left_before.ravel()[0]) ** 2,
        (abs(left_before.ravel()[1]) ** 2 + abs(right_before.ravel()[0]) ** 2) / 2.0,
        abs(right_before.ravel()[1]) ** 2,
    ], rtol=1.0e-15)
    assert np.isnan(result.rcs_phase.ravel()[1])
    np.testing.assert_allclose(result.rcs.ravel()[[0, 2]], [left_before.ravel()[0], right_before.ravel()[1]], rtol=1.0e-15)
    np.testing.assert_array_equal(left.rcs, left_before)
    np.testing.assert_array_equal(right.rcs, right_before)


@pytest.mark.parametrize("policy", ["priority-first", "priority-last", "power-mean", "coherent-mean"])
def test_merge_preserves_tiny_phase_and_uses_bounded_readers(policy):
    left = _solver([1 + 1.0e-18j, 1 + 2.0e-18j, 1 + 3.0e-18j], azimuths=(0, 1, 2))
    validation = RcsGrid._complete_authoritative_raw_arrays
    calls = []

    def counted_validation(grid):
        calls.append(grid)
        return validation(grid)

    with (
        mock.patch("GRIM_Backend.datasets.combine._JOIN_MERGE_BLOCK_CELLS", 1),
        mock.patch.object(RcsGrid, "_complete_authoritative_raw_arrays", counted_validation),
    ):
        result, report = RcsGrid.stitch_many(left, policy=policy, return_report=True)

    assert calls == [left]
    np.testing.assert_allclose(result.rcs_phase.ravel(), [1.0e-18, 2.0e-18, 3.0e-18], rtol=1.0e-15, atol=0.0)
    reference = RcsGrid.stitch_many(
        _solver([1.0, 1.0, 1.0], azimuths=(0, 1, 2)), policy=policy
    )
    residual = result.coherent_subtract(reference).rcs_power.ravel()
    np.testing.assert_allclose(
        residual, 4.0 * np.pi * np.array([1.0e-36, 4.0e-36, 9.0e-36]),
        rtol=2.0e-15, atol=0.0,
    )
    with pytest.raises(MemoryError):
        RcsGrid.stitch_many(left, policy=policy, max_output_bytes=report["estimated_peak_bytes"] - 1)


@pytest.mark.parametrize("policy", ["priority-first", "priority-last", "power-mean"])
def test_malformed_raw_metadata_keeps_power_only_samples_available(policy):
    left = _ordinary([4.0, 9.0], [np.nan, 0.3], azimuths=(0, 1))
    left.extra.update(rcs_amp_real=np.array([1.0]), rcs_amp_imag=np.array([2.0]))

    result = RcsGrid.stitch_many(left, policy=policy)

    np.testing.assert_allclose(result.rcs_power, left.rcs_power, rtol=1.0e-15)
    np.testing.assert_allclose(result.rcs_phase, left.rcs_phase, rtol=1.0e-15, equal_nan=True)


@pytest.mark.parametrize("policy", ["priority-first", "priority-last", "power-mean"])
@pytest.mark.parametrize("raw_problem", ["wrong_shape", "missing_field", "unsupported_quantity"])
def test_unusable_raw_metadata_preserves_cached_float32_samples_exactly(policy, raw_problem):
    source = _ordinary([10.0, 3.0], [0.3, 0.7], azimuths=(0, 1), dtype=np.float32)
    raw_shape = (1,) if raw_problem == "wrong_shape" else source.rcs_power.shape
    raw_real = np.ones(raw_shape)
    if raw_problem == "missing_field":
        raw_real.ravel()[-1] = np.nan
    source.extra.update(rcs_amp_real=raw_real, rcs_amp_imag=np.zeros(raw_shape))
    if raw_problem == "unsupported_quantity":
        source.units["rcs_linear_quantity"] = "power_ratio"
        source.units["rcs_log_unit"] = "dB"
    source_power = source.rcs_power.copy()
    source_phase = source.rcs_phase.copy()
    default_reader, default_dtype = source._bounded_complex_slice_reader()
    detailed_reader, detailed_dtype, authoritative = source._bounded_complex_slice_reader(include_authority=True)
    assert not authoritative
    assert default_dtype == detailed_dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(default_reader(np.s_[:]), detailed_reader(np.s_[:]))

    result = RcsGrid.stitch_many(source, policy=policy)

    np.testing.assert_array_equal(result.rcs_power, source_power)
    np.testing.assert_array_equal(result.rcs_phase, source_phase)
    np.testing.assert_array_equal(source.rcs_power, source_power)
    np.testing.assert_array_equal(source.rcs_phase, source_phase)
