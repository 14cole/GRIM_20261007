"""Known-answer checks for zero fields with physically undefined phase."""

import json

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import coherent_divide, decimate_axis


def _grid(power, phase, dtype):
    power = np.asarray(power, dtype=dtype).reshape(-1, 1, 1, 1)
    phase = np.asarray(phase, dtype=dtype).reshape(power.shape)
    return RcsGrid(
        np.arange(power.shape[0], dtype=float), [0.0], [10.0], ["HH"],
        rcs_power=power, rcs_phase=phase,
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_reconstructs_without_phase_in_every_field_reader(dtype):
    grid = _grid([0, 4, np.nan, 4], [np.nan, np.nan, 0, 0], dtype)
    expected = np.asarray([0, complex(np.nan, np.nan), complex(np.nan, np.nan), 2])
    read, reader_dtype = grid._bounded_complex_slice_reader()
    whole = (slice(None), slice(None), slice(None), slice(None))
    for actual in (grid.rcs, grid.rcs_slice(whole), read(whole)):
        np.testing.assert_allclose(actual.ravel(), expected, equal_nan=True)
        assert actual.dtype == (np.complex64 if dtype == np.float32 else np.complex128)
    assert reader_dtype == np.dtype(dtype)
    assert grid.rcs_slice((0, 0, 0, 0)) == 0j
    assert np.isnan(grid.rcs_phase[0, 0, 0, 0])
    assert np.isnan(grid.rcs_power[2, 0, 0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_is_identity_in_coherent_arithmetic_and_counts_as_usable(dtype):
    zero = _grid([0, 0], [np.nan, np.nan], dtype)
    field = _grid([4, 9], [0, np.pi / 2], dtype)
    zero_power = zero.rcs_power.copy()
    zero_phase = zero.rcs_phase.copy()
    cases = (
        (zero.coherent_add(field), field.rcs),
        (field.coherent_add(zero), field.rcs),
        (field.coherent_add_many(zero, zero), field.rcs),
        (field.coherent_subtract(zero), field.rcs),
        (zero.coherent_subtract(field), -field.rcs),
    )
    for result, expected in cases:
        np.testing.assert_allclose(result.rcs, expected, rtol=1e-6, atol=1e-6)
        qa = json.loads(result.extra["coherent_sample_qa_json"])
        assert qa["usable_sample_count"] == 2
        assert qa["masked_sample_count"] == 0
    np.testing.assert_array_equal(zero.rcs_power, zero_power)
    np.testing.assert_array_equal(zero.rcs_phase, zero_phase)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_zero_inputs_are_valid_coherent_results(dtype):
    zero = _grid([0, 0], [np.nan, np.nan], dtype)
    for result in (
        zero.coherent_add(zero),
        zero.coherent_add_many(zero, zero),
        zero.coherent_subtract(zero),
        zero.statistics_dataset("mean", axes=("azimuth",), domain="complex"),
        decimate_axis(zero, axis="azimuth", factor=2, mode="coherent"),
    ):
        np.testing.assert_array_equal(result.rcs_power, 0)
        np.testing.assert_array_equal(result.rcs, 0j)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_numerator_divides_but_zero_denominator_remains_undefined(dtype):
    numerator = _grid([0, 4, 0, np.nan, 4], [np.nan, 0, np.nan, 0, np.nan], dtype)
    denominator = _grid([4, 0, 0, 4, 4], [0, np.nan, np.nan, 0, 0], dtype)
    result = coherent_divide(numerator, denominator)
    np.testing.assert_allclose(
        result.rcs_power.ravel(), [0, np.nan, np.nan, np.nan, np.nan], equal_nan=True,
    )
    assert result.rcs[0, 0, 0, 0] == 0j
    assert result.linear_quantity() == "power_ratio"
    zero = _grid([0, 0], [np.nan, np.nan], dtype)
    field = _grid([4, 4], [0, 0], dtype)
    np.testing.assert_array_equal(coherent_divide(zero, field).rcs_power, 0)
    with pytest.raises(ValueError, match="nonzero denominator"):
        coherent_divide(field, zero)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complex_mean_and_decimation_include_zeros_but_exclude_missing_samples(dtype):
    # Fields [0, 2, missing, unknown] have mean 1, hence power 1.
    grid = _grid([0, 4, np.nan, 4], [np.nan, 0, 0, np.nan], dtype)
    mean = grid.statistics_dataset("mean", axes=("azimuth",), domain="complex")
    decimated = decimate_axis(grid, axis="azimuth", factor=4, mode="coherent")
    np.testing.assert_allclose(mean.rcs_power, 1)
    np.testing.assert_allclose(decimated.rcs_power, 1)
    # Repeated decimation must also retain zero outputs whose phase is NaN.
    zero = _grid([0, 0, 0, 0], [np.nan, np.nan, np.nan, np.nan], dtype)
    first = decimate_axis(zero, axis="azimuth", factor=2, mode="coherent")
    assert np.isnan(first.rcs_phase).all()
    second = decimate_axis(first, axis="azimuth", factor=2, mode="coherent")
    np.testing.assert_array_equal(second.rcs_power, 0)
    np.testing.assert_array_equal(second.rcs, 0j)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_invalid_power_and_nonzero_unknown_phase_stay_missing_in_arithmetic(dtype):
    left = _grid([0, np.nan, 4], [np.nan, 0, np.nan], dtype)
    right = _grid([4, 4, 4], [0, 0, 0], dtype)
    for result in (left.coherent_add(right), left.coherent_subtract(right)):
        np.testing.assert_allclose(result.rcs_power.ravel(), [4, np.nan, np.nan], equal_nan=True)
        qa = json.loads(result.extra["coherent_sample_qa_json"])
        assert qa["usable_sample_count"] == 1
        assert qa["masked_sample_count"] == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("phase_wrap", ["-180_180", "0_360"])
def test_constructor_phase_wrap_preserves_in_range_precision_and_half_open_endpoints(dtype, phase_wrap):
    source_phase = np.asarray(
        [-1.0e-18, 1.0e-18, -np.pi, np.pi, 0.0, 2.0 * np.pi, -2.0 * np.pi, np.nan],
        dtype=dtype,
    )
    phase_before = source_phase.copy()
    source = RcsGrid(
        np.arange(source_phase.size, dtype=float), [0.0], [10.0], ["HH"],
        rcs_power=np.ones((source_phase.size, 1, 1, 1), dtype=dtype),
        rcs_phase=source_phase.reshape(-1, 1, 1, 1),
        units={"phase_wrap": phase_wrap},
    )
    result = source.rcs_phase.ravel()
    lower, upper = (-np.pi, np.pi) if phase_wrap == "-180_180" else (0.0, 2.0 * np.pi)
    in_range = np.isfinite(source_phase) & (source_phase >= lower) & (source_phase < upper)
    np.testing.assert_array_equal(result[in_range], source_phase[in_range])
    finite = result[np.isfinite(result)]
    assert np.all(finite >= lower)
    assert np.all(finite < upper)
    assert np.isnan(result[-1])
    assert result[1] == source_phase[1]
    if phase_wrap == "-180_180":
        assert result[0] == source_phase[0]
        np.testing.assert_array_equal(result[2:4], np.asarray([-np.pi, -np.pi], dtype=dtype))
    else:
        assert result[5] == 0.0
    np.testing.assert_array_equal(source_phase, phase_before)
