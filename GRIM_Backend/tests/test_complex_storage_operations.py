"""Operations retain complex samples without round trips through polar form."""

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import (
    coherent_divide, decimate_axis, duplicate_dataset, offset_db, shift_dataset,
    time_gate, translate_phase_center,
)


def _grid(dtype=np.complex128):
    field = np.asarray([
        1.0 + 1e-18j, -2.0 + 1e-18j, 1e-18 + 3.0j,
        1.0 - 1e-18j, -2.0 - 1e-18j, -1e-18 - 3.0j,
    ], dtype=dtype).reshape(2, 1, 3, 1)
    return RcsGrid(
        [0.0, 10.0], [5.0], [1.0, 1.1, 1.2], ["HH"], rcs=field,
        units={"frequency": "GHz", "azimuth": "deg", "elevation": "deg",
               "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"},
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_exact_sample_operations_preserve_quadratures_and_precision(dtype):
    source = _grid(dtype)
    expected = source.rcs.copy()
    cases = [
        (duplicate_dataset(source), expected),
        (source.axis_crop(azimuths=[10.0]), expected[1:]),
        (source.shift_azimuth(2.0), expected),
        (source.shift_elevation(2.0), expected),
        (source.convert_axis_units(frequency="Hz"), expected),
        (source.edit_axis_value("azimuth", 0, 20.0), expected[::-1]),
        (source.mirror_about_azimuth(0.0), expected[::-1]),
        (source.swap_elevation_azimuth(), expected.swapaxes(0, 1)),
        (source.wrap_phase("0_360"), expected),
        (source.round_axes(0, frequencies=False), expected),
    ]
    for result, field in cases:
        assert result.storage_kind == "complex"
        assert result.rcs.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.rcs, field)
    assert source.storage_kind == "complex"
    np.testing.assert_array_equal(source.rcs, expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_coherent_arithmetic_preserves_small_quadrature_components(dtype):
    source = _grid(dtype)
    unity = RcsGrid(
        source.azimuths, source.elevations, source.frequencies,
        source.polarizations, rcs=np.ones(source.shape, dtype=dtype), units=source.units,
    )
    expected = source.rcs.copy()
    for result, field in (
        (source.coherent_subtract(unity), expected - 1),
        (source.coherent_add(unity), expected + 1),
        (coherent_divide(source, unity), expected),
    ):
        assert result.storage_kind == "complex"
        assert result.rcs.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.rcs, field)
    assert source.storage_kind == "complex"


def test_offsets_rotate_and_scale_complex_samples_directly():
    source = _grid()
    expected = source.rcs.copy()
    shifted = shift_dataset(source, phase_degrees=30.0)
    scaled = offset_db(source, 20.0)
    assert shifted.storage_kind == scaled.storage_kind == "complex"
    np.testing.assert_array_equal(shifted.rcs, expected * np.exp(1j * np.pi / 6))
    np.testing.assert_array_equal(scaled.rcs, expected * 10.0)
    assert source.storage_kind == "complex"


def test_incoherent_operations_and_unknown_phase_keep_power_only_storage():
    source = _grid()
    result = source.incoherent_add(source)
    assert result.storage_kind == "power"
    np.testing.assert_allclose(result.power, 2 * np.abs(source.rcs) ** 2)
    assert np.isnan(np.asarray(result.phase)).all()
    assert np.isnan(result.rcs).all()
    cropped = result.axis_crop(azimuths=[0.0])
    assert cropped.storage_kind == "power"
    assert np.isnan(np.asarray(cropped.phase)).all()


def test_mixed_phase_crop_does_not_invent_complex_samples():
    source = RcsGrid(
        [0.0, 1.0], [0.0], [1.0], ["HH"],
        rcs_power=np.array([4.0, 9.0]).reshape(2, 1, 1, 1),
        rcs_phase=np.array([0.0, np.nan]).reshape(2, 1, 1, 1),
    )
    duplicate = duplicate_dataset(source)
    np.testing.assert_array_equal(duplicate.power, source.power)
    np.testing.assert_array_equal(duplicate.phase, source.phase)
    assert duplicate.rcs[0, 0, 0, 0] == 2.0
    assert np.isnan(duplicate.rcs[1, 0, 0, 0])


def test_time_gate_retains_complex_output_and_source_storage():
    frequencies = np.linspace(1.0, 2.0, 32)
    field = np.exp(-1j * 4 * np.pi * frequencies * 1e9 * 0.2 / 299792458.0)
    source = RcsGrid(
        [0.0], [0.0], frequencies, ["HH"], rcs=field.reshape(1, 1, -1, 1),
        units={"frequency": "GHz"},
    )
    result = time_gate(source, start_m=-0.2, stop_m=0.8, compensate=False)
    assert source.storage_kind == result.storage_kind == "complex"
    assert np.isfinite(result.rcs).all()


def test_complex_regridding_and_decimation_do_not_round_trip_phase():
    source = _grid()
    expected = source.rcs.copy()
    interpolated = source.interpolate_axis("azimuth", [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(interpolated.rcs[[0, 2]], expected)
    np.testing.assert_array_equal(interpolated.rcs[1], (expected[0] + expected[1]) / 2)
    decimated = decimate_axis(source, axis="frequency", factor=3, mode="coherent")
    np.testing.assert_array_equal(decimated.rcs, np.mean(expected, axis=2, keepdims=True))
    assert interpolated.storage_kind == decimated.storage_kind == "complex"
    assert source.storage_kind == "complex"


def test_phase_center_translation_uses_the_existing_field():
    source = _grid()
    expected = source.rcs.copy()
    translated = translate_phase_center(source, x_m=0.01, y_m=0.0, z_m=0.0)
    projection = np.cos(np.deg2rad(source.azimuths)) * np.cos(np.deg2rad(5.0)) * 0.01
    ramp = -4 * np.pi * projection[:, None] * source.frequencies[None, :] * 1e9 / 299792458.0
    np.testing.assert_allclose(translated.rcs, expected * np.exp(1j * ramp[:, None, :, None]), rtol=2e-15)
    assert source.storage_kind == translated.storage_kind == "complex"
