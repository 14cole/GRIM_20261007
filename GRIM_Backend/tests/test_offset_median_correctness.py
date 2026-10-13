"""Known-answer cases for precise offsets and physical median windows."""

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import medianize_azimuth, offset_db


def _raw_grid(raw, quantity, frequencies=(1.0,)):
    raw = np.asarray(raw, dtype=np.complex128).reshape(-1, 1, len(frequencies), 1)
    if quantity == "sigma_2d":
        k = 2.0 * np.pi * np.asarray(frequencies) * 1.0e9 / C0
        field = raw / (2.0 * np.sqrt(k))[None, None, :, None]
    else:
        field = raw * np.sqrt(4.0 * np.pi)
    return RcsGrid(
        np.arange(raw.shape[0], dtype=float), [0.0], frequencies, ["VV"],
        rcs_power=np.asarray(np.abs(field) ** 2, dtype=np.float32),
        rcs_phase=np.asarray(np.angle(field), dtype=np.float32),
        units={
            "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
            "rcs_linear_quantity": quantity,
            "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm",
        },
        extra={"rcs_amp_real": raw.real.copy(), "rcs_amp_imag": raw.imag.copy()},
    ), field


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
@pytest.mark.parametrize("offset", [0.0, 6.0])
def test_offset_preserves_small_coherent_residual(quantity, offset):
    epsilon = 1.0e-8
    left, left_field = _raw_grid([1.0 + epsilon], quantity)
    right, right_field = _raw_grid([1.0], quantity)
    # The cached display values have already rounded away the difference.
    np.testing.assert_array_equal(left.rcs_power, right.rcs_power)
    power_before = left.rcs_power.copy()
    raw_before = left.extra["rcs_amp_real"].copy()

    shifted_left = offset_db(left, offset)
    shifted_right = offset_db(right, offset)
    residual = shifted_left.coherent_subtract(shifted_right)
    expected = np.abs(left_field - right_field) ** 2 * 10.0 ** (offset / 10.0)

    np.testing.assert_allclose(residual.rcs_power, expected, rtol=2.0e-7, atol=0.0)
    assert shifted_left.rcs_power.dtype == np.float64
    np.testing.assert_array_equal(left.rcs_power, power_before)
    np.testing.assert_array_equal(left.extra["rcs_amp_real"], raw_before)
    assert not np.shares_memory(shifted_left.rcs_power, left.rcs_power)


@pytest.mark.parametrize("reference_phase", [0.0, np.pi, -np.pi])
@pytest.mark.parametrize("offset", [0.0, 6.0])
def test_offset_preserves_small_phase_residual_across_signed_wrap(reference_phase, offset):
    left, left_field = _raw_grid([np.exp(1j * (reference_phase + 1.0e-8))], "sigma_3d")
    right, right_field = _raw_grid([np.exp(1j * reference_phase)], "sigma_3d")
    for source in (left, right):
        source.units["phase_wrap"] = "-180_180"

    residual = offset_db(left, offset).coherent_subtract(offset_db(right, offset))

    expected = np.abs(left_field - right_field) ** 2 * 10.0 ** (offset / 10.0)
    np.testing.assert_allclose(residual.rcs_power, expected, rtol=2.0e-7, atol=0.0)


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
@pytest.mark.parametrize("offset", [-6.0, 0.0, 6.0])
def test_offset_normalizes_authoritative_field_at_each_frequency(quantity, offset):
    raw = np.asarray([[1.0 + 0.2j, -2.0 + 1.0j], [0.0j, np.nan + 1j * np.nan]])
    source, field = _raw_grid(raw, quantity, frequencies=(1.0, 4.0))
    source.rcs_phase[:] = np.nan  # Raw quadratures still provide valid phase.

    result = offset_db(source, offset)

    np.testing.assert_allclose(
        result.rcs_power, np.abs(field) ** 2 * 10.0 ** (offset / 10.0),
        rtol=2.0e-15, atol=0.0, equal_nan=True,
    )
    np.testing.assert_allclose(
        result.rcs, field * 10.0 ** (offset / 20.0),
        rtol=2.0e-15, atol=0.0, equal_nan=True,
    )
    assert result.linear_quantity() == quantity
    assert np.isnan(result.rcs_power[-1, 0, -1, 0])
    assert result.rcs_power[-1, 0, 0, 0] == 0.0


def test_offset_preserves_power_only_ratio_and_missing_samples():
    source = RcsGrid(
        [0.0, 1.0, 2.0], [0.0], [1.0], ["VV"],
        rcs_power=np.asarray([0.0, 4.0, np.nan]).reshape(3, 1, 1, 1),
        units={"rcs_linear_quantity": "power_ratio", "rcs_log_unit": "dB"},
    )

    result = offset_db(source, 10.0)

    np.testing.assert_allclose(result.rcs_power.ravel(), [0.0, 40.0, np.nan])
    assert np.isnan(result.rcs_phase).all()
    assert result.linear_quantity() == "power_ratio"


@pytest.mark.parametrize("raw_shape", [(1024, 512), (1, 512_000)])
def test_offset_reads_raw_field_in_bounded_blocks(monkeypatch, raw_shape):
    source, _ = _raw_grid(
        np.full(raw_shape, 1.0 + 0.25j), "sigma_3d",
        frequencies=np.arange(1.0, raw_shape[1] + 1.0),
    )
    original_reader = source._authoritative_raw_amplitude_from_pair
    block_sizes = []

    def read(pair, selection=None):
        assert selection is not None, "offset must not allocate a full complex grid"
        block_sizes.append(source.rcs_power[selection].size)
        return original_reader(pair, selection)

    monkeypatch.setattr(source, "_authoritative_raw_amplitude_from_pair", read)
    result = offset_db(source, 0.0)

    assert len(block_sizes) > 1
    assert max(block_sizes) * 48 <= 16 * 1024**2
    np.testing.assert_allclose(result.rcs_power, 4.0 * np.pi * 1.0625, rtol=1e-15)


def _median_grid(azimuths, power, angle_unit):
    azimuths = np.asarray(azimuths, dtype=float)
    if angle_unit == "rad":
        azimuths = np.deg2rad(azimuths)
    return RcsGrid(
        azimuths, [0.0], [1.0], ["VV"],
        rcs_power=np.asarray(power, dtype=float).reshape(-1, 1, 1, 1),
        units={"azimuth": angle_unit, "frequency": "GHz"},
    )


@pytest.mark.parametrize("angle_unit", ["deg", "rad"])
def test_nonperiodic_median_does_not_fill_empty_windows(angle_unit):
    source = _median_grid([0.0, 1.0, 100.0, 101.0], [0.0, 4.0, 16.0, 36.0], angle_unit)
    before = source.rcs_power.copy()

    result = medianize_azimuth(
        source, window_degrees=2.0, slide_degrees=10.0, periodic=False,
    )

    centers = np.rad2deg(result.azimuths) if angle_unit == "rad" else result.azimuths
    np.testing.assert_allclose(centers, np.arange(1.0, 92.0, 10.0))
    assert result.rcs_power[0, 0, 0, 0] == 2.0  # Median includes measured zero.
    assert np.isnan(result.rcs_power[1:]).all()
    assert np.isnan(result.rcs_phase).all()
    np.testing.assert_array_equal(source.rcs_power, before)


@pytest.mark.parametrize("angle_unit", ["deg", "rad"])
def test_periodic_median_preserves_gaps_and_counts_seam_once(angle_unit):
    source = _median_grid(
        [0.0, 1.0, 100.0, 101.0, 359.0, 360.0],
        [0.0, 4.0, 16.0, 36.0, 30.0, 0.0], angle_unit,
    )

    result = medianize_azimuth(
        source, window_degrees=2.0, slide_degrees=10.0, periodic=True,
    )

    expected = np.full(36, np.nan)
    expected[0] = 4.0  # Median of 359, 0, and 1 degrees, with 360 counted once.
    expected[10] = 26.0  # Only the measured 100- and 101-degree samples.
    np.testing.assert_allclose(result.rcs_power.ravel(), expected, equal_nan=True)


def test_median_window_with_only_missing_measurements_stays_missing():
    source = _median_grid([0.0, 1.0, 2.0, 3.0], [np.nan, np.nan, 8.0, 10.0], "deg")
    with pytest.warns(RuntimeWarning, match="All-NaN slice"):
        result = medianize_azimuth(
            source, window_degrees=1.0, slide_degrees=2.0, periodic=False,
        )
    np.testing.assert_allclose(result.rcs_power.ravel(), [np.nan, 9.0])
