"""Analytic RF examples and finite-support contracts for engineering diagnostics."""
from dataclasses import replace

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.engineering_tools import (
    EngineeringCut, dataset_cut, finite_runs, level_crossings, measure_trace,
    phase_trace, phase_slope_fit, aligned_cuts, phase_residual, channel_comparison,
)


def make_grid(field=None, *, frequencies=None, azimuths=(0.,), polarizations=("HH",), units=None):
    frequencies = np.linspace(9., 10., 21) if frequencies is None else np.asarray(frequencies)
    shape = (len(azimuths), 1, len(frequencies), len(polarizations))
    values = np.ones(shape, complex) if field is None else np.broadcast_to(np.asarray(field, complex), shape).copy()
    return RcsGrid(azimuths, [0.], frequencies, polarizations, rcs=values,
                   units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                          "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm", **(units or {})})


def cut(field, x=None, *, unit="GHz"):
    field = np.asarray(field, dtype=complex)
    x = np.arange(len(field), dtype=float) if x is None else np.asarray(x, float)
    power = np.abs(field) ** 2
    return EngineeringCut(x, power, field, np.zeros(len(x)), "frequency", unit,
                          "dBsm", "test", "+jwt", [])


def test_peak_lobe_crossings_and_region_statistics_have_analytic_width():
    x = np.arange(7.)
    y = np.array([-9., -6., -3., 0., -3., -6., -9.])
    report = measure_trace(x, y, level=-4.5, drop=3., region=(1., 5.))
    assert report["peak"] == {"index": 3, "x": 3., "y": 0.}
    assert report["minimum"]["y"] == -6.
    assert report["crossings"] == [1.5, 4.5]
    assert report["bandwidth"] == {"left": 2., "right": 4., "width": 2.}
    assert report["mean"] == -3.6
    assert report["coverage"] == 1.


def test_crossings_and_bandwidth_never_bridge_missing_or_large_gaps():
    assert level_crossings([0, 1, 2], [-2, np.nan, 2], 0).size == 0
    assert level_crossings([0, 10], [-2, 2], 0, maximum_gap=2).size == 0
    report = measure_trace(np.arange(5.), [-5, -1, 0, np.nan, -5], drop=3.)
    assert report["bandwidth"] is None
    assert report["coverage"] == .8


def test_plateau_crossings_and_local_peaks_use_native_indices():
    np.testing.assert_array_equal(level_crossings(np.arange(6), [-1, 0, 0, 0, 0, 1], 0), [1, 4])
    report = measure_trace(np.arange(9), [0, 3, 0, 1, 1, 1, 0, 4, 0])
    assert [p["index"] for p in report["peaks"]] == [1, 4, 7]
    assert report["peak"]["index"] == 7


def test_empty_region_reports_no_measurement():
    report = measure_trace([0, 1], [3, 5], region=(10, 20))
    assert report["finite_count"] == 0 and "peak" not in report


def test_zero_drop_width_reports_the_contiguous_peak_plateau():
    report = measure_trace([0, 1, 2, 3], [0, 2, 2, 0], drop=0)
    assert report["bandwidth"] == {"left": 1., "right": 2., "width": 1.}
    assert measure_trace([0, 1, 2], [0, 2, 0], drop=0)["bandwidth"]["width"] == 0


@pytest.mark.parametrize("unit,scale", [("GHz", 1e9), ("MHz", 1e6), ("Hz", 1.)])
def test_sigma2d_cut_preserves_authoritative_precision_and_undefined_zero_phase(unit, scale):
    frequencies = np.array([9e9, 10e9, 11e9]) / scale
    raw = np.array([1.0000000002 + .234567890123j, 0j, 2.0000000004 - .7j]).reshape(1, 1, -1, 1)
    field = raw / (2 * np.sqrt(2 * np.pi * frequencies * scale / C0))[None, None, :, None]
    dataset = RcsGrid([0.], [0.], frequencies, ["HH"],
                      rcs_power=(np.abs(field) ** 2).astype(np.float32), rcs_phase=np.angle(field).astype(np.float32),
                      units={"frequency": unit, "rcs_linear_quantity": "sigma_2d", "rcs_log_unit": "dBke"},
                      extra={"rcs_amp_real": raw.real, "rcs_amp_imag": raw.imag, "raw_complex_amplitude_preserved": True})
    result = dataset_cut(dataset)
    np.testing.assert_allclose(result.field, field.ravel(), rtol=3e-15, atol=0)
    np.testing.assert_allclose(result.power, np.abs(field.ravel()) ** 2, rtol=3e-15)
    assert result.level_unit == "dBke"
    assert np.isneginf(result.db[1])
    assert np.isnan(phase_trace(result.x, result.field)["wrapped"][1])


def test_power_only_cut_keeps_magnitude_without_inventing_phase():
    dataset = RcsGrid([0], [0], [1, 2, 3], ["HH"], rcs_power=np.array([1., 4., 9.]).reshape(1, 1, 3, 1))
    result = dataset_cut(dataset)
    np.testing.assert_allclose(result.power, [1, 4, 9])
    assert not np.any(phase_trace(result.x, result.field)["valid"])


def test_phase_unwrap_resets_at_weak_and_missing_samples():
    angles = np.deg2rad([170, -170, -150, 0, 170, -170, 0, -170, 170])
    field = np.exp(1j * angles); field[3] *= 1e-6; field[6] = complex(np.nan, np.nan)
    report = phase_trace(np.arange(9.), field, floor_db=-40.)
    np.testing.assert_allclose(np.rad2deg(report["unwrapped"][[0, 1, 2, 4, 5, 7, 8]]), [170, 190, 210, 170, 190, -170, -190])
    assert report["finite_count"] == 7
    assert [len(run) for run in report["runs"]] == [3, 2, 2]


@pytest.mark.parametrize("time_sign,phase_sign", [("+jwt", -1), ("-jwt", 1)])
def test_phase_slope_sorted_hz_known_delay_and_fit_quality(time_sign, phase_sign):
    f = np.linspace(9e9, 10e9, 101)
    delay = .65e-9
    field = np.exp(phase_sign * 2j * np.pi * f * delay)
    result = phase_slope_fit(f[::-1], field[::-1], time_sign=time_sign)
    fit = result["fits"][0]
    assert fit["apparent_delay_s"] == pytest.approx(delay, rel=1e-12)
    assert fit["r_squared"] == pytest.approx(1)
    assert fit["residual_rms_deg"] < 1e-10
    assert fit["alias_period_s"] == pytest.approx(1e-7)
    assert "not target range" in result["note"] and "aliasing" in result["note"]


def test_slope_respects_selected_region_and_separate_finite_runs():
    f = np.arange(11.) * 1e6 + 9e9
    field = np.exp(-2j * np.pi * f * 1e-9)
    field[5] = np.nan + 1j * np.nan
    result = phase_slope_fit(f, field, region_hz=(f[1], f[9]))
    assert [fit["samples"] for fit in result["fits"]] == [4, 4]
    assert result["finite_count"] == 8
    assert all(fit["apparent_delay_s"] == pytest.approx(1e-9) for fit in result["fits"])


def test_sparse_phase_runs_give_diagnostics_without_a_fabricated_fit():
    result = phase_slope_fit([1, 2, 5, 6], np.ones(4), maximum_gap_hz=2)
    assert len(result["fits"]) == 2
    assert all("note" in fit for fit in result["fits"])
    assert np.all(np.isnan(result["fitted"]))


@pytest.mark.parametrize("time_sign,phase_sign", [("+jwt", -1), ("-jwt", 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_slope_preserves_native_gaps_after_frequency_sorting(time_sign, phase_sign, reverse):
    native = np.array([0, 1, 2, 50, 51, 52])
    frequency = 9e9 + native * 10e6
    field = np.exp(phase_sign * 2j * np.pi * (frequency - 9e9) * 5e-9)
    # The second group has an independent phase reference. Its slope is still
    # 5 ns, and neither reference may be inferred through the omitted samples.
    field[3:] *= np.exp(.73j)
    if reverse:
        frequency, field, native = frequency[::-1], field[::-1], native[::-1]
    snapshots = tuple(value.copy() for value in (frequency, field, native))
    result = phase_slope_fit(frequency, field, native_indices=native, time_sign=time_sign)
    assert [fit["samples"] for fit in result["fits"]] == [3, 3]
    assert [run.tolist() for run in result["runs"]] == [[0, 1, 2], [3, 4, 5]]
    for fit in result["fits"]:
        assert fit["apparent_delay_s"] == pytest.approx(5e-9, rel=1e-12)
        assert fit["r_squared"] == pytest.approx(1.)
        assert fit["residual_rms_deg"] < 1e-10
    for actual, before in zip((frequency, field, native), snapshots):
        np.testing.assert_array_equal(actual, before)


def test_phase_unwrap_resets_at_native_gaps_without_discarding_endpoints():
    native = np.array([0, 1, 2, 50, 51, 52])
    wrapped_deg = np.array([170, -170, -150, -170, 170, 150])
    result = phase_trace(native * .01, np.exp(1j * np.deg2rad(wrapped_deg)),
                         native_indices=native)
    np.testing.assert_allclose(np.rad2deg(result["unwrapped"]),
                               [170, 190, 210, -170, -190, -210])
    assert result["finite_count"] == result["total_count"] == 6
    assert [run.tolist() for run in result["runs"]] == [[0, 1, 2], [3, 4, 5]]


def test_native_gaps_combine_with_region_weak_samples_and_physical_gap_limit():
    native = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11])
    frequency = np.array([1, 2, 3, 4, 5, 6, 7, 20, 21.])
    field = np.ones(len(frequency), complex)
    field[5] = 1e-6
    result = phase_trace(frequency, field, native_indices=native, floor_db=-40,
                         maximum_gap=2, region=(2, 20))
    assert [run.tolist() for run in result["runs"]] == [[1, 2], [3, 4], [6], [7]]
    assert result["finite_count"] == 6
    assert np.isnan(result["unwrapped"][[0, 5, 8]]).all()


def test_full_native_sweep_keeps_existing_phase_and_delay_results():
    frequency = np.linspace(9e9, 10e9, 101)
    field = np.exp(-2j * np.pi * (frequency - frequency[0]) * 5e-9)
    before = phase_slope_fit(frequency, field)
    after = phase_slope_fit(frequency, field, native_indices=np.arange(len(frequency)))
    assert after["fits"] == before["fits"]
    for name in ("wrapped", "unwrapped", "fitted", "residual", "valid"):
        np.testing.assert_array_equal(after[name], before[name])


def test_native_singletons_remain_visible_without_a_cross_gap_fit():
    frequency = np.array([9e9, 9.5e9, 10e9])
    result = phase_slope_fit(frequency, np.ones(3), native_indices=[0, 50, 100])
    assert [fit["samples"] for fit in result["fits"]] == [1, 1, 1]
    assert all("apparent_delay_s" not in fit for fit in result["fits"])
    assert np.isfinite(result["unwrapped"]).all()
    assert np.isnan(result["fitted"]).all()


def test_phase_residual_matches_physical_units_without_interpolation():
    a = cut(np.exp(1j * np.array([.3, .6, .9])), [9, 10, 11])
    b = cut(np.exp(1j * np.array([.1, .4, .7])), [9000, 10000, 11000], unit="MHz")
    report = phase_residual(a, b)
    assert report["common_count"] == report["valid_count"] == 3
    np.testing.assert_allclose(report["phase_difference_rad"], .2, atol=1e-15)
    assert report["circular_mean_deg"] == pytest.approx(np.rad2deg(.2))
    assert aligned_cuts(a, replace(b, x=np.array([9001, 10001, 11001])))[0].size == 0


def test_ratio_and_reciprocity_report_known_channel_values():
    a, b = cut(2 * np.exp(.4j) * np.ones(4)), cut(np.ones(4))
    report = channel_comparison(a, b)
    np.testing.assert_allclose(report["ratio_db"], 20 * np.log10(2))
    assert report["circular_mean_deg"] == pytest.approx(np.rad2deg(.4))
    identical = channel_comparison(a, a)
    assert identical["normalized_complex_mismatch"] == 0
    assert identical["circular_mean_deg"] == pytest.approx(0, abs=1e-12)


def test_power_only_ratio_is_available_but_complex_mismatch_is_not():
    a = cut([1, 2, 3]); b = cut([1, 1, 1])
    a.field[:] = np.nan + 1j * np.nan
    report = channel_comparison(a, b)
    np.testing.assert_allclose(report["ratio_db"], 10 * np.log10([1, 4, 9]))
    assert report["normalized_complex_mismatch"] is None
    assert report["valid_count"] == 0


def test_circular_mean_does_not_claim_direction_for_opposite_angles():
    report = phase_residual(cut([1, -1]), cut([1, 1]))
    assert report["circular_mean_deg"] is None


def test_empty_coordinate_matching_is_valid():
    assert aligned_cuts(cut([]), cut([]))[0].size == 0


def test_invalid_coordinate_does_not_discard_valid_common_samples():
    left = cut([1, 2, 3], [9, 10, np.nan])
    right = cut([1, 2, 3], [9, 10, 11])
    ia, ib = aligned_cuts(left, right)
    np.testing.assert_array_equal(ia, [0, 1]); np.testing.assert_array_equal(ib, [0, 1])
    assert phase_residual(left, right)["common_count"] == 2
