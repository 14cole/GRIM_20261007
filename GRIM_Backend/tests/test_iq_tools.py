"""Analytic phasors, physical units, and native sampling for I/Q diagnostics."""
from dataclasses import replace

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.plotting.engineering_tools import EngineeringCut
from GRIM_Backend.plotting.iq_tools import (
    field_unit, iq_series, compare_iq, compensate_iq, repeat_iq,
)


def cut(field, x=None, *, unit="GHz", level_unit="dBsm", axis="frequency", label="test"):
    field = np.asarray(field, dtype=complex)
    x = np.arange(len(field), dtype=float) + 9. if x is None else np.asarray(x, float)
    power = np.abs(field) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10. * np.log10(power)
        if level_unit == "dBke":
            scale = {"Hz": 1., "MHz": 1.e6, "GHz": 1.e9}[unit] if axis == "frequency" else 1.e9
            frequency = x * scale if axis == "frequency" else 10.e9
            db += 10. * np.log10(2. * np.pi * frequency / C0)
    return EngineeringCut(x, power, field, db, axis, unit, level_unit, label, "+jwt", [])


@pytest.mark.parametrize("unit,expected", [("dBsm", "m"), ("dBke", "sqrt(m)"), ("dB", "dimensionless")])
def test_labels_describe_field_quantity(unit, expected):
    source = cut([1, 2j], level_unit=unit)
    assert field_unit(source) == expected
    assert iq_series(source)["display_unit"] == expected


def test_weak_values_and_nulls_remain_visible_but_zero_phase_is_undefined():
    source = cut([1, 1j, 1.e-5j, 0, complex(np.nan, 0), -1])
    result = iq_series(source, floor_db=-60.)
    np.testing.assert_array_equal(result["valid"], [1, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(result["phase_valid"], [1, 1, 0, 0, 0, 1])
    np.testing.assert_allclose(result["phase_deg"][[0, 1, 5]], [0, 90, 180])
    assert np.isnan(result["phase_deg"][3])
    assert [row.tolist() for row in result["runs"]] == [[0, 1, 2, 3], [5]]
    phase_only = iq_series(source, scaling="phase_only")
    assert not phase_only["valid"][3]
    assert phase_only["field"][2] == pytest.approx(1j)


def test_common_peak_comparison_uses_one_scale_and_signed_native_difference():
    a, b = cut([1, 2j]), cut([2, -4j])
    report = compare_iq(a, b, scaling="common_peak")
    assert report["left"]["scale"] == report["right"]["scale"] == report["difference"]["scale"] == 4.
    np.testing.assert_allclose(report["left"]["field"], [.25, .5j])
    np.testing.assert_allclose(report["right"]["field"], [.5, -1j])
    np.testing.assert_allclose(report["difference"]["field"], [-.25, 1.5j])
    np.testing.assert_allclose(report["difference"]["native_field"], [-1, 6j])
    np.testing.assert_allclose(report["left"]["db"], a.db)
    assert report["left"]["display_unit"] == "dimensionless"


def test_all_zero_common_peak_does_not_create_nan_or_invent_phase():
    result = compare_iq(cut([0, 0]), cut([0, 0]), scaling="common_peak")
    np.testing.assert_array_equal(result["difference"]["field"], [0j, 0j])
    assert np.all(result["difference"]["valid"])
    assert not np.any(result["difference"]["phase_valid"])
    assert np.all(np.isneginf(result["difference"]["db"]))


def test_comparison_converts_units_but_does_not_interpolate_or_bridge_missing_matches():
    a = cut([1, 2, 3, 4, 5], x=[9, 10, 11, 12, 13])
    b = cut([10, 30, 50], x=[9000, 11000, 13000], unit="MHz")
    result = compare_iq(a, b)
    np.testing.assert_array_equal(result["indices_left"], [0, 2, 4])
    np.testing.assert_array_equal(result["indices_right"], [0, 1, 2])
    np.testing.assert_allclose(result["right"]["x"], [9, 11, 13])
    for name in ("left", "right", "difference"):
        assert [run.tolist() for run in result[name]["runs"]] == [[0], [1], [2]]
    unmatched = compare_iq(a, cut([1, 2], x=[9.001, 10.001]))
    assert unmatched["common_count"] == 0


def test_large_coordinate_gap_and_region_split_trajectory_without_deleting_native_indices():
    result = iq_series(cut([1, 2, 3, 4], x=[1, 2, 9, 10]), maximum_gap=2, region=(2, 10))
    assert [row.tolist() for row in result["runs"]] == [[1], [2, 3]]
    np.testing.assert_array_equal(result["native_indices"], [0, 1, 2, 3])


def test_selected_sweep_keeps_original_indices_and_never_connects_unselected_samples():
    a = replace(cut([1, 2, 3], x=[9, 10, 12]), native_indices=np.array([4, 5, 7]))
    b = replace(cut([2, 3, 4], x=[9, 10, 12]), native_indices=np.array([9, 10, 12]))
    series = iq_series(a)
    np.testing.assert_array_equal(series["native_indices"], [4, 5, 7])
    assert [run.tolist() for run in series["runs"]] == [[0, 1], [2]]
    report = compare_iq(a, b)
    np.testing.assert_array_equal(report["indices_left"], [0, 1, 2])
    np.testing.assert_array_equal(report["left"]["native_indices"], [4, 5, 7])
    np.testing.assert_array_equal(report["right"]["native_indices"], [9, 10, 12])
    assert [run.tolist() for run in report["difference"]["runs"]] == [[0, 1], [2]]
    compensated = compensate_iq(a, phase_offset_deg=90)["compensated"]
    np.testing.assert_array_equal(compensated["native_indices"], [4, 5, 7])
    repeat = repeat_iq([a, b], 10.)
    np.testing.assert_array_equal(repeat["indices"], [5, 10])
    np.testing.assert_array_equal(repeat["positions"], [1, 1])
    assert repeat["reference_index"] == 1 and repeat["reference_native_index"] == 5


def test_metadata_differences_are_notes_but_incompatible_quantity_cannot_be_subtracted():
    a = cut([1, 2])
    b = replace(cut([1, 2]), time_convention="-jwt", notes=["phase_reference: unspecified"])
    assert any("Time-convention" in note for note in compare_iq(a, b)["notes"])
    with pytest.raises(ValueError, match="same physical"):
        compare_iq(a, cut([1, 2], level_unit="dBke"))


def test_explicit_linear_quantity_controls_units_and_compatibility_over_log_label():
    a, b = cut([1, 2], level_unit="dB"), cut([1, 2], level_unit="dB")
    a.linear_quantity = "sigma_2d"
    b.linear_quantity = "ratio"
    assert field_unit(a) == "sqrt(m)"
    assert field_unit(b) == "dimensionless"
    with pytest.raises(ValueError, match="same physical"):
        compare_iq(a, b)
    report = repeat_iq([a, b], 9.)
    assert report["count"] == 1 and len(report["skipped"]) == 1


def test_metadata_stays_with_each_series_and_repeat_sample():
    a = replace(cut([1, 2]), linear_quantity="sigma_3d", polarization="HH",
                fixed_coordinates={"azimuth": (30., "deg"), "elevation": (0., "deg")})
    b = replace(cut([2, 3], level_unit="dB"), linear_quantity="sigma_3d", polarization="HH",
                fixed_coordinates={"azimuth": (np.pi / 6, "rad"), "elevation": (0., "rad")})
    report = compare_iq(a, b)
    assert report["left"]["linear_quantity"] == "sigma_3d"
    assert report["left"]["fixed_coordinates"]["azimuth"] == (30., "deg")
    assert report["right"]["fixed_coordinates"]["azimuth"] == (np.pi / 6, "rad")
    assert report["right"]["level_unit"] == "dB"
    repeated = repeat_iq([a, b], 9.)
    assert [row["level_unit"] for row in repeated["metadata"]] == ["dBsm", "dB"]
    assert repeated["metadata"][1]["fixed_coordinates"]["azimuth"] == (np.pi / 6, "rad")


def test_power_ratio_alias_uses_dimensionless_fields_and_matches_legacy_ratio():
    a = replace(cut([1, 2], level_unit="dB"), linear_quantity="power_ratio")
    b = replace(cut([2, 3], level_unit="dB"), linear_quantity="ratio")
    assert field_unit(a) == field_unit(b) == "dimensionless"
    report = compare_iq(a, b)
    assert report["common_count"] == 2
    assert report["left"]["linear_quantity"] == "ratio"


@pytest.mark.parametrize("unit,coordinates", [("GHz", [9, 10, 11]), ("MHz", [9000, 10000, 11000])])
def test_difference_dbke_preserves_wavenumber_factor_at_each_frequency(unit, coordinates):
    a = cut([0, 3j, 2], x=coordinates, unit=unit, level_unit="dBke")
    b = cut([1, 1j, 2], x=coordinates, unit=unit, level_unit="dBke")
    difference = compare_iq(a, b)["difference"]
    expected = cut([-1, 2j, 0], x=coordinates, unit=unit, level_unit="dBke")
    np.testing.assert_allclose(difference["db"], expected.db)


def test_angular_difference_dbke_recovers_fixed_frequency_reference_from_either_cut():
    a = cut([0, 0], x=[0, 10], unit="deg", level_unit="dBke", axis="azimuth")
    b = cut([1, 2j], x=[0, 10], unit="deg", level_unit="dBke", axis="azimuth")
    np.testing.assert_allclose(compare_iq(a, b)["difference"]["db"], b.db)


def test_angular_difference_dbke_uses_fixed_frequency_when_left_is_zero_and_right_plain_db():
    a = replace(cut([0, 0], x=[0, 10], unit="deg", level_unit="dBke", axis="azimuth"),
                linear_quantity="sigma_2d", fixed_coordinates={"frequency": (10000., "MHz")})
    b = replace(cut([1, 2j], x=[0, 10], unit="deg", level_unit="dB", axis="azimuth"),
                linear_quantity="sigma_2d", fixed_coordinates={"frequency": (10., "GHz")})
    expected = b.db + 10. * np.log10(2. * np.pi * 10.e9 / C0)
    np.testing.assert_allclose(compare_iq(a, b)["difference"]["db"], expected)
    # Missing frequency and a zero-only dBke cut leave no valid conversion
    # reference. A plain dB label cannot supply the dBke wavenumber factor.
    unknown = compare_iq(replace(a, fixed_coordinates={}), b)["difference"]
    assert np.all(np.isnan(unknown["db"]))
    np.testing.assert_allclose(unknown["native_field"], [-1, -2j])


@pytest.mark.parametrize("sign", ["+jwt", "-jwt"])
@pytest.mark.parametrize("unit,frequency_scale", [("GHz", 1.e9), ("MHz", 1.e6), ("Hz", 1.)])
def test_compensation_removes_known_delay_and_reference_phase_without_changing_levels(sign, unit, frequency_scale):
    hz = np.linspace(9.e9, 10.e9, 41)
    fref = 9.3e9
    delay = 1.2e-9
    offset = .7
    direction = -1 if sign == "+jwt" else 1
    field = 2. * np.exp(1j * (offset + direction * 2. * np.pi * (hz - fref) * delay))
    source = cut(field, x=hz / frequency_scale, unit=unit)
    before = source.field.copy()
    result = compensate_iq(source, phase_offset_deg=np.rad2deg(offset), delay_ns=1.2,
                           reference_frequency=fref / frequency_scale, time_sign=sign,
                           scaling="common_peak")
    np.testing.assert_allclose(result["compensated"]["native_field"], 2., atol=1.e-13)
    np.testing.assert_allclose(result["compensated"]["field"], 1., atol=1.e-13)
    np.testing.assert_array_equal(source.field, before)
    np.testing.assert_array_equal(result["compensated"]["db"], source.db)
    assert result["reference_frequency"] == fref / frequency_scale


def test_phase_offset_works_for_angle_cut_and_does_not_invent_delay_coordinate():
    source = cut([1j, 2j, 0], x=[0, 10, 20], unit="deg", axis="azimuth")
    result = compensate_iq(source, phase_offset_deg=90)
    np.testing.assert_allclose(result["compensated"]["native_field"], [1, 2, 0], atol=1.e-14)
    assert result["reference_frequency"] is None
    with pytest.raises(ValueError, match="frequency sweep"):
        compensate_iq(source, delay_ns=1)


def test_repeatability_snaps_once_then_matches_same_physical_coordinate_and_reports_skips():
    a = cut([100, 1 + 1j, 100], x=[9, 10, 11], label="A")
    b = cut([100, 1 - 1j, 100], x=[9000, 10000, 11000], unit="MHz", label="B")
    missing = cut([5, 6], x=[9.9, 10.1], label="No exact sample")
    nonfinite = cut([1, complex(np.nan, 0), 1], x=[9, 10, 11], label="Missing field")
    result = repeat_iq([a, b, missing, nonfinite], 10.2)
    assert result["coordinate"] == 10.
    assert result["requested_coordinate"] == 10.2
    assert result["count"] == 2
    assert result["labels"] == ["A", "B"]
    np.testing.assert_array_equal(result["indices"], [1, 1])
    np.testing.assert_array_equal(result["cut_indices"], [0, 1])
    np.testing.assert_array_equal(result["native_field"], [1 + 1j, 1 - 1j])
    assert result["mean_field"] == 1.
    assert result["rms_distance"] == 1.
    assert len(result["skipped"]) == 2


def test_repeatability_phase_only_excludes_null_display_without_losing_native_zero():
    result = repeat_iq([cut([0]), cut([2j])], 9., scaling="phase_only")
    assert result["count"] == 2
    np.testing.assert_array_equal(result["valid"], [False, True])
    np.testing.assert_array_equal(result["phase_valid"], [False, True])
    assert result["native_field"][0] == 0j
    assert result["display_mean_field"] == 1j
    assert result["mean_field"] == 1j


def test_helpers_never_modify_input_arrays_or_notes():
    a, b = cut([1j, 0, 3j]), cut([2j, 1, 1j])
    snapshots = [(c.x.copy(), c.field.copy(), c.power.copy(), c.db.copy(), list(c.notes)) for c in (a, b)]
    iq_series(a, scaling="common_peak", region=(9, 10))
    compare_iq(a, b, scaling="phase_only")
    compensate_iq(a, phase_offset_deg=20, delay_ns=.1)
    repeat_iq([a, b], 10, scaling="common_peak")
    for source, snapshot in zip((a, b), snapshots):
        for actual, expected in zip((source.x, source.field, source.power, source.db), snapshot[:4]):
            np.testing.assert_array_equal(actual, expected)
        assert source.notes == snapshot[4]
