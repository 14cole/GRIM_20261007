"""Azimuth wrapping moves coordinates without rounding complex responses."""

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.operations import operation_events


def _grid(azimuths, field, *, unit="deg", extra=None, **kwargs):
    field = np.asarray(field)
    field = field.reshape(len(azimuths), 1, -1, 1)
    return RcsGrid(
        azimuths, [0.0], np.arange(field.shape[2], dtype=float) + 1.0,
        ["HH"], rcs=field,
        units={"azimuth": unit, "elevation": "deg", "frequency": "GHz",
               "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"},
        extra=extra, **kwargs,
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("unit", ["deg", "rad"])
@pytest.mark.parametrize("mode", ["0_360", "-180_180"])
def test_wrap_preserves_exact_quadratures_and_native_roundtrip(tmp_path, dtype, unit, mode):
    azimuths = np.array([-170.0, -10.0, 40.0, 200.0])
    if unit == "rad":
        azimuths = np.deg2rad(azimuths)
    field = np.array([
        -1 + 1e-6j, -1 - 1e-18j, 1e-18 + 2j, 2 - 1e-18j,
    ], dtype=dtype)
    source = _grid(azimuths, field, unit=unit, history="original")
    original = source.rcs.copy()
    original_power = np.asarray(source.power).copy()
    original_phase = np.asarray(source.phase).copy()
    period = 2 * np.pi if unit == "rad" else 360.0
    offset = 0.0 if mode == "0_360" else period / 2
    expected_angles = (azimuths + offset) % period - offset
    order = np.argsort(expected_angles, kind="stable")
    expected = original[order]

    result, report = source.wrap_azimuth(mode, return_report=True)

    np.testing.assert_array_equal(result.azimuths, expected_angles[order])
    assert result.storage_kind == "complex"
    assert result.rcs.dtype == np.dtype(dtype)
    assert result.rcs.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(result.power, original_power[order])
    np.testing.assert_array_equal(result.phase, original_phase[order])
    assert report["merged_coordinate_count"] == 0
    assert report["discarded_conflict_cell_count"] == 0
    assert source.history == "original"
    assert result.history.startswith(source.history + "\n")
    events = operation_events(result)
    assert len(events) == 1
    assert events[0]["operation"] == "wrap_azimuth"
    assert events[0]["parameters"] == {
        "mode": mode, "seam_conflict": "first", "return_report": True,
    }
    assert not operation_events(source)
    assert source.storage_kind == "complex"
    assert source.rcs.tobytes() == original.tobytes()
    np.testing.assert_array_equal(source.azimuths, azimuths)

    # Cancellation is the sensitive end-to-end use case: a coordinate edit
    # must not change the tiny residual hidden by a large real component.
    reference = _grid(result.azimuths, -expected.real.astype(dtype), unit=unit)
    residual = result.coherent_add(reference)
    np.testing.assert_array_equal(residual.rcs, expected - expected.real)
    restored = RcsGrid.load(result.save(tmp_path / "wrapped"))
    assert restored.rcs.dtype == np.dtype(dtype)
    assert restored.rcs.tobytes() == expected.tobytes()
    assert restored.history == result.history
    assert operation_events(restored) == events


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_wrap_retains_iq_after_reading_legacy_arrays(dtype):
    source = _grid([-10.0, 10.0], np.array([-1 + 1e-6j, -2 + 1e-18j], dtype=dtype))
    original = source.rcs.copy()
    # Compatibility clients may inspect these arrays without modifying them.
    np.asarray(source.rcs_power)
    np.asarray(source.rcs_phase)
    result = source.wrap_azimuth("0_360")
    assert result.rcs.tobytes() == original[::-1].tobytes()
    assert source.rcs.tobytes() == original.tobytes()


def test_wrap_honors_edits_to_legacy_arrays():
    source = _grid([-10.0, 10.0], [-1 + 1e-6j, 2 + 1e-18j])
    source.rcs_power[0] = 9.0
    source.rcs_phase[0] = np.nan
    result = source.wrap_azimuth("0_360")
    np.testing.assert_array_equal(result.power[:, 0, 0, 0], [4.0, 9.0])
    assert np.isnan(result.phase[1]).all()
    assert np.isnan(result.rcs[1]).all()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_wrap_keeps_signed_zero_and_missing_cells(dtype):
    field = np.array([complex(-0.0, -0.0), complex(np.nan, np.nan), -1 + 1e-6j], dtype=dtype)
    source = _grid([-10.0, 0.0, 10.0], field)
    result = source.wrap_azimuth("0_360")
    assert result.rcs.tobytes() == source.rcs[[1, 2, 0]].tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("mode", ["0_360", "-180_180"])
def test_wrap_complex_seam_keeps_first_and_fills_gaps_exactly(dtype, mode):
    missing = complex(np.nan, np.nan)
    field = np.array([
        [-1 + 1e-6j, missing, -3 + 1e-18j],
        [-2 + 1e-18j, 4 - 1e-18j, 1e-18 + 2j],
        [-1.1 + 0.5j, -2 - 1e-18j, -3 + 1e-18j],
    ], dtype=dtype)
    source = _grid([0.0, 180.0, 360.0], field, history="original")
    original = source.rcs.copy()
    expected = original[:2].copy()
    expected[0, 0, 1, 0] = original[2, 0, 1, 0]
    if mode == "-180_180":
        expected = expected[::-1]

    result, report = source.wrap_azimuth(mode, return_report=True)

    assert result.rcs.dtype == np.dtype(dtype)
    assert result.rcs.tobytes() == expected.tobytes()
    assert report["merged_coordinate_count"] == 1
    assert report["conflicting_coordinate_count"] == 1
    assert report["discarded_conflict_cell_count"] == 1
    assert "discarded 1 conflicting cell(s)" in result.history
    assert source.rcs.tobytes() == original.tobytes()
    with pytest.raises(ValueError, match="conflicting finite seam"):
        source.wrap_azimuth(mode, seam_conflict="error")
    assert source.rcs.tobytes() == original.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_wrap_preserves_existing_equivalent_seam_tolerance(dtype):
    # A slightly different repeated measurement remains acceptable under the
    # existing power/phase tolerances; retain the opening I/Q verbatim.
    field = np.array([-1 + 1e-6j, -1 + 1.1e-6j], dtype=dtype)
    source = _grid([0.0, 360.0], field)
    result, report = source.wrap_azimuth("0_360", seam_conflict="error", return_report=True)
    assert result.rcs.tobytes() == source.rcs[:1].tobytes()
    assert report["conflicting_coordinate_count"] == 0


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_wrap_keeps_raw_solver_authority_and_metadata_policy(tmp_path, quantity):
    field = np.array([-1 + 1e-18j, -2 - 1e-18j, -1 + 1e-18j], dtype=np.complex128).reshape(3, 1, 1, 1)
    source = RcsGrid(
        [-180.0, 0.0, 180.0], [0.0], [2.0], ["HH"],
        rcs=field, rcs_storage_domain="raw_field",
        units={"azimuth": "deg", "frequency": "GHz", "rcs_linear_quantity": quantity,
               "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm"},
        extra={"amplitude_convention": "solver field", "producer_array": np.ones(field.shape)},
    )
    expected_physical = source.rcs[[1, 0]].copy()
    result = source.wrap_azimuth("0_360")
    assert result.storage_domain == "raw_field"
    np.testing.assert_array_equal(result.native_sample_payload()["rcs_complex"], field[[1, 0]])
    np.testing.assert_array_equal(result.rcs, expected_physical)
    assert "producer_array" not in result.extra
    assert result.extra["amplitude_convention"] == "solver field"
    restored = RcsGrid.load(result.save(tmp_path / "raw"))
    np.testing.assert_array_equal(restored.native_sample_payload()["rcs_complex"], field[[1, 0]])
    np.testing.assert_array_equal(source.native_sample_payload()["rcs_complex"], field)


@pytest.mark.parametrize("phase", [None, [0.5, np.nan, 4 * np.pi, 1.0]])
def test_wrap_preserves_unknown_and_exceptional_phase(phase):
    power = np.array([4.0, 9.0, 16.0, 0.0]).reshape(4, 1, 1, 1)
    phase_array = None if phase is None else np.array(phase).reshape(power.shape)
    source = RcsGrid([-30.0, -20.0, 10.0, 20.0], [0.0], [1.0], ["HH"],
                     rcs_power=power, rcs_phase=phase_array)
    result = source.wrap_azimuth("0_360")
    order = [2, 3, 0, 1]
    np.testing.assert_array_equal(result.power, power[order])
    np.testing.assert_array_equal(result.phase, np.asarray(source.phase)[order])
    assert result.storage_kind == source.storage_kind
    np.testing.assert_array_equal(source.power, power)


def test_wrap_seam_fills_known_phase_without_discarding_unknown_power():
    power = np.array([[4.0, 9.0], [4.0, 9.0]]).reshape(2, 1, 2, 1)
    phase = np.array([[np.nan, np.nan], [0.5, np.nan]]).reshape(power.shape)
    source = RcsGrid([0.0, 360.0], [0.0], [1.0, 2.0], ["HH"],
                     rcs_power=power, rcs_phase=phase)
    result = source.wrap_azimuth("0_360", seam_conflict="error")
    np.testing.assert_array_equal(result.power, power[:1])
    np.testing.assert_array_equal(result.phase, phase[1:])
    assert np.isnan(result.rcs[0, 0, 1, 0])
    np.testing.assert_array_equal(source.phase, phase)
