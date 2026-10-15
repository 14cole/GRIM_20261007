"""Physical median-window membership is independent of angular storage units."""

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.operations import operation_events
from GRIM_Backend.datasets.transforms import medianize_azimuth


def _grid(azimuth, power, unit="deg", *, phase=None):
    azimuth = np.asarray(azimuth, dtype=float)
    if unit == "rad":
        azimuth = np.deg2rad(azimuth)
    shape = (azimuth.size, 1, 1, 1)
    return RcsGrid(
        azimuth, [0.0], [10.0], ["HH"],
        rcs_power=np.asarray(power, dtype=float).reshape(shape),
        rcs_phase=None if phase is None else np.asarray(phase).reshape(shape),
        history="source history",
        units={"azimuth": unit, "frequency": "GHz", "rcs_linear_quantity": "sigma_3d"},
        extra={"time_convention": "exp(+jwt)", "polarization_basis": "H/V"},
    )


@pytest.mark.parametrize("unit", ["deg", "rad"])
@pytest.mark.parametrize("periodic", [False, True])
def test_exact_window_edges_preserve_known_median_and_source(unit, periodic):
    azimuth = np.arange(0.0, 361.0, 10.0)
    power = 1.0 + np.arange(azimuth.size) ** 2
    power[-1] = power[0]
    source = _grid(azimuth, power, unit, phase=np.full(azimuth.size, np.nan))
    original_axis = source.azimuths.copy()
    original_power = np.asarray(source.power).copy()
    original_phase = np.asarray(source.phase).copy()

    result, report = medianize_azimuth(
        source, window_degrees=40.0, slide_degrees=20.0,
        periodic=periodic, return_report=True,
    )

    centers = np.rad2deg(result.azimuths) if unit == "rad" else result.azimuths
    at_80 = int(np.flatnonzero(np.isclose(centers, 80.0, rtol=0, atol=1e-12))[0])
    assert result.power[at_80, 0, 0, 0] == 65.0  # Both 60- and 100-degree edges.
    assert report == {
        "schema": "grim.medianize-report.v1", "periodic": periodic,
        "seam_merged": periodic, "seam_conflict": "first",
        "discarded_seam_cell_count": 0,
    }
    assert np.isnan(result.phase).all()
    assert source.history == "source history"
    assert result.history.startswith(source.history + "\n")
    events = operation_events(result)
    assert len(events) == 1
    assert events[0]["operation"] == "medianize_azimuth"
    assert events[0]["parameters"] == {
        "window_degrees": 40.0, "slide_degrees": 20.0, "periodic": periodic,
        "seam_conflict": "first", "return_report": True,
    }
    assert not operation_events(source)
    assert result.units == source.units
    assert result.extra["time_convention"] == source.extra["time_convention"]
    assert result.extra["polarization_basis"] == source.extra["polarization_basis"]
    np.testing.assert_array_equal(source.azimuths, original_axis)
    np.testing.assert_array_equal(source.power, original_power)
    np.testing.assert_array_equal(source.phase, original_phase)


@pytest.mark.parametrize("step,window,slide", [(10.0, 40.0, 20.0), (1.0, 10.0, 5.0), (0.1, 10.0, 5.0)])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("origin", [0.0, -180.0, 37.3])
def test_smooth_response_matches_between_units_and_integer_window_oracle(step, window, slide, periodic, origin):
    sample_number = np.arange(round(360 / step) + 1)
    azimuth = origin + sample_number * step
    power = 1.0 + 15.0 * (sample_number * step / 360.0) ** 2
    power[-1] = power[0]
    source = _grid(azimuth, power)
    radians = source.convert_axis_units(azimuth="rad")
    kwargs = dict(window_degrees=window, slide_degrees=slide, periodic=periodic)
    degrees_result = medianize_azimuth(source, **kwargs)
    radians_result = medianize_azimuth(radians, **kwargs)

    # Integer sample distances give the exact population independently of
    # floating-point degree/radian arithmetic in the implementation.
    source_indices = sample_number[:-1] if periodic else sample_number
    half_samples = round(window / (2 * step))
    center_indices = np.arange(degrees_result.shape[0]) * round(slide / step)
    if not periodic:
        center_indices += half_samples
    reference = []
    for center in center_indices:
        distance = np.abs(source_indices - center)
        if periodic:
            distance = np.minimum(distance, round(360 / step) - distance)
        reference.append(np.median(power[source_indices[distance <= half_samples]]))
    np.testing.assert_array_equal(np.asarray(degrees_result.power).ravel(), reference)
    np.testing.assert_array_equal(radians_result.power, degrees_result.power)
    np.testing.assert_allclose(np.rad2deg(radians_result.azimuths), degrees_result.azimuths, rtol=0, atol=2e-13)


@pytest.mark.parametrize("unit", ["deg", "rad"])
@pytest.mark.parametrize("periodic", [False, True])
def test_nonuniform_axis_includes_both_edges_but_excludes_nearby_outside_samples(unit, periodic):
    # 2e-12 degrees is tiny physically but still distinctly outside float64
    # angular conversion/arithmetic roundoff at these coordinates.
    azimuth = [0.0, 11.0, 29.0, 60.0 - 2e-12, 60.0, 67.0, 80.0, 94.0, 100.0, 100.0 + 2e-12, 140.0]
    # Both edge values are above the interior median, so losing either edge
    # (or both) changes the answer; large outside values expose overinclusion.
    power = [1, 2, 3, 500, 100, 10, 20, 30, 200, 600, 100]
    result = medianize_azimuth(
        _grid(azimuth, power, unit), window_degrees=40.0,
        slide_degrees=20.0, periodic=periodic,
    )
    centers = np.rad2deg(result.azimuths) if unit == "rad" else result.azimuths
    at_80 = np.flatnonzero(np.isclose(centers, 80.0, rtol=0, atol=1e-12))[0]
    assert result.power[at_80, 0, 0, 0] == 30.0


@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_periodic_seam_counts_each_direction_once_and_excludes_outside_samples(unit):
    azimuth = [0.0, 1.0, 1.0 + 2e-12, 89.0, 180.0, 270.0, 359.0 - 2e-12, 359.0, 360.0]
    power = [10, 30, 1000, 50, 60, 70, 2000, 20, 10]
    result, report = medianize_azimuth(
        _grid(azimuth, power, unit), window_degrees=2.0,
        slide_degrees=90.0, periodic=True, return_report=True,
    )
    assert result.power[0, 0, 0, 0] == 20.0
    assert report["seam_merged"]
    assert report["discarded_seam_cell_count"] == 0


@pytest.mark.parametrize("unit", ["deg", "rad"])
@pytest.mark.parametrize("periodic", [False, True])
def test_tiny_windows_near_zero_do_not_inherit_fixed_degree_tolerance(unit, periodic):
    azimuth = np.array([0.0, 0.25, 0.5, 1.0, 1.01, 1.5, 2.0]) * 1e-12
    power = [10, 20, 30, 40, 1000, 2000, 3000]
    # A 2e-12-wide periodic window is centered on zero; the nonperiodic
    # 1e-12-wide window is centered on 0.5e-12. Both include indices 0..3.
    result = medianize_azimuth(
        _grid(azimuth, power, unit), window_degrees=2e-12 if periodic else 1e-12,
        slide_degrees=360.0 if periodic else 1e-12, periodic=periodic,
    )
    assert result.power[0, 0, 0, 0] == 25.0


@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_narrow_window_at_nonzero_coordinate_preserves_membership(unit):
    azimuth = np.array([30.0, 30.0 + 1e-8, 30.0 + 2e-8, 30.0 + 3e-8])
    result = medianize_azimuth(
        _grid(azimuth, [10, 20, 30, 40], unit),
        window_degrees=2e-8, slide_degrees=1e-8, periodic=False,
    )
    np.testing.assert_array_equal(np.asarray(result.power).ravel(), [20, 30])


@pytest.mark.parametrize("unit", ["deg", "rad"])
def test_final_center_really_outside_the_span_is_not_added(unit):
    result = medianize_azimuth(
        _grid([0.0, 20.0, 40.0, 60.0, 80.0, 100.0 - 2e-12], [1, 2, 3, 4, 5, 6], unit),
        window_degrees=40.0, slide_degrees=20.0, periodic=False,
    )
    centers = np.rad2deg(result.azimuths) if unit == "rad" else result.azimuths
    np.testing.assert_allclose(centers, [20.0, 40.0, 60.0], rtol=0, atol=2e-14)


@pytest.mark.parametrize("unit", ["deg", "rad"])
@pytest.mark.parametrize("slide", [1e-16, 1e-14, 1.0])
def test_a_single_fitting_center_is_not_duplicated_by_endpoint_roundoff(unit, slide):
    result = medianize_azimuth(
        _grid([30.0, 40.0], [10.0, 20.0], unit),
        window_degrees=10.0, slide_degrees=slide, periodic=False,
    )
    centers = np.rad2deg(result.azimuths) if unit == "rad" else result.azimuths
    np.testing.assert_allclose(centers, [35.0], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(np.asarray(result.power).ravel(), [15.0])
