"""Independent physical and workflow regressions from the plotting audit."""
import json

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import decimate_axis, gate_geometry, regrid_axis, shift_dataset


def raw_grid(frequencies=(10.04, 10.16), *, unit="GHz", quantity="sigma_2d"):
    frequencies = np.asarray(frequencies, dtype=float)
    raw = np.asarray([1.000000001 + 0.2j, 2.000000003 - 0.4j])[:len(frequencies)].reshape(1, 1, -1, 1)
    scale = 1e9 if unit == "GHz" else 1e6
    if quantity == "sigma_2d":
        field = raw / (2 * np.sqrt(2 * np.pi * frequencies * scale / C0))[None, None, :, None]
    else:
        field = raw * np.sqrt(4 * np.pi)
    return RcsGrid([0.], [0.], frequencies, ["HH"],
                   rcs_power=(np.abs(field) ** 2).astype(np.float32),
                   rcs_phase=np.angle(field).astype(np.float32),
                   units={"azimuth": "deg", "elevation": "deg", "frequency": unit,
                          "rcs_linear_quantity": quantity,
                          "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm"},
                   extra={"rcs_amp_real": raw.real.copy(), "rcs_amp_imag": raw.imag.copy(),
                          "raw_complex_amplitude_preserved": True,
                          "solver_certification": "source-only"})


@pytest.mark.parametrize("unit,frequencies,decimals", [("GHz", (10.04, 10.16), 1), ("MHz", (10000.04, 10100.16), 1)])
def test_round_frequency_preserves_normalized_raw_response(unit, frequencies, decimals):
    source = raw_grid(frequencies, unit=unit)
    before = source.rcs.copy()
    result = source.round_frequencies(decimals)
    np.testing.assert_allclose(result.rcs, before, rtol=3e-15, atol=0)
    np.testing.assert_allclose(result.rcs_power, source.rcs_power, rtol=8 * np.finfo(float).eps)
    assert result.audit()["errors"] == []
    assert "rescaled" in result.extra["frequency_relabel_normalization"]
    assert "solver_certification" not in result.extra
    np.testing.assert_array_equal(source.frequencies, frequencies)


def test_inline_frequency_relabel_reorders_without_changing_physical_field():
    source = raw_grid()
    before = source.rcs.copy()
    result = source.edit_axis_value("frequency", 0, 10.3)
    np.testing.assert_allclose(result.rcs, before[:, :, [1, 0], :], rtol=3e-15, atol=0)
    np.testing.assert_allclose(result.rcs_power, source.rcs_power[:, :, [1, 0], :], rtol=8 * np.finfo(float).eps)
    assert result.audit()["errors"] == []
    assert "solver_certification" not in result.extra


def test_round_to_zero_keeps_permissive_relabel_and_reports_invalid_coordinate():
    source = raw_grid((0.04,))
    result = source.round_frequencies(1)
    assert result.frequencies.item() == 0
    np.testing.assert_allclose(result.rcs, source.rcs, rtol=3e-15)
    assert "rcs_amp_real" not in result.extra
    errors = result.audit()["errors"]
    assert errors
    assert not any(error["code"] == "raw_complex_power_mismatch" for error in errors)


@pytest.mark.parametrize("phase", [0., 360., 31.25])
@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_phase_shift_keeps_authoritative_complex_precision(phase, quantity):
    source = raw_grid(quantity=quantity)
    expected = source.rcs * np.exp(1j * np.deg2rad(phase))
    result = shift_dataset(source, phase_degrees=phase)
    np.testing.assert_allclose(result.rcs, expected, rtol=3e-15, atol=0)
    if phase == 0:
        np.testing.assert_array_equal(result.rcs, source.rcs)


def sweep_grid(count=26):
    return RcsGrid([0.], [0.], 8. + .1 * np.arange(count), ["HH"],
                   rcs=np.ones((1, 1, count, 1), dtype=complex),
                   units={"frequency": "GHz"})


def test_decimation_default_retains_samples_and_reports_nonuniform_tail():
    source = sweep_grid()
    result = decimate_axis(source, axis="frequency", factor=3, mode="coherent")
    np.testing.assert_allclose(np.diff(result.frequencies), [.3] * 7 + [.25])
    report = json.loads(result.extra["decimation_json"])
    assert report["output_uniform"] is False
    assert report["discarded_source_coordinate_count"] == 0
    assert report["partial_bin_sample_count"] == 2
    assert "nonuniform" in result.history


def test_explicit_drop_tail_produces_uniform_fft_compatible_sweep():
    source = sweep_grid()
    result = decimate_axis(source, axis="frequency", factor=3, mode="coherent", partial="drop")
    np.testing.assert_allclose(result.frequencies, [8.1, 8.4, 8.7, 9., 9.3, 9.6, 9.9, 10.2])
    np.testing.assert_allclose(result.rcs, 1.)
    assert gate_geometry(result)["resolution_m"] > 0
    report = json.loads(result.extra["decimation_json"])
    assert report["discarded_source_coordinate_count"] == 2
    assert report["output_uniform"] is True
    assert "explicitly discarded 2" in result.history
    assert len(source.frequencies) == 26


def test_decimation_drop_tail_with_no_full_bin_explains_resolution():
    with pytest.raises(ValueError, match="keep it or choose a smaller factor"):
        decimate_axis(sweep_grid(2), axis="frequency", factor=3, partial="drop")


def gap_grid():
    return RcsGrid([0., 1., 4., 5.], [0.], [10.], ["HH"],
                   rcs=np.asarray([1., 2., 5., 6.], dtype=complex).reshape(4, 1, 1, 1))


def test_regrid_permissive_default_and_optional_gap_mask_preserve_real_samples():
    source = gap_grid()
    targets = np.arange(0., 5.1, .5)
    permissive = regrid_axis(source, "azimuth", values=targets)
    limited = regrid_axis(source, "azimuth", values=targets, max_gap=1.1)
    np.testing.assert_allclose(permissive.rcs.ravel(), 1 + targets)
    excluded = (targets > 1) & (targets < 4)
    assert np.all(np.isnan(limited.rcs_power.ravel()[excluded]))
    np.testing.assert_allclose(limited.rcs.ravel()[~excluded], (1 + targets)[~excluded])
    report = json.loads(limited.extra["regrid_support_json"])
    assert report["measured_coordinate_count"] == 4
    assert report["interpolated_coordinate_count"] == 2
    assert report["excluded_coordinate_count"] == 5
    assert report["finite_power_sample_count"] == 6


def test_measured_only_policy_retains_missing_sample_and_masks_invented_coordinates():
    source = gap_grid()
    source.rcs_power[1, 0, 0, 0] = np.nan
    source.rcs_phase[1, 0, 0, 0] = np.nan
    result = regrid_axis(source, "azimuth", values=np.arange(6.), support="measured-only")
    np.testing.assert_array_equal(np.isfinite(result.rcs_power.ravel()), [True, False, False, False, True, True])
    report = json.loads(result.extra["regrid_support_json"])
    assert report["measured_coordinate_count"] == 4
    assert report["finite_power_sample_count"] == 3


def test_roundoff_after_unit_conversion_keeps_exact_samples_and_local_missingness():
    coordinates = np.deg2rad(np.arange(11) * .1)
    values = np.arange(11, dtype=float).reshape(11, 1, 1, 1) + 1
    values[6] = np.nan
    source = RcsGrid(coordinates, [0.], [10.], ["HH"],
                     rcs_power=values, rcs_phase=np.zeros_like(values),
                     units={"azimuth": "rad"})
    targets = np.deg2rad(.1) * np.arange(11)
    result = regrid_axis(source, "azimuth", values=targets, support="measured-only")
    np.testing.assert_allclose(result.rcs_power, values, rtol=1e-15, equal_nan=True)
    report = json.loads(result.extra["regrid_support_json"])
    assert report["measured_coordinate_count"] == 11
    assert report["excluded_coordinate_count"] == 0
    # Meaningful sub-bin offsets are not silently treated as measured samples.
    shifted = regrid_axis(source, "azimuth", values=[coordinates[7] + 1e-8], support="measured-only")
    assert np.isnan(shifted.rcs_power.item())


def test_decimate_dialog_shows_tail_choice_and_actual_sampling():
    from PySide6.QtWidgets import QApplication
    from GRIM_Backend.ui.dataset_dialogs import DecimateDialog
    app = QApplication.instance() or QApplication([])
    dialog = DecimateDialog(dataset_entries=[("Sweep", sweep_grid())])
    try:
        dialog._axis.setCurrentIndex(2)
        dialog._factor.setValue(3)
        assert dialog.get_params()["partial"] == "keep"
        assert "NONUNIFORM" in dialog._preview.text()
        dialog._partial.setCurrentIndex(1)
        assert "8 output coordinates, uniform" in dialog._preview.text()
        assert "2 trailing coordinates discarded" in dialog._preview.text()
    finally:
        dialog.deleteLater()
        app.processEvents()


def test_regrid_dialog_default_and_optional_coverage_preview():
    from PySide6.QtWidgets import QApplication
    from GRIM_Backend.ui.dataset_dialogs import RegridDialog
    app = QApplication.instance() or QApplication([])
    dialog = RegridDialog(gap_grid())
    try:
        assert dialog.get_params()["max_gap"] is None
        assert dialog.get_params()["support"] == "interpolate"
        dialog._spin_step.setValue(.5)
        dialog._max_gap.setValue(1.1)
        dialog._limit_gap.setChecked(True)
        assert "4 measured, 2 interpolated, 5 policy-masked coordinates" in dialog._summary.text()
        assert "missing response samples remain missing" in dialog._summary.text()
    finally:
        dialog.deleteLater()
        app.processEvents()
