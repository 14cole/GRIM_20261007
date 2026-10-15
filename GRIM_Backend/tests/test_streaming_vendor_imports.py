"""Bounded vendor table parsing and shared memory planning contracts."""
from __future__ import annotations

import csv
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io import cst

SENTRI_HEADER = [
    "freq_MHz", "theta_deg", "phi_deg", "rcs_pp_dBsm", "efield_phase_pp_deg",
    "rcs_tt_dBsm", "efield_phase_tt_deg", "rcs_pt_dBsm", "efield_phase_pt_deg",
    "rcs_tp_dBsm", "efield_phase_tp_deg",
]
CST_FLAT = ["Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)", "Polarity",
            "Magnitude(dBsm)", "Phase(deg)", "IQ"]
CST_WIDE = ["Frequency(GHz)", "Theta(deg)", "Phi(deg)",
            "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)"]


def write_rows(tmp_path, rows, name="samples.csv"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    return path


@pytest.mark.parametrize("kind", ["sentri", "flat", "wide"])
def test_import_streams_and_planning_matches_the_loader_budget(tmp_path, kind):
    if kind == "sentri":
        rows = [SENTRI_HEADER, [1000, 90, 0, 0, 20, -10, -40, -20, 180, -30, 90],
                [2000, 80, 360, 0, 20, -10, -40, -20, 180, -30, 90]]
        loader = RcsGrid.read_SENTRi
    elif kind == "flat":
        rows = [CST_FLAT, [0, 0, 1, "VV", 0, 90, "1j"],
                [10, 10, 2, "HH", 0, 90, "1j"]]
        loader = RcsGrid.read_CST
    else:
        rows = [CST_WIDE, [1, 90, 0, 0, 90], [2, 80, 10, 0, 90]]
        loader = RcsGrid.read_CST
    path = write_rows(tmp_path, rows, "samples.cst_data" if kind == "flat" else "samples.csv")
    with mock.patch.object(cst, "_read_cst_delimited_rows", side_effect=AssertionError("whole table retained")):
        with mock.patch.object(cst.np, "full", side_effect=AssertionError("sample allocation during planning")):
            with mock.patch.object(cst.np, "zeros", side_effect=AssertionError("sample allocation during planning")):
                estimate = cst.estimate_vendor_text_import(path)
        grid = loader(path, max_output_bytes=estimate["estimated_peak_bytes"])
        assert tuple(estimate["shape"]) == grid.shape
        assert estimate["retained_bytes"] == grid.extra["dense_import_allocation_bytes"]
        assert estimate["estimated_peak_bytes"] == grid.extra["dense_import_peak_bytes"]
        with pytest.raises(MemoryError, match="exceeding"):
            loader(path, max_output_bytes=estimate["estimated_peak_bytes"] - 1)


def test_generic_linear_csv_is_not_misidentified_as_vendor(tmp_path):
    path = write_rows(tmp_path, [["azimuth_deg", "elevation_deg", "frequency_ghz",
                                  "polarization", "rcs_power", "rcs_phase"],
                                 [0, 0, 1, "VV", 1, 0]])
    assert cst.estimate_vendor_text_import(path) is None


def test_unrecognized_table_plan_is_none(tmp_path):
    path = write_rows(tmp_path, [["name", "value"], ["one", 1]])
    assert cst.estimate_vendor_text_import(path) is None


def test_iq_summary_counts_rows_once_including_equivalent_duplicates(tmp_path):
    path = write_rows(tmp_path, [CST_FLAT, [0, 0, 1, "VV", 0, 90, "1j"],
                                [0, 360, 1, "VV", 0, 90, "1j"],
                                [0, 10, 1, "VV", "", "", "1j"],
                                [0, 20, 1, "VV", 0, 90, "opaque"]], "summary.cst_data")
    grid = RcsGrid.read_CST(path)
    assert grid.extra["cst_iq_rows_validated"] == 2
    assert grid.extra["cst_iq_only_rows"] == 1
    assert grid.extra["cst_iq_unparsed_fallback_rows"] == 1


def test_source_duplicate_remains_detectable_after_preferred_seam_replacement(tmp_path):
    def row(phi, db):
        return [1000, 90, phi, db, 0, db, 0, db, 0, db, 0]
    path = write_rows(tmp_path, [SENTRI_HEADER, row(0, 0), row(360, 10), row(0, 20)])
    with pytest.raises(ValueError, match=r"line 4: conflicting duplicate SENTRi sample at source phi=0; first defined on line 2"):
        RcsGrid.read_SENTRi(path)


def test_aliases_beyond_seams_keep_exact_duplicate_evidence(tmp_path):
    def row(phi, db):
        return [1000, 90, phi, db, 0, db, 0, db, 0, db, 0]
    path = write_rows(tmp_path, [SENTRI_HEADER, row(720, 0), row(0, 0), row(360, 0), row(720, 10)])
    with pytest.raises(ValueError, match=r"line 5: conflicting duplicate SENTRi sample at source phi=720; first defined on line 2"):
        RcsGrid.read_SENTRi(path)


@pytest.mark.parametrize("magnitude,phase", [(0, 720), (-float("inf"), 45), (0, "")])
def test_exceptional_phase_stays_lossless(tmp_path, magnitude, phase):
    path = write_rows(tmp_path, [CST_FLAT, [0, 0, 1, "VV", magnitude, phase, ""]], "phase.cst_data")
    grid = RcsGrid.read_CST(path)
    power = float(grid.rcs_power[0, 0, 0, 0])
    assert power == (0.0 if np.isneginf(magnitude) else 1.0)
    result = float(grid.rcs_phase[0, 0, 0, 0])
    if phase == "":
        assert np.isnan(result)
    else:
        assert result == float(np.deg2rad(phase))


def test_source_change_between_passes_is_reported(tmp_path):
    path = write_rows(tmp_path, [CST_WIDE, [1, 90, 0, 0, 0]])
    rows = cst._DelimitedRows(path)
    assert len(list(rows)) == 2
    with path.open("a", encoding="utf-8") as stream:
        stream.write("2,90,0,0,0\n")
    with pytest.raises(ValueError, match="changed during import"):
        list(rows)


def test_materializing_compatibility_helper_still_returns_rows(tmp_path):
    path = write_rows(tmp_path, [["metadata"], CST_WIDE, [1, 90, 0, 0, 0]])
    rows = cst._read_cst_delimited_rows(path)
    assert isinstance(rows, list)
    assert rows[0] == ["metadata"]
    assert rows[-1] == ["1", "90", "0", "0", "0"]


LEGACY_HEADER = ["theta(deg)", "phi(deg)", "abs(rcs)(dbm^2)",
                 "abs(theta)(dbm^2)", "phase(theta)(deg)",
                 "abs(phi)(dbm^2)", "phase(phi)(deg)", "ax.ratio(db)"]


def test_legacy_plan_and_typed_duplicate_tracking_preserve_original_precision(tmp_path):
    # Float32 rounding is wider than the physical duplicate tolerance. Comparing
    # a later row to rounded output would incorrectly reject this exact repeat.
    row = [10, 20, 0, -13.123456789, 31.123456789, -17.246813579, 24.987654321, 0]
    path = write_rows(tmp_path, [LEGACY_HEADER, row, row], "f=1GHz_samples.txt")
    estimate = cst.estimate_vendor_text_import(path)
    assert estimate["source_format"] == "legacy theta/phi TXT"
    grid = RcsGrid.load_theta_phi_txt(path, max_output_bytes=estimate["estimated_peak_bytes"])
    np.testing.assert_array_equal(grid.power_slice()[0, 0, 0, :2],
                                  np.float32(10 ** (np.array([row[3], row[5]]) / 10)))
    np.testing.assert_array_equal(grid.phase_slice()[0, 0, 0, :2],
                                  np.float32(np.deg2rad([row[4], row[6]])))
    assert grid.power_slice().dtype == np.float32
    assert grid.extra["dense_import_peak_bytes"] == estimate["estimated_peak_bytes"]


def test_sentri_far_wrapped_coordinate_keeps_near_zero_tolerance(tmp_path):
    row = [1000, 90, 720.0 + 1e-10, 0, 0, 0, 0, 0, 0, 0, 0]
    path = write_rows(tmp_path, [SENTRI_HEADER, row])
    grid = RcsGrid.read_SENTRi(path)
    np.testing.assert_array_equal(grid.azimuths, [0.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("wrap", ["", "0_360", "-180_180"])
@pytest.mark.parametrize("mode", ["known", "unknown", "partial", "zero", "unwrapped"])
def test_bounded_constructor_matches_original_sample_semantics(dtype, wrap, mode):
    from GRIM_Backend.io.samples import construct_parsed_grid

    power = np.array([1, 2, 3, np.nan, 0], dtype=dtype).reshape(-1, 1, 1, 1)
    phase = np.array([-np.pi, -1e-15, np.pi, 0.5, 0], dtype=dtype).reshape(power.shape)
    if mode == "unknown":
        phase[:] = np.nan
    elif mode == "partial":
        phase[1] = np.nan
    elif mode == "zero":
        phase[-1] = 1
    elif mode == "unwrapped":
        phase[2] = 8 * np.pi
    units = {"phase_wrap": wrap} if wrap else {}
    axes = (np.arange(5), [0], [1], ["VV"])
    expected = RcsGrid(*axes, rcs_power=power.copy(), rcs_phase=phase.copy(), units=units)
    actual = construct_parsed_grid(RcsGrid, *axes, rcs_power=power.copy(),
                                   rcs_phase=phase.copy(), units=units)
    assert actual.rcs_domain == expected.rcs_domain
    assert actual._storage_kind == expected._storage_kind
    assert actual.units == expected.units
    np.testing.assert_array_equal(actual.power_slice(), expected.power_slice())
    np.testing.assert_array_equal(actual.phase_slice(), expected.phase_slice())


def test_flat_csv_missing_power_does_not_invent_a_phase(tmp_path):
    from GRIM_Backend.io.loaders import load_dataset

    path = write_rows(tmp_path, [["azimuth", "elevation", "frequency", "polarization",
                                  "magnitude_linear", "phase_deg", "phase_wrap"],
                                 [0, 0, 1, "VV", 1, 720, "0_360"],
                                 [1, 0, 1, "VV", "", 45, "0_360"],
                                 [2, 0, 1, "VV", 0, 90, "0_360"]])
    grid = load_dataset(str(path))
    np.testing.assert_array_equal(grid.power_slice().reshape(-1), [1, np.nan, 0])
    np.testing.assert_array_equal(grid.phase_slice().reshape(-1), [0, np.nan, np.pi / 2])
    assert grid.rcs_domain == "power_phase"


def test_flat_csv_estimate_does_not_apply_or_validate_load_budget(tmp_path, monkeypatch):
    from GRIM_Backend.io.csv import load_flat_csv

    monkeypatch.setenv("GRIM_MAX_CSV_GRID_GB", "invalid")
    path = write_rows(tmp_path, [["azimuth", "elevation", "frequency", "polarization",
                                  "magnitude_linear"], [0, 0, 1, "VV", 1]])
    estimate = load_flat_csv(path, None, None, _estimate_only=True)
    assert estimate["shape"] == (1, 1, 1, 1)
    assert estimate["estimated_peak_bytes"] > estimate["retained_bytes"]


def test_flat_csv_negative_zero_matches_canonical_power_cleaning(tmp_path):
    from GRIM_Backend.io.loaders import load_dataset

    path = write_rows(tmp_path, [["azimuth", "elevation", "frequency", "polarization",
                                  "magnitude_linear", "phase_deg"], [0, 0, 1, "VV", -0.0, 90]])
    grid = load_dataset(str(path))
    assert not np.signbit(grid.power_slice()).any()
    assert float(grid.phase_slice().item()) == np.pi / 2
