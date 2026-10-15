"""GHOST writes GRIM format version 2 by default and reads both versions."""
import json
import math
import os
import sys
import tracemalloc
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPOSITORY = ROOT.parents[1]
if (REPOSITORY / "GRIM_Backend").is_dir() and str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from ghost_backend.io import grim
from ghost_backend.io.grim import C0, export_result_to_grim, load_grim_npz
from ghost_backend.data_tools.cem_tools.native_samples import decode_native_samples
from ghost_backend.execution.recovery_export import GRID_KEYS

VERSION_ONE_ARRAYS = {"rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"}
VERSION_TWO_KEYS = {"grim_format_version", "storage_kind", "storage_domain", "rcs_complex"}
# Key order and dtype kinds of a version-1 monostatic export with a raw pair.
VERSION_ONE_LAYOUT = [
    ("azimuths", "f"), ("elevations", "f"), ("frequencies", "f"), ("polarizations", "U"),
    ("polarization_alias_primary", "U"), ("polarization_aliases_json", "U"),
    ("rcs_power", "f"), ("rcs_phase", "f"), ("rcs_domain", "U"), ("power_domain", "U"),
    ("source_path", "U"), ("history", "U"), ("units", "U"), ("phase_reference", "U"),
    ("amplitude_convention", "U"), ("raw_complex_amplitude_preserved", "b"),
    ("rcs_amp_real", "f"), ("rcs_amp_imag", "f"), ("complex_field_domain", "U"),
    ("solver_metadata_json", "U"),
]
FLOAT32_EPS = np.finfo(np.float32).eps


def solver_result(quantity="sigma_2d", *, bistatic=False, seed=3):
    """A small, physically consistent monostatic or bistatic solver result."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(-30., 30., 7)
    incidences = (0., 20.) if bistatic else (None,)
    rows = []
    for frequency in (1., 1.5, 2.):
        k0 = 2. * math.pi * frequency * 1.e9 / C0
        for incidence in incidences:
            for angle in angles:
                amplitude = complex(*rng.standard_normal(2))
                linear = (abs(amplitude) ** 2 / (4. * k0) if quantity == "sigma_2d"
                          else 4. * math.pi * abs(amplitude) ** 2)
                rows.append(dict(
                    frequency_ghz=frequency, theta_scat_deg=float(angle),
                    theta_inc_deg=float(angle) if incidence is None else incidence,
                    rcs_linear=linear, rcs_db=10. * math.log10(linear),
                    rcs_amp_real=amplitude.real, rcs_amp_imag=amplitude.imag,
                    linear_residual=1.e-12,
                ))
    return dict(
        solver="2d_bie_mom_rcs" if quantity == "sigma_2d" else "bor_mom_rcs",
        scattering_mode="bistatic" if bistatic else "monostatic", polarization="TM",
        rcs_linear_quantity=quantity, rcs_log_unit="dBke" if quantity == "sigma_2d" else "dBsm",
        samples=rows, metadata={"note": "format version test"},
    )


def raw_field_payloads(shape, quantity, *, seed=5, zero_cells=True):
    """Return the version-1 and version-2 payloads for one random raw field."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    field *= 10. ** rng.uniform(-3., 3., shape)
    if zero_cells:
        field[0, 0, 0, 0] = 0.
        field[-1, -1, -1, -1] = complex(-0., 0.)
    frequencies = np.linspace(1., 2., shape[2])
    if quantity == "sigma_2d":
        scale_sq = (1. / (4. * 2. * math.pi * frequencies * 1.e9 / C0))[None, None, :, None]
    else:
        scale_sq = 4. * math.pi
    metadata = dict(
        azimuths=np.linspace(-90., 90., shape[0]), elevations=np.linspace(-10., 10., shape[1]),
        frequencies=frequencies, polarizations=np.asarray(["VV", "HH", "VH", "HV"][:shape[3]]),
        units=json.dumps({"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                          "rcs_linear_quantity": quantity}),
        rcs_domain="power_phase", power_domain="linear_rcs",
        raw_complex_amplitude_preserved=np.asarray(True),
    )
    phase = np.arctan2(field.imag, field.real)
    phase[(field.real == 0.) & (field.imag == 0.)] = 0.
    legacy = dict(metadata, rcs_power=((field.real ** 2 + field.imag ** 2) * scale_sq).astype(np.float32),
                  rcs_phase=phase.astype(np.float32), rcs_amp_real=field.real.copy(),
                  rcs_amp_imag=field.imag.copy())
    compact = dict(metadata, grim_format_version=np.asarray(2, dtype=np.int16),
                   storage_kind="complex", storage_domain="raw_field", rcs_complex=field)
    return legacy, compact, field


def assert_within_version_one_tolerances(decoded, legacy):
    """Power within 16 float32 eps and phase within 2e-5 rad, as the readers check."""
    power, expected = decoded["rcs_power"].astype(np.float64), legacy["rcs_power"].astype(np.float64)
    tolerance = 16. * FLOAT32_EPS * np.maximum(power, expected) + np.finfo(np.float32).tiny
    assert np.all(np.abs(power - expected) <= tolerance)
    difference = decoded["rcs_phase"].astype(np.float64) - legacy["rcs_phase"].astype(np.float64)
    wrapped = np.arctan2(np.sin(difference), np.cos(difference))
    assert float(np.max(np.abs(wrapped))) <= 2.e-5
    np.testing.assert_array_equal(decoded["rcs_amp_real"], legacy["rcs_amp_real"])
    np.testing.assert_array_equal(decoded["rcs_amp_imag"], legacy["rcs_amp_imag"])


@pytest.fixture
def unset_format_environment():
    with mock.patch.dict(os.environ):
        os.environ.pop(grim.GRIM_FORMAT_VERSION_ENV, None)
        yield


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_default_and_version_one_exports_load_identically_in_grim(tmp_path, unset_format_environment, quantity):
    RcsGrid = pytest.importorskip("GRIM_Backend.datasets.api").RcsGrid
    result = solver_result(quantity)
    [legacy] = export_result_to_grim(result, str(tmp_path / "legacy"), format_version=1)
    [compact] = export_result_to_grim(result, str(tmp_path / "compact"))
    with np.load(legacy, allow_pickle=False) as one, np.load(compact, allow_pickle=False) as two:
        assert set(one.files) - set(two.files) == VERSION_ONE_ARRAYS
        assert set(two.files) - set(one.files) == VERSION_TWO_KEYS
        assert two["grim_format_version"].dtype == np.int16 and int(two["grim_format_version"]) == 2
        assert str(two["storage_kind"]) == "complex" and str(two["storage_domain"]) == "raw_field"
        assert two["rcs_complex"].dtype == np.complex128
        np.testing.assert_array_equal(two["rcs_complex"].real, one["rcs_amp_real"])
        np.testing.assert_array_equal(two["rcs_complex"].imag, one["rcs_amp_imag"])
        for key in set(one.files) & set(two.files):
            np.testing.assert_array_equal(one[key], two[key], err_msg=key)
        stored_power, stored_phase = one["rcs_power"], one["rcs_phase"]
    assert os.path.getsize(compact) < os.path.getsize(legacy)

    first, second = RcsGrid.load(legacy), RcsGrid.load(compact)
    for axis in ("azimuths", "elevations", "frequencies", "polarizations"):
        np.testing.assert_array_equal(getattr(second, axis), getattr(first, axis))
    np.testing.assert_array_equal(second.extra["rcs_amp_real"], first.extra["rcs_amp_real"])
    np.testing.assert_array_equal(second.extra["rcs_amp_imag"], first.extra["rcs_amp_imag"])
    assert second.linear_quantity() == first.linear_quantity() == quantity
    np.testing.assert_allclose(second.rcs_power, first.rcs_power, rtol=1.e-12)
    np.testing.assert_allclose(second.rcs_phase, first.rcs_phase, rtol=0., atol=1.e-12)
    np.testing.assert_allclose(second.rcs_power, stored_power, rtol=2. * FLOAT32_EPS)
    np.testing.assert_allclose(second.rcs_phase, stored_phase, rtol=0., atol=4. * FLOAT32_EPS)


def test_version_one_layout_keys_dtypes_and_values_are_unchanged(tmp_path):
    result = solver_result()
    payload = grim._build_grid_for_samples(result["samples"], "TM", rcs_log_unit="dBke",
                                           rcs_linear_quantity="sigma_2d")
    payload["solver_metadata_json"] = grim._solver_metadata_json(result)
    saved = grim._save_grim_npz(payload, str(tmp_path / "legacy"), format_version=1)
    with np.load(saved, allow_pickle=False) as data:
        assert [(key, data[key].dtype.kind) for key in data.files] == VERSION_ONE_LAYOUT
        assert data["rcs_power"].dtype == data["rcs_phase"].dtype == np.float32
        assert data["rcs_amp_real"].dtype == data["rcs_amp_imag"].dtype == np.float64
        for key in data.files:
            np.testing.assert_array_equal(data[key], np.asarray(payload[key]), err_msg=key)
        assert bool(data["raw_complex_amplitude_preserved"]) is True


def test_payload_without_raw_pair_keeps_the_version_one_layout(tmp_path, unset_format_environment):
    [saved] = export_result_to_grim(solver_result(), str(tmp_path / "power"),
                                    preserve_raw_complex_amplitude=False)
    with np.load(saved, allow_pickle=False) as data:
        assert not VERSION_TWO_KEYS & set(data.files)
        assert {"rcs_power", "rcs_phase"} <= set(data.files)
        assert not {"rcs_amp_real", "rcs_amp_imag"} & set(data.files)
        assert bool(data["raw_complex_amplitude_preserved"]) is False


@pytest.mark.parametrize("setting, expected", [("1", 1), ("2", 2), ("", 2), (" 1 ", 1), ("3", 2)])
def test_environment_selects_the_default_layout(tmp_path, setting, expected):
    with mock.patch.dict(os.environ, {grim.GRIM_FORMAT_VERSION_ENV: setting}):
        assert grim._resolve_grim_format_version(None) == expected
        [saved] = export_result_to_grim(solver_result(), str(tmp_path / "exported"))
        with np.load(saved, allow_pickle=False) as data:
            assert ("rcs_complex" in data.files) == (expected == 2)
        # An explicit version always wins over the environment.
        [explicit] = export_result_to_grim(solver_result(), str(tmp_path / "explicit"),
                                           format_version=3 - expected)
        with np.load(explicit, allow_pickle=False) as data:
            assert ("rcs_complex" in data.files) == (expected == 1)


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_unsupported_explicit_versions_are_rejected(tmp_path, version):
    with pytest.raises(ValueError, match="format_version"):
        export_result_to_grim(solver_result(), str(tmp_path / "bad"), format_version=version)
    assert not list(tmp_path.iterdir())


def test_bistatic_batches_and_validation_follow_the_selected_layout(tmp_path, unset_format_environment):
    result = solver_result(bistatic=True)
    written = export_result_to_grim(result, str(tmp_path / "bistatic"))
    assert len(written) == 2
    for path in written:
        with np.load(path, allow_pickle=False) as data:
            assert VERSION_TWO_KEYS <= set(data.files) and not VERSION_ONE_ARRAYS & set(data.files)
    legacy = export_result_to_grim(result, str(tmp_path / "legacy"), format_version=1)
    for path in legacy:
        with np.load(path, allow_pickle=False) as data:
            assert VERSION_ONE_ARRAYS <= set(data.files) and not VERSION_TWO_KEYS & set(data.files)
    # The payload is still validated against its amplitude before writing.
    result["samples"][0]["rcs_linear"] *= 1.01
    with pytest.raises(ValueError, match="inconsistent"):
        export_result_to_grim(result, str(tmp_path / "rejected"))
    assert not list(tmp_path.glob("rejected*"))


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_decoded_version_two_matches_version_one_within_reader_tolerances(tmp_path, unset_format_environment, quantity):
    result = solver_result(quantity)
    [legacy] = export_result_to_grim(result, str(tmp_path / "legacy"), format_version=1)
    [compact] = export_result_to_grim(result, str(tmp_path / "compact"))
    with load_grim_npz(legacy) as one, load_grim_npz(compact) as two:
        assert set(one.files) == set(two.files)
        assert two["rcs_power"].dtype == two["rcs_phase"].dtype == np.float32
        assert_within_version_one_tolerances(two, one)
        for key in set(one.files) - VERSION_ONE_ARRAYS:
            np.testing.assert_array_equal(one[key], two[key], err_msg=key)


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_decoder_expands_raw_field_without_a_second_complex_array(quantity):
    legacy, compact, field = raw_field_payloads((2000, 1, 250, 4), quantity)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        decoded = decode_native_samples(dict(compact))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Float32 power and phase (8 bytes per cell) plus bounded block scratch;
    # a second full-grid complex128 copy alone would reach field.nbytes.
    assert peak < field.nbytes, (peak, field.nbytes)
    assert decoded["rcs_power"].dtype == decoded["rcs_phase"].dtype == np.float32
    # Compatibility components are lazy views; explicit ndarray conversion
    # intentionally detaches a writable legacy component.
    assert decoded["rcs_amp_real"].field is field
    assert decoded["rcs_amp_imag"].field is field
    assert_within_version_one_tolerances(decoded, legacy)
    assert decoded["rcs_phase"][0, 0, 0, 0] == 0. and decoded["rcs_phase"][-1, -1, -1, -1] == 0.
    assert not VERSION_TWO_KEYS & set(decoded)
    assert bool(decoded["raw_complex_amplitude_preserved"])


def test_decoder_keeps_the_physical_domain_branch_in_float64():
    _legacy, compact, field = raw_field_payloads((6, 2, 3, 2), "sigma_3d", zero_cells=False)
    compact["storage_domain"] = "physical"
    decoded = decode_native_samples(dict(compact))
    assert decoded["rcs_power"].dtype == decoded["rcs_phase"].dtype == np.float64
    np.testing.assert_array_equal(decoded["rcs_power"], np.abs(field) ** 2)
    np.testing.assert_array_equal(decoded["rcs_phase"], np.angle(field))
    np.testing.assert_allclose(decoded["rcs_amp_real"] + 1j * decoded["rcs_amp_imag"],
                               field / math.sqrt(4. * math.pi), rtol=1.e-15)


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_decoded_power_passes_assembly_consistency_checks(tmp_path, unset_format_environment, quantity):
    from ghost_backend.assembly.fields import _load_grim, _load_grim_sample
    [compact] = export_result_to_grim(solver_result(quantity), str(tmp_path / "compact"))
    whole = _load_grim(compact)
    single = _load_grim_sample(compact, frequency=1.5, azimuth=0., elevation=0.)
    with load_grim_npz(compact) as data:
        np.testing.assert_array_equal(whole["_amp"], data["rcs_amp_real"] + 1j * data["rcs_amp_imag"])
    np.testing.assert_array_equal(single["_amp"], whole["_amp"][3:4, :, 1:2, :])


def test_recovery_grid_keys_include_the_complex_field():
    assert "rcs_complex" in GRID_KEYS
