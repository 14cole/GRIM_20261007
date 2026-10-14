"""Native compatibility, sample authority, and compact-storage regressions."""
from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.memory import _preflight_native_archive_allocation


def axes(n=3):
    return dict(azimuths=np.arange(n, dtype=float), elevations=np.array([0.]),
                frequencies=np.array([3.]), polarizations=np.array(["VV"]))


def write_payload(path, payload):
    with path.open("wb") as stream:
        np.savez(stream, **payload)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_roundtrip_uses_one_exact_authority(tmp_path, dtype):
    field = np.array([1. + 1.e-10j, -2. + 3.j, np.nan + 1j * np.nan], dtype=dtype).reshape(3, 1, 1, 1)
    grid = RcsGrid(**axes(), rcs=field, units={"frequency": "GHz"},
                   extra={"phase_reference": "origin", "sample_provenance": np.arange(3)})
    path = grid.save(tmp_path / "complex")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["grim_format_version"].item() == 2
        assert archive["storage_kind"].item() == "complex"
        assert archive["storage_domain"].item() == "physical"
        assert "rcs_power" not in archive and "rcs_phase" not in archive
        assert archive["rcs_complex"].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(archive["rcs_complex"], field)
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.rcs, field)
    assert restored.rcs.dtype == np.dtype(dtype)
    assert restored._phase_reference() == "origin"
    np.testing.assert_array_equal(restored.extra["sample_provenance"], np.arange(3))


def test_power_only_roundtrip_never_invents_phase_and_omits_phase_array(tmp_path):
    power = np.array([4., 0., np.nan], dtype=np.float32).reshape(3, 1, 1, 1)
    grid = RcsGrid(**axes(), rcs_power=power)
    path = grid.save(tmp_path / "power", compressed=False)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["storage_kind"].item() == "power"
        assert "rcs_phase" not in archive and "rcs_complex" not in archive
        np.testing.assert_array_equal(archive["rcs_power"], power)
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.power, power)
    assert np.isnan(np.asarray(restored.phase)).all()
    assert np.isnan(restored.rcs[0]).all()
    assert (restored.rcs[1] == 0.).all()


def test_mixed_phase_roundtrip_preserves_unphased_power(tmp_path):
    power = np.array([4., 9., 0.], dtype=np.float64).reshape(3, 1, 1, 1)
    phase = np.array([.2, np.nan, np.nan], dtype=np.float64).reshape(power.shape)
    grid = RcsGrid(**axes(), rcs_power=power, rcs_phase=phase)
    path = grid.save(tmp_path / "mixed")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["storage_kind"].item() == "power_phase"
        assert "rcs_complex" not in archive
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.power, power)
    np.testing.assert_array_equal(restored.phase, phase)


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_raw_solver_roundtrip_preserves_field_and_normalization_without_duplicates(tmp_path, quantity):
    raw = np.array([1. + 1.e-13j, -2. + .25j, np.nan + 1j * np.nan], dtype=np.complex128).reshape(3, 1, 1, 1)
    scale = (1. / (2. * np.sqrt(2. * np.pi * 3.e9 / C0)) if quantity == "sigma_2d"
             else np.sqrt(4. * np.pi))
    units = {"frequency": "GHz", "rcs_linear_quantity": quantity,
             "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm"}
    legacy = dict(axes(), rcs_power=(np.abs(raw * scale) ** 2).astype(np.float32),
                  rcs_phase=np.angle(raw).astype(np.float32), rcs_amp_real=raw.real,
                  rcs_amp_imag=raw.imag, raw_complex_amplitude_preserved=True,
                  units=json.dumps(units), amplitude_convention="solver field")
    original = tmp_path / "legacy-raw.grim"
    write_payload(original, legacy)
    grid = RcsGrid.load(original)
    path = grid.save(tmp_path / "compact-raw")
    with np.load(path, allow_pickle=False) as archive:
        assert archive["storage_domain"].item() == "raw_field"
        assert not ({"rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"} & set(archive.files))
        np.testing.assert_array_equal(archive["rcs_complex"], raw)
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.extra["rcs_amp_real"], raw.real)
    np.testing.assert_array_equal(restored.extra["rcs_amp_imag"], raw.imag)
    np.testing.assert_allclose(restored.rcs, raw * scale, rtol=2.e-15, atol=0., equal_nan=True)
    assert restored.linear_quantity() == quantity
    assert restored.extra["amplitude_convention"].item() == "solver field"


def test_legacy_power_phase_archive_remains_readable(tmp_path):
    power = np.array([4., 9., np.nan]).reshape(3, 1, 1, 1)
    phase = np.array([.2, np.nan, np.nan]).reshape(power.shape)
    path = tmp_path / "legacy.grim"
    write_payload(path, dict(axes(), rcs_power=power, rcs_phase=phase, units="{}"))
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.power, power)
    np.testing.assert_array_equal(restored.phase, phase)


@pytest.mark.parametrize("edit,match", [
    ({"grim_format_version": 3}, "unsupported grim_format_version"),
    ({"storage_kind": "unknown"}, "unsupported storage_kind"),
    ({"storage_domain": "voltage"}, "unsupported storage_domain"),
    ({"rcs_power": np.ones((3, 1, 1, 1))}, "inconsistent samples"),
    ({"rcs_complex": np.ones((3, 1, 1, 1))}, "invalid rcs_complex"),
    ({"rcs_complex": np.full((3, 1, 1, 1), complex(np.nan, 0.))}, "one-sided NaN"),
    ({"rcs_complex": np.full((3, 1, 1, 1), complex(np.inf, 0.))}, "infinite"),
])
def test_malformed_new_authority_is_rejected(tmp_path, edit, match):
    payload = dict(axes(), grim_format_version=2, storage_kind="complex",
                   storage_domain="physical", rcs_complex=np.ones((3, 1, 1, 1), complex))
    payload.update(edit)
    path = tmp_path / "malformed.grim"
    write_payload(path, payload)
    with pytest.raises(ValueError, match=match):
        RcsGrid.load(path)


def test_compact_archive_preflight_honors_limit_before_numpy_extraction(tmp_path):
    grid = RcsGrid(**axes(), rcs=np.ones((3, 1, 1, 1), dtype=np.complex128))
    path = grid.save(tmp_path / "bounded", compressed=False)
    report = _preflight_native_archive_allocation(path, max_output_bytes=1024**2)
    with mock.patch("numpy.load") as loader:
        with pytest.raises(MemoryError, match="peak memory|uncompressed native members"):
            RcsGrid.load(path, max_output_bytes=report["estimated_peak_bytes"] - 1)
        loader.assert_not_called()
    restored = RcsGrid.load(path, max_output_bytes=report["estimated_peak_bytes"])
    np.testing.assert_array_equal(restored.rcs, grid.rcs)


def test_embedded_and_file_native_payloads_share_the_same_decoder(tmp_path):
    field = np.array([1. + 1.e-14j, 3. - 5.j, 2. + 1.j]).reshape(3, 1, 1, 1)
    grid = RcsGrid(**axes(), rcs=field, source_path="source.pio", history="coherent sum")
    payload = grid._native_archive_payload()
    restored = RcsGrid._from_native_arrays(payload, source_path_override="assembly")
    np.testing.assert_array_equal(restored.rcs, field)
    assert restored.source_path == "assembly"
    assert restored.history == "coherent sum"
