"""Compact solver writes, bounded consumers, and legacy mutation semantics."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.data_tools.cem_tools import native_samples as codec
from ghost_backend.data_tools.cem_tools import grim_native as cem
from ghost_backend.io.grim import _save_grim_npz, _build_grid_for_samples
from ghost_backend.assembly.fields import _load_grim, _load_grim_sample
from ghost_backend.hpc.common import read_unit_grims


def data(quantity="sigma_3d", delta=False):
    raw = np.array([1.+1.e-20j, -2.-3.e-20j, 0.+0.j, 2.+3.j]).reshape(2, 1, 2, 1)
    metadata = dict(azimuths=np.array([0., 90.]), elevations=np.array([0.]),
        frequencies=np.array([2., 3.]), polarizations=np.array(["VV"]),
        rcs_domain="delta" if delta else "power_phase", power_domain="delta_amp_sq" if delta else "linear_rcs",
        units=json.dumps({"frequency": "GHz", "rcs_linear_quantity": quantity,
                          "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm"}))
    return codec.legacy_components(metadata, raw), raw


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_solver_writer_stores_one_exact_raw_authority_and_grim_can_read(tmp_path, quantity):
    payload, raw = data(quantity)
    path = _save_grim_npz(payload, str(tmp_path / "native.grim"))
    with np.load(path, allow_pickle=False) as archive:
        assert archive["grim_format_version"].item() == 2
        assert archive["storage_kind"].item() == "complex"
        assert archive["storage_domain"].item() == "raw_field"
        assert not {"rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"} & set(archive.files)
        np.testing.assert_array_equal(archive["rcs_complex"], raw)
    from GRIM_Backend.datasets.grid import RcsGrid
    loaded = RcsGrid.load(path)
    _, scale = codec.units_and_scale(payload)
    np.testing.assert_array_equal(loaded.rcs, raw * scale)
    np.testing.assert_array_equal(_load_grim(path)["_amp"], raw)


def test_producer_keeps_one_field_and_derived_samples_are_sliceable():
    rows = [dict(theta_scat_deg=angle, frequency_ghz=freq, rcs_linear=4*np.pi*abs(amp)**2,
                 rcs_amp_real=amp.real, rcs_amp_imag=amp.imag)
            for angle, freq, amp in [(0., 1., 2.+1.e-18j), (90., 1., 3.-1.e-18j)]]
    payload = _build_grid_for_samples(rows, "VV", rcs_linear_quantity="sigma_3d", rcs_log_unit="dBsm")
    field = codec.raw_amplitude(payload)
    assert payload["rcs_power"].field is payload["rcs_phase"].field
    assert payload["rcs_amp_real"].field is payload["rcs_amp_imag"].field is payload["rcs_power"].field
    assert np.shares_memory(field, payload["rcs_power"].field)
    np.testing.assert_array_equal(field[:, 0, 0, 0].imag, [1.e-18, -1.e-18])


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_save_assembly_hpc_and_cem_do_not_materialize_full_derived_arrays(tmp_path, monkeypatch, quantity):
    payload, raw = data(quantity)
    path = tmp_path / "VV_2GHz_native.grim"
    # HPC units have one frequency per file.
    raw = raw[:, :, :1]
    payload["frequencies"] = np.array([2.])
    payload = codec.legacy_components(payload, raw)
    def no_full_component(*args, **kwargs):
        raise AssertionError("A full derived component was materialized")
    monkeypatch.setattr(codec.SampleComponent, "__array__", no_full_component)
    _save_grim_npz(payload, str(path))
    whole = _load_grim(path)
    np.testing.assert_array_equal(whole["_amp"], raw)
    sample = _load_grim_sample(path, frequency=2., azimuth=90., elevation=0.)
    np.testing.assert_array_equal(sample["_amp"], raw[1:2])
    restored = cem.load_grim(path)
    np.testing.assert_array_equal(cem.raw_amplitude(restored, "test"), raw)
    np.testing.assert_array_equal(read_unit_grims(tmp_path)[0]["amp"], raw[:, 0, :, 0])


def test_explicit_legacy_component_edits_are_saved_without_old_authority(tmp_path):
    payload, raw = data()
    decoded = codec.decode_native_samples(codec.encode_native_samples(payload))
    decoded["rcs_amp_real"] *= 2.
    decoded["rcs_amp_imag"] *= 2.
    decoded["rcs_power"] *= 4.
    path = cem.save_grim_atomic(decoded, tmp_path / "edited.grim")
    restored = cem.load_grim(path)
    np.testing.assert_array_equal(cem.raw_amplitude(restored, "test"), raw * 2.)


@pytest.mark.parametrize('component', ['rcs_power', 'rcs_phase'])
@pytest.mark.parametrize('edit', ['setitem', 'iadd', 'imul'])
def test_solver_writer_keeps_small_explicit_component_edits(tmp_path, component, edit):
    payload, _ = data()
    payload = codec.decode_native_samples(codec.encode_native_samples(payload))
    original_component = payload[component]
    old = payload[component][1, 0, 1, 0]
    if edit == 'setitem':
        payload[component][1, 0, 1, 0] = np.nextafter(old, np.asarray(np.inf, dtype=old.dtype))
    elif edit == 'iadd':
        delta = np.zeros(payload[component].shape, dtype=old.dtype)
        delta[1, 0, 1, 0] = np.nextafter(old, np.asarray(np.inf, dtype=old.dtype)) - old
        payload[component] += delta
    else:
        factor = np.ones(payload[component].shape, dtype=old.dtype)
        factor[1, 0, 1, 0] += np.finfo(old.dtype).eps
        payload[component] *= factor
    assert payload[component] is original_component
    assert payload[component][1, 0, 1, 0] != old
    expected = np.array(payload[component], copy=True)
    path = _save_grim_npz(payload, str(tmp_path / 'edited.grim'))
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive['storage_kind']) == 'power_phase'
        np.testing.assert_array_equal(archive[component], expected)
        assert not {'rcs_amp_real', 'rcs_amp_imag'} & set(archive.files)
        restored = codec.decode_native_samples(dict(archive))
    np.testing.assert_array_equal(restored[component], expected)
    from GRIM_Backend.datasets.grid import RcsGrid
    grid = RcsGrid.load(path)
    actual = grid.power if component == 'rcs_power' else grid.phase
    np.testing.assert_allclose(actual[1, 0, 1, 0], expected[1, 0, 1, 0], rtol=1e-14, atol=1e-14)
    if component == 'rcs_phase':
        # GRIM wraps the float32 value just below -pi to the equivalent +pi.
        np.testing.assert_allclose(np.exp(1j*np.asarray(actual)), np.exp(1j*expected), atol=4*np.finfo(np.float32).eps)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("phase", [None, 0., 1., 7.])
def test_zero_and_unknown_or_unwrapped_phase_remain_distinct(phase):
    payload, _ = data()
    payload = {k: v for k, v in payload.items() if k not in codec.SAMPLES}
    payload["rcs_power"] = np.zeros((2, 1, 2, 1))
    payload["rcs_phase"] = np.full((2, 1, 2, 1), np.nan if phase is None else phase)
    compact = codec.encode_native_samples(payload)
    restored = codec.decode_native_samples(compact)
    np.testing.assert_array_equal(restored["rcs_power"], payload["rcs_power"])
    np.testing.assert_array_equal(restored["rcs_phase"], payload["rcs_phase"])
    assert compact["storage_kind"] == ("power" if phase is None else "complex" if phase == 0 else "power_phase")


def test_delta_field_units_and_metadata_survive_cem_save(tmp_path):
    payload, raw = data("sigma_2d", delta=True)
    payload["_custom_ancillary"] = np.array([42.])
    path = cem.save_grim_atomic(payload, tmp_path / "delta.grim")
    restored = cem.load_grim(path)
    np.testing.assert_array_equal(cem.raw_amplitude(restored, "delta"), raw)
    np.testing.assert_allclose(restored["rcs_power"], np.abs(raw)**2, rtol=2.e-15)
    assert str(restored["rcs_domain"]) == "delta" and str(restored["power_domain"]) == "delta_amp_sq"
    np.testing.assert_array_equal(restored["_custom_ancillary"], [42.])


def test_empty_grid_fails_without_looping():
    payload, _ = data()
    compact = codec.encode_native_samples(payload)
    compact["azimuths"] = np.array([])
    compact["rcs_complex"] = np.empty((0, 1, 2, 1), complex)
    with pytest.raises(ValueError, match="nonempty"):
        codec.decode_native_samples(compact)


@pytest.mark.parametrize("edit", ["fill", "at", "asarray", "small_power", "small_phase"])
def test_legacy_mutations_are_never_discarded(edit):
    payload, _ = data()
    decoded = codec.decode_native_samples(codec.encode_native_samples(payload))
    if edit == "fill":
        decoded["rcs_power"].fill(1.)
    elif edit == "at":
        np.add.at(decoded["rcs_power"], (0, 0, 0, 0), 2.)
    elif edit == "asarray":
        np.asarray(decoded["rcs_power"])[0, 0, 0, 0] += 2.
    elif edit == "small_power":
        decoded["rcs_power"][0, 0, 0, 0] *= 1.0000001
    else:
        decoded["rcs_phase"][0, 0, 0, 0] += 1.e-7
    expected = {k: np.array(decoded[k]) for k in ("rcs_power", "rcs_phase")}
    restored = codec.decode_native_samples(codec.encode_native_samples(decoded))
    for key, value in expected.items():
        np.testing.assert_array_equal(restored[key], value)


def test_full_turn_phase_is_not_silently_wrapped():
    payload, _ = data()
    payload = {k: v for k, v in payload.items() if k not in codec.SAMPLES}
    payload["units"] = json.dumps({"rcs_linear_quantity": "sigma_3d", "phase_wrap": "0_360"})
    payload["rcs_power"] = np.ones((2, 1, 2, 1)) * (4*np.pi)
    payload["rcs_phase"] = np.full((2, 1, 2, 1), 2*np.pi)
    payload["rcs_amp_real"], payload["rcs_amp_imag"] = np.ones((2, 1, 2, 1)), np.zeros((2, 1, 2, 1))
    compact = codec.encode_native_samples(payload)
    assert compact["storage_kind"] == "power_phase"
    np.testing.assert_array_equal(codec.decode_native_samples(compact)["rcs_phase"], payload["rcs_phase"])


def test_large_fortran_grid_validation_uses_bounded_derived_blocks(tmp_path, monkeypatch):
    metadata = dict(azimuths=np.arange(257.), elevations=np.arange(3.),
                    frequencies=np.linspace(1., 3., 401), polarizations=np.array(["VV"]),
                    units=json.dumps({"frequency": "GHz", "rcs_linear_quantity": "sigma_2d"}),
                    rcs_domain="power_phase", power_domain="linear_rcs")
    field = np.full((257, 3, 401, 1), 1.+.5j, order="F")
    payload = codec.legacy_components(metadata, field)
    original = codec.SampleComponent.__getitem__
    sizes = []
    def bounded(self, selection):
        value = original(self, selection)
        sizes.append(value.size)
        return value
    monkeypatch.setattr(codec.SampleComponent, "__getitem__", bounded)
    monkeypatch.setattr(codec.SampleComponent, "__array__", lambda *args, **kwargs: pytest.fail("full component"))
    path = _save_grim_npz(payload, str(tmp_path / "large.grim"))
    np.testing.assert_array_equal(_load_grim(path)["_amp"], field)
    assert sizes and max(sizes) <= 262144


def test_compact_join_reads_components_in_blocks_and_preserves_inputs(monkeypatch):
    first, raw = data()
    second = codec.legacy_components(first, raw * (1. + .5j))
    second["frequencies"] = np.array([4., 5.])
    def no_full_component(*args, **kwargs):
        pytest.fail("join expanded a complete compatibility component")
    monkeypatch.setattr(codec.SampleComponent, "__array__", no_full_component)
    merged = cem.join_payloads([first, second], axis="frequencies")
    np.testing.assert_array_equal(codec.raw_amplitude(merged), np.concatenate([raw, raw * (1. + .5j)], axis=2))
    np.testing.assert_array_equal(codec.raw_amplitude(first), raw)
    np.testing.assert_array_equal(codec.raw_amplitude(second), raw * (1. + .5j))
    assert not np.shares_memory(codec.raw_amplitude(merged), raw)


def test_decoder_blocks_a_single_wide_leading_row(monkeypatch):
    shape = (1, 1, 300_000, 2)
    payload = dict(azimuths=[0.], elevations=[0.], frequencies=np.arange(1., shape[2]+1.),
                   polarizations=['VV', 'HH'], grim_format_version=2, storage_kind='complex',
                   storage_domain='raw_field', rcs_complex=np.ones(shape, complex),
                   units=json.dumps({'frequency': 'GHz', 'rcs_linear_quantity': 'sigma_3d'}))
    original = codec.np.isfinite
    checked = []
    def bounded(value, *args, **kwargs):
        checked.append(np.size(value))
        return original(value, *args, **kwargs)
    monkeypatch.setattr(codec.np, 'isfinite', bounded)
    result = codec.decode_native_samples(payload)
    assert checked and max(checked) <= codec.DECODE_BLOCK_CELLS
    assert result['rcs_power']._detached is None
