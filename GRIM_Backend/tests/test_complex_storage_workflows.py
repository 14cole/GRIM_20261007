"""Storage stays canonical through plotting, audit, and embedded assemblies."""
import base64
import io

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.assembly.tree import _grid_to_b64, _b64_to_grid
from GRIM_Backend.io.batch import _grim_save_compression_decision
from GRIM_Backend.plotting.engineering_tools import dataset_cut
from GRIM_Backend.plotting.image_analysis import heatmap_display_plane
from GRIM_Backend.plotting.modes.isar_mode import _selected_data_token


def grid(dtype=np.complex128):
    field = (np.arange(24).reshape(3, 1, 4, 2) + 1j).astype(dtype)
    result = RcsGrid([0., 1., 2.], [0.], [8., 9., 10., 11.], ['HH', 'VV'], rcs=field,
                     units={'frequency': 'GHz', 'rcs_linear_quantity': 'sigma_3d'})
    return result, field


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_plot_cut_image_audit_and_save_policy_keep_complex_storage(dtype):
    ds, field = grid(dtype)
    cut = dataset_cut(ds, polarization='HH')
    np.testing.assert_array_equal(cut.field, field[0, 0, :, 0])
    image = heatmap_display_plane(ds, [np.arange(3), [0], np.arange(4), [0]],
                                  'azimuth', 'frequency', linear=True)
    np.testing.assert_allclose(image, np.abs(field[:, 0, :, 0]) ** 2, rtol=1.e-6)
    ds.audit()
    decision = _grim_save_compression_decision(ds)
    assert isinstance(decision, dict)
    payload = ds.native_sample_payload()
    assert payload['storage_kind'] == 'complex'
    np.testing.assert_array_equal(payload['rcs_complex'], field)


def test_assembly_embedding_uses_one_representation_and_reads_legacy():
    ds, field = grid()
    encoded = _grid_to_b64(ds)
    with np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False) as data:
        assert 'rcs_complex' in data.files
        assert 'rcs_power' not in data.files
        assert 'rcs_phase' not in data.files
    restored = _b64_to_grid(encoded)
    np.testing.assert_array_equal(restored.rcs, field)
    assert restored.native_sample_payload()['storage_kind'] == 'complex'
    legacy = io.BytesIO()
    np.savez(legacy, azimuths=ds.azimuths, elevations=ds.elevations,
             frequencies=ds.frequencies, polarizations=ds.polarizations,
             rcs_power=np.abs(field) ** 2, rcs_phase=np.angle(field))
    restored_legacy = _b64_to_grid(base64.b64encode(legacy.getvalue()).decode('ascii'))
    np.testing.assert_allclose(restored_legacy.rcs, field, atol=1.e-13)


def test_isar_cache_detects_legacy_array_edits_and_retains_source_on_reads():
    ds, field = grid()
    key = lambda: _selected_data_token(ds, np.arange(3), 0, np.arange(4), 0)
    before = key()
    assert ds.native_sample_payload()['storage_kind'] == 'complex'
    _ = ds.rcs_power
    np.testing.assert_array_equal(ds.rcs, field)
    after_read = key()
    ds.rcs_power[0, 0, 0, 0] = 100.
    assert key() != after_read
    assert np.abs(ds.rcs_slice((0, 0, 0, 0))) ** 2 == pytest.approx(100.)
    assert before is not None


@pytest.mark.parametrize('domain', ['physical', 'raw_field'])
def test_native_validation_does_not_materialize_whole_derived_arrays(tmp_path, monkeypatch, domain):
    from GRIM_Backend.datasets.storage import SampleComponent
    ds, expected = grid()
    if domain == 'raw_field':
        ds = RcsGrid(ds.azimuths, ds.elevations, ds.frequencies, ds.polarizations,
                     rcs=expected, rcs_storage_domain=domain, units=ds.units)
    physical = ds.rcs.copy()
    def disallow_full_conversion(*args, **kwargs):
        raise AssertionError('native validation must use bounded sample slices')
    monkeypatch.setattr(SampleComponent, '__array__', disallow_full_conversion)
    destination = tmp_path / 'bounded.grim'
    ds.save(destination)
    restored = RcsGrid.load(destination)
    np.testing.assert_array_equal(restored.rcs, physical)
    assert restored.storage_kind == 'complex'


def test_isar_cache_tracks_raw_fields_in_exceptional_phase_storage():
    raw = np.array([0., 2.]).reshape(2, 1, 1, 1) / np.sqrt(4*np.pi)
    ds = RcsGrid([0., 1.], [0.], [10.], ['HH'],
                 rcs_power=np.array([0., 4.]).reshape(2, 1, 1, 1),
                 rcs_phase=np.array([np.nan, 0.]).reshape(2, 1, 1, 1),
                 extra={'rcs_amp_real': raw, 'rcs_amp_imag': np.zeros_like(raw)})
    assert ds.native_sample_payload()['storage_kind'] == 'power_phase'
    before = _selected_data_token(ds, [0, 1], 0, [0], 0)
    ds.extra['rcs_amp_real'][1, 0, 0, 0] *= 2.
    after = _selected_data_token(ds, [0, 1], 0, [0], 0)
    assert before != after
    assert np.isnan(ds.phase[0, 0, 0, 0])
