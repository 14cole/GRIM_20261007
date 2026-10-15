"""Phase-sign correction preserves storage semantics and reverses range images."""
import json
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.operations import operation_events
from GRIM_Backend.datasets.result_storage import operation_result_storage
from GRIM_Backend.datasets.transforms import flip_phase


def make_grid(*, field=None, power=None, phase=None, units=None, extra=None, raw=False):
    values = field if field is not None else power
    shape = np.shape(values)
    return RcsGrid(np.arange(shape[0]) * 10., [0.], np.linspace(8., 12., shape[2]), ['VV'],
                   **({'rcs': field, 'rcs_storage_domain': 'raw_field' if raw else 'physical'}
                      if field is not None else {'rcs_power': power, 'rcs_phase': phase}),
                   units={'frequency': 'GHz', 'azimuth': 'deg', 'elevation': 'deg',
                          'rcs_linear_quantity': 'sigma_3d', 'rcs_log_unit': 'dBsm', **(units or {})},
                   extra=extra, history='Loaded source')


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('storage', ['memory', 'disk', 'auto'])
def test_exact_conjugation_double_flip_input_immutable_and_managed_storage(dtype, storage, tmp_path):
    values = np.array([1+2j, -3-4j, 0j, complex(np.nan, np.nan),
                       -2+0j, 1e-5-2e-4j, 0+3j, 4+5j], dtype=dtype).reshape(2, 1, 4, 1)
    source = make_grid(field=values)
    original = source.rcs
    with operation_result_storage(storage, ram_budget_bytes=0, cache_dir=tmp_path):
        flipped = flip_phase(source)
        restored = flip_phase(flipped)
    assert flipped._complex.dtype == np.dtype(dtype)
    assert flipped.sample_storage == ('memory' if storage == 'memory' else 'disk')
    assert not np.shares_memory(source._complex, flipped._complex)
    np.testing.assert_array_equal(flipped.rcs, np.conj(original))
    np.testing.assert_array_equal(flipped.power, source.power)
    np.testing.assert_array_equal(restored.rcs, original)
    np.testing.assert_array_equal(source.rcs, original)
    assert source.history == 'Loaded source'
    assert not operation_events(source)
    assert [event['operation'] for event in operation_events(restored)] == ['flip_phase', 'flip_phase']


@pytest.mark.parametrize('wrap', ['', '-180_180', '0_360'])
def test_unknown_phase_zero_amplitude_and_phase_wrap_are_preserved(wrap):
    power = np.array([4., 0., 9., np.nan, 1., 0.]).reshape(1, 1, 6, 1)
    phase = np.array([np.nan, np.nan, 1.2, np.nan, 0., 1.5]).reshape(power.shape)
    source = make_grid(power=power, phase=phase, units={'phase_wrap': wrap})
    flipped, restored = flip_phase(source), flip_phase(flip_phase(source))
    expected = -phase
    if wrap == '0_360':
        expected = np.remainder(expected, 2*np.pi)
    np.testing.assert_array_equal(flipped.power, power)
    np.testing.assert_allclose(flipped.phase, expected, rtol=0, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(restored.phase, phase, rtol=0, atol=1e-15, equal_nan=True)
    assert flipped.phase_coverage() == source.phase_coverage() == 'partial'


def test_power_only_and_unwrapped_phase_do_not_gain_or_lose_information():
    power = np.array([1., 0., np.nan, 4.], dtype=np.float32).reshape(1, 1, 4, 1)
    source = make_grid(power=power)
    flipped = flip_phase(source)
    assert flipped.storage_kind == 'power'
    assert flipped._phase is None
    np.testing.assert_array_equal(flipped.power, power)
    phase = np.array([20., 4., np.nan, -30.], dtype=np.float32).reshape(power.shape)
    source = make_grid(power=power, phase=phase)
    flipped = flip_phase(source)
    np.testing.assert_array_equal(flipped.phase, -phase)
    np.testing.assert_array_equal(flip_phase(flipped).phase, phase)


@pytest.mark.parametrize('quantity', ['sigma_2d', 'sigma_3d'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_raw_solver_authority_and_precision_survive_native_roundtrip(quantity, dtype, tmp_path):
    values = np.array([1.123456789+2.456789012j, -3.141592654-4.23456789j, 0j, 2-5j], dtype=dtype).reshape(1, 1, 4, 1)
    source = make_grid(field=values, raw=True,
                       units={'rcs_linear_quantity': quantity,
                              'rcs_log_unit': 'dBke' if quantity == 'sigma_2d' else 'dBsm'},
                       extra={'amplitude_convention': 'F raw solver amplitude',
                              'solver_certification': 'source only'})
    with operation_result_storage('disk', cache_dir=tmp_path):
        flipped = flip_phase(source)
    assert flipped.storage_domain == 'raw_field'
    assert flipped._complex.dtype == values.dtype
    assert 'solver_certification' not in flipped.extra
    assert flipped.extra['amplitude_convention'] == source.extra['amplitude_convention']
    np.testing.assert_array_equal(flipped._complex, np.conj(values))
    np.testing.assert_array_equal(flipped.power, source.power)
    np.testing.assert_array_equal(flipped.rcs, np.conj(source.rcs))
    path = flipped.save(tmp_path / f'{quantity}-{dtype.__name__}.grim')
    restored = RcsGrid.load(path)
    assert restored.storage_domain == 'raw_field'
    np.testing.assert_array_equal(restored._complex, flipped._complex)
    np.testing.assert_array_equal(restored.power, source.power)
    np.testing.assert_array_equal(flip_phase(restored)._complex, values)


def test_attached_raw_pair_remains_authoritative_and_exceptional_phase_stays_unknown():
    values = np.array([1+2j, 3-4j, 0j, -5+6j]).reshape(1, 1, 4, 1)
    source = make_grid(field=values)
    raw = values * 0.5
    source.extra.update(rcs_amp_real=raw.real.copy(), rcs_amp_imag=raw.imag.copy())
    flipped = flip_phase(source)
    assert flipped.storage_domain == 'raw_field'
    np.testing.assert_array_equal(flipped.rcs, np.conj(source.rcs))
    np.testing.assert_array_equal(flipped._complex, np.conj(raw))
    power = 4*np.pi * np.abs(raw)**2
    phase = np.array([np.nan, -0.9, 1.2, 0.]).reshape(power.shape)
    polar = make_grid(power=power, phase=phase)
    polar.extra.update(rcs_amp_real=raw.real.copy(), rcs_amp_imag=raw.imag.copy())
    flipped = flip_phase(polar)
    np.testing.assert_array_equal(flipped.phase, -phase)
    np.testing.assert_array_equal(flipped.power, polar.power)
    np.testing.assert_array_equal(flipped.extra['rcs_amp_real'], raw.real)
    np.testing.assert_array_equal(flipped.extra['rcs_amp_imag'], -raw.imag)


def test_phase_correction_retains_declared_interpretation_and_records_it(tmp_path):
    source = make_grid(field=np.ones((1, 1, 4, 1), dtype=complex) * (1+2j),
                       units={'time_convention': 'exp(-jwt)'},
                       extra={'phase_reference': 'fixed origin, exp(-jwt)',
                              'range_phase_convention': 'exp(+j*2*k*R)'})
    flipped = flip_phase(source)
    assert flipped.units == source.units
    for key in ('phase_reference', 'range_phase_convention'):
        assert flipped.extra[key] == source.extra[key]
    detail = json.loads(flipped.extra['phase_sign_correction_json'])
    assert 'declared time and phase conventions retained' in detail['interpretation']
    assert operation_events(flipped)[-1]['details'] == detail
    restored = RcsGrid.load(flipped.save(tmp_path / 'flipped.grim'))
    assert restored.extra['phase_sign_correction_json'] == flipped.extra['phase_sign_correction_json']
    assert operation_events(restored) == operation_events(flipped)


def test_native_roundtrip_preserves_unknown_phase(tmp_path):
    source = make_grid(power=np.array([1., 0., 3., np.nan]).reshape(1, 1, 4, 1),
                       phase=np.array([np.nan, 1., -.5, np.nan]).reshape(1, 1, 4, 1))
    flipped = flip_phase(source)
    restored = RcsGrid.load(flipped.save(tmp_path / 'partial.grim'))
    np.testing.assert_array_equal(restored.power, source.power)
    np.testing.assert_array_equal(restored.phase, -source.phase)


def test_assembly_hash_is_invalidated_without_losing_source_lineage():
    from GRIM_Backend.assembly.tree import (
        _assembly_response_physics_sha256, _assert_no_shared_body_totals,
    )
    values = np.array([1+2j, 3-4j, 5+6j, 7+8j, 9+10j, 11-12j]).reshape(1, 1, 2, 3)
    metadata = {'phase_reference': 'fixed origin, exp(+jwt)',
                'amplitude_convention': 'F raw solver amplitude',
                'complex_field_domain': 'coherent radar frame far field amplitude'}
    source = RcsGrid([0.], [0.], [8., 9.], ['VV', 'HH', 'VH'],
                     rcs=values, rcs_storage_domain='raw_field',
                     units={'rcs_linear_quantity': 'sigma_3d'}, extra=metadata)
    original_digest = _assembly_response_physics_sha256(source)
    source.extra['assembly_base_response_sha256'] = original_digest
    delta = RcsGrid([0.], [0.], [8., 9.], ['VV', 'HH', 'VH'],
                    rcs=values * .1, rcs_storage_domain='raw_field',
                    units=source.units,
                    extra={**metadata, 'assembly_response_role': 'features_only_delta',
                           'assembly_base_sha256': 'a' * 64,
                           'assembly_base_response_sha256': original_digest})
    _assert_no_shared_body_totals([source, delta])
    flipped = flip_phase(source)
    assert _assembly_response_physics_sha256(flipped) != original_digest
    assert 'assembly_base_response_sha256' not in flipped.extra
    assert flipped.extra['assembly_source_base_response_sha256'] == original_digest
    assert flipped.extra['assembly_base_response_transform'] == 'flip-phase'
    assert source.extra['assembly_base_response_sha256'] == original_digest
    with pytest.raises(ValueError, match='proven matching'):
        _assert_no_shared_body_totals([flipped, delta])
    # Once the numerical response is restored, Assembly can prove its match
    # from the actual samples instead of trusting an obsolete declaration.
    restored = flip_phase(flipped)
    assert _assembly_response_physics_sha256(restored) == original_digest
    _assert_no_shared_body_totals([restored, delta])


@pytest.mark.parametrize('format_name', ['ptm', 'pio'])
def test_ptm_and_pio_retain_flipped_iq_without_implicit_sign_conversion(format_name, tmp_path):
    angle = np.linspace(-2.5, 2.5, 3*64).reshape(3, 1, 64, 1)
    source = make_grid(field=(np.exp(1j*angle) * 3.).astype(np.complex64))
    flipped = flip_phase(source)
    path = tmp_path / ('flipped.' + format_name)
    getattr(flipped, 'save_' + format_name)(path)
    restored = getattr(RcsGrid, 'load_' + format_name)(path)
    np.testing.assert_allclose(restored.rcs, np.conj(source.rcs), rtol=1e-7, atol=1e-7)
    assert not restored.units.get('time_convention')


@pytest.mark.parametrize('time_convention', ['exp(+jwt)', 'exp(-jwt)'])
def test_phase_flip_reverses_known_azimuth_downrange_peak(time_convention):
    from GRIM_Backend.datasets.transforms import _declared_time_sign
    from GRIM_Backend.plotting.modes.az_vs_range_mode import _form_range_image
    count, peak = 64, 9
    history = 3*np.exp(-2j*np.pi*peak*np.arange(count)/count)
    source = make_grid(field=np.tile(history, (2, 1)).reshape(2, 1, count, 1),
                       units={'time_convention': time_convention})

    def image(grid):
        field = grid.rcs[:, 0, :, 0]
        if _declared_time_sign(grid) == '-jwt':
            field = np.conj(field)
        return _form_range_image(field, np.ones(field.shape), np.ones(count))[0]

    original, flipped, restored = image(source), image(flip_phase(source)), image(flip_phase(flip_phase(source)))
    coordinates = np.arange(count) - count//2
    original_peak = coordinates[np.argmax(np.abs(original), axis=1)]
    flipped_peak = coordinates[np.argmax(np.abs(flipped), axis=1)]
    np.testing.assert_array_equal(flipped_peak, -original_peak)
    np.testing.assert_allclose(np.max(np.abs(flipped), axis=1), 3., rtol=1e-6)
    np.testing.assert_array_equal(restored, original)


def test_operation_blocks_conjugation_without_whole_grid_scratch():
    source = make_grid(field=np.full((3, 1, 100_000, 1), 1+2j, dtype=np.complex64))
    original = np.conjugate
    sizes = []

    def conjugate(values, *, out):
        sizes.append(values.size)
        assert out is not None
        return original(values, out=out)

    with mock.patch('GRIM_Backend.datasets.transforms.np.conjugate', side_effect=conjugate):
        result = flip_phase(source)
    assert len(sizes) > 1 and max(sizes) <= 262_144
    np.testing.assert_array_equal(result.rcs[0, 0, :2, 0], [1-2j, 1-2j])


def test_public_workspace_and_recorder_make_operation_replayable(tmp_path):
    from GRIM_Backend.scripting.workspace import flip_phase as exported
    from GRIM_Backend.scripting.recorder import DatasetReference, PythonScriptRecorder
    assert exported is flip_phase
    source = make_grid(field=np.full((1, 1, 4, 1), 1+2j))
    path = source.save(tmp_path / 'source.grim')
    recorder = PythonScriptRecorder()
    source_ref = DatasetReference('source', 'source', str(path))
    output_ref = DatasetReference('flipped', 'flipped', None)
    recorder.record_function(output_ref, 'flip_phase', [source_ref])
    namespace = {'__file__': str(tmp_path / 'replay.py')}
    exec(compile(recorder.script, str(tmp_path / 'replay.py'), 'exec'), namespace)
    results = [value for value in namespace.values() if isinstance(value, RcsGrid)]
    assert any(np.array_equal(value.rcs, np.conj(source.rcs)) for value in results)
