"""Reject inadmissible requests without attempting the dangerous allocations."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import math

import numpy as np
import pytest

from ibc import compute, sweep_admission as admission
from ibc.compute import LayerConfig, LoadedLayer, ConstantMaterial, UncertaintyConfig
from ibc.mix_analysis import evaluate_mix_performance, build_mix_display
from ibc.tolerance_analysis import PreparedStudy
from ibc.tolerance_config import validate_setup
from ibc.batch import ibc_batch_frequency_count
from ibc.ui import ImpedanceGui


class UnallocatedAxis:
    """Count-only input: any attempt to read/build its contents is a failure."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError('Allocated/iterated an inadmissible axis')


@pytest.mark.parametrize('start,stop,step,expected', [
    (1., 20., 1e-9, 19_000_000_001),
    (1., 1., 1e-300, 1),
    (1., 2., .3, 4),
    (8., 12., .1, 41),
    (0., .3, .1, 4),
    (1., 2., 2., 1),
])
def test_count_preserves_anchored_endpoints(start, stop, step, expected):
    assert admission.sweep_count(start, stop, step) == expected
    if expected <= 100:
        assert compute.make_sweep(start, stop, step) == [start + i * step for i in range(expected)]


@pytest.mark.parametrize('args', [
    (math.nan, 1, 1), (0, math.inf, 1), (0, 1, math.nan),
    (0, 1, 0), (1, 0, 1), (0, 1, -1),
    (-1e308, 1e308, 1), (1, 20, 5e-324),
])
def test_invalid_or_overflowing_count_is_actionable_value_error(args):
    with pytest.raises(ValueError):
        admission.sweep_count(*args)


def test_hostile_sweep_rejected_before_range_or_numpy_allocation():
    with mock.patch.object(compute, 'range', create=True, side_effect=AssertionError('range reached')), \
         mock.patch.object(compute.np, 'asarray', side_effect=AssertionError('array reached')):
        with pytest.raises(ValueError, match='19,000,000,001.*1,000,000'):
            compute.make_frequency_sweep(1, 20, 1e-9)


def test_exact_axis_boundary_without_allocating_values():
    assert admission.validate_axis_count(1_000_000) == 1_000_000
    with pytest.raises(ValueError, match='1,000,001'):
        admission.validate_axis_count(1_000_001)


@pytest.mark.parametrize('kwargs', [
    {'frequency_count': math.nan}, {'frequency_count': 1.5},
    {'frequency_count': True}, {'frequency_count': 0},
    {'frequency_count': 2, 'layer_count': math.nan},
    {'frequency_count': 2, 'extra_bytes': math.inf},
    {'frequency_count': 2, 'layer_count': -1},
])
def test_noninteger_counts_cannot_bypass_budget(kwargs):
    with pytest.raises(ValueError):
        admission.validate_grid(**kwargs)


def test_response_product_is_bounded_even_when_both_axes_are_small():
    with pytest.raises(ValueError, match='4,000,000 response points'):
        admission.validate_grid(2000, 2000)
    # Just at the response boundary, sufficiently light retention still fits.
    assert admission.validate_grid(2000, 1000, metric_count=1) < admission.MAX_STUDY_BYTES


def test_captured_polarizations_and_uncertainty_are_counted():
    assert admission.validate_analysis_grid(1000, 200) < admission.MAX_STUDY_BYTES
    assert admission.validate_analysis_grid(1000, 200, polarizations=2) < admission.MAX_STUDY_BYTES
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_analysis_grid(1000, 200, uncertainty=True, polarizations=2)


def test_inverse_cache_and_retained_plots_have_independent_budgets():
    # All axes and the response product are small; the per-layer cache is not.
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_inverse_grid(1000, 100, layer_count=100, case_count=9,
                                        design_count=1, top_n=1)
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_inverse_grid(1000, 100, layer_count=1, case_count=9,
                                        design_count=100, top_n=100)
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_inverse_grid(1, 1, layer_count=1, case_count=1,
                                        design_count=np.int64(2**62), top_n=1)


def test_mix_component_and_retained_result_products_are_bounded():
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_mix_grid(100_000, 1, component_count=100)
    with pytest.raises(ValueError, match='512 MiB'):
        admission.validate_mix_grid(1000, 100, retained_results=100)


@pytest.mark.parametrize('entry', ['metrics', 'impedance', 'properties', 'waves'])
def test_vector_api_guards_before_reading_axis(entry):
    freqs = UnallocatedAxis(19_000_000_001)
    with pytest.raises(ValueError, match='desktop limit'):
        if entry == 'metrics':
            compute.compute_angle_metrics_many(freqs, 0, [], 'te')
        elif entry == 'impedance':
            compute.compute_stack_impedance_many(freqs, [], 'pec')
        elif entry == 'properties':
            compute.prepare_layer_properties_many(freqs, [])
        else:
            compute.prepare_layer_wave_terms_many(freqs, 0, [], 'te')


@pytest.mark.parametrize('numpy_available', [True, False])
def test_dense_grid_guard_precedes_preparation_and_allocation(numpy_available):
    frequencies, angles = UnallocatedAxis(2000), UnallocatedAxis(2000)
    with mock.patch('ibc.ui.NUMPY_AVAILABLE', numpy_available), \
         mock.patch('ibc.ui.prepare_layer_properties_many', side_effect=AssertionError('prepared')), \
         mock.patch('ibc.ui.np.zeros', side_effect=AssertionError('allocated')):
        with pytest.raises(ValueError, match='4,000,000 response points'):
            ImpedanceGui._compute_heatmap_data(None, [], 'te', angles, frequencies)
        with pytest.raises(ValueError, match='4,000,000 response points'):
            ImpedanceGui._compute_thickness_data(None, [], 0, angles, 'te', 0, frequencies)


def test_internal_modes_budget_envelopes_before_nominal_allocation():
    sentinel = mock.Mock(side_effect=AssertionError('nominal allocated'))
    fake = SimpleNamespace(_compute_heatmap_data=sentinel, _compute_thickness_data=sentinel)
    uncertainty = UncertaintyConfig(True, 5, 5, 5)
    with pytest.raises(ValueError, match='512 MiB'):
        ImpedanceGui._compute_angle_mode(fake, Path('unused.csv'), [], uncertainty,
                                        UnallocatedAxis(300), UnallocatedAxis(1000), 'te')
    with pytest.raises(ValueError, match='512 MiB'):
        ImpedanceGui._compute_thickness_mode(fake, Path('unused.csv'), [], 0, uncertainty,
                                            UnallocatedAxis(300), UnallocatedAxis(1000), 'te', 0)
    sentinel.assert_not_called()


def test_mix_and_tolerance_services_guard_before_interpolation():
    with pytest.raises(ValueError, match='response points'):
        evaluate_mix_performance(SimpleNamespace(freq_ghz=UnallocatedAxis(2000)), .1,
                                 {'angles': UnallocatedAxis(2000)})
    with pytest.raises(ValueError, match='response points'):
        build_mix_display([], 'unused', .1, UnallocatedAxis(2000),
                          performance={'angles': UnallocatedAxis(2000)})
    with pytest.raises(ValueError, match='response points'):
        PreparedStudy([], UnallocatedAxis(2000), UnallocatedAxis(2000), ['te'], [])


@pytest.mark.parametrize('nf,na,layers,designs,kept', [
    (2000, 2000, 1, 1, 1),
    (1000, 100, 100, 1, 1),
    (1000, 100, 1, 100, 100),
])
def test_inverse_service_rejects_before_identity_files_or_prepared_cache(nf, na, layers, designs, kept):
    from ibc.design_search import InverseSearchRequest, run_inverse_search
    request = InverseSearchRequest(
        layer_snapshot=[object()] * layers, target_freqs=UnallocatedAxis(nf),
        target_angles=UnallocatedAxis(na), wave_pol='te',
        uncertainty_cfg=UncertaintyConfig(True, 5, 5, 5), score_mode='worst',
        checkpoint=None, grid=SimpleNamespace(total=designs), top_n=kept,
        target_freq_desc='test', a_start=0., a_stop=80., numpy_available=True)
    with mock.patch('ibc.design_search.search_identity', side_effect=AssertionError('identity read')), \
         mock.patch('ibc.design_search.prepare_layer_wave_terms_many', side_effect=AssertionError('cache allocated')):
        with pytest.raises(ValueError, match='desktop'):
            run_inverse_search(request, stop_requested=lambda: False, progress=mock.Mock(),
                               score_candidate=mock.Mock())


def test_mix_search_rejects_retained_result_product_before_material_reads():
    from ibc.design_search import MixSearchRequest, run_mix_search
    request = MixSearchRequest(
        uncertainty_cfg=UncertaintyConfig(False, 0, 0, 0), comp_snapshot=[{}, {}],
        target_freqs=UnallocatedAxis(1000), rule_norm='linear', property_mode=False,
        performance_mode=True, target=None, thickness_in=.1,
        performance_config={'angles': UnallocatedAxis(100)}, score_mode='worst',
        search_seed=1, lower=[0, 0], upper=[1, 1], max_evals=100, top_n=100,
        refine=False, prop_desc='', target_desc='', numpy_available=True)
    reader = mock.Mock(side_effect=AssertionError('material read'))
    with pytest.raises(ValueError, match='512 MiB'):
        run_mix_search(request, read_table=reader)
    reader.assert_not_called()


def test_existing_tolerance_and_batch_policies_keep_count_first_checks():
    with pytest.raises(ValueError, match='too large'):
        validate_setup({'f_start': '1', 'f_stop': '20', 'f_step': '1e-9'})
    with pytest.raises(ValueError, match='desktop range'):
        ibc_batch_frequency_count(1, 20, 5e-324)


@pytest.fixture
def workspace(tmp_path):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    window = ImpedanceGui()
    window.layers = [LayerConfig(.1, False, '', '', 0., material_source="constant")]
    window._refresh_layers()
    for name in ('output_var', 'angle_output_var', 'thk_output_var'):
        getattr(window, name).set(str(tmp_path / 'out.csv'))
    window.uncertainty_var.set(False)
    window.angle_uncertainty_var.set(False)
    window.thk_uncertainty_var.set(False)
    yield window
    window.deleteLater()
    app.processEvents()


@pytest.mark.parametrize('mode,prefix', [('impedance', ''), ('off_angle', 'angle_'), ('thickness', 'thk_')])
def test_gui_rejects_before_sweep_allocation_dispatch_or_output_confirmation(workspace, mode, prefix):
    for suffix, value in [('start', '1'), ('stop', '20'), ('step', '1e-9')]:
        getattr(workspace, f'{prefix}f_{suffix}_var').set(value)
    with mock.patch('ibc.ui.make_frequency_sweep', side_effect=AssertionError('sweep allocated')), \
         mock.patch('ibc.ui.messagebox.showerror') as error, \
         mock.patch.object(workspace, '_run_background_task') as dispatch, \
         mock.patch.object(workspace, '_confirm_output_replacements') as confirm:
        getattr(workspace, '_compute_' + mode)()
    assert '19,000,000,001' in error.call_args.args[1]
    dispatch.assert_not_called()
    confirm.assert_not_called()


def test_gui_cross_product_rejects_before_either_axis_allocation(workspace):
    workspace.angle_f_start_var.set('1')
    workspace.angle_f_stop_var.set('20')
    workspace.angle_f_step_var.set('0.001')
    workspace.angle_start_var.set('0')
    workspace.angle_stop_var.set('80')
    workspace.angle_step_var.set('0.01')
    with mock.patch('ibc.ui.make_frequency_sweep', side_effect=AssertionError('frequency allocated')), \
         mock.patch('ibc.ui.make_sweep', side_effect=AssertionError('angle allocated')), \
         mock.patch('ibc.ui.messagebox.showerror') as error:
        workspace._compute_off_angle()
    assert 'response points' in error.call_args.args[1]


def test_gui_inverse_rejects_before_allocating_axes(workspace):
    workspace.inv_freq_mode_var.set('Band')
    workspace.inv_target_start_var.set('1')
    workspace.inv_target_stop_var.set('20')
    workspace.inv_target_step_var.set('1e-9')
    with mock.patch('ibc.ui.make_frequency_sweep', side_effect=AssertionError('frequency allocated')), \
         mock.patch('ibc.ui.messagebox.showerror') as error, \
         mock.patch.object(workspace, '_run_background_task') as dispatch:
        workspace._run_inverse_design()
    assert '19,000,000,001' in error.call_args.args[1]
    dispatch.assert_not_called()
    with mock.patch('ibc.inverse_workflow.make_frequency_sweep', side_effect=AssertionError('frequency allocated')):
        with pytest.raises(ValueError, match='19,000,000,001'):
            workspace._inverse_setup_values()


def test_equal_angle_preserves_single_value_without_validating_unused_step(workspace):
    workspace.angle_f_start_var.set('1')
    workspace.angle_f_stop_var.set('2')
    workspace.angle_f_step_var.set('.5')
    workspace.angle_start_var.set('30')
    workspace.angle_stop_var.set('30')
    workspace.angle_step_var.set('not used')
    with mock.patch('ibc.ui.messagebox.showerror') as error, \
         mock.patch.object(workspace, '_confirm_output_replacements', return_value=True), \
         mock.patch.object(workspace, '_run_background_task') as dispatch:
        workspace._compute_off_angle()
    error.assert_not_called()
    dispatch.assert_called_once()


def test_prepared_empty_frequency_api_remains_compatible():
    layers = [LoadedLayer(.001, False, 0, ConstantMaterial(2 - .1j, 1), None),
              LoadedLayer(0, False, 0, None, None, True, 100)]
    assert compute.prepare_layer_properties_many([], layers) == [([], []), None]
    terms = compute.prepare_layer_wave_terms_many([], 0, layers, 'te')
    assert terms[1] is None
    assert all(len(values) == 0 for values in (*terms[0], terms[0].upper, terms[0].lower))
    assert compute.compute_stack_impedance_many([], layers, 'pec') == []
    assert all(values == [] for values in compute.compute_angle_metrics_many([], 0, layers, 'te').values())


def test_optional_heatmap_frequency_grid_is_admitted_before_list_creation():
    fake = SimpleNamespace(f_start_var=SimpleNamespace(get=lambda: '1'),
                           f_stop_var=SimpleNamespace(get=lambda: '20'),
                           f_step_var=SimpleNamespace(get=lambda: '.01'))
    with mock.patch('ibc.ui.make_frequency_sweep', side_effect=AssertionError('allocated')):
        with pytest.raises(ValueError, match='response points'):
            ImpedanceGui._compute_heatmap_data(fake, [], 'te', UnallocatedAxis(2000))


@pytest.mark.parametrize('oversized_axis', ['frequencies', 'angles'])
def test_coating_generator_is_bounded_before_existing_product_check(oversized_axis):
    from ibc.ghost_coating import assess_scalar_coating
    def values():
        yield from range(100001)
        raise AssertionError('Read beyond the bounded generator prefix')
    frequencies = values() if oversized_axis == 'frequencies' else [1.]
    angles = values() if oversized_axis == 'angles' else [0.]
    with mock.patch('ibc.ghost_coating.compute_stack_impedance_many', side_effect=AssertionError('computed')):
        with pytest.raises(ValueError, match='200,000'):
            assess_scalar_coating(frequencies, [object()], angles)
