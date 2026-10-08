"""Physical parity and cache admission contracts for the shared solve pipeline."""
import copy
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.twod import solver
from ghost_backend.twod.preparation import preparation_scope, forecast_cache
from ghost_backend.execution import selection
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.twod.assembly import scatter
from test_experimental_cpu import fixture
from general_fixtures import fixture as general_fixture


def amplitudes(rows):
    return np.asarray([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in rows])


class BistaticReuseTests(unittest.TestCase):
    def arguments(self, snapshot, frequencies=None):
        return dict(geometry_snapshot=snapshot, frequencies_ghz=frequencies or [.6],
            incidence_angles_deg=[13., 80.], observation_angles_deg=[0., 49., 120.],
            geometry_units='meters', execution_options=dict(assembly_threads=1, blas_threads=1))

    def check_channels(self, arguments, combined):
        for channel, polarization in (('VV', 'TE'), ('HH', 'TM')):
            independent = solver.solve_bistatic_rcs_2d_single_polarization(
                **arguments, polarization=polarization, solver_method='direct')
            np.testing.assert_allclose(amplitudes(combined['co_solved_samples'][channel]),
                                       amplitudes(independent['samples']), rtol=3e-12, atol=1e-14)

    def test_non_sheet_mesh_preparation_is_shared_once_per_frequency(self):
        arguments = self.arguments(fixture('pec', 24), [.6, .8])
        with mock.patch.object(solver, '_build_linear_mesh_interface_aware',
                               wraps=solver._build_linear_mesh_interface_aware) as build, \
                mock.patch.object(solver.MaterialLibrary, 'from_entries',
                                  wraps=solver.MaterialLibrary.from_entries) as materials:
            combined = solver.solve_bistatic_rcs_2d(**arguments)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(materials.call_count, 1)
        self.assertTrue(combined['metadata']['co_solve_shared_discretization'])
        self.check_channels(arguments, combined)

    def test_sheet_meshes_remain_separate_for_polarizations(self):
        arguments = self.arguments(general_fixture('sheet', 24))
        with mock.patch.object(solver, '_build_linear_mesh_interface_aware',
                               wraps=solver._build_linear_mesh_interface_aware) as build:
            combined = solver.solve_bistatic_rcs_2d(**arguments)
        self.assertEqual(build.call_count, 2)
        self.assertEqual([call.kwargs.get('polarization') for call in build.call_args_list], ['TE', 'TM'])
        self.check_channels(arguments, combined)

    def test_paired_multiregion_assembly_preserves_both_physical_channels(self):
        arguments = self.arguments(fixture('coated', 24))
        with mock.patch.object(scatter, 'assemble_pair', wraps=scatter.assemble_pair) as pair:
            combined = solver.solve_bistatic_rcs_2d(**arguments)
        self.assertEqual(pair.call_count, 1)
        self.check_channels(arguments, combined)


class ForecastReuseTests(unittest.TestCase):
    def arguments(self):
        return dict(geometry_snapshot=fixture('pec', 24), frequencies_ghz=[.6],
                    elevations_deg=[0., 90.], geometry_units='meters', max_panels=100000)

    def test_hits_reprice_live_storage_and_readmit_without_rebuilding_mesh(self):
        options = validate_options(dict(factorization='adaptive', compressed_storage_mib=0))
        budget = [10.]
        def estimate(*args, **kwargs):
            from ghost_backend.linalg.hierarchical import factor_mode
            from ghost_backend.compressed.runtime import storage_budget
            return 6. if factor_mode() == 'dense' else 1. + storage_budget()/1024**3
        with preparation_scope(), mock.patch.object(solver, '_solve_memory_limit_gb', side_effect=lambda: budget[0]), \
                mock.patch.object(solver, '_estimate_memory_gb', side_effect=estimate), \
                mock.patch.object(solver, '_build_panels', wraps=solver._build_panels) as panels:
            first = selection.select_backend(self.arguments(), options)
            calls = panels.call_count
            self.assertEqual(first['selected'], 'dense')
            self.assertAlmostEqual(first['candidates']['compressed']['peak_gb'], 7.)
            first['meshes'].clear()  # callers cannot mutate a cached record
            budget[0] = 4.
            second = selection.select_backend(self.arguments(), options)
            self.assertTrue(second['forecast_reused'])
            self.assertEqual(second['selected'], 'compressed')
            self.assertAlmostEqual(second['candidates']['compressed']['peak_gb'], 3.4)
            self.assertTrue(second['meshes'])
            self.assertEqual(panels.call_count, calls)
            budget[0] = .5
            with self.assertRaises(MemoryError):
                selection.select_backend(self.arguments(), options)
            self.assertEqual(panels.call_count, calls)

    def test_changed_inputs_and_independent_runs_do_not_reuse_forecasts(self):
        options = validate_options(dict(factorization='adaptive'))
        arguments = self.arguments()
        with mock.patch.object(selection, '_forecast_backend', wraps=selection._forecast_backend) as forecast, \
                mock.patch.object(solver, '_solve_memory_limit_gb', return_value=10.):
            with preparation_scope():
                selection.select_backend(arguments, options)
                selection.select_backend(copy.deepcopy(arguments), options)
                self.assertEqual(forecast.call_count, 1)
                selection.select_backend(dict(arguments, elevations_deg=[0., 45.]), options)
                selection.select_backend(dict(arguments, frequencies_ghz=[.8]), options)
                changed = copy.deepcopy(arguments)
                changed['geometry_snapshot']['segments'][0]['properties'][1] = '2'
                selection.select_backend(changed, options)
                selection.select_backend(arguments, dict(options, far_quadrature_order=12))
                self.assertEqual(forecast.call_count, 5)
            self.assertIsNone(forecast_cache())
            with preparation_scope():
                selection.select_backend(arguments, options)
            self.assertEqual(forecast.call_count, 6)

    def test_effective_allocation_changes_reprice_the_reused_forecast(self):
        # The forecast records are built once per run; a different CPU/RAM
        # allocation reuses them and reprices cost and peaks under the current
        # one, equal to a fresh forecast (October 2026 audit, R-2D-7: the
        # sweep planner selects under each worker's share).
        options = validate_options(dict(factorization='adaptive', assembly_threads='auto'))
        patches = (mock.patch.object(solver, '_solve_memory_limit_gb', return_value=10.),
                   mock.patch('ghost_backend.execution.options.host_assembly_threads', return_value=8))
        with preparation_scope(), patches[0], patches[1],                 mock.patch.object(selection, '_forecast_backend', wraps=selection._forecast_backend) as forecast:
            with execution_scope(options, assembly_threads=1, memory_budget_gib=8.):
                first = selection.select_backend(self.arguments(), options)
                selection.select_backend(self.arguments(), options)
            self.assertEqual(forecast.call_count, 1)
            with execution_scope(options, assembly_threads=2, memory_budget_gib=8.):
                threads = selection.select_backend(self.arguments(), options)
            with execution_scope(options, assembly_threads=2, memory_budget_gib=4.):
                budget = selection.select_backend(self.arguments(), options)
            self.assertEqual(forecast.call_count, 1)
            self.assertFalse(first.get('forecast_reused', False))
            self.assertTrue(threads['forecast_reused'] and budget['forecast_reused'])
            self.assertNotEqual([c['cost'] for c in first['candidates'].values()],
                                [c['cost'] for c in threads['candidates'].values()])
        with preparation_scope(), patches[0], patches[1]:
            with execution_scope(options, assembly_threads=2, memory_budget_gib=4.):
                fresh = selection.select_backend(self.arguments(), options)
        self.assertEqual(budget['selected'], fresh['selected'])
        for mode, candidate in fresh['candidates'].items():
            self.assertAlmostEqual(budget['candidates'][mode]['cost'], candidate['cost'], places=12)
            self.assertAlmostEqual(budget['candidates'][mode]['peak_gb'], candidate['peak_gb'], places=12)

    def test_cached_plan_honors_event_and_checkpoint_cancellation(self):
        options = validate_options(dict(factorization='adaptive'))
        with preparation_scope(), mock.patch.object(solver, '_solve_memory_limit_gb', return_value=10.):
            selection.select_backend(self.arguments(), options)
            event = threading.Event()
            event.set()
            with self.assertRaises(InterruptedError):
                selection.select_backend(dict(self.arguments(), abort_event=event), options)
            with self.assertRaises(InterruptedError):
                selection.select_backend(self.arguments(), options,
                    checkpoint=mock.Mock(side_effect=InterruptedError('cancelled')))

    def test_setup_forecasts_frequency_units_reused_by_execution_requests(self):
        from ghost_backend.runs.setup import RunSetupMixin, two_d_request
        from ghost_backend.runs.quality import accuracy_target_policy
        record = two_d_request([.6, .8], [0., 90.], 'meters', False, 'standard', 'monostatic', [])
        record['execution_options'] = validate_options(dict(record['execution_options'],
            assembly_threads=1, blas_threads=1, mesh_strategy='global'))
        snapshot = fixture('pec', 24)
        with preparation_scope(), mock.patch.object(solver, '_solve_memory_limit_gb', return_value=10.), \
                mock.patch.object(selection, '_forecast_backend', wraps=selection._forecast_backend) as forecast:
            RunSetupMixin._run_setup_summary(None, snapshot, '', record)
            self.assertEqual(forecast.call_count, 2)
            result = selection.select_backend(dict(geometry_snapshot=snapshot,
                material_base_dir='', geometry_units='meters', frequencies_ghz=record['frequencies_ghz'],
                elevations_deg=record['angles_deg'], solver_method=record['solver_method'],
                max_panels=100000, mesh_convergence_policy=accuracy_target_policy('standard')),
                record['execution_options'], False)
            self.assertTrue(result['forecast_reused'])
            self.assertEqual(forecast.call_count, 2)
            for frequency in record['frequencies_ghz']:
                result = selection.select_backend(dict(geometry_snapshot=snapshot,
                    material_base_dir='', geometry_units='meters', frequencies_ghz=[frequency],
                    elevations_deg=record['angles_deg'], solver_method=record['solver_method'],
                    max_panels=100000, mesh_convergence_policy=accuracy_target_policy('standard')),
                    record['execution_options'], False)
                self.assertTrue(result['forecast_reused'])
            self.assertEqual(forecast.call_count, 2)


if __name__ == '__main__':
    unittest.main()
