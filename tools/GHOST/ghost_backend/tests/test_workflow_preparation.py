"""Checkpoint-aware admission and run-owned input reuse keep safety boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ghost_backend.twod import checkpoints
from ghost_backend.twod.preparation import preparation_scope
from ghost_backend.runs.setup import RunSetupMixin, two_d_request
from ghost_backend.runs.bor_setup import resource_summary
from test_pipeline_performance import small_result
from test_sweep_preparation import rectangle
from test_bor_run_setup import recipe


class CheckpointPreflightTests(unittest.TestCase):
    def test_probe_is_read_only_and_only_trusts_current_identity_and_digest(self):
        arguments = dict(geometry_snapshot={}, frequencies_ghz=[.6, .8], elevations_deg=[0.])
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / 'absent'
            self.assertEqual(checkpoints.missing_frequencies(arguments, directory, {}, 'double', True), [.6, .8])
            self.assertFalse(directory.exists())
            identity = checkpoints.input_identity(arguments, {}, 'double', True)
            store = checkpoints.FrequencyCheckpoints(directory, identity, True)
            store.save(.6, small_result(.6))
            self.assertEqual(checkpoints.missing_frequencies(arguments, directory, {}, 'double', True), [.8])
            changed = dict(arguments, elevations_deg=[90.])
            self.assertEqual(checkpoints.missing_frequencies(changed, directory, {}, 'double', True), [.6, .8])
            store._path(.6).write_bytes(b'corrupt')
            self.assertEqual(checkpoints.missing_frequencies(arguments, directory, {}, 'double', True), [.6, .8])

    def test_probe_cancellation_precedes_input_hashing(self):
        def cancel():
            raise InterruptedError('canceled')
        with mock.patch.object(checkpoints, 'input_identity') as identity:
            with self.assertRaises(InterruptedError):
                checkpoints.missing_frequencies({}, '.', {}, 'double', True, checkpoint=cancel)
        identity.assert_not_called()

    def test_complete_checkpoint_preflight_needs_no_solve_ram(self):
        value = two_d_request([.6, .8], [0.], 'meters', True, 'standard', 'monostatic', [])
        with preparation_scope(), mock.patch('ghost_backend.execution.selection.select_backend',
                side_effect=MemoryError('no new numerical solve fits')):
            summary = RunSetupMixin._run_setup_summary(None, rectangle(), None, value,
                forecast_frequencies=[])
        self.assertIn('no new solve forecast', summary)
        self.assertIn('2 frequencies', summary)

    def test_partial_checkpoint_preflight_only_forecasts_missing_frequency(self):
        value = two_d_request([.6, .8], [0.], 'meters', True, 'standard', 'monostatic', [])
        selection = dict(selected='dense', dense_peak_gib=.1, admission_budget_gib=1.)
        with preparation_scope(), mock.patch('ghost_backend.execution.selection.select_backend',
                return_value=selection) as forecast:
            RunSetupMixin._run_setup_summary(None, rectangle(), None, value, forecast_frequencies=[.8])
        self.assertEqual(forecast.call_count, 1)
        self.assertEqual(forecast.call_args.args[0]['frequencies_ghz'], [.8])

    def test_complete_checkpoints_do_not_skip_input_validation(self):
        value = two_d_request([.6], [0.], 'meters', True, 'standard', 'monostatic', [])
        with self.assertRaisesRegex(ValueError, 'no segments'):
            RunSetupMixin._run_setup_summary(None, {}, None, value, forecast_frequencies=[])

    def test_bor_cached_preflight_skips_both_planning_paths(self):
        from ghost_backend import run_local_bor
        path = Path(__file__).resolve().parents[1] / 'geometry/geometries/body.geo'
        snapshot, base = run_local_bor._load_snapshot(str(path))
        value = dict(recipe(), bor_options={'factorization': 'auto'})
        with mock.patch('ghost_backend.bor.dispatch.resolve_automatic_plan', side_effect=AssertionError('plan')), \
                mock.patch('ghost_backend.bor.dispatch.estimate_bor_resources', side_effect=AssertionError('forecast')):
            summary = resource_summary(snapshot, base, value, forecast_frequencies=[])
        self.assertIn('no new solve forecast', summary)
        self.assertIn('2 frequencies', summary)


class WorkflowInputReuseTests(unittest.TestCase):
    def test_bor_preview_reuses_parsed_materials_within_but_not_between_runs(self):
        from ghost_backend.bor.dispatch import estimate_bor_resources
        from ghost_backend.twod.solver import MaterialLibrary
        from test_bor_physics_regression import _pec_sphere_snapshot
        snapshot = _pec_sphere_snapshot(explicit_elements=-12)
        kwargs = dict(geometry_units='meters', workers=1, mesh_certification=False,
                      bor_options={'factorization': 'dense'})
        with mock.patch.object(MaterialLibrary, 'from_entries', wraps=MaterialLibrary.from_entries) as load:
            with preparation_scope():
                estimate_bor_resources(snapshot, .6, [0., 90.], **kwargs)
                changed = copy.deepcopy(snapshot)
                changed['segments'][0]['properties'][1] = '-14'
                estimate_bor_resources(changed, .8, [0., 90.], **kwargs)
                self.assertEqual(load.call_count, 1)
            estimate_bor_resources(snapshot, .6, [0., 90.], **kwargs)
            self.assertEqual(load.call_count, 2)

    def test_paired_input_check_deduplicates_only_identical_claims_at_one_boundary(self):
        from ghost_backend import run_local_bor
        unit = dict(geometry='example.geo', geometry_input_sha256='first')
        pair = [dict(unit, polarization=pol) for pol in ('VV', 'HH')]
        context = dict(geometry_units='meters')
        with mock.patch('ghost_backend.assembly.fields.geometry_input_fingerprint', return_value='first') as fingerprint:
            run_local_bor._verify_channel_inputs(pair, context)
            self.assertEqual(fingerprint.call_count, 1)
            # A fresh boundary checks the file again and catches intervening changes.
            fingerprint.return_value = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                run_local_bor._verify_channel_inputs(pair, context)
            self.assertEqual(fingerprint.call_count, 2)
        pair[1]['geometry_input_sha256'] = 'incompatible'
        with mock.patch('ghost_backend.assembly.fields.geometry_input_fingerprint', return_value='first'):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                run_local_bor._verify_channel_inputs(pair, context)

    def test_bor_reused_material_models_keep_numerical_warnings_local(self):
        from ghost_backend.bor import dispatch
        from ghost_backend.twod.solver import MaterialLibrary
        from test_bor_physics_regression import _pec_sphere_snapshot
        original = dispatch.solve_bor
        calls = []
        def warned_solve(*args, **kwargs):
            result = original(*args, **kwargs)
            if not calls:
                result.setdefault('warnings', []).append('first solve only')
            calls.append(1)
            return result
        arguments = dict(geometry_snapshot=_pec_sphere_snapshot(explicit_elements=-12),
            elevations_deg=[90.], geometry_units='meters', workers=1,
            bor_options={'factorization': 'dense'})
        with preparation_scope(), mock.patch.object(dispatch, 'solve_bor', side_effect=warned_solve), \
                mock.patch.object(MaterialLibrary, 'from_entries', wraps=MaterialLibrary.from_entries) as load:
            first = dispatch.solve_monostatic_rcs_bor_survey(frequencies_ghz=[.6], **arguments)
            second = dispatch.solve_monostatic_rcs_bor_survey(frequencies_ghz=[.8], **arguments)
            self.assertEqual(load.call_count, 1)
        self.assertIn('first solve only', first['metadata']['warnings'])
        self.assertNotIn('first solve only', second['metadata']['warnings'])

    def test_node_plan_reprices_live_resources_with_one_material_parse(self):
        from ghost_backend import run_hpc_bor_monostatic as driver
        from ghost_backend.twod.solver import MaterialLibrary
        from test_bor_physics_regression import _pec_sphere_snapshot
        from test_hpc_bor_resource_binding import manifest
        with mock.patch.object(MaterialLibrary, 'from_entries', wraps=MaterialLibrary.from_entries) as load:
            plan = driver._compute_resource_plan(dict(frequency_ghz=.6),
                _pec_sphere_snapshot(explicit_elements=-12), None, manifest(), 1, 8.)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(plan['cpu_reservation'], 1)
        self.assertLessEqual(plan['memory_reservation_gib'], 8.)

    def test_local_frequency_planning_shares_only_run_owned_inputs(self):
        from ghost_backend import run_local_bor as driver
        from ghost_backend.twod.solver import MaterialLibrary
        from test_bor_physics_regression import _pec_sphere_snapshot
        units = [dict(geometry='example.geo', geometry_stem='example', frequency_ghz=f,
            channel_units=[dict(polarization='VV')]) for f in (.6, .8)]
        with mock.patch.object(driver, '_load_snapshot', return_value=(
                _pec_sphere_snapshot(explicit_elements=-12), None)), \
                mock.patch.object(driver.hpc_scheduler, 'predict_bor_extent', return_value=(.3, .1)), \
                mock.patch.multiple(driver, GEOMETRY_UNITS='meters', WORKERS_PER_UNIT=1,
                                    MESH_CERTIFICATION=False, BOR_EXECUTION_OPTIONS={'factorization': 'dense'}), \
                mock.patch.object(MaterialLibrary, 'from_entries', wraps=MaterialLibrary.from_entries) as load:
            costs = driver._plan(units, [90.])
        self.assertEqual(len(costs), 2)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(all(unit['resource_estimate']['estimated_peak_gb'] > 0 for unit in units))


if __name__ == '__main__':
    unittest.main()
