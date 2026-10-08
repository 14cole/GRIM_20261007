"""Bounded checkpoint failure recovery and BoR frequency resume contracts."""
import copy
import errno
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from ghost_backend.twod import checkpoints
from ghost_backend.bor.checkpoints import run_checkpointed as run_bor
from ghost_backend.bor.checkpoints import merge_frequency_results
from ghost_backend.execution.options import allocated_memory_budget
from ghost_backend.twod.samples import compact_samples
from test_pipeline_performance import small_result


def arguments(frequencies=(.6, .8, 1.)):
    return dict(geometry_snapshot={}, frequencies_ghz=list(frequencies), elevations_deg=[0., 90.])


def bor_result(frequency, certified=True):
    result = small_result(frequency, certified)
    result['solver'] = 'bor_mom_rcs'
    result['metadata'].update(frequency_count=1, aspect_count=2, elevation_count=2,
        output_aspect_count=2, residual_nonfinite_count=1,
        per_frequency=[dict(frequency_ghz=frequency, mode_converged=True)],
        quality_gate=dict(passed=True, values=dict(nonfinite_sample_count=1)),
        mesh_convergence=dict(passed=certified, polarizations=dict(VV=dict(
            sample_count=2, complex_rms_normalized=frequency / 100))))
    return result


class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.identity = mock.patch.object(checkpoints, 'input_identity', return_value='recovery-test')
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def run_synthetic(self, solve, args, directory):
        return checkpoints.run_checkpointed(solve, args, directory, {}, 'double', True)

    def test_reused_frequencies_decode_only_once_and_corruption_recomputes(self):
        solve = mock.Mock(side_effect=lambda **kw: small_result(kw['frequencies_ghz'][0]))
        with tempfile.TemporaryDirectory() as directory:
            self.run_synthetic(solve, arguments(), directory)
            solve.reset_mock()
            loads = []
            original = checkpoints.FrequencyCheckpoints.load
            def load(store, frequency):
                loads.append(frequency)
                return original(store, frequency)
            with mock.patch.object(checkpoints.FrequencyCheckpoints, 'load', load):
                result = self.run_synthetic(solve, arguments(), directory)
            self.assertEqual(loads, [.6, .8, 1.])
            solve.assert_not_called()
            self.assertEqual(result['metadata']['frequency_checkpoints']['reused'], 3)
            store = checkpoints.FrequencyCheckpoints(directory, 'recovery-test', True)
            store._path(.8).write_bytes(b'corrupt')
            self.run_synthetic(solve, arguments(), directory)
            self.assertEqual(solve.call_args.kwargs['frequencies_ghz'], [.8])
            self.assertEqual(solve.call_count, 1)

    def test_disk_full_continues_with_bounded_reserved_unsaved_results(self):
        allocations = []
        def solve(**kw):
            allocations.append(allocated_memory_budget())
            return small_result(kw['frequencies_ghz'][0])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=1.), \
                mock.patch.object(checkpoints.FrequencyCheckpoints, 'save', side_effect=OSError(errno.ENOSPC, 'full')):
            result = self.run_synthetic(solve, arguments(), directory)
        self.assertEqual(len(result['samples']), 6)
        self.assertFalse(result['metadata'].get('partial_result', False))
        self.assertIsNone(allocations[0])
        self.assertLess(allocations[2], allocations[1])
        self.assertLess(allocations[1], 1.)
        self.assertEqual(result['metadata']['frequency_checkpoints']['persisted'], 0)
        self.assertEqual(len(result['metadata']['frequency_checkpoints']['write_warnings']), 3)

    def test_large_unsaved_result_returns_explicit_partial_without_next_solve(self):
        solve = mock.Mock(side_effect=lambda **kw: small_result(kw['frequencies_ghz'][0]))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=.000001), \
                mock.patch.object(checkpoints.FrequencyCheckpoints, 'save', side_effect=OSError(errno.ENOSPC, 'full')):
            result = self.run_synthetic(solve, arguments(), directory)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(len(result['samples']), 2)
        self.assertTrue(result['metadata']['partial_result'])
        self.assertEqual(result['metadata']['remaining_frequencies_ghz'], [.8, 1.])
        self.assertEqual(result['metadata']['frequency_checkpoints']['persisted'], 0)

    def test_single_frequency_disk_full_preserves_complete_result(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(checkpoints.FrequencyCheckpoints, 'save', side_effect=OSError(errno.ENOSPC, 'full')):
            result = self.run_synthetic(lambda **kw: small_result(.6), arguments([.6]), directory)
        self.assertEqual(len(result['samples']), 2)
        self.assertFalse(result['metadata'].get('partial_result', False))
        self.assertIn('could not be saved', result['metadata']['warnings'][0])

    def test_reduced_budget_rejection_preserves_previously_unsaved_samples(self):
        solve = mock.Mock(side_effect=[small_result(.6), MemoryError('reservation exhausted')])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=1.), \
                mock.patch.object(checkpoints.FrequencyCheckpoints, 'save', side_effect=OSError(errno.ENOSPC, 'full')):
            result = self.run_synthetic(solve, arguments(), directory)
        self.assertEqual(len(result['samples']), 2)
        self.assertTrue(result['metadata']['partial_result'])
        self.assertEqual(result['metadata']['remaining_frequencies_ghz'], [.8, 1.])

    def test_bor_duplicate_frequency_request_keeps_both_result_sets(self):
        solve = mock.Mock(side_effect=lambda **kw: bor_result(kw['frequencies_ghz'][0]))
        with tempfile.TemporaryDirectory() as directory:
            result = run_bor(solve, arguments([.6, .6]), directory, {}, True)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(len(result['samples']), 4)
        self.assertEqual(result['metadata']['frequency_count'], 2)

    def test_changed_checkpoint_after_reuse_probe_still_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = checkpoints.FrequencyCheckpoints(directory, 'recovery-test', True)
            store.save(.6, small_result(.6))
            def solve(**kw):
                store._path(.6).write_bytes(b'concurrent corruption')
                return small_result(.8)
            with self.assertRaisesRegex(IOError, 'changed before result export'):
                self.run_synthetic(solve, arguments([.6, .8]), directory)

    def test_bor_cancel_resume_retains_only_completed_certified_frequencies(self):
        event, calls = threading.Event(), []
        def solve(**kw):
            frequency = kw['frequencies_ghz'][0]
            calls.append(frequency)
            if len(calls) == 1:
                event.set()
            return bor_result(frequency)
        args = dict(arguments(), abort_event=event)
        with tempfile.TemporaryDirectory() as directory, compact_samples():
            with self.assertRaises(InterruptedError):
                run_bor(solve, args, directory, {}, True)
            event.clear()
            result = run_bor(solve, args, directory, {}, True)
        self.assertEqual(calls, [.6, .8, 1.])
        self.assertEqual(result['metadata']['frequency_checkpoints']['reused'], 1)
        self.assertEqual(result['metadata']['frequency_count'], 3)
        self.assertEqual(result['metadata']['aspect_count'], 2)
        self.assertEqual(result['metadata']['residual_nonfinite_count'], 3)
        self.assertEqual(result['metadata']['quality_gate']['values']['nonfinite_sample_count'], 3)
        self.assertEqual(len(result['metadata']['per_frequency']), 3)
        self.assertEqual(result['metadata']['mesh_convergence']['polarizations']['VV']['sample_count'], 6)
        self.assertIn('worst-frequency', result['metadata']['mesh_convergence']['aggregation'])


class BorCheckpointIdentityTests(unittest.TestCase):
    def test_bor_source_and_options_are_part_of_identity(self):
        args = arguments()
        before = checkpoints.input_identity(args, {}, 'double', False, solver_kind='bor')
        changed_options = checkpoints.input_identity(args, {'near_refinement': 1}, 'double', False, solver_kind='bor')
        original = Path.read_bytes
        def read(path):
            value = original(path)
            return value + b'changed BoR kernel' if path.name == 'solver.py' and path.parent.name == 'bor' else value
        with mock.patch.object(Path, 'read_bytes', read):
            changed_source = checkpoints.input_identity(args, {}, 'double', False, solver_kind='bor')
        self.assertNotEqual(before, changed_options)
        self.assertNotEqual(before, changed_source)
        self.assertNotEqual(before, checkpoints.input_identity(args, {}, 'double', False))

    def test_bor_merge_preserves_numeric_and_channel_request_order(self):
        values = [bor_result(1.), bor_result(.6)]
        result = merge_frequency_results(iter(copy.deepcopy(values)), [1., .6])
        self.assertEqual([row['frequency_ghz'] for row in result['samples']], [.6, .6, 1., 1.])
        self.assertEqual([row['frequency_ghz'] for row in result['co_solved_samples']['VV']], [1., .6])

    def test_real_bor_survey_checkpoint_resume_preserves_fields(self):
        from ghost_backend.bor.dispatch import solve_monostatic_rcs_bor_survey
        from test_bor_physics_regression import _pec_sphere_snapshot
        args = dict(geometry_snapshot=_pec_sphere_snapshot(explicit_elements=-20),
                    frequencies_ghz=[.8, .6], elevations_deg=[90., 0., 180.],
                    geometry_units='meters', workers=1,
                    bor_options={'factorization': 'dense'})
        with tempfile.TemporaryDirectory() as directory, compact_samples():
            actual = run_bor(solve_monostatic_rcs_bor_survey, args, directory, args['bor_options'], False)
            fail = mock.Mock(side_effect=AssertionError('completed BoR frequency was recomputed'))
            restored = run_bor(fail, args, directory, args['bor_options'], False)
        self.assertEqual(list(actual['samples']), list(restored['samples']))
        for pol in ('VV', 'HH'):
            self.assertEqual(list(actual['co_solved_samples'][pol]), list(restored['co_solved_samples'][pol]))
        self.assertEqual(restored['metadata']['frequency_checkpoints']['reused'], 2)
        self.assertEqual(actual['metadata']['quality_gate'], restored['metadata']['quality_gate'])


if __name__ == '__main__':
    unittest.main()
