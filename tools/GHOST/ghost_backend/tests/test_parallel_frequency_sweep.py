"""Frequency scheduling preserves bounded recovery and numerical contracts."""
from concurrent.futures import Future
from contextlib import ExitStack
import queue
import tempfile
import threading
import unittest
from unittest import mock

from ghost_backend.execution import frequency_sweep as fs
from ghost_backend.execution.options import allocated_cpu_budget, allocated_memory_budget
from ghost_backend.twod import checkpoints
from ghost_backend.twod.preparation import mesh_frequencies
from test_pipeline_performance import small_result

REAL_PLAN = fs._plan


class ImmediateExecutor:
    payloads = []
    save_failure = False
    cancel_after = None

    def __init__(self, **kwargs):
        self.payloads.clear()

    def submit(self, function, payload):
        self.payloads.append(payload)
        result = small_result(payload['frequency'])
        result['metadata']['runtime_profile'] = dict(stage_seconds={'operators': .1},
            sampled_peak_process_rss_bytes=1234, sampled_peak_process_tree_rss_bytes=2345,
            sampled_peak_process_tree_private_bytes=3456)
        future = Future()
        if self.save_failure:
            future.set_result(dict(frequency=payload['frequency'], result=result,
                warning='disk full', profile=result['metadata']['runtime_profile']))
        else:
            store = checkpoints.FrequencyCheckpoints(payload['directory'], payload['identity'], True)
            store.save(payload['frequency'], result)
            future.set_result(dict(frequency=payload['frequency'], result=None,
                warning=None, profile=result['metadata']['runtime_profile']))
        if self.cancel_after is not None:
            self.cancel_after.set()
        return future

    def shutdown(self, **kwargs):
        pass


def arguments():
    return dict(geometry_snapshot={}, frequencies_ghz=[1., .6, .8], elevations_deg=[0., 90.])


class ParallelFrequencyTests(unittest.TestCase):
    def setUp(self):
        ImmediateExecutor.save_failure = False
        ImmediateExecutor.cancel_after = None
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(checkpoints, 'input_identity', return_value='parallel-test'))
        self.stack.enter_context(mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=4.))
        self.stack.enter_context(mock.patch('ghost_backend.execution.options.blas_core_budget', return_value=4))
        self.stack.enter_context(mock.patch.object(fs, '_solver_name', return_value='solve_monostatic_rcs_2d_certified'))
        self.stack.enter_context(mock.patch.object(fs, 'ProcessPoolExecutor', ImmediateExecutor))
        self.stack.enter_context(mock.patch.object(fs, '_plan', side_effect=self.plan))
        self.stack.enter_context(mock.patch.object(fs, '_close', side_effect=lambda executor, stop, failed: executor.shutdown()))
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())

    @staticmethod
    def plan(args, options, certified, frequencies, workers, budget, cores):
        return [dict(frequency=f, cpus=2, memory_gib=.75, cost=10*f, selection=None)
                for f in sorted(frequencies, reverse=True)]

    def run_sweep(self, args=None):
        solve = mock.Mock(side_effect=lambda **kw: small_result(kw['frequencies_ghz'][0]))
        result = checkpoints.run_checkpointed(solve, args or arguments(), self.directory,
            {}, 'double', True, frequency_workers=2)
        return result, solve

    def test_parallel_completion_merges_request_order_and_resumes(self):
        result, solve = self.run_sweep()
        solve.assert_not_called()
        self.assertEqual([r['frequency_ghz'] for r in result['samples']], [1., 1., .6, .6, .8, .8])
        self.assertEqual([p['frequency'] for p in ImmediateExecutor.payloads], [1., .8, .6])
        evidence = result['metadata']['frequency_execution']
        self.assertEqual(evidence['maximum_active_workers'], 2)
        self.assertLessEqual(evidence['peak_reserved_gib'], evidence['memory_budget_gib'])
        profile = result['metadata']['runtime_profile']
        self.assertIsNone(profile['sampled_peak_process_rss_bytes'])
        self.assertIsNone(profile['sampled_peak_process_tree_rss_bytes'])
        self.assertIsNone(profile['sampled_peak_process_tree_private_bytes'])
        self.assertEqual(profile['sampled_peak_frequency_worker_process_rss_bytes'], 1234)
        self.assertIn('not aggregate sweep memory', profile['memory_semantics'])
        resumed, solve = self.run_sweep()
        solve.assert_not_called()
        self.assertEqual(resumed['metadata']['frequency_checkpoints']['reused'], 3)

    def test_disk_failure_preserves_all_completed_samples(self):
        ImmediateExecutor.save_failure = True
        result, solve = self.run_sweep()
        solve.assert_not_called()
        self.assertEqual(len(result['samples']), 6)
        self.assertFalse(result['metadata'].get('partial_result', False))
        self.assertEqual(result['metadata']['frequency_checkpoints']['persisted'], 0)

    def test_unsaved_limit_stops_new_admissions_and_returns_ordered_partial(self):
        ImmediateExecutor.save_failure = True
        with mock.patch.object(checkpoints, '_retained_bytes', return_value=80*1024**2):
            result, solve = self.run_sweep()
        solve.assert_not_called()
        self.assertEqual(result['metadata']['remaining_frequencies_ghz'], [.6])
        self.assertEqual([r['frequency_ghz'] for r in result['samples']], [1., 1., .8, .8])
        self.assertTrue(result['metadata']['partial_result'])

    def test_cancel_keeps_workers_completed_checkpoint_files(self):
        event = threading.Event()
        ImmediateExecutor.cancel_after = event
        with self.assertRaises(InterruptedError):
            self.run_sweep(dict(arguments(), abort_event=event))
        store = checkpoints.FrequencyCheckpoints(self.directory, 'parallel-test', True)
        self.assertTrue(store.available(1.))
        self.assertFalse(store.available(.8))

    def test_cpu_and_memory_limits_decline_parallelism(self):
        with mock.patch('ghost_backend.execution.options.blas_core_budget', return_value=1):
            result, solve = self.run_sweep()
        self.assertEqual(solve.call_count, 3)
        self.assertNotIn('frequency_execution', result['metadata'])

    def test_auto_keeps_tiny_sweeps_in_process(self):
        def small_plan(*args):
            return [dict(row, cost=.01) for row in self.plan(*args)]
        solve = mock.Mock(side_effect=lambda **kw: small_result(kw['frequencies_ghz'][0]))
        with mock.patch.object(fs, '_plan', side_effect=small_plan):
            result = checkpoints.run_checkpointed(solve, arguments(), self.directory,
                {}, 'double', True, frequency_workers='auto')
        self.assertEqual(solve.call_count, 3)
        self.assertNotIn('frequency_execution', result['metadata'])

    def test_oversized_frequency_does_not_disable_two_smaller_workers(self):
        def select(args, options, certified):
            if args['frequencies_ghz'] == [1.]:
                raise MemoryError('large frequency needs the full parent reservation')
            return dict(selected='dense', retry_order=[], candidates=dict(
                dense=dict(cost=10*args['frequencies_ghz'][0], peak_gb=.6)))
        with mock.patch.object(fs, '_plan', REAL_PLAN), \
                mock.patch('ghost_backend.execution.selection.select_backend', side_effect=select):
            result, solve = self.run_sweep()
        self.assertEqual([p['frequency'] for p in ImmediateExecutor.payloads], [.8, .6])
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.kwargs['frequencies_ghz'], [1.])
        self.assertEqual(result['metadata']['frequency_execution']['maximum_active_workers'], 2)
        self.assertEqual([r['frequency_ghz'] for r in result['samples']], [1., 1., .6, .6, .8, .8])
        self.assertFalse(result['metadata'].get('partial_result', False))

    def test_worker_preserves_full_mesh_scope_and_allocation(self):
        payload = dict(solver='solve_monostatic_rcs_2d_certified', frequency=.6,
            arguments=dict(geometry_snapshot={}, elevations_deg=[0.]), frequencies=[.6, 1.],
            options={}, precision='double', cpus=2, memory_gib=2., selection=None,
            directory=self.directory, identity='worker-test', certified=True)
        observed = []
        def solve(**kw):
            observed.append((mesh_frequencies(kw['frequencies_ghz']),
                             allocated_cpu_budget(), allocated_memory_budget()))
            return small_result(.6)
        with mock.patch.object(fs, '_STOP', threading.Event()), mock.patch.object(fs, '_PROGRESS', queue.Queue()), \
                mock.patch('ghost_backend.twod.solver.solve_monostatic_rcs_2d_certified', side_effect=solve):
            result = fs._compute(payload)
        self.assertEqual(observed, [((.6, 1.), 2, 2.)])
        self.assertIsNone(result['result'])


class SpawnedNumericalContractTests(unittest.TestCase):
    def test_real_spawned_sweep_keeps_fixed_reference_mesh_and_fields(self):
        import numpy as np
        from ghost_backend.execution.options import execution_scope, validate_options
        from ghost_backend.twod import solver
        from ghost_backend.twod.samples import compact_samples
        from test_sweep_preparation import rectangle
        args = dict(geometry_snapshot=rectangle(), frequencies_ghz=[1., 2.],
            elevations_deg=[0., 90.], geometry_units='meters', mesh_reference_ghz=1.,
            solver_method='experimental_cpu')
        options = validate_options(dict(factorization='dense', mesh_strategy='global',
                                       assembly_threads=1, blas_threads=1))
        outputs = []
        with mock.patch('ghost_backend.execution.options.blas_core_budget', return_value=2):
            for workers in (1, 2):
                with tempfile.TemporaryDirectory() as directory, compact_samples(), \
                        execution_scope(options, memory_budget_gib=4., assembly_threads=2):
                    outputs.append(checkpoints.run_checkpointed(solver.solve_monostatic_rcs_2d_survey,
                        args, directory, options, 'double', False, frequency_workers=workers))
        self.assertEqual(outputs[1]['metadata']['frequency_execution']['maximum_active_workers'], 2)
        for output in outputs:
            counts = [r['metadata']['panel_count'] for r in output['metadata']['frequency_metadata']]
            self.assertEqual(counts, [66, 66])
        fields = [np.array([complex(r['rcs_amp_real'], r['rcs_amp_imag']) for r in out['samples']])
                  for out in outputs]
        np.testing.assert_array_equal(*fields)


class CheckpointPreparationLifecycleTests(unittest.TestCase):
    def test_direct_api_shares_resources_and_closes_once_on_success_or_cancel(self):
        from ghost_backend.twod.preparation import preparation_scope, run_resources
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(checkpoints, 'input_identity', return_value='lifecycle-test'):
                event = threading.Event()
                resource = mock.Mock()
                observed = []

                def solve(**kw):
                    # Production solvers enter their own nested preparation scope.
                    with preparation_scope():
                        resources = run_resources()
                        observed.append(resources.setdefault('test', resource))
                        resource.close.assert_not_called()
                        if cancel and len(observed) == 2:
                            event.set()
                        return small_result(kw['frequencies_ghz'][0])

                args = dict(arguments(), frequencies_ghz=[.6, .8], abort_event=event)
                if cancel:
                    with self.assertRaises(InterruptedError):
                        checkpoints.run_checkpointed(solve, args, directory, {}, 'double', True)
                else:
                    checkpoints.run_checkpointed(solve, args, directory, {}, 'double', True)
                self.assertEqual(observed, [resource, resource])
                resource.close.assert_called_once_with()
                self.assertIsNone(run_resources())

    def test_existing_outer_scope_keeps_resource_ownership(self):
        from ghost_backend.twod.preparation import preparation_scope, run_resources
        resource = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(checkpoints, 'input_identity', return_value='outer-lifecycle-test'):
            with preparation_scope():
                run_resources()['test'] = resource
                solve = mock.Mock(side_effect=lambda **kw: small_result(kw['frequencies_ghz'][0]))
                checkpoints.run_checkpointed(solve, dict(arguments(), frequencies_ghz=[.6, .8]),
                    directory, {}, 'double', True)
                self.assertIs(run_resources()['test'], resource)
                resource.close.assert_not_called()
            resource.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
