"""Fresh runs must recompute, preserve fields and avoid all checkpoint I/O."""
from concurrent.futures import Future
from contextlib import ExitStack
import queue
import threading
import unittest
from unittest import mock

import numpy as np
from ghost_backend.execution import frequency_sweep as fs
from ghost_backend.execution.fresh_sweep import run_fresh
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.twod import checkpoints, solver
from ghost_backend.twod.samples import compact_samples
from test_pipeline_performance import small_result
from test_sweep_preparation import rectangle


class MemoryExecutor:
    payloads = []
    cancel = None
    def __init__(self, **kwargs):
        self.payloads.clear()
    def submit(self, function, payload):
        self.payloads.append(payload)
        result = small_result(payload['frequency'])
        profile = dict(stage_calls={'factorization':4},stage_seconds={'factorization':.2})
        result['metadata']['runtime_profile'] = profile
        future=Future()
        future.set_result(dict(frequency=payload['frequency'], result=result,
                               profile=profile, warning=None))
        if self.cancel is not None:
            self.cancel.set()
        return future
    def shutdown(self, **kwargs):
        pass


class FreshSweepTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(checkpoints, 'input_identity',
                                                  side_effect=AssertionError('cache fingerprint queried')))
        self.stack.enter_context(mock.patch.object(checkpoints, 'FrequencyCheckpoints',
                                                  side_effect=AssertionError('cache opened')))
        self.stack.enter_context(mock.patch.object(solver, '_solve_memory_limit_gb',return_value=4.))
        self.arguments=dict(geometry_snapshot={},frequencies_ghz=[1.,.6,.8],elevations_deg=[0.,90.])
        MemoryExecutor.cancel=None
    def parallel(self):
        self.stack.enter_context(mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=4))
        self.stack.enter_context(mock.patch.object(fs,'_solver_name',return_value='solve_monostatic_rcs_2d_certified'))
        self.stack.enter_context(mock.patch.object(fs,'ProcessPoolExecutor',MemoryExecutor))
        self.stack.enter_context(mock.patch.object(fs,'_close',side_effect=lambda e,s,f:e.shutdown()))
        self.stack.enter_context(mock.patch.object(fs,'_plan',side_effect=lambda a,o,c,f,w,b,cores:
            [dict(frequency=x,cpus=2,memory_gib=.75,cost=10.,selection=None) for x in sorted(f,reverse=True)]))
    def test_repeated_sequential_runs_recompute_every_frequency(self):
        solve=mock.Mock(side_effect=lambda **kw:small_result(kw['frequencies_ghz'][0]))
        for repeat in range(2):
            result=run_fresh(solve,self.arguments,{},'double',True,frequency_workers=1)
            self.assertEqual(solve.call_count,3*(repeat+1))
            self.assertNotIn('frequency_checkpoints',result['metadata'])
            self.assertFalse(result['metadata']['frequency_execution']['persistent_solve_cache'])
        self.assertEqual([r['frequency_ghz'] for r in result['samples']],[1.,1.,.6,.6,.8,.8])
    def test_repeated_parallel_runs_recompute_and_preserve_requested_order(self):
        self.parallel()
        solve=mock.Mock()
        for repeat in range(2):
            result=run_fresh(solve,self.arguments,{},'double',True,frequency_workers=2)
            self.assertEqual(len(MemoryExecutor.payloads),3)
            self.assertTrue(all(p['directory'] is None and p['identity'] is None for p in MemoryExecutor.payloads))
            self.assertEqual([r['frequency_ghz'] for r in result['samples']],[1.,1.,.6,.6,.8,.8])
            self.assertEqual(result['metadata']['runtime_profile']['stage_calls']['factorization'],12)
            self.assertEqual(result['metadata']['frequency_execution']['reused_frequencies'],0)
            self.assertNotIn('frequency_checkpoints',result['metadata'])
        solve.assert_not_called()
    def test_worker_returns_fields_without_writing_checkpoint(self):
        payload=dict(solver='solve_monostatic_rcs_2d_certified',frequency=.6,
            arguments=dict(geometry_snapshot={},elevations_deg=[0.]),frequencies=[.6,1.],
            options={},precision='double',cpus=1,memory_gib=2.,selection=None,
            directory=None,identity=None,certified=True)
        with mock.patch.object(fs,'_STOP',threading.Event()),mock.patch.object(fs,'_PROGRESS',queue.Queue()), \
                mock.patch.object(solver,'solve_monostatic_rcs_2d_certified',return_value=small_result(.6)):
            result=fs._compute(payload)
        self.assertIsNotNone(result['result'])
        self.assertIsNone(result['warning'])
    def test_cancel_then_restart_does_not_resume(self):
        self.parallel()
        event=threading.Event()
        args=dict(self.arguments,abort_event=event)
        MemoryExecutor.cancel=event
        with self.assertRaisesRegex(InterruptedError,'no result was published'):
            run_fresh(mock.Mock(),args,{},'double',True,frequency_workers=2)
        event.clear();MemoryExecutor.cancel=None
        result=run_fresh(mock.Mock(),args,{},'double',True,frequency_workers=2)
        self.assertEqual(len(MemoryExecutor.payloads),3)
        self.assertEqual(result['metadata']['frequency_execution']['computed_frequencies'],3)
    def test_retained_results_reduce_later_solve_memory(self):
        from ghost_backend.execution.options import allocated_memory_budget
        observed=[]
        def solve(**kw):
            observed.append(allocated_memory_budget())
            return small_result(kw['frequencies_ghz'][0])
        run_fresh(solve,self.arguments,{},'double',True,frequency_workers=1)
        self.assertGreater(observed[0],observed[1])
        self.assertGreater(observed[1],observed[2])
    def test_large_retained_results_fail_without_publishing_partial_result(self):
        solve=mock.Mock(side_effect=lambda **kw:small_result(kw['frequencies_ghz'][0]))
        with mock.patch.object(checkpoints,'_retained_bytes',side_effect=lambda v:0 if not v else 3*1024**3):
            with self.assertRaisesRegex(MemoryError,'Run a smaller'):
                run_fresh(solve,self.arguments,{},'double',True,frequency_workers=1)
        self.assertEqual(solve.call_count,1)
    def test_bor_repeated_runs_keep_duplicates_and_recompute(self):
        def solve(**kw):
            value=small_result(kw['frequencies_ghz'][0])
            value['solver']='bor_mom_rcs'
            return value
        called=mock.Mock(side_effect=solve)
        args=dict(self.arguments,frequencies_ghz=[.6,.8,.6])
        for repeat in range(2):
            result=run_fresh(called,args,{},'double',True,solver_kind='bor',frequency_workers=1)
            self.assertEqual(called.call_count,2*(repeat+1))
            self.assertEqual(len(result['samples']),6)
            self.assertEqual(result['metadata']['frequency_count'],3)
    def test_report_uses_total_fresh_wall_time(self):
        from ghost_backend.runs.quality import solver_report_text
        report=solver_report_text(dict(execution_wall_seconds=1.,runtime_profile=dict(wall_seconds=7.),
            frequency_execution=dict(persistent_solve_cache=False,computed_frequencies=3)))
        self.assertIn('Elapsed: 7.000 s',report)
        self.assertIn('Fresh calculation: 3 frequencies computed',report)


class SpawnedFreshFieldsTests(unittest.TestCase):
    def test_spawned_bor_recomputes_with_matching_complex_fields(self):
        from ghost_backend.bor import dispatch
        from test_bor_frequency_sweep import arguments
        args=arguments()
        outputs=[]
        with compact_samples(),execution_scope({},assembly_threads=4,memory_budget_gib=8.), \
                mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=4), \
                mock.patch.object(checkpoints,'FrequencyCheckpoints',side_effect=AssertionError('cache opened')):
            for workers in (1,2,2):
                outputs.append(run_fresh(dispatch.solve_monostatic_rcs_bor_survey,args,args['bor_options'],
                                         'double',False,solver_kind='bor',frequency_workers=workers))
        for result in outputs[1:]:
            self.assertEqual(result['metadata']['frequency_execution']['maximum_active_workers'],2)
            self.assertEqual(result['metadata']['frequency_execution']['computed_frequencies'],2)
            self.assertEqual(result['metadata']['frequency_execution']['reused_frequencies'],0)
            self.assertTrue(result['metadata']['runtime_profile']['stage_calls'])
            self.assertNotIn('frequency_checkpoints',result['metadata'])
            for pol in ('VV','HH'):
                for key in ('rcs_amp_real','rcs_amp_imag'):
                    np.testing.assert_allclose([r[key] for r in outputs[0]['co_solved_samples'][pol]],
                        [r[key] for r in result['co_solved_samples'][pol]],rtol=5e-12,atol=2e-14)

    def test_spawned_fresh_fields_match_sequential_without_checkpoint_files(self):
        args=dict(geometry_snapshot=rectangle(),frequencies_ghz=[1.,2.],
            elevations_deg=[0.,90.],geometry_units='meters',mesh_reference_ghz=1.,solver_method='experimental_cpu')
        options=validate_options(dict(factorization='dense',mesh_strategy='global',assembly_threads=1,blas_threads=1))
        outputs=[]
        with mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=2), \
                mock.patch.object(checkpoints,'FrequencyCheckpoints',side_effect=AssertionError('cache opened')):
            for workers in (1,2,2):
                with compact_samples(),execution_scope(options,memory_budget_gib=4.,assembly_threads=2):
                    outputs.append(run_fresh(solver.solve_monostatic_rcs_2d_survey,args,options,'double',False,
                                             frequency_workers=workers))
        for output in outputs[1:]:
            self.assertEqual(output['metadata']['frequency_execution']['maximum_active_workers'],2)
            self.assertGreater(output['metadata']['runtime_profile']['stage_calls']['factorization'],0)
            self.assertNotIn('frequency_checkpoints',output['metadata'])
            self.assertEqual([r['metadata']['panel_count'] for r in output['metadata']['frequency_metadata']],[66,66])
            for key in ('rcs_amp_real','rcs_amp_imag'):
                np.testing.assert_array_equal([r[key] for r in outputs[0]['samples']],
                                              [r[key] for r in output['samples']])


if __name__=='__main__':unittest.main()
