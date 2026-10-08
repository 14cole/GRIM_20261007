"""Durable per-run recovery without reuse between independent solves."""
import base64
import copy
import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from ghost_backend.execution.recovery import RecoveryRun, DiskSamples, cleanup_exported, sha
from ghost_backend.execution.recovery_export import export_run
from ghost_backend.execution.fresh_sweep import run_fresh
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.execution import frequency_sweep as fs
from ghost_backend.twod import solver
from ghost_backend.twod.samples import compact_samples
from ghost_backend.io.grim import export_result_to_grim
from test_sweep_preparation import rectangle
from test_pipeline_performance import small_result


def arguments():
    return dict(geometry_snapshot=rectangle(), frequencies_ghz=[1.,2.],
        elevations_deg=[0.,90.], geometry_units='meters', solver_method='experimental_cpu')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def create(self, args=None):
        return RecoveryRun.create(self.root/'Run_Recovery',args or arguments(),{},'double',True)
    def test_unique_runs_and_coordinate_snapshot(self):
        args=arguments()
        first=self.create(args)
        args['geometry_snapshot']['segments'][0]['point_pairs'][0]['x1']+=1e-10
        second=self.create(args)
        self.assertNotEqual(first.directory,second.directory)
        self.assertNotEqual(first.request()['arguments']['geometry_snapshot'],args['geometry_snapshot'])
        first.save(1.,small_result(1.))
        self.assertEqual(second.completed(),[])
        self.assertTrue((first.directory/'inputs/geometry.geo').is_file())
    def test_material_original_changes_do_not_change_captured_worker_inputs(self):
        csv='frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n1000000000,3,-0.1,1,0\n2000000000,3,-0.1,1,0\n'
        material=self.root/'material.csv';material.write_text(csv)
        args=arguments();args['material_base_dir']=str(self.root)
        args['geometry_snapshot']['dielectrics']=[['1','material.csv']]
        run=self.create(args)
        material.write_text(csv.replace(',3,',',4,'))
        observed=[]
        def solve(**kw):
            from ghost_backend.twod.geometry import MaterialLibrary
            library=MaterialLibrary.from_entries([],kw['geometry_snapshot']['dielectrics'],kw['material_base_dir'])
            observed.append(library.get_medium(1,kw['frequencies_ghz'][0])[0].real)
            return small_result(kw['frequencies_ghz'][0])
        payload=dict(solver='solve_monostatic_rcs_2d_certified',frequency=1.,arguments=args,
            frequencies=[1.,2.],options={},precision='double',cpus=1,memory_gib=2.,selection=None,
            directory=None,identity=None,certified=True,recovery_directory=str(run.directory))
        with mock.patch.object(solver,'solve_monostatic_rcs_2d_certified',side_effect=solve), \
                mock.patch.object(fs,'_STOP',threading.Event()),mock.patch.object(fs,'_PROGRESS',queue.Queue()):
            returned=fs._compute(payload)
        self.assertEqual(observed,[3.])
        self.assertIsNone(returned['result'])
        self.assertEqual(run.completed(),[1.])
        later=self.create(args)
        self.assertIn(',4,',(later.directory/'inputs/material.csv').read_text())
    def test_modified_captured_inputs_stop_instead_of_saving(self):
        run=self.create()
        (run.directory/'inputs/geometry.geo').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Captured input changed'):
            run.save(1.,small_result(1.))
        self.assertEqual(run.completed(),[])
    def test_cancel_preserves_first_frequency_and_next_run_recomputes_it(self):
        event=threading.Event();args=dict(arguments(),abort_event=event);calls=[]
        def solve(**kw):
            calls.append(kw['frequencies_ghz'][0]);event.set()
            return small_result(calls[-1])
        first=self.create(args)
        with self.assertRaises(InterruptedError):
            run_fresh(solve,args,{},'double',True,frequency_workers=1,recovery=first)
        recovered=RecoveryRun.open(first.directory).result()
        self.assertTrue(recovered['metadata']['partial_result'])
        self.assertEqual(recovered['metadata']['remaining_frequencies_ghz'],[2.])
        self.assertIsInstance(recovered['samples'],DiskSamples)
        self.assertEqual(len(recovered['samples']),2)
        self.assertEqual(first.manifest['state'],'interrupted')
        event.clear();second=self.create(args)
        with self.assertRaises(InterruptedError):
            run_fresh(solve,args,{},'double',True,frequency_workers=1,recovery=second)
        self.assertEqual(calls,[1.,1.])
        self.assertNotEqual(first.directory,second.directory)
    def test_failed_completion_marker_does_not_publish_frequency(self):
        run=self.create();run.save(1.,small_result(1.))
        original=os.replace
        def replace(source,target):
            if str(target)==str(run.store._path(2.).with_suffix('.json')):
                raise OSError('disk full')
            return original(source,target)
        with mock.patch('ghost_backend.twod.checkpoints.os.replace',side_effect=replace):
            with self.assertRaisesRegex(OSError,'disk full'):
                run.save(2.,small_result(2.))
        self.assertEqual(run.completed(),[1.])
        self.assertEqual(run.result()['metadata']['remaining_frequencies_ghz'],[2.])
    def test_corrupt_output_is_excluded_and_lazy_reader_checks_again(self):
        run=self.create();run.save(1.,small_result(1.));run.save(2.,small_result(2.))
        view=run.result()
        run.store._path(1.).write_bytes(b'corrupt')
        with self.assertRaisesRegex(IOError,'missing or corrupt'):
            next(iter(view['samples']))
        self.assertEqual(run.completed(),[2.])
        self.assertTrue(run.result()['metadata']['partial_result'])
    def test_completed_outputs_are_not_retained_in_memory(self):
        args=arguments();args['frequencies_ghz']=list(np.arange(1.,21.))
        run=self.create(args)
        with mock.patch('ghost_backend.twod.checkpoints._retained_bytes',wraps=lambda v:0) as size:
            result=run_fresh(lambda **kw:small_result(kw['frequencies_ghz'][0]),args,{},'double',True,
                             frequency_workers=1,recovery=run)
        self.assertEqual(size.call_count,1)
        self.assertIsInstance(result['samples'],DiskSamples)
        self.assertIsNone(result['samples'].reader.value)
        self.assertEqual(len(result['samples']),40)

    def test_abrupt_process_exit_keeps_completed_frequency(self):
        run=self.create()
        script = '''import os, sys
sys.path.insert(0, sys.argv[2])
from test_pipeline_performance import small_result
from ghost_backend.execution.recovery import RecoveryRun
run=RecoveryRun.open(sys.argv[1])
run.save(1.,small_result(1.))
(run.directory/'frequencies/unfinished.writing').write_bytes(b'incomplete')
os._exit(23)
'''
        child=subprocess.run([sys.executable,'-B','-c',script,str(run.directory),str(Path(__file__).parent)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode,23,child.stderr)
        recovered=RecoveryRun.open(run.directory).result()
        self.assertEqual(recovered['metadata']['run_recovery']['completed'],1)
        self.assertEqual(recovered['metadata']['remaining_frequencies_ghz'],[2.])

    def test_cleanup_waits_for_every_view_and_preserves_changed_final_file(self):
        run=self.create();run.save(1.,small_result(1.));run.save(2.,small_result(2.))
        output=self.root/'verified.grim';output.write_bytes(b'final')
        run.status('exported',exports=[dict(path=str(output),sha256=sha(output))])
        first=run.result();second=run.result()
        cleanup_exported(run.directory)
        self.assertTrue(run.directory.exists())
        del first;gc.collect()
        self.assertEqual(len(second['samples']),4)
        self.assertTrue(run.directory.exists())
        output.write_bytes(b'changed')
        del second;gc.collect()
        self.assertTrue(run.directory.exists())
        output.write_bytes(b'final');cleanup_exported(run.directory)
        self.assertFalse(run.directory.exists())

    def test_failed_export_preserves_previous_file_and_recovery(self):
        run=self.create();run.save(1.,small_result(1.));run.save(2.,small_result(2.))
        output=self.root/'existing.grim';output.write_bytes(b'previous')
        value=run.result()
        with self.assertRaisesRegex(OSError,'disk full'):
            export_run(value,[str(output)],mock.Mock(side_effect=OSError('disk full')))
        self.assertEqual(output.read_bytes(),b'previous')
        self.assertEqual(run.completed(),[1.,2.])
        self.assertEqual(list(run.directory.glob('export_*')),[])


class NumericalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.options=validate_options(dict(factorization='dense',mesh_strategy='global',assembly_threads=1,blas_threads=1))
    def test_bor_preflight_prices_only_the_active_frequency_output(self):
        from ghost_backend.runs.setup import RunSetupMixin
        from ghost_backend.bor import dispatch
        from test_bor_run_setup import recipe
        from test_bor_frequency_sweep import arguments as bor_arguments
        value=recipe();value['bor_options']={'factorization':'auto'}
        with mock.patch.object(dispatch,'resolve_automatic_plan',return_value=('dense',None)) as choose, \
                mock.patch.object(dispatch,'estimate_bor_resources',return_value=dict(
                    estimated_peak_gb=1.,mesh_elements=20,active_mode_workers=1)) as estimate:
            RunSetupMixin._run_setup_summary(None,bor_arguments()['geometry_snapshot'],'',value,
                                            output_frequency_count=1)
        self.assertEqual(choose.call_args.args[0]['frequency_count'],1)
        self.assertTrue(all(call.kwargs['frequency_count']==1 for call in estimate.call_args_list))
    def test_real_spawned_2d_fields_and_streaming_export_match(self):
        args=arguments()
        run=RecoveryRun.create(self.root/'Run_Recovery',args,self.options,'double',False)
        with compact_samples(),execution_scope(self.options,memory_budget_gib=4.,assembly_threads=2), \
                mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=2):
            expected=run_fresh(solver.solve_monostatic_rcs_2d_survey,args,self.options,'double',False,frequency_workers=1)
            actual=run_fresh(solver.solve_monostatic_rcs_2d_survey,args,self.options,'double',False,frequency_workers=2,recovery=run)
        self.assertEqual(actual['metadata']['frequency_execution']['maximum_active_workers'],2)
        self.assertEqual(run.completed(),[1.,2.])
        for key in ('rcs_amp_real','rcs_amp_imag'):
            np.testing.assert_allclose([r[key] for r in actual['samples']],[r[key] for r in expected['samples']],rtol=1e-12,atol=1e-13)
        [reference]=export_result_to_grim(expected,str(self.root/'reference.grim'))
        with mock.patch.object(DiskSamples,'__iter__',side_effect=AssertionError('whole run read by exporter')):
            [written]=export_run(actual,[str(self.root/'joined.grim')],lambda value,path:export_result_to_grim(value,path))
        with np.load(reference,allow_pickle=False) as a,np.load(written,allow_pickle=False) as b:
            for key in ('azimuths','frequencies','polarizations','rcs_power','rcs_phase','rcs_amp_real','rcs_amp_imag'):
                np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-13) if a[key].dtype.kind not in 'US' else np.testing.assert_array_equal(a[key],b[key])
            meta=json.loads(str(b['solver_metadata_json']))
            self.assertEqual(len(meta['sample_diagnostics']),len(expected['samples']))
            inputs=json.loads(str(b['recovery_inputs_json']))
            self.assertEqual(base64.b64decode(inputs['files_base64']['geometry_snapshot.json']),
                             (run.directory/'inputs/geometry_snapshot.json').read_bytes())
        self.assertTrue(run.directory.exists())
        del actual;gc.collect()
        self.assertFalse(run.directory.exists())
    def test_bor_radar_grid_and_embedded_body_arrays_match(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PySide6.QtWidgets import QApplication
        from ghost_backend.ui.app import GhostWorkspace
        from ghost_backend.bor import dispatch
        from ghost_backend.assembly.fields import radar_grid_aspects
        from test_bor_frequency_sweep import arguments as bor_arguments
        app=QApplication.instance() or QApplication([])
        workspace=GhostWorkspace();self.addCleanup(workspace.close)
        args=bor_arguments();args['frequencies_ghz']=[.6,.7]
        grid=dict(azimuths_deg=[0.,90.,180.],elevations_deg=[0.],axis_az_deg=0.,axis_el_deg=0.,roll_deg=0.)
        args['elevations_deg']=radar_grid_aspects(grid['azimuths_deg'],grid['elevations_deg'],0.,0.).tolist()
        context=dict(solver_kind='bor',snapshot=args['geometry_snapshot'],units='meters',radar_grid=grid)
        run=RecoveryRun.create(self.root/'Run_Recovery',args,args['bor_options'],'double',False,solver_kind='bor',context=context)
        with compact_samples(),execution_scope({},assembly_threads=4,memory_budget_gib=8.), \
                mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=4), \
                mock.patch.object(dispatch,'estimate_bor_resources',wraps=dispatch.estimate_bor_resources) as estimate:
            expected=run_fresh(dispatch.solve_monostatic_rcs_bor_survey,args,args['bor_options'],'double',False,solver_kind='bor',frequency_workers=1)
            estimate.reset_mock()
            actual=run_fresh(dispatch.solve_monostatic_rcs_bor_survey,args,args['bor_options'],'double',False,solver_kind='bor',frequency_workers=2,recovery=run)
        self.assertTrue(estimate.called)
        self.assertTrue(all(call.kwargs['frequency_count']==1 for call in estimate.call_args_list))
        tab=workspace.solver_tab;tab.last_solve_context=context
        [reference]=tab._export_result_files(expected,str(self.root/'reference.grim'),source_path='',history='test')
        [written]=tab._export_result_files(actual,str(self.root/'joined.grim'),source_path='',history='test')
        with np.load(reference,allow_pickle=False) as a,np.load(written,allow_pickle=False) as b:
            for key in a.files:
                if key=='solver_metadata_json':continue
                if a[key].dtype.kind in 'fc':np.testing.assert_allclose(a[key],b[key],rtol=5e-12,atol=2e-14,err_msg=key)
                else:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            meta=json.loads(str(b['solver_metadata_json']))
            self.assertEqual(set(meta['metadata']['per_frequency']),{'0.6','0.7'})

    def test_real_desktop_worker_and_explicit_partial_recovery(self):
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        from ghost_backend.ui.app import GhostWorkspace
        from ghost_backend.ui.solver import _SolveWorker
        from ghost_backend.runs.setup import DEFAULT_QUALITY
        from test_experimental_cpu import fixture
        app=QApplication.instance() or QApplication([])
        workspace=GhostWorkspace();self.addCleanup(workspace.close)
        snapshot=fixture('pec',48)
        context=dict(solver_kind='2d',snapshot=snapshot,units='meters',uses_geometry_tab=False)
        worker=_SolveWorker(snapshot=snapshot,source_path='',base_dir='',frequencies=[.6,.8],
            elevations=[0.,90.],units='meters',quality_thresholds=DEFAULT_QUALITY,
            solver_method='experimental_cpu',mesh_certification=True,
            execution_options={'factorization':'adaptive'},recovery_root=str(self.root/'Run_Recovery'),
            recovery_context=context)
        completed,errors=[],[]
        worker.finished.connect(lambda result,path:completed.append(result));worker.error.connect(errors.append)
        worker.run()
        self.assertFalse(errors,errors);self.assertEqual(len(completed),1)
        self.assertTrue(completed[0]['metadata']['mesh_convergence_certified'])
        self.assertEqual(worker.recovery.completed(),[.6,.8])
        # An interrupted/corrupt second frequency is omitted, never filled.
        worker.recovery.store._path(.8).with_suffix('.json').unlink()
        tab=workspace.solver_tab
        with mock.patch.object(QFileDialog,'getOpenFileName',return_value=(str(worker.recovery.directory/'run.json'),'')), \
                mock.patch.object(QMessageBox,'warning') as warning:
            tab.btn_recover.click()
        warning.assert_not_called()
        self.assertIn('Recovered 1 of 2',tab.lbl_status.text())
        self.assertTrue(tab.last_result['metadata']['partial_result'])
        self.assertTrue(tab.btn_export.isEnabled())
        [written]=tab._export_result_files(tab.last_result,str(self.root/'partial.grim'),source_path='',history='test')
        with np.load(written,allow_pickle=False) as data:
            np.testing.assert_array_equal(data['frequencies'],[.6])
            meta=json.loads(str(data['solver_metadata_json']))
            self.assertTrue(meta['metadata']['partial_result'])
        self.assertEqual(RecoveryRun.open(worker.recovery.directory).manifest['state'],'partial_exported')
        self.assertTrue(worker.recovery.directory.exists())

    def test_bistatic_multiple_incidence_exports_match(self):
        args=arguments();args.pop('solver_method');args['incidence_angles_deg']=args.pop('elevations_deg')
        args['observation_angles_deg']=[0.,45.,90.]
        run=RecoveryRun.create(self.root/'Run_Recovery',args,self.options,'double',False)
        with compact_samples(),execution_scope(self.options,memory_budget_gib=4.):
            expected=run_fresh(solver.solve_bistatic_rcs_2d_survey,args,self.options,'double',False,frequency_workers=1)
            actual=run_fresh(solver.solve_bistatic_rcs_2d_survey,args,self.options,'double',False,frequency_workers=1,recovery=run)
        reference=export_result_to_grim(expected,str(self.root/'reference.grim'))
        from ghost_backend.ui.solver import _planned_export_paths
        written=export_run(actual,_planned_export_paths(actual,str(self.root/'joined.grim')),
                           lambda value,path:export_result_to_grim(value,path))
        self.assertEqual(len(written),2)
        for left,right in zip(reference,written):
            with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
                for key in ('frequencies','azimuths','rcs_amp_real','rcs_amp_imag'):
                    np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
