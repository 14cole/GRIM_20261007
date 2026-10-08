"""Work-aware timing calibration stays local, conservative, and RAM-admitted."""
import copy
from unittest import mock

import pytest

from ghost_backend.execution import policy, timing_history as history
from ghost_backend.execution.options import validate_options
from ghost_backend.execution.stage_timing import measured_stages, observation


def meshes(d=1000,degree=1,formulation='multi_region'):
    return [dict(unknowns=d,formulation=formulation,polynomial_degree=degree,polarization=pol,
                 resources=dict(nodes=d,system_dofs=d,operator_entries=2*d*d,
                                geometric_near_pairs=10*d,basis_width=degree+1,operator_matrices=6))
            for pol in ('TE','TM')]


def request(name,frequency=10.,family='geometry-material-host-library-source',angles=100):
    descriptor=dict(family=family,frequency_ghz=frequency,angle_count=angles)
    return history.RequestKey(name,descriptor,dict(descriptor,options=validate_options({})))


def selection(d=1000):
    return dict(selected='dense',retry_order=['compressed'],admission_budget_gib=8.,
                meshes=meshes(d),candidates=dict(dense=dict(cost=10.,peak_gb=4.),
                                                compressed=dict(cost=11.,peak_gb=2.)))


def metadata(mode,d=1000,scale=1.):
    stages=dict(factorization=3*scale,excitation=.2*scale,rhs_compression=1.*scale,
                linear_solve=.9*scale,rhs_solve=.8*scale,far_field=.3*scale)
    if mode=='dense':stages['operators']=4*scale
    else:stages.update(compressed_assembly=3.5*scale,compressed_memory_sampling=.5*scale,operators=3*scale)
    return dict(quality_gate=dict(passed=True),dense_largest_system=d,
        stage_cost_meshes=meshes(d),runtime_profile=dict(stage_seconds=stages),
        experimental_cpu=dict(systems=[dict(unknowns=d,factorizations=1) for _ in range(2)]))


@pytest.fixture(autouse=True)
def private_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR',str(tmp_path))


def train(mode,req=None,d=1000,scale=1.,repeat=3):
    for _ in range(repeat):history.record(req or request(mode),mode,10*scale,metadata(mode,d,scale))


def test_prior_has_separate_stage_work_and_actual_dense_factor_regime():
    r=meshes()[0]['resources']
    ordinary=policy.stage_work(r,100,'dense')
    more_near=policy.stage_work(dict(r,geometric_near_pairs=100000),100,'dense')
    assert more_near['assembly_near']>ordinary['assembly_near']
    assert policy.relative_cost(r,100,'compressed')>policy.relative_cost(r,100,'dense')
    assert policy.relative_cost(r,100,'compressed')!=1.4*policy.relative_cost(r,100,'dense')
    assert policy.stage_work(r,200,'dense')['rhs']==2*ordinary['rhs']
    threaded=policy.stage_work(r,100,'dense',4,4)
    assert threaded['factorization']==ordinary['factorization']/4
    with mock.patch.dict('os.environ',{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'1000'}):
        assert policy.stage_work(dict(r,system_dofs=999),100,'dense')['factor_regime']=='lu'
        assert policy.stage_work(r,100,'dense')['factor_regime']=='hodlr'
        bigger=policy.stage_work(dict(r,system_dofs=2000),100,'dense')
        assert bigger['factorization']<8*policy.stage_work(r,100,'dense')['factorization']
    with mock.patch.dict('os.environ',{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'0'}):
        assert policy.stage_work(dict(r,system_dofs=2000),100,'dense')['factor_regime']=='lu'


def test_stage_seconds_do_not_double_count_nested_timers():
    result=measured_stages(metadata('compressed'),'compressed',10.)
    assert result==dict(assembly=4.,factorization=3.,rhs=1.5,other=1.5)
    assert sum(result.values())==10.
    assert measured_stages(metadata('compressed'),'compressed',5.) is None


def test_work_threads_auto_respects_physical_core_budget():
    with mock.patch('ghost_backend.execution.options.allocated_cpu_budget',return_value=8), \
         mock.patch('ghost_backend.execution.options.blas_core_budget',return_value=2), \
         mock.patch('ghost_backend.execution.options.effective_assembly_threads',return_value=8):
        assert policy.work_threads(1500,{'blas_threads':'auto'})==(8,2)
        assert policy.work_threads(1500,{'blas_threads':6})==(8,6)


def test_work_scaling_uses_independent_successful_backend_samples_across_half_ghz_step():
    train('dense',request('dense-at-10',10.),scale=1.)
    train('compressed',request('compressed-at-10.1',10.1),scale=.35)
    # The original 2% proximity heuristic cannot transfer this 5% frequency step.
    with mock.patch.object(history, 'read', wraps=history.read) as read:
        revised=history.adjust(selection(1050),request('target',10.5))
    read.assert_called_once_with()
    assert revised['selected']=='compressed'
    assert revised['timing_model']=='matched_stage_rates_v1'
    evidence=revised['stage_timing_evidence']['predictions']
    assert evidence['compressed']['samples']==3
    assert evidence['dense']['stage_seconds']['factorization']==pytest.approx(3*1.05**3)
    assert revised['candidates']['dense']['peak_gb']==4.


@pytest.mark.parametrize('change', ['frequency','dofs','angles','family','formulation','basis','regime'])
def test_stage_model_rejects_unsupported_extrapolation(change):
    train('dense');train('compressed',scale=.35)
    target=request('target',10.5);chosen=selection(1050)
    if change=='frequency':target=request('target',12.)
    if change=='dofs':chosen=selection(1250)
    if change=='angles':target=request('target',10.5,angles=125)
    if change=='family':target=request('target',10.5,family='other-material-or-library')
    if change=='formulation':chosen['meshes']=meshes(1050,formulation='sheet')
    if change=='basis':chosen['meshes']=meshes(1050,degree=2)
    if change=='regime':
        with mock.patch.dict('os.environ',{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'1025'}):
            assert history.adjust(chosen,target)['selected']=='dense'
        return
    assert history.adjust(chosen,target)['selected']=='dense'


def test_stage_model_needs_both_backends_repetition_stability_and_winning_margin():
    train('compressed',scale=.35)
    assert history.adjust(selection(),request('target',10.5))['selected']=='dense'
    train('dense',repeat=2)
    assert history.adjust(selection(),request('target',10.5))['selected']=='dense'
    train('dense',repeat=1)
    assert history.adjust(selection(),request('target',10.5))['selected']=='compressed'
    train('compressed',scale=.9,repeat=5)
    assert history.adjust(selection(),request('target',10.5))['selected']=='dense'


def test_stage_model_preserves_worker_admission_even_for_clear_measured_winner():
    train('dense',scale=.3);train('compressed',scale=1.)
    chosen=selection();chosen['selected']='compressed'
    assert history.adjust(chosen,request('target',10.5),batch=True)['selected']=='compressed'


def test_bad_actual_system_counts_and_retries_never_train_stages():
    m=metadata('dense');m['experimental_cpu']['systems'][0]['unknowns']=999
    assert observation(request('counts'),'dense',10.,m) is None
    m=metadata('dense');m['backend_selection']=dict(failed_attempts=[dict(backend='compressed')])
    history.record(request('retried'),'dense',10.,m)
    assert not history.read()


def test_corrupt_stage_cache_and_unwritable_cache_are_optional(monkeypatch):
    train('dense');train('compressed',scale=.35)
    entries=history.read()
    entries['dense']['_stages']['dense'][0]['features']['work']['rhs']='bad'
    with mock.patch.object(history,'read',return_value=entries):
        assert history.adjust(selection(),request('target',10.5))['selected']=='dense'
    monkeypatch.setattr(history,'open',mock.Mock(side_effect=PermissionError('read-only')),raising=False)
    train('dense')  # Successful results remain usable when optional persistence fails.


@pytest.mark.parametrize('field',['frequency','unknowns'])
def test_nonfinite_stage_cache_cannot_bypass_locality_guards(field):
    train('dense');train('compressed',scale=.35)
    entries=history.read()
    for row in entries['compressed']['_stages']['compressed']:
        if field=='frequency':row['frequency_ghz']=float('nan')
        else:row['features']['unknowns'][0]=float('nan')
    with mock.patch.object(history,'read',return_value=entries):
        assert history.adjust(selection(),request('target',10.5))['selected']=='dense'


@pytest.mark.parametrize('state_key',['experimental_cpu','cpu_kernel_execution'])
def test_actual_execution_mesh_telemetry_trains_explicit_run(state_key):
    m=metadata('compressed')
    m['experimental_cpu']['stage_cost_meshes']=m.pop('stage_cost_meshes')
    m[state_key]=m.pop('experimental_cpu')
    assert observation(request('actual'),'compressed',10.,m) is not None
