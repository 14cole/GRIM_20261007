"""Nearby backend timing reuse requires paired evidence, guards and live RAM admission."""
from unittest import mock
import time

import pytest

from ghost_backend.execution import timing_history as history


def key(name, frequency=1., angles=101, family='same-geometry-host-source-settings'):
    return history.RequestKey(name,dict(family=family,frequency_ghz=frequency,angle_count=angles))


def choice(unknowns=1000):
    return dict(selected='dense',retry_order=['compressed'],admission_budget_gib=8,
        meshes=[dict(unknowns=unknowns)],candidates=dict(
            dense=dict(cost=10.,peak_gb=4.),compressed=dict(cost=14.,peak_gb=1.)))


def train(request, dense=(10.,10.1,9.9), compressed=(4.,4.1,3.9), unknowns=1000):
    for mode,values in [('dense',dense),('compressed',compressed)]:
        for seconds in values:
            history.record(request,mode,seconds,dict(quality_gate=dict(passed=True),dense_largest_system=unknowns))


@pytest.fixture(autouse=True)
def private_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR',str(tmp_path))


def test_nearby_paired_measurements_can_replace_prior_without_changing_ram_forecasts():
    train(key('source'))
    initial=choice(1010)
    revised=history.adjust(initial,key('nearby',1.01,103))
    assert revised['selected']=='compressed'
    assert revised['timing_model']=='nearby_request_host_paired_v1'
    assert revised['nearby_timing_evidence']['source_request']=='source'
    assert initial['selected']=='dense'
    assert revised['candidates']['compressed']['peak_gb']==1.
    assert revised['candidates']['dense']['peak_gb']==4.


@pytest.mark.parametrize('target,size', [
    (key('too far frequency',1.03),1000),
    (key('too many angles',1.,107),1000),
    (key('too many unknowns'),1030),
    (key('changed material/source/host',family='different'),1000),
])
def test_nearby_reuse_rejects_workload_changes(target,size):
    train(key('source'))
    assert history.adjust(choice(size),target)['selected']=='dense'


@pytest.mark.parametrize('dense,compressed', [
    ((10.,10.),(4.,4.)),                 # Too few repeats for interpolation.
    ((8.,10.,12.),(4.,4.,4.)),           # Noisy measurements.
    ((10.,10.,10.),(8.,8.,8.)),          # Too little winning margin.
])
def test_nearby_reuse_rejects_weak_or_noisy_evidence(dense,compressed):
    train(key('source'),dense,compressed)
    assert history.adjust(choice(),key('target',1.01))['selected']=='dense'


def test_two_unpaired_backend_requests_do_not_make_calibration():
    train(key('dense-only'),compressed=())
    train(key('compressed-only',1.005),dense=())
    assert history.adjust(choice(),key('target',1.01))['selected']=='dense'


def test_conflicting_nearby_winners_leave_prior_intact():
    train(key('compressed-wins'))
    train(key('dense-wins',1.005),dense=(2.,2.,2.),compressed=(8.,8.,8.))
    assert history.adjust(choice(),key('target',1.01))['selected']=='dense'


def test_expired_and_changed_mesh_samples_do_not_transfer():
    with mock.patch.object(history.time,'time',return_value=time.time()-history.MAX_AGE-1):
        train(key('old'))
    assert history.adjust(choice(),key('target',1.01))['selected']=='dense'
    train(key('source'))
    # A changed adaptive outcome cannot relabel earlier timings with new DOFs.
    history.record(key('source'),'compressed',4.,dict(quality_gate=dict(passed=True),dense_largest_system=1010))
    assert history.adjust(choice(),key('target',1.01))['selected']=='dense'


def test_same_exact_key_cannot_relabel_old_timings_with_new_reservation_family():
    train(key('source'))
    request=key('source',family='different-reservation')
    history.record(request,'dense',10.,dict(quality_gate=dict(passed=True),dense_largest_system=1000))
    target=key('target',1.01,family='different-reservation')
    assert history.adjust(choice(),target)['selected']=='dense'
    # The existing exact-request series is not replaced by nearby bookkeeping.
    assert history.measured_costs('source')['compressed']==4.


def test_exact_history_still_overrides_nearby_and_worker_ram_is_preserved():
    train(key('source'),dense=(2.,2.,2.),compressed=(8.,8.,8.))
    initial=choice();initial['selected']='compressed'
    assert history.adjust(initial,key('target',1.01),batch=True)['selected']=='compressed'
    train(key('target',1.01),dense=(10.,10.),compressed=(3.,3.))
    revised=history.adjust(choice(),key('target',1.01))
    assert revised['selected']=='compressed'
    assert revised['timing_model']=='identical_request_host_median_v1'


@pytest.mark.parametrize('failure', [
    dict(backend_selection=dict(failed_attempts=[dict(backend='compressed')])),
    dict(adaptive_mesh=dict(fallback=True)),
    dict(adaptive_mesh=dict(fallback=False, conservative_retry=dict(from_coarsening=8., to_coarsening=4.))),
    dict(dense_fallback_reasons=['hierarchical solve rejected']),
    dict(compressed_factors=[dict(coarse_rejection='stalled')]),
    dict(hierarchical_factors=[dict(builds=2)]),
    dict(hierarchical_factors=dict(VV=[dict(coarse_rejection='rank')],HH=[])),
    dict(channel_metadata=dict(VV=dict(compressed_factors=[dict(compact_preconditioner='RAM')]))),
    dict(frequency_metadata=[dict(metadata=dict(backend_selection=dict(failed_attempts=[{}])))]),
    dict(frequency_metadata=[dict(metadata={'adaptive_mesh': {'conservative_retry': {'reason': 'quality'}}})]),
])
def test_retried_or_fallback_runs_never_train(failure):
    metadata=dict(quality_gate=dict(passed=True),dense_largest_system=1000,**failure)
    history.record(key('invalid'),'dense',10.,metadata)
    assert not history.read()


def test_descriptor_keeps_geometry_settings_and_irregular_angles_exact():
    payload=dict(geometry={'segments':[[0,0,1,0]]},frequencies_ghz=[1.],
                 elevations_deg=[0.,45.,90.],options={'mesh_strategy':'global'})
    args=dict(solver_method='experimental_cpu')
    original=history._nearby_descriptor(payload,{},args)
    changed=dict(payload,frequencies_ghz=[1.01],elevations_deg=[0.,30.,60.,90.])
    assert history._nearby_descriptor(changed,{},args)['family']==original['family']
    for change in [dict(geometry={'segments':[[0,0,2,0]]}),
                   dict(elevations_deg=[0.,40.,90.]),
                   dict(options={'mesh_strategy':'local'})]:
        assert history._nearby_descriptor(dict(payload,**change),{},args)['family']!=original['family']
    assert history._nearby_descriptor(payload,{},dict(solver_method='direct'))['family']!=original['family']
    assert history._nearby_descriptor(dict(payload,frequencies_ghz=[1.,1.01]),{},args) is None


def test_nearby_does_not_cross_blas_or_hierarchy_thresholds():
    train(key('source'),unknowns=1024)
    assert history.adjust(choice(1030),key('target',1.01))['selected']=='dense'
    with mock.patch.dict('os.environ',{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'10000'}):
        train(key('larger'),unknowns=9950)
        assert history.adjust(choice(10010),key('target',1.01))['selected']=='dense'


def test_exact_timing_identity_includes_inherited_fixed_mesh_frequencies():
    from ghost_backend.execution.options import validate_options
    from ghost_backend.twod.preparation import sweep_mesh_scope
    args=dict(geometry_snapshot=dict(segments=[],ibcs=[],dielectrics=[]),
              frequencies_ghz=[1.],elevations_deg=[0.,90.],mesh_reference_ghz=1.)
    with sweep_mesh_scope([1.,2.]):
        first=history.request_key(args,validate_options({}),'solve_monostatic_rcs_2d')
    with sweep_mesh_scope([1.,3.]):
        second=history.request_key(args,validate_options({}),'solve_monostatic_rcs_2d')
    assert first!=second
    assert first.nearby['family']!=second.nearby['family']


@pytest.mark.parametrize('kind', ['exact', 'nearby', 'absent'])
def test_one_snapshot_preserves_exact_nearby_and_uncalibrated_rankings(kind):
    if kind != 'absent':
        train(key('source'))
    request = key('source') if kind == 'exact' else key('target', 1.01)
    expected = history.adjust(choice(1010), request)
    entries = history.read()
    # A later cache change must not split one decision across different data.
    with mock.patch.object(history, 'read', side_effect=[entries, {}]) as read:
        actual = history.adjust(choice(1010), request)
    read.assert_called_once_with()
    assert actual == expected
    assert actual['selected'] == ('dense' if kind == 'absent' else 'compressed')
    assert actual['candidates']['dense']['peak_gb'] == 4.
