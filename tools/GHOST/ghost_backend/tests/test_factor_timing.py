"""Factor crossover training needs repeated matched full runs, with safe fallback."""
import copy
import time
from unittest import mock

import pytest

from ghost_backend.execution import factor_timing, timing_history as history
from ghost_backend.linalg import crossover, hierarchical


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('GHOST_HIERARCHICAL_MIN_UNKNOWNS', raising=False)


def request(variant, family='same-request-source-host-reservation'):
    return history.RequestKey(variant, factor=family)


def metadata(variant, seconds, count=2):
    return dict(quality_gate=dict(passed=True), experimental_cpu=dict(systems=[
        dict(unknowns=7500, factor_variant=variant, factor_work_seconds=seconds,
             factorizations=1, factor_rebuilds=0, factor_fallback=False)
        for _ in range(count)]))


def train(variant, samples, **kwargs):
    for seconds in samples:
        history.record(request(variant), 'dense', 10.*seconds, metadata(variant, seconds, **kwargs))


def selected(key=None):
    return factor_timing.choices(key or request('current'), history.read(), history.MAX_AGE)


def test_paired_runs_choose_factor_and_preserve_explicit_override(monkeypatch):
    train('lu', (5.,5.1)); train('hodlr', (3.,3.1))
    choices, evidence=selected()
    assert choices == {7500:'hodlr'}
    assert evidence['7500']['systems_per_run']==2
    assert evidence['7500']['samples']=={'lu':2,'hodlr':2}
    with crossover.scope(choices):
        assert hierarchical.automatic_hierarchical(7500)
        monkeypatch.setenv('GHOST_HIERARCHICAL_MIN_UNKNOWNS','10000')
        assert not hierarchical.automatic_hierarchical(7500)
    monkeypatch.delenv('GHOST_HIERARCHICAL_MIN_UNKNOWNS')
    assert not hierarchical.automatic_hierarchical(7500)


def test_polarizations_are_not_independent_repeat_measurements():
    train('lu',(5.,));train('hodlr',(3.,))
    assert selected()[0]=={}


@pytest.mark.parametrize('samples', [(4.85,4.86), (2.,4.)])
def test_small_or_noisy_wins_keep_prior(samples):
    train('lu',(5.,5.1));train('hodlr',samples)
    assert selected()[0]=={}


def test_different_workload_and_system_count_do_not_transfer():
    train('lu',(5.,5.1));train('hodlr',(3.,3.1))
    assert selected(request('changed',family='other-source-or-reservation'))[0]=={}
    train('hodlr',(3.,3.1),count=1)
    assert selected()[0]=={}


def test_expired_samples_do_not_transfer():
    with mock.patch.object(history.time,'time',return_value=time.time()-history.MAX_AGE-10):
        train('lu',(5.,5.1));train('hodlr',(3.,3.1))
    assert selected()[0]=={}


@pytest.mark.parametrize('field,value', [('factor_fallback',True),('factor_rebuilds',1),
                                      ('factor_work_failed',True),('factor_work_seconds',float('nan'))])
def test_failed_or_invalid_factor_never_trains(field,value):
    item=metadata('hodlr',3.)
    item['experimental_cpu']['systems'][0][field]=value
    history.record(request('bad'),'dense',30.,item)
    assert not history.read().get('bad',{}).get('_factorizations')


def test_joint_storage_recovery_is_not_clean_timing():
    item=metadata('lu',5.)
    item['adaptive_mesh']=dict(polynomial_pair=[dict(fallback='joint_storage_rejection')])
    history.record(request('bad'),'dense',50.,item)
    assert history.read()=={}


def test_descriptor_excludes_only_factor_choice_and_honors_memory_and_other_algorithms(monkeypatch):
    payload=dict(geometry='same',source='same',runtime_overrides={})
    options=dict(factorization='dense',ram_budget_gib=12.,assembly_threads=4)
    first=factor_timing.descriptor(payload,options)
    monkeypatch.setenv('GHOST_HIERARCHICAL_MIN_UNKNOWNS','5000')
    assert factor_timing.descriptor(payload,options)==first
    assert factor_timing.descriptor(payload,dict(options,ram_budget_gib=8.))!=first
    monkeypatch.setenv('GHOST_POLYNOMIAL_PAIR','off')
    assert factor_timing.descriptor(payload,options)!=first


def test_malformed_optional_history_is_ignored():
    malformed={'bad':dict(_factorizations=[None,{},dict(family=request('x').factor,
         time=time.time(),groups={'bad':{},'7500':dict(variant='lu',seconds=-1,systems=2)})])}
    assert factor_timing.choices(request('x'),malformed,history.MAX_AGE)==({}, {})


def test_install_updates_only_current_scope():
    with crossover.scope({}):
        crossover.install({7500:'hodlr'})
        assert crossover.chosen(7500)=='hodlr'
        with crossover.scope({7500:'lu'}):
            assert crossover.chosen(7500)=='lu'
        assert crossover.chosen(7500)=='hodlr'
    assert crossover.chosen(7500) is None


def test_learned_factor_policy_separates_whole_run_history_but_preserves_training():
    key=history.RequestKey('exact', nearby=dict(family='nearby', frequency_ghz=10.),
                           stage=dict(family='stage', options={'basis_order':3}), factor='factor')
    assert history.with_factor_choices(key,{}) is key
    changed=history.with_factor_choices(key,{7500:'hodlr'})
    assert changed!=key and changed.factor==key.factor
    assert changed.nearby['family']!=key.nearby['family']
    assert changed.stage['family']!=key.stage['family']
    assert changed.nearby['frequency_ghz']==10.
    assert changed.stage['options']==key.stage['options']
    assert history.with_factor_choices(key,{7500:'hodlr'})==changed
