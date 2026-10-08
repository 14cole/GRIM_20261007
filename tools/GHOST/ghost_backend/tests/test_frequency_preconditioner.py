"""Opt-in inverse reuse never keeps old operators or weakens physical checks."""
import gc
import weakref
from unittest import mock
import numpy as np
import pytest

from ghost_backend.compressed.factor import CompressedFactor
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed import recycling
from ghost_backend.execution.options import execution_scope,validate_options
from ghost_backend.twod.preparation import preparation_scope
from test_compressed_path import Exact


def operator(a):
    return StreamedOperator(Exact(np.asarray(a,complex)),np.arange(len(a))[:,None],tile=64)


def factor(a,identity=b'mesh-region-dof-pol',frequency=10.,**kwargs):
    return CompressedFactor(operator(a),recycling_key=identity,recycling_frequency=frequency,**kwargs)


@pytest.fixture
def enabled():
    with execution_scope(validate_options(dict(frequency_preconditioner='reuse',ram_budget_gib=2.))),preparation_scope():
        yield


def test_default_off_retains_nothing():
    with execution_scope(validate_options({})),preparation_scope():
        f=factor(np.eye(16));f.solve(np.ones(16));f.retain_preconditioner()
        assert f.factor is not None
        assert recycling.cache() is None


def test_borrowed_inverse_solves_new_operator_and_old_operator_is_released(enabled):
    a=np.eye(80)*2.+np.ones((80,80))*.01
    f=factor(a);f.solve(np.ones((80,3)));old=weakref.ref(f.a);tree=weakref.ref(f.factor)
    f.retain_preconditioner();retained=recycling.live_bytes()
    assert retained>0 and f.factor is None
    del f;gc.collect();assert old() is None and tree() is not None
    b=np.arange(160).reshape(80,2)+1j
    new=a+np.eye(80)*.005;g=factor(new,frequency=10.05)
    assert g.recycled and g.event['factorizations']==0 and recycling.live_bytes()==0
    for trans,matrix in ((0,new),(1,new.T),(2,new.conj().T)):
        np.testing.assert_allclose(g.inverse(b,trans=trans),np.linalg.solve(matrix,b),rtol=1e-12,atol=1e-12)
    assert g.event['max_backward_error']<=1e-12
    evidence=g.event['frequency_preconditioner']
    assert evidence['original_wavenumber']==evidence['original_frequency_coordinate']==10.
    assert evidence['current_frequency_coordinate']==10.05
    assert evidence['coordinate_units']=='rad/m'


def test_hertz_recycling_evidence_does_not_mislabel_wavenumber(enabled):
    f=factor(np.eye(16),frequency=10e9,recycling_coordinate_units='Hz')
    f.solve(np.ones(16));f.retain_preconditioner()
    g=factor(np.eye(16),frequency=10.05e9,recycling_coordinate_units='Hz')
    assert g.recycled
    evidence=g.event['frequency_preconditioner']
    assert evidence['original_frequency_coordinate']==evidence['original_frequency_hz']==10e9
    assert evidence['current_frequency_coordinate']==10.05e9
    assert evidence['coordinate_units']=='Hz'
    assert 'original_wavenumber' not in evidence
    np.testing.assert_allclose(g.solve(np.ones(16)),1.,atol=1e-13,rtol=1e-13)


def test_failed_reuse_rebuilds_fresh_without_looser_error_gate(enabled):
    f=factor(np.eye(80));f.solve(np.ones(80));f.retain_preconditioner()
    new=np.diag(np.geomspace(1.,1000.,80));g=factor(new,frequency=10.05)
    assert g.recycled
    b=np.ones(80);result=g.solve(b)
    assert not g.recycled and g.event['factorizations']==1
    assert g.event['frequency_preconditioner']['rejected']
    np.testing.assert_allclose(result,1/np.diag(new),rtol=1e-12,atol=1e-14)
    assert g.event['max_backward_error']<=1e-12


@pytest.mark.parametrize('identity,frequency',[(b'different-region-order',10.05),(b'mesh-region-dof-pol',12.)])
def test_topology_and_frequency_changes_do_not_reuse(enabled,identity,frequency):
    f=factor(np.eye(16));f.solve(np.ones(16));f.retain_preconditioner()
    g=factor(np.eye(16),identity=identity,frequency=frequency)
    assert not g.recycled and g.event['factorizations']==1


def test_cache_is_byte_bounded_closes_and_does_not_keep_callback_state(enabled):
    class State:
        def check(self):pass
    state=State();ref=weakref.ref(state)
    f=factor(np.eye(16),checkpoint=state.check);f.solve(np.ones(16));f.retain_preconditioner()
    owner=recycling.cache();assert 0<owner.bytes<=owner.capacity<=recycling.MAX_BYTES
    del f,state;gc.collect();assert ref() is None
    owner.close();assert owner.bytes==0 and not owner.entries


def test_remaining_cache_is_subtracted_from_inverse_storage_budget(enabled):
    f=factor(np.eye(16),identity=b'other-polarization');f.solve(np.ones(16));f.retain_preconditioner()
    retained=recycling.live_bytes();new=operator(np.eye(16))
    g=CompressedFactor(new,storage_budget_bytes=1000000)
    assert g.budget==1000000-new.bytes-retained


def test_cancelled_reused_solve_propagates_without_rebuild_or_cache(enabled):
    f=factor(np.eye(16));f.solve(np.ones(16));f.retain_preconditioner()
    g=factor(np.eye(16),frequency=10.05)
    g.checkpoint=mock.Mock(side_effect=InterruptedError('cancelled'))
    with pytest.raises(InterruptedError):g.solve(np.ones(16))
    assert g.event['factorizations']==0 and recycling.live_bytes()==0


def test_reused_solves_do_not_train_cold_timing_history():
    from ghost_backend.execution.timing_history import _clean_success
    assert not _clean_success(dict(compressed_factors=[dict(frequency_preconditioner=dict(reused=True))]),'compressed')


def test_rejected_borrow_is_released_before_replacement_build():
    """Other cached inverses can consume room after a candidate is removed."""
    class Candidate:
        bytes = 500
    references = []
    def take(*args):
        candidate = Candidate()
        candidate.payload = np.zeros(1000)
        references.append(weakref.ref(candidate))
        return candidate, 10.
    def build():
        gc.collect()
        assert references[0]() is None
    new = operator(np.eye(16, dtype=complex))
    with mock.patch.object(recycling, 'take', side_effect=take), \
            mock.patch.object(recycling, 'live_bytes', return_value=600), \
            mock.patch.object(CompressedFactor, '_fresh_build', side_effect=build) as replacement:
        CompressedFactor(new, recycling_key=b'candidate', recycling_frequency=10.05,
                         storage_budget_bytes=new.bytes+1000)
    replacement.assert_called_once()


def test_cache_evicts_when_live_capacity_shrinks(enabled):
    f = factor(np.eye(16, dtype=complex))
    f.solve(np.ones(16))
    old = weakref.ref(f.factor)
    f.retain_preconditioner()
    owner = recycling.cache()
    assert owner.bytes > 0 and old() is not None
    with mock.patch.object(recycling, 'capacity_bytes', return_value=owner.bytes//2):
        assert recycling.live_bytes() == 0
        gc.collect()
        assert old() is None
        assert not owner.entries


@pytest.mark.parametrize('analytic',[False,True])
def test_dense_forecast_accounts_for_cache_even_after_backend_change(analytic):
    from ghost_backend.twod import solver
    with execution_scope(validate_options(dict(factorization='dense'))):
        with mock.patch.object(recycling,'capacity_bytes',return_value=0):
            before=solver._estimate_memory_gb(80,False,dense_resources={'analytic_zero':analytic})
        with mock.patch.object(recycling,'capacity_bytes',return_value=128*1024**2):
            after=solver._estimate_memory_gb(80,False,dense_resources={'analytic_zero':analytic})
    assert after-before==pytest.approx(.125)


def test_capacity_query_evicts_before_forecast_uses_a_smaller_reservation(enabled):
    f=factor(np.eye(16));f.solve(np.ones(16));f.retain_preconditioner()
    owner=recycling.cache();previous=owner.bytes
    with mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb',return_value=previous/(.1*1024**3)):
        capacity=recycling.capacity_bytes()
        assert capacity==previous//2
        assert owner.bytes==0


def test_two_large_polarizations_do_not_evict_each_other_before_reuse(enabled):
    first=factor(np.eye(16),identity=b'TE');first.solve(np.ones(16));first.retain_preconditioner()
    owner=recycling.cache();retained=owner.bytes
    with mock.patch.object(recycling,'capacity_bytes',return_value=retained+16):
        other=factor(np.eye(16),identity=b'TM');other.solve(np.ones(16));other.retain_preconditioner()
        assert other.factor is not None
        following=factor(np.eye(16),identity=b'TE',frequency=10.05)
        assert following.recycled


def test_tight_storage_cap_drops_optional_cache_before_building(enabled):
    f=factor(np.eye(16),identity=b'other');f.solve(np.ones(16));f.retain_preconditioner()
    assert recycling.live_bytes()>0
    new=operator(np.eye(16));g=CompressedFactor(new,storage_budget_bytes=new.bytes+6000)
    assert recycling.live_bytes()==0
    assert g.event['factorizations']==1


def test_concurrent_modes_transfer_each_cached_tree_to_only_one_owner():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    owner=recycling.InverseCache(2**20)
    tree=SimpleNamespace(inverse_only=True,bytes=1024,root=SimpleNamespace(leaf=True))
    assert owner.put(b'mode',10.,tree)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:owner.take(b'mode',10.01,2**20),range(32)))
    assert sum(result is not None for result in results)==1
    assert owner.bytes==0 and not owner.entries


def test_larger_opt_in_capacity_still_obeys_five_percent_reservation():
    with execution_scope(validate_options(dict(frequency_preconditioner='reuse',ram_budget_gib=12.))),preparation_scope():
        assert recycling.capacity_bytes()==512*1024**2
    with execution_scope(validate_options(dict(frequency_preconditioner='reuse',ram_budget_gib=2.))),preparation_scope():
        assert recycling.capacity_bytes()==int(.05*2*1024**3)
