"""Checked block products and order projection against independent dense algebra."""
import os
from unittest import mock
import numpy as np
import pytest
from scipy import sparse

from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed.factor import CompressedFactor
from ghost_backend.compressed.projection import project_operator
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from test_compressed_path import Exact


def operator(n=640):
    rng=np.random.default_rng(572)
    u=rng.standard_normal((n,8))+1j*rng.standard_normal((n,8))
    v=rng.standard_normal((8,n))+1j*rng.standard_normal((8,n))
    a=4*np.eye(n)+u@v/n
    return a,StreamedOperator(Exact(a),np.arange(n)[:,None],tile=64)


@pytest.mark.parametrize('trans',[0,1,2])
def test_subblock_products_preserve_permuted_partial_tile_indices(trans):
    a,op=operator(160)
    rows=np.arange(3,151,3)[::-1];cols=np.r_[np.arange(0,53,2),np.arange(91,155,3)]
    rng=np.random.default_rng(184)
    b=rng.standard_normal((len(cols) if trans==0 else len(rows),9))+1j
    block=a[np.ix_(rows,cols)]
    expected=block if trans==0 else block.T if trans==1 else block.conj().T
    # A block product must not silently reconstruct through coefficient queries.
    with mock.patch.object(op,'_get',side_effect=AssertionError('dense reconstruction')):
        result=op.block_matmul(rows,cols,b,op.plan(rows),op.plan(cols),trans=trans)
    np.testing.assert_allclose(result,expected@b,rtol=2e-13,atol=2e-13)


def test_batched_inverse_keeps_original_operator_and_adjoint_quality():
    a,op=operator()
    rng=np.random.default_rng(52);b=rng.standard_normal((len(a),4))+1j
    before=(op.row_error.copy(),op.column_error.copy())
    with mock.patch.dict(os.environ,{'GHOST_COMPRESSED_INVERSE_BUILDER':'randomized'}):
        factor=CompressedFactor(op,diagnostics={})
    assert factor.factor.evidence['sampled_blocks']>=2
    assert factor.factor.evidence['inverse_builder']=='randomized'
    for trans,matrix in ((0,a),(1,a.T),(2,a.conj().T)):
        result=factor.inverse(b,trans=trans)
        np.testing.assert_allclose(matrix@result,b,rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(op.row_error,before[0])
    np.testing.assert_array_equal(op.column_error,before[1])
    assert factor.event['max_backward_error']<=1e-12


def test_rejected_sampled_build_uses_aca_without_weakening_the_field_gate():
    a,op=operator()
    with mock.patch.dict(os.environ,{'GHOST_COMPRESSED_INVERSE_BUILDER':'randomized'}), \
            mock.patch('ghost_backend.linalg.hierarchical.compress_sampled',
                       side_effect=HierarchicalRejected('unhelpful sampled range')):
        factor=CompressedFactor(op)
    assert factor.factor.evidence['sampled_fallbacks']>=2
    result=factor.solve(np.ones(len(a),complex))
    np.testing.assert_allclose(a@result,1.,rtol=1e-12,atol=1e-12)
    op.row_error.fill(1.)
    with pytest.raises(HierarchicalRejected):factor.inverse(np.ones(len(a),complex))


def test_order_projection_propagates_fine_coefficient_errors_and_uses_factors():
    a,fine=operator(96)
    rng=np.random.default_rng(963)
    # Real, non-interpolatory, signed sparse supports deliberately exercise
    # the absolute prolongation factors in the incoming error bound.
    p=sparse.csc_matrix((np.tile([1.,-.25,.4],32),
                         (np.arange(96),np.repeat(np.arange(32),3))),shape=(96,32))
    perturbation=(rng.standard_normal(a.shape)+1j*rng.standard_normal(a.shape))*1e-10
    true=a+perturbation
    fine.row_error+=np.sum(abs(perturbation),axis=1)
    fine.column_error+=np.sum(abs(perturbation),axis=0)
    with mock.patch.object(fine,'_get',side_effect=AssertionError('fine block reconstruction')):
        coarse=project_operator(fine,p,np.arange(32)[:,None],2**24,tile=16)
    expected=np.asarray(p.T @ true @ p)
    actual=coarse.matmul(np.eye(32,dtype=complex))
    difference=abs(expected-actual)
    assert np.all(difference.sum(axis=1)<=coarse.row_error+1e-12)
    assert np.all(difference.sum(axis=0)<=coarse.column_error+1e-12)
    assert coarse.evidence['geometry_coefficients']==0
    assert coarse.evidence['projected_from_unknowns']==96


def test_order_projection_rejects_bad_maps_and_short_budget_without_changing_fine():
    a,fine=operator(48)
    before=fine.matmul(np.ones(48))
    p=sparse.eye(48,format='csc')
    with pytest.raises(MemoryError):project_operator(fine,p,np.arange(48)[:,None],1)
    with pytest.raises(ValueError):project_operator(fine,sparse.csc_matrix((48,2)),np.arange(2)[:,None],2**20)
    np.testing.assert_array_equal(fine.matmul(np.ones(48)),before)


def test_regional_pair_projects_both_channels_and_consumes_retained_cubic_operators(tmp_path):
    from test_polynomial_pair import inputs
    from ghost_backend.twod.assembly import polynomial_pair as pp
    from ghost_backend.twod.assembly.session import AssemblySession,_SESSION
    from ghost_backend.twod.assembly.scatter import assemble_multi
    from ghost_backend.execution.options import execution_scope,validate_options
    from ghost_backend.compressed import runtime
    options=validate_options(dict(factorization='compressed',assembly_threads=1,ram_budget_gib=12.,
                                  temporary_directory=str(tmp_path)))
    with execution_scope(options):
        mesh,infos=inputs('mixed',count=48);fine=pp.cubic_mesh(mesh)
        references={pol:assemble_multi(mesh,infos,pol,8,8)[0] for pol in ('TE','TM')}
        session=AssemblySession();session.compressed_partner=(mesh,infos)
        with mock.patch.dict(os.environ,{'GHOST_POLYNOMIAL_PAIR':'auto'}), \
                _SESSION.override(session),pp.polynomial_pair_scope(3) as pair:
            coarse_te,_=runtime.regional(mesh,infos,'TE')
            assert len(pair.pending)==2
            retained={key[1]:value[0].operator for key,value in pair.pending.items()}
            assert pp.retained_bytes('compressed')==0
            assert all(not op.tiles for op in retained.values())
            coarse_tm,_=runtime.regional(mesh,infos,'TM')
            for pol,op in (('TE',coarse_te),('TM',coarse_tm)):
                result=op.matmul(np.eye(len(op),dtype=complex))
                np.testing.assert_allclose(result,references[pol],rtol=2e-9,atol=2e-12)
                assert op.evidence['construction']=='qualified_fine_galerkin_projection'
                found,_=runtime.regional(fine,infos,pol)
                assert found is retained[pol]
            assert not pair.pending
            assert len([e for e in pair.evidence if e['action']=='project_quadratic'])==1
    assert not list(tmp_path.glob('ghost-tm-*'))
    assert not list(tmp_path.glob('ghost-polynomial-*'))


def test_regional_pair_budget_decline_does_not_retain_fine_operators():
    from test_polynomial_pair import inputs
    from ghost_backend.twod.assembly import polynomial_pair as pp
    from ghost_backend.twod.assembly.session import AssemblySession,_SESSION
    from ghost_backend.execution.options import execution_scope,validate_options
    from ghost_backend.compressed.runtime import _projected_regional
    with execution_scope(validate_options(dict(factorization='compressed',ram_budget_gib=12.))):
        mesh,infos=inputs('mixed',count=48);session=AssemblySession();session.compressed_partner=(mesh,infos)
        with mock.patch.dict(os.environ,{'GHOST_POLYNOMIAL_PAIR':'auto'}), \
                _SESSION.override(session),pp.polynomial_pair_scope(3) as pair, \
                mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb',return_value=.001):
            assert _projected_regional(mesh,infos,'TE',8,8) is None
            assert not pair.pending and session.compressed_partner[0] is mesh
            assert pair.evidence[-1]['action']=='independent_assembly'


def test_projection_storage_rejection_recovers_independent_assembly(tmp_path):
    from test_polynomial_pair import inputs
    from ghost_backend.twod.assembly import polynomial_pair as pp
    from ghost_backend.twod.assembly.session import AssemblySession,_SESSION
    from ghost_backend.execution.options import execution_scope,validate_options
    from ghost_backend.compressed import runtime
    with execution_scope(validate_options(dict(factorization='compressed',ram_budget_gib=12.,
                                               temporary_directory=str(tmp_path)))):
        mesh,infos=inputs('mixed',count=48)
        session=AssemblySession();session.compressed_partner=(mesh,infos)
        with mock.patch.dict(os.environ,{'GHOST_POLYNOMIAL_PAIR':'auto'}), \
                _SESSION.override(session),pp.polynomial_pair_scope(3) as pair, \
                mock.patch('ghost_backend.compressed.projection.project_operator',
                           side_effect=MemoryError('actual retained ranks exceeded allowance')):
            coarse,_=runtime.regional(mesh,infos,'TE')
            partner,_=runtime.regional(mesh,infos,'TM')
            assert coarse.evidence.get('construction')!='qualified_fine_galerkin_projection'
            assert not pair.pending
            assert any(e.get('fallback')=='joint_storage_rejection' for e in pair.evidence)
            assert np.all(np.isfinite(coarse.matmul(np.ones(len(coarse)))))
            assert np.all(np.isfinite(partner.matmul(np.ones(len(partner)))))
    assert not list(tmp_path.glob('ghost-tm-*'))


def test_explicit_far_sampling_keeps_its_query_path():
    from ghost_backend.compressed.operator import reciprocal_enabled
    from ghost_backend.compressed.regional_coefficients import PreparedOracle
    from ghost_backend.execution.options import execution_scope,validate_options
    oracle=PreparedOracle.__new__(PreparedOracle)
    with mock.patch.dict(os.environ,{'GHOST_COMPRESSED_RECIPROCAL':'auto'}):
        with execution_scope(validate_options(dict(compressed_far_method='full'))):
            assert reciprocal_enabled(oracle)
        with execution_scope(validate_options(dict(compressed_far_method='verified_cur'))):
            assert not reciprocal_enabled(oracle)

