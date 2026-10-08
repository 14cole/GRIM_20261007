"""Packed inverse products preserve physical algebra and bounded ownership."""
from unittest import mock
import os
import numpy as np
import pytest
from ghost_backend.compressed import block_products as bp
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed.inverse import CompressedSystem
from ghost_backend.execution.options import execution_scope, validate_options


def operator(n=2048):
    rng = np.random.default_rng(489)
    u = rng.normal(size=(n, 6))+1j*rng.normal(size=(n, 6))
    v = (rng.normal(size=(6, n))+1j*rng.normal(size=(6, n)))/n
    source = type('Source', (), dict(n=n))()
    op = StreamedOperator(source, np.arange(n)[:, None], tile=256, assemble=False, budget=2**28)
    for i, rows in enumerate(op.groups):
        for j, cols in enumerate(op.groups):
            op.tiles[i, j] = (5*np.eye(len(rows))+u[rows]@v[:, cols], None) if i == j else (u[rows], v[:, cols])
    return op, u, v


@pytest.mark.parametrize('trans', [0, 1, 2])
def test_packed_partial_permuted_products_match_independent_dense_algebra(trans):
    op, u, v = operator()
    rng = np.random.default_rng(81)
    rows, cols = rng.permutation(op.n)[:1123], rng.permutation(op.n)[:1151]
    a = u[rows]@v[:, cols]+5*(rows[:, None] == cols[None, :])
    x = rng.normal(size=(len(cols) if trans == 0 else len(rows), 17))+1j
    expected = (a if trans == 0 else a.T if trans == 1 else a.conj().T)@x
    with execution_scope(validate_options(dict(assembly_threads=4, blas_threads=4))), \
            mock.patch.object(bp, 'WORKSPACE_BYTES', 1024**2), bp.product_scope() as team:
        apply = bp.prepared_product(op)
        actual = apply(rows, cols, x, trans=trans)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
        again = apply(rows, cols, x, trans=trans)
        np.testing.assert_array_equal(actual, again)
        assert team.evidence['products'] == 2
        assert team.evidence['teams'] == 1
        assert team.evidence['chunks'] > 2
        assert team.evidence['peak_workspace_bytes'] <= 1024**2
    assert team.pool is None


def test_single_cpu_and_custom_semantics_do_not_start_team():
    op, _, _ = operator()
    rows, cols = op.order[:1024], op.order[1024:]
    x = np.ones((1024, 9), complex)
    with execution_scope(validate_options(dict(assembly_threads=1))), bp.product_scope() as team:
        expected = op.block_matmul(rows, cols, x)
        np.testing.assert_array_equal(bp.prepared_product(op)(rows, cols, x), expected)
        assert team.evidence['products'] == 0 and team.pool is None
    class Custom(StreamedOperator):
        def block_matmul(self, *args, **kwargs): return 'custom'
    with bp.product_scope() as team:
        assert bp.prepared_product(object.__new__(Custom))(rows, cols, x) == 'custom'
        assert team.pool is None


def test_reduced_allocation_does_not_reuse_a_larger_team():
    from ghost_backend.execution.options import cpu_allocation_scope
    op, _, _ = operator()
    rows, cols = op.order[:1024], op.order[1024:]
    x = np.ones((1024, 9), complex)
    with execution_scope(validate_options(dict(assembly_threads=4))), bp.product_scope() as team:
        apply = bp.prepared_product(op)
        expected = apply(rows, cols, x)
        before = team.evidence['products']
        with cpu_allocation_scope(1):
            np.testing.assert_allclose(apply(rows, cols, x), expected, rtol=1e-14, atol=1e-14)
        assert team.evidence['products'] == before


def test_cancellation_waits_for_running_groups_and_closes_pool():
    op, _, _ = operator()
    rows, cols = op.order[:1024], op.order[1024:]
    with execution_scope(validate_options(dict(assembly_threads=4))), pytest.raises(InterruptedError):
        with bp.product_scope() as team:
            apply = bp.prepared_product(op)
            apply(rows, cols, np.ones((1024, 8), complex))
            import threading
            def cancel():
                if threading.current_thread().name.startswith('ghost-inverse-products'):
                    raise InterruptedError('cancelled')
            op.checkpoint = cancel
            apply(rows, cols, np.ones((1024, 8), complex))
    assert team.pool is None


def test_inverse_scope_releases_team_and_preserves_adjoint_solve():
    op, u, v = operator()
    with execution_scope(validate_options(dict(assembly_threads=4, blas_threads=4))), \
            mock.patch.dict(os.environ, GHOST_COMPRESSED_INVERSE_BUILDER='randomized'):
        factor = CompressedSystem(op, op.coordinates, tolerance=1e-8, inverse_only=True)
    assert factor.evidence['block_products']['products'] > 0
    assert bp._ACTIVE.get() is None
    rhs = np.ones((op.n, 3), complex)
    for trans in (0, 1, 2):
        x = factor.apply(rhs, solve=True, trans=trans)
        image = 5*x + (u@(v@x) if trans == 0 else v.T@(u.T@x) if trans == 1 else v.conj().T@(u.conj().T@x))
        np.testing.assert_allclose(image, rhs, rtol=3e-8, atol=3e-8)


def test_default_aca_does_not_start_product_team_and_diagnostic_threads_restore():
    from ghost_backend.execution.thread_control import threadpool_info
    op, _, _ = operator(256)
    with execution_scope(validate_options(dict(assembly_threads=4, blas_threads=4))), mock.patch.dict(os.environ):
        os.environ.pop('GHOST_COMPRESSED_INVERSE_BUILDER', None)
        os.environ.pop('GHOST_COMPRESSED_INVERSE_THREADS', None)
        before = [row['num_threads'] for row in threadpool_info() if row.get('user_api') == 'blas']
        factor = CompressedSystem(op, op.coordinates, inverse_only=True)
        after = [row['num_threads'] for row in threadpool_info() if row.get('user_api') == 'blas']
    assert factor.evidence['inverse_builder'] == 'aca'
    assert factor.evidence['inverse_blas_threads'] == 1
    assert factor.evidence['block_products']['teams'] == 0
    assert before == after


@pytest.mark.parametrize('builder,inverse_only,expected', [
    ('aca', True, 1), ('aca', False, 'configured'),
    ('randomized', True, 'configured'), ('randomized', False, 'configured')])
def test_automatic_inverse_threads_apply_only_to_aca_inverse(builder, inverse_only, expected):
    op, _, _ = operator(64)
    with mock.patch.dict(os.environ, GHOST_COMPRESSED_INVERSE_BUILDER=builder):
        os.environ.pop('GHOST_COMPRESSED_INVERSE_THREADS', None)
        factor = CompressedSystem(op, op.coordinates, inverse_only=inverse_only)
    assert factor.evidence['inverse_blas_threads'] == expected


def test_explicit_configured_inverse_threads_retains_caller_policy():
    op, _, _ = operator(64)
    with mock.patch.dict(os.environ, GHOST_COMPRESSED_INVERSE_BUILDER='aca',
                         GHOST_COMPRESSED_INVERSE_THREADS='configured'), \
            mock.patch.object(bp, 'product_scope', wraps=bp.product_scope) as scope:
        factor = CompressedSystem(op, op.coordinates, inverse_only=True)
    assert factor.evidence['inverse_blas_threads'] == 'configured'
    assert scope.call_args.kwargs['blas_threads'] is None


def test_overlapping_single_thread_inverse_scopes_restore_only_after_last_exit():
    from ghost_backend.execution import options
    from ghost_backend.execution.thread_control import threadpool_info, threadpool_limits
    with threadpool_limits(limits=4, user_api='blas'):
        before = [row['num_threads'] for row in threadpool_info() if row.get('user_api') == 'blas']
        outer, inner = bp.product_scope(enabled=False, blas_threads=1), bp.product_scope(enabled=False, blas_threads=1)
        outer.__enter__()
        inner.__enter__()
        try:
            outer.__exit__(None, None, None)
            assert options._POOL_BLAS_USERS == 1
            assert all(row['num_threads'] == 1 for row in threadpool_info() if row.get('user_api') == 'blas')
        finally:
            inner.__exit__(None, None, None)
        after = [row['num_threads'] for row in threadpool_info() if row.get('user_api') == 'blas']
        assert before == after


def test_matching_single_thread_scope_does_not_take_the_global_blas_lock():
    from ghost_backend.execution.thread_control import threadpool_limits
    with execution_scope(validate_options(dict(blas_threads=1))), threadpool_limits(limits=1), \
            mock.patch('ghost_backend.execution.options.linear_algebra_threads',
                       side_effect=AssertionError('Already matching BLAS scope was serialized')):
        with bp.product_scope(enabled=False, blas_threads=1) as team:
            assert team.pool is None


def test_inverse_thread_diagnostic_honors_explicit_blas_cap():
    from ghost_backend.execution.options import cpu_allocation_scope
    op, _, _ = operator(64)
    with execution_scope(validate_options(dict(blas_threads=1))), cpu_allocation_scope(4), \
            mock.patch.dict(os.environ, GHOST_COMPRESSED_INVERSE_THREADS='2'):
        with pytest.raises(ValueError, match='configured BLAS cap'):
            CompressedSystem(op, op.coordinates, inverse_only=True)


@pytest.mark.parametrize('setting', ['0', '-1', '1.5', 'many', '3'])
def test_invalid_inverse_thread_diagnostic_rejects(setting):
    from ghost_backend.execution.options import cpu_allocation_scope
    op, _, _ = operator(64)
    with cpu_allocation_scope(2), mock.patch.dict(os.environ, GHOST_COMPRESSED_INVERSE_THREADS=setting):
        with pytest.raises(ValueError, match='CPU allocation'):
            CompressedSystem(op, op.coordinates, inverse_only=True)
