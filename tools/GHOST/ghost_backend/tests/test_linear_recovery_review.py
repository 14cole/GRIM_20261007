"""Independent complex-valued checks of batched recovery and owned BoR LU."""
from unittest import mock
import numpy as np
import pytest
from ghost_backend.compressed.factor import CompressedFactor
from ghost_backend.bor.factor import ModalFactor
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from ghost_backend.linalg import residual_spool


@pytest.mark.parametrize('trans', [0, 1, 2])
def test_gmres_different_krylov_dimensions_with_complex_nonnormal_matrix(trans):
    rng = np.random.default_rng(507)
    a = np.zeros((7, 7), complex)
    a[0, 0] = 1. + .2j
    a[1:, 1:] = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)) + 5 * np.eye(6)
    operators = (a, a.T, a.conj().T)
    class Operator:
        row_norm, column_norm = np.sum(abs(a), axis=1), np.sum(abs(a), axis=0)
        row_error = column_error = np.zeros(7)
        def __len__(self): return 7
        def matmul(self, values, direction): return operators[direction] @ values
    class Preconditioner:
        def apply(self, values, **kwargs): return .5 * values
    instance = CompressedFactor.__new__(CompressedFactor)
    instance.recycled = False
    instance.a, instance.factor = Operator(), Preconditioner()
    instance.checkpoint = lambda: None
    instance.tolerance = 2e-10
    instance.event = dict(max_refinements=0, refinement_steps=0, gmres_columns=0, max_backward_error=0.)
    rhs = np.column_stack((np.eye(7)[:, 0], rng.normal(size=7) + 1j * rng.normal(size=7),
                           np.eye(7)[:, 3], np.zeros(7)))
    solution = instance.inverse(rhs, trans=trans)
    np.testing.assert_allclose(operators[trans] @ solution, rhs, rtol=0, atol=5e-14)


@pytest.mark.parametrize('order', ['C', 'F'])
def test_bor_rejected_factor_owned_spool_preserves_complex_transpose_and_condition(order, tmp_path):
    rng = np.random.default_rng(811)
    original = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)) + 8 * np.eye(4)
    matrix = np.array(original, order=order, copy=True)
    factor = ModalFactor(matrix, 1, True, owned=True, residual_storage=('auto', str(tmp_path)))
    original_condition = factor.condition
    class Rejecting:
        def solve(self, *args, **kwargs): raise HierarchicalRejected('exercise owned recovery')
    factor.hierarchical = Rejecting()
    rhs = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    try:
        with mock.patch.object(residual_spool, 'auto_spooled', return_value=True):
            solution = factor.solve(rhs)
        assert factor.trans == (1 if order == 'C' else 0)
        assert factor.event['residual_storage'] == 'disk'
        np.testing.assert_allclose(original @ solution, rhs, rtol=0, atol=5e-15)
        np.testing.assert_allclose(factor.a @ rhs, original @ rhs, rtol=0, atol=1e-14)
        np.testing.assert_allclose(original @ factor.inverse(rhs), rhs, rtol=0, atol=5e-15)
        # LAPACK reports an estimate, not the exact inverse norm; spooling
        # must preserve the original condition estimate for this matrix.
        np.testing.assert_allclose(factor.condition, original_condition, rtol=1e-12)
    finally:
        factor.close()


def test_concurrent_lu_copies_cannot_admit_the_same_remaining_ram():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    matrix = np.eye(4, dtype=complex)
    resident = [matrix.nbytes]
    start = threading.Barrier(2)
    original = np.array
    def copy(*args, **kwargs):
        # Release the GIL while an admitted copy is still not resident. A
        # second unguarded capacity check would pass against the same RSS.
        time.sleep(.025)
        result = original(*args, **kwargs)
        resident[0] += result.nbytes
        return result
    def worker():
        start.wait(timeout=3)
        try:
            return residual_spool.copy_for_lu(matrix)
        except MemoryError:
            return None
    with mock.patch('ghost_backend.twod.solver._process_rss_bytes', side_effect=lambda: resident[0]), \
            mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', return_value=3 * matrix.nbytes / 1024**3), \
            mock.patch.object(residual_spool, 'COPY_MARGIN_BYTES', 0), \
            mock.patch.object(residual_spool.np, 'array', copy), ThreadPoolExecutor(2) as executor:
        first, second = executor.submit(worker), executor.submit(worker)
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert sum(result is not None for result in results) == 1
    assert resident[0] == 2 * matrix.nbytes
