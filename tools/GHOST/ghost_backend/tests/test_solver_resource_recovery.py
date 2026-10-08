"""Failure recovery must retain accuracy without bypassing resource admission."""
import types
import weakref
from unittest import mock

import numpy as np
import pytest

from ghost_backend.bor.factor import ModalFactor
from ghost_backend.compressed import factor as cf
from ghost_backend.execution.options import execution_scope
from ghost_backend.linalg.dense import DenseFactor
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from ghost_backend.linalg import residual_spool


@pytest.mark.parametrize('trans', [0, 1, 2])
@pytest.mark.parametrize('reverse', [False, True])
def test_gmres_happy_breakdown_does_not_stop_other_columns(trans, reverse):
    a = np.diag([1., 1., -1.]).astype(complex)
    class Operator:
        row_norm = column_norm = np.ones(3)
        row_error = column_error = np.zeros(3)
        def __len__(self): return 3
        def matmul(self, x, direction): return a @ x
    class HalfIdentity:
        def apply(self, x, **kwargs): return .5*x
    instance = cf.CompressedFactor.__new__(cf.CompressedFactor)
    instance.recycled = False
    instance.a, instance.factor = Operator(), HalfIdentity()
    instance.checkpoint = lambda: None
    instance.tolerance = 2e-10  # No tighter retry can hide a failed Krylov solve.
    instance.event = dict(max_refinements=0, refinement_steps=0, gmres_columns=0,
                          max_backward_error=0.)
    b = np.column_stack(([1., 0., 0.], [0., 1., 1./9])).astype(complex)
    if reverse: b = b[:, ::-1]
    x = instance.inverse(b, trans=trans)
    np.testing.assert_allclose(a @ x, b, rtol=0, atol=3e-15)
    assert instance.event['gmres_columns'] == 2
    assert instance.event['max_backward_error'] < 1e-14


def test_compressed_memory_retry_releases_failed_arrays():
    refs = []
    class Construction:
        def __init__(self, *args, **kwargs):
            self.payload = np.ones(100_000, complex)
            refs.append(weakref.ref(self.payload))
            if len(refs) == 1: raise MemoryError('full preconditioner')
            assert refs[0]() is None
            self.evidence = {}
    operator = types.SimpleNamespace(n=1, row_norm=np.ones(1), row_error=np.zeros(1),
        bytes=0, evidence={}, checkpoint=lambda: None, coordinates=np.zeros((1, 1)))
    with mock.patch.object(cf, 'CompressedSystem', Construction):
        instance = cf.CompressedFactor(operator, storage_budget_bytes=64*1024**2,
                                       check_precision=False)
    assert instance.tolerance == cf.COMPACT_PRECONDITIONER_TOLERANCE


@pytest.mark.parametrize('entry', ['solve', 'inverse'])
def test_bor_lu_recovery_releases_rejected_factor_before_allocation(entry):
    factor = ModalFactor(np.eye(3, dtype=complex), 0, False)
    class Rejecting:
        def __init__(self): self.payload = np.ones(100_000, complex)
        def solve(self, *args, **kwargs): raise HierarchicalRejected('stalled')
    factor.hierarchical = Rejecting()
    ref = weakref.ref(factor.hierarchical.payload)
    actual_lu = factor._factor_lu
    def observe(*args):
        assert ref() is None
        return actual_lu(*args)
    factor._factor_lu = observe
    b = np.ones((3, 1), complex)
    np.testing.assert_array_equal(getattr(factor, entry)(b), b)


def test_copy_admission_obeys_smaller_reservation_and_resident_usage():
    gib = 1024**3
    with execution_scope(dict(ram_budget_gib=3), memory_budget_gib=1.8), \
         mock.patch('ghost_backend.twod.solver._detect_available_gb', return_value=64), \
         mock.patch('ghost_backend.twod.solver._process_rss_bytes', return_value=int(.9*gib)):
        assert not residual_spool.copy_fits(int(.8*gib))
        assert residual_spool.auto_spooled(int(.8*gib))
        assert residual_spool.copy_fits(int(.5*gib))
    with execution_scope(dict(ram_budget_gib=1)), \
         mock.patch('ghost_backend.twod.solver._detect_available_gb', return_value=64), \
         mock.patch('ghost_backend.twod.solver._process_rss_bytes', return_value=int(.9*gib)):
        # An explicit budget also protects matrices below the 512 MiB threshold.
        assert residual_spool.auto_spooled(1024**2)


@pytest.mark.parametrize('kind', ['2d', 'bor'])
def test_failed_auto_spool_cannot_fall_through_to_unadmitted_lu(kind, tmp_path):
    a = np.eye(4, dtype=complex, order='F')
    with execution_scope(dict(dense_residual_storage='auto', temporary_directory=str(tmp_path))), \
         mock.patch.object(residual_spool, 'auto_spooled', return_value=True), \
         mock.patch.object(residual_spool, 'copy_fits', return_value=False), \
         mock.patch.object(residual_spool, 'ResidualSpool', side_effect=OSError('disk full')):
        with pytest.raises(MemoryError, match='RAM budget'):
            if kind == '2d': DenseFactor(a, owned_matrix=True)
            else: ModalFactor(a, 0, False, owned=True)


def test_bor_owned_fallback_can_spool_and_preserves_original_matrix_residuals(tmp_path):
    a = np.array([[3., 1.], [2., 4.]], dtype=complex, order='F')
    original = a.copy()
    factor = ModalFactor(a, 0, False, owned=True,
                         residual_storage=('auto', str(tmp_path)))
    class Rejecting:
        def solve(self, *args, **kwargs): raise HierarchicalRejected('stalled')
    factor.hierarchical = Rejecting()
    b = np.ones((2, 1), complex)
    with mock.patch.object(residual_spool, 'auto_spooled', return_value=True):
        x = factor.solve(b)
    try:
        assert factor.event['residual_storage'] == 'disk'
        np.testing.assert_allclose(original @ x, b, atol=1e-14)
        assert factor.event['max_backward_error'] < 1e-14
    finally:
        factor.close()
