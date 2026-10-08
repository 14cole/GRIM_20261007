"""Bounded factor products and RHS workspaces retain certified solve paths."""
from unittest import mock
import numpy as np
import pytest
import scipy.linalg as la
from ghost_backend.linalg import hierarchical as hf
from ghost_backend.compressed import factor as cf


@pytest.mark.parametrize('order', ['C', 'F'])
def test_block_products_bound_each_gather_and_match_full_complex_matrix(order):
    rng = np.random.default_rng(324)
    a = np.array(rng.normal(size=(83, 91))+1j*rng.normal(size=(83, 91)), order=order)
    rows, cols = rng.permutation(83)[:71], rng.permutation(91)[:67]
    block = hf.Block(a, rows, cols, lambda: None)
    rhs = rng.normal(size=(67, 5))+1j*rng.normal(size=(67, 5))
    basis = rng.normal(size=(71, 7))+1j*rng.normal(size=(71, 7))
    sizes = []
    original = hf.Block.dense
    def gather(part):
        sizes.append(np.prod(part.shape)*16)
        return original(part)
    with mock.patch.object(hf, 'PRODUCT_PANEL_BYTES', 4096), mock.patch.object(hf.Block, 'dense', gather):
        actual, projected = block.matmul(rhs), block.project(basis)
    assert max(sizes) <= 4096
    assert len(sizes) > 2
    expected = a[np.ix_(rows, cols)]
    np.testing.assert_allclose(actual, expected@rhs, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(projected, basis.conj().T@expected, rtol=3e-14, atol=3e-14)


def test_sampled_compression_does_not_gather_full_block_and_retains_accuracy():
    rng = np.random.default_rng(753)
    a = (rng.normal(size=(200, 4))+1j*rng.normal(size=(200, 4))) @ rng.normal(size=(4, 190))
    block = hf.Block(a, np.arange(200), np.arange(190), lambda: None)
    original = hf.Block.dense
    def gather(part):
        assert np.prod(part.shape)*16 <= 8192
        return original(part)
    with mock.patch.object(hf, 'PRODUCT_PANEL_BYTES', 8192), mock.patch.object(hf.Block, 'dense', gather):
        u, v, error = hf.compress_sampled(block, 1e-10, 32, rng)
    assert error < 1e-10
    np.testing.assert_allclose(u@v, a, rtol=2e-12, atol=2e-12)
    with mock.patch.object(block, 'checkpoint', side_effect=InterruptedError('stop')):
        with pytest.raises(InterruptedError): block.matmul(np.ones((190, 1)))
        with pytest.raises(InterruptedError): block.project(np.ones((200, 1)))


def old_solve(node, b, trans=0):
    """Prior inverse algorithm, independent of the new workspace recursion."""
    if node.leaf:
        return la.lu_solve(node.lu, b, trans=trans, check_finite=False)
    n = node.left.n
    if node.lu is None:
        return np.vstack((old_solve(node.left, b[:n], trans), old_solve(node.right, b[n:], trans)))
    if trans == 0:
        z1, z2 = old_solve(node.left, b[:n]), old_solve(node.right, b[n:])
        small = np.vstack((node.v12@z2, node.v21@z1))
        correction = la.lu_solve(node.lu, small, check_finite=False)
        r = node.e1.shape[1]
        return np.vstack((z1-node.e1@correction[:r], z2-node.e2@correction[r:]))
    adj = lambda a: a.T if trans == 1 else a.conj().T
    small = np.vstack((adj(node.e1)@b[:n], adj(node.e2)@b[n:]))
    correction = la.lu_solve(node.lu, small, trans=trans, check_finite=False)
    r = node.e1.shape[1]
    return np.vstack((old_solve(node.left, b[:n]-adj(node.v21)@correction[r:], trans),
                      old_solve(node.right, b[n:]-adj(node.v12)@correction[:r], trans)))


@pytest.mark.parametrize('trans', [0, 1, 2])
def test_workspace_inverse_matches_old_recursion_without_stacking_full_rhs(trans):
    rng = np.random.default_rng(742)
    n = 129
    a = 10*np.eye(n) + (rng.normal(size=(n, 3))+1j*rng.normal(size=(n, 3))) @ rng.normal(size=(3, n))/n
    with mock.patch.object(hf, 'LEAF_SIZE', 32):
        factor = hf.HierarchicalFactor(a)
    b = rng.normal(size=(n, 9))+1j*rng.normal(size=(n, 9))
    expected = old_solve(factor.root, b, trans)
    original = np.vstack
    def small_stack(values):
        assert sum(len(value) for value in values) < 32
        return original(values)
    with mock.patch.object(hf.np, 'vstack', small_stack):
        actual = factor.root.solve(b, trans)
        inplace = b.copy()
        factor.root.solve_into(inplace, inplace, trans)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(inplace, expected, rtol=2e-14, atol=2e-14)
    op = a if trans == 0 else a.T if trans == 1 else a.conj().T
    assert np.linalg.norm(op@factor.solve(b, trans)-b)/np.linalg.norm(b) < 1e-12


@pytest.mark.parametrize('trans', [0, 1, 2])
def test_gmres_basis_is_contiguous_and_never_conjugated_as_a_tensor(trans):
    rng = np.random.default_rng(678)
    n, count = 35, 4
    a = 3*np.eye(n)+(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/np.sqrt(n)
    b = rng.normal(size=(n,count))+1j*rng.normal(size=(n,count))
    class Operator:
        row_norm = np.abs(a).sum(axis=1)
        column_norm = np.abs(a).sum(axis=0)
        def matmul(self, x, direction):
            return (a if direction == 0 else a.T if direction == 1 else a.conj().T)@x
    class Identity:
        def apply(self, x, **kwargs):
            assert x.flags.f_contiguous or x.flags.c_contiguous
            return x.copy()
    instance = cf.CompressedFactor.__new__(cf.CompressedFactor)
    instance.recycled = False
    instance.a, instance.factor, instance.checkpoint = Operator(), Identity(), lambda: None
    original = np.conjugate
    def bounded_conjugate(value, **kwargs):
        assert value.ndim <= 2
        return original(value, **kwargs)
    with mock.patch.object(cf.np, 'conjugate', bounded_conjugate):
        actual = instance._gmres(b, np.zeros_like(b), trans)
    op = a if trans == 0 else a.T if trans == 1 else a.conj().T
    np.testing.assert_allclose(actual, np.linalg.solve(op,b), rtol=1e-11, atol=1e-12)
