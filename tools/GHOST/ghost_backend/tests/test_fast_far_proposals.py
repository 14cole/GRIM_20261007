"""Experimental coefficient sampling must never certify unseen entries."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ghost_backend.compressed.fast_far import FarProposal, propose, separated_supports


class MatrixOracle:
    def __init__(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.queries = []

    def get_with_error(self, rows, cols):
        self.queries.append((len(rows), len(cols)))
        raw = self.matrix[np.ix_(rows, cols)]
        return raw, np.zeros(raw.shape)


class FastFarProposalTests(unittest.TestCase):
    def test_sampled_low_rank_tile_requires_complete_validation(self):
        rng = np.random.default_rng(42)
        left = rng.normal(size=(128, 3)) + 1j*rng.normal(size=(128, 3))
        right = rng.normal(size=(3, 128)) + 1j*rng.normal(size=(3, 128))
        matrix = left @ right
        oracle = MatrixOracle(matrix)
        ids = np.arange(128)
        with patch('ghost_backend.compressed.fast_far.separated_supports', return_value=True):
            candidate = propose(oracle, ids, ids)
        self.assertIsNotNone(candidate)
        self.assertFalse(candidate.evidence['certified'])
        self.assertTrue(candidate.evidence['full_validation_required'])
        self.assertLess(candidate.evidence['queried_entries'], matrix.size)
        tail = np.full(matrix.shape, 1e-16)
        payload, result, error, accepted = candidate.validate(matrix, tail)
        self.assertTrue(accepted)
        self.assertEqual(payload[0].shape[1], 3)
        np.testing.assert_array_equal(error, tail + abs(matrix-result))
        self.assertEqual(candidate.evidence['validation'], 'all_coefficients')

    def test_unseen_localized_error_falls_back_to_exact_coefficients(self):
        candidate = FarProposal(np.ones((64, 1)), np.ones((1, 64)))
        exact = np.ones((64, 64), complex)
        exact[37, 29] += .01
        tail = np.zeros(exact.shape)
        payload, result, error, accepted = candidate.validate(exact, tail)
        self.assertFalse(accepted)
        self.assertIs(result, exact)
        self.assertIs(payload[0], exact)
        self.assertIsNone(payload[1])
        self.assertIs(error, tail)

    def test_near_supports_never_sampled(self):
        oracle = MatrixOracle(np.eye(128, dtype=complex))
        with patch('ghost_backend.compressed.fast_far.separated_supports', return_value=False):
            self.assertIsNone(propose(oracle, np.arange(128), np.arange(128)))
        self.assertEqual(oracle.queries, [])

    def test_separation_uses_full_support_not_only_basis_node(self):
        geometry = SimpleNamespace(
            p0=np.array([[0., 0.], [10., 0.]]),
            segments=np.array([[1., 0.], [1., 0.]]),
            lengths=np.array([1., 1.]),
            elements_touching=lambda nodes: np.isin([0, 1], np.asarray(nodes)//2))
        source = SimpleNamespace(geometry=geometry, mesh=SimpleNamespace(nodes=[None]*4), n=4, kind='robin')
        self.assertTrue(separated_supports(source, np.array([0]), np.array([2])))
        geometry.segments[0, 0] = 9.99
        geometry.lengths[0] = 9.99
        self.assertFalse(separated_supports(source, np.array([0]), np.array([2])))

    def test_nonlocal_thin_layer_excluded(self):
        source = SimpleNamespace(geometry=object(), mesh=SimpleNamespace(nodes=[None]*4), n=8, kind='thin')
        self.assertFalse(separated_supports(source, np.array([0]), np.array([6])))

    def test_high_rank_tile_rejects_proposal(self):
        rng = np.random.default_rng(91)
        oracle = MatrixOracle(rng.normal(size=(128, 128)))
        with patch('ghost_backend.compressed.fast_far.separated_supports', return_value=True):
            self.assertIsNone(propose(oracle, np.arange(128), np.arange(128)))

    def test_full_validation_rejects_invalid_error_budget(self):
        candidate = FarProposal(np.ones((8, 1)), np.ones((1, 8)))
        with self.assertRaises(ValueError):
            candidate.validate(np.ones((8, 8)), -np.ones((8, 8)))

    def test_paired_sampling_preserves_shared_queries(self):
        rng = np.random.default_rng(87)
        matrix = rng.normal(size=(128, 3)) @ rng.normal(size=(3, 128))
        first, second = MatrixOracle(matrix), MatrixOracle(2j*matrix)
        class Pair:
            n = 128
            oracles = (first, second)
            def get_with_error(self, rows, cols):
                return [source.get_with_error(rows, cols) for source in self.oracles]
        with patch('ghost_backend.compressed.fast_far.separated_supports', return_value=True):
            candidates = propose(Pair(), np.arange(128), np.arange(128))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(first.queries, second.queries)
        self.assertTrue(candidates[0].validate(matrix, np.zeros(matrix.shape))[3])
        self.assertTrue(candidates[1].validate(2j*matrix, np.zeros(matrix.shape))[3])


if __name__ == '__main__':
    unittest.main()
