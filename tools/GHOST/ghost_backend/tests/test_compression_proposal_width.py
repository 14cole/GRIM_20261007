"""Large tile proposals retain full coefficient and storage safeguards."""
import unittest
from unittest.mock import patch
import numpy as np

from ghost_backend.compressed.operator import tile_payload
from ghost_backend.linalg.sweep import _qr_basis
from ghost_backend.execution.thread_control import threadpool_limits


class CompressionProposalTests(unittest.TestCase):
    def test_moderate_rank_tile_avoids_full_width_qr(self):
        rng = np.random.default_rng(107)
        left = rng.normal(size=(300, 30)) + 1j*rng.normal(size=(300, 30))
        right = rng.normal(size=(30, 300)) + 1j*rng.normal(size=(30, 300))
        with threadpool_limits(limits=1):
            raw = left @ right
            tail = np.full(raw.shape, 1e-16)
            with patch('ghost_backend.compressed.operator._qr_basis', wraps=_qr_basis) as qr:
                payload, reconstructed, error, accepted = tile_payload(raw, tail.copy(), 1e-14, 'qr')
        self.assertTrue(accepted)
        self.assertEqual(qr.call_args_list[0].args[0].shape, (300, 48))
        self.assertTrue(all(call.args[0].shape[0] <= 48 for call in qr.call_args_list[1:]))
        self.assertEqual(payload[0].shape[1], 30)
        self.assertLess(sum(a.nbytes for a in payload), raw.nbytes)
        self.assertLessEqual(np.linalg.norm(raw-reconstructed), 1e-14*np.linalg.norm(raw))
        np.testing.assert_array_equal(error, tail+abs(raw-reconstructed))

    def test_unsampled_localized_change_is_checked(self):
        raw = np.ones((300, 300), dtype=complex)
        sampled = set(np.linspace(0, 299, 48).astype(int))
        column = next(i for i in range(300) if i not in sampled)
        raw[121, column] += .01j
        with threadpool_limits(limits=1):
            with patch('ghost_backend.compressed.operator._qr_basis', wraps=_qr_basis) as qr:
                payload, reconstructed, error, accepted = tile_payload(raw, np.zeros(raw.shape), 1e-14, 'qr')
        self.assertEqual(qr.call_args_list[0].args[0].shape, (300, 48))
        self.assertEqual(qr.call_args_list[-1].args[0].shape, (300, 300))
        self.assertLessEqual(np.linalg.norm(raw-reconstructed), 1e-14*np.linalg.norm(raw))
        np.testing.assert_array_equal(error, abs(raw-reconstructed))

    def test_high_rank_rectangular_tile_keeps_dense_storage(self):
        rng = np.random.default_rng(203)
        raw = rng.normal(size=(256, 300)) + 1j*rng.normal(size=(256, 300))
        tail = np.full(raw.shape, 1e-12)
        with threadpool_limits(limits=1):
            payload, reconstructed, error, accepted = tile_payload(raw, tail, 1e-14, 'qr')
        self.assertFalse(accepted)
        self.assertIs(payload[0], raw)
        self.assertIsNone(payload[1])
        self.assertIs(reconstructed, raw)
        self.assertIs(error, tail)

    def test_trimmed_weak_complex_direction_remains_in_error_bound(self):
        # A direction below the final tolerance but above the proposal threshold
        # should be removed by the small QR, and its actual error still counted.
        columns = np.arange(400)
        raw = np.zeros((400, 400), complex)
        raw[0] = 1./20
        raw[1] = 4e-8*np.exp(2j*np.pi*columns/400)/20
        tail = np.full(raw.shape, 3e-13)
        with threadpool_limits(limits=1):
            with patch('ghost_backend.compressed.operator._qr_basis', wraps=_qr_basis) as qr:
                payload, reconstructed, error, accepted = tile_payload(raw, tail.copy(), 1e-7, 'qr')
        self.assertTrue(accepted)
        self.assertEqual(qr.call_args_list[0].args[0].shape, (400, 64))
        self.assertEqual(payload[0].shape[1], 1)
        self.assertLess(np.linalg.norm(raw-reconstructed), 1e-7*np.linalg.norm(raw))
        self.assertGreater(np.linalg.norm(raw-reconstructed), 3e-8)
        np.testing.assert_array_equal(error, tail+abs(raw-reconstructed))

    def test_sampling_and_trim_errors_cannot_each_spend_full_budget(self):
        # The proposal misses one unsampled localized direction and the small
        # QR trims another. Their combined error exceeds the complete-tile
        # tolerance even though each individual error is below that tolerance.
        rows, columns = 300, 400
        sampled = set(np.linspace(0, columns-1, 48).astype(int))
        missing = next(index for index in range(columns) if index not in sampled)
        raw = np.zeros((rows, columns), complex)
        raw[0] = 1./20
        raw[1] = 5e-8*np.exp(2j*np.pi*np.arange(columns)/columns)/20
        raw[2, missing] = .8e-7j
        tail = np.full(raw.shape, 2e-13)
        with threadpool_limits(limits=1):
            with patch('ghost_backend.compressed.operator._qr_basis', wraps=_qr_basis) as qr:
                payload, reconstructed, error, accepted = tile_payload(raw, tail.copy(), 1e-7, 'qr')
        self.assertTrue(accepted)
        self.assertEqual(qr.call_args_list[0].args[0].shape, (rows, 48))
        self.assertEqual(qr.call_args_list[-1].args[0].shape, raw.shape)
        self.assertLessEqual(np.linalg.norm(raw-reconstructed), 1e-7*np.linalg.norm(raw))
        self.assertLess(sum(value.nbytes for value in payload), raw.nbytes)
        np.testing.assert_array_equal(error, tail+abs(raw-reconstructed))


if __name__ == '__main__':
    unittest.main()
