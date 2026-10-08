"""Exact spool ownership, bounded mode windows and original physical coefficients."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, options
from ghost_backend.bor.spooled_far import SpooledFarBlocks
from ghost_backend.bor.streaming import StreamingFarBlocks, StreamingSpillError


class ExactSpoolTests(unittest.TestCase):
    def test_all_families_signed_modes_owned_windows_and_cleanup(self):
        surface = bor.BorPecSolver(bor.sphere_generatrix(.035,12),1e9)
        zs = np.full(surface.g.rho.shape,45+7j)
        kwargs = dict(mfie=True,ibc_zs_pt=zs,workers=1,tile_threads=1)
        with tempfile.TemporaryDirectory() as directory:
            reference = StreamingFarBlocks(surface,4,mode_block=5,**kwargs)
            actual = SpooledFarBlocks(surface,4,mode_block=1,spill=directory,**kwargs)
            try:
                self.assertEqual(actual.n_sweeps,1)
                self.assertEqual(actual.mode_block,1)
                self.assertEqual(actual.spilled_gb(),reference.memory_gb())
                rows,cols = np.array([0,2,7,8]),np.array([8,3,0])
                for mode in (0,1,-1,4,-4,2,0):
                    for family in ('efie','mfie','ibc'):
                        expected = reference.query_blocks(family,mode,rows,cols)
                        found = actual.query_blocks(family,mode,rows,cols)
                        for a,b in zip(found,expected):
                            np.testing.assert_allclose(a,b,rtol=3e-13,atol=1e-16)
                        # A caller's returned slice must not alias the window.
                        found[0][:] = 999
                        np.testing.assert_allclose(actual.query_blocks(family,mode,rows,cols)[0],
                                                   expected[0],rtol=3e-13,atol=1e-16)
                    self.assertEqual(actual.Z.shape[1],1)
                    self.assertNotIsInstance(actual.Z,np.memmap)
                    self.assertLessEqual(actual.memory_gb(),reference.memory_gb()/5*1.000001)
                self.assertEqual(actual.n_sweeps,1)
                self.assertGreater(actual.evidence['reloads'],1)
            finally:
                actual.close()
                reference.close()
            self.assertEqual(list(Path(directory).iterdir()),[])
            with self.assertRaises(RuntimeError):
                actual.query_blocks('efie',0,rows,cols)

    def test_cancellation_during_build_closes_files(self):
        surface = bor.BorPecSolver(bor.sphere_generatrix(.035,8),1e9)
        surface._checkpoint = mock.Mock(side_effect=RuntimeError('cancelled'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                SpooledFarBlocks(surface,3,mode_block=2,spill=directory,tile_threads=1)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_failed_window_read_closes_partial_payload_and_old_range(self):
        surface = bor.BorPecSolver(bor.sphere_generatrix(.035,12),1e9)
        with tempfile.TemporaryDirectory() as directory:
            actual = SpooledFarBlocks(surface,3,mfie=True,mode_block=1,spill=directory,tile_threads=1)
            original = actual._read_family
            def failed_second_family(family,lo,hi):
                if family == 'mfie':
                    raise IOError('read interrupted')
                return original(family,lo,hi)
            with mock.patch.object(actual,'_read_family',side_effect=failed_second_family):
                with self.assertRaisesRegex(IOError,'read interrupted'):
                    actual.query_blocks('efie',2,[0],[1])
            self.assertTrue(actual._closed)
            self.assertIsNone(actual.Z)
            self.assertEqual(list(Path(directory).iterdir()),[])
            with self.assertRaises(RuntimeError):
                actual.query_blocks('efie',0,[0],[1])

    def test_interrupted_error_is_cancellation_not_disk_fallback(self):
        with mock.patch.object(bor,'COMPRESSED_FAR_CACHE_MIN_NODES',1), \
                mock.patch.object(bor,'plan_compressed_far_spill',return_value=('not-used',1.)), \
                mock.patch('ghost_backend.bor.spooled_far.SpooledFarBlocks',
                           side_effect=InterruptedError('cancelled')):
            with self.assertRaisesRegex(InterruptedError,'cancelled'):
                bor.solve_bor(bor.sphere_generatrix(.035,8),1e9,[0.,180.],
                              bor_options=dict(factorization='compressed'))

    def test_spooled_physical_fields_match_query_only(self):
        points = bor.sphere_generatrix(.035,8)
        kwargs = dict(freq_hz=1e9,thetas_deg=[20.,90.,150.],n_modes=9,workers=1,
                      bor_options=dict(factorization='compressed'))
        reference = bor.solve_bor(points,**kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(bor,'COMPRESSED_FAR_CACHE_MIN_NODES',1), \
                mock.patch.object(bor,'COMPRESSED_FAR_CACHE_GB',.000032), \
                mock.patch('ghost_backend.bor.streaming.spill_directory',return_value=directory):
            actual = bor.solve_bor(points,**kwargs)
            self.assertEqual(list(Path(directory).iterdir()),[])
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[field],reference[field],rtol=3e-9,atol=1e-13)
        self.assertGreater(actual['stream_spill_gb'],0.)
        self.assertEqual(actual['stream_sweeps'],1)

    def test_failed_optional_disk_cache_reverts_to_original_queries(self):
        points = bor.sphere_generatrix(.035,8)
        kwargs = dict(freq_hz=1e9,thetas_deg=[0.,180.],n_modes=9,workers=1,
                      bor_options=dict(factorization='compressed'))
        reference = bor.solve_bor(points,**kwargs)
        with mock.patch.object(bor,'COMPRESSED_FAR_CACHE_MIN_NODES',1), \
                mock.patch.object(bor,'plan_compressed_far_spill',return_value=('not-used',1.)), \
                mock.patch('ghost_backend.bor.spooled_far.SpooledFarBlocks',
                           side_effect=StreamingSpillError('disk unavailable')):
            actual = bor.solve_bor(points,**kwargs)
        for field in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[field],reference[field],rtol=3e-11,atol=1e-13)
        self.assertEqual(actual['stream_spill_gb'],0.)
        self.assertEqual(actual['stream_cache_fallback'],'disk unavailable')
        self.assertEqual(actual['modal_execution']['systems'][0]['compression_tile'],32)


if __name__ == '__main__':
    unittest.main()
