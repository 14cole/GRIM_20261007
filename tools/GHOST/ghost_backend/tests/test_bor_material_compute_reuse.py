"""Material performance changes retain original coefficients and solve gates."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, streaming
from ghost_backend.bor.compressed_cross import CompressedCrossFarBlocks, _verified_component
from ghost_backend.bor.options import option_scope, validate_options


def _cross(close=False, medium=None):
    if close:
        p,q = bor.sphere_generatrix(.05,12),bor.sphere_generatrix(.04,8)
    else:
        p = np.column_stack((np.full(25,.04),np.linspace(.2,0.,25)))
        q = np.column_stack((np.full(19,.08),np.linspace(.9,.7,19)))
    return bor.BorCrossOperators(bor.BorPecSolver(p,1e9,medium=medium),
                                 bor.BorPecSolver(q,1e9,medium=medium))


class RectangularStorageTests(unittest.TestCase):
    def test_full_coefficient_check_rejects_bad_svd_proposal(self):
        import ghost_backend.bor.compressed_cross as cc
        raw = np.ones((12,10),complex)
        u,s,v = np.linalg.svd(raw,full_matrices=False)
        u = u.copy();u[0,0] += .1
        with mock.patch.object(cc.la,'svd',return_value=(u,s,v)):
            entry,error = _verified_component(raw,1e-10,True)
        self.assertEqual(entry[0],'dense')
        np.testing.assert_array_equal(entry[1],raw)
        self.assertEqual(error,0.)

    def test_rectangular_modes_match_dense_coefficients_and_own_near_rules(self):
        for close in (False,True):
            for medium in (None,(2.5-.2j,1.1-.01j)):
                with self.subTest(close=close,medium=medium):
                    cross = _cross(close,medium)
                    dense = streaming.StreamingCrossFarBlocks(cross,4,workers=1,tile_threads=1,mode_block=5)
                    compact = CompressedCrossFarBlocks(cross,4,workers=1,mode_block=5,tile_budget_gb=.05)
                    try:
                        for mode in (0,1,-1,4,-4):
                            for family, getter in (('efie','efie_blocks'),('ibc','bracket_blocks')):
                                expected = np.asarray(getattr(dense,getter)(mode))
                                actual = np.asarray(getattr(compact,getter)(mode))
                                self.assertLess(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-300),2e-10)
                        self.assertLessEqual(compact.evidence['max_relative_block_error'],1e-10)
                        self.assertLessEqual(compact.evidence['stored_gb'],streaming.estimate_rectangular_streaming_gb(
                            cross.sp.gen.n_elems,cross.sq.gen.n_elems,4))
                        self.assertLessEqual(compact.memory_gb(),compact.evidence['stored_gb']+.05/4)
                        if close:
                            self.assertTrue(cross.near_pairs)
                            self.assertEqual(compact.evidence['lowrank_components'],0)
                        else:
                            self.assertGreater(compact.evidence['lowrank_components'],0)
                    finally:
                        dense.close();compact.close()

    def test_destination_assembly_matches_tables_with_variable_ranges_and_spill(self):
        cross = _cross(True,(2.4-.3j,1.))
        cross.prepare(3,workers=1)
        expected = {(kind,m):getattr(cross,'assemble_'+kind)(m,3) for kind in ('T','P') for m in (0,1,-1,3)}
        raw_near = {pair:{kind:value.copy() for kind,value in record.items()}
                    for pair,record in cross._cache[3].items()}
        with tempfile.TemporaryDirectory() as directory:
            for spill in (None,directory):
                stream = CompressedCrossFarBlocks(cross,3,workers=1,mode_block=2,spill=spill,tile_budget_gb=.05)
                cross._stream = stream
                try:
                    for (kind,m), wanted in expected.items():
                        holder = np.full((wanted.shape[0]+2,wanted.shape[1]+3),123.+4j)
                        destination = holder[:wanted.shape[0],:wanted.shape[1]]
                        actual = getattr(cross,'assemble_'+kind)(m,3,out=destination)
                        self.assertIs(actual,destination)
                        np.testing.assert_allclose(actual,wanted,rtol=2e-10,atol=2e-12*np.max(abs(wanted)))
                        np.testing.assert_array_equal(holder[-2:],123.+4j)
                    for pair,record in raw_near.items():
                        for kind,value in record.items():
                            np.testing.assert_array_equal(cross._cache[3][pair][kind],value)
                    if spill:
                        self.assertGreater(stream.spilled_gb(),0)
                        self.assertGreater(stream.memory_gb(),0.)  # compact numeric indexes remain resident
                        self.assertLess(stream.memory_gb(),stream.spilled_gb())
                        path = Path(stream._spill.path)
                    else:
                        self.assertGreater(stream.n_sweeps,1)
                finally:
                    cross.close_streaming()
                if spill:
                    self.assertFalse(path.exists())

    def test_selection_keeps_single_precision_on_original_store(self):
        cross = _cross()
        with option_scope(validate_options(dict(far_compression='on'))):
            try:
                cross.enable_streaming(2,workers=1)
                self.assertIsInstance(cross._stream,CompressedCrossFarBlocks)
                cross.enable_streaming(2,workers=1,single_blocks=True)
                self.assertIsInstance(cross._stream,streaming.StreamingCrossFarBlocks)
            finally:
                cross.close_streaming()

    def test_legacy_unbanded_sampling_keeps_original_store_and_backend_label(self):
        from ghost_backend.bor import kernels
        cross = _cross()
        with option_scope(validate_options(dict(far_compression='on'))), mock.patch.object(kernels,'BANDED_FFT',False):
            try:
                cross.enable_streaming(1,workers=1)
                self.assertIsInstance(cross._stream,streaming.StreamingCrossFarBlocks)
                self.assertIn(streaming.sampling_backend_name(cross._stream),('native_c','numpy'))
            finally:
                cross.close_streaming()

    def test_concurrent_different_ranges_keep_matching_descriptors_and_values(self):
        cross = _cross()
        dense = streaming.StreamingCrossFarBlocks(cross,3,workers=1,tile_threads=1)
        expected = {m:np.asarray(dense.efie_blocks(m)) for m in (0,3)}
        compact = CompressedCrossFarBlocks(cross,3,workers=1,mode_block=2,tile_budget_gb=.05)
        entered,release,second_started = threading.Event(),threading.Event(),threading.Event()
        original = compact._load_array
        def paused_load(*args):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test did not release the first range read.')
            return original(*args)
        def second_read():
            second_started.set()
            return np.asarray(compact.efie_blocks(3))
        try:
            with mock.patch.object(compact,'_load_array',paused_load), ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(lambda:np.asarray(compact.efie_blocks(0)))
                try:
                    self.assertTrue(entered.wait(5))
                    second = pool.submit(second_read)
                    self.assertTrue(second_started.wait(5))
                    with self.assertRaises(TimeoutError):
                        second.result(timeout=.1)
                finally:
                    release.set()
                for future,mode in ((first,0),(second,3)):
                    np.testing.assert_allclose(future.result(timeout=5),expected[mode],rtol=2e-10,atol=1e-13)
        finally:
            release.set()
            dense.close();compact.close()

    def test_index_budget_falls_back_without_reducing_modes_or_precision(self):
        import ghost_backend.bor.compressed_cross as cc
        cross = _cross()
        with option_scope(validate_options(dict(far_compression='on'))), mock.patch.object(cc,'CROSS_STORAGE_CHUNK_BYTES',2**40):
            try:
                cross.enable_streaming(4,workers=1,mode_block=5,tile_budget_gb=.05)
                self.assertIsInstance(cross._stream,streaming.StreamingCrossFarBlocks)
                self.assertEqual(cross._stream.m_max,4)
                self.assertEqual(cross._stream.dtype,np.complex128)
                self.assertIn('indexes',cross._stream.evidence['compression_fallback'])
            finally:
                cross.close_streaming()

    def test_numeric_arena_spans_chunks_without_retaining_component_objects(self):
        import ghost_backend.bor.compressed_cross as cc
        cross = _cross()
        reference = streaming.StreamingCrossFarBlocks(cross,2,workers=1,tile_threads=1)
        with mock.patch.object(cc,'CROSS_STORAGE_CHUNK_BYTES',4096):
            compact = CompressedCrossFarBlocks(cross,2,workers=1,tile_budget_gb=.01)
        try:
            self.assertGreater(len(compact._arena),1)
            self.assertTrue(all(a.dtype==np.int64 for family in compact._blocks.values() for a in family.values()))
            for getter in ('efie_blocks','bracket_blocks'):
                np.testing.assert_allclose(getattr(compact,getter)(2),getattr(reference,getter)(2),rtol=2e-10,atol=1e-13)
        finally:
            reference.close();compact.close()


class MaterialFactorTests(unittest.TestCase):
    def test_material_coordinates_use_checked_factor_and_lu_fallback(self):
        from ghost_backend.linalg import hierarchical
        from ghost_backend.tests.test_bor_physics_regression import _hemisphere, _bulged_upper_interface
        outer,core = bor.sphere_generatrix(.035,10),bor.sphere_generatrix(.02,8)
        calls = [lambda:bor.solve_bor_dielectric(outer,1e9,[0.,45.,90.],2.5-.2j,workers=1),
                 lambda:bor.solve_bor_coated_pec(outer,core,1e9,[0.,45.,90.],2.5-.2j,workers=1),
                 lambda:bor.solve_bor_partial_coating(
                     _bulged_upper_interface(.035,4),_hemisphere(.035,4),[_hemisphere(.035,4,False)],
                     1e9,[0.,45.,90.],2.5-.2j,workers=1),
                 lambda:bor.solve_bor_coating_patch(
                     _bulged_upper_interface(.035,4),_hemisphere(.035,4),[_hemisphere(.035,4,False)],core,
                     1e9,[0.,45.,90.],eps_inner=2.2-.06j,mu_inner=1.-.01j,
                     eps_patch=3.-.1j,mu_patch=1.-.02j,workers=1)]
        for solve in calls:
            with self.subTest(solve=solve):
                with mock.patch.object(hierarchical,'automatic_hierarchical',return_value=False):
                    reference = solve()
                with mock.patch.object(hierarchical,'automatic_hierarchical',return_value=True):
                    actual = solve()
                    with mock.patch.object(hierarchical,'HierarchicalFactor',side_effect=hierarchical.HierarchicalRejected('test rejection')):
                        fallback = solve()
                systems = actual['modal_execution']['systems']
                self.assertTrue(all(event['backend']=='hodlr' for event in systems))
                self.assertTrue(all('hierarchical_fallback' in event for event in fallback['modal_execution']['systems']))
                for key in ('amp_vv','amp_hh'):
                    np.testing.assert_allclose(actual[key],reference[key],rtol=1e-11,atol=1e-13)
                    np.testing.assert_allclose(fallback[key],reference[key],rtol=1e-13,atol=1e-14)

    def test_complete_lossy_coated_solve_preserves_fields_with_compressed_cross(self):
        outer,core = bor.sphere_generatrix(.035,20),bor.sphere_generatrix(.02,14)
        common = dict(points_outer=outer,points_core=core,freq_hz=1e9,thetas_deg=[0.,45.,90.,137.],
                      eps_r=2.5-.2j,mu_r=1.1-.03j,workers=1,assembly='streaming',table_precision='double')
        with option_scope(validate_options(dict(far_compression='off'))):
            reference = bor.solve_bor_coated_pec(**common)
        with option_scope(validate_options(dict(far_compression='on'))):
            actual = bor.solve_bor_coated_pec(**common)
        evidence = actual['stream_far_compression']['cross_outer_core']
        self.assertEqual(evidence['coefficient_check'],'all_original_coefficients')
        self.assertLessEqual(evidence['max_relative_block_error'],1e-10)
        for key in ('amp_vv','amp_hh'):
            np.testing.assert_allclose(actual[key],reference[key],rtol=2e-10,atol=2e-12)


if __name__=='__main__':
    unittest.main()
