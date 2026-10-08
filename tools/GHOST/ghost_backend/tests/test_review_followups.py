"""Independent operator checks added after the September implementation review."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver as td, operators as ops
from ghost_backend.bor import solver as bor
from ghost_backend.twod.assembly.compact import CompactOperator
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.execution.cpu import CPUState, _STATE
from test_compact_multi_region import prepared


class FusedRegionalOperatorTests(unittest.TestCase):
    def test_all_four_operators_match_independent_assembly(self):
        for degree in (1, 2, 3):
            with execution_scope(validate_options({'basis_order':degree})), _STATE.override(CPUState()):
                mesh, infos, k = prepared('coated', 'TE', 16)
                n = len(mesh.nodes)
                rows, cols = np.arange(0,n,3), np.arange(1,n,2)
                mask = np.arange(len(mesh.elements)) >= len(mesh.elements)//2
                for wave in (k, k*(1.6-.03j)):
                    for native in (False, True):
                        with self.subTest(degree=degree, k=wave, native=native), patch.object(ops, '_NATIVE_FAR',native):
                            outputs = [CompactOperator(n,rows,cols) for _ in range(4)]
                            td._assemble_linear_operator_matrices_multi(mesh,wave,True,[mask],
                                output_node_ids_many=[(rows,cols)], operator_outputs=[outputs[:2]],
                                additional_operator_outputs=[outputs[2:]])
                            sk = td._assemble_linear_operator_matrices_multi(mesh,wave,True,[mask],
                                output_node_ids_many=[(rows,cols)])[0]
                            _, d = td._assemble_linear_operator_matrices_multi(mesh,wave,False,[mask],
                                compute_single_layer=False,output_node_ids_many=[(rows,cols)])[0]
                            w = td._assemble_linear_hypersingular_matrix(mesh,wave,
                                source_element_mask=mask,output_node_ids=(rows,cols))
                            for got, ref in zip(outputs,(*sk,d,w)):
                                scale = max(float(np.max(abs(ref.values))),1e-20)
                                np.testing.assert_allclose(got.values,ref.values,rtol=2e-9,atol=2e-11*scale)

    def test_dense_regional_solve_never_uses_the_separate_hypersingular_pass(self):
        from ghost_backend.twod.formulations.regions import assemble_system
        from ghost_backend.compressed.regional_coefficients import PreparedOracle
        mesh, infos, _ = prepared('coated', 'TE', 16)
        with patch.object(td, '_assemble_linear_hypersingular_matrix',side_effect=AssertionError('unfused W pass')):
            matrix, _ = assemble_system(mesh,infos,'TE')
            oracle = PreparedOracle(mesh,infos,'TE')
            ids = np.arange(oracle.n)
            native = oracle.get(ids,ids)
        np.testing.assert_allclose(native,matrix,rtol=2e-10,atol=2e-12)


class NearPairCleanupTests(unittest.TestCase):
    def test_cross_surface_consumer_closes_iterator_when_storage_fails(self):
        closed = []
        def results():
            try:
                yield object(), object()
            finally:
                closed.append(True)
        iterator = results()  # Retain a reference so cleanup cannot rely on GC.
        cross = bor.BorCrossOperators(
            bor.BorPecSolver(bor.sphere_generatrix(.1, 8), 1e9),
            bor.BorPecSolver(bor.sphere_generatrix(.09, 8), 1e9))
        cross._stream, cross._cache, cross.near_pairs = object(), {}, [(0, 0)]
        with patch.object(bor, '_iter_near_pairs', return_value=iterator), \
             patch.object(cross, '_store_near', side_effect=RuntimeError('storage failed')):
            with self.assertRaisesRegex(RuntimeError, 'storage failed'):
                cross.prepare(0)
        self.assertEqual(closed, [True])

    def test_same_surface_consumer_closes_iterator_when_contraction_fails(self):
        closed = []
        def results():
            try:
                yield np.zeros((1, 1)), None  # Invalid contraction shape.
            finally:
                closed.append(True)
        iterator = results()
        solver = bor.BorPecSolver(bor.sphere_generatrix(.1, 8), 1e9)
        with patch.object(bor, '_iter_near_pairs', return_value=iterator):
            with self.assertRaises(ValueError):
                solver._prepare_near_contractions('mfie', [(0, 0)], 0)
        self.assertEqual(closed, [True])
        self.assertEqual(solver._near_contractions, {})


if __name__ == '__main__':
    unittest.main()
