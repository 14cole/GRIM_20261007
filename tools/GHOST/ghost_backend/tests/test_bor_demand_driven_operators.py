"""Omitted BoR work must retain coefficients, reciprocity and solve checks."""
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, streaming
from ghost_backend.bor.compressed_cross import CompressedCrossFarBlocks
from ghost_backend.bor.options import option_scope, validate_options
from ghost_backend.bor.tiled import TileExpression, expression, modal_matrix


class TileAssignmentTests(unittest.TestCase):
    def counted(self, values):
        entries = []
        def query(rows, cols):
            entries.extend((int(r), int(c)) for r in rows for c in cols)
            return values[np.ix_(rows, cols)].copy()
        return TileExpression(values.shape, query), entries

    def test_full_augmented_assignments_query_each_primitive_once(self):
        matrix = modal_matrix((6, 6), compressed=True)
        values = [np.full((6, 6), x, complex) for x in (1+2j, 3-1j, 4)]
        tracked = [self.counted(value) for value in values]
        for operand, _ in tracked:
            matrix[:, :] += operand
        rows, cols = np.array([4, 1, 2]), np.array([5, 0])
        np.testing.assert_array_equal(matrix.get(rows, cols), sum(values)[np.ix_(rows, cols)])
        for _, entries in tracked:
            self.assertEqual(len(entries), len(rows)*len(cols))

    def test_partial_overwrite_never_queries_replaced_original_entries(self):
        original = np.arange(42).reshape(6, 7).astype(complex)
        matrix, old_entries = self.counted(original)
        replacement, new_entries = self.counted(np.full((3, 2), 100+3j))
        assigned_rows, assigned_cols = [4, 1, 3], [5, 0]
        matrix[assigned_rows, assigned_cols] = replacement
        wanted = original.copy()
        wanted[np.ix_(assigned_rows, assigned_cols)] = 100+3j
        rows, cols = np.array([5, 3, 1, 0, 4]), np.array([5, 2, 0, 6])
        np.testing.assert_array_equal(matrix.get(rows, cols), wanted[np.ix_(rows, cols)])
        replaced = {(r, c) for r in assigned_rows for c in assigned_cols}
        self.assertFalse(replaced.intersection(old_entries))
        self.assertEqual(len(old_entries), len(rows)*len(cols)-len(replaced))
        self.assertEqual(len(new_entries), len(replaced))

    def test_overlapping_additions_keep_immutable_slices_and_sparse_projection(self):
        original = np.arange(25).reshape(5, 5).astype(complex)
        matrix = expression(original)
        previous = matrix[[3, 1], [4, 0]]
        wanted = original.copy()
        matrix[1:4, 0:3] += 2j
        wanted[1:4, 0:3] += 2j
        matrix[2:5, 1:4] += matrix[0:3, 0:3]
        wanted[2:5, 1:4] += wanted[0:3, 0:3].copy()
        np.testing.assert_array_equal(previous.get([0, 1], [0, 1]), original[np.ix_([3, 1], [4, 0])])
        np.testing.assert_array_equal(matrix.get(np.arange(5), np.arange(5)), wanted)
        q = csr_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1j, 0, 0], [0, -1, 0]]))
        reduced = matrix.reduce(q)
        np.testing.assert_array_equal(reduced.get([2, 0, 1], [1, 2]), (q.conj().T @ wanted @ q)[np.ix_([2, 0, 1], [1, 2])])

    def test_nonintersecting_and_empty_queries_skip_replacement(self):
        values = np.arange(20).reshape(4, 5)
        matrix = expression(values)
        replacement, calls = self.counted(np.ones((1, 2)))
        matrix[[2], [1, 3]] = replacement
        np.testing.assert_array_equal(matrix.get([0, 1], [1, 3]), values[np.ix_([0, 1], [1, 3])])
        self.assertEqual(matrix.get([], [1, 3]).shape, (0, 2))
        self.assertEqual(calls, [])

    def test_full_replacement_query_does_not_expose_source_storage(self):
        values = np.arange(12).reshape(3, 4).astype(complex)
        replacement = TileExpression(values.shape, lambda rows, cols: values)
        matrix = modal_matrix(values.shape, compressed=True)
        matrix[:, :] = replacement
        tile = matrix.get(np.arange(3), np.arange(4))
        self.assertFalse(np.shares_memory(tile, values))
        tile[:] = -100
        np.testing.assert_array_equal(values, np.arange(12).reshape(3, 4))


def make_cross(need_p=True, close=True, medium=None):
    p = bor.sphere_generatrix(.035, 6)
    q = bor.sphere_generatrix(.025, 5)
    if not close:
        q = q + [0, .15]
    return bor.BorCrossOperators(bor.BorPecSolver(p, 1e9, medium=medium),
                                 bor.BorPecSolver(q, 1e9, medium=medium), need_p=need_p)


class CrossFamilyPreparationTests(unittest.TestCase):
    def test_table_and_near_preparation_skip_p_and_preserve_t(self):
        for precision in (np.complex64, np.complex128):
            with self.subTest(precision=precision):
                reference = make_cross(medium=(2.4-.15j, 1.))
                actual = make_cross(False, medium=(2.4-.15j, 1.))
                reference.sp._table_dtype = actual.sp._table_dtype = precision
                reference.prepare(2)
                with mock.patch.object(bor, 'nonnegative_bracket_tables', side_effect=AssertionError('unused P table')):
                    actual.prepare(2, workers=2)
                self.assertIsNone(actual._B)
                self.assertTrue(actual._cache[2])
                self.assertTrue(all(set(record) == {'efie'} for record in actual._cache[2].values()))
                self.assertLessEqual(actual.near_quadrature_error_max, actual.near_rtol)
                for mode in (0, 1, -1, 2):
                    expected = reference.assemble_T(mode, 2)
                    np.testing.assert_allclose(actual.assemble_T(mode, 2), expected,
                        rtol=3e-5, atol=3e-8*np.max(abs(expected)))
                with self.assertRaisesRegex(ValueError, 'EFIE only'):
                    actual.assemble_P(1, 2)

    def test_streams_skip_p_storage_sampling_and_spill_for_signed_modes(self):
        cross = make_cross(False, close=False)
        reference = streaming.StreamingCrossFarBlocks(make_cross(close=False), 3, workers=1, tile_threads=1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for store_class in (streaming.StreamingCrossFarBlocks, CompressedCrossFarBlocks):
                    for spill in (None, directory):
                        with self.subTest(store=store_class.__name__, spill=spill is not None):
                            if store_class is streaming.StreamingCrossFarBlocks:
                                patch = mock.patch.object(store_class, '_sample_brackets', side_effect=AssertionError('unused P sample'))
                            else:
                                patch = nullcontext()
                            with patch:
                                actual = store_class(cross, 3, workers=1, tile_threads=1,
                                                     mode_block=2, spill=spill, tile_budget_gb=.05)
                                try:
                                    self.assertIsNone(actual.B)
                                    for mode in (0, -1, 3, -3):
                                        np.testing.assert_allclose(actual.efie_blocks(mode), reference.efie_blocks(mode),
                                                                   rtol=2e-10, atol=1e-13)
                                    if isinstance(actual, CompressedCrossFarBlocks):
                                        self.assertEqual(set(actual._blocks), {'efie'})
                                        self.assertLessEqual(actual.evidence['stored_gb'],
                                            streaming.estimate_rectangular_streaming_gb(6, 5, 3, False))
                                    elif spill is None:
                                        self.assertEqual(actual.memory_gb(), actual.Z.nbytes / 1e9)
                                    if spill:
                                        # These small tiles stay dense; one family's disk payload is exact.
                                        self.assertEqual(actual.spilled_gb(), streaming.estimate_rectangular_streaming_gb(6, 5, 3, False))
                                    with self.assertRaisesRegex(ValueError, 'EFIE only'):
                                        actual.bracket_blocks(0)
                                finally:
                                    actual.close()
        finally:
            reference.close()

    def test_reciprocal_and_symbolic_reverse_keep_the_same_family_request(self):
        actual = make_cross(False, close=False)
        reverse = bor._reverse_cross(actual)
        self.assertFalse(reverse.need_p)
        independent = bor.BorCrossOperators(actual.sq, actual.sp, need_p=False)
        for mode in (0, 1, -1):
            np.testing.assert_allclose(reverse.assemble_T(mode, 2), independent.assemble_T(mode, 2),
                                       rtol=2e-12, atol=1e-13)
        actual.sp._compressed = True
        symbolic_reverse = bor._reverse_cross(actual)
        self.assertFalse(symbolic_reverse.need_p)
        self.assertFalse(getattr(symbolic_reverse, 'derived', False))

    def test_storage_estimates_charge_only_requested_cross_families(self):
        modes, pair_count = 9, 11
        full = bor.BorCrossStorage(24, 20, 16, pair_count, 192, (64, 128))
        filtered = full._replace(has_rotated_pv=False)
        both = bor.bor_operator_storage_bytes(modes, [], [full])
        one = bor.bor_operator_storage_bytes(modes, [], [filtered])
        self.assertEqual(one['tables'], 24*20*(modes+2)*16)
        self.assertEqual(one['near']*2, both['near'])
        self.assertLess(one['fft_workspace'], both['fft_workspace'])
        self.assertEqual(one['tables']/1e9, bor.estimate_bor_cross_table_gb(6, 5, modes, 4, 4, has_rotated_pv=False))
        full_cross, actual = make_cross(), make_cross(False)
        self.assertLess(bor.estimate_bor_operator_storage_gb(modes, [], [actual]),
                        bor.estimate_bor_operator_storage_gb(modes, [], [full_cross]))


def conductor_system():
    p = bor.sphere_generatrix(.025, 6)
    q = bor.sphere_generatrix(.020, 5) + [0, .12]
    return bor._MultiRegionBor([(p, True), (q, True)],
        [dict(medium=None, bounds=[(0, 1), (1, 1)], exterior=True)], 1e9)


def all_families():
    original = bor.BorCrossOperators.__init__
    def init(self, *args, **kwargs):
        kwargs['need_p'] = True
        original(self, *args, **kwargs)
    return mock.patch.object(bor.BorCrossOperators, '__init__', init)


class FormulationDemandTests(unittest.TestCase):
    def test_multiregion_conductor_matrix_and_fields_match_all_families(self):
        filtered = conductor_system()
        self.assertTrue(all(not cross.need_p for cross in filtered.X.values()))
        with all_families():
            reference = conductor_system()
        filtered.prepare(2)
        reference.prepare(2)
        for mode in (0, 1, 2):
            np.testing.assert_allclose(filtered.assemble(mode, 2)[0], reference.assemble(mode, 2)[0], rtol=2e-12, atol=1e-12)
        common = dict(freq_hz=1e9, thetas_deg=[0., 47., 90., 143.], n_modes=8,
                      mode_tol=1e-6, workers=1, progress=None, check_abort=None,
                      formulation='separate-conductors', extra={}, assembly='tables', table_precision='double')
        expected = bor._solve_multiregion(reference, **common)
        actual = bor._solve_multiregion(filtered, **common)
        for field in ('amp_vv', 'amp_hh'):
            np.testing.assert_allclose(actual[field], expected[field], rtol=2e-11, atol=1e-13)
        self.assertEqual(actual['modes_used'], expected['modes_used'])

    def test_symbolic_regional_additions_match_dense_modal_matrix(self):
        dense = conductor_system()
        dense.prepare(2)
        expected_modes = [dense.assemble(mode, 2)[0] for mode in (0, 1, 2)]
        with option_scope(validate_options(dict(factorization='compressed'))):
            symbolic = conductor_system()
            symbolic.prepare(2)
            for mode in (0, 1, 2):
                expected = expected_modes[mode]
                actual, _ = symbolic.assemble(mode, 2)
                self.assertIsInstance(actual, TileExpression)
                np.testing.assert_allclose(actual.get(np.arange(actual.shape[0]), np.arange(actual.shape[1])),
                                           expected, rtol=3e-12, atol=3e-12)

    def test_closed_internal_core_retains_magnetic_cross_family(self):
        surfaces = [(bor.sphere_generatrix(.055, 6), False),
                    (bor.sphere_generatrix(.025, 5), True)]
        regions = [dict(medium=None, bounds=[(0, 1)], exterior=True),
                   dict(medium=(2.4-.1j, 1.), bounds=[(0, -1), (1, 1)])]
        system = bor._MultiRegionBor(surfaces, regions, 1e9)
        self.assertTrue(system.core_cfie[1])
        self.assertTrue(all(cross.need_p for cross in system.X.values()))

    def test_partial_band_uses_union_of_both_bare_impedance_requests(self):
        def arc(start, end):
            theta = np.linspace(start, end, 4)
            points = .035*np.column_stack((np.sin(theta), np.cos(theta)))
            points[abs(points[:, 0]) < 1e-15, 0] = 0.
            return points
        covered = arc(np.pi/3, 2*np.pi/3)
        interface = covered.copy()
        interface[:, 0] += .006*np.sin(np.linspace(0, np.pi, len(interface)))
        bare = [arc(0, np.pi/3), arc(2*np.pi/3, np.pi)]
        original = bor.BorCrossOperators.__init__
        for impedances in ([None, None], [None, 60+5j], [60+5j, None]):
            captured = []
            def capture(self, *args, **kwargs):
                original(self, *args, **kwargs)
                captured.append(self)
            common = dict(freq_hz=1e9, thetas_deg=[0., 45., 90.], eps_r=2.5-.2j,
                          bare_zs=impedances, n_modes=8, workers=1, assembly='tables', table_precision='double')
            with self.subTest(impedances=impedances), mock.patch.object(bor.BorCrossOperators, '__init__', capture):
                actual = bor.solve_bor_partial_coating(interface, covered, bare, **common)
            bare_cross = [cross for cross in captured if
                np.array_equal(cross.sp.gen.nodes, bare[0]) and np.array_equal(cross.sq.gen.nodes, bare[1])]
            self.assertEqual(len(bare_cross), 1)
            self.assertEqual(bare_cross[0].need_p, any(z is not None for z in impedances))
            if not bare_cross[0].need_p:
                self.assertIsNone(bare_cross[0]._B)
            with all_families():
                expected = bor.solve_bor_partial_coating(interface, covered, bare, **common)
            for field in ('amp_vv', 'amp_hh'):
                np.testing.assert_allclose(actual[field], expected[field], rtol=3e-5, atol=1e-10)
            self.assertEqual(actual['modes_used'], expected['modes_used'])


if __name__ == '__main__':
    unittest.main()
