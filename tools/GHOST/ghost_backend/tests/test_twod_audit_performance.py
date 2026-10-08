"""Quadrature-axis correctness and bounded, equivalent near assembly."""
from pathlib import Path
import gc
import sys
import unittest
import weakref
from unittest import mock
from concurrent.futures import Future

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import operators as ops, polynomial_quadrature as pq
from ghost_backend.twod.geometry import LinearElement, LinearMesh, LinearNode
from ghost_backend.execution.cpu import CPUState, _STATE
from ghost_backend.twod.assembly.native import far
from test_audit_fixes_2d_operators import mesh_of, circle, fused


def separated_mesh():
    nodes, elements = [], []
    for index, (a, b) in enumerate((((0., 0.), (.001, 0.)), ((4., .2), (5., .4)))):
        a, b = np.asarray(a), np.asarray(b)
        length = np.linalg.norm(b - a)
        tangent = (b - a) / length
        nodes.extend((LinearNode(a, (2*index, 0)), LinearNode(b, (2*index+1, 0))))
        elements.append(LinearElement('pair', 2, 0, 0, 0, (2*index, 2*index+1), a, b,
            (a+b)/2, tangent, np.array([-tangent[1], tangent[0]]), length, index))
    return LinearMesh(nodes, elements)


class QuadratureAndMemoryTests(unittest.TestCase):
    def test_unequal_far_rules_match_independent_tensor_quadrature(self):
        mesh = separated_mesh()
        for observer, source, obs_order, src_order in ((0, 1, 2, 32), (1, 0, 32, 2)):
            for derivative in (False, True):
                for k in (15., 15.-.7j):
                    with self.subTest(observer=observer, derivative=derivative, k=k), \
                            mock.patch.object(ops, '_FAR_GRADED', False), \
                            _STATE.override(CPUState()), \
                            mock.patch.object(far, 'far_block', side_effect=AssertionError('Unequal native rule')):
                        mask = np.arange(2) == source
                        rows, cols = mesh.elements[observer].node_ids, mesh.elements[source].node_ids
                        actual = ops._assemble_linear_operator_matrices_multi(mesh, k, derivative, [mask],
                            obs_order=obs_order, src_order=src_order, output_node_ids_many=[(rows, cols)])[0]
                    expected = ops._integrate_linear_pair_box_sk_vectorized(
                        mesh.elements[observer], mesh.elements[source], k, derivative,
                        (0., 1.), (0., 1.), obs_order, src_order)
                    for observed, reference in zip(actual, expected):
                        np.testing.assert_allclose(observed.values, reference, rtol=2e-12, atol=1e-18)

    def test_near_contractions_match_scalar_pair_rules(self):
        mesh = mesh_of([(0., 0.), (.07, .01), (.08, .04), (.02, .09)])
        pairs = [(mesh.elements[0], mesh.elements[2]), (mesh.elements[1], mesh.elements[3])]
        touching = [(mesh.elements[0], mesh.elements[1]), (mesh.elements[2], mesh.elements[1])]
        shared = [(False, True), (True, False)]
        for k in (12., 12.-3j):
            for derivative in (False, True):
                for want_s, want_k in ((True, True), (True, False), (False, True)):
                    with self.subTest(k=k, derivative=derivative, channels=(want_s, want_k)):
                        actual = ops._integrate_linear_pairs_box_sk_batched(mesh.elements, [0, 1], [2, 3],
                            k, derivative, 16, want_s, want_k)
                        references = [ops._integrate_linear_pair_box_sk_vectorized(a, b, k, derivative,
                            (0., 1.), (0., 1.), 16, 16, want_s, want_k) for a, b in pairs]
                        for channel in (0, 1):
                            np.testing.assert_allclose(actual[channel], [r[channel] for r in references],
                                                       rtol=3e-13, atol=1e-17)
                        actual = ops._integrate_linear_touching_pairs_sk_batched(
                            touching, shared, k, derivative, 9, want_s, want_k)
                        references = [ops._integrate_linear_touching_duffy_sk_vectorized(a, b, k,
                            derivative, (0., 1.), (0., 1.), *ends, 9, want_s, want_k)
                            for (a, b), ends in zip(touching, shared)]
                        for channel in (0, 1):
                            np.testing.assert_allclose(actual[channel], [r[channel] for r in references],
                                                       rtol=3e-13, atol=1e-17)

    def test_unrequested_full_near_arrays_are_not_allocated(self):
        mesh = mesh_of(circle(.06, 24))
        original_zeros = np.zeros
        original_expand, original_classify = ops._expand_near_chunks, ops._near_fixed_order_positions
        for want_s, want_k in ((True, False), (False, True), (True, True)):
            state = dict(shape=None, classify=False, allocations=0)
            def expand(*args, **kwargs):
                pair_ids = original_expand(*args, **kwargs)
                state['shape'] = (len(pair_ids[0]), 2, 2)
                return pair_ids
            def classify(*args, **kwargs):
                state['classify'] = True
                return original_classify(*args, **kwargs)
            def zeros(shape, *args, **kwargs):
                if not state['classify'] and isinstance(shape, tuple) and shape == state['shape']:
                    state['allocations'] += 1
                return original_zeros(shape, *args, **kwargs)
            with mock.patch.object(ops, '_expand_near_chunks', expand), \
                    mock.patch.object(ops, '_near_fixed_order_positions', classify), mock.patch.object(np, 'zeros', zeros):
                result = ops._assemble_linear_operator_matrices(mesh, 12., True,
                    compute_single_layer=want_s, compute_double_layer=want_k)
            self.assertEqual(state['allocations'], int(want_s) + int(want_k))
            self.assertTrue(all(np.isfinite(a).all() for a in result))

    def test_near_map_bounds_pending_work_and_releases_consumed_results(self):
        submitted, shutdown = [], []
        class ImmediatePool:
            def __init__(self, max_workers):
                self.workers = max_workers
            def submit(self, function, job):
                submitted.append(job)
                future = Future()
                future.set_result(function(job))
                return future
            def shutdown(self, **kwargs):
                shutdown.append(kwargs)
        with mock.patch('concurrent.futures.ThreadPoolExecutor', ImmediatePool):
            iterator = pq.map_checked(lambda value: np.array([value]), list(range(20)), 2)
            first = next(iterator)
            self.assertEqual(len(submitted), 4)
            ref = weakref.ref(first)
            del first
            second = next(iterator)
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(int(second[0]), 1)
            self.assertEqual(len(submitted), 5)
            iterator.close()
        self.assertEqual(shutdown, [dict(wait=True, cancel_futures=True)])

    def test_chunked_scatter_preserves_all_four_operators(self):
        mesh = mesh_of(circle(.06, 40))
        reference = fused(mesh, 12.-.7j)
        maue_sizes = []
        original = ops._maue_blocks
        def bounded(blocks, *args, **kwargs):
            maue_sizes.append(len(blocks))
            return original(blocks, *args, **kwargs)
        with mock.patch.object(ops, '_NEAR_SCATTER_MAX_PAIRS', 3), \
                mock.patch.object(ops, '_maue_blocks', bounded):
            actual = fused(mesh, 12.-.7j)
        self.assertTrue(maue_sizes)
        self.assertLessEqual(max(maue_sizes), 3)
        for result, expected in zip(actual, reference):
            np.testing.assert_array_equal(result, expected)
        coefficients = np.linspace(.2, 1.3, len(mesh.elements))*(1.+.7j)
        reference = ops._assemble_linear_operator_matrices(mesh, 12.-.7j, True,
            single_layer_observation_coefficients=coefficients)
        with mock.patch.object(ops, '_NEAR_SCATTER_MAX_PAIRS', 3):
            actual = ops._assemble_linear_operator_matrices(mesh, 12.-.7j, True,
                single_layer_observation_coefficients=coefficients)
        for result, expected in zip(actual, reference):
            np.testing.assert_array_equal(result, expected)

    def test_near_map_stops_admitting_jobs_on_cancellation(self):
        calls = []
        def checkpoint():
            if len(calls) == 2:
                raise InterruptedError('cancelled')
        def run(index):
            calls.append(index)
            return index
        with self.assertRaises(InterruptedError):
            list(pq.map_checked(run, list(range(100)), 1, checkpoint))
        self.assertEqual(calls, [0, 1])


if __name__ == '__main__':
    unittest.main()
