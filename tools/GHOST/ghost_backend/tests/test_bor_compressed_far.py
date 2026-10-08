"""Hierarchical (compressed) BoR far blocks against the dense streamed blocks."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.bor import compressed_far as cf
from ghost_backend.bor import solver as bor_solver
from ghost_backend.bor import streaming as bor_streaming
from ghost_backend.bor.near_storage import mode_sign
from ghost_backend.bor.options import option_scope, validate_options

C0 = bor_solver.C0
FREQUENCY_HZ = 1.0e9


def _solver(elements=160, ka=6.0):
    radius = ka * C0 / (2.0 * math.pi * FREQUENCY_HZ)
    return bor_solver.BorPecSolver(bor_solver.sphere_generatrix(radius, elements), FREQUENCY_HZ)


def _small_tree():
    """Leaves and cross-approximation sizes that give a 161-node surface every block kind."""
    return (mock.patch.object(cf, "FAR_COMPRESSION_LEAF", 12),
            mock.patch.object(cf, "FAR_COMPRESSION_ACA_MIN_NODES", 20))


def _relative(actual, expected):
    difference = sum(float(np.sum(np.abs(a - e) ** 2)) for a, e in zip(actual, expected))
    total = sum(float(np.sum(np.abs(e) ** 2)) for e in expected)
    return math.sqrt(difference / max(total, 1e-300))


class PartitionTests(unittest.TestCase):
    def test_partition_covers_every_node_pair_once_and_is_symmetric(self):
        solver = _solver(120)
        clusters = cf.cluster_tree(solver.Nn, 8)
        admissible, dense = cf.block_partition(clusters, solver.gen.nodes,
                                               solver._near_sources_by_element)
        self.assertTrue(admissible)
        cover = np.zeros((solver.Nn, solver.Nn), int)
        for i, j in admissible + dense:
            (a0, a1, _), (b0, b1, _) = clusters[i], clusters[j]
            cover[a0:a1, b0:b1] += 1
        np.testing.assert_array_equal(cover, 1)
        self.assertEqual(set(admissible), {(j, i) for i, j in admissible})
        self.assertEqual(set(dense), {(j, i) for i, j in dense})
        # No admissible block holds a near element pair.
        near = solver._near_sources_by_element
        for i, j in admissible:
            (a0, a1, _), (b0, b1, _) = clusters[i], clusters[j]
            e0, e1 = max(a0 - 1, 0), min(a1, solver.gen.n_elems)
            f0, f1 = max(b0 - 1, 0), min(b1, solver.gen.n_elems)
            for e in range(e0, e1):
                self.assertFalse(any(f0 <= f < f1 for f in near[e]))


class CompressedBlockTests(unittest.TestCase):
    def test_every_mode_and_family_matches_the_streamed_blocks(self):
        solver = _solver()
        m_max = 9
        leaf, aca = _small_tree()
        with leaf, aca:
            store = cf.CompressedFarBlocks(solver, m_max, efie=True, mfie=True, workers=2)
        dense = bor_streaming.StreamingFarBlocks(solver, m_max, efie=True, mfie=True)
        self.assertGreater(store.evidence["lowrank_blocks"], 0)
        self.assertLess(store.memory_gb(), dense.memory_gb())
        Nn = solver.Nn
        for m in range(m_max + 1):
            self.assertLess(_relative(store.efie_blocks(m), dense.efie_blocks(m)), 1e-9)
            expected = tuple(np.zeros((Nn, Nn), complex) for _ in range(4))
            dense.add_blocks("mfie", m, expected, 1.0)
            self.assertLess(_relative(store.bracket_blocks("mfie", m), expected), 1e-9)
        # The public write paths: signed modes, scale, in-place accumulation.
        for m in (-3, 4):
            quads = tuple(np.full((Nn, Nn), np.nan, complex) for _ in range(4))
            reference = tuple(np.empty((Nn, Nn), complex) for _ in range(4))
            store.write_efie_blocks(m, quads, 0.25 - 1j)
            dense.write_efie_blocks(m, reference, 0.25 - 1j)
            self.assertTrue(all(np.all(np.isfinite(quad)) for quad in quads))
            self.assertLess(_relative(quads, reference), 1e-9)
            targets = tuple(np.ones((Nn, Nn), complex) for _ in range(4))
            reference = tuple(np.ones((Nn, Nn), complex) for _ in range(4))
            store.add_blocks("mfie", m, targets, -0.5)
            dense.add_blocks("mfie", m, reference, -0.5)
            self.assertLess(_relative(targets, reference), 1e-9)
            blocks, signs = store.stored_blocks("efie", m)
            self.assertEqual(signs, tuple(mode_sign(uv, m) for uv in range(4)))
        store.close()
        with self.assertRaises(RuntimeError):
            store.efie_blocks(0)
        dense.close()

    def test_rotated_pv_families_match(self):
        solver = _solver(120, 4.0)
        zs = np.linspace(20.0, 80.0, solver.P) + 5j
        leaf, aca = _small_tree()
        for kwargs in (dict(ibc_zs_pt=zs), dict(pmchwt=True)):
            with self.subTest(**{key: True for key in kwargs}):
                with leaf, aca:
                    store = cf.CompressedFarBlocks(solver, 6, efie=False, **kwargs)
                dense = bor_streaming.StreamingFarBlocks(solver, 6, efie=False, **kwargs)
                self.assertIsNotNone(store.B)
                self.assertIsNone(store.Z)
                self.assertEqual(store.rot_pv_unit_source, bool(kwargs.get("pmchwt")))
                for m in range(7):
                    expected = tuple(np.zeros((solver.Nn, solver.Nn), complex) for _ in range(4))
                    dense.add_blocks("ibc", m, expected, 1.0)
                    self.assertLess(_relative(store.bracket_blocks("ibc", m), expected), 1e-9)

    def test_broken_process_pool_finishes_on_threads(self):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        solver = _solver(80, 3.0)
        geometry = cf._Geometry(solver, 3, ("mfie",), None, False)
        executor = cf._Executor(geometry, 2, 0)

        class BrokenPool:
            def submit(self, *args):
                future = Future()
                future.set_exception(BrokenProcessPool("worker died"))
                return future

            def shutdown(self, **kwargs):
                pass

        executor.pool, executor.workers = BrokenPool(), 2
        jobs = [("leaf", "mfie", (0, 20), (0, 20)), ("leaf", "mfie", (20, 40), (0, 20))]
        results = executor.run(jobs, None, 1e-10, solver.Nn, lambda: None)
        executor.close()
        self.assertEqual(set(results), {("mfie", (0, 20), (0, 20)), ("mfie", (20, 40), (0, 20))})
        self.assertIn("process pool failed", executor.backend)
        reference = cf._Tiles(geometry).block("mfie", (20, 40), (0, 20))
        np.testing.assert_allclose(results[("mfie", (20, 40), (0, 20))][1],
                                   reference.transpose(1, 0, 2, 3), rtol=1e-13, atol=0)

    def test_single_precision_storage(self):
        solver = _solver(120, 4.0)
        leaf, aca = _small_tree()
        with leaf, aca:
            double = cf.CompressedFarBlocks(solver, 5, efie=True, mfie=True)
            single = cf.CompressedFarBlocks(solver, 5, efie=True, mfie=True, dtype=np.complex64)
        # Single retains the original expanded-factor rounding. Double may
        # now share Q across slices, so half of its *expanded* payload is the
        # correct single-precision expectation.
        expected = 0
        for store in double._blocks.values():
            for kind, payload in store.values():
                if kind == 'dense':
                    expected += payload.nbytes // 2
                elif kind == 'shared':
                    basis, rows = payload
                    expected += sum(8*basis.shape[0]*left.shape[1]+right.nbytes//2
                                    for row in rows for left,right in row)
                else:
                    expected += sum((left.nbytes+right.nbytes)//2 for row in payload for left,right in row)
        self.assertAlmostEqual(single.memory_gb(), expected/1e9, places=12)
        self.assertLess(_relative(single.efie_blocks(3), double.efie_blocks(3)), 1e-6)


class SelectionAndPlanningTests(unittest.TestCase):
    def test_option_selects_store_and_prices_it(self):
        with self.assertRaises(ValueError):
            validate_options(dict(far_compression="always"))
        for setting, expected in (("on", True), ("off", False), ("auto", False)):
            with option_scope(validate_options(dict(far_compression=setting))):
                self.assertEqual(cf.far_compression_selected(161), expected)
        with option_scope(validate_options(dict(far_compression="auto"))):
            self.assertTrue(cf.far_compression_selected(cf.FAR_COMPRESSION_MIN_NODES))
        elements, modes = 4000, 40
        with option_scope(validate_options(dict(far_compression="off"))):
            dense = bor_streaming.estimate_streaming_gb(elements, modes, "cfie")
        with option_scope(validate_options(dict(far_compression="on"))):
            compressed = bor_streaming.estimate_streaming_gb(elements, modes, "cfie")
            self.assertEqual(compressed, cf.estimate_compressed_far_gb(elements, modes, "cfie"))
            self.assertEqual(bor_streaming.estimate_streaming_block_gb(elements, modes, 1, "cfie"),
                             compressed / (modes + 1))
            # The retained compressed band now obeys the same hard budget.
            block, held, workers = bor_streaming.plan_streaming_mode_block(
                elements, modes, "cfie", False, False, compressed / 10, 8)
            self.assertEqual((block, workers), (4, 4))
            self.assertLessEqual(held, compressed / 10)
            self.assertEqual(bor_streaming.plan_stream_spill(block, modes + 1, compressed,
                                                            allow_spill=False)[0], None)
        self.assertLess(compressed, dense / 5)

    def test_solve_bor_matches_dense_streaming(self):
        radius = 6.0 * C0 / (2.0 * math.pi * FREQUENCY_HZ)
        points = bor_solver.sphere_generatrix(radius, 160)
        thetas = np.linspace(0.0, 180.0, 13)
        results = {}
        leaf, aca = _small_tree()
        for setting in ("off", "on"):
            with leaf, aca, option_scope(validate_options(dict(far_compression=setting))):
                results[setting] = bor_solver.solve_bor(points, FREQUENCY_HZ, thetas,
                                                        assembly="streaming", workers=2)
        compressed = results["on"]["stream_far_compression"]
        self.assertIsNotNone(compressed)
        self.assertIsNone(results["off"]["stream_far_compression"])
        self.assertEqual(results["on"]["stream_spill_gb"], 0.0)
        for key in ("amp_vv", "amp_hh"):
            expected = np.asarray(results["off"][key])
            np.testing.assert_allclose(results["on"][key], expected, rtol=0,
                                       atol=1e-8 * float(np.max(np.abs(expected))))


if __name__ == "__main__":
    unittest.main()
