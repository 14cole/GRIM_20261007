"""BoR compute reuse preserves complex operators and bounds retained memory."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor


def _retained_angular_bytes(record):
    arrays = [value for value in vars(record).values() if isinstance(value, np.ndarray)]
    arrays += [value for pair in record._checkpoints.values() for value in pair]
    return sum(value.nbytes for value in arrays)


class AngularCheckpointTests(unittest.TestCase):
    def test_checkpoints_preserve_every_recurrence_order_and_direct_fallback(self):
        rho = np.r_[np.linspace(.01, .3, 25), 0., 1e-12]
        z = np.linspace(-2., 2., rho.size)
        aspects = np.array([0., 1e-9, .5, 25., 90., 165., 180.])
        for top in (16, 65, 145, 400):
            plain = bor._AngularChunk(400., rho, z, aspects, top)
            orders = bor._angular_checkpoint_orders(top, bor.ANGULAR_CHECKPOINT_MAX_COUNT)
            record = bor._AngularChunk(400., rho, z, aspects, top, orders)
            self.assertLessEqual(len(orders), bor.ANGULAR_CHECKPOINT_MAX_COUNT)
            self.assertEqual(record.nbytes, _retained_angular_bytes(record))
            self.assertEqual(record.direct.dtype, np.bool_)
            modes = list(range(top)) if top < 150 else list(range(0, top, 7)) + [top - 1]
            for mode in modes + [-1, -7, -(top - 1)]:
                for actual, expected in zip(record.triplet(mode), plain.triplet(mode)):
                    np.testing.assert_array_equal(actual, expected)
            self.assertTrue(all(not value.flags.writeable
                                for pair in record._checkpoints.values() for value in pair))

    def test_parallel_modes_share_immutable_checkpoints(self):
        rho = np.linspace(.05, .2, 64)
        z = np.linspace(-1., 1., 64)
        aspects = np.array([0., 45., 90., 180.])
        record = bor._AngularChunk(150., rho, z, aspects, 128,
                                  bor._angular_checkpoint_orders(128, 8))
        modes = [0, 32, 64, 96, 127, -1, -16, -79] * 4
        expected = [record.triplet(mode) for mode in modes]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(record.triplet, modes))
        for observed, reference in zip(actual, expected):
            for value, wanted in zip(observed, reference):
                np.testing.assert_array_equal(value, wanted)

    def test_reservation_matches_actual_arrays_and_degrades_under_tight_budget(self):
        gc.collect()
        before = bor._ANGULAR_CACHE_USED[0]
        surface = bor.BorPecSolver(bor.sphere_generatrix(.1, 12), 1e9)
        surface._angular_top = 128
        aspects = np.zeros(4)
        count = surface.P * len(aspects)
        allowance = bor.ANGULAR_BASE_BYTES_PER_VALUE * count + 16 * len(aspects) + 16 * count
        with mock.patch.object(bor, 'ANGULAR_CACHE_BUDGET_BYTES', before + allowance):
            record = surface._shared_angular_chunk(aspects, 2)
            self.assertIsNotNone(record)
            self.assertEqual(len(record._checkpoints), 1)
            self.assertEqual(record.nbytes, _retained_angular_bytes(record))
            self.assertEqual(bor._ANGULAR_CACHE_USED[0] - before, record.nbytes)
            self.assertLessEqual(bor._ANGULAR_CACHE_USED[0], before + allowance)
            # A different chunk cannot displace or overrun the admitted one.
            self.assertIsNone(surface._shared_angular_chunk(aspects + 1., 2))
        surface = record = None
        gc.collect()
        self.assertEqual(bor._ANGULAR_CACHE_USED[0], before)

    def test_failed_construction_releases_its_reservation(self):
        before = bor._ANGULAR_CACHE_USED[0]
        surface = bor.BorPecSolver(bor.sphere_generatrix(.1, 8), 1e9)
        with mock.patch.object(bor, '_AngularChunk', side_effect=RuntimeError('construction failed')):
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                surface._shared_angular_chunk(np.array([0., 90.]), 2)
        self.assertEqual(bor._ANGULAR_CACHE_USED[0], before)


class NonnegativeNearTests(unittest.TestCase):
    def test_production_kernels_and_contractions_drop_no_positive_modes(self):
        surface = bor.BorPecSolver(bor.sphere_generatrix(.12, 10), 1e9)
        cap = 12
        for pair in ((3, 3), (3, 4), (3, 6)):
            e, f = pair
            points = (bor._same_surface_points(surface.gen, e, f, ('efie', 'mfie', 'ibc'), 4)
                      if abs(e - f) <= 1 else bor._regular_cell_points(8))
            args = (surface.gen, e, surface.gen, f, surface.k, cap, ('efie', 'mfie', 'ibc'), points)
            signed = bor._contract_near_points(*args, signed=True)
            with mock.patch.object(bor, 'mfie_kernels_near', wraps=bor.mfie_kernels_near) as mfie, \
                    mock.patch.object(bor, 'ibc_kernels_near', wraps=bor.ibc_kernels_near) as ibc:
                half = bor._contract_near_points(*args, signed=False)
            self.assertTrue(mfie.called and ibc.called)
            self.assertTrue(all(call.kwargs['signed'] is False for call in mfie.call_args_list + ibc.call_args_list))
            for kind in half:
                self.assertEqual(half[kind].shape, (4, cap + 1, 2, 2))
                np.testing.assert_allclose(half[kind], signed[kind][:, cap:], rtol=2e-13, atol=1e-15)


class ModeCpuAllocationTests(unittest.TestCase):
    def test_nested_teams_share_the_unit_allocation_and_preserve_its_profile(self):
        from ghost_backend.compressed import operator
        from ghost_backend.execution import options as execution

        reference = None
        for workers, allocation in ((1, 8), (4, 8), (12, 2)):
            for profile in (None, {'assembly_threads': 6}):
                with self.subTest(workers=workers, allocation=allocation, profile=profile):
                    seen = []

                    def capture(stage):
                        seen.append((stage, execution.allocated_cpu_budget(),
                                     operator._thread_budget(), execution.current_options()))

                    def assemble(mode):
                        capture('assemble')
                        # Actual nested executor threads must inherit the mode's
                        # share; observing only the parent thread misses this.
                        operator._run_groups(range(4), lambda group: capture('nested'), 8)
                        return np.eye(2, dtype=complex), None

                    def rhs(mode, theta, pol):
                        capture('rhs')
                        return np.ones(2, complex)

                    def farfield(mode, value, theta, pol):
                        capture('farfield')
                        return complex(value[0]) * .01 ** abs(mode)

                    with mock.patch.object(execution, '_usable_logical_cpus', return_value=16), \
                            execution.cpu_allocation_scope(allocation), \
                            (nullcontext() if profile is None else execution.execution_scope(profile)):
                        original_profile = execution.current_options()
                        with mock.patch.object(operator, 'ThreadPoolExecutor', wraps=ThreadPoolExecutor) as nested, \
                                mock.patch.object(bor, 'ThreadPoolExecutor', wraps=ThreadPoolExecutor) as modes:
                            fields, _, stats = bor._mode_sweep(
                                2, [30., 90.], ['VV', 'HH'], 3, 1e-6,
                                assemble, rhs, farfield, workers=workers)
                        admitted = stats['modal_execution']['worker_plan']['workers']
                        self.assertEqual(admitted, min(workers, allocation, 4))
                        self.assertEqual(stats['modal_execution']['worker_plan']['requested_workers'], workers)
                        self.assertEqual(stats['modal_execution']['worker_plan']['cpu_budget'], allocation)
                        self.assertEqual(modes.call_args.kwargs['max_workers'], admitted)
                        share = allocation // admitted
                        team_size = min(share, 6) if profile else share
                        self.assertEqual(stats['modal_execution']['cpu_budget_per_mode'], share)
                        self.assertEqual({value[0] for value in seen},
                                         {'assemble', 'nested', 'rhs', 'farfield'})
                        self.assertTrue(all(value[1:] == (share, team_size, original_profile)
                                            for value in seen))
                        self.assertEqual(nested.called, team_size > 1)
                        self.assertTrue(all(call.kwargs['max_workers'] == min(4, team_size)
                                            for call in nested.call_args_list))
                        self.assertEqual(execution.allocated_cpu_budget(), allocation)
                        self.assertEqual(execution.current_options(), original_profile)
                    if reference is None:
                        reference = fields
                    else:
                        np.testing.assert_array_equal(fields, reference)


def _reference_p(surface, mode, cap):
    tt, tf, ft, ff = surface._rot_pv_blocks(mode, cap)
    return np.block([[-tf, tt], [-ff, ft]])


class MaterialAssemblyReuseTests(unittest.TestCase):
    def test_destination_assembly_matches_independent_operators(self):
        for streaming in (False, True):
            exterior = bor.BorPecSolver(bor.sphere_generatrix(.1, 10), 1e9)
            interior = bor.BorPecSolver(bor.sphere_generatrix(.1, 10), 1e9,
                                       medium=(2.7 - .2j, 1.1 - .03j))
            cap, size = 4, 2 * exterior.Nn
            try:
                for surface in (exterior, interior):
                    if streaming:
                        surface.enable_streaming(cap, efie=True, pmchwt=True,
                                                 tile_budget_gb=.1, workers=1)
                    surface.prepare_operators(cap, efie=True, ibc=True, workers=1)
                for mode in (0, 1, -1, 4):
                    pe, pi = (_reference_p(surface, mode, cap) for surface in (exterior, interior))
                    # Destination can be a strided submatrix of the vehicle system.
                    holder = np.full((size + 3, size + 3), 123 + 4j)
                    destination = holder[:size, :size]
                    self.assertIs(exterior.assemble_pmchwt_P(mode, cap, out=destination), destination)
                    np.testing.assert_array_equal(destination, pe)
                    np.testing.assert_array_equal(holder[size:, :], 123 + 4j)
                    te, ti = (surface.assemble_mode(mode, cap) for surface in (exterior, interior))
                    for weight in (1., np.exp(.25j)):
                        ratio = (bor.ETA0 / interior.eta) ** 2
                        combined = bor.ETA0 * (pe + weight * pi)
                        if weight != 1.:
                            combined = bor._add_rotation_mass_into(
                                combined, exterior, -(0.5 * (1. - weight) * bor.ETA0))
                        reference = np.block([[te + weight * ti, -combined],
                                              [combined, te + (weight * ratio) * ti]])
                        holder = np.full((2 * size + 5, 2 * size + 5), 123 + 4j)
                        target = holder[:2 * size, :2 * size]
                        bor._assemble_pmchwt_interface_into(target, exterior, interior,
                                                            mode, cap, weight, ratio)
                        np.testing.assert_allclose(target, reference, rtol=2e-14, atol=2e-15)
                        np.testing.assert_array_equal(holder[2 * size:, :], 123 + 4j)
                self.assertIsNone(exterior._B_T)
                self.assertIsNone(interior._B_T)
            finally:
                exterior.close_streaming()
                interior.close_streaming()

    def test_production_storage_does_not_reserve_unused_dense_bases(self):
        surface = bor.BorSurfaceStorage(300, 101, 16, 500, True, True, True, None)
        for streaming in (False, True):
            parts = bor.bor_operator_storage_bytes(20, [surface], streaming=streaming)
            self.assertEqual(parts['basis'], 0.)
            self.assertGreater(parts['near'], 0.)


def _efie_near_asymmetry(pairs, values) -> 'float':
    """Largest relative violation of EFIE reciprocity over retained near blocks.

    ``values`` is the ``[4, modes, 4 * len(pairs)]`` retained storage of
    ``_prepare_near_contractions`` (2x2 blocks flattened row-major).
    """
    if not len(pairs):
        return 0.0
    modes = values.shape[1]
    blocks = values.reshape(4, modes, len(pairs), 2, 2)
    scale = np.max(np.abs(values), axis=(0, 2))
    scale = np.where(scale > 0.0, scale, 1.0)
    index = {tuple(pair): i for i, pair in enumerate(pairs)}
    worst = 0.0
    for (e, f), i in index.items():
        j = index.get((f, e))
        if j is None or j < i:
            continue
        own, mirror = blocks[:, :, i], blocks[:, :, j].transpose(0, 1, 3, 2)
        violation = np.max(np.abs(np.stack([
            own[0] - mirror[0],
            own[3] - mirror[3],
            own[1] + mirror[2],
            own[2] + mirror[1],
        ])), axis=(0, 2, 3))
        worst = max(worst, float(np.max(violation / scale)))
    return worst


class CompactModalStorageTests(unittest.TestCase):
    def test_raw_reciprocity_diagnostic_survives_coalescing(self):
        from ghost_backend.bor.near_storage import EfieReciprocity, reciprocal_pair_order
        rng = np.random.default_rng(733)
        pairs = [(0,0), (0,1), (0,2), (1,0), (1,1), (2,0)]
        raw = rng.normal(size=(4,5,len(pairs),2,2)) + 1j*rng.normal(size=(4,5,len(pairs),2,2))
        expected = _efie_near_asymmetry(pairs, raw.reshape(4,5,-1))
        diagnostic = EfieReciprocity(5)
        for pair in reciprocal_pair_order(pairs):
            diagnostic.add(pair, raw[:,:,pairs.index(pair)])
        self.assertEqual(diagnostic.value(), expected)

    def test_shared_basis_preserves_slices_without_another_approximation(self):
        from ghost_backend.bor.compressed_far import _slices, CompressedFarBlocks
        rng = np.random.default_rng(31)
        p,s,nm,r,k = 100,80,6,12,3
        U = rng.normal(size=(p,r)) + 1j*rng.normal(size=(p,r))
        pieces = [(rng.normal(size=(r,k))+1j*rng.normal(size=(r,k))) @
                  (rng.normal(size=(k,s))+1j*rng.normal(size=(k,s))) for _ in range(4*nm)]
        V = np.stack(pieces).reshape(4,nm,r,s).transpose(2,0,1,3).reshape(r,-1)
        weights = np.linspace(.5,1.5,nm)
        original = _slices(U,V,nm,s,weights,1e-9)
        shared = _slices(U,V,nm,s,weights,1e-9,shared=True)
        self.assertEqual(shared[0], 'shared')
        basis, rows = shared[1]
        old_bytes = sum(a.nbytes+b.nbytes for row in original for a,b in row)
        new_bytes = basis.nbytes+sum(a.nbytes+b.nbytes for row in rows for a,b in row)
        self.assertLess(new_bytes, old_bytes)
        for m in range(nm):
            actual = CompressedFarBlocks._values(shared,m)
            for uv,(left,right) in enumerate(original[m]):
                np.testing.assert_array_equal(basis @ rows[m][uv][0],left)
                np.testing.assert_array_equal(actual[uv],left@right)


if __name__ == '__main__':
    unittest.main()
