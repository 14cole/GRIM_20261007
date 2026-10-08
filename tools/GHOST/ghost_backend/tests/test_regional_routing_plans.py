"""Shared regional maps retain exact coefficients, queries and worker ownership."""
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from ghost_backend.compressed import regional_coefficients as regional
from ghost_backend.execution.options import execution_scope, validate_options
from test_compact_multi_region import prepared


class RegionalRoutingPlanTests(unittest.TestCase):
    def setUp(self):
        scope = execution_scope(validate_options(dict(assembly_threads=1, blas_threads=1)))
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_reused_maps_match_uncached_routes_for_unsorted_queries(self):
        rng = np.random.default_rng(513)
        for kind in ('pec', 'ibc', 'lossy', 'magnetic', 'coated', 'layered', 'mixed', 'equal_k'):
            for pol in ('TE', 'TM'):
                with self.subTest(kind=kind, polarization=pol):
                    mesh, infos, _ = prepared(kind, pol, 16)
                    oracle = regional.PreparedOracle(mesh, infos, pol, cut=32)
                    select = regional._selected_route
                    def uncached(*args):
                        return select(*args[:-1])
                    n = oracle.n
                    cases = [(rng.permutation(n), rng.permutation(n)),
                             (rng.permutation(n)[:n//2], rng.permutation(n)[:n//3]),
                             (np.array([], int), np.arange(n))]
                    for rows, columns in cases:
                        expected = oracle.get_with_error(rows, columns)
                        with patch.object(regional, '_selected_route', side_effect=uncached):
                            actual = oracle.get_with_error(rows, columns)
                        for a, b in zip(actual, expected):
                            np.testing.assert_array_equal(a, b)

    def test_interning_keeps_distinct_maps_and_weights_separate(self):
        first = np.array([-1, 4, 1, -1, 7])
        second = np.array([2, -1, 6, 3, -1])
        maps = {}
        a = regional._route_plan(first, second, np.ones(5), maps)
        b = regional._route_plan(first.copy(), second.copy(), np.arange(5.), maps)
        c = regional._route_plan(second, first, np.ones(5), maps)
        for index in (1, 2, 3, 4):
            self.assertIs(a[index], b[index])
        self.assertIsNot(a[0], b[0])
        self.assertIsNot(a[3], c[3])
        self.assertEqual(len(maps), 2)
        rows, columns = np.array([7, 1]), np.array([6, 2])
        rd, cd = np.full(8, -1), np.full(8, -1)
        rd[rows], cd[columns] = np.arange(2), np.arange(2)
        selected = ({}, {})
        ar = regional._selected_route(a, rd, cd, (1, 7), (2, 6), 5, selected)
        br = regional._selected_route(b, rd, cd, (1, 7), (2, 6), 5, selected)
        self.assertIs(ar[0][0], br[0][0])
        self.assertIs(ar[0][1], br[0][1])
        np.testing.assert_array_equal(ar[0][0], [-1, -1, 1, -1, 0])
        np.testing.assert_array_equal(ar[0][1], [1, -1, 0, -1, -1])

    def test_map_reuse_is_query_local_and_survives_worker_serialization(self):
        mesh, infos, _ = prepared('layered', 'TE', 24)
        oracle = regional.PreparedOracle(mesh, infos, 'TE', cut=32)
        copy = pickle.loads(pickle.dumps(oracle))
        ids = np.arange(oracle.n)
        first = oracle.get(ids[::-1], ids)
        oracle.get(ids[::2], ids[1::2])
        np.testing.assert_array_equal(first, oracle.get(ids[::-1], ids))
        np.testing.assert_array_equal(first, copy.get(ids[::-1], ids))

    def test_open_material_junction_matches_dense_regional_assembly(self):
        from test_2d_capability_acceptance import TopologyAcceptanceTests
        from ghost_backend.twod import solver
        from ghost_backend.twod.formulations import regions
        snapshot = TopologyAcceptanceTests._partial_coating(4)
        materials = solver.MaterialLibrary.from_entries(snapshot['ibcs'], snapshot['dielectrics'], base_dir='.')
        frequency = .6
        k0 = 2*np.pi*frequency*1e9/solver.C0
        wavelength = solver._mesh_wavelength_for_snapshot(snapshot, materials, frequency)[0]
        panels = solver._build_panels(snapshot, 1., wavelength)
        for pol in ('TE', 'TM'):
            infos = solver._build_coupled_panel_info(panels, materials, frequency, pol, k0)
            mesh, _ = solver._build_linear_mesh_interface_aware(panels, infos)
            oracle = regional.PreparedOracle(mesh, infos, pol, cut=None)
            reference, _ = regions._assemble_system_fresh(mesh, infos, pol)
            ids = np.random.default_rng(78).permutation(oracle.n)
            np.testing.assert_allclose(oracle.get(ids, ids[::-1]), reference[np.ix_(ids, ids[::-1])],
                                       rtol=2e-13, atol=1e-18)


if __name__ == '__main__':
    unittest.main()
