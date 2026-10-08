"""Sweep planning reuse, fixed-reference sizing and batch-invariant marking."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from ghost_backend.execution.options import validate_options
from ghost_backend.execution.selection import select_backend
from ghost_backend.twod.preparation import (
    ForecastCache, preparation_scope, forecast_cache, sweep_mesh_scope, mesh_frequencies,
)
from ghost_backend.twod.adaptivity import Indicators


def rectangle():
    corners = [(-.1, -.02), (-.1, .02), (.1, .02), (.1, -.02), (-.1, -.02)]
    return dict(segments=[dict(name='PEC', seg_type=2, properties=['2', '0', '0', '0', '0'],
        point_pairs=[dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1])
                     for a, b in zip(corners, corners[1:])])], ibcs=[], dielectrics=[])


class SweepPreparationTests(unittest.TestCase):
    def test_large_sweep_reuses_all_forecasts_and_uses_digest_keys(self):
        args = dict(geometry_snapshot=rectangle(), frequencies_ghz=[1.], elevations_deg=[0.])
        options = validate_options({})
        forecast = dict(selected='dense', meshes=[], candidates=dict(dense=dict(cost=1., peak_gb=1.)))
        with preparation_scope(), patch('ghost_backend.execution.selection._forecast_backend', return_value=forecast) as build, \
                patch('ghost_backend.execution.selection._refresh_memory_forecast'):
            for _ in range(2):
                for i in range(256):
                    select_backend(dict(args, frequencies_ghz=[1. + i*.01]), options)
            self.assertEqual(build.call_count, 256)
            self.assertTrue(all(isinstance(key, bytes) and len(key) == 32 for key in forecast_cache()))
            self.assertLessEqual(forecast_cache().retained_bytes, forecast_cache().max_bytes)

    def test_forecast_byte_budget_evicts_and_rejects_oversize_record(self):
        cache = ForecastCache(max_bytes=1600)
        for index in range(20):
            cache[str(index)] = {'text': str(index)*100}
            self.assertLessEqual(cache.retained_bytes, cache.max_bytes)
        self.assertLess(len(cache), 20)
        cache['oversize'] = 'x'*2000
        self.assertNotIn('oversize', cache)

    def test_nested_sweep_scope_preserves_original_request(self):
        with sweep_mesh_scope([1., 2.]):
            with sweep_mesh_scope([1.]):
                self.assertEqual(mesh_frequencies([1.]), (1., 2.))
        self.assertEqual(mesh_frequencies([3.]), (3.,))

    def test_sizing_frequencies_are_part_of_forecast_identity(self):
        args = dict(geometry_snapshot=rectangle(), frequencies_ghz=[1.], elevations_deg=[0.], mesh_reference_ghz=1.)
        options = validate_options({})
        with preparation_scope(), patch('ghost_backend.execution.selection._forecast_backend', return_value={}) as build:
            for served in ([1., 2.], [1., 3.]):
                with sweep_mesh_scope(served):
                    select_backend(args, options)
            self.assertEqual(build.call_count, 2)

    def test_fixed_reference_sweep_and_diagnostic_meshes_agree(self):
        from ghost_backend.twod import solver
        args = dict(geometry_snapshot=rectangle(), frequencies_ghz=[1., 2.], elevations_deg=[0., 90.],
            geometry_units='meters', mesh_reference_ghz=1., solver_method='direct',
            execution_options=dict(mesh_strategy='global', basis_order=1, factorization='dense',
                                   blas_threads=1, assembly_threads=1))
        combined = solver.solve_monostatic_rcs_2d(**args)
        diagnostic = solver.solve_monostatic_rcs_2d_single_polarization(polarization='TE', **args)
        count = diagnostic['metadata']['panel_count']
        self.assertEqual([r['metadata']['panel_count'] for r in combined['metadata']['frequency_metadata']], [count, count])
        def amplitudes(rows):
            return np.array([complex(row['rcs_amp_real'], row['rcs_amp_imag']) for row in rows])
        np.testing.assert_allclose(amplitudes(combined['co_solved_samples']['VV']),
                                   amplitudes(diagnostic['samples']), rtol=2e-12, atol=1e-14)
        self.assertEqual(combined['metadata']['operator_cache_scope'], 'one_frequency')
        certified = solver.solve_monostatic_rcs_2d_certified(**args)
        gates = [r['metadata']['mesh_convergence'] for r in certified['metadata']['frequency_metadata']]
        self.assertEqual([gate['base_panel_count'] for gate in gates], [count, count])
        self.assertTrue(all(gate['fine_panel_count'] > count for gate in gates))

    def test_fixed_reference_forecast_scope_matches_execution(self):
        args = dict(geometry_snapshot=rectangle(), frequencies_ghz=[1., 2.], elevations_deg=[0., 90.],
                    geometry_units='meters', mesh_reference_ghz=1.)
        options = validate_options(dict(mesh_strategy='global', factorization='adaptive'))
        with preparation_scope():
            plan = select_backend(args, options)
        counts = {r['panels'] for r in plan['meshes']}
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts, {66})

    def test_marking_is_independent_of_angle_batch_partition(self):
        degree = 3
        elements = [SimpleNamespace(node_ids=tuple(range(4*i, 4*i+4)),
                    primitive_key=str(i//2), length=.1*(i+1)) for i in range(8)]
        mesh = SimpleNamespace(elements=elements)
        rng = np.random.default_rng(182)
        density = rng.normal(size=(32, 39)) + 1j*rng.normal(size=(32, 39))
        full, partitioned = Indicators(), Indicators()
        full.observe(mesh, density)
        for first, last in ((0, 1), (1, 8), (8, 25), (25, 39)):
            partitioned.observe(mesh, density[:, first:last])
        np.testing.assert_allclose(list(full.scores.values()), list(partitioned.scores.values()), rtol=5e-15)
        self.assertEqual(full.marked(), partitioned.marked())

    def test_near_pair_count_matches_all_pair_definition(self):
        from ghost_backend.twod.formulations.regions import geometric_near_pair_count
        rng = np.random.default_rng(240)
        centers = rng.uniform(-1., 1., (150, 2))
        centers[:4] = [[0., 0.], [.3, 0.], [.6, 0.], [.9, 0.]]
        for lengths in (np.full(150, .1), np.exp(rng.uniform(-9., 0., 150))):
            expected = np.count_nonzero(np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
                                        <= 3*np.maximum(lengths[:, None], lengths[None, :]))
            self.assertEqual(geometric_near_pair_count(centers, lengths), expected)
        self.assertEqual(geometric_near_pair_count(np.empty((0, 2)), np.array([])), 0)


if __name__ == '__main__':
    unittest.main()
