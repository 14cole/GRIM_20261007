"""Round 12: one BoR operator-storage model for the run-time gate and the previews, on every geometry kind."""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch
from ghost_backend.twod import solver as td

MEDIA = [['1', '2.2', '-0.06', '1.0', '-0.01'], ['2', '3.0', '-0.10', '1.0', '-0.02'], ['3', '4.0', '-0.2', '1.0', '0.0']]


def _chain(name, kind, points, positive=0, negative=0, ibc=0):
    points = np.asarray(points, float)
    return dict(name=name, seg_type=kind, properties=[str(kind), '1', str(ibc), str(positive), str(negative)],
                point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                             for a, b in zip(points[:-1], points[1:])])


def _part(radius, count, start=0., stop=math.pi):
    theta = np.linspace(start, stop, count+1)
    points = np.column_stack([radius*np.sin(theta), radius*np.cos(theta)])
    points[np.abs(points[:, 0]) < 1e-15, 0] = 0.
    return points


def _corrugated(folds):
    points = [[0., .1], [.1, .1]]
    for index in range(2*folds):
        level = .1-.2*(index+1)/(2*folds)
        points.append([points[-1][0], level])
        points.append([.03 if points[-1][0] > .05 else .1, level])
    points[-1] = [0., -.1]
    return np.array(points)


class Admitted(Exception):
    pass


def _gate(call):
    guard, seen = bor._guard_bor_dense_memory, []
    def gate(*args, **kwargs):
        seen.append(guard(*args, **kwargs))
        raise Admitted()
    with patch.object(bor, '_guard_bor_dense_memory', gate):
        try:
            call()
        except Admitted:
            pass
    return seen[0]


class OneStorageModelTests(unittest.TestCase):
    def test_the_gate_is_the_model_on_the_solvers_own_records(self):
        outer, core = bor.BorPecSolver(bor.sphere_generatrix(.1, 30), 1e9), bor.BorPecSolver(bor.sphere_generatrix(.095, 28), 1e9)
        cross = bor.BorCrossOperators(outer, core)
        self.assertGreater(len(cross.near_pairs), 0)
        near = (bor.bor_near_cache_bytes(outer._near_pair_count, outer.Nn, 2, 12)+bor.bor_near_cache_bytes(core._near_pair_count, core.Nn, 2, 12)
                + len(cross.near_pairs)*2*4*13*4*16)
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                gate = bor.estimate_bor_operator_storage_gb(12, ((outer, True, False, True), (core, True, True, False)),
                                                            (cross,), constraint_dofs=40, streaming=streaming)
                surfaces = [bor.BorSurfaceStorage(outer.P, outer.Nn, 16, outer._near_pair_count, True, False, True, None),
                            bor.BorSurfaceStorage(core.P, core.Nn, 16, core._near_pair_count, True, True, False, None)]
                crosses = [bor.BorCrossStorage(outer.P, core.P, 16, len(cross.near_pairs), cross.near_max_order, None)]
                parts = bor.bor_operator_storage_bytes(12, surfaces, crosses, 40, streaming=streaming)
                self.assertEqual(parts['near'], near)                 # same- and cross-surface contractions
                self.assertEqual(parts['projection'], 3*40*6*16)          # sparse m = 0, 1, |m| >= 2 transforms
                retained = parts['tables']+parts['basis']+parts['near']+parts['projection']
                bound = (bor.BOR_RETAINED_STORAGE_FACTOR*retained+max(parts['fft_workspace'], parts['near_workspace']))/1e9
                # records without sample counts price the FFT workspace at its bound: never below the gate
                self.assertGreaterEqual(bound, gate)
                self.assertLess(bound-gate, .26 if not streaming else 1e-12)
                if streaming:
                    self.assertEqual((parts['tables'], parts['basis'], parts['fft_workspace']), (0., 0., 0.))

    def test_the_model_is_monotone(self):
        base = bor.BorSurfaceStorage(400, 101, 16, 900, True, True, False, None)
        reference = sum(bor.bor_operator_storage_bytes(20, [base], constraint_dofs=50).values())
        for change in (dict(points=440), dict(nodes=120), dict(near_pairs=2000), dict(ibc=True)):
            with self.subTest(change=change):
                self.assertGreater(sum(bor.bor_operator_storage_bytes(20, [base._replace(**change)], constraint_dofs=50).values()),
                                   reference)

    def test_preview_tables_are_the_solvers_table_formulas(self):
        for formulation, impedance in (('cfie', True), ('cfie', False), ('efie', True), ('efie', False)):
            with self.subTest(formulation=formulation, impedance=impedance):
                kinds = bor.conductor_operator_kinds(formulation, impedance)
                model = dispatch._layout_storage([(150, True)], 20, kinds=[kinds])['tables']/1e9
                self.assertAlmostEqual(model, bor.estimate_bor_table_gb(150, 20, formulation, impedance, bor.FAR_GAUSS_ORDER, False), places=12)
        layout = [(60, False), (40, True)]
        # One cross table per surface pair: dense solves derive the reverse blocks by reciprocity.
        order = bor.FAR_GAUSS_ORDER
        coated = (2*bor.estimate_bor_table_gb(60, 20, 'efie', True, order, False)+bor.estimate_bor_table_gb(40, 20, 'cfie', False, order, False)
                  + bor.estimate_bor_cross_table_gb(60, 40, 20))
        self.assertAlmostEqual(dispatch._estimate_multisurface_operator_gb(layout, 20), coated, places=12)
        self.assertAlmostEqual(dispatch._estimate_multisurface_operator_gb(layout, 20, single_tables=True), coated/2, places=12)

    def test_preview_covers_the_gate_for_layered_and_banded_bodies(self):
        # never compared before round 12; 1 to 2 mm layers make cross-surface near pairs dominate
        full = lambda radius, count: _part(radius, count)
        upper = lambda radius, count: _part(radius, count, 0., math.pi/2)
        lower = lambda radius, count: _part(radius, count, math.pi/2, math.pi)
        bodies = {
            'layered': dict(dielectrics=MEDIA[:2], ibcs=[], segments=[
                _chain('outer', 3, full(.100, 40), 1), _chain('mid', 5, full(.099, 40), 1, 2), _chain('core', 4, full(.098, 40), 2)]),
            'layered patch': dict(dielectrics=MEDIA[:2], ibcs=[], segments=[
                _chain('patch', 3, upper(.085, 16)*(1+.15*np.sin(2*np.linspace(0., math.pi/2, 17)))[:, None], 1),
                _chain('mid covered', 5, upper(.085, 16), 1, 2), _chain('mid bare', 3, lower(.085, 16), 2),
                _chain('core', 4, full(.06, 24), 2)]),
            'layered_n': dict(dielectrics=MEDIA, ibcs=[], segments=[
                _chain('outer', 3, full(.100, 32), 1), _chain('i12', 5, full(.099, 32), 1, 2),
                _chain('i23', 5, full(.098, 32), 2, 3), _chain('core', 4, full(.097, 32), 3)]),
            'banded': dict(dielectrics=MEDIA[:2], ibcs=[], segments=[
                _chain('outer upper', 3, upper(.100, 24), 1), _chain('outer lower', 3, lower(.100, 24), 2),
                _chain('wall', 5, [[.100, 0.], [.098, 0.]], 2, 1),
                _chain('core upper', 4, upper(.098, 24), 1), _chain('core lower', 4, lower(.098, 24), 2)])}
        kinds = set()
        for label, snapshot in bodies.items():
            for assembly in ('tables', 'streaming'):
                with self.subTest(body=label, assembly=assembly), \
                        patch.object(td, '_solve_memory_limit_gb', return_value=1e4), \
                        patch.object(bor, '_solve_memory_limit_gb', return_value=1e4):
                    common = dict(geometry_units='meters', n_modes=20, workers=1, assembly=assembly,
                                  bor_options=dict(factorization='dense'))
                    preview = dispatch.estimate_bor_resources(snapshot, 3., [0., 60.], mesh_certification=False, **common)
                    gate = _gate(lambda: dispatch.solve_monostatic_rcs_bor(snapshot, [3.], [0., 60.], **common))
                    kinds.add(preview['geometry_kind'])
                    self.assertGreaterEqual(preview['estimated_peak_gb'], gate)
                    self.assertLess(preview['estimated_peak_gb'], gate+1.5)
        self.assertEqual(kinds, {'layered', 'layered_n', 'banded'})

    def test_partial_coating_preview_prices_the_impedance_maps_of_bare_pieces(self):
        from test_bor_physics_regression import _partial_coating_snapshot
        snapshot = _partial_coating_snapshot(explicit_elements=12)
        bare = [segment for segment in snapshot['segments'] if int(segment['properties'][0]) == 2]
        self.assertTrue(bare)
        coated = dict(snapshot, ibcs=[['7', 'constant', '100', '50', '0', '0']],
                      segments=[dict(segment, properties=[segment['properties'][0], segment['properties'][1], '7']
                                     + list(segment['properties'][3:])) if segment in bare else segment
                                for segment in snapshot['segments']])
        common = dict(geometry_units='meters', n_modes=20, workers=1, assembly='streaming', mesh_certification=False,
                      bor_options=dict(factorization='dense'))
        with patch.object(td, '_solve_memory_limit_gb', return_value=1e4), patch.object(bor, '_solve_memory_limit_gb', return_value=1e4):
            plain = dispatch.estimate_bor_resources(snapshot, 2., [0., 60.], **common)
            mapped = dispatch.estimate_bor_resources(coated, 2., [0., 60.], **common)
            gate = _gate(lambda: dispatch.solve_monostatic_rcs_bor(coated, [2.], [0., 60.], geometry_units='meters', n_modes=20,
                                                                   workers=1, assembly='streaming', bor_options=dict(factorization='dense')))
        self.assertGreater(mapped['persistent_assembly_gb'], plain['persistent_assembly_gb'])
        self.assertGreaterEqual(mapped['estimated_peak_gb'], gate)

    def test_direct_chooser_counts_the_near_pairs_of_the_supplied_generatrix(self):
        points = _corrugated(30)
        solver = bor.BorPecSolver(points, .05e9)
        supplied = dict(points=points, freq_hz=.05e9, thetas_deg=[0., 60.], zs=100+50j, n_modes=40, workers=1, assembly='tables')
        self.assertEqual(dispatch._direct_near_pair_counts(supplied), {(0, 0): int(solver._near_pair_count)})
        self.assertEqual(dispatch._direct_near_pair_counts(dict(supplied, points=np.zeros((3, 2)))), {})  # the solve reports why
        layout, _ = dispatch._direct_surface_layout(supplied)
        with patch.object(td, '_solve_memory_limit_gb', return_value=1e4), patch.object(bor, '_solve_memory_limit_gb', return_value=1e4):
            planned = dispatch._direct_dense_plan(supplied, layout, 40, 4)['estimated_peak_gb']
            gate = _gate(lambda: bor.solve_bor(points, .05e9, [0., 60.], zs=100+50j, n_modes=40, workers=1, assembly='tables',
                                               bor_options=dict(factorization='dense')))
        self.assertGreaterEqual(planned, gate)      # the stencil floor alone priced this body 0.24 GB below its gate


if __name__ == '__main__':
    unittest.main()
