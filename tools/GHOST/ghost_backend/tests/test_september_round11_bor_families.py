"""Round 11: the BoR previews assumed two near-operator families per surface (Codex's review of round 10, finding 1)."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor, dispatch
from ghost_backend.twod import solver as td

RADIUS = .1


def _chain(name, kind, points, ibc=0):
    return dict(name=name, seg_type=kind, properties=[str(kind), '1', str(ibc), '0', '0'],
                point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                             for a, b in zip(points[:-1], points[1:])])


def _corrugated(folds):
    """Codex's closed axis-to-axis corrugated conductor: many geometrically close panels."""
    points = [[0., .1], [.1, .1]]
    for index in range(2*folds):
        level = .1-.2*(index+1)/(2*folds)
        points.append([points[-1][0], level])
        points.append([.03 if points[-1][0] > .05 else .1, level])
    points[-1] = [0., -.1]
    return np.array(points)


class Admitted(Exception):
    pass


class NearOperatorFamilyTests(unittest.TestCase):
    """The previews assumed two near-operator families per surface; an impedance CFIE retains three."""

    IMPEDANCE = [['1', 'constant', '100', '50', '0', '0']]

    @staticmethod
    def _gate(call):
        guard, seen = bor._guard_bor_dense_memory, []
        def gate(*args, **kwargs):
            seen.append((bool(kwargs.get('streaming')), guard(*args, **kwargs)))
            raise Admitted()
        with patch.object(bor, '_guard_bor_dense_memory', gate):
            try:
                call()
            except Admitted:
                pass
        return seen

    def test_the_solve_states_its_operator_families_once(self):
        self.assertEqual(bor.conductor_operator_kinds('cfie', True), (True, True, True))
        self.assertEqual(bor.conductor_operator_kinds('cfie', False), (True, True, False))
        self.assertEqual(bor.conductor_operator_kinds('efie', True), (True, False, True))
        self.assertEqual(bor.conductor_operator_kinds('efie', False), (True, False, False))
        self.assertEqual(bor.conductor_operator_kinds('mfie', False), (False, True, False))
        # solve_bor prepares exactly the families it prices
        prepared = []
        def stop(solver, modes, **kwargs):
            prepared.append((kwargs['efie'], kwargs['mfie'], kwargs['ibc']))
            raise Admitted()
        for formulation, zs in (('cfie', 100+50j), ('cfie', None), ('efie', 100+50j)):
            with patch.object(bor.BorPecSolver, 'prepare_operators', stop), self.assertRaises(Admitted):
                bor.solve_bor(bor.sphere_generatrix(RADIUS, 12), 1e9, [0.], formulation=formulation, zs=zs,
                              workers=1, bor_options=dict(factorization='dense'))
            self.assertEqual(prepared[-1], bor.conductor_operator_kinds(formulation, zs is not None))

    def test_preview_and_gate_price_the_same_near_cache(self):
        points = _corrugated(20)
        solver = bor.BorPecSolver(points, .05e9)
        elements, pairs = len(points)-1, int(solver._near_pair_count)
        self.assertGreater(pairs, 8*elements)                    # near dominated: above the preview's floor
        for formulation, impedance in (('cfie', True), ('cfie', False), ('efie', True), ('efie', False)):
            with self.subTest(formulation=formulation, impedance=impedance):
                kinds = bor.conductor_operator_kinds(formulation, impedance)
                gate = bor.estimate_bor_operator_storage_gb(20, ((solver, *kinds),), streaming=True)
                preview = dispatch._estimate_junction_auxiliary_gb([(elements, True)], 20, {(0, 0): pairs}, [sum(kinds)])
                # the gate: 1.10 x near cache + 128 MB of near workspace
                self.assertAlmostEqual(preview['near_gb']+.128, gate, places=9)
        three = dispatch._estimate_junction_auxiliary_gb([(elements, True)], 20, {(0, 0): pairs}, [3])['near_gb']
        default = dispatch._estimate_junction_auxiliary_gb([(elements, True)], 20, {(0, 0): pairs})['near_gb']
        self.assertAlmostEqual(three, 1.5*default, places=9)     # material sides keep their two families

    def test_preview_covers_the_gate_on_near_dominated_conductors(self):
        points = _corrugated(30)
        bodies = dict(impedance=dict(segments=[_chain('ibc', 2, points, ibc=1)], ibcs=self.IMPEDANCE, dielectrics=[]),
                      pec=dict(segments=[_chain('pec', 2, points)], ibcs=[], dielectrics=[]))
        for label, snapshot in bodies.items():
            for assembly in ('tables', 'streaming'):
                for backend in ('dense', 'compressed'):
                    if backend == 'compressed' and assembly == 'streaming':
                        continue
                    with self.subTest(body=label, assembly=assembly, backend=backend), \
                            patch.object(td, '_solve_memory_limit_gb', return_value=1e4), \
                            patch.object(bor, '_solve_memory_limit_gb', return_value=1e4):
                        common = dict(geometry_units='meters', n_modes=40, workers=1, assembly=assembly,
                                      bor_options=dict(factorization=backend))
                        preview = dispatch.estimate_bor_resources(snapshot, .05, [0., 60.], mesh_certification=False, **common)
                        gates = self._gate(lambda: dispatch.solve_monostatic_rcs_bor(snapshot, [.05], [0., 60.], **common))
                        # Before, on the impedance body: streamed 0.141 GB and compressed 0.146 GB
                        # BELOW their gates (tables 0.013 GB above; 0.038 GB below at 201 panels).
                        self.assertGreaterEqual(preview['estimated_peak_gb'], gates[0][1])
                        self.assertLess(preview['estimated_peak_gb'], gates[0][1]+.25)

    def test_automatic_snapshot_call_is_admitted_where_the_old_preview_chose_tables(self):
        # Codex's reproduction: 201 panels, cap 20, tables just under the solvers' 2 GB
        # rule. The preview priced the table plan at 4.239 GB, its gate asks 4.277 GB;
        # at any limit in between the chooser imposed tables and the solve was
        # rejected, although the streamed plan needs 2.92 GB.
        snapshot = dict(segments=[_chain('corrugated IBC', 2, _corrugated(50), ibc=1)], ibcs=self.IMPEDANCE, dielectrics=[])
        arguments = dict(geometry_snapshot=snapshot, frequencies_ghz=[.05], elevations_deg=[0., 60.],
                         geometry_units='meters', n_modes=20, workers=1)
        with patch.object(td, '_solve_memory_limit_gb', return_value=1e4), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=1e4):
            table_gate = self._gate(lambda: dispatch.solve_monostatic_rcs_bor(
                **arguments, assembly='tables', bor_options=dict(factorization='dense')))[0][1]
        limit = table_gate-.01
        with patch.object(td, '_solve_memory_limit_gb', return_value=limit), \
                patch.object(bor, '_solve_memory_limit_gb', return_value=limit):
            self.assertEqual(dispatch.resolve_automatic_plan(dict(arguments)), ('dense', 'streaming'))
            gates = self._gate(lambda: dispatch.solve_monostatic_rcs_bor(**arguments))   # raised BorAdmissionError
        self.assertEqual([streaming for streaming, _ in gates], [True])
        self.assertLess(gates[0][1], limit)

if __name__ == '__main__':
    unittest.main()
