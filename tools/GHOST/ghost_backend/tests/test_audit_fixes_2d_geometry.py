"""September audit: 2D geometry validation, orientation checks and panel/node construction.

Each test reproduces a verified failure of the previous code and asserts the fix
(F1-F13 and the S1-S3 follow-ups of the audit report).
"""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import geometry as g
from ghost_backend.twod import solver as td
from ghost_backend.twod import adaptive_geometry as ag
from ghost_backend.geometry import io as gio

DENSE_2D = dict(factorization='dense', mesh_strategy='global')


def _pairs(points):
    return [dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
            for a, b in zip(points[:-1], points[1:])]


def _segment(name, points, seg_type=2, n=0, ibc=0, pos=0, neg=0):
    return dict(name=name, seg_type=seg_type, properties=[str(seg_type), str(n), str(ibc), str(pos), str(neg)],
                point_pairs=_pairs(points))


def _snapshot(segments, ibcs=(), dielectrics=()):
    return dict(segments=list(segments), ibcs=[list(row) for row in ibcs], dielectrics=[list(row) for row in dielectrics])


def _circle(radius, count, cw=True, center=(0.0, 0.0), start=0.0):
    """Vertices as drawn with cos/sin: the last one closes onto the first only to ~1e-17 m."""
    theta = np.linspace(start, start + (-2.0 if cw else 2.0) * np.pi, count + 1)
    return list(zip(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))


def _square(half, cw=True, center=(0.0, 0.0)):
    cx, cy = center
    points = [(cx - half, cy - half), (cx - half, cy + half), (cx + half, cy + half), (cx + half, cy - half),
              (cx - half, cy - half)]
    return points if cw else points[::-1]


def _mesh(snapshot, frequency, pol, scale=1.0):
    materials = g.MaterialLibrary.from_entries(snapshot['ibcs'], snapshot['dielectrics'], '.')
    wavelength = g._mesh_wavelength_for_snapshot(snapshot, materials, frequency)[0]
    panels = g._build_panels(snapshot, scale, wavelength, materials=materials, frequencies_ghz=[frequency])
    infos = g._build_coupled_panel_info(panels, materials, frequency, pol, 2 * np.pi * frequency * 1e9 / g.C0)
    mesh, _ = g._build_linear_mesh_interface_aware(panels, infos, polarization=pol)
    return panels, infos, mesh


def _amplitudes(snapshot, frequency, pol, angles=(0., 60., 90., 180.), units='meters'):
    result = td.solve_monostatic_rcs_2d_single_polarization(snapshot, [frequency], list(angles), pol,
        geometry_units=units, strict_quality_gate=False, execution_options=dict(DENSE_2D))
    amplitudes = np.array([complex(s['rcs_amp_real'], s['rcs_amp_imag']) for s in result['samples']])
    return amplitudes, result['metadata']


def _orientation_errors(segments):
    try:
        g._check_segment_orientation_or_raise(segments, 1.0)
    except ValueError as exc:
        return str(exc)
    return ''


class ExactPrimitiveEndsTests(unittest.TestCase):
    """F1: the last discretized point was p0 + (p1 - p0)*1.0, one ulp off the shared vertex."""

    def test_discretized_and_hp_ends_are_the_drawn_vertices(self):
        rng = np.random.default_rng(3)
        for _ in range(2000):
            a, b = rng.uniform(-3, 3, 2) * 0.0254, rng.uniform(-3, 3, 2) * 0.0254
            count = int(rng.integers(1, 40))
            points = g._discretize_primitive(a, b, count)
            self.assertTrue(np.array_equal(points[0], a) and np.array_equal(points[-1], b))
            _, hp_points = ag.panel_parameters({'_2d_hp_coarsening': 4.}, 0, a, b, count, False, 1.0, set())
            self.assertTrue(np.array_equal(hp_points[0], a) and np.array_equal(hp_points[-1], b))

    def test_inch_quadrilateral_meshes_one_node_per_panel(self):
        # vertex (1.0000625, 0.5) in: previously 13 nodes for 12 panels, no closed loop,
        # so TM silently used the SLP Robin-BIE and TE raised an open-contour error
        points = [(-1.3, -0.9), (-1.3, 0.5), (1.0000625, 0.5), (5.3, -0.9), (-1.3, -0.9)]
        snapshot = _snapshot([_segment('q', points, n=3)])
        panels = g._build_panels(snapshot, 0.0254, 1.0)
        materials = g.MaterialLibrary.from_entries([], [], '.')
        for pol in ('TM', 'TE'):
            infos = g._build_coupled_panel_info(panels, materials, 1.0, pol, 2 * np.pi * 1e9 / g.C0)
            mesh, _ = g._build_linear_mesh_interface_aware(panels, infos, polarization=pol)
            self.assertEqual((len(panels), len(mesh.nodes)), (12, 12))
            self.assertEqual(len(g._closed_conductor_loop_panels(panels, infos, 1e-9)), 12)
        self.assertEqual(len(g._build_linear_mesh(panels).nodes), 12)


class ClosingEdgeTests(unittest.TestCase):
    """F2: point-in-polygon skipped the closing edge poly[-1] -> poly[0] (a ~5e-18 m seam)."""

    def test_point_in_polygon_counts_the_closing_edge(self):
        disk = _circle(0.1, 96)
        self.assertNotEqual(disk[0], disk[-1])            # the seam of a cos/sin circle
        self.assertTrue(gio._point_in_polygon(0.0, 0.0, disk))
        self.assertTrue(gio._point_in_polygon(-0.05, 0.0, disk))
        self.assertFalse(gio._point_in_polygon(0.2, 0.0, disk))
        # an exactly closed polygon is unchanged
        closed = disk[:-1] + [disk[0]]
        self.assertTrue(gio._point_in_polygon(0.0, 0.0, closed))

    def test_void_on_the_seam_ray_of_a_disk_is_winding_checked(self):
        disk = _segment('disk', _circle(0.1, 96), 3, pos=1)
        cw_void = [(-0.03, -0.02), (-0.03, 0.02), (0.03, 0.02), (0.03, -0.02), (-0.03, -0.02)]
        wrong = _segment('void', cw_void, 3, pos=1)                 # previously accepted: 2.7-6.3 dB wrong
        right = _segment('void', cw_void[::-1], 3, pos=1)
        self.assertIn("'void'", _orientation_errors([disk, wrong]))
        self.assertEqual(_orientation_errors([disk, right]), '')

    def test_disjoint_square_and_circle_pass(self):
        # the circle's seam ray passed through the square: a valid layout was refused
        square = _segment('square', _square(0.05))
        circle = _segment('circle', _circle(0.02, 64, center=(0.3, 0.0)))
        self.assertEqual(_orientation_errors([square, circle]), '')


class AirParityTests(unittest.TestCase):
    """F3: the nesting depth counted TYPE 1 sheet loops and TYPE 4/5 contours."""

    def test_pec_inside_a_closed_sheet_shell(self):
        shell = _segment('shell', _circle(0.08, 96, start=0.3), 1, ibc=1)
        pec_cw = _segment('pec', _square(0.03))
        pec_ccw = _segment('pec', _square(0.03, cw=False))
        self.assertEqual(_orientation_errors([shell, pec_cw]), '')          # previously refused
        self.assertIn("'pec'", _orientation_errors([shell, pec_ccw]))      # previously accepted

    def test_air_void_in_a_layered_body(self):
        outer = _segment('outer', _circle(0.1, 96), 3, pos=1)
        inner = _segment('inner', _circle(0.06, 64), 5, pos=1, neg=2)       # CW: eps 1 outside, eps 2 inside
        void = _square(0.02, cw=False)
        self.assertEqual(_orientation_errors([outer, inner, _segment('void', void, 3, pos=2)]), '')
        self.assertIn("'void'", _orientation_errors([outer, inner, _segment('void', void[::-1], 3, pos=2)]))


class NodeWeldingTests(unittest.TestCase):
    """F4: validation calls endpoints <= 1e-9 m apart connected; the mesher keyed nodes on a fixed grid."""

    def test_welder_joins_points_across_a_grid_line(self):
        welder = g._NodeWelder()
        first = welder.key(np.array([0.5e-9 - 1.5e-10, 0.2]))
        self.assertEqual(welder.key(np.array([0.5e-9 + 1.5e-10, 0.2])), first)     # 3e-10 m, other cell
        self.assertNotEqual(welder.key(np.array([2.6e-9, 0.2])), first)            # 2.3e-9 m: distinct
        self.assertEqual(welder.lookup(np.array([0.5e-9, 0.2 + 9e-10])), first)
        self.assertIsNone(welder.lookup(np.array([0.0, 0.3])))

    def test_near_coincident_vertex_copies_mesh_as_one_node(self):
        pairs = [dict(x1=-1.0, y1=-1.0, x2=-1.0, y2=1.0), dict(x1=-1.0, y1=1.0, x2=1.00000001, y2=1.0),
                 dict(x1=1.00000002, y1=1.0, x2=1.0, y2=-1.0), dict(x1=1.0, y1=-1.0, x2=-1.0, y2=-1.0)]
        snapshot = dict(segments=[dict(name='sq', seg_type=2, properties=['2', '0', '0', '0', '0'], point_pairs=pairs)],
                        ibcs=[], dielectrics=[])
        g.validate_geometry_snapshot_for_solver(snapshot, '.', 0.0254)     # 2.54e-10 m: connected
        materials = g.MaterialLibrary.from_entries([], [], '.')
        panels = g._build_panels(snapshot, 0.0254, g.C0 / 1e9, materials=materials, frequencies_ghz=[1.0])
        for pol in ('TE', 'TM'):
            infos = g._build_coupled_panel_info(panels, materials, 1.0, pol, 2 * np.pi * 1e9 / g.C0)
            mesh, _ = g._build_linear_mesh_interface_aware(panels, infos, polarization=pol)
            self.assertEqual(len(mesh.nodes), len(panels))
            self.assertEqual(len(g._closed_conductor_loop_panels(panels, infos, 1e-9)), len(panels))
        self.assertEqual(len(g._build_linear_mesh(panels).nodes), len(panels))


class ShortPrimitiveTests(unittest.TestCase):
    """F5: sub-snap primitives meshed as degenerate elements. F6: 2e-9..1e-6 m primitives were 'cracks'."""

    @staticmethod
    def _corner_stub(stub):
        c = 0.05
        return _snapshot([_segment('sq', [(-c, -c), (-c, c), (c - stub, c), (c, c), (c, -c), (-c, -c)], ibc=1)],
                         ibcs=[['1', 'constant', '75', '-20', '0', '0']])

    @staticmethod
    def _notch(width):
        c = 0.05
        return _snapshot([_segment('notch', [(-c, -c), (-c, c), (0.0, c), (0.0, c - width), (width, c - width),
                                             (width, c), (c, c), (c, -c), (-c, -c)])])

    def test_primitives_within_the_node_tolerance_are_rejected(self):
        for stub in (5e-10, 9e-10):
            snapshot = self._corner_stub(stub)
            with self.assertRaisesRegex(ValueError, 'mesh node tolerance'):
                g.validate_geometry_snapshot_for_solver(snapshot, '.', 1.0)
            with self.assertRaisesRegex(ValueError, 'mesh node tolerance'):
                g._build_panels(snapshot, 1.0, 0.1)

    def test_short_legitimate_primitives_validate_and_mesh(self):
        for make in (self._corner_stub, self._notch):
            for length in (2e-9, 1e-8, 1e-7, 1e-6):
                with self.subTest(shape=make.__name__, length=length):
                    report = g.validate_geometry_snapshot_for_solver(make(length), '.', 1.0)
                    self.assertEqual(report['warnings'], [])
                    panels, _, mesh = _mesh(make(length), 3.0, 'TM')
                    self.assertEqual(len(mesh.nodes), len(panels))
                    self.assertFalse(any(e.node_ids[0] == e.node_ids[1] for e in mesh.elements))

    def test_real_cracks_and_overlaps_are_still_refused(self):
        crack = _snapshot([_segment('a', [(0.0, 0.0), (0.01, 0.0)], 1, ibc=1),
                           _segment('b', [(0.0100001, 0.0), (0.02, 0.0)], 1, ibc=1)],
                          ibcs=[['1', 'constant', '100', '0', '0', '0']])
        with self.assertRaisesRegex(ValueError, 'Geometry crack'):
            g.validate_geometry_snapshot_for_solver(crack, '.', 1.0)
        # a short primitive doubling back over its neighbour overlaps it
        hairpin = _snapshot([_segment('h', [(0.0, 0.0), (0.01, 0.0), (0.01 - 5e-8, 0.0)], 1, ibc=1)],
                            ibcs=[['1', 'constant', '100', '0', '0', '0']])
        with self.assertRaisesRegex(ValueError, 'Collinear overlapping'):
            g.validate_geometry_snapshot_for_solver(hairpin, '.', 1.0)


class SurfaceWaveDensityTests(unittest.TestCase):
    """F7: the mesh ignored the slow bound surface wave of strongly reactive impedances."""

    FREQUENCY = 3.0

    def test_bound_wave_index(self):
        eta = g.ETA0
        index = g._bound_surface_wave_index
        self.assertAlmostEqual(index(-60j), math.sqrt(1 + (eta / 60) ** 2), places=12)      # TM, X < 0
        self.assertAlmostEqual(index(600j), math.sqrt(1 + (600 / eta) ** 2), places=12)     # TE, X > 0
        self.assertAlmostEqual(index(-2000j), math.sqrt(1 + (eta / 2000) ** 2), places=12)  # weak TM wave
        self.assertEqual(index(0j), 0.0)
        self.assertEqual(index(75 - 20j), 0.0)          # bound but damped within a fraction of a wavelength
        self.assertEqual(index(eta), 0.0)               # matched absorber guides nothing
        # a TYPE 4 law sees the coating medium
        self.assertAlmostEqual(index(600j, 4.0 + 0j, 1.0 + 0j),
                               2.0 * math.sqrt(1 + (600 / (eta / 2.0)) ** 2), places=10)

    def _square(self, law, n=0):
        return _snapshot([_segment('sq', _square(0.05), n=n, ibc=1 if law else 0)],
                         ibcs=[['1', 'constant', repr(law.real), repr(law.imag), '0', '0']] if law else [])

    def _count(self, snapshot, frequencies=(3.0,)):
        materials = g.MaterialLibrary.from_entries(snapshot['ibcs'], [], '.')
        wavelength = g._conservative_mesh_wavelength_for_frequencies(snapshot, materials, frequencies)[0]
        return len(g._build_panels(snapshot, 1.0, wavelength, materials=materials, frequencies_ghz=list(frequencies)))

    def test_reactive_segments_are_sized_for_their_bound_wave(self):
        pec = self._count(self._square(0j))
        self.assertEqual(pec, 80)
        self.assertEqual(self._count(self._square(75 - 20j)), pec)
        self.assertEqual(self._count(self._square(-2000j)), 4 * 21)                    # weak TM wave, index 1.018
        self.assertEqual(self._count(self._square(-60j)), 4 * 81)                      # index 6.4, capped at 4x
        self.assertEqual(self._count(self._square(-20j)), pec)                         # index 18.9: unresolvable
        self.assertEqual(self._count(self._square(600j)), 4 * 38)                      # index 1.88
        self.assertEqual(self._count(self._square(600j, n=10)), 40)                    # explicit N is kept
        # a fixed mesh serving several frequencies is sized at each of them
        self.assertEqual(self._count(self._square(600j), (1.0, 3.0)), 4 * 38)
        # only the reactive segment is refined
        mixed = _snapshot([_segment('pec', _square(0.05)), _segment('ibc', _square(0.05, center=(0.5, 0.0)), ibc=1)],
                          ibcs=[['1', 'constant', '0', '600', '0', '0']])
        materials = g.MaterialLibrary.from_entries(mixed['ibcs'], [], '.')
        panels = g._build_panels(mixed, 1.0, g.C0 / 3e9, materials=materials, frequencies_ghz=[3.0])
        self.assertEqual(sum(p.name == 'pec' for p in panels), 80)
        self.assertEqual(sum(p.name == 'ibc' for p in panels), 4 * 38)

    def test_resource_forecast_uses_the_solved_mesh(self):
        from ghost_backend.execution.options import validate_options
        from ghost_backend.execution.selection import select_backend
        from ghost_backend.hpc import scheduler
        snapshot = self._square(-60j)
        arguments = dict(geometry_snapshot=snapshot, frequencies_ghz=[self.FREQUENCY], elevations_deg=[0.],
                         geometry_units='meters')
        _, metadata = _amplitudes(snapshot, self.FREQUENCY, 'TM', angles=(0.,))
        self.assertEqual(metadata['panel_count'], 4 * 81)
        planned = select_backend(dict(arguments, polarization='TM'), validate_options(dict(DENSE_2D)))
        self.assertEqual({record['panels'] for record in planned['meshes']}, {4 * 81})
        materials = g.MaterialLibrary.from_entries(snapshot['ibcs'], [], '.')
        records = scheduler._resource_records_for_frequency(td, snapshot, materials, self.FREQUENCY,
                                                            [('TM', 'TM')], 1., 20000)
        self.assertEqual(records['TM']['panels'], 4 * 81)

    def test_reactive_cylinder_matches_the_series_at_default_density(self):
        # 128-gon, ka = 6.3, TE +600j: 0.44 dB at 20 panels per free-space wavelength
        from ghost_backend.validation.cylinder import sigma_impedance_cylinder
        snapshot = _snapshot([_segment('c', _circle(0.1, 128), ibc=1)], ibcs=[['1', 'constant', '0', '600', '0', '0']])
        result = td.solve_monostatic_rcs_2d_single_polarization(snapshot, [3.0], [0.], 'TE', geometry_units='meters',
            strict_quality_gate=False, execution_options=dict(DENSE_2D))
        exact = sigma_impedance_cylinder(0.1, 600j, 3e9, 'TE')
        error_db = abs(10 * np.log10(result['samples'][0]['rcs_linear'] / exact))
        self.assertEqual(result['metadata']['panel_count'], 256)
        self.assertLess(error_db, 0.05)


class PanelCountToleranceTests(unittest.TestCase):
    """F8: a plain ceil turned round-off and 0.2 % length excess into a doubled panel count."""

    def test_exact_multiples_and_explicit_counts(self):
        for length, wavelength in ((0.3, 0.3), (0.1, 0.1), (0.015, 0.3), (0.7, 0.7), (0.9, 0.3)):
            self.assertEqual(g._panel_count_from_n(0, length, wavelength), round(20 * length / wavelength))
        self.assertEqual(g._panel_count_from_n(-40, 0.7, 0.7), 40)
        self.assertEqual(g._panel_count_from_n(7, 1.0, 0.1), 7)
        self.assertEqual(g._panel_count_from_n(0, 1e-6, 1.0), 1)
        self.assertEqual(g._panel_count_from_n(0, 1.002 * 0.05, 1.0), 1)      # was 2: a doubled primitive
        self.assertEqual(g._panel_count_from_n(0, 1.06 * 0.05, 1.0), 2)       # beyond the 5 % tolerance

    def test_regular_polygon_has_no_cost_cliff(self):
        snapshot = _snapshot([_segment('c', _circle(0.1, 128))])
        counts = [len(g._build_panels(snapshot, 1.0, g.C0 / (f * 1e9))) for f in (3.05, 3.06, 3.10, 3.25)]
        self.assertEqual(counts, [128, 128, 128, 256])


class CertificationBudgetTests(unittest.TestCase):
    """Fine certification mesh: max(n + 1, ceil(1.5 n)) per primitive doubled dense polylines."""

    def test_symmetric_chains_keep_symmetric_fine_meshes(self):
        # Mirror-image primitives (equal lengths and base counts) are refined
        # together: an odd leftover no longer refines one side only, which
        # made the certified fields of symmetric coupons asymmetric (1e-4).
        rng = np.random.default_rng(3)
        for _ in range(200):
            half = int(rng.integers(1, 6))
            lengths = list(rng.uniform(0.01, 0.3, half))
            base = [int(n) for n in rng.integers(1, 9, half)]
            middle_length = [float(rng.uniform(0.01, 0.3))] if rng.random() < 0.5 else []
            middle_base = [int(rng.integers(1, 9))] if middle_length else []
            chain_lengths = lengths + middle_length + lengths[::-1]
            chain_base = base + middle_base + base[::-1]
            fine = g._certification_fine_counts(chain_base, chain_lengths, 1.5)
            self.assertEqual(fine, fine[::-1])
            total = sum(chain_base)
            target = max(total + 1, math.ceil(1.5 * total))
            self.assertGreaterEqual(sum(fine), target)
            self.assertLessEqual(sum(fine), target + max(2, math.ceil(0.01 * target)))
            self.assertTrue(all(f >= b for f, b in zip(fine, chain_base)))

    def test_fine_counts_follow_a_chain_budget(self):
        lengths = [1.0] * 798
        counts = g._certification_fine_counts([1] * 798, lengths, 1.5)
        self.assertEqual(sum(counts), 1197)
        self.assertTrue(all(c in (1, 2) for c in counts))
        refined = np.flatnonzero(np.array(counts) == 2)
        self.assertLess(np.max(np.diff(refined)), 4)                 # spread along the chain
        # the longest panels are refined first, never below the base count
        mixed = g._certification_fine_counts([20, 1, 1, 1, 1], [2.0, 0.05, 0.05, 0.05, 0.05], 1.5)
        self.assertEqual(mixed, [32, 1, 1, 1, 1])
        self.assertEqual(g._certification_fine_counts([20], [2.0], 1.5), [30])
        self.assertEqual(g._certification_fine_counts([1], [2.0], 1.5), [2])

    def test_fine_mesh_of_a_densely_drawn_polyline(self):
        from ghost_backend.runs.quality import scale_snapshot_panel_density
        snapshot = _snapshot([_segment('c', _circle(0.3, 798))])
        wavelength = g.C0 / 1e9
        base = g._build_panels(snapshot, 1.0, wavelength)
        fine = scale_snapshot_panel_density(snapshot, 1.5)
        fine['_2d_certification_refinement_factor'] = 1.5
        fine['_2d_certification_base_segment_n'] = ['0']
        self.assertEqual((len(base), len(g._build_panels(fine, 1.0, wavelength))), (798, 1197))


class HpEligibilityTests(unittest.TestCase):
    """F9: hp eligibility ignored that coarsening stops at one element per drawn primitive."""

    @staticmethod
    def _ngon(sides, radius):
        return _snapshot([_segment('c', _circle(radius, sides))])

    def test_predicted_size(self):
        self.assertEqual(ag.predicted_hp_size([(10, False), (1, False), (9, True)]), (20, 3 + 1 + 9))

    def test_faceted_input_that_cannot_coarsen_is_refused(self):
        # one element per drawn primitive: the certified hp pair took 11.3 s against 6.9 s for P1
        materials = g.MaterialLibrary.from_entries([], [], '.')
        eligible, reason = ag.eligible_snapshot(self._ngon(1024, 0.5), materials, [3.0], 1.0, None)
        self.assertFalse(eligible)
        self.assertIn('Drawn primitives limit coarsening', reason)

    def test_coarsenable_polygon_is_admitted(self):
        # 640 reference panels on 64 primitives (192 hp elements): previously refused as small,
        # although the certified hp pair took 1.1 s against 3.1 s for P1
        materials = g.MaterialLibrary.from_entries([], [], '.')
        counts = g._reference_panel_counts(self._ngon(64, 0.5), 1.0, g.C0 / 3e9, materials, [3.0])
        self.assertEqual(ag.predicted_hp_size(counts), (640, 192))
        self.assertEqual(ag.eligible_snapshot(self._ngon(64, 0.5), materials, [3.0], 1.0, None), (True, ''))
        self.assertTrue(ag.eligible_snapshot(self._ngon(128, 0.5), materials, [3.0], 1.0, None)[0])


class OrientationScalingTests(unittest.TestCase):
    """F10: every closed contour was point-in-polygon tested against every other one."""

    def test_disjoint_contours_skip_point_in_polygon(self):
        segments = [_segment(f'c{i}_{j}', _circle(0.015, 8, center=(0.05 * i, 0.05 * j)))
                    for i in range(30) for j in range(30)]
        calls = []
        real = gio._point_in_polygon

        def counting(*args):
            calls.append(1)
            return real(*args)
        with patch.object(gio, '_point_in_polygon', counting):
            findings = gio.check_orientation_consistency(gio.chains_from_snapshot_segments(segments))
        self.assertEqual(findings, [])
        self.assertLess(len(calls), 100)          # 810,000 before the bounding-box filter


class ExplicitFloorTests(unittest.TestCase):
    """F11: the explicit-N safety floor used the shortest wavelength of any material in the model."""

    def test_unrelated_high_index_rod_does_not_raise_an_air_segment_floor(self):
        pec = _segment('pec', _square(0.05), n=20)
        rod = _segment('rod', _circle(0.004, 64, center=(0.3, 0.001), start=0.2), 3, pos=1)
        snapshot = _snapshot([pec, rod], dielectrics=[['1', '100', '0', '1', '0']])
        materials = g.MaterialLibrary.from_entries([], snapshot['dielectrics'], '.')
        wavelength = g._mesh_wavelength_for_snapshot(snapshot, materials, 3.0)[0]
        panels = g._build_panels(snapshot, 1.0, wavelength, materials=materials, frequencies_ghz=[3.0])
        self.assertEqual(sum(p.name == 'pec' for p in panels), 80)
        # the floor still applies at the segment's own material wavelength
        thin = _snapshot([_segment('rod', _square(0.05), 3, n=2, pos=1)], dielectrics=[['1', '100', '0', '1', '0']])
        with self.assertRaisesRegex(ValueError, 'safety floor'):
            g._build_panels(thin, 1.0, wavelength, materials=materials, frequencies_ghz=[3.0])


class TableRangeTests(unittest.TestCase):
    """F12: a sweep frequency one ulp past a table end raised."""

    def test_ends_absorb_round_off(self):
        table = g.ComplexTable(freqs_ghz=np.array([0.8, 1.2]), values=np.array([12 - 4j, 17 - 1j]))
        self.assertEqual(table.sample(0.8 + 4 * 0.1), 17 - 1j)                  # 1.2000000000000002
        self.assertEqual(table.sample(np.nextafter(0.8, 0.0)), 12 - 4j)
        with self.assertRaisesRegex(ValueError, r'1\.2000000001\d* GHz is outside the characterized range'):
            table.sample(1.2000000001)
        medium = g.MediumTable(freqs_ghz=np.array([0.8, 1.2]), eps_values=np.array([2 - 0.1j, 3 - 0.1j]),
                               mu_values=np.array([1 + 0j, 1 + 0j]))
        self.assertEqual(medium.sample(1.2000000000000002), (3 - 0.1j, 1 + 0j))
        with self.assertRaisesRegex(ValueError, r'0\.7999\d* GHz is outside'):
            medium.sample(0.79999)


class FacetingAdvisoryTests(unittest.TestCase):
    """F13: faceting error is invisible to mesh certification; it is now reported as information."""

    def _information(self, snapshot, frequency):
        materials = g.MaterialLibrary.from_entries([], [], '.')
        g._build_panels(snapshot, 1.0, g.C0 / (frequency * 1e9), materials=materials, frequencies_ghz=[frequency])
        return materials.information, materials.warnings

    def test_coarse_polygon_approximating_a_curve_is_reported(self):
        ka30 = 30 * g.C0 / (2 * np.pi * 0.1) / 1e9
        information, warnings = self._information(_snapshot([_segment('c', _circle(0.1, 32))]), ka30)
        self.assertTrue(any('Faceting advisory' in item and "'c'" in item for item in information))
        self.assertEqual(warnings, [])
        self.assertEqual(self._information(_snapshot([_segment('c', _circle(0.1, 256))]), ka30)[0], [])
        self.assertEqual(self._information(_snapshot([_segment('sq', _square(0.1))]), ka30)[0], [])

    def test_a_per_solve_notice_sink_receives_the_advisory(self):
        class Sink:
            def __init__(self):
                self.information, self.warnings = [], []

            def inform_once(self, message):
                self.information.append(message)

            def warn_once(self, message):
                self.warnings.append(message)
        ka10 = 10 * g.C0 / (2 * np.pi * 0.1) / 1e9
        materials, sink = g.MaterialLibrary.from_entries([], [], '.'), Sink()
        g._build_panels(_snapshot([_segment('c', _circle(0.1, 32))]), 1.0, g.C0 / (ka10 * 1e9),
                        materials=materials, frequencies_ghz=[ka10], notices=sink)
        self.assertEqual(len(sink.information), 1)
        self.assertIn('Faceting advisory', sink.information[0])
        self.assertEqual((sink.warnings, materials.warnings), ([], []))


class InterfaceSignatureTests(unittest.TestCase):
    """S3: pos/neg flags a TYPE ignores entered the node signature and split closed contours."""

    def test_ignored_flags_do_not_split_nodes(self):
        c = 0.05
        half_a, half_b = [(-c, -c), (-c, c), (c, c)], [(c, c), (c, -c), (-c, -c)]
        dielectrics = [['1', '3', '0', '1', '0']]
        for seg_type, junk in ((2, dict(pos=1, neg=1)), (3, dict(pos=1, neg=1))):
            with self.subTest(seg_type=seg_type):
                base = dict(pos=1) if seg_type == 3 else {}
                clean = _snapshot([_segment('a', half_a, seg_type, **base), _segment('b', half_b, seg_type, **base)],
                                  dielectrics=dielectrics)
                stray = _snapshot([_segment('a', half_a, seg_type, **base), _segment('b', half_b, seg_type, **junk)],
                                  dielectrics=dielectrics)
                for pol in ('TM', 'TE'):
                    reference, meta_clean = _amplitudes(clean, 3.0, pol)
                    values, meta_stray = _amplitudes(stray, 3.0, pol)
                    self.assertEqual(meta_stray['linear_node_count'], meta_clean['linear_node_count'])
                    self.assertEqual(meta_stray['formulation'], meta_clean['formulation'])
                    np.testing.assert_allclose(values, reference, rtol=1e-12, atol=0)


class Type4WindingTests(unittest.TestCase):
    """S2: closed TYPE 4 chains without a matching closed TYPE 3/5 parent were never winding-checked."""

    def test_second_coating_layer_core(self):
        outer = _segment('outer', _circle(0.1, 96), 3, pos=1)
        middle = _segment('mid', _circle(0.08, 80), 5, pos=1, neg=2)        # CW: eps 1 outside, eps 2 inside
        core_cw = _segment('core', _circle(0.06, 64), 4, pos=2)
        core_ccw = _segment('core', _circle(0.06, 64, cw=False), 4, pos=2)
        self.assertEqual(_orientation_errors([outer, middle, core_cw]), '')
        # previously accepted, 17-31 dB wrong
        self.assertIn("Segment 'core' (TYPE 4)", _orientation_errors([outer, middle, core_ccw]))

    def test_dielectric_filled_cavity_in_a_pec_body(self):
        body = _segment('pec', _circle(0.1, 96))
        wall_ccw = _segment('wall', _circle(0.05, 64, cw=False), 4, pos=1)   # dielectric inside
        wall_cw = _segment('wall', _circle(0.05, 64), 4, pos=1)
        self.assertEqual(_orientation_errors([body, wall_ccw]), '')
        self.assertIn("Segment 'wall' (TYPE 4)", _orientation_errors([body, wall_cw]))

    def test_no_inference_across_a_stitched_cavity_wall(self):
        # the cavity wall is two open TYPE 4 segments: the medium around the coated island
        # is the cavity filling, not the PEC body, so the island is not judged against the body
        body = _segment('pec', _circle(0.1, 96))
        wall = _circle(0.06, 64, cw=False)
        halves = [_segment('wall_a', wall[:33], 4, pos=1), _segment('wall_b', wall[32:], 4, pos=1)]
        island = _segment('island', _circle(0.02, 32), 4, pos=1)                # PEC inside, coated: CW
        self.assertEqual(_orientation_errors([body] + halves + [island]), '')


if __name__ == '__main__':
    unittest.main()
