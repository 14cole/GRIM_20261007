"""Analytic checks for optional assembly response corrections and sidedness."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghost_backend.assembly.fields import (
    point_scatterer_amplitude, prepare_point_pattern, sum_features,
)
from ghost_backend.assembly.line_expansion import C0, SeamCoefficients, expand_perimeter
from ghost_backend.geometry.occlusion import Occluder
from test_point_scatter_physics import _pattern_dict, _local_frame, _jones_from_tensor


class LineResponseControlsTests(unittest.TestCase):
    def setUp(self):
        self.length = 0.2
        self.coefficients = SeamCoefficients(
            1.0, np.array([0., 90., 180.]), np.ones(3, complex), np.ones(3, complex)
        )
        self.placement = dict(
            delta=self.coefficients,
            perimeter=np.array([[[-0.1, 0., 0.], [0.1, 0., 0.]]]),
            normal=np.array([0., 0., 1.]),
        )

    def test_per_response_zero_corrections_and_distinct_tm_te_defaults(self):
        direction = np.array([[0., 0., 1.]])
        default = sum_features(None, [self.placement], direction, 1.)
        uncorrected = sum_features(None, [dict(self.placement, psi_tm_deg=0., psi_te_deg=0.)], direction, 1.)
        # At the pole the canonical V/H axes are x/y. The x-directed span is TM.
        amplitude = self.length / (4. * math.pi)
        np.testing.assert_allclose(uncorrected['amp_vv'], amplitude, atol=1e-14)
        np.testing.assert_allclose(uncorrected['amp_hh'], amplitude, atol=1e-14)
        np.testing.assert_allclose(default['amp_vv'], amplitude*np.exp(1j*np.deg2rad(166.9)), atol=1e-14)
        np.testing.assert_allclose(default['amp_hh'], amplitude*np.exp(1j*np.deg2rad(-9.2)), atol=1e-14)
        # Both placements remain independent when they share the coefficient table.
        combined = sum_features(None, [self.placement, dict(self.placement, psi_tm_deg=0., psi_te_deg=0.)], direction, 1.)
        for channel in ('amp_vv', 'amp_hh', 'amp_vh'):
            np.testing.assert_allclose(combined[channel], default[channel]+uncorrected[channel], atol=1e-14)

    def test_taper_off_restores_supplied_field_and_keeps_front_gate(self):
        angles = np.deg2rad([2., 5., 15., -5., 0.])
        directions = np.column_stack((np.zeros(len(angles)), np.cos(angles), np.sin(angles)))
        common = dict(
            segments=self.placement['perimeter'], coefficients=self.coefficients,
            normal_fn=lambda points: np.tile([0., 0., 1.], (len(points), 1)),
            directions=directions, frequency_ghz=1.,
        )
        uncorrected = self.length / (4. * math.pi)
        expected_off = uncorrected * np.array([1., 1., 1., 0., 0.])
        expected_on = uncorrected * np.array([0.5-0.5*np.cos(0.2*np.pi), 0.5, 1., 0., 0.])
        for batch in (1, 3):
            off = expand_perimeter(**common, grazing_taper_deg=0., _look_batch_size=batch)
            on = expand_perimeter(**common, _look_batch_size=batch)
            for channel in ('F_vv', 'F_hh'):
                np.testing.assert_allclose(off[channel], expected_off, atol=1e-14)
                np.testing.assert_allclose(on[channel], expected_on, atol=1e-14)
            np.testing.assert_allclose(off['F_vh'], 0., atol=1e-14)
        placement = dict(self.placement, psi_tm_deg=0., psi_te_deg=0., grazing_taper_deg=0.)
        actual = sum_features(None, [placement], directions, 1.)
        np.testing.assert_allclose(actual['amp_vv'], expected_off, atol=1e-14)


class PointResponseControlsTests(unittest.TestCase):
    def test_two_sided_arbitrary_frame_matches_cartesian_dyadic(self):
        tensor = np.array([[.6+.1j, .12-.03j, -.07+.02j],
                           [.12-.03j, .35-.08j, .05+.04j],
                           [-.07+.02j, .05+.04j, .2+.06j]])
        pattern = prepare_point_pattern(_pattern_dict(tensor, frequencies=(1.,)))
        normal = np.array([1., -2., 3.])
        roll = np.array([2., 1., .5])
        frame = _local_frame(normal, roll)
        azimuth = np.deg2rad([15., 80., 150., 210., 285., 340.])
        elevation = np.deg2rad([-10., -25., -40., -60., -75., -90.])
        local_dirs = np.column_stack((np.cos(elevation)*np.cos(azimuth),
                                     np.cos(elevation)*np.sin(azimuth), np.sin(elevation)))
        directions = local_dirs @ frame.T
        location = np.array([.037, -.021, .014])
        actual = point_scatterer_amplitude(pattern, location, normal, directions, 1., roll_ref=roll, one_sided=False)
        default = point_scatterer_amplitude(pattern, location, normal, directions, 1., roll_ref=roll)
        body_tensor = frame @ tensor @ frame.T
        expected = []
        for direction in directions:
            azimuth = math.atan2(direction[1], direction[0])
            transverse = math.hypot(direction[0], direction[1])
            horizontal = np.array([-math.sin(azimuth), math.cos(azimuth), 0.])
            vertical = np.array([direction[2]*math.cos(azimuth), direction[2]*math.sin(azimuth), -transverse])
            jones = _jones_from_tensor(body_tensor, vertical, horizontal)
            phase = np.exp(4j*math.pi*1e9/C0*(direction @ location))
            expected.append(np.array([jones[0, 0], jones[1, 1], jones[0, 1]])*phase)
        for index, channel in enumerate(('F_vv', 'F_hh', 'F_vh')):
            np.testing.assert_allclose(actual[channel], np.asarray(expected)[:, index], rtol=2e-11, atol=2e-13)
            np.testing.assert_array_equal(default[channel], 0.)

    def test_mixed_sidedness_cache_and_geometric_shadowing_are_independent(self):
        pattern = prepare_point_pattern(_pattern_dict(np.eye(3), frequencies=(1.,)))
        point = dict(pattern=pattern, location=np.array([0., 0., 1.]),
                     aperture_normal=np.array([0., 0., 1.]), roll_ref=np.array([1., 0., 0.]))
        directions = np.array([[0., 0., 1.], [0., 0., -1.]])
        points = [point, dict(point, one_sided=False)]
        free = sum_features(None, [], directions, 1., points=points)
        phase = np.exp(4j*math.pi*1e9/C0*directions[:, 2])
        np.testing.assert_allclose(free['amp_vv'], np.array([2., 1.])*phase, atol=1e-12)

        plate = Occluder(np.array([[[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.]],
                                   [[-2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]]]))
        shadowed = sum_features(None, [], directions, 1., points=points, occluder=plate)
        np.testing.assert_allclose(shadowed['amp_vv'], np.array([2., 0.])*phase, atol=1e-12)
        # A point below the plate facing down has an unobstructed backside look
        # downward when its local normal points up. Packed masks must not cull it.
        below = dict(point, location=np.array([0., 0., -1.]), one_sided=False)
        visible_back = sum_features(None, [], directions, 1., points=[below], occluder=plate)
        np.testing.assert_allclose(visible_back['amp_vv'], [0., phase[0]], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
