"""Gap measurements retain narrow clearances and stored primitive topology."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.geometry.measurements import (
    closest_primitive_points, closest_segment_points, point_to_primitive,
)


def segment(*primitives):
    return SimpleNamespace(x=[v for p in primitives for v in (p[0], p[2])],
                           y=[v for p in primitives for v in (p[1], p[3])])


class PrimitiveMeasurementTests(unittest.TestCase):
    def test_local_projection_measures_point_zero_one_gap(self):
        point, gap = point_to_primitive((3.5, 0), (0, .01, 10, .01))
        self.assertEqual(point, (3.5, .01))
        self.assertAlmostEqual(gap, .01)

    def test_projection_clamps_to_endpoint(self):
        self.assertEqual(point_to_primitive((3, 2), (0, 0, 1, 0)), ((1, 0), math.sqrt(8)))

    def test_point_primitives_do_not_divide_by_zero(self):
        self.assertEqual(point_to_primitive((3, 4), (0, 0, 0, 0)), ((0, 0), 5))
        self.assertEqual(closest_primitive_points((0, 0, 0, 0), (3, 4, 3, 4))[2], 5)

    def test_parallel_overlap_has_positive_gap(self):
        first, second, gap = closest_primitive_points((0, 0, 10, 0), (2, .01, 8, .01))
        self.assertAlmostEqual(gap, .01)
        self.assertAlmostEqual(first[0], second[0])

    def test_crossing_is_zero_away_from_endpoints(self):
        first, second, gap = closest_primitive_points((0, 0, 2, 2), (0, 2, 2, 0))
        self.assertEqual((first, second, gap), ((1, 1), (1, 1), 0))

    def test_almost_parallel_crossing_is_not_discarded(self):
        first, second, gap = closest_primitive_points((0, 0, 1, 1e-12), (0, 1e-12, 1, 0))
        self.assertEqual(gap, 0)
        self.assertAlmostEqual(first[0], .5)
        self.assertEqual(first, second)

    def test_shared_endpoint_and_sloping_collinear_overlap_are_zero(self):
        self.assertEqual(closest_primitive_points((0, 0, 1, 1), (1, 1, 2, 0))[2], 0)
        self.assertEqual(closest_primitive_points((0, 0, 3, 3), (1, 1, 2, 2))[2], 0)

    def test_disjoint_collinear_primitives_retain_gap(self):
        self.assertEqual(closest_primitive_points((0, 0, 1, 0), (2, 0, 3, 0)),
                         ((1, 0), (2, 0), 1))

    def test_reversing_and_swapping_primitives_preserves_distance(self):
        a, b = (1, 2, 5, 7), (2, 9, 8, 3)
        expected = closest_primitive_points(a, b)[2]
        for first, second in ((b, a), (a[2:] + a[:2], b), (a, b[2:] + b[:2])):
            self.assertAlmostEqual(closest_primitive_points(first, second)[2], expected)

    def test_small_positive_distance_is_not_snapped_to_contact(self):
        self.assertAlmostEqual(closest_primitive_points((0, 0, 1, 0), (0, 1e-12, 1, 1e-12))[2],
                               1e-12, delta=1e-25)

    def test_nonfinite_input_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                point_to_primitive((value, 0), (0, 0, 1, 0))
            with self.assertRaises(ValueError):
                closest_primitive_points((0, 0, 1, 0), (0, value, 1, 1))


class RowMeasurementTests(unittest.TestCase):
    def test_row_minimum_reports_primitive_indices(self):
        first = segment((0, 0, 1, 0), (3, 0, 4, 0))
        second = segment((3, .01, 4, .01), (0, 1, 1, 1))
        result = closest_segment_points(first, second)
        self.assertAlmostEqual(result['distance'], .01)
        self.assertEqual((result['first_primitive'], result['second_primitive']), (1, 0))

    def test_disconnected_pairs_do_not_invent_connecting_line(self):
        first = segment((0, 0, 1, 0), (3, 0, 4, 0))
        second = segment((2, -.1, 2, .1))
        self.assertEqual(closest_segment_points(first, second)['distance'], 1)

    def test_shared_tip_minimum_and_local_clearance_have_distinct_meanings(self):
        first = segment((0, 0, 10, 0))
        second = segment((0, 0, 0, .01), (0, .01, 10, .01))
        self.assertEqual(closest_segment_points(first, second)['distance'], 0)
        self.assertAlmostEqual(point_to_primitive((5, 0), (0, .01, 10, .01))[1], .01)

    def test_empty_row_has_no_measurement(self):
        self.assertIsNone(closest_segment_points(segment(), segment((0, 0, 1, 1))))

    def test_malformed_coordinate_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            closest_segment_points(SimpleNamespace(x=[0, 1, 2], y=[0, 1]), segment())

    def test_search_can_be_canceled(self):
        calls = []
        def checkpoint():
            calls.append(1)
            if len(calls) == 2:
                raise InterruptedError('Canceled')
        first = segment(*[(i, 0, i + .5, 0) for i in range(20)])
        second = segment(*[(i, 1, i + .5, 1) for i in range(20)])
        with self.assertRaises(InterruptedError):
            closest_segment_points(first, second, checkpoint)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
