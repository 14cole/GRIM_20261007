"""Preview topology and screen-space overlay regression tests."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.geometry.preview import build_material_faces, select_normal_samples, visible_segment_midpoint


def segment(points, left="air", right="PEC"):
    pairs = list(zip(points, points[1:]))
    return SimpleNamespace(x=[p[0] for pair in pairs for p in pair],
                           y=[p[1] for pair in pairs for p in pair],
                           materials=(left, "blue", right, "gray"))


def faces(segments, **kwargs):
    return build_material_faces(segments, lambda s: s.materials, **kwargs)


class MaterialFaceTests(unittest.TestCase):
    def test_thin_region_with_three_way_material_junctions(self):
        # A narrow coating above a substrate shares a boundary. All four
        # corners are individual rows; endpoints at the seam have degree 3.
        segments = [
            segment([(0, 0), (10, 0), (10, 1)], "d1", "air"),
            segment([(10, 1), (0, 1)], "d1", "d2"),
            segment([(0, 1), (0, 0)], "d1", "air"),
            segment([(10, 1), (10, 1.01), (0, 1.01), (0, 1)], "d2", "air"),
        ]
        result = faces(segments)
        self.assertEqual(len(result), 2)
        by_label = {item["label"]: item for item in result}
        self.assertEqual(set(by_label), {"d1", "d2"})
        self.assertAlmostEqual(by_label["d2"]["area"], 0.1)
        self.assertTrue(all(item["consistent"] for item in result))

    def test_nested_thin_shell_preserves_hole_and_depth(self):
        outer = segment([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], "d1", "air")
        inner = segment([(.01, .01), (9.99, .01), (9.99, 9.99), (.01, 9.99), (.01, .01)], "air", "d1")
        result = sorted(faces([inner, outer]), key=lambda item: item["depth"])
        self.assertEqual([item["label"] for item in result], ["d1", "air"])
        self.assertEqual([item["depth"] for item in result], [0, 1])
        self.assertAlmostEqual(result[0]["area"] - result[1]["area"], .3996)

    def test_discontinuous_primitives_do_not_invent_a_closing_edge(self):
        broken = segment([(0, 0), (1, 0), (1, 1), (0, 0)])
        broken.x[2:4] = [0, 1]
        broken.y[2:4] = [1, 1]
        self.assertEqual(faces([broken]), [])

    def test_inconsistent_materials_flag_only_the_bad_cell(self):
        boundary = [segment([(0, 0), (2, 0), (2, 2)], "d1", "air"),
                    segment([(2, 2), (0, 2), (0, 0)], "d2", "air")]
        result = faces(boundary)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["consistent"])
        self.assertEqual(result[0]["rows"], [0, 1])

    def test_bor_axis_closure_is_virtual_and_component_local(self):
        first = segment([(0, 0), (1, 1), (0, 2)], "PEC", "air")
        second = segment([(0, 4), (1, 5), (0, 6)], "PEC", "air")
        original = (list(first.x), list(first.y))
        self.assertEqual(faces([first, second]), [])
        result = faces([first, second], close_axis=True)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(item["area"] == 1 for item in result))
        self.assertEqual((first.x, first.y), original)

    def test_bor_off_axis_open_chain_is_not_closed(self):
        profile = segment([(1, 0), (2, 1), (1, 2)])
        self.assertEqual(faces([profile], close_axis=True), [])

    def test_close_endpoints_across_rounding_buckets_are_joined(self):
        # Old round(x / tolerance) keys separate points less than tolerance apart.
        a = segment([(0, 0), (1, 0), (1, 1)], "PEC", "air")
        b = segment([(1 + 4e-10, 1), (0, 1), (0, 0)], "PEC", "air")
        self.assertEqual(len(faces([a, b])), 1)

    def test_reversing_primitives_and_material_sides_retains_fill(self):
        original = segment([(0, 0), (1, 0), (1, .01), (0, .01), (0, 0)], "d1", "air")
        reversed_segment = SimpleNamespace(x=original.x[::-1], y=original.y[::-1],
                                           materials=("air", "gray", "d1", "blue"))
        first = faces([original])[0]
        second = faces([reversed_segment])[0]
        self.assertEqual(first["label"], second["label"])
        self.assertAlmostEqual(first["area"], second["area"])


class NormalSampleTests(unittest.TestCase):
    def test_dense_neighbors_are_thinned_and_selected_row_has_priority(self):
        candidates = [{"midpoint": (x, y), "row": i, "priority": 2}
                      for i, (x, y) in enumerate([(0, 0), (0, 1), (10, 0), (40, 0)])]
        candidates[1]["priority"] = 0
        result = select_normal_samples(candidates)
        self.assertEqual([item["row"] for item in result], [1, 3])

    def test_reversing_input_keeps_same_arrow_anchors(self):
        candidates = [{"midpoint": (x, 0), "row": 0} for x in (0, 20, 40, 60)]
        first = select_normal_samples(candidates)
        second = select_normal_samples(reversed(candidates))
        self.assertEqual(first, second)

    def test_viewport_excludes_nonfinite_and_offscreen_samples(self):
        candidates = [{"midpoint": point} for point in ((0, 0), (50, 50), (100, 50), (math.nan, 0))]
        self.assertEqual(select_normal_samples(candidates, bounds=(10, 10, 90, 90)), [candidates[1]])

    def test_arrow_footprints_do_not_overlap_even_when_anchors_are_spaced(self):
        first = {"midpoint": (0, 0), "footprint": (-5, -5, 30, 5)}
        second = {"midpoint": (40, 0), "footprint": (25, -5, 50, 5)}
        self.assertEqual(select_normal_samples([first, second]), [first])



class VisibleSegmentTests(unittest.TestCase):
    def test_long_segment_keeps_an_anchor_in_zoomed_view(self):
        self.assertEqual(visible_segment_midpoint((-100, 5), (1000, 5), (0, 0, 10, 10)), (5, 5))

    def test_clipped_diagonal_anchor(self):
        self.assertEqual(visible_segment_midpoint((-10, -10), (10, 10), (0, 0, 10, 10)), (5, 5))

    def test_visible_segment_retains_midpoint(self):
        self.assertEqual(visible_segment_midpoint((2, 3), (8, 5), (0, 0, 10, 10)), (5, 4))

    def test_reversed_segment_has_identical_anchor(self):
        a, b, box = (-151.6, 14.3), (350.7, 5.4), (0, 0, 80, 20)
        self.assertEqual(visible_segment_midpoint(a, b, box), visible_segment_midpoint(b, a, box))

    def test_disjoint_and_nonfinite_segments_are_excluded(self):
        self.assertIsNone(visible_segment_midpoint((-10, 1), (-5, 1), (0, 0, 10, 10)))
        self.assertIsNone(visible_segment_midpoint((math.nan, 1), (5, 1), (0, 0, 10, 10)))

    def test_boundary_and_point_segments(self):
        self.assertEqual(visible_segment_midpoint((-1, 0), (11, 0), (0, 0, 10, 10)), (5, 0))
        self.assertEqual(visible_segment_midpoint((4, 4), (4, 4), (0, 0, 10, 10)), (4, 4))

if __name__ == "__main__":
    unittest.main()
