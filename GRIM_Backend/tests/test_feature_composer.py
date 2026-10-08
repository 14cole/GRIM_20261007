"""Feature authoring checks: geometry, schema, and the direct-add dialog."""
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from GRIM_Backend.assembly.feature_composer import (
    AddFeatureDialog, GUI_AVAILABLE, compose_line_rows, compose_point_rows,
)
from GRIM_Backend.assembly.model import LINE_PLACEMENT_COLUMNS, POINT_PLACEMENT_COLUMNS


class ComposeFeatureRowsTests(unittest.TestCase):
    def test_point_row_places_repeats_and_constructs_orthonormal_frame(self):
        rows = compose_point_rows("Wing fasteners", "fastener_response", (1, 2, 3),
                                  (0, 0, 5), (4, 0, 4), count=3, spacing=(0, .25, 0))
        self.assertEqual(len(rows[0]), len(POINT_PLACEMENT_COLUMNS))
        self.assertEqual([row[0] for row in rows], ["wing_fasteners_001", "wing_fasteners_002", "wing_fasteners_003"])
        self.assertEqual([tuple(map(float, row[2:5])) for row in rows], [(1, 2, 3), (1, 2.25, 3), (1, 2.5, 3)])
        self.assertEqual(tuple(map(float, rows[0][5:])), (0, 0, 1, 1, 0, 0))
        self.assertEqual(rows[0][1], "fastener_response")

    def test_point_rejects_nonfinite_degenerate_or_overlapping_geometry(self):
        arguments = {"name": "fastener", "dataset_id": "response", "position": (0, 0, 0), "normal": (0, 0, 1), "roll": (1, 0, 0)}
        cases = [({"position": (math.nan, 0, 0)}, "finite"),
                 ({"normal": (0, 0, 0)}, "nonzero"),
                 ({"roll": (0, 0, 2)}, "parallel"),
                 ({"count": 2}, "nonzero repeat spacing"),
                 ({"count": 1.5}, "whole number"),
                 ({"count": True}, "whole number"),
                 ({"count": 10001}, "whole number"),
                 ({"spacing": (math.inf, 0, 0)}, "finite"),
                 ({"count": 3, "position": (1e308, 0, 0), "spacing": (1e308, 0, 0)}, "finite"),
                 ({"count": 2, "position": (1e20, 0, 0), "spacing": (1e-10, 0, 0)}, "too small")]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                compose_point_rows(**(arguments | overrides))

    def test_closed_line_continuity_and_per_vertex_normals(self):
        vertices = [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0)]
        normals = [(0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)]
        rows = compose_line_rows("Door gap", "gap", vertices, normals, closed=True)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(len(row) == len(LINE_PLACEMENT_COLUMNS) for row in rows))
        self.assertEqual([row[2] for row in rows], ["1", "2", "3", "4"])
        self.assertEqual({row[:2] for row in rows}, {("door_gap", "gap")})
        for first, second in zip(rows, rows[1:] + rows[:1]):
            self.assertEqual(first[6:9], second[3:6])
            self.assertEqual(first[12:15], second[9:12])
        self.assertEqual(rows[-1][6:9], ("0", "0", "0"))

    def test_line_rejects_bad_topology_and_normal_frames(self):
        cases = [([(0, 0, 0)], [(0, 0, 1)], False, "two vertices"),
                 ([(0, 0, 0), (1, 0, 0)], [(0, 0, 1)] * 2, True, "three vertices"),
                 ([(0, 0, 0), (0, 0, 0)], [(0, 0, 1)] * 2, False, "coincide"),
                 ([(0, 0, 0), (1, 0, 0)], [(0, 0, 0)] * 2, False, "nonzero"),
                 ([(0, 0, 0), (1, 0, 0)], [(1, 0, 0)] * 2, False, "parallel"),
                 ([(0, 0, 0), (1, 0, 0)], [(0, 0, 1), (0, 0, -1)], False, "oppose"),
                 ([(0, 0, 0), (1, 0, 0), (0, 0, 0)], [(0, 0, 1)] * 3, False, "backtracks"),
                 ([(0, 0, 0), (1, 0, 0), (0, 0, 0)], [(0, 0, 1)] * 3, True, "repeated last"),
                 ([(0, 0, 0), (math.inf, 0, 0)], [(0, 0, 1)] * 2, False, "finite"),
                 ([(0, 0, 0), (1, 0, 0)], [(0, 0, 1)], False, "one outward normal")]
        for vertices, normals, closed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compose_line_rows("gap", "response", vertices, normals, closed)


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 is unavailable")
class AddFeatureDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.response = Path(self.directory.name) / "feature.grim"
        self.response.touch()

    def dialog(self, kind="point", units="meters", **kwargs):
        dialog = AddFeatureDialog(kind, units=units, **kwargs)
        self.addCleanup(dialog.close)
        dialog.response_edit.setText(str(self.response))
        return dialog

    def test_requires_explicit_units_and_shows_errors_inline(self):
        dialog = self.dialog(units="")
        dialog.accept()
        self.assertEqual(dialog.result(), 0)
        self.assertIn("Choose placement units", dialog.error_label.text())
        dialog.units_combo.setCurrentIndex(dialog.units_combo.findData("inches"))
        dialog.accept()
        self.assertEqual(dialog.result(), 1)
        self.assertEqual(dialog.feature_definition()["units"], "inches")

    def test_existing_units_are_locked_and_repeats_need_deliberate_spacing(self):
        dialog = self.dialog()
        self.assertFalse(dialog.units_combo.isEnabled())
        dialog.count_spin.setValue(4)
        dialog.accept()
        self.assertIn("repeat spacing", dialog.error_label.text())
        dialog.spacing_fields[1].setText("0.2")
        dialog.accept()
        self.assertEqual(dialog.result(), 1)
        rows = dialog.feature_definition()["rows"]
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(float(rows[-1][3]), .6)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.response])

    def test_reuse_response_and_name_presets_preserve_custom_names(self):
        dialog = self.dialog(dataset_choices={"Rivet": str(self.response)})
        dialog.response_choice.setCurrentIndex(1)
        self.assertEqual(dialog.name_edit.text(), "Rivet")
        self.assertEqual(dialog.response_edit.text(), str(self.response))
        dialog.name_edit.setText("Nose inlets")
        dialog.feature_type.setCurrentText("Inlet")
        self.assertEqual(dialog.name_edit.text(), "Nose inlets")

    def test_file_and_table_errors_stay_in_dialog(self):
        point = self.dialog()
        point.response_edit.setText(str(self.response.with_suffix(".csv")))
        point.accept()
        self.assertIn("existing .grim", point.error_label.text())
        line = self.dialog("line")
        line.vertices_table.item(1, 0).setText("NaN")
        line.accept()
        self.assertEqual(line.result(), 0)
        self.assertIn("Vertex 2", line.error_label.text())

    def test_line_starter_authors_one_segment_and_close_loop_needs_third_vertex(self):
        line = self.dialog("line")
        self.assertEqual(len(line.feature_definition()["rows"]), 1)
        line.close_loop.setChecked(True)
        line.accept()
        self.assertIn("three vertices", line.error_label.text())
        line._append_vertex((1, 1, 0, 0, 0, 1))
        line.accept()
        self.assertEqual(line.result(), 1)
        self.assertEqual(len(line.feature_definition()["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
