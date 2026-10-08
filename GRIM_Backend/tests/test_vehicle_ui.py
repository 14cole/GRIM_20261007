"""Named feature controls must mirror physical membership and request readiness."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "GHOST"))

from GRIM_Backend.assembly.feature_composer import compose_line_rows, compose_point_rows
from GRIM_Backend.assembly.model import FeatureWorkflowAdapter
from GRIM_Backend.assembly.panel import GUI_AVAILABLE


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 is unavailable")
class VehicleFeatureUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from ghost_backend.assembly.workflow import discover_feature_dataset_ids
        cls.app = QApplication.instance() or QApplication([])
        cls.service = FeatureWorkflowAdapter(dict, discover_feature_dataset_ids,
                                             lambda request: request, lambda plan: plan)

    def setUp(self):
        from GRIM_Backend.assembly.panel import FeatureAssemblyPanel
        from GRIM_Backend.tests.test_feature_assembly_panel import _write_minimal_base_grim
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.response = self.root / "feature.grim"
        self.response.write_bytes(b"physical response content is validated by the backend")
        body = self.root / "body.grim"
        _write_minimal_base_grim(body, embedded_bor=True)
        with patch.dict(os.environ, GRIM_ASSEMBLY_DRAFT_DIR=str(self.root / "drafts")):
            self.panel = FeatureAssemblyPanel(service=self.service)
        self.addCleanup(self.close_panel)
        self.panel.set_base_grim(str(body))
        self.panel.set_output_grim(str(self.root / "vehicle.grim"))
        self.point_id = self.add_point("Fasteners", count=3)
        self.line_id = self.panel.add_vehicle_feature(dict(kind="line", name="Panel gap", units="meters",
            response=str(self.response), rows=compose_line_rows("gap", "response",
                [(0, 0, 1), (1, 0, 1), (1, 1, 1)], [(0, 0, 1)] * 3)))
        self.panel._geometry_preview_timer.stop()

    def close_panel(self):
        self.panel._geometry_preview_timer.stop()
        for editor in list(self.panel._placement_editors.values()):
            editor.reject()
        self.panel._recipe_dirty = False
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()

    def add_point(self, name, count=1):
        dataset_id = self.panel.add_vehicle_feature(dict(kind="point", name=name, units="meters",
            response=str(self.response), rows=compose_point_rows(name, "response", (0, 0, 1),
                (0, 0, 1), (1, 0, 0), count=count, spacing=(.1, 0, 0))))
        self.panel._geometry_preview_timer.stop()
        return dataset_id

    def controls(self, kind):
        return self.panel.vehicle_feature_controls[kind]

    def test_clear_all_resets_vehicle_without_touching_source_or_output_files(self):
        from PySide6.QtTest import QSignalSpy
        from PySide6.QtWidgets import QTableWidgetItem
        from GRIM_Backend.assembly.values import FeatureAssemblyValues

        panel = self.panel
        output = Path(panel.output_picker.path())
        output.write_bytes(b"previously saved vehicle response")
        sources = [Path(panel.base_picker.path()), self.response, output,
                   Path(panel.point_csv_picker.path()), Path(panel.line_csv_picker.path())]
        original_bytes = {path: path.read_bytes() for path in sources}
        panel._recipe_path = self.root / "saved.assembly.json"
        panel._recipe_path.write_bytes(b"previous saved recipe")
        original_bytes[panel._recipe_path] = panel._recipe_path.read_bytes()
        panel._recipe_source_warnings = ("old source warning",)
        panel.model.values.excluded_point_placement_ids.add(panel.model.point_instances[0][0])
        panel._surface_binding_checked = {"old": True}
        panel._surface_dimensions_text = "old dimensions"
        panel.wing_table.setRowCount(1)
        panel.wing_table.setItem(0, 0, QTableWidgetItem("section.geo"))
        panel.wing_corner_table.setRowCount(1)
        panel.wing_output_picker.set_path("wing.grim")
        panel.wing_mirror.setChecked(True)
        panel.wing_angle_step.setValue(2.0)
        panel.wing_grid_fields["frequencies_ghz"].setText("2, 3")
        panel.recipe_name_edit.setText("Old vehicle")
        panel.recipe_variant_edit.setText("Option B")
        panel._validated_plan_current = panel._preview_is_current = True
        panel._calculate_pending = True
        panel._geometry_preview_pending = True
        panel._geometry_preview_timer.start()
        # Clearing must work even if inputs could not be parsed or saved.
        panel.shadow_bias.setText("invalid")
        panel.study_fields["study_frequencies_ghz"].setText("invalid")
        cleared = QSignalSpy(panel.assembly_cleared)
        with patch.object(panel, "_start_operation") as start:
            panel.clear_all_button.click()
            self.app.processEvents()
        start.assert_not_called()
        self.assertEqual(cleared.count(), 1)
        values = panel.model.values
        expected = FeatureAssemblyValues()
        for name in expected.__dataclass_fields__:
            if name == "skin_tol_m":
                self.assertAlmostEqual(getattr(values, name), getattr(expected, name))
            else:
                self.assertEqual(getattr(values, name), getattr(expected, name), name)
        self.assertEqual(panel.point_mapping.mapping(), {})
        self.assertEqual(panel.line_mapping.mapping(), {})
        self.assertEqual(panel.wing_table.rowCount(), 0)
        self.assertEqual(panel.wing_corner_table.rowCount(), 0)
        self.assertEqual(panel.wing_output_picker.path(), "")
        self.assertFalse(panel.wing_mirror.isChecked())
        self.assertEqual(panel.wing_angle_step.value(), 0.5)
        self.assertEqual(panel.workflow_tabs.currentIndex(), 0)
        self.assertEqual(panel.recipe_name_edit.text(), "Vehicle feature assembly")
        self.assertEqual(panel.recipe_variant_edit.text(), "Baseline")
        self.assertIsNone(panel._recipe_path)
        self.assertFalse(panel._recipe_dirty)
        self.assertFalse(panel._validated_plan_current)
        self.assertFalse(panel._preview_is_current)
        self.assertFalse(panel._calculate_pending)
        self.assertFalse(panel._geometry_preview_pending)
        self.assertFalse(panel._geometry_preview_timer.isActive())
        self.assertFalse(panel.build_button.isEnabled())
        self.assertIs(panel.service(), self.service)
        for path, contents in original_bytes.items():
            self.assertEqual(path.read_bytes(), contents)
        # A reset does not disconnect the workflow: a new import/add is usable.
        self.add_point("Fresh feature")
        self.assertEqual(panel.model.point_dataset_ids, ("Fresh_feature",))

    def test_clear_all_is_blocked_during_jobs_or_placement_edits(self):
        from PySide6.QtTest import QSignalSpy
        from types import SimpleNamespace
        from threading import Event

        panel = self.panel
        original = panel.point_csv_picker.path()
        cleared = QSignalSpy(panel.assembly_cleared)
        panel._thread = SimpleNamespace(isRunning=lambda: True)
        panel._set_busy(True)
        try:
            self.assertFalse(panel.clear_all_button.isEnabled())
            self.assertFalse(panel.clear_all())
            self.assertEqual(panel.point_csv_picker.path(), original)
        finally:
            panel._thread = None
            panel._set_busy(False)
        panel._placement_editors["point"] = object()
        panel._sync_editor_context_lock()
        try:
            self.assertFalse(panel.clear_all_button.isEnabled())
            self.assertFalse(panel.clear_all())
            self.assertEqual(panel.point_csv_picker.path(), original)
        finally:
            panel._placement_editors.clear()
            panel._sync_editor_context_lock()
        self.assertEqual(cleared.count(), 0)
        self.assertTrue(panel.clear_all_button.isEnabled())
        scan = SimpleNamespace(cancel=Event())
        panel.point_mapping._library_scan = scan
        try:
            self.assertTrue(panel.clear_all())
            self.assertTrue(scan.cancel.is_set())
        finally:
            panel.point_mapping._library_scan = None

    def test_point_csv_import_discovers_groups_and_preserves_source_and_lines(self):
        from PySide6.QtTest import QTest
        from GRIM_Backend.assembly.panel import POINT_PLACEMENT_COLUMNS

        coordinates = self.root / "vehicle-points.csv"
        source = (",".join(POINT_PLACEMENT_COLUMNS) + "\n"
                  "fastener-1,fastener,0,0,1,0,0,1,1,0,0\n"
                  "inlet-1,inlet,1,0,1,0,0,1,1,0,0\n"
                  "fastener-2,fastener,2,0,1,0,0,1,1,0,0\n").encode("utf-8")
        coordinates.write_bytes(source)
        line_path = self.panel.model.values.line_locations_csv
        line_source = Path(line_path).read_bytes()
        line_instances = self.panel.model.line_instances
        self.assertFalse(self.controls("point")["advanced"].header.isChecked())

        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(str(coordinates), "")) as browse, \
                patch.object(self.panel, "_schedule_geometry_preview"):
            self.controls("point")["import"].click()
            deadline = time.monotonic() + 5
            while self.panel.job_is_running() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.app.processEvents()
            self.assertFalse(self.panel.job_is_running(), "CSV discovery did not finish")
        browse.assert_called_once()

        self.assertEqual(self.panel.workflow_tabs.currentIndex(), 1)
        self.assertTrue(self.controls("point")["advanced"].header.isChecked())
        self.assertEqual(self.panel.model.values.point_locations_csv, str(coordinates))
        self.assertEqual(self.panel.model.point_dataset_ids, ("fastener", "inlet"))
        self.assertEqual(self.panel.model.point_instances, (
            ("fastener-1", "fastener"), ("inlet-1", "inlet"),
            ("fastener-2", "fastener")))
        table = self.controls("point")["table"]
        self.assertEqual(table.rowCount(), 2)
        self.assertEqual([table.item(row, 2).text() for row in range(2)], ["2", "1"])
        self.assertEqual(self.panel.model.missing_dataset_mappings(),
                         ("point:fastener", "point:inlet"))
        self.assertFalse(self.panel.build_button.isEnabled())
        self.assertEqual(coordinates.read_bytes(), source)
        self.assertEqual(self.panel.model.values.line_locations_csv, line_path)
        self.assertEqual(Path(line_path).read_bytes(), line_source)
        self.assertEqual(self.panel.model.line_instances, line_instances)
        self.assertEqual(self.panel.line_mapping.mapping(), {self.line_id: str(self.response)})

    def test_cancel_point_csv_import_keeps_existing_placements_and_details_collapsed(self):
        controls = self.controls("point")
        previous_path = self.panel.model.values.point_locations_csv
        previous_source = Path(previous_path).read_bytes()
        previous_instances = self.panel.model.point_instances
        self.assertFalse(controls["advanced"].header.isChecked())
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")), \
                patch.object(self.panel, "refresh_dataset_ids") as discover:
            controls["import"].click()
        discover.assert_not_called()
        self.assertFalse(controls["advanced"].header.isChecked())
        self.assertEqual(self.panel.model.values.point_locations_csv, previous_path)
        self.assertEqual(self.panel.point_csv_picker.path(), previous_path)
        self.assertEqual(Path(previous_path).read_bytes(), previous_source)
        self.assertEqual(self.panel.model.point_instances, previous_instances)
        self.assertEqual(self.panel.point_mapping.mapping(), {self.point_id: str(self.response)})

    def test_line_csv_import_uses_connected_segments_and_preserves_points(self):
        from PySide6.QtTest import QTest
        from GRIM_Backend.assembly.panel import LINE_PLACEMENT_COLUMNS

        coordinates = self.root / "vehicle-panel-gaps.csv"
        source = (",".join(LINE_PLACEMENT_COLUMNS) + "\n"
                  "door-1,panel_gap,1,0,0,1,1,0,1,0,0,1,0,0,1\n"
                  "door-1,panel_gap,2,1,0,1,1,1,1,0,0,1,0,0,1\n").encode("utf-8")
        coordinates.write_bytes(source)
        point_path = self.panel.model.values.point_locations_csv
        point_source = Path(point_path).read_bytes()
        point_instances = self.panel.model.point_instances
        controls = self.controls("line")
        self.assertFalse(controls["advanced"].header.isChecked())

        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName",
                   return_value=(str(coordinates), "")) as browse, \
                patch.object(self.panel, "_schedule_geometry_preview"), \
                patch.object(self.panel, "_add_vehicle_feature") as composer:
            controls["import"].click()
            deadline = time.monotonic() + 5
            while self.panel.job_is_running() and time.monotonic() < deadline:
                QTest.qWait(10)
            self.app.processEvents()
            self.assertFalse(self.panel.job_is_running(), "CSV discovery did not finish")
        browse.assert_called_once()
        composer.assert_not_called()

        self.assertEqual(self.panel.workflow_tabs.currentIndex(), 2)
        self.assertTrue(controls["advanced"].header.isChecked())
        self.assertEqual(self.panel.model.values.line_locations_csv, str(coordinates))
        self.assertEqual(self.panel.model.line_dataset_ids, ("panel_gap",))
        self.assertEqual(self.panel.model.line_instances, (("door-1", "panel_gap", 2),))
        self.assertEqual(controls["table"].rowCount(), 1)
        self.assertEqual(controls["table"].item(0, 2).text(), "1")
        self.assertEqual(self.panel.model.missing_dataset_mappings(), ("line:panel_gap",))
        self.assertFalse(self.panel.build_button.isEnabled())
        self.assertEqual(coordinates.read_bytes(), source)
        self.assertEqual(self.panel.model.values.point_locations_csv, point_path)
        self.assertEqual(Path(point_path).read_bytes(), point_source)
        self.assertEqual(self.panel.model.point_instances, point_instances)
        self.assertEqual(self.panel.point_mapping.mapping(), {self.point_id: str(self.response)})

    def test_manual_placement_controls_are_only_available_through_more_tools(self):
        for kind in ("point", "line"):
            controls = self.controls(kind)
            with self.subTest(kind=kind):
                self.assertTrue(controls["import"].property("primaryAction"))
                for key in ("add", "edit"):
                    self.assertTrue(controls[key].isHidden())
                    action = controls[f"{key}_action"]
                    self.assertIn(action, self.panel.manual_placement_menu.actions())
                    self.assertTrue(action.isEnabled())
                with patch.object(self.panel, "_add_vehicle_feature") as add, \
                        patch.object(self.panel, "_edit_vehicle_placements") as edit:
                    controls["add_action"].trigger()
                    controls["edit_action"].trigger()
                    add.assert_called_once_with(kind)
                    edit.assert_called_once_with(kind)

                for mode in ("job", "editor"):
                    context = (patch.object(self.panel, "job_is_running", return_value=True)
                               if mode == "job" else
                               patch.object(self.panel, "_placement_editors", {kind: object()}))
                    with self.subTest(mode=mode), context:
                        self.panel._update_vehicle_selection_actions(kind)
                        for key in ("import", "add", "edit", "add_action", "edit_action"):
                            self.assertFalse(controls[key].isEnabled())
                    self.panel._update_vehicle_selection_actions(kind)
                controls["table"].setCurrentCell(-1, -1)
                self.panel._update_vehicle_selection_actions(kind)
                self.assertTrue(controls["add_action"].isEnabled())
                self.assertFalse(controls["edit_action"].isEnabled())

    def test_family_double_click_assigns_response_without_opening_placement_editor(self):
        for kind in ("point", "line"):
            with self.subTest(kind=kind), \
                    patch.object(self.panel, "_change_vehicle_response") as response, \
                    patch.object(self.panel, "_edit_vehicle_placements") as edit:
                self.controls(kind)["table"].cellDoubleClicked.emit(0, 1)
                response.assert_called_once_with(kind)
                edit.assert_not_called()

    def test_repeat_and_gap_rows_select_new_groups_and_remove_keeps_other_kind(self):
        points, lines = self.controls("point"), self.controls("line")
        self.assertEqual(points["table"].rowCount(), 1)
        self.assertEqual(points["table"].item(0, 2).text(), "3")
        self.assertEqual(lines["table"].item(0, 2).text(), "1")
        self.assertEqual(self.panel._vehicle_selected_dataset("line"), self.line_id)
        added = self.add_point("Antenna")
        self.assertEqual(self.panel._vehicle_selected_dataset("point"), added)
        self.assertTrue(points["edit"].isEnabled())
        line_path = self.panel.model.values.line_locations_csv
        self.panel._remove_vehicle_feature_clicked("point")
        self.assertEqual(self.panel.model.point_dataset_ids, (self.point_id,))
        self.assertEqual(self.panel._vehicle_selected_dataset("point"), self.point_id)
        self.assertEqual(self.panel.model.values.line_locations_csv, line_path)
        self.assertEqual(self.panel.line_mapping.mapping(), {self.line_id: str(self.response)})

    def test_partial_membership_updates_both_views_and_group_toggle_restores_all(self):
        from PySide6.QtCore import Qt
        point_ids = [item[0] for item in self.panel.model.point_instances]
        leaf = self.panel.spatial_feature_tree._instance_items[("point", point_ids[1])]
        self.panel._preview_is_current = self.panel._validated_plan_current = True
        leaf.setCheckState(2, Qt.Unchecked)
        table = self.controls("point")["table"]
        self.assertEqual(table.item(0, 0).checkState(), Qt.PartiallyChecked)
        self.assertEqual(table.item(0, 2).text(), "2/3")
        self.assertFalse(self.panel._preview_is_current)
        self.assertFalse(self.panel._validated_plan_current)
        self.assertEqual(self.panel.model.values.excluded_point_placement_ids, {point_ids[1]})
        table.item(0, 0).setCheckState(Qt.Unchecked)
        self.assertEqual(self.panel.model.values.excluded_point_placement_ids, set(point_ids))
        self.assertEqual(self.panel.model.active_point_dataset_ids(), ())
        self.assertEqual(table.item(0, 2).text(), "0/3")
        table.item(0, 0).setCheckState(Qt.Checked)
        self.assertEqual(self.panel.model.values.excluded_point_placement_ids, set())
        self.assertEqual(self.panel.model.values.excluded_line_ids, set())
        self.assertEqual(table.item(0, 2).text(), "3")
        for identifier in point_ids:
            self.assertEqual(self.panel.spatial_feature_tree._instance_items[("point", identifier)].checkState(2), Qt.Checked)

    def test_excluded_group_does_not_require_response_and_reenable_updates_readiness(self):
        from PySide6.QtCore import Qt
        table = self.controls("point")["table"]
        self.assertTrue(self.panel.build_button.isEnabled(), self.panel.next_step_label.text())
        table.item(0, 0).setCheckState(Qt.Unchecked)
        self.panel.point_mapping.set_path(self.point_id, "")
        self.assertEqual(self.panel.point_mapping.missing_ids(), ())
        self.assertEqual(self.panel.model.missing_dataset_mappings(), ())
        self.assertTrue(self.panel.build_button.isEnabled(), self.panel.next_step_label.text())
        self.panel.point_mapping.set_path(self.point_id, str(self.root / "missing.grim"))
        self.assertTrue(self.panel.build_button.isEnabled(), self.panel.next_step_label.text())
        table.item(0, 0).setCheckState(Qt.Checked)
        self.assertFalse(self.panel.build_button.isEnabled())
        self.assertIn("missing feature response", self.panel.next_step_label.text())
        self.panel.point_mapping.set_path(self.point_id, str(self.response))
        self.assertTrue(self.panel.build_button.isEnabled(), self.panel.next_step_label.text())

    def test_change_response_updates_visible_mapping_and_invalidates_review(self):
        replacement = self.root / "replacement.grim"
        replacement.write_bytes(b"different response")
        self.panel._preview_is_current = self.panel._validated_plan_current = True
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(replacement), "")):
            self.panel._change_vehicle_response("point")
        self.assertEqual(self.panel.model.values.point_datasets[self.point_id], str(replacement))
        table = self.controls("point")["table"]
        self.assertEqual(table.item(0, 3).text(), "replacement.grim")
        self.assertEqual(self.panel._vehicle_selected_dataset("point"), self.point_id)
        self.assertFalse(self.panel._preview_is_current)
        self.assertFalse(self.panel._validated_plan_current)
        self.assertEqual(self.panel.model.values.line_datasets[self.line_id], str(self.response))

    def test_busy_and_open_editor_lock_family_edits_and_ignore_queued_membership(self):
        from PySide6.QtCore import Qt
        controls = self.controls("point")
        for mode in ("job", "editor"):
            context = patch.object(self.panel, "job_is_running", return_value=True) if mode == "job" else patch.object(self.panel, "_placement_editors", {"point": object()})
            with self.subTest(mode=mode), context:
                self.panel._update_vehicle_selection_actions("point")
                self.assertFalse(controls["table"].isEnabled())
                for key in ("import", "add", "edit", "response", "remove"):
                    self.assertFalse(controls[key].isEnabled())
                controls["table"].item(0, 0).setCheckState(Qt.Unchecked)
                self.assertEqual(self.panel.model.values.excluded_point_placement_ids, set())
                self.assertEqual(controls["table"].item(0, 0).checkState(), Qt.Checked)
                with patch.object(self.panel.point_mapping, "_browse") as browse, patch.object(self.panel, "_edit_placements") as edit:
                    self.panel._change_vehicle_response("point")
                    self.panel._edit_vehicle_placements("point")
                    browse.assert_not_called()
                    edit.assert_not_called()
                with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName") as import_csv:
                    self.panel._import_vehicle_placements("point")
                    import_csv.assert_not_called()
            self.panel._update_vehicle_selection_actions("point")
            self.assertTrue(controls["table"].isEnabled())
            self.assertTrue(controls["edit"].isEnabled())

    def test_selected_group_opens_at_its_rows_and_blocks_calculation_until_editor_closes(self):
        added = self.add_point("Antenna")
        self.panel._edit_vehicle_placements("point")
        editor = self.panel._placement_editors["point"]
        self.assertEqual(editor.rows()[editor.table.currentRow()][1], added)
        self.assertFalse(self.panel.build_button.isEnabled())
        self.assertFalse(self.panel.preview_button.isEnabled())
        self.assertIn("Apply or close", self.panel.next_step_label.text())
        with patch.object(self.panel, "_start_operation") as start, patch.object(self.panel, "_show_error") as error:
            for action in (self.panel.calculate_and_save, self.panel.validate_and_preview,
                           self.panel.assemble_and_save):
                action()
                self.assertIn("open placement editor", str(error.call_args.args[0]))
            start.assert_not_called()
        self.assertFalse((self.root / "vehicle.grim").exists())
        editor.reject()
        self.assertEqual(self.panel._placement_editors, {})
        self.assertTrue(self.panel.build_button.isEnabled(), self.panel.next_step_label.text())
        self.assertTrue(self.controls("point")["table"].isEnabled())

    def test_family_table_stays_compact_for_small_and_large_vehicle_lists(self):
        controls = self.controls("point")
        self.assertLessEqual(controls["table"].height(), 100)
        self.assertTrue(controls["description"].isHidden())
        for index in range(6):
            self.add_point(f"Antenna {index}")
        self.assertEqual(controls["table"].rowCount(), 7)
        self.assertLessEqual(controls["table"].height(), 180)
        self.assertEqual(self.panel._vehicle_selected_dataset("point"), "Antenna_5")


if __name__ == "__main__":
    unittest.main()
