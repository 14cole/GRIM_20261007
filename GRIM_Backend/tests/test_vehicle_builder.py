"""Vehicle authoring must preserve imported geometry and current membership."""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "GHOST"))

from GRIM_Backend.assembly.model import (
    FeatureAssemblyValues, FeatureWorkflowAdapter,
    POINT_PLACEMENT_COLUMNS, LINE_PLACEMENT_COLUMNS,
)
from GRIM_Backend.assembly.vehicle_builder import (
    prepare_vehicle_feature, prepare_vehicle_feature_removal,
    portable_vehicle_values, VehicleBuilderMixin,
)


def write_rows(path, kind, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(POINT_PLACEMENT_COLUMNS if kind == "point" else LINE_PLACEMENT_COLUMNS)
        writer.writerows(rows)


POINT = ["fastener_1", "fastener", "0", "0", "1", "0", "0", "1", "1", "0", "0"]
LINE = ["gap_1", "gap", "1", "0", "0", "1", "1", "0", "1", "0", "0", "1", "0", "0", "1"]


class VehicleBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The genuine parser exercises strict row, orientation and chain rules.
        from ghost_backend.assembly.workflow import discover_feature_dataset_ids
        cls.service = FeatureWorkflowAdapter(dict, discover_feature_dataset_ids,
                                             lambda request: request, lambda plan: plan)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.drafts = self.root / "drafts"
        self.response = self.root / "response.grim"
        self.response.write_bytes(b"response bytes validated during physical preparation")

    def definition(self, kind="point", name="fastener"):
        return dict(kind=kind, name=name, units="meters", response="response.grim",
                    rows=[list(POINT if kind == "point" else LINE)])

    def add(self, values, definition=None):
        return prepare_vehicle_feature(values, definition or self.definition(),
                                        directory=self.drafts, service=self.service)

    def test_add_preserves_import_and_exclusions_assigns_unique_family(self):
        source = self.root / "points.csv"
        write_rows(source, "point", [POINT])
        original = source.read_bytes()
        values = FeatureAssemblyValues(base_dir=str(self.root), coordinate_units="meters",
            point_locations_csv="points.csv", point_datasets={"fastener": "original.grim"},
            excluded_point_placement_ids={"fastener_1"})
        before = copy.deepcopy(values)
        edit = self.add(values)
        self.assertEqual(values, before)
        self.assertEqual(source.read_bytes(), original)
        self.assertNotEqual(edit.created_path, source)
        self.assertEqual(edit.dataset_id, "fastener_2")
        self.assertEqual(edit.model.values.point_datasets["fastener"], "original.grim")
        self.assertEqual(edit.model.values.point_datasets["fastener_2"], str(self.response))
        self.assertEqual(edit.model.values.excluded_point_placement_ids, {"fastener_1"})
        self.assertEqual(edit.model.point_instances,
                         (("fastener_1", "fastener"), ("fastener_1_2", "fastener_2")))
        self.assertTrue(edit.model.requirements_are_current("point"))

    def test_line_add_preserves_chain_grouping_and_other_kind(self):
        points = self.root / "points.csv"
        lines = self.root / "lines.csv"
        write_rows(points, "point", [POINT])
        write_rows(lines, "line", [LINE])
        values = FeatureAssemblyValues(base_dir=str(self.root), coordinate_units="meters",
            point_locations_csv="points.csv", line_locations_csv="lines.csv",
            point_datasets={"fastener": "response.grim"}, line_datasets={"gap": "response.grim"},
            excluded_point_placement_ids={"fastener_1"}, excluded_line_ids={"gap_1"})
        definition = self.definition("line", "gap")
        second = list(LINE)
        second[2], second[3], second[6] = "2", "1", "2"
        definition["rows"].append(second)
        edit = self.add(values, definition)
        self.assertEqual(edit.instance_count, 1)
        self.assertEqual(edit.model.line_instances, (("gap_1", "gap", 1), ("gap_1_2", "gap_2", 2)))
        self.assertEqual(edit.model.values.excluded_line_ids, {"gap_1"})
        self.assertEqual(edit.model.values.excluded_point_placement_ids, {"fastener_1"})
        self.assertTrue(edit.model.requirements_are_current("point"))

    def test_parser_failure_leaves_values_and_imported_source_intact(self):
        source = self.root / "points.csv"
        write_rows(source, "point", [POINT])
        original = source.read_bytes()
        values = FeatureAssemblyValues(base_dir=str(self.root), coordinate_units="meters",
            point_locations_csv="points.csv", point_datasets={"fastener": "response.grim"})
        before = copy.deepcopy(values)
        definition = self.definition()
        definition["rows"][0][2] = "not-a-number"
        with self.assertRaises(ValueError):
            self.add(values, definition)
        self.assertEqual(values, before)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(list(self.drafts.glob("*.csv")), [])

    def test_first_feature_sets_units_but_does_not_reinterpret_existing_units(self):
        values = FeatureAssemblyValues(base_dir=str(self.root))
        edit = self.add(values)
        self.assertEqual(edit.model.values.coordinate_units, "meters")
        self.assertEqual(values.coordinate_units, "")
        values.coordinate_units = "inches"
        with self.assertRaisesRegex(ValueError, "Assembly placement units"):
            self.add(values)

    def test_remove_family_preserves_other_rows_and_clears_last_csv(self):
        first = self.add(FeatureAssemblyValues(base_dir=str(self.root)))
        second = self.add(first.model.values, self.definition(name="antenna"))
        second.model.values.excluded_point_placement_ids = {"fastener_1"}
        before_bytes = second.created_path.read_bytes()
        removed = prepare_vehicle_feature_removal(second.model.values, "point", "fastener",
            directory=self.drafts, service=self.service)
        self.assertEqual(second.created_path.read_bytes(), before_bytes)
        self.assertEqual(removed.model.point_dataset_ids, ("antenna",))
        self.assertEqual(removed.model.values.excluded_point_placement_ids, set())
        empty = prepare_vehicle_feature_removal(removed.model.values, "point", "antenna",
            directory=self.drafts, service=self.service)
        self.assertEqual(empty.model.values.point_locations_csv, "")
        self.assertEqual(empty.model.values.point_datasets, {})
        self.assertEqual(empty.model.point_instances, ())

    def test_recipe_copies_only_managed_geometry_and_preserves_saved_variant(self):
        edit = self.add(FeatureAssemblyValues(base_dir=str(self.root)))
        imported = self.root / "imported_lines.csv"
        write_rows(imported, "line", [LINE])
        edit.model.values.line_locations_csv = str(imported)
        recipe = self.root / "vehicle.assembly.json"
        before = copy.deepcopy(edit.model.values)
        portable = portable_vehicle_values(before, recipe, draft_directory=self.drafts)
        copied = Path(portable.point_locations_csv)
        self.assertEqual(copied.parent, recipe.parent)
        self.assertEqual(copied.read_bytes(), edit.created_path.read_bytes())
        self.assertEqual(portable.line_locations_csv, str(imported))
        second = portable_vehicle_values(before, recipe, draft_directory=self.drafts)
        self.assertNotEqual(portable.point_locations_csv, second.point_locations_csv)
        self.assertEqual(edit.model.values, before)

    def test_edit_guard_blocks_busy_operation_or_open_editor(self):
        class Host(VehicleBuilderMixin):
            busy = False
            _placement_editors = {}
            def job_is_running(self):
                return self.busy
        host = Host()
        host.busy = True
        with self.assertRaisesRegex(ValueError, "current Assembly operation"):
            host._ensure_vehicle_editable()
        host.busy = False
        host._placement_editors = {"point": object()}
        with self.assertRaisesRegex(ValueError, "open placement editor"):
            host._ensure_vehicle_editable()


try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is unavailable")
class VehiclePanelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_authored_point_and_gap_reach_real_coherent_saved_response(self):
        """Exercise the new add actions through physical preparation and output."""
        import numpy as np
        from dataclasses import replace
        from GRIM_Backend.assembly.panel import FeatureAssemblyPanel
        from GRIM_Backend.tests.test_feature_assembly_panel import (
            _write_closed_box_facet, _write_isotropic_line_delta,
        )
        from ghost_backend.assembly import workflow
        from ghost_backend.assembly.line_expansion import C0, PSI_HH_DEG, PSI_VV_DEG
        from ghost_backend.tests.test_point_scatter_physics import (
            _write_point_grim, _write_component_grim,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body, mesh = root / "body.grim", root / "body.facet"
            point_response, line_response = root / "point.grim", root / "gap.grim"
            output = root / "vehicle.grim"
            _write_closed_box_facet(mesh)
            clean = np.full((2, 2, 1, 3), 0.01 + 0.003j, dtype=complex)
            _write_component_grim(body, [0., 45.], [30., 60.], [2.], clean)
            _write_point_grim(point_response, np.diag([.006+.001j, .003-.001j, .002+.001j]), [2.])
            _write_isotropic_line_delta(line_response, frequency_ghz=2.,
                installed_coefficient=.31-.12j, c0=C0,
                psi_hh_deg=PSI_HH_DEG, psi_vv_deg=PSI_VV_DEG)
            with patch.dict(os.environ, GRIM_ASSEMBLY_DRAFT_DIR=str(root / "drafts")):
                panel = FeatureAssemblyPanel(service=workflow)
            try:
                panel.set_base_grim(str(body))
                panel.set_surface_mesh(str(mesh))
                panel.surface_units.setCurrentIndex(panel.surface_units.findData("meters"))
                panel.set_output_grim(str(output))
                panel.add_vehicle_feature(dict(kind="point", name="Antenna", units="meters",
                    response=str(point_response), rows=[
                        ["antenna", "unused", "0", "0.1", "0.1", "0", "0", "1", "1", "0", "0"],
                    ]))
                panel.add_vehicle_feature(dict(kind="line", name="Panel gap", units="meters",
                    response=str(line_response), rows=[
                        ["gap", "unused", "1", "-0.08", "-0.1", "0.1", "0.08", "-0.1", "0.1", "0", "0", "1", "0", "0", "1"],
                    ]))
                panel._geometry_preview_timer.stop()
                panel._pull_values()
                dispatch = panel.model.assemble(workflow)
                self.assertTrue(output.is_file())
                self.assertEqual(len(dispatch.plan.point_placements), 1)
                self.assertEqual(len(dispatch.plan.line_placements), 1)
                point_only = root / "point_only.grim"
                line_only = root / "line_only.grim"
                workflow.execute_feature_assembly(workflow.prepare_feature_assembly(replace(
                    dispatch.plan.request, output_grim=point_only, enabled_line_ids=())))
                workflow.execute_feature_assembly(workflow.prepare_feature_assembly(replace(
                    dispatch.plan.request, output_grim=line_only, enabled_point_placement_ids=())))
                def field(path):
                    with np.load(path, allow_pickle=False) as data:
                        return data["rcs_amp_real"] + 1j * data["rcs_amp_imag"]
                total, only_point, only_line = field(output), field(point_only), field(line_only)
                self.assertGreater(float(np.linalg.norm(only_point-clean)), 1e-8)
                self.assertGreater(float(np.linalg.norm(only_line-clean)), 1e-8)
                np.testing.assert_allclose(total, only_point + only_line - clean, rtol=1e-10, atol=1e-12)
            finally:
                panel._geometry_preview_timer.stop()
                panel._recipe_dirty = False
                panel.close()
                panel.deleteLater()
                self.app.processEvents()

    def test_add_remove_and_saved_variant_survive_draft_removal(self):
        from GRIM_Backend.assembly.panel import FeatureAssemblyPanel
        from GRIM_Backend.assembly.recipe import read_feature_assembly_recipe
        from ghost_backend.assembly import workflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drafts = root / "drafts"
            response = root / "feature.grim"
            response.write_bytes(b"validated later")
            with patch.dict(os.environ, GRIM_ASSEMBLY_DRAFT_DIR=str(drafts)):
                panel = FeatureAssemblyPanel(service=workflow)
            try:
                point_id = panel.add_vehicle_feature(dict(kind="point", name="Fasteners", units="meters",
                    response=str(response), rows=[POINT]))
                self.assertEqual(point_id, "Fasteners")
                panel.model.values.excluded_point_placement_ids = {"fastener_1"}
                panel._refresh_spatial_feature_tree()
                line_id = panel.add_vehicle_feature(dict(kind="line", name="Panel gap", units="meters",
                    response=str(response), rows=[LINE]))
                panel._geometry_preview_timer.stop()
                self.assertEqual(panel.point_mapping.mapping(), {point_id: str(response)})
                self.assertEqual(panel.line_mapping.mapping(), {line_id: str(response)})
                self.assertEqual(panel.model.values.excluded_point_placement_ids, {"fastener_1"})
                self.assertEqual(panel.vehicle_feature_controls["point"]["table"].rowCount(), 1)
                self.assertEqual(panel.vehicle_feature_controls["line"]["table"].rowCount(), 1)
                recipe = panel.save_recipe_path(root / "vehicle.assembly.json")
                variant = panel.create_variant_path(root / "variant.assembly.json", "Variant")
                # The unique managed sources can disappear without breaking a saved recipe.
                for source in drafts.glob("*.csv"):
                    source.unlink()
                for saved in (recipe, variant):
                    loaded = read_feature_assembly_recipe(saved)
                    self.assertEqual(loaded.source_warnings, ())
                    self.assertTrue(Path(loaded.values.point_locations_csv).is_file())
                    self.assertTrue(Path(loaded.values.line_locations_csv).is_file())
                panel.load_recipe_path(recipe, refresh=False)
                panel.model.discover_dataset_ids(workflow)
                panel._apply_requirements_to_tables()
                panel.remove_vehicle_feature("point", point_id)
                self.assertEqual(panel.point_csv_picker.path(), "")
                self.assertEqual(panel.line_mapping.mapping(), {line_id: str(response)})
            finally:
                panel._geometry_preview_timer.stop()
                panel._recipe_dirty = False
                panel.close()
                panel.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
