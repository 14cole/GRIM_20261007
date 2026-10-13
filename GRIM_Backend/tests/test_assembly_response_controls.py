"""Regression coverage for per-response Assembly modeling assumptions."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from GRIM_Backend.assembly.model import FeatureAssemblyFormModel, FeatureAssemblyValues
from GRIM_Backend.assembly.recipe import read_feature_assembly_recipe, write_feature_assembly_recipe
from GRIM_Backend.assembly.values import LINE_RESPONSE_DEFAULTS, normalize_response_settings
from GRIM_Backend.tests.test_feature_assembly_panel import _FakeWorkflow, _ready_point_model, _close_panel_without_prompt


class ResponseSettingsModelTests(unittest.TestCase):
    def test_response_settings_reach_request_and_invalidate_validated_plan(self):
        model = _ready_point_model()
        model.values.line_locations_csv = "lines.csv"
        model.update_dataset_requirements({"point_dataset_ids": ("fastener",), "line_dataset_ids": ("gap",)})
        model.values.line_datasets = {"gap": "gap.grim"}
        workflow = _FakeWorkflow()
        settings = {"psi_tm_deg": 0.0, "psi_te_deg": 27.5, "grazing_taper_deg": 0.0}
        model.values.line_response_settings = {"gap": settings, "unused": dict(LINE_RESPONSE_DEFAULTS)}
        model.values.point_response_settings = {"fastener": {"one_sided": False}}
        request = model.build_request(workflow)
        self.assertEqual(request.kwargs["line_response_settings"], {"gap": settings})
        self.assertEqual(request.kwargs["point_response_settings"], {"fastener": {"one_sided": False}})
        for field, replacement in (("psi_tm_deg", 5), ("psi_te_deg", 10), ("grazing_taper_deg", 7)):
            model.prepare_preview(workflow)
            self.assertTrue(model.validated_plan_is_current(workflow))
            model.values.line_response_settings["gap"][field] = replacement
            self.assertFalse(model.validated_plan_is_current(workflow))
        model.prepare_preview(workflow)
        model.values.point_response_settings["fastener"]["one_sided"] = True
        self.assertFalse(model.validated_plan_is_current(workflow))

    def test_recipe_round_trip_and_v5_default_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            values = FeatureAssemblyValues(
                point_response_settings={"antenna": {"one_sided": False}},
                line_response_settings={"gap": {"psi_tm_deg": 0, "psi_te_deg": -23, "grazing_taper_deg": 0}},
            )
            path = write_feature_assembly_recipe(values, Path(folder) / "vehicle", name="Vehicle", variant="Custom")
            loaded = read_feature_assembly_recipe(path)
            self.assertEqual(loaded.values.point_response_settings, values.point_response_settings)
            self.assertEqual(loaded.values.line_response_settings, values.line_response_settings)
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["version"] = 5
            del doc["values"]["line_response_settings"]
            del doc["values"]["point_response_settings"]
            path.write_text(json.dumps(doc), encoding="utf-8")
            legacy = read_feature_assembly_recipe(path).values
            self.assertEqual(legacy.point_response_settings, {})
            self.assertEqual(legacy.line_response_settings, {})

    def test_only_invalid_response_values_are_rejected(self):
        self.assertEqual(normalize_response_settings({"gap": {"psi_tm_deg": -720}}, "line")["gap"],
                         {**LINE_RESPONSE_DEFAULTS, "psi_tm_deg": -720})
        self.assertEqual(normalize_response_settings({"gap": {"grazing_taper_deg": 120}}, "line")["gap"]["grazing_taper_deg"], 120)
        for field, value in (("psi_tm_deg", float("nan")), ("psi_te_deg", float("inf")), ("grazing_taper_deg", -1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                normalize_response_settings({"gap": {field: value}}, "line")
        with self.assertRaises(ValueError):
            normalize_response_settings({"antenna": {"one_sided": "false"}}, "point")


class ResponseSettingsGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_phase_presets_preserve_taper_and_allow_custom_phase(self):
        from GRIM_Backend.assembly.panel import _ResponseModelDialog
        dialog = _ResponseModelDialog("line", "gap", {"grazing_taper_deg": 0})
        self.assertEqual(dialog.settings(), {**LINE_RESPONSE_DEFAULTS, "grazing_taper_deg": 0})
        dialog.preset.setCurrentIndex(1)
        self.assertEqual(dialog.settings(), {"psi_tm_deg": 0, "psi_te_deg": 0, "grazing_taper_deg": 0})
        dialog.fields["psi_tm_deg"].setText("37.125")
        dialog.fields["psi_tm_deg"].textEdited.emit("37.125")
        self.assertEqual(dialog.preset.currentIndex(), 2)
        self.assertEqual(dialog.settings()["psi_tm_deg"], 37.125)
        dialog.preset.setCurrentIndex(0)
        self.assertEqual(dialog.settings(), {**LINE_RESPONSE_DEFAULTS, "grazing_taper_deg": 0})
        dialog.reject()

    def test_point_pattern_choices_are_explicit_and_independent(self):
        from GRIM_Backend.assembly.panel import _ResponseModelDialog
        dialog = _ResponseModelDialog("point", "antenna", {})
        self.assertEqual(dialog.settings(), {"one_sided": True})
        dialog.sidedness.setCurrentIndex(1)
        self.assertEqual(dialog.settings(), {"one_sided": False})
        dialog.reject()

    def test_panel_settings_dirty_and_invalidate_then_survive_recipe_loading(self):
        from GRIM_Backend.assembly.panel import FeatureAssemblyPanel
        panel = FeatureAssemblyPanel(service=_FakeWorkflow())
        try:
            panel.model.update_dataset_requirements({"point_dataset_ids": ("antenna",), "line_dataset_ids": ("gap",)})
            panel._apply_requirements_to_tables()
            panel._validated_plan_current = True
            panel._recipe_dirty = False
            panel.line_mapping.set_response_settings("gap", {"psi_tm_deg": 0, "psi_te_deg": 15, "grazing_taper_deg": 0})
            panel.point_mapping.set_response_settings("antenna", {"one_sided": False})
            self.assertFalse(panel._validated_plan_current)
            self.assertTrue(panel._recipe_dirty)
            self.assertIn("taper off", panel.line_mapping._model_buttons["gap"].text())
            panel._pull_values()
            self.assertEqual(panel.model.values.point_response_settings, {"antenna": {"one_sided": False}})
            expected = {"gap": {"psi_tm_deg": 0, "psi_te_deg": 15, "grazing_taper_deg": 0}}
            with tempfile.TemporaryDirectory() as folder:
                path = write_feature_assembly_recipe(panel.model.values, Path(folder) / "vehicle", name="Vehicle", variant="Custom")
                panel.load_recipe_path(path, refresh=False)
                panel._pull_values()
                self.assertEqual(panel.model.values.line_response_settings, expected)
                self.assertEqual(panel.point_mapping.response_settings(), {"antenna": {"one_sided": False}})
        finally:
            _close_panel_without_prompt(panel)
