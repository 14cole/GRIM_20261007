"""Behavior regressions for the jointly reviewed Assembly fixes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/GHOST'))
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QFontDatabase, QFont
from GRIM_Backend.assembly.panel import FeatureAssemblyPanel, _OperationWorker, POINT_PLACEMENT_COLUMNS
from GRIM_Backend.assembly.placement_editor import PlacementEditor
from GRIM_Backend.assembly.model import FeatureAssemblyFormModel, read_feature_assembly_recipe
from GRIM_Backend.assembly.workflow import suggest_response_mapping
from ghost_backend.assembly import workflow


class AssemblyImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if Path('C:/Windows/Fonts/segoeui.ttf').is_file():
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
            cls.app.setFont(QFont('Segoe UI', 9))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, GRIM_ASSEMBLY_DRAFT_DIR=self.temp.name)
        self.env.start()
        self.panel = FeatureAssemblyPanel(service=workflow)

    def tearDown(self):
        self.panel._geometry_preview_timer.stop()
        for editor in tuple(self.panel._placement_editors.values()):
            editor._saved_rows = editor.rows()
            editor.reject()
        self.panel._recipe_dirty = False
        self.panel.close()
        self.panel.deleteLater()
        self.app.processEvents()
        self.env.stop()
        self.temp.cleanup()

    def test_editor_context_is_locked_and_programmatic_changes_fail_closed(self):
        p = self.panel
        p.coordinate_units.setCurrentIndex(p.coordinate_units.findData('inches'))
        p._edit_placements('point')
        editor = p._placement_editors['point']
        editor.add()
        p._update_workflow_readiness()
        self.assertFalse(p.coordinate_units.isEnabled())
        self.assertFalse(p.base_picker.isEnabled())
        p.coordinate_units.setCurrentIndex(p.coordinate_units.findData('meters'))
        with mock.patch.object(workflow, 'discover_feature_dataset_ids') as validate:
            editor.save()
            validate.assert_not_called()
        self.assertIn('context', editor.status.text())
        self.assertFalse(editor.save_path.exists())
        with self.assertRaisesRegex(RuntimeError, 'placement editor'):
            p.load_recipe_path(Path(self.temp.name)/'irrelevant.json')
        editor._saved_rows = editor.rows()
        editor.reject()
        self.assertTrue(p.coordinate_units.isEnabled())

    def test_cell_undo_does_not_scan_or_copy_all_rows(self):
        e = PlacementEditor('point', columns=POINT_PLACEMENT_COLUMNS, units='meters')
        try:
            e.change([[str(i),'bolt','0','0','0','0','0','1','1','0','0'] for i in range(10000)])
            with mock.patch.object(e, 'rows', side_effect=AssertionError('full-table scan')):
                e.table.item(5000, 2).setText('0.123')
                e.undo()
                self.assertEqual(e.table.item(5000,2).text(), '0')
                e.redo()
                self.assertEqual(e.table.item(5000,2).text(), '0.123')
            self.assertLess(e._history[-1][1], 200)
            e.change(e.rows()[:3])
            e.undo()
            self.assertEqual(e.table.rowCount(), 10000)
            self.assertEqual(e.rows()[5000][2], '0.123')
            e.undo()
            self.assertEqual(e.rows()[5000][2], '0')
        finally:
            e._saved_rows = e.rows(); e.reject()

    def test_bad_completion_is_transactional_and_reported_through_real_signal(self):
        p = self.panel
        p.model.update_dataset_requirements({'point_dataset_ids':['old'], 'point_instances':[('kept','old')]})
        p._apply_requirements_to_tables()
        p._active_kind = 'discover'
        p._discovery_paths = (p.point_csv_picker.path(),p.line_csv_picker.path())
        errors = []
        p.build_failed.connect(errors.append)
        worker = _OperationWorker(lambda: None)
        worker.succeeded.connect(p._operation_succeeded)
        worker.succeeded.emit({'point_dataset_ids':['new'], 'point_instances':[('a','new'),('a','new')]})
        self.assertEqual(p.model.point_dataset_ids, ('old',))
        self.assertEqual(p.model.point_instances, (('kept','old'),))
        self.assertTrue(errors)
        self.assertIn('duplicate',errors[0])
        self.assertFalse(p._validated_plan_current)

    def test_unnamed_draft_retains_units_and_edits_without_recipe_prompt(self):
        p = self.panel
        p.coordinate_units.setCurrentIndex(p.coordinate_units.findData('meters'))
        p.output_picker.set_path(str(Path(self.temp.name)/'result.grim'))
        p._recipe_dirty = True
        with mock.patch.object(QMessageBox,'warning') as prompt:
            self.assertTrue(p.request_close())
            prompt.assert_not_called()
        loaded = read_feature_assembly_recipe(p._draft_path)
        self.assertEqual(loaded.values.coordinate_units, 'meters')
        self.assertTrue(loaded.values.output_grim.endswith('result.grim'))

    def test_warning_details_are_bounded_and_acknowledgement_stays_visible(self):
        p=self.panel
        p.validation_profile.setCurrentIndex(2)
        p._pull_values()
        p._show_validation_qa(SimpleNamespace(validation_warnings=['x'*280 for _ in range(200)],point_records=[],line_records=[]))
        self.assertLess(len(p.validation_warning_label.text()), 100)
        self.assertEqual(p.validation_warning_text.toPlainText().count('x'),56000)
        self.assertLessEqual(p.validation_warning_text.maximumHeight(),180)
        self.assertFalse(p.validation_warning_ack.isHidden())

    def test_disclosure_title_and_open_state_survive_refresh(self):
        section=self.panel.advanced_section
        section.set_title('Updated options')
        section.header.setChecked(True)
        self.assertEqual(section.header.text(),'−  Updated options')
        section.header.setChecked(False)
        self.assertEqual(section.header.text(),'+  Updated options')
        geometry=self.panel.body_geometry_section
        geometry.header.setChecked(True)
        self.panel.shadow.setChecked(True)
        self.panel.shadow.setChecked(False)
        self.assertTrue(geometry.header.isChecked())

    def test_calculate_chains_validation_and_stops_for_strict_warnings(self):
        p=self.panel
        with mock.patch.object(p.model,'validated_plan_is_current',return_value=False), \
             mock.patch.object(p,'validate_and_preview') as validate:
            p.calculate_and_save()
            validate.assert_called_once()
        for count, expected in [(0,1),(1,0)]:
            p._calculate_pending=True; p._validated_plan_current=True
            p._active_kind='preview'; p._validation_warning_count=count
            with mock.patch.object(p,'_set_busy'), mock.patch.object(p,'assemble_and_save') as build:
                p._operation_thread_finished()
                self.app.processEvents()
                self.assertEqual(build.call_count,expected)

    def test_stale_exclusions_are_reported(self):
        model=FeatureAssemblyFormModel()
        model.values.excluded_point_placement_ids={'old-name'}
        model.update_dataset_requirements({'point_dataset_ids':['bolt'],'point_instances':[('new-name','bolt')]})
        self.assertEqual(model.enabled_point_placement_ids,('new-name',))
        self.assertIn('old-name',model.membership_advisories[0])

    def test_library_scan_is_case_insensitive_cancellable_and_preserves_mapping(self):
        root=Path(self.temp.name)
        (root/'Bolt.GRIM').write_bytes(b'a'); (root/'kept.grim').write_bytes(b'b')
        unique,_,_=suggest_response_mapping(['bolt','kept'],root,{'kept':'chosen.grim'})
        self.assertEqual(set(unique),{'bolt'})
        with self.assertRaises(InterruptedError):
            suggest_response_mapping(['bolt'],root,cancel_check=lambda:True)


if __name__ == '__main__': unittest.main()
