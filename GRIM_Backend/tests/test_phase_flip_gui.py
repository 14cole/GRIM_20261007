"""Phase correction is reachable, non-destructive, and recorded from the GUI."""
import time
from unittest import mock

import numpy as np
from PySide6.QtCore import Qt

from GRIM_Backend.datasets.operations import operation_events
from test_plot_analysis_features import _WindowCase


class PhaseFlipGuiTests(_WindowCase):
    def wait_for_operation(self):
        deadline = time.monotonic() + 10
        while self.window._background_job_active() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window._background_job_active())

    def test_button_flips_each_selected_dataset_without_changing_sources(self):
        self.select_rows(0, 1)
        sources = self.datasets[:2]
        for index, source in enumerate(sources):
            source.rcs_phase[:] = .4 + index * .2
        before = [source.rcs.copy() for source in sources]
        count = self.window.table.rowCount()
        recorder = self.window.python_recorder
        with mock.patch.object(recorder, 'record_function', wraps=recorder.record_function) as record:
            self.window.btn_flip_phase.click()
            self.wait_for_operation()
        self.assertEqual(self.window.table.rowCount(), count + 2)
        self.assertEqual(record.call_count, 2)
        for output_index in range(2):
            item = self.window.table.item(count + output_index, 0)
            index = 0 if item.text().startswith('Run 1 ') else 1
            source = sources[index]
            result = item.data(Qt.UserRole)
            self.assertIsNot(result, source)
            np.testing.assert_allclose(result.rcs, before[index].conjugate())
            np.testing.assert_allclose(source.rcs, before[index])
            np.testing.assert_allclose(result.rcs_power, source.rcs_power)
            self.assertEqual(operation_events(result)[-1]['operation'], 'flip_phase')
            self.assertEqual(record.call_args_list[output_index].args[1], 'flip_phase')
            self.assertIn('Phase flipped', item.text())

    def test_action_availability_and_double_flip_for_unknown_phase(self):
        self.window.table.clearSelection()
        self.window._update_dataset_action_states()
        self.assertFalse(self.window.btn_flip_phase.isEnabled())
        self.select_rows(0)
        source = self.datasets[0]
        source.rcs_phase[:] = np.nan
        self.assertTrue(self.window.btn_flip_phase.isEnabled())
        self.window.btn_flip_phase.click()
        self.wait_for_operation()
        row = self.window.table.rowCount() - 1
        self.select_rows(row)
        self.window.btn_flip_phase.click()
        self.wait_for_operation()
        restored = self.window.table.item(self.window.table.rowCount() - 1, 0).data(Qt.UserRole)
        np.testing.assert_array_equal(restored.rcs_power, source.rcs_power)
        self.assertTrue(np.isnan(restored.rcs_phase).all())
