"""Keep physical cut selections when the active dataset changes."""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

from GRIM_Backend.datasets.grid import RcsGrid
import GRIM_Backend.ui.app as grim_cut_gui
from test_gui_shell import (
    _FakeFeatureWorkflow, _FakeFreddyIntegration, _FakeGhostIntegration,
    _MemorySettings, _RecordingWindow,
)


def _grid(*, azimuths=(-2, -1, 0, 1, 2), elevations=(-20, 0, 20),
          frequencies=(8, 9, 10), polarizations=("HH", "VV"), radians=False,
          hz=False):
    angles = np.deg2rad if radians else np.asarray
    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    return RcsGrid(
        angles(azimuths), angles(elevations),
        np.asarray(frequencies) * (1e9 if hz else 1), polarizations,
        rcs_power=np.ones(shape), rcs_phase=np.zeros(shape),
        units={"azimuth": "rad" if radians else "deg",
               "elevation": "rad" if radians else "deg",
               "frequency": "Hz" if hz else "GHz"},
    )


class ParameterSelectionPersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        for name, replacement in (
            ("GhostIntegrationWidget", _FakeGhostIntegration),
            ("FreddyIntegrationWidget", _FakeFreddyIntegration),
        ):
            patch = mock.patch.object(grim_cut_gui, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(grim_cut_gui, "load_ghost_module",
                                  return_value=_FakeFeatureWorkflow())
        patch.start()
        self.addCleanup(patch.stop)
        self.window = _RecordingWindow(settings=_MemorySettings())
        self.window.btn_auto_plot.setChecked(False)

    def tearDown(self):
        self.window.deleteLater()
        self.app.processEvents()

    def add(self, grid):
        row = self.window.table.rowCount()
        self.window._add_dataset_row(grid, f"Dataset {row}", "", "")
        return row

    def activate(self, row):
        self.window.table.setCurrentCell(
            row, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows
        )

    def select(self, axis, values):
        widget = self.widget(axis)
        blocked = widget.blockSignals(True)
        try:
            widget.clearSelection()
            for row in range(widget.count()):
                item = widget.item(row)
                item.setSelected(item.data(Qt.UserRole) in values)
        finally:
            widget.blockSignals(blocked)
        if axis == "polarization":
            self.window._on_polarization_selection_changed()

    def widget(self, axis):
        return getattr(self.window, {"frequency": "list_freq", "elevation": "list_elev",
                                    "azimuth": "list_az", "polarization": "list_pol"}[axis])

    def selected(self, axis):
        return self.window._selected_values(self.widget(axis))

    def test_first_dataset_keeps_existing_defaults(self):
        self.activate(self.add(_grid()))
        self.assertEqual(self.selected("frequency"), [8])
        self.assertEqual(self.selected("elevation"), [-20])
        self.assertEqual(self.selected("polarization"), ["VV"])
        self.assertEqual(self.selected("azimuth"), [-2, -1, 0, 1, 2])

    def test_switch_preserves_waterline_frequency_and_multiple_polarizations(self):
        first, second = self.add(_grid()), self.add(_grid(polarizations=("VV", "HH")))
        self.activate(first)
        self.select("frequency", [9])
        self.select("elevation", [0])
        self.select("polarization", ["HH", "VV"])
        with mock.patch.object(self.window, "_maybe_autoplot") as autoplot:
            self.activate(second)
        self.assertEqual(self.selected("frequency"), [9])
        self.assertEqual(self.selected("elevation"), [0])
        self.assertEqual(set(self.selected("polarization")), {"HH", "VV"})
        self.assertEqual(autoplot.call_count, 1)

    def test_switch_converts_physical_cuts_across_hz_and_radians(self):
        first, second = self.add(_grid()), self.add(_grid(radians=True, hz=True))
        self.activate(first)
        self.select("frequency", [9])
        self.select("elevation", [20])
        self.select("azimuth", [-1, 0, 1])
        self.activate(second)
        np.testing.assert_allclose(self.selected("frequency"), [9e9])
        np.testing.assert_allclose(self.selected("elevation"), np.deg2rad([20]))
        np.testing.assert_allclose(self.selected("azimuth"), np.deg2rad([-1, 0, 1]))
        self.activate(first)
        self.assertEqual(self.selected("frequency"), [9])
        self.assertEqual(self.selected("elevation"), [20])
        self.assertEqual(self.selected("azimuth"), [-1, 0, 1])

    def test_contiguous_span_keeps_new_native_samples_inside_range(self):
        first = self.add(_grid())
        second = self.add(_grid(azimuths=np.arange(-3, 3.1, .5)))
        self.activate(first)
        self.select("azimuth", [-1, 0, 1])
        self.activate(second)
        self.assertEqual(self.selected("azimuth"), [-1, -.5, 0, .5, 1])

    def test_separate_cut_choices_stay_separate(self):
        first, second = self.add(_grid()), self.add(_grid(frequencies=(8, 8.5, 9, 9.5, 10)))
        self.activate(first)
        self.select("frequency", [8, 10])
        self.activate(second)
        self.assertEqual(self.selected("frequency"), [8, 10])

    def test_unavailable_fixed_cut_uses_default_instead_of_nearest(self):
        first, second = self.add(_grid()), self.add(_grid(frequencies=(7, 8, 9.1)))
        self.activate(first)
        self.select("frequency", [9])
        self.select("elevation", [0])
        self.activate(second)
        self.assertEqual(self.selected("frequency"), [7])
        self.assertEqual(self.selected("elevation"), [0])
        self.assertIn("frequency unavailable; using default", self.window.status.currentMessage())
        self.assertEqual(self.window._pending_parameter_selection_notice,
                         self.window.status.currentMessage())

        # A subsequent successful switch must not carry the earlier fallback
        # notice into its pending automatic plot.
        self.select("frequency", [8])
        self.activate(first)
        self.assertEqual(self.selected("frequency"), [8])
        self.assertEqual(self.window._pending_parameter_selection_notice, "")

    def test_preserves_choices_through_transient_empty_table_selection(self):
        first, second = self.add(_grid()), self.add(_grid())
        self.activate(first)
        self.select("frequency", [10])
        self.select("elevation", [0])
        self.window.table.clearSelection()
        self.assertIsNone(self.window.active_dataset)
        self.activate(second)
        self.assertEqual(self.selected("frequency"), [10])
        self.assertEqual(self.selected("elevation"), [0])

    def test_missing_polarization_keeps_available_selected_channels(self):
        first, second = self.add(_grid()), self.add(_grid(polarizations=("VV", "HV")))
        self.activate(first)
        self.select("polarization", ["HH", "VV"])
        self.activate(second)
        self.assertEqual(self.selected("polarization"), ["VV"])
        self.assertIn("polarization limited to available values", self.window.status.currentMessage())

    def test_no_samples_at_restored_cut_falls_back_after_availability_filtering(self):
        first = self.add(_grid())
        other = _grid()
        other.rcs_power[:, :, 1, :] = np.nan
        second = self.add(other)
        self.activate(first)
        self.select("frequency", [9])
        self.activate(second)
        self.assertEqual(self.selected("frequency"), [8])
        self.assertIn("frequency unavailable; using default", self.window.status.currentMessage())

    def test_available_discrete_cuts_survive_partial_sample_availability(self):
        first = self.add(_grid())
        other = _grid()
        other.rcs_power[:, :, 2, :] = np.nan
        second = self.add(other)
        self.activate(first)
        self.select("frequency", [8, 10])
        self.activate(second)
        self.assertEqual(self.selected("frequency"), [8])
        self.assertIn("frequency limited to available values", self.window.status.currentMessage())


if __name__ == "__main__":
    unittest.main()
