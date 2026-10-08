"""Radar-grid inputs must reach the BoR solve and the exported complex field."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication
from ghost_backend.ui.app import GhostWorkspace
from ghost_backend.runs.bor_setup import resource_summary
from ghost_backend.assembly.fields import load_body_grim, require_body_radar_support


class BorRadarGridGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.workspace = GhostWorkspace()
        self.tab = self.workspace.solver_tab
        self.tab.cmb_solver_kind.setCurrentIndex(self.tab.cmb_solver_kind.findData("bor"))

    def tearDown(self):
        self.tab._solve_worker = None
        self.tab._solve_thread = None
        self.tab._set_solving_state(False)
        self.workspace.close()
        self.workspace.deleteLater()
        self.app.processEvents()

    def test_elevation_modes_visibility_and_busy_state(self):
        tab = self.tab
        self.assertEqual(tab._capture_run_setup()["radar_grid"]["elevations_deg"], [0.0])
        self.assertFalse(tab.edit_bor_elev_list.isHidden())
        self.assertTrue(tab.bor_elev_sweep_row.isHidden())
        tab.cmb_bor_elev_mode.setCurrentIndex(1)
        tab.edit_bor_elev_start.setText("-30")
        tab.edit_bor_elev_stop.setText("30")
        tab.edit_bor_elev_step.setText("15")
        expected = [-30., -15., 0., 15., 30.]
        self.assertEqual(tab._capture_run_setup()["radar_grid"]["elevations_deg"], expected)
        self.assertTrue(tab.edit_bor_elev_list.isHidden())
        self.assertFalse(tab.bor_elev_sweep_row.isHidden())
        tab._set_solving_state(True)
        for control in (tab.cmb_bor_elev_mode, tab.edit_bor_elev_start,
                        tab.edit_bor_elev_stop, tab.edit_bor_elev_step):
            self.assertFalse(control.isEnabled())
        tab._set_solving_state(False)
        self.assertTrue(tab.edit_bor_elev_step.isEnabled())
        tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData("2d"))
        self.assertTrue(tab.cmb_bor_elev_mode.isHidden())
        self.assertTrue(tab.lbl_bor_elev_mode.isHidden())
        self.assertTrue(tab.bor_elev_sweep_row.isHidden())
        self.assertNotIn("radar_grid", tab._capture_run_setup())
        tab.cmb_solver_kind.setCurrentIndex(tab.cmb_solver_kind.findData("bor"))
        self.assertEqual(tab._capture_run_setup()["radar_grid"]["elevations_deg"], expected)

    def test_invalid_angles_and_oversized_grid_fail_before_solving(self):
        tab = self.tab
        for invalid in ("", "nan", "inf", "-91", "91", "0, 0", "bad"):
            with self.subTest(elevations=invalid):
                tab.edit_bor_elev_list.setText(invalid)
                with self.assertRaises(ValueError):
                    tab._capture_run_setup()
        tab.cmb_bor_elev_mode.setCurrentIndex(1)
        tab.edit_bor_elev_step.setText("0")
        with self.assertRaisesRegex(ValueError, "step must be > 0"):
            tab._capture_run_setup()
        tab.cmb_bor_elev_mode.setCurrentIndex(0)
        tab.edit_bor_elev_list.setText("0")
        tab.edit_elev_list.setText("0, 360")
        with self.assertRaisesRegex(ValueError, "do not include both"):
            tab._capture_run_setup()
        tab.edit_elev_list.setText(",".join(str(a) for a in range(360)))
        tab.edit_bor_elev_list.setText(",".join(str(e / 2) for e in range(-180, 181)))
        with mock.patch("ghost_backend.assembly.fields.radar_grid_aspects") as aspects:
            with self.assertRaisesRegex(ValueError, "100,000"):
                tab._capture_run_setup()
            aspects.assert_not_called()

    def test_solve_and_export_use_captured_grid_with_complex_polarization_rotation(self):
        tab = self.tab
        tab.edit_freq_list.setText("1, 2")
        tab.edit_elev_list.setText("30, 90, 210")
        tab.edit_bor_elev_list.setText("-30, 0, 30")
        tab.cmb_units.setCurrentText("meters")
        tab.chk_mesh_certification.setChecked(False)
        geometry = Path(__file__).resolve().parents[1] / "geometry/geometries/body.geo"
        tab.edit_geo_path.setText(str(geometry))
        with mock.patch("ghost_backend.ui.solver.QThread.start"), mock.patch(
            "ghost_backend.ui.solver.QMessageBox.critical"
        ) as errors:
            tab._run_solver()
        errors.assert_not_called()
        worker = tab._solve_worker
        self.assertIsNotNone(worker)
        context = copy.deepcopy(tab._pending_solve_context)
        azimuths, elevations = [30., 90., 210.], [-30., 0., 30.]
        # Independent horizontal-axis mapping, including noninteger aspects.
        az = np.deg2rad(azimuths)[:, None]
        el = np.deg2rad(elevations)[None, :]
        theta = np.rad2deg(np.arccos(np.cos(az) * np.cos(el)))
        expected_aspects = np.unique(np.round(theta, 12))
        np.testing.assert_allclose(worker.elevations, expected_aspects, atol=1e-10, rtol=0)
        self.assertEqual(worker.preflight_setup["radar_grid"], context["radar_grid"])
        channels = {"VV": [], "HH": []}
        for frequency in worker.frequencies:
            for angle in worker.elevations:
                for pol, amplitude in (
                    ("VV", frequency * (2 + np.sin(np.deg2rad(angle)) + .5j)),
                    ("HH", frequency * (1 + np.cos(np.deg2rad(angle)) - .25j)),
                ):
                    channels[pol].append(dict(frequency_ghz=frequency, theta_inc_deg=angle,
                        polarization=pol, rcs_amp_real=amplitude.real, rcs_amp_imag=amplitude.imag))
        result = dict(solver="bor_mom_rcs", scattering_mode="monostatic",
                      polarizations=["VV", "HH"], polarization_mapping={"VV": "VV", "HH": "HH"},
                      co_solved_samples=channels, metadata={})
        def solve_frequency(**kwargs):
            requested = set(kwargs["frequencies_ghz"])
            subset = {pol: [row for row in rows if row["frequency_ghz"] in requested]
                      for pol, rows in channels.items()}
            return dict(result, samples=subset["VV"] + subset["HH"], co_solved_samples=subset)
        with tempfile.TemporaryDirectory() as checkpoints:
            worker.checkpoint_directory = checkpoints
            with mock.patch("ghost_backend.ui.solver.solve_monostatic_rcs_bor_survey",
                            side_effect=solve_frequency) as solve:
                result = worker._run_bor()
                self.assertEqual(solve.call_count, 2)
                np.testing.assert_allclose(solve.call_args.kwargs["elevations_deg"], expected_aspects)
                self.assertEqual(len(result["co_solved_samples"]["VV"]), len(channels["VV"]))
        tab.last_solve_context = context
        tab._set_solving_state(False)
        tab.edit_elev_list.setText("0")
        tab.edit_bor_elev_list.setText("0")
        with tempfile.TemporaryDirectory() as directory:
            [saved] = tab._export_result_files(result, str(Path(directory) / "grid.grim"),
                                               source_path=str(geometry), history="GUI radar-grid regression")
            with np.load(saved, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["azimuths"], azimuths)
                np.testing.assert_array_equal(data["elevations"], elevations)
                np.testing.assert_array_equal(data["frequencies"], [1., 2.])
                self.assertEqual(data["polarizations"].tolist(), ["VV", "HH", "VH"])
                grid = json.loads(str(data["requested_radar_grid_json"]))
                self.assertEqual(grid["elevations_deg"], elevations)
                amp = data["rcs_amp_real"] + 1j * data["rcs_amp_imag"]
                self.assertEqual(amp.shape, (3, 3, 2, 3))
                np.testing.assert_allclose(data["rcs_power"], 4 * np.pi * np.abs(amp)**2,
                                           rtol=1e-6, atol=np.finfo(np.float32).tiny)
                # Meridian V is the body-axis projection into the transverse plane.
                # Its radar V/H projections yield the independent co-pol rotation.
                denominator = np.sin(np.deg2rad(theta))
                meridian_v_dot_radar_v = np.sin(el) * np.cos(az) / denominator
                meridian_v_dot_radar_h = -np.sin(az) / denominator
                body_vv = 2 + np.sin(np.deg2rad(theta)) + .5j
                body_hh = 1 + np.cos(np.deg2rad(theta)) - .25j
                expected_vv = body_vv * meridian_v_dot_radar_v**2 + body_hh * meridian_v_dot_radar_h**2
                expected_hh = body_vv * meridian_v_dot_radar_h**2 + body_hh * meridian_v_dot_radar_v**2
                expected_vh = (body_vv - body_hh) * meridian_v_dot_radar_v * meridian_v_dot_radar_h
                for i, expected in enumerate((expected_vv, expected_hh, expected_vh)):
                    np.testing.assert_allclose(amp[:, :, 0, i], expected, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(amp[:, :, 1, i], 2 * expected, atol=1e-12, rtol=0)
            bodies = load_body_grim(saved)
            np.testing.assert_allclose(bodies[1.]["theta_deg"], expected_aspects)
            require_body_radar_support(bodies, [1., 2.], azimuths, elevations)
        worker.deleteLater()

    def test_preflight_reports_requested_output_grid(self):
        self.tab.edit_elev_list.setText("0, 90, 180")
        self.tab.edit_bor_elev_list.setText("-30, 30")
        request = self.tab._capture_run_setup()
        from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot
        geometry = Path(__file__).resolve().parents[1] / "geometry/geometries/body.geo"
        snapshot = build_geometry_snapshot(*parse_geometry(geometry.read_text()))
        with mock.patch("ghost_backend.bor.dispatch.estimate_bor_resources",
                        return_value={"estimated_peak_gb": 1., "mesh_elements": 20,
                                      "active_mode_workers": 2}):
            summary = resource_summary(snapshot, "", request)
        self.assertIn("3 azimuths \u00d7 2 elevations", summary)


if __name__ == "__main__":
    unittest.main()
