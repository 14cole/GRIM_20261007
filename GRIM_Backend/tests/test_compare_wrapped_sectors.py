"""RF Compare supports real matched samples across either angular seam."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import compare_mode
from test_plot_renderer_correctness import _RendererHarness, _Checked, _Spin, _Visible


def _sweep(angles, *, unit="deg"):
    angles = np.asarray(angles, dtype=float)
    native = np.deg2rad(angles) if unit == "rad" else angles
    power = 2.0 + np.cos(np.deg2rad(angles))
    return RcsGrid(
        native, [0.0], [9.0], ["HH"],
        rcs_power=power[:, None, None, None],
        rcs_phase=np.zeros((angles.size, 1, 1, 1)),
        units={"azimuth": unit, "elevation": unit, "frequency": "GHz"},
    )


def _harness(left, right, *, start, end, show_all=False, phase=False):
    owner = _RendererHarness(
        [("analysis", left), ("test", right)],
        selections={"azimuth": left.azimuths, "elevation": [0.0], "frequency": [9.0]},
        phase=phase,
    )
    owner.compare_sector_bar = _Visible()
    owner.spin_compare_az_min = _Spin()
    owner.spin_compare_az_max = _Spin()
    owner.chk_compare_show_all_azimuths = _Checked(False)
    # Initialize selection-dependent controls before applying the user's bounds.
    compare_mode._comparison_azimuth_sector(owner, left, left.azimuths)
    owner.spin_compare_az_min.setValue(start)
    owner.spin_compare_az_max.setValue(end)
    owner.chk_compare_show_all_azimuths.setChecked(show_all)
    return owner


class CompareWrappedSectorTests(unittest.TestCase):
    def test_curves_default_to_solid_and_respect_explicit_styles(self):
        angles = np.arange(-180.0, 181.0, 5.0)
        left, right = _sweep(angles), _sweep(angles)
        owner = _harness(left, right, start=-180.0, end=180.0)
        self.addCleanup(owner.plot_figure.clear)
        compare_mode.render(owner)
        self.assertEqual([line.get_linestyle() for line in owner.plot_figure.axes[0].lines], ["-", "-"])
        owner._dataset_plot_styles = {owner._dataset_plot_key(right): {"linestyle": "--"}}
        compare_mode.render(owner)
        self.assertEqual([line.get_linestyle() for line in owner.plot_figure.axes[0].lines], ["-", "--"])

    def test_signed_seam_sector_is_contiguous_and_counts_endpoint_once(self):
        angles = np.arange(-180.0, 181.0, 5.0)
        left, right = _sweep(angles), _sweep(angles)
        left.rcs_power[:, 0, 0, 0] = 1.0
        right.rcs_power[:, 0, 0, 0] = 1.0
        left.rcs_power[angles == 175.0, 0, 0, 0] = 9.0
        right.rcs_power[angles == -175.0, 0, 0, 0] = 9.0
        owner = _harness(left, right, start=170.0, end=-170.0)
        with mock.patch.object(compare_mode, "rf_agreement_statistics", wraps=compare_mode.rf_agreement_statistics) as stats:
            compare_mode.render(owner)
        self.assertIn("updated", owner.status.message)
        top, residual = owner.plot_figure.axes
        expected = [170.0, 175.0, 180.0, 185.0, 190.0]
        np.testing.assert_allclose(top.lines[0].get_xdata(), expected)
        np.testing.assert_allclose(residual.lines[-1].get_xdata(), expected)
        np.testing.assert_allclose(stats.call_args.args[0], expected)
        self.assertIn("Peak shift: -10 deg", top.get_title())
        self.assertIn("Statistics range: 170 to -170 deg", top.get_title())
        self.assertIn("continuous across seam", residual.get_xlabel())

    def test_show_all_keeps_native_sweep_and_highlights_both_sector_pieces(self):
        angles = np.arange(-180.0, 181.0, 5.0)
        left, right = _sweep(angles), _sweep(angles)
        right.rcs_power[np.abs(angles) < 170.0, 0, 0, 0] *= 100.0
        owner = _harness(left, right, start=170.0, end=-170.0, show_all=True)
        with mock.patch.object(compare_mode, "rf_agreement_statistics", wraps=compare_mode.rf_agreement_statistics) as stats:
            compare_mode.render(owner)
        top, residual = owner.plot_figure.axes
        np.testing.assert_allclose(top.lines[0].get_xdata(), angles)
        np.testing.assert_allclose(stats.call_args.args[0], [170, 175, 180, 185, 190])
        self.assertIn("Overall match: 100.0/100", top.get_title())
        self.assertEqual([(p.get_x(), p.get_width()) for p in residual.patches], [(170.0, 10.0), (-180.0, 10.0)])

    def test_zero_seam_wrap_uses_measured_values_without_interpolation(self):
        angles = np.arange(0.0, 361.0, 5.0)
        left, right = _sweep(angles), _sweep(angles)
        owner = _harness(left, right, start=350.0, end=10.0)
        with mock.patch.object(compare_mode, "rf_agreement_statistics", wraps=compare_mode.rf_agreement_statistics) as stats:
            compare_mode.render(owner)
        np.testing.assert_allclose(stats.call_args.args[0], [350, 355, 360, 365, 370])
        expected_indices = [70, 71, 0, 1, 2]
        np.testing.assert_allclose(stats.call_args.args[1], left.rcs_power[expected_indices, 0, 0, 0])
        self.assertIn("Overall match: 100.0/100", owner.plot_figure.axes[0].get_title())

    def test_radian_reference_with_degree_dataset_uses_radian_period(self):
        angles = np.arange(-180.0, 181.0, 5.0)
        left, right = _sweep(angles, unit="rad"), _sweep(angles)
        owner = _harness(left, right, start=np.deg2rad(170.0), end=np.deg2rad(-170.0))
        with mock.patch.object(compare_mode, "rf_agreement_statistics", wraps=compare_mode.rf_agreement_statistics) as stats:
            compare_mode.render(owner)
        expected = np.deg2rad([170, 175, 180, 185, 190])
        np.testing.assert_allclose(stats.call_args.args[0], expected)
        for line in owner.plot_figure.axes[0].lines:
            np.testing.assert_allclose(line.get_xdata(), expected)
        self.assertIn("Azimuth (rad)", owner.plot_figure.axes[1].get_xlabel())

    def test_sparse_sector_keeps_only_existing_common_coordinates(self):
        left = _sweep([-179, -175, -170, 0, 170, 175, 179])
        right = _sweep([-179, -170, 0, 170, 175, 179])
        owner = _harness(left, right, start=170, end=-170)
        with mock.patch.object(compare_mode, "rf_agreement_statistics", wraps=compare_mode.rf_agreement_statistics) as stats:
            compare_mode.render(owner)
        np.testing.assert_allclose(stats.call_args.args[0], [170, 175, 179, 181, 190])
        self.assertEqual(len(owner.plot_figure.axes[0].lines[0].get_xdata()), 5)

    def test_phase_sector_uses_the_same_contiguous_order(self):
        angles = np.arange(-180.0, 181.0, 5.0)
        left, right = _sweep(angles), _sweep(angles)
        left.rcs_phase[:, 0, 0, 0] = np.deg2rad(179.0)
        right.rcs_phase[:, 0, 0, 0] = np.deg2rad(-179.0)
        owner = _harness(left, right, start=170, end=-170, phase=True)
        with mock.patch.object(compare_mode, "phase_agreement_statistics", wraps=compare_mode.phase_agreement_statistics) as stats:
            compare_mode.render(owner)
        np.testing.assert_allclose(stats.call_args.args[0], [170, 175, 180, 185, 190])
        np.testing.assert_allclose(owner.plot_figure.axes[1].lines[-1].get_ydata(), -2.0)

    def test_seam_endpoint_duplicates_alone_do_not_make_two_samples(self):
        left, right = _sweep([-180.0, 0.0, 180.0]), _sweep([-180.0, 0.0, 180.0])
        owner = _harness(left, right, start=179.0, end=-179.0)
        compare_mode.render(owner)
        self.assertIn("fewer than 2 distinct matched samples", owner.status.message)


if __name__ == "__main__":
    unittest.main()
