"""Numerical and renderer regressions for the September 2026 plotting audit."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
import unittest
import warnings
from unittest import mock

import numpy as np
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import (
    common, compare_mode, azimuth_rect_mode, azimuth_polar_mode,
    frequency_mode, elevation_sweep_mode, waterfall_mode,
)
from GRIM_Backend.scripting.plotting import _display_values
from test_plot_renderer_correctness import _RendererHarness


def grid(az=(0., 1., 2.), el=(0.,), fr=(10.,), *, power=None, phase=None, units=None, extra=None):
    shape = (len(az), len(el), len(fr), 1)
    return RcsGrid(az, el, fr, ["HH"], rcs_power=np.ones(shape) if power is None else power,
                   rcs_phase=phase, units=units, extra=extra)


def harness(dataset, *, phase=False, others=()):
    return _RendererHarness([("reference", dataset), *others], phase=phase, selections={
        "azimuth": dataset.azimuths, "elevation": dataset.elevations,
        "frequency": dataset.frequencies,
    })


class NumericalAuditTests(unittest.TestCase):
    def test_rf_score_is_invariant_to_common_scale_including_sectors(self):
        x = np.arange(1., 25.)
        expected = compare_mode.rf_agreement_statistics(x, x, x * 1.01)
        for scale in (1.e-300, 1.e-150, 1.e-10, 1., 1.e100, 1.e300):
            with self.subTest(scale=scale):
                actual = compare_mode.rf_agreement_statistics(x, x * scale, x * scale * 1.01)
                self.assertAlmostEqual(actual.score, expected.score, places=8)
                self.assertAlmostEqual(actual.mae_db, expected.mae_db, places=10)
                np.testing.assert_allclose([s.score for s in actual.sectors],
                                           [s.score for s in expected.sectors], atol=1.e-8)

    def test_rf_zeros_and_constants_have_explicit_behavior(self):
        for scale in (1.e-300, 1., 1.e300):
            self.assertEqual(compare_mode._lin_concordance(np.ones(5)*scale, np.ones(5)*scale), 1.)
            self.assertEqual(compare_mode._lin_concordance(np.ones(5)*scale, np.ones(5)*scale*2), 0.)
        self.assertEqual(compare_mode._lin_concordance(np.zeros(5), np.zeros(5)), 1.)
        self.assertEqual(compare_mode._linear_nrmse(np.zeros(5), np.zeros(5)), 0.)
        with self.assertRaisesRegex(ValueError, "positive"):
            compare_mode.rf_agreement_statistics([0, 1], [0, 0], [0, 0])
        positive_only = compare_mode.rf_agreement_statistics([1, 2], [1.e-200, 1.], [1.e-210, 1.])
        including_zero = compare_mode.rf_agreement_statistics([0, 1, 2], [0, 1.e-200, 1.], [0, 1.e-210, 1.])
        self.assertEqual(positive_only, including_zero)
        self.assertAlmostEqual(positive_only.mae_db, 50.)

    def test_unfloored_dbke_handles_subnormal_width_without_product_underflow(self):
        dataset = grid(units={"frequency": "Hz", "rcs_log_unit": "dBke", "rcs_linear_quantity": "sigma_2d"})
        width = np.nextafter(0., 1.)
        result = dataset.linear_to_dbke(width, 1., eps=0.)
        expected = 10.*(np.log10(width) + np.log10(2*np.pi/299792458.))
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(float(result), expected)

    def test_circular_median_minimizes_geodesic_loss_and_is_order_independent(self):
        values = np.array([-180., -160., -140., -60., 40.])
        self.assertAlmostEqual(float(common.circular_median_degrees(values)), -160.)
        rng = np.random.default_rng(23091)
        for n in range(1, 20):
            values = rng.uniform(-180., 180., n)
            actual = float(common.circular_median_degrees(values))
            cost = lambda angle: np.abs((values-angle+180.) % 360.-180.).sum()
            self.assertLessEqual(cost(actual), min(cost(value) for value in values) + 1.e-8)
            self.assertAlmostEqual(actual, float(common.circular_median_degrees(values[::-1])))
        for values in ([179., -179.], [90., -90.], [0., 90., 180., 270.]):
            answers = [float(common.circular_median_degrees(order)) for order in itertools.permutations(values)]
            np.testing.assert_allclose(answers, answers[0])
        self.assertTrue(np.isnan(common.circular_median_degrees([np.nan, np.nan])))

    def test_phase_envelope_is_shortest_covering_arc_with_bounded_blocks(self):
        expected = None
        with mock.patch.object(common, "REDUCTION_BLOCK_CELLS", 4):
            for order in itertools.permutations([-170., -60., 60., 170.]):
                envelope = common.StreamingEnvelope(phase_degrees=True)
                try:
                    for value in order:
                        envelope.update([value, np.nan, 179. if value < 0 else -179.])
                    low, high, count = envelope.result()
                    np.testing.assert_allclose(high-low, [240., np.nan, 2.], equal_nan=True)
                    np.testing.assert_array_equal(count, [4, 0, 4])
                    if expected is None:
                        expected = low.copy(), high.copy()
                    np.testing.assert_allclose(low, expected[0], equal_nan=True)
                    np.testing.assert_allclose(high, expected[1], equal_nan=True)
                    # Reading a result does not end the streaming accumulator.
                    envelope.update([0., np.nan, 180.])
                    self.assertEqual(envelope.result()[2][0], 5)
                finally:
                    envelope.close()

    def test_line_vectorization_matches_independent_bucket_oracle(self):
        rng = np.random.default_rng(8127)
        for n, budget, scratch in ((101, 8, 13), (3601, 2000, 262144), (10007, 17, 79)):
            y = rng.normal(size=n)
            y[::19] = np.nan
            y[20:30] = 7.
            x = np.arange(n)
            selected = {0, n-1}
            boundaries = np.linspace(1, n-1, (budget-2)//3+1, dtype=int)
            for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                valid = [i for i in range(lo, hi) if np.isfinite(y[i])]
                missing = [i for i in range(lo, hi) if not np.isfinite(y[i])]
                if valid:
                    selected.update((min(valid, key=lambda i: y[i]), max(valid, key=lambda i: y[i])))
                if missing:
                    selected.add(missing[0])
            with mock.patch.object(common, "REDUCTION_BLOCK_CELLS", scratch):
                actual_x, actual_y, changed = common.decimate_line(x, y, budget)
            self.assertTrue(changed)
            np.testing.assert_array_equal(actual_x, sorted(selected))
            np.testing.assert_array_equal(actual_y, y[sorted(selected)])
        with mock.patch.object(common, "MAX_LINE_POINTS", 7):
            self.assertLessEqual(len(common.decimate_line(x, y)[0]), 7)
            self.assertLessEqual(len(common.decimate_envelope(x, y, y)[0]), 7)
        integers = np.full(20, 2**60, dtype=np.int64)
        integers[7] += 1
        self.assertIn(7, common.decimate_line(np.arange(20), integers, 8)[0])

    def test_magnitude_only_interpolation_skips_complex_but_keeps_raw_authority(self):
        power = np.array([0., 1., np.nan, 9.])[:, None, None, None]
        dataset = grid(az=[0., 1., 2., 3.], power=power)
        with mock.patch.object(dataset, "_interp_complex_axis", side_effect=AssertionError("unneeded complex interpolation")):
            output = dataset.interpolate_axis("azimuth", [0., .5, 1., 1.5, 2., 3.])
        np.testing.assert_allclose(output.rcs_power[:, 0, 0, 0], [0., .5, 1., np.nan, np.nan, 9.], equal_nan=True)
        self.assertTrue(np.all(np.isnan(output.rcs_phase)))
        # Raw solver amplitudes override missing phase and the nominal powers.
        raw = np.array([1., 3.])[:, None, None, None]
        raw_grid = grid(az=[0., 1.], extra={"rcs_amp_real": raw, "rcs_amp_imag": np.zeros_like(raw)})
        output = raw_grid.interpolate_axis("azimuth", [.5])
        self.assertAlmostEqual(output.rcs_power.item(), 4.*4.*np.pi)
        self.assertAlmostEqual(output.rcs_phase.item(), 0.)
        phased = grid(az=[0., 1.], phase=np.array([0., np.pi])[:, None, None, None])
        output = phased.interpolate_axis("azimuth", [.5])
        self.assertLess(output.rcs_power.item(), 1.e-30)


class RendererAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_positive_log_values_match_compare_statistics_and_python_replay(self):
        for log_unit, quantity in (("dBsm", "sigma_3d"), ("dBke", "sigma_2d"), ("dB", "power_ratio")):
            units = {"rcs_log_unit": log_unit, "rcs_linear_quantity": quantity}
            a = grid(power=np.full((3, 1, 1, 1), 1.e-200), units=units)
            b = grid(power=np.full((3, 1, 1, 1), 1.e-210), units=units)
            h = harness(a, others=[("other", b)])
            compare_mode.render(h)
            upper, residual = h.plot_figure.axes[:2]
            np.testing.assert_allclose(upper.lines[0].get_ydata()-upper.lines[1].get_ydata(), 100., atol=1.e-10)
            np.testing.assert_allclose(residual.lines[1].get_ydata(), 100., atol=1.e-10)
            self.assertEqual(residual.get_ylabel(), "Difference (dB)")
            self.assertIn("100.00 dB", upper.get_title())
            values = _display_values(a, np.array([1.e-200, 0., np.nan]), frequency=10., phase=False, scale="dbsm")
            self.assertTrue(np.isfinite(values[0]))
            self.assertTrue(np.all(np.isnan(values[1:])))
            h.plot_figure.clear()
        zero = grid(power=np.zeros((3, 1, 1, 1)))
        h = harness(zero)
        azimuth_rect_mode.render(h)
        self.assertEqual(len(h.plot_ax.lines), 0)
        self.assertIn("Zero/nonpositive", h.status.message)

    def test_hold_uses_exact_coordinates_and_aggregation_selection(self):
        a = grid(fr=[10., 10.000001])
        for mode in (azimuth_rect_mode, azimuth_polar_mode, elevation_sweep_mode):
            h = harness(a)
            h.btn_hold.checked = True
            mode.render(h)
            self.assertEqual(len(h.plot_ax.lines), 2)
            mode.render(h)
            self.assertEqual(len(h.plot_ax.lines), 2)
            self.assertNotEqual(h.plot_ax.lines[0]._grim_trace_key, h.plot_ax.lines[1]._grim_trace_key)
        a = grid(el=[0., .0000001], fr=[9., 10.])
        h = harness(a)
        h.btn_hold.checked = True
        frequency_mode.render(h)
        self.assertEqual(len(h.plot_ax.lines), 2)
        h._selections[h.list_az] = [0., 2.]
        frequency_mode.render(h)
        self.assertEqual(len(h.plot_ax.lines), 4)  # Same range, different P50 contributors.

    def test_ambiguous_pbp_is_controlled_and_preserves_existing_canvas(self):
        a = grid(fr=[10., 10.0000005, 10.000001])
        b = grid(fr=[10., 10.000001])
        h = harness(a, others=[("other", b)])
        old_line = h.plot_ax.plot([0, 1], [2, 3])[0]
        h.btn_pbp.checked = True
        frequency_mode.render(h)
        self.assertEqual(list(h.plot_ax.lines), [old_line])
        self.assertIn("one-to-one", h.status.message)
        self.assertIn("regrid", h.status.message)
        np.testing.assert_array_equal(common.unique_axis_selection(a.frequencies, a.frequencies, 1.e-6), [0, 1, 2])

    def test_real_spinboxes_preserve_hz_bounds_fit_and_singletons(self):
        a = grid(fr=[8.e9, 9.e9, 11.e9], units={"frequency": "Hz"})
        for mode in (frequency_mode, waterfall_mode):
            h = harness(a)
            widgets = []
            for name in ("xmin", "xmax", "ymin", "ymax"):
                spin = QDoubleSpinBox()
                spin.setRange(-1.e9, 1.e9)  # Deliberately reproduce the old GUI limit.
                spin.setDecimals(6)
                spin.setValue(-180. if name.endswith("min") else 180.)
                setattr(h, "spin_plot_"+name, spin)
                widgets.append(spin)
            mode.render(h)
            if mode is frequency_mode:
                self.assertEqual(h.plot_ax.get_xlim(), (8.e9, 11.e9))
            else:
                self.assertEqual(h.plot_ax.get_ylim(), (8.e9, 11.e9))
            h._fit_both()
            limits = h.plot_ax.get_xlim() if mode is frequency_mode else h.plot_ax.get_ylim()
            self.assertLessEqual(limits[0], 8.e9)
            self.assertGreaterEqual(limits[1], 11.e9)
            common.set_spin_value(h.spin_plot_xmin, 1.e-15)
            self.assertEqual(h.spin_plot_xmin.value(), 1.e-15)
            h.spin_plot_xmin.setValue(10.)
            h.spin_plot_xmax.setValue(10.)
            h.spin_plot_ymin.setValue(3.)
            h.spin_plot_ymax.setValue(3.)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                h._apply_plot_limits()
            self.assertFalse(any("identical" in str(w.message) for w in caught))
            self.assertLess(h.spin_plot_xmin.value(), h.spin_plot_xmax.value())
            self.assertEqual(h.plot_ax.get_xlim(), (h.spin_plot_xmin.value(), h.spin_plot_xmax.value()))

    def test_cap_stops_computation_after_visible_curves_but_not_pbp(self):
        a = grid(fr=np.arange(1., 16.))
        a.rcs_power[:, :, :2, :] = np.nan
        h = harness(a)
        original = h._display_from_values
        with mock.patch.object(common, "MAX_LINE_SERIES", 3), mock.patch.object(h, "_display_from_values", wraps=original) as convert:
            azimuth_rect_mode.render(h)
            self.assertEqual(convert.call_count, 5)  # Skip two empty cuts, then retain three.
            self.assertEqual(len(h.plot_ax.lines), 3)
            self.assertIn("10 further candidate", h.status.message)
        h = harness(a)
        h.btn_pbp.checked = True
        original = h._display_from_values
        with mock.patch.object(common, "MAX_LINE_SERIES", 3), mock.patch.object(h, "_display_from_values", wraps=original) as convert:
            azimuth_rect_mode.render(h)
            self.assertEqual(convert.call_count, 15)

    def test_display_budget_preserves_data_peak_and_runtime_configuration(self):
        a = grid(az=np.arange(10001.), fr=np.arange(1., 5.))
        a.rcs_power[5432, :, :, :] = 1.e6
        before = a.rcs_power.copy()
        h = harness(a)
        with mock.patch.object(common, "TOTAL_LINE_POINT_TARGET", 8000):
            azimuth_rect_mode.render(h)
        self.assertLessEqual(sum(len(line.get_xdata()) for line in h.plot_ax.lines), 8000)
        for line in h.plot_ax.lines:
            self.assertIn(5432., line.get_xdata())
            self.assertEqual(max(line.get_ydata()), 60.)
        np.testing.assert_array_equal(a.rcs_power, before)
        self.assertIn("figure exports", h.status.message)

    def test_irregular_waterfall_hover_uses_physical_cells_without_path_lookup(self):
        a = grid(az=[0., 1., 10.], fr=[8., 9., 11.], power=np.arange(1., 10.).reshape(3, 1, 3, 1))
        h = harness(a)
        waterfall_mode.render(h)
        mesh = h.plot_ax.collections[0]
        np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], [-.5, .5, 5.5, 14.5])
        with mock.patch.object(mesh, "get_cursor_data", side_effect=AssertionError("slow path")):
            value = h._hover_z_from_axes(h.plot_ax, SimpleNamespace(xdata=1., ydata=9.))
            self.assertAlmostEqual(value, 10*np.log10(5.))
            self.assertIsNone(h._hover_z_from_axes(h.plot_ax, SimpleNamespace(xdata=100., ydata=9.)))
        a.rcs_power[1, 0, 1, 0] = np.nan
        waterfall_mode.render(h)
        with mock.patch.object(h.plot_ax.collections[0], "get_cursor_data", side_effect=AssertionError("slow path")):
            self.assertIsNone(h._hover_z_from_axes(h.plot_ax, SimpleNamespace(xdata=1., ydata=9.)))
        self.assertIsNone(getattr(mesh, "_paths", None))


if __name__ == "__main__":
    unittest.main()
