"""Compare actual CDF/survival path geometry with independent sample counts."""

from __future__ import annotations

from io import BytesIO
import re
import xml.etree.ElementTree as ET

import numpy as np

from test_plot_analysis_features import _WindowCase


def horizontal_height(line, threshold):
    """Read the path, not the input ranks or the Matplotlib drawstyle name."""
    vertices = line.get_path().vertices
    crossings = [a[1] for a, b in zip(vertices[:-1], vertices[1:])
                 if a[0] < threshold < b[0] and a[1] == b[1]]
    if len(crossings) != 1:
        raise AssertionError(f"Expected one horizontal crossing at {threshold}: {crossings}")
    return crossings[0]


class ExceedanceStepTests(_WindowCase):
    def distribution(self, powers, *, exceedance=True, linear=True):
        self.datasets[0].rcs_power[:] = np.asarray(powers)[:, None, None, None]
        self.select_rows(0)
        self.window.combo_plot_scale.setCurrentIndex(
            self.window.combo_plot_scale.findData("linear" if linear else "dbsm")
        )
        self.window.analysis_controls.combo_cdf.setCurrentIndex(int(exceedance))
        self.plot("_plot_cdf")
        return self.lines(self.keys[0])[0]

    def assert_empirical_path(self, line, samples, *, exceedance):
        samples = np.asarray(samples)
        levels = np.unique(samples)
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        np.testing.assert_allclose(x[1:-1], levels)
        # At a discontinuity the line includes a vertical connector. Its stored
        # knot and the appropriate one-sided horizontal limit give inclusivity.
        for index, level in enumerate(levels):
            expected = 100.0 * np.mean(samples >= level if exceedance else samples <= level)
            self.assertAlmostEqual(y[index + 1], expected)
            side = -np.inf if exceedance else np.inf
            self.assertAlmostEqual(horizontal_height(line, np.nextafter(x[index + 1], side)), expected)
        thresholds = [(x[0] + levels[0]) / 2, (levels[-1] + x[-1]) / 2]
        thresholds.extend((levels[:-1] + levels[1:]) / 2)
        for threshold in thresholds:
            expected = 100.0 * np.mean(samples >= threshold if exceedance else samples <= threshold)
            self.assertAlmostEqual(horizontal_height(line, threshold), expected)

    def test_audit_five_distinct_samples_show_eighty_percent_between_first_two(self):
        for linear in (True, False):
            with self.subTest(linear=linear):
                line = self.distribution([1, 2, 3, 4, 5], linear=linear)
                values = np.arange(1, 6) if linear else self.dbsm(np.arange(1, 6))
                self.assertAlmostEqual(horizontal_height(line, (values[0] + values[1]) / 2), 80.0)
                self.assert_empirical_path(line, values, exceedance=True)

    def test_ties_drop_as_one_group_and_exact_levels_are_inclusive(self):
        for exceedance in (True, False):
            for linear in (True, False):
                with self.subTest(exceedance=exceedance, linear=linear):
                    values = np.asarray([1, 1, 2, 2, 5])
                    line = self.distribution(values, exceedance=exceedance, linear=linear)
                    self.assert_empirical_path(
                        line, values if linear else self.dbsm(values), exceedance=exceedance
                    )
                    self.assertEqual(len(line.get_xdata()), 5)  # Three levels and two tails.

    def test_constant_population_draws_the_complete_single_jump(self):
        for exceedance in (True, False):
            with self.subTest(exceedance=exceedance):
                line = self.distribution([2, 2, 2, 2, 2], exceedance=exceedance)
                self.assert_empirical_path(line, [2] * 5, exceedance=exceedance)
                self.assertIn("5 samples", line.get_label())

    def test_cdf_remains_right_continuous_between_distinct_samples(self):
        line = self.distribution([1, 2, 3, 4, 5], exceedance=False)
        self.assert_empirical_path(line, [1, 2, 3, 4, 5], exceedance=False)
        self.assertEqual(horizontal_height(line, 1.5), 20.0)

    def test_existing_missing_and_log_zero_policy_stays_explicit(self):
        line = self.distribution([0, 1, 2, np.nan, np.inf], linear=True)
        self.assert_empirical_path(line, [0, 1, 2], exceedance=True)
        self.assertIn("3 samples", line.get_label())
        line = self.distribution([0, 1, 2, np.nan, np.inf], linear=False)
        self.assert_empirical_path(line, self.dbsm([1, 2]), exceedance=True)
        self.assertIn("2 samples", line.get_label())
        self.assertIn("Zero/nonpositive samples are omitted", self.window.status.currentMessage())

    def test_frequency_dependent_dbke_is_ranked_after_conversion(self):
        dataset = self.datasets[0]
        dataset.units.update(rcs_linear_quantity="width_2d", rcs_log_unit="dBke")
        dataset.rcs_power[:] = np.asarray([1, 1, 2, 2, 5])[:, None, None, None]
        self.select_rows(0, freqs=(0, 7))
        self.window.analysis_controls.combo_cdf.setCurrentIndex(1)
        self.plot("_plot_cdf")
        line, = self.lines(self.keys[0])
        frequencies = dataset.frequencies[[0, 7]] * 1.0e9
        power = np.asarray([1, 1, 2, 2, 5])[:, None]
        expected = 10.0 * np.log10(power * 2.0 * np.pi * frequencies / 299_792_458.0)
        self.assert_empirical_path(line, expected.ravel(), exceedance=True)
        self.assertIn("10 samples", line.get_label())

    def test_overlaid_curves_have_complete_tails_to_shared_display_bounds(self):
        self.select_rows(0, 1, 2)
        self.window.analysis_controls.combo_cdf.setCurrentIndex(1)
        self.plot("_plot_cdf")
        all_levels = np.concatenate([self.lines(key)[0].get_xdata()[1:-1] for key in self.keys])
        pad = 0.02 * (max(all_levels) - min(all_levels))
        bounds = [min(all_levels) - pad, max(all_levels) + pad]
        for key in self.keys:
            line, = self.lines(key)
            np.testing.assert_allclose(line.get_xdata()[[0, -1]], bounds)
            np.testing.assert_allclose(line.get_ydata()[[0, -1]], [100, 0])

    def test_svg_export_contains_the_same_corrected_path_geometry(self):
        line = self.distribution([1, 1, 2, 2, 5])
        self.assertEqual(horizontal_height(line, 1.5), 60.0)
        line.set_gid("verified-exceedance")
        figure = self.window.plot_figure
        original_dpi = figure.dpi
        try:
            figure.set_dpi(72)
            figure.canvas.draw()
            expected = line.get_transform().transform(line.get_path().vertices)
            expected[:, 1] = figure.bbox.height - expected[:, 1]
            stream = BytesIO()
            figure.savefig(stream, format="svg")
        finally:
            figure.set_dpi(original_dpi)
        root = ET.fromstring(stream.getvalue())
        path = root.find(".//*[@id='verified-exceedance']/{http://www.w3.org/2000/svg}path")
        self.assertIsNotNone(path)
        actual = np.asarray([float(value) for value in re.findall(
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.attrib["d"]
        )]).reshape(-1, 2)
        np.testing.assert_allclose(actual, expected, atol=1.0e-5)
