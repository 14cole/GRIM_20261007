"""Display regressions for false white stripes between acquired azimuths."""
import unittest

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from GRIM_Backend.plotting.modes import az_vs_range_mode
from test_reconstruction_accuracy import point_grid, harness


class AzimuthRangeDisplayTests(unittest.TestCase):
    def display(self, angles, *, width=1., max_side=1414):
        image = np.arange(5*len(angles), dtype=float).reshape(5, len(angles))
        edges, y, data, reduced = az_vs_range_mode._range_display_grid(
            angles, np.arange(5.), image, azimuth_width=width, max_side=max_side)
        return edges, y, data, reduced, image

    def test_regular_stride_uses_selected_spacing_without_blank_columns(self):
        edges, _, data, reduced, original = self.display([0., 2., 4., 6.])
        np.testing.assert_allclose(edges, [-1., 1., 3., 5., 7.])
        np.testing.assert_array_equal(data, original)
        self.assertFalse(reduced)

    def test_small_spacing_variations_share_midpoint_edges(self):
        angles = np.array([0., .99, 2.01, 3., 4.02, 5.])
        edges, _, data, reduced, original = self.display(angles, width=.98)
        np.testing.assert_allclose(edges[1:-1], (angles[:-1]+angles[1:])/2)
        np.testing.assert_array_equal(data, original)
        self.assertFalse(reduced)

    def test_float32_angle_rounding_does_not_create_stripes(self):
        angles = np.arange(350., 360., .1).astype(np.float32).astype(float)
        _, _, data, _, original = self.display(angles, width=np.diff(angles).min())
        np.testing.assert_array_equal(data, original)

    def test_change_in_sampling_density_is_contiguous(self):
        angles = np.r_[np.arange(0., 2., .1), np.arange(2., 11., 1.)]
        edges, _, data, _, original = self.display(angles, width=.1)
        np.testing.assert_array_equal(data, original)
        self.assertTrue(np.all(np.diff(edges) > 0))

    def test_large_regular_stride_selection_stays_inside_display_budget(self):
        edges, y, data, reduced, original = self.display(np.arange(0., 6000., 2.), max_side=64)
        self.assertTrue(reduced)
        self.assertLessEqual(max(data.shape), 64)
        self.assertFalse(np.isnan(data).any())
        self.assertEqual(np.max(data), np.max(original))
        np.testing.assert_allclose(edges[[0, -1]], [-1., 5999.])
        np.testing.assert_allclose(y[[0, -1]], [-.5, 4.5])

    def test_real_disconnected_sectors_stay_blank_without_moving_peaks(self):
        angles = [0., 2., 4., 90., 92., 94.]
        edges, _, data, _, original = self.display(angles)
        np.testing.assert_allclose(edges, [-1., 1., 3., 5., 89., 91., 93., 95.])
        self.assertTrue(np.isnan(data[:, 3]).all())
        np.testing.assert_array_equal(data[:, :3], original[:, :3])
        np.testing.assert_array_equal(data[:, 4:], original[:, 3:])

    def test_radian_axes_have_the_same_display_support(self):
        angles = np.array([0., 2., 4., 90., 92., 94.])
        deg = self.display(angles)
        rad = self.display(np.deg2rad(angles), width=np.deg2rad(1.))
        np.testing.assert_allclose(rad[0], np.deg2rad(deg[0]))
        np.testing.assert_allclose(rad[2], deg[2], equal_nan=True)

    def test_missing_range_profile_is_still_unknown(self):
        image = np.ones((5, 4))
        image[:, 2] = np.nan
        _, _, data, _ = az_vs_range_mode._range_display_grid([0., 2., 4., 6.], np.arange(5.),
            image, azimuth_width=1., max_side=1414)
        self.assertTrue(np.isnan(data[:, 2]).all())
        self.assertTrue(np.isfinite(data[:, [0, 1, 3]]).all())

    def test_actual_renderer_joins_stride_selection_and_retains_peak_coordinates(self):
        dataset = point_grid(np.arange(61.), np.linspace(8e9, 12e9, 129))
        # Brightest acquired return at 20 degrees.
        dataset.rcs_power[20] *= 100
        owner = harness(dataset)
        previous = owner._selected_indices
        owner._selected_indices = lambda widget: set(range(0, 61, 2)) if widget is owner.list_az else previous(widget)
        az_vs_range_mode.render(owner)
        self.assertIn('updated', owner.status.message)
        image = owner.plot_ax.images[0]
        values = image.get_array()
        self.assertFalse(np.ma.getmaskarray(values).any())
        row, column = np.unravel_index(np.argmax(values), values.shape)
        left, right, bottom, top = image.get_extent()
        self.assertAlmostEqual(left+(column+.5)*(right-left)/values.shape[1], 20.)
        self.assertAlmostEqual(bottom+(row+.5)*(top-bottom)/values.shape[0], 0.)
        FigureCanvasAgg(owner.plot_figure).draw()
        owner.plot_figure.clear()


if __name__ == '__main__':
    unittest.main()
