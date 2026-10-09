"""Independent point-target regressions for ISAR and azimuth/range imaging."""
from types import SimpleNamespace
import unittest

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.isar.geometry import draw_image, reduced_axis_edges
from GRIM_Backend.isar.interpolation import resample_pair
from GRIM_Backend.isar.quality import reconstruction_advisories
from GRIM_Backend.plotting.modes import isar_mode, isar_render, az_vs_range_mode
from test_plot_renderer_correctness import _RendererHarness

C0 = 299792458.0


def point_grid(azimuth, frequency, x=0., y=0.):
    a, f = np.asarray(azimuth), np.asarray(frequency)
    # Independent far-field monostatic phase law; do not use the imaging operator.
    field = np.exp(-4j*np.pi/C0 * f[None, :] *
                   (x*np.sin(np.deg2rad(a[:, None])) + y*np.cos(np.deg2rad(a[:, None]))))
    return RcsGrid(a, [0.], f, ['VV'], rcs=field[:, None, :, None],
        units={'frequency': 'Hz', 'azimuth': 'deg', 'elevation': 'deg', 'time_convention': 'exp(+jwt)'},
        extra={'phase_reference': 'fixed origin', 'measurement_geometry': 'far-field monostatic',
               'motion_compensation': 'stable', 'range_phase_convention': 'S~exp(-j*2*k*R)'})


def harness(dataset):
    owner = _RendererHarness([('point', dataset)], selections=dict(azimuth=dataset.azimuths,
        elevation=dataset.elevations, frequency=dataset.frequencies, polarization=dataset.polarizations))
    owner._selected_indices = lambda widget: set(range(len(owner._selected_values(widget))))
    owner._isar_window = np.hanning
    owner.combo_isar_units = SimpleNamespace(currentText=lambda: 'm')
    owner.chk_isar_square = SimpleNamespace(isChecked=lambda: False)
    return owner


class ReconstructionAccuracyTests(unittest.TestCase):
    def test_accurate_off_origin_on_grid_point_preserves_amplitude_and_position(self):
        az, freq = np.linspace(-10, 10, 401), np.linspace(8e9, 12e9, 401)
        origin = isar_mode.form_isar(point_grid(az, freq), reconstruction='accurate')[0][0]
        x = origin['x_range'][np.argmin(abs(origin['x_range']-5.))]
        y = origin['y_range'][np.argmin(abs(origin['y_range']-5.))]
        band = isar_mode.form_isar(point_grid(az, freq, x, y), reconstruction='accurate')[0][0]
        index = np.unravel_index(np.argmax(band['magnitude']), band['magnitude'].shape)
        self.assertAlmostEqual(band['x_range'][index[0]], x)
        self.assertAlmostEqual(band['y_range'][index[1]], y)
        self.assertAlmostEqual(float(band['magnitude'][index]), 1., delta=.002)

    def test_uneven_frequency_processing_preserves_off_origin_point(self):
        frequency = np.linspace(8e9, 12e9, 401) + np.sin(np.arange(401))*2e6
        distance = 134*C0/(2*401*1e7)
        data = np.exp(-4j*np.pi/C0*frequency*distance)[None, :]
        target, history, weights, _ = az_vs_range_mode._prepare_uniform_frequency_history(frequency, data)
        image, usable = az_vs_range_mode._form_range_image(history, weights, np.hanning(len(target)))
        axis = np.fft.fftshift(np.fft.fftfreq(len(target), np.mean(np.diff(target))))*C0/2
        peak = int(np.argmax(abs(image[0])))
        self.assertTrue(usable[0])
        self.assertGreater(abs(image[0, peak]), .98)  # Formerly 0.791.
        self.assertLess(abs(axis[peak]-distance), abs(axis[1]-axis[0])/2)

    def test_missing_phase_stencils_do_not_spread_as_known_observations(self):
        source = np.arange(40, dtype=float)
        target = np.arange(.5, 39., 1.)
        weights = np.ones((40, 2), np.float32)
        weights[20, 0] = 0
        field = weights.astype(np.complex64)
        out, coverage = resample_pair(source, field, weights, target, axis=0, support=np.ones(39, bool))
        np.testing.assert_allclose(out, coverage, atol=2e-7)
        self.assertEqual(coverage[19, 0], .5)
        self.assertEqual(coverage[20, 0], .5)
        self.assertTrue(np.all(coverage[:, 1] == 1))

    def test_nonuniform_gap_stencils_never_join_separate_bands(self):
        source = np.r_[np.arange(20.), np.arange(100., 120.)]
        target = np.arange(.5, 119., 1.)
        support, _ = isar_mode._interpolation_support(source, target)
        field = np.r_[np.ones(20), np.full(20, 10.)].astype(np.complex64)[:, None]
        out, coverage = resample_pair(source, field, np.ones(field.shape, np.float32), target, axis=0, support=support)
        np.testing.assert_allclose(out[target < 19], 1)
        np.testing.assert_allclose(out[target > 100], 10)
        np.testing.assert_array_equal(out[~support], 0)
        np.testing.assert_array_equal(coverage[~support], 0)

    def test_narrow_band_auto_forms_point_and_reports_support_tradeoff(self):
        az, freq = np.linspace(-5, 5, 129), np.linspace(10e9, 10.01e9, 129)
        band = isar_mode.form_isar(point_grid(az, freq), reconstruction='auto')[0][0]
        self.assertTrue(band['accurate_pfa'])
        self.assertAlmostEqual(float(band['magnitude'].max()), 1., places=6)
        self.assertGreater(band['sampling']['range_resolution'], 0)
        self.assertLess(band['accuracy_plan']['cartesian_support_scale'], 1)
        self.assertIn('resolution is reduced', ' '.join(reconstruction_advisories([band])))

    def test_range_processing_precision_matches_double_reference(self):
        rng = np.random.default_rng(42)
        data = (rng.normal(size=(4, 1601))+1j*rng.normal(size=(4, 1601))).astype(np.complex64)
        weights = np.ones(data.shape, np.float32)
        weights[:, 25:30] = 0
        weights[-1] = 0
        taper = np.hanning(data.shape[1])
        expected = np.fft.fftshift(np.fft.ifft(data*taper[None, :]*weights, axis=1), axes=1)
        expected[:-1] /= (np.sum(weights*taper[None, :], axis=1)/data.shape[1])[:-1, None]
        actual, usable = az_vs_range_mode._form_range_image(data, weights, taper)
        self.assertEqual(actual.dtype, np.complex64)
        self.assertTrue(np.isnan(actual[-1]).all())
        self.assertFalse(usable[-1])
        self.assertLess(np.linalg.norm(actual[:-1]-expected[:-1])/np.linalg.norm(expected[:-1]), 1e-6)

    def test_partial_display_cells_keep_true_coordinates(self):
        x, y = np.arange(1617.), np.arange(9.)
        magnitude = np.zeros((len(x), len(y)), np.float32)
        magnitude[-1, 4] = 1.
        reduced = isar_mode._decimate_display_max(magnitude, 1414)
        band = dict(x_range=x, y_range=y, display_x_edges=reduced_axis_edges(x, 1414),
                    display_y_edges=reduced_axis_edges(y, 1414))
        figure = Figure()
        mesh = draw_image(figure.add_subplot(), band, reduced)
        coordinates = mesh.get_coordinates()
        np.testing.assert_allclose(coordinates[0, -2:, 0], [1615.5, 1616.5])
        np.testing.assert_allclose(coordinates[:, 0, 1], np.arange(-.5, 9))
        self.assertEqual(float(mesh.get_array().max()), 1.)

    def test_formation_passes_partial_block_edges_through_actual_renderer(self):
        dataset = point_grid(np.linspace(-2, 2, 17), np.linspace(8e9, 12e9, 1801))
        bands, elapsed = isar_mode.form_isar(dataset, reconstruction='fast', decimate_display=True)
        band = bands[0]
        self.assertTrue(band['display_decimated'])
        owner = harness(dataset)
        isar_render.display_results(owner, dict(dataset=dataset, unit_name='m', az_target_deg=None,
            elevation_deg=0, pol_idx=0, recon='fft'), bands, elapsed)
        mesh = owner.plot_ax.collections[0]
        y = band['y_range']
        dy = y[1]-y[0]
        np.testing.assert_allclose(mesh.get_coordinates()[-2:, 0, 1], [y[-1]-dy/2, y[-1]+dy/2])
        FigureCanvasAgg(owner.plot_figure).draw()
        owner.plot_figure.clear()

    def test_acquired_data_warning_visible_even_when_sparse_solver_converged(self):
        dataset = point_grid(np.linspace(-2, 2, 129), np.linspace(8e9, 10e9, 129))
        bands, elapsed = isar_mode.form_isar(dataset, reconstruction='sparse')
        bands[0]['native_residual'].update(high_model_mismatch=True, relative_complex_l2_residual=1.268)
        owner = harness(dataset)
        isar_render.display_results(owner, dict(dataset=dataset, unit_name='m', az_target_deg=None,
            elevation_deg=0, pol_idx=0, recon='sparse'), bands, elapsed)
        self.assertIn('ACCURACY WARNING', owner.status.message[:100])
        self.assertIn('126.8%', owner.status.message)
        self.assertIn('converged', owner.status.message)
        owner.plot_figure.clear()

    def test_composite_surfaces_worst_sublook_accuracy(self):
        band = {'resolved_reconstruction': 'sparse', 'composite_look_diagnostics': [
            {'native_residual': {'status': 'computed', 'high_model_mismatch': True,
             'relative_complex_l2_residual': value, 'sample_count': 100, 'source_sample_count': 1000}}
            for value in [.5, 1.2, .8]]}
        notes = reconstruction_advisories([band])
        self.assertEqual(len(notes), 1)
        self.assertIn('120.0%', notes[0])

    def test_azimuth_range_applies_band_and_wrapped_aperture(self):
        dataset = point_grid([0., 1., 180., 359.], np.linspace(8e9, 12e9, 401))
        owner = harness(dataset)
        owner.chk_isar_freq_band = SimpleNamespace(isChecked=lambda: True)
        owner.spin_isar_freq_min = SimpleNamespace(value=lambda: 9e9)
        owner.spin_isar_freq_max = SimpleNamespace(value=lambda: 11e9)
        owner.chk_isar_aperture = SimpleNamespace(isChecked=lambda: True)
        owner.spin_isar_ap_center = SimpleNamespace(value=lambda: 0.)
        owner.spin_isar_ap_width = SimpleNamespace(value=lambda: 4.)
        az_vs_range_mode.render(owner)
        mesh = owner.plot_ax.collections[0]
        self.assertEqual(mesh.get_array().shape[0], 201)
        self.assertTrue(np.ma.getmaskarray(mesh.get_array())[:, 2].all())
        np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], [-.5, .5, 1.5, 358.5, 359.5])
        owner.plot_figure.clear()

    def test_invalid_frequency_band_stops_before_forming_an_image(self):
        owner = harness(point_grid([0., 1.], np.linspace(8e9, 12e9, 20)))
        owner.chk_isar_freq_band = SimpleNamespace(isChecked=lambda: True)
        owner.spin_isar_freq_min = SimpleNamespace(value=lambda: 12e9)
        owner.spin_isar_freq_max = SimpleNamespace(value=lambda: 9e9)
        az_vs_range_mode.render(owner)
        self.assertIn('max must exceed min', owner.status.message)
        self.assertFalse(owner.plot_ax.images)

    def test_sparse_composite_axis_prediction_matches_actual_cartesian_formation(self):
        dataset = point_grid(np.linspace(-5, 5, 65), np.linspace(10e9, 10.01e9, 65))
        predicted = isar_mode._predict_sublook_scene_axes(dataset, list(range(65)), dataset.frequencies,
            reconstruction='sparse', unit_scale=1., elevation_deg=0., az_center_deg=None)
        band = isar_mode.form_isar(dataset, reconstruction='sparse', l1_iterations=20)[0][0]
        np.testing.assert_allclose(predicted[0], band['x_range'])
        np.testing.assert_allclose(predicted[1], band['y_range'])

    def test_nonuniform_resampling_can_cancel_between_blocks(self):
        source = np.arange(40, dtype=float)
        field = np.ones((40, 2), dtype=np.complex64)
        with self.assertRaises(InterruptedError):
            resample_pair(source, field, field.real, source[:-1]+.5, axis=0,
                support=np.ones(39, bool), cancel_check=lambda: True)


if __name__ == '__main__':
    unittest.main()
