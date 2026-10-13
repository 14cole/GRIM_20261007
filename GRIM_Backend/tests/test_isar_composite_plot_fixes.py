"""Physical regressions for composite look widths, panel frames, and coverage."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from GRIM_Backend.isar.artifact import build_isar_manifest, load_isar_artifact, save_isar_artifact
from GRIM_Backend.isar.comparison import compare_images, hydrate_band
from GRIM_Backend.isar.geometry import angular_sublooks, pool_image_peaks
from GRIM_Backend.plotting.modes import isar_mode, isar_render
from GRIM_Backend.scripting.plotting import plot_datasets
from test_isar_improvements import grid, params
from test_reconstruction_accuracy import harness, point_grid


class CompositePlotFixTests(unittest.TestCase):
    def test_sublooks_distribute_remainder_without_narrow_tail_or_lost_samples(self):
        for span in (20.1, 21., 30.):
            angles = np.arange(-span / 2, span / 2 + .01, .05)
            for overlap in (0., .5):
                with self.subTest(span=span, overlap=overlap):
                    looks = angular_sublooks(range(len(angles)), angles, overlap=overlap)
                    self.assertEqual(set(sum(looks, [])), set(range(len(angles))))
                    widths = [np.ptp(angles[look]) for look in looks]
                    self.assertTrue(all(9.9 <= width <= 10. + 1e-10 for width in widths))

    def test_auto_composite_point_does_not_acquire_a_short_look_ridge(self):
        # The original 20.1-degree selection produced a two-sample final look:
        # a unit point became an almost full-strength ridge one meter away.
        angles = np.arange(-10.05, 10.05 + .01, .05)
        source = point_grid(angles, np.linspace(9e9, 11e9, 129))
        # Exercise all application defaults: Fast PFA, Hanning, Auto aperture,
        # and the even-sized 1024-pixel composite grid.
        band = isar_mode.form_isar(source)[0][0]
        self.assertEqual(band['composite'], 3)
        self.assertTrue(all(look['span_degrees'] >= 9.9 for look in band['composite_sublooks']))
        x, y, magnitude = band['x_range'], band['y_range'], band['magnitude']
        peak = np.nanmax(magnitude)
        ridge_y = -np.tan(np.deg2rad(10.025))
        off_target = magnitude[np.argmin(abs(x - 1.)), np.argmin(abs(y - ridge_y))]
        self.assertLess(20 * np.log10(off_target / peak), -40.)
        half_power_area = np.sum(magnitude >= peak / np.sqrt(2)) * np.diff(x)[0] * np.diff(y)[0]
        self.assertLess(half_power_area, .01)

    def test_mixed_panels_and_export_label_each_actual_coordinate_frame(self):
        angles = np.r_[np.linspace(30., 35., 51), np.linspace(90., 120., 301)]
        source = point_grid(angles, np.linspace(9e9, 11e9, 65), x=.35, y=.45)
        bands, elapsed = isar_mode.form_isar(source, reconstruction='accurate',
            window='Rectangular', composite_side=128, scene_half_extent_m=(2., 2.))
        self.assertFalse(bands[0].get('composite'))
        self.assertTrue(bands[1].get('composite'))
        owner = harness(source)
        isar_render.display_results(owner, dict(dataset=source, unit_name='m',
            az_target_deg=None, elevation_deg=0., pol_idx=0, recon='accurate'), bands, elapsed)
        expected = [('Cross-Range at 32.5° (m)', 'Range at 32.5° (m)'),
                    ('Cross-Range at 0° (m)', 'Down-Range at 0° (m)')]
        self.assertEqual([(ax.get_xlabel(), ax.get_ylabel()) for ax in owner.plot_axes], expected)
        self.assertIn('Coherent + Wide-Aperture Composite', owner.plot_figure._suptitle.get_text())
        with mock.patch.object(isar_mode, 'form_isar', return_value=(bands, elapsed)):
            figure = plot_datasets([('point', source)], mode='isar_image',
                azimuths=angles, elevations=[0.], frequencies=source.frequencies,
                polarization='VV', show_colorbar=False, isar_options={'length_unit': 'm'})
        self.assertEqual([(ax.get_xlabel(), ax.get_ylabel()) for ax in figure.axes], expected)
        owner.plot_figure.clear()
        figure.clear()

    def test_uncovered_composite_pixels_stay_missing_and_measured_zeros_stay_zero(self):
        azimuth = np.linspace(40., 50., 65)
        frequency = np.linspace(9e9, 11e9, 65)
        source = grid(azimuth, frequency, np.zeros((65, 65), np.complex64))
        options = dict(reconstruction='accurate', window='Rectangular')
        coherent = isar_mode.form_isar(source, aperture_mode='coherent', **options)[0][0]
        bands, elapsed = isar_mode.form_isar(source, aperture_mode='composite',
            composite_side=64, **options)
        band = bands[0]
        x, y = band['x_range'], band['y_range']
        theta = np.deg2rad(45.)
        look_x = x[:, None] * np.cos(theta) - y[None, :] * np.sin(theta)
        look_y = x[:, None] * np.sin(theta) + y[None, :] * np.cos(theta)
        unsupported = ((look_x < coherent['x_range'][0]) | (look_x >= coherent['x_range'][-1])
            | (look_y < coherent['y_range'][0]) | (look_y >= coherent['y_range'][-1]))
        self.assertGreater(np.mean(unsupported), .05)
        np.testing.assert_array_equal(np.isnan(band['magnitude']), unsupported)
        np.testing.assert_array_equal(band['magnitude'][~unsupported], 0.)

        manifest = build_isar_manifest(source, params(source, aperture_mode='composite', composite_side=64), bands, elapsed)
        with tempfile.TemporaryDirectory() as directory:
            path = save_isar_artifact(Path(directory) / 'coverage', bands, manifest)
            saved_manifest, saved_bands = load_isar_artifact(path)
        restored = hydrate_band(saved_bands[0], saved_manifest)
        np.testing.assert_array_equal(restored['magnitude'], band['magnitude'])
        result = compare_images(manifest, band, saved_manifest, restored)
        np.testing.assert_array_equal(np.isnan(result['delta_db']), unsupported)
        self.assertEqual(result['statistics']['pixels'], int(np.count_nonzero(~unsupported)))
        self.assertEqual(result['statistics']['mean_delta_db'], 0.)

        owner = harness(source)
        for scale in ('linear', 'dbsm'):
            owner.combo_plot_scale.data = scale
            isar_render.display_results(owner, dict(dataset=source, unit_name='m',
                az_target_deg=None, elevation_deg=0., pol_idx=0, recon='accurate'), bands, elapsed)
            mesh = owner._isar_meshes[0]
            np.testing.assert_array_equal(np.ma.getmaskarray(mesh.get_array()), unsupported.T)
            np.testing.assert_array_equal(np.ma.getmaskarray(mesh._grim_image_record.values), unsupported.T)
            self.assertEqual(float(mesh.get_array().compressed()[0]), 0. if scale == 'linear' else -120.)
        owner.plot_figure.clear()

        pooled, _, _ = pool_image_peaks(band['magnitude'], 2, [(i, i + 2) for i in range(0, 64, 2)])
        expected_missing = unsupported.reshape(32, 2, 32, 2).all(axis=(1, 3))
        np.testing.assert_array_equal(np.isnan(pooled), expected_missing)
        np.testing.assert_array_equal(pooled[~expected_missing], 0.)

    def test_composite_artifact_does_not_accept_infinity_as_missing_coverage(self):
        source = grid(az=np.linspace(40., 50., 65))
        bands, elapsed = isar_mode.form_isar(source, aperture_mode='composite', composite_side=32)
        manifest = build_isar_manifest(source, params(source, aperture_mode='composite'), bands, elapsed)
        bands[0]['magnitude'][0, 0] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                save_isar_artifact(Path(directory) / 'invalid', bands, manifest)
