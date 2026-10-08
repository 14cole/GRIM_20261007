"""Native mixed-dB overlays through report extraction, preview and export."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.reports import plot_data, report
from test_ppt_plot_data import _grid


def mixed_grids(*, angle_unit="deg", frequency_unit="GHz"):
    azimuths = np.asarray((90.0, -90.0, 0.0))
    elevations = np.asarray((20.0, -20.0, 0.0))
    frequencies = np.asarray((2.0, 1.0))
    az = azimuths[:, None, None, None]
    el = elevations[None, :, None, None]
    freq = frequencies[None, None, :, None]
    sigma = 10.0 ** ((10.0 + az * 0.01 + el * 0.02 + freq) / 10.0)
    width = C0 / (2.0 * np.pi * freq * 1.0e9) * (
        10.0 ** ((-10.0 + az * 0.02 - el * 0.01 + 3.0 * freq) / 10.0)
    )
    kwargs = dict(
        azimuths=np.deg2rad(azimuths) if angle_unit == "rad" else azimuths,
        elevations=np.deg2rad(elevations) if angle_unit == "rad" else elevations,
        frequencies=frequencies * 1.0e9 if frequency_unit == "Hz" else frequencies,
        angle_unit=angle_unit,
        frequency_unit=frequency_unit,
        phase=np.deg2rad(az * 0.2 + el * 0.1 + freq),
    )
    return (
        ("Test", _grid(power=sigma, **kwargs)),
        ("Analysis", _grid(power=width, quantity="sigma_2d", log_unit="dBke", **kwargs)),
    )


def specs_for(datasets, *, quantity="magnitude", native_frequency=2.0):
    return (
        plot_data.build_azimuth_specs(
            datasets, frequencies=(native_frequency,), elevation=0.0,
            polarization="VV", kind="azimuth_rect", quantity=quantity,
        )[0],
        plot_data.build_azimuth_specs(
            datasets, frequencies=(native_frequency,), elevation=0.0,
            polarization="VV", kind="azimuth_polar", quantity=quantity,
        )[0],
        plot_data.build_elevation_specs(
            datasets, frequencies=(native_frequency,), azimuth=0.0,
            polarization="VV", quantity=quantity,
        )[0],
        plot_data.build_frequency_spec(
            datasets, azimuth=0.0, elevation=0.0,
            polarization="VV", quantity=quantity,
        ),
    )


class PptMixedDbTests(unittest.TestCase):
    def test_native_values_and_labels_in_every_exact_report_family(self):
        for units in (("deg", "GHz"), ("rad", "Hz")):
            with self.subTest(units=units):
                datasets = mixed_grids(angle_unit=units[0], frequency_unit=units[1])
                specs = specs_for(datasets, native_frequency=2e9 if units[1] == "Hz" else 2.0)
                for spec in specs:
                    with self.subTest(kind=spec.kind):
                        self.assertEqual(spec.y_label, "Mixed dB")
                        self.assertEqual([series.label for series in spec.series],
                                         ["Test [dBsm]", "Analysis [dBke]"])
                        x = np.asarray(spec.series[0].x)
                        if spec.kind.startswith("azimuth"):
                            expected = (12.0 + x * 0.01, -4.0 + x * 0.02)
                        elif spec.kind == "elevation":
                            expected = (12.0 + x * 0.02, -4.0 - x * 0.01)
                        else:
                            expected = (10.0 + x, -10.0 + x * 3.0)
                        for series, values in zip(spec.series, expected):
                            np.testing.assert_allclose(series.y, values, atol=1e-12)

    def test_frequency_band_p50_retains_each_native_logarithmic_quantity(self):
        spec = plot_data.build_frequency_spec(
            mixed_grids(), azimuth=None, elevation=0.0, polarization="VV",
            azimuth_band=(-90.0, 90.0), azimuth_percentile=50.0,
        )
        self.assertEqual(spec.y_label, "Mixed dB")
        self.assertEqual([series.label for series in spec.series],
                         ["Test [dBsm]", "Analysis [dBke]"])
        np.testing.assert_allclose(spec.series[0].y, [11.0, 12.0], atol=1e-12)
        np.testing.assert_allclose(spec.series[1].y, [-7.0, -4.0], atol=1e-12)
        self.assertIn("P50", spec.title)

    def test_phase_overlays_have_no_magnitude_unit_labels(self):
        for spec in specs_for(mixed_grids(), quantity="phase"):
            with self.subTest(kind=spec.kind):
                self.assertEqual(spec.y_label, "Phase (deg)")
                self.assertEqual([series.label for series in spec.series], ["Test", "Analysis"])
                np.testing.assert_allclose(spec.series[0].y, spec.series[1].y)
                x = np.asarray(spec.series[0].x)
                expected = (x * 0.2 + 2.0 if spec.kind.startswith("azimuth") else
                            x * 0.1 + 2.0 if spec.kind == "elevation" else x)
                np.testing.assert_allclose(spec.series[0].y, expected, atol=1e-12)

    def test_homogeneous_2d_and_ratio_reports_use_physical_quantity_labels(self):
        for dataset, expected_label in (
            (mixed_grids()[1], "Scattering Width (dBke)"),
            (("Ratio", _grid(elevations=(-20.0, 0.0, 20.0),
                             quantity="power_ratio", log_unit="dB")), "Power Ratio (dB)"),
        ):
            for spec in specs_for((dataset,)):
                with self.subTest(label=expected_label, kind=spec.kind):
                    self.assertEqual(spec.y_label, expected_label)
                    self.assertEqual(spec.series[0].label, dataset[0])

    def test_mixed_labels_reach_powerpoint_assets_and_image_manifest(self):
        from matplotlib.figure import Figure

        plots = specs_for(mixed_grids())
        plan = report.combine_plans(
            report.plan_azimuth_slides(plots[:3], master_legend=True),
            report.plan_frequency_slides(plots[3:], master_legend=True),
        )
        expected_labels = ["Test [dBsm]", "Analysis [dBke]"]
        for slide in plan.slides:
            self.assertEqual([entry.label for entry in slide.master_legend], expected_labels)
        captured_axes = []
        savefig = Figure.savefig

        def inspect_save(figure, *args, **kwargs):
            for axes in figure.axes:
                if axes.get_ylabel():
                    captured_axes.append((axes.get_ylabel(),
                                          [line.get_label() for line in axes.lines]))
            return savefig(figure, *args, **kwargs)

        class CaptureWriter:
            def write(self, export_plan, rendered_images, destination, **kwargs):
                self.plan = export_plan
                self.images = {key: Path(path).read_bytes() for key, path in rendered_images.items()}
                destination.write_bytes(b"export writer reached")

        writer = CaptureWriter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(Figure, "savefig", inspect_save):
                report.export_powerpoint_report(plan, root / "mixed.pptx", writer=writer)
            self.assertEqual(len(captured_axes), 4)
            self.assertTrue(all(label == "Mixed dB" and labels == expected_labels
                                for label, labels in captured_axes))
            self.assertEqual(len(writer.images), 6)
            self.assertTrue(all(data.startswith(b"\x89PNG\r\n\x1a\n")
                                for data in writer.images.values()))
            exported = report.export_report_images(plan, root / "images")
            manifest = json.loads((exported / "manifest.json").read_text())
            for slide in manifest["slides"]:
                self.assertEqual(slide["legend"]["series_labels"], expected_labels)
                for plot in slide["plots"]:
                    self.assertEqual(plot["axis_labels"]["y"], "Mixed dB")
                    self.assertEqual(plot["units"]["y"], "Mixed dB")
                    self.assertEqual(plot["series_labels"], expected_labels)


if __name__ == "__main__":
    unittest.main()
