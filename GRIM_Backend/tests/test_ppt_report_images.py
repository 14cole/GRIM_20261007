"""PowerPoint-free image publication shares the presentation render contract."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from GRIM_Backend.reports import report


def plot(identifier, kind='azimuth_rect', title='9 GHz | Elevation 20 deg | VV'):
    return report.PlotSpec(identifier, kind, title,
        'Frequency (GHz)' if kind == 'frequency' else 'Azimuth (deg)', 'RCS (dBsm)',
        (report.PlotSeries.from_values([0., 30., 60.], [-10., -8., -12.], label='Model A'),
         report.PlotSeries.from_values([0., 30., 60.], [-11., -9., -11.], label='Range')))


def fake_render(value, destination, **kwargs):
    Path(destination).write_bytes(b'fake PNG for publication tests')
    return Path(destination)


class ReportImageExportTests(unittest.TestCase):
    def plan(self):
        return report.combine_plans(
            report.plan_azimuth_slides([plot('az'), plot('polar', 'azimuth_polar'), plot('el', 'elevation')],
                slide_titles='Rocket body — VV', footer='Chosen comparison cuts', master_legend=True),
            report.plan_frequency_slides([plot('frequency', 'frequency', 'Frequency Sweep | Azimuth 30 deg | VV')],
                slide_titles='Frequency sweep', master_legend=True))

    def test_images_equal_powerpoint_assets_and_pixel_dimensions_without_com(self):
        from PIL import Image
        plan = self.plan()
        captured = {}
        class CaptureWriter:
            def write(self, plan, rendered_images, destination, **kwargs):
                captured.update({key: path.read_bytes() for key, path in rendered_images.items()})
                destination.write_bytes(b'fake presentation')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            report.export_powerpoint_report(plan, base / 'reference.pptx', writer=CaptureWriter(), dpi=100)
            with mock.patch.object(report, 'PowerPointComBridge', side_effect=AssertionError('COM called')), \
                    mock.patch.object(report, 'pythoncom', None), mock.patch.object(report, 'win32com', None), \
                    mock.patch.dict('sys.modules', {'pptx': None}):
                exported = report.export_report_images(plan, base / 'images', dpi=100)
            self.assertEqual(exported, (base / 'images').resolve())
            manifest = json.loads((exported / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['plot_count'], 4)
            self.assertEqual(manifest['slide_count'], 2)
            self.assertEqual(manifest['dpi'], 100)
            for slide_index, slide in enumerate(manifest['slides']):
                self.assertEqual(slide['title'], plan.slides[slide_index].title)
                for plot_index, item in enumerate(slide['plots']):
                    image_path = exported / item['relative_path']
                    self.assertEqual(image_path.read_bytes(), captured[slide_index, plot_index])
                    placement = plan.slides[slide_index].plots[plot_index]
                    with Image.open(image_path) as image:
                        self.assertEqual(image.format, 'PNG')
                        self.assertAlmostEqual(image.width, placement.frame.width / 72. * 100, delta=1.)
                        self.assertAlmostEqual(image.height, placement.frame.height / 72. * 100, delta=1.)
                    self.assertEqual(item['plot_id'], placement.plot.plot_id)
                    self.assertEqual(item['type'], placement.plot.kind)
                    self.assertEqual(item['selection_caption'], placement.plot.title)
                    self.assertEqual(item['series_labels'], ['Model A', 'Range'])
                    self.assertEqual(item['units']['y'], 'dBsm')
                legend = exported / slide['legend']['relative_path']
                self.assertEqual(legend.read_bytes(), captured[slide_index, report.MASTER_LEGEND_IMAGE_INDEX])
            self.assertEqual(sorted(path.name for path in exported.iterdir()),
                             sorted([slide['folder'] for slide in manifest['slides']] + ['manifest.json']))

    def test_short_safe_unique_names_preserve_original_captions_in_manifest(self):
        dangerous = '../CON.: <test> " / \\ | ? * \x00 ' + 'Long title ' * 20
        plots = [plot(f'../plot-{index}', title=dangerous) for index in range(13)]
        plan = report.plan_azimuth_slides(plots, slide_titles=dangerous)
        with tempfile.TemporaryDirectory() as directory:
            output = report.export_report_images(plan, Path(directory) / 'images', renderer=fake_render)
            manifest = json.loads((output / 'manifest.json').read_text())
            names = []
            for index, slide in enumerate(manifest['slides']):
                self.assertEqual(slide['title'], plan.slides[index].title)
                self.assertLessEqual(len(slide['folder']), 40)
                for item in slide['plots']:
                    self.assertEqual(item['title'], dangerous)
                    self.assertLessEqual(len(item['file']), 80)
                    path = output / item['relative_path']
                    self.assertTrue(path.resolve().is_relative_to(output.resolve()))
                    self.assertIsNone(re.search(r'[<>:"/\\|?*\x00-\x1f]', path.name))
                    self.assertFalse(path.name.endswith((' ', '.')))
                    names.append(item['relative_path'].casefold())
            self.assertEqual(len(names), len(set(names)))
        for name in ('CON', 'con.txt', 'LPT1', 'COM¹.foo', 'AUX', 'NUL', 'CONOUT$'):
            self.assertTrue(report._image_export_name(name).startswith('_'))
        self.assertEqual(report._image_export_name(' . / \\ '), 'Untitled')

    def test_existing_nonempty_directory_or_file_refused_before_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            occupied = root / 'occupied'
            occupied.mkdir()
            empty = root / 'empty'
            empty.mkdir()
            sentinel = occupied / 'important.txt'
            sentinel.write_text('keep')
            for destination in (occupied, sentinel, empty):
                with self.subTest(destination=destination), \
                        mock.patch.object(report, 'render_plan_images', side_effect=AssertionError('rendered')):
                    with self.assertRaises(FileExistsError):
                        report.export_report_images(self.plan(), destination)
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertEqual(set(root.iterdir()), {occupied, empty})
            self.assertEqual(list(empty.iterdir()), [])

    def test_render_failure_cleans_staging_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'images'
            def failing(value, path, **kwargs):
                Path(path).write_bytes(b'partial')
                raise RuntimeError('renderer failed')
            with self.assertRaisesRegex(RuntimeError, 'renderer failed'):
                report.export_report_images(self.plan(), output, renderer=failing, legend_renderer=fake_render)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_publish_failure_does_not_publish_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'images'
            with mock.patch.object(report.os, 'rename', side_effect=OSError('publication failed')):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    report.export_report_images(self.plan(), output, renderer=fake_render, legend_renderer=fake_render)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_new_files_appearing_during_render_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'images'
            def concurrent_write(value, path, **kwargs):
                output.mkdir(exist_ok=True)
                (output / 'keep.txt').write_text('new content')
                return fake_render(value, path, **kwargs)
            with self.assertRaises(FileExistsError):
                report.export_report_images(self.plan(), output, renderer=concurrent_write, legend_renderer=fake_render)
            self.assertEqual((output / 'keep.txt').read_text(), 'new content')
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_manifest_failure_and_destination_links_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'images'
            with mock.patch.object(report.Path, 'write_text', side_effect=OSError('manifest failed')):
                with self.assertRaisesRegex(OSError, 'manifest failed'):
                    report.export_report_images(self.plan(), output, renderer=fake_render, legend_renderer=fake_render)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with mock.patch.object(report.Path, 'is_symlink', return_value=True):
                with self.assertRaisesRegex(ValueError, 'not a link'):
                    report.export_report_images(self.plan(), output, renderer=fake_render, legend_renderer=fake_render)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_external_renderer_asset_is_copied_and_preserved(self):
        plan = report.plan_frequency_slides([plot('single', 'frequency')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'images'
            external = root / 'reusable.png'
            external.write_bytes(b'reusable PNG')
            exported = report.export_report_images(plan, output, renderer=lambda *args, **kwargs: external)
            self.assertEqual(external.read_bytes(), b'reusable PNG')
            manifest = json.loads((exported / 'manifest.json').read_text())
            self.assertNotIn('legend', manifest['slides'][0])
            self.assertEqual((exported / manifest['slides'][0]['plots'][0]['relative_path']).read_bytes(), b'reusable PNG')

    def test_long_windows_path_is_rejected_before_rendering_or_creating_folders(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(report.sys, 'platform', 'win32'):
            root = Path(directory)
            output = root / ('long parent ' * 7) / ('another parent ' * 6) / 'images'
            with mock.patch.object(report, 'render_plan_images', side_effect=AssertionError('rendered')):
                with self.assertRaisesRegex(ValueError, 'shorter destination'):
                    report.export_report_images(self.plan(), output)
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_dpi_and_empty_renderer_output_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'images'
            for dpi in (True, 0, 71, 100.5):
                with self.subTest(dpi=dpi), self.assertRaises(ValueError):
                    report.export_report_images(self.plan(), output, dpi=dpi)
            def empty(value, path, **kwargs):
                Path(path).touch()
                return Path(path)
            with self.assertRaisesRegex(RuntimeError, 'nonempty image'):
                report.export_report_images(self.plan(), output, renderer=empty, legend_renderer=empty)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
