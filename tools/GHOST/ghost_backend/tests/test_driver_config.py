"""BoR driver configuration travels with unchanged drivers and remains bound to run identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))
import ghost_backend.runs.config as config
import ghost_backend.hpc.common as hpc_common
import ghost_backend.execution.provenance as workflow_provenance


class DriverConfigurationTests(unittest.TestCase):
    def test_configuration_preserves_source_and_worker_fingerprint(self):
        for name in ('run_hpc_bor_monostatic.py', 'run_local_bor.py'):
            with self.subTest(driver=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = BACKEND / name
                staged = hpc_common.configure_driver(source, root / 'driver.py', {'OUTPUT_DIR': str(root / 'results')})
                self.assertEqual(source.read_bytes(), staged.read_bytes())
                kind, keys = config.driver_contract(source)
                namespace = {'__name__': 'imported_driver'}
                loaded = config.load_driver_configuration(namespace, staged, kind, keys)
                self.assertEqual(namespace['OUTPUT_DIR'], str(root / 'results'))
                worker = root / 'worker.py'
                worker.write_bytes(staged.read_bytes())
                config.copy_configuration(loaded, worker)
                worker_loaded = config.load_driver_configuration({}, worker, kind, keys)
                first = workflow_provenance.source_bundle_fingerprint(config.configuration_source_records(staged, loaded))
                second = workflow_provenance.source_bundle_fingerprint(config.configuration_source_records(worker, worker_loaded))
                self.assertEqual(first, second)
                loaded.path.write_text(loaded.path.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'changed after loading'):
                    config.configuration_source_records(staged, loaded)

    def test_invalid_settings_fail_before_creating_driver(self):
        for settings in ({'MISSPELLED_OPTION': 1}, {'N_NODES': True},
                         {'FREQUENCIES_GHZ': [float('nan')]}, {'JOB_PROLOGUE': 'text'},
                         {'MESH_CERTIFICATION': 'false'}):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'driver.py'
                with self.assertRaises(ValueError):
                    hpc_common.configure_driver(BACKEND / 'run_hpc_bor_monostatic.py', path, settings)
                self.assertFalse(path.exists())

    def test_explicit_configuration_overrides_adjacent_path_and_rejects_wrong_solver(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'explicit.json'
            payload = config.configuration_payload('bor', {'OUTPUT_DIR': 'chosen'}, ['OUTPUT_DIR'])
            config.write_configuration(path, payload)
            with mock.patch.object(sys, 'argv', ['driver.py', '--config', str(path)]):
                namespace = {'__name__': '__main__'}
                config.load_driver_configuration(namespace, root / 'driver.py', 'bor', ['OUTPUT_DIR'])
                self.assertEqual(namespace['OUTPUT_DIR'], 'chosen')
                with self.assertRaisesRegex(ValueError, 'solver kind'):
                    config.load_driver_configuration(namespace, root / 'driver.py', '2d', ['OUTPUT_DIR'])

    def test_two_dimensional_drivers_have_no_json_configuration(self):
        with self.assertRaisesRegex(ValueError, 'BoR drivers only'):
            config.configuration_payload('2d', {'OUTPUT_DIR': 'chosen'}, ['OUTPUT_DIR'])
        for name in ('run_hpc_monostatic.py', 'run_local_monostatic.py'):
            with self.subTest(driver=name), self.assertRaisesRegex(ValueError, 'configuration contract'):
                config.driver_contract(BACKEND / name)


if __name__ == '__main__':
    unittest.main()
