"""Round 12: both HPC drivers end to end on bodies with impedance junctions.

The configured SLURM worker entry points are run locally without submission, as test_automatic_hpc does: plan, then the
worker in a fresh interpreter in which importing Qt raises.  (The Qt names below make the suite runner give this module
its own interpreter; it needs no Qt.)
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND.parent), str(BACKEND/'tests')]
from ghost_backend.bor import solver as bor

RADIUS = .1


GEOMETRY = """Title: PEC and impedance halves of a rectangle
Segment: pec 2
properties: 2 24 0 0 0
-.02 -.01 -.02 .01
-.02 .01 .02 .01
Segment: coated 2
properties: 2 24 1 0 0
.02 .01 .02 -.01
.02 -.01 -.02 -.01
IBCS_Resistances:
{law}
Dielectrics:
"""


class HpcDriverJunctionTests(unittest.TestCase):
    """The configured SLURM worker entry point, run locally without submission (as test_automatic_hpc does)."""

    def _run(self, folder, law):
        from general_fixtures import configured_2d_driver
        from ghost_backend.hpc.common import latest_run_dir, run_status
        work = Path(folder)
        (work/'geometry').mkdir()
        (work/'empty').mkdir()
        (work/'geometry'/'body.geo').write_text(GEOMETRY.format(law=law))
        settings = dict(FREQUENCIES_GHZ=[1.], AZIMUTHS_DEG=[0., 45., 90.], GEOMETRY_UNITS='meters', N_NODES=1, N_JOBS=1,
                        MAX_WORKERS_PER_NODE=1, MESH_CERTIFICATION=True, OUTPUT_DIR=str(work/'runs'), SUBMIT=False,
                        FRD_DIR=str(work/'geometry'), OPN_DIR=str(work/'empty'))
        driver = configured_2d_driver(BACKEND/'run_hpc_monostatic.py', work/'driver.py', settings)
        (work/'sitecustomize.py').write_text(
            'import sys\nclass NoGui:\n    def find_spec(self, fullname, path=None, target=None):\n'
            '        if fullname.split(".")[0] in ("PySide6", "PySide2", "PyQt5", "PyQt6"):\n'
            '            raise RuntimeError("GUI imported into HPC worker")\nsys.meta_path.insert(0, NoGui())\n')
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(BACKEND.parent))), OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', GHOST_TIMING_CACHE_DIR=str(work/'timings'),
                   PYTHONDONTWRITEBYTECODE='1')
        for arguments in ([], None):
            if arguments is None:
                arguments = ['--worker', str(latest_run_dir(work/'runs')), '0', '0']
            process = subprocess.run([sys.executable, str(driver), *arguments], env=env, cwd=work, capture_output=True,
                                     text=True, timeout=300)
            self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
        directory = latest_run_dir(work/'runs')
        self.assertTrue(run_status(directory)['attestation_verified'])
        forecast = json.loads((directory/'schedule.json').read_text())['units'][0]
        results = list((directory/'results').rglob('*.grim'))
        self.assertEqual(len(results), 1)
        with np.load(results[0], allow_pickle=False) as archive:
            metadata = json.loads(str(archive['solver_metadata_json'].reshape(()).item()))['metadata']
        self.assertTrue(metadata['mesh_convergence_certified'])
        return forecast, metadata

    def test_the_scheduler_forecasts_the_graded_mesh_the_worker_solves(self):
        # 4 x 24 panels, 144 on the certification mesh. At 1 GHz the 40 mm sides have 180 panels per wavelength
        # (7 levels) and the 20 mm sides 360 (8); each junction joins one of each. The certification mesh gives
        # each chain ceil(1.5 B) panels, longest panels first (48 + 24 per 72), so its junctions grade 8 + 8.
        for law, base, fine in (('1 constant 75 -20 0 0', 96+2*(7+8), 144+2*(8+8)), ('1 constant 1 0 0 0', 126, 176),
                                ('1 constant 0 0 0 0', 96, 144)):
            with self.subTest(law=law), tempfile.TemporaryDirectory() as folder:
                forecast, metadata = self._run(folder, law)
                self.assertEqual((forecast['nodes'], forecast['fine_nodes']), (base, fine))
                self.assertEqual(metadata['panel_count'], fine)


class HpcBorDriverTests(unittest.TestCase):
    """The configured BoR SLURM worker entry point, run locally without submission, in an interpreter without Qt."""

    @staticmethod
    def _geometry(law, elements=24):
        points = bor.sphere_generatrix(RADIUS, elements)
        rows = lambda part: '\n'.join('{:.12g} {:.12g} {:.12g} {:.12g}'.format(*a, *b) for a, b in zip(part[:-1], part[1:]))
        half = elements//2
        return ('Title: PEC and impedance halves of a sphere\nSegment: pec 2\nproperties: 2 1 0 0 0\n'+rows(points[:half+1])
                + '\nSegment: coated 2\nproperties: 2 1 1 0 0\n'+rows(points[half:])+'\nIBCS_Resistances:\n'+law+'\nDielectrics:\n')

    def test_the_worker_solves_the_graded_generatrix(self):
        from ghost_backend.hpc.common import configure_driver, latest_run_dir
        backend = Path(__file__).resolve().parents[1]
        for law, elements, graded, formulation in (('1 constant 100 50 0 0', 32, 1, 'IBC-CFIE'), ('1 constant 0 0 0 0', 24, 0, 'PEC CFIE')):
            with self.subTest(law=law), tempfile.TemporaryDirectory() as folder:
                work = Path(folder)
                (work/'geometry').mkdir()
                (work/'geometry'/'sphere.geo').write_text(self._geometry(law))
                driver = configure_driver(backend/'run_hpc_bor_monostatic.py', work/'driver.py', dict(
                    GEOMETRY_DIRS=[str(work/'geometry')], FREQUENCIES_GHZ=[1.5], AZIMUTHS_DEG=[0., 90.], ELEVATIONS_DEG=[0., 30.],
                    OUTPUT_DIR=str(work/'runs'), N_NODES=1, N_JOBS=1, GEOMETRY_UNITS='meters', MESH_CERTIFICATION=False,
                    WORKERS_PER_UNIT=1, SUBMIT=False))
                (work/'sitecustomize.py').write_text(
                    'import sys\nclass NoGui:\n    def find_spec(self, fullname, path=None, target=None):\n'
                    '        if fullname.split(".")[0] in ("PySide6", "PySide2", "PyQt5", "PyQt6"):\n'
                    '            raise RuntimeError("GUI imported into HPC worker")\nsys.meta_path.insert(0, NoGui())\n')
                env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(backend.parent))), OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                for arguments in ([], None):
                    if arguments is None:
                        arguments = ['--worker', str(latest_run_dir(work/'runs')), '0', '0']
                    process = subprocess.run([sys.executable, str(driver), *arguments], env=env, cwd=work, capture_output=True,
                                             text=True, timeout=600)
                    self.assertEqual(process.returncode, 0, process.stdout+process.stderr)
                directory = latest_run_dir(work/'runs')
                manifest = json.loads((directory/'manifest.json').read_text())
                self.assertTrue(all(unit['estimated_peak_gb'] > 0 for unit in manifest['units']))
                files = sorted((directory/'results'/'by_frequency').glob('*.grim'))
                self.assertEqual([path.name[:2] for path in files], ['HH', 'VV'])
                for path in files:
                    with np.load(path, allow_pickle=False) as archive:
                        metadata = json.loads(str(archive['solver_metadata_json'].reshape(()).item()))['metadata']
                    record = metadata['per_frequency'][0]
                    self.assertEqual((record['mesh_elements_total'], record['graded_impedance_junctions']), (elements, graded))
                    self.assertIn(formulation, metadata['formulation'])
                self.assertTrue((directory/'results'/'sphere.grim').is_file())


if __name__ == '__main__':
    unittest.main()
