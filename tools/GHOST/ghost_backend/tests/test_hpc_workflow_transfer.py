"""Real source-checkout transfer and frozen HPC-worker regression checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND.parent), str(BACKEND / "tests")]
from ghost_backend.hpc import bundle
from ghost_backend.hpc.common import latest_run_dir, run_status
from ghost_backend.hpc.runtime_snapshot import snapshot_backend_runtime
from general_fixtures import configured_2d_driver


class HpcWorkflowTransferTests(unittest.TestCase):
    def test_portable_bor_stage_runs_real_driver_with_empty_pythonpath(self):
        # This exercises the subprocess boundary that the bundle unit tests mock.
        with tempfile.TemporaryDirectory(prefix="ghost-stage-import-") as raw:
            root = Path(raw)
            request = root / "request"
            bundle.create_portable_bundle(
                request, solver="bor",
                geometries=[{"role": "BOR", "path": str(
                    BACKEND / "geometry/geometries/BOR/pec_cylinder_r4_h10_in.geo")}],
                settings={"FREQUENCIES_GHZ": [0.1], "AZIMUTHS_DEG": [0., 90.],
                          "ELEVATIONS_DEG": [0.], "MESH_CERTIFICATION": False,
                          "WORKERS_PER_UNIT": 1, "CORES_PER_NODE": 1})
            with mock.patch.object(bundle, "_linux_staging_available", return_value=True), \
                    mock.patch.dict(os.environ, {"PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}):
                result = bundle.stage_portable_bundle(
                    request, root / "workspace", run_driver=True, submit=False)
            log = Path(result["log_path"]).read_text(encoding="utf-8")
            self.assertTrue(result["ok"], log)
            self.assertTrue(result["driver_ran"])
            self.assertFalse(result["submitted"])
            run = Path(result["run_dir"])
            self.assertTrue((run / "manifest.json").is_file())
            self.assertTrue((run / "submit_job0.slurm").is_file())
            self.assertTrue((run / "runtime/ghost_backend/bor/solver.py").is_file())

    def test_config_and_solver_edits_cannot_change_frozen_2d_run(self):
        with tempfile.TemporaryDirectory(prefix="ghost-frozen-worker-") as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            import_root = snapshot_backend_runtime(BACKEND, source)
            backend = import_root / "ghost_backend"
            geometry = root / "geometry"
            geometry.mkdir()
            empty = root / "empty"
            empty.mkdir()
            (geometry / "rectangle.geo").write_text(
                "Title: frozen worker\nSegment: rectangle 2\nproperties: 2 24 0 0 0\n"
                "-.02 -.01 -.02 .01\n-.02 .01 .02 .01\n.02 .01 .02 -.01\n"
                ".02 -.01 -.02 -.01\nIBCS_Resistances:\nDielectrics:\n")
            driver = backend / "run_hpc_monostatic.py"
            configured_2d_driver(driver, driver, dict(
                FREQUENCIES_GHZ=[1.], AZIMUTHS_DEG=[0., 90.], GEOMETRY_UNITS="meters",
                N_NODES=1, N_JOBS=1, MAX_WORKERS_PER_NODE=1, CORES_PER_NODE=1,
                MESH_CERTIFICATION=False, OUTPUT_DIR=str(root / "runs"), SUBMIT=False,
                FRD_DIR=str(geometry), OPN_DIR=str(empty)))
            env = dict(os.environ, PYTHONPATH=str(import_root), PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       GHOST_TIMING_CACHE_DIR=str(root / "timings"))
            def invoke(path, *args):
                return subprocess.run([sys.executable, "-B", str(path), *args],
                    cwd=root, env=env, capture_output=True, text=True, timeout=120)
            submitted = invoke(driver)
            self.assertEqual(submitted.returncode, 0, submitted.stdout + submitted.stderr)
            run = latest_run_dir(root / "runs")
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["runtime_pythonpath"], "runtime")
            self.assertIn(str(run / "runtime"), (run / "submit_job0.slurm").read_text())
            # Modify only a private checkout, representing another configured run
            # and a subsequent solver update after this run was submitted.
            configured_2d_driver(driver, driver, {"FREQUENCIES_GHZ": [2.]})
            original_solver = backend / "twod/constants.py"
            original_solver.write_text(original_solver.read_text() + "\n# later checkout update\n")
            configured = run / "driver_configured.py"
            completed = invoke(configured, "--worker", str(run), "0", "0")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(run_status(run)["attestation_verified"])
            self.assertEqual(len(list((run / "results").rglob("*.grim"))), 1)
            # The frozen copy remains integrity checked, including on output reuse.
            frozen_solver = run / "runtime/ghost_backend/twod/constants.py"
            frozen_solver.write_text(frozen_solver.read_text() + "\n# unexpected mutation\n")
            rejected = invoke(configured, "--worker", str(run), "0", "0")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("source/native artifacts differ", rejected.stdout + rejected.stderr)


if __name__ == "__main__":
    unittest.main()
