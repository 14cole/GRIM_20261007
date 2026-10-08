"""The automatic run profile and its driver and GUI resolution."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.execution.options import (
    DEFAULTS, automatic_options, automatic_run, efficient_defaults, from_environment, validate_for_run, validate_options, blas_thread_reservation,
    host_assembly_threads,
)
from ghost_backend.runs.execution import driver_options


class EfficientDefaultsTests(unittest.TestCase):
    def test_automatic_run_certifies_and_backends_preserve_pec_and_mixed_fields(self):
        import numpy as np
        from ghost_backend.twod.solver import solve_monostatic_rcs_2d_certified
        from test_experimental_cpu import fixture, fields
        for material in ('pec', 'mixed'):
            results = []
            for factorization, method in (('dense', 'direct'), ('compressed', 'experimental_cpu')):
                options = dict(efficient_defaults(), factorization=factorization, mesh_strategy='global')
                result = solve_monostatic_rcs_2d_certified(fixture(material,64), [.6], list(range(361)),
                    geometry_units='meters', solver_method=method, execution_options=options)
                self.assertTrue(result['metadata']['mesh_convergence_certified'])
                results.append(result)
            for pol in ('VV','HH'):
                a,b = fields(results[0],pol),fields(results[1],pol)
                self.assertLess(float(np.max(abs(a-b))/np.max(abs(a))), 1e-10)
            run = automatic_run('monostatic')
            result = solve_monostatic_rcs_2d_certified(fixture(material,64), [.6], list(range(361)),
                geometry_units='meters', solver_method=run['solver_method'], execution_options=run['execution_options'])
            self.assertTrue(result['metadata']['mesh_convergence_certified'])
            # The profile's 'auto' assembly threads resolve on the solving host.
            self.assertEqual(result['metadata']['execution_threads'],
                dict(assembly=host_assembly_threads(),blas=efficient_defaults()['blas_threads']))

    def test_automatic_run_matches_each_scattering_capability(self):
        monostatic, bistatic = automatic_run('monostatic'), automatic_run('bistatic')
        self.assertEqual((monostatic['solver_method'], monostatic['lu_precision']), ('auto', 'double'))
        self.assertEqual(monostatic['execution_options'], automatic_options())
        self.assertEqual((bistatic['solver_method'], bistatic['execution_options']['factorization'],
                          bistatic['execution_options']['mesh_strategy']), ('direct', 'dense', 'global'))
        for scattering, run in (('monostatic', monostatic), ('bistatic', bistatic)):
            validate_for_run(run['execution_options'], run['solver_method'], run['lu_precision'], scattering)
        self.assertEqual(automatic_options(12)['ram_budget_gib'], 12.)
        with self.assertRaises(ValueError):
            automatic_run('forward')

    def test_preset_and_launch_overrides_do_not_change_serialized_v1_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(os, 'cpu_count', return_value=8):
            profile = efficient_defaults()
            self.assertEqual((profile['factorization'], profile['compressed_storage_mib'],
                              profile['assembly_threads'], profile['blas_threads']), ('adaptive',0,'auto','auto'))
            self.assertEqual((profile['rhs_compression'], profile['angle_batch_size']), ('auto',256))
            self.assertIsNone(profile['ram_budget_gib'])
            self.assertEqual(from_environment(profile), profile)
            self.assertEqual(validate_options({}), DEFAULTS)
            self.assertEqual(from_environment()['factorization'], 'dense')
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense', 'OPENBLAS_NUM_THREADS':'1'}):
                changed = from_environment(profile)
                self.assertEqual((changed['factorization'],changed['blas_threads']), ('dense',1))
            profile['assembly_threads'] = 99
            self.assertEqual(efficient_defaults()['assembly_threads'], 'auto')
        with mock.patch.object(os, 'cpu_count', return_value=1):
            self.assertEqual(blas_thread_reservation(efficient_defaults()), 1)
            self.assertEqual(efficient_defaults()['assembly_threads'], 'auto')

    def test_drivers_resolve_to_the_automatic_profile(self):
        import run_local_monostatic as local
        import run_hpc_monostatic as hpc
        with mock.patch.dict(os.environ, {}, clear=True):
            for driver in (local,hpc):
                for name in ('SOLVER_METHOD', 'LU_PRECISION', 'SOLVE_PRESET', 'EXECUTION_OPTIONS', '_CONFIG_KEYS'):
                    self.assertFalse(hasattr(driver, name), name)
                self.assertTrue(driver.MESH_CERTIFICATION)
                self.assertEqual(driver_options(vars(driver)), automatic_options())
                self.assertEqual(driver_options(dict(vars(driver), MAX_SOLVE_GB=24))['ram_budget_gib'], 24.)


if __name__ == '__main__':
    unittest.main()
