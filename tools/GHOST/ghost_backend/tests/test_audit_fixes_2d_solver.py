"""Regressions for the September 2026 2-D solver audit fixes.

1  per-solve warnings (no accumulation across a frequency sweep)
2  mesh-convergence dB/phase checks referred to a level floor (deep nulls)
3  kernel tables for direct, mixed-precision and boundary-density paths
4  deterministic condition estimate
5  merged metadata propagates non-finite values
6  memory detection edge cases and GiB labels
7  samples in request order
8  automatic assembly threads resolved on the executing host
9  operator cache key includes the far-quadrature/tile settings
"""
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import ghost_backend.twod.solver as rcs
import ghost_backend.execution.cpu as cpu
from ghost_backend.execution import options
from ghost_backend.linalg.refined_lu import linear_precision
from ghost_backend.runs.quality import (
    PRODUCTION_MESH_CONVERGENCE_DEFAULTS,
    accuracy_target_policy,
    evaluate_mesh_convergence,
)

GIB = 1024 ** 3
GLOBAL_DENSE = dict(mesh_strategy='global', basis_order=1, factorization='dense', blas_threads=1)


def circle(radius, count, seg_type=2, ibc=0, material=0, n_prop=1):
    theta = np.linspace(0.0, -2.0 * np.pi, count + 1)
    pairs = [dict(x1=float(radius * np.cos(theta[i])), y1=float(radius * np.sin(theta[i])),
                  x2=float(radius * np.cos(theta[i + 1])), y2=float(radius * np.sin(theta[i + 1])))
             for i in range(count)]
    return dict(name='circle', seg_type=seg_type,
                properties=[str(seg_type), str(n_prop), str(ibc), str(material), '0'], point_pairs=pairs)


def plate(n_prop='0', length=0.8, thickness=0.02):
    corners = [(-length / 2, -thickness / 2), (-length / 2, thickness / 2), (length / 2, thickness / 2),
               (length / 2, -thickness / 2), (-length / 2, -thickness / 2)]
    pairs = [dict(x1=p[0], y1=p[1], x2=q[0], y2=q[1]) for p, q in zip(corners[:-1], corners[1:])]
    return dict(segments=[dict(name='plate', seg_type=2, properties=['2', str(n_prop), '0', '0', '0'],
                               point_pairs=pairs)], ibcs=[], dielectrics=[])


def mixed_body():
    """A lossy dielectric body beside a PEC body (both air-facing)."""
    from ghost_backend.tests.general_fixtures import fixture
    return fixture('mixed', 128)


def fields(result, pol):
    return np.array([complex(row['rcs_amp_real'], row['rcs_amp_imag']) for row in result['co_solved_samples'][pol]])


def relative(a, b):
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def rows(frequency, angles, amplitudes, k):
    out = []
    for angle, value in zip(angles, amplitudes):
        linear = abs(value) ** 2 / (4 * k)
        out.append(dict(frequency_ghz=frequency, theta_inc_deg=float(angle), theta_scat_deg=float(angle),
                        rcs_amp_real=float(value.real), rcs_amp_imag=float(value.imag), rcs_linear=linear,
                        rcs_db=10 * math.log10(max(linear, 1e-12))))
    return out


def gate(base, fine, target='standard', **override):
    policy = dict(accuracy_target_policy(target), **override)
    policy.pop('fine_factor')
    return evaluate_mesh_convergence(dict(samples=base), dict(samples=fine), policy.pop('rms_limit_db'),
                                     policy.pop('max_abs_limit_db'), **policy)


class WarningScopeTests(unittest.TestCase):
    """Fix 1: each solve reports and gates only the notices it raised."""

    def test_eight_frequency_certified_sweep_with_large_estimates_passes(self):
        geometry = dict(segments=[circle(0.07, 48)], ibcs=[], dielectrics=[])
        frequencies = [1.0 + 0.1 * i for i in range(8)]
        calls = []

        def estimate(*args, **kwargs):
            # >8 GiB and different for every mesh, as large real solves are.
            calls.append(1)
            return 8.0 + 0.3 * len(calls)

        profile = dict(GLOBAL_DENSE, far_quadrature_order=12)
        with mock.patch.object(rcs, '_estimate_memory_gb', estimate), \
                mock.patch.object(rcs, '_solve_memory_limit_gb', lambda *a, **k: 1e6):
            result = rcs.solve_monostatic_rcs_2d_certified(
                geometry, frequencies, [0.0, 90.0], geometry_units='meters', solver_method='direct',
                execution_options=profile)
        metadata = result['metadata']
        self.assertTrue(metadata['mesh_convergence_certified'])
        self.assertTrue(metadata['quality_gate']['passed'])
        self.assertGreaterEqual(len(calls), 32)
        override = [text for text in metadata['warnings'] if 'Far-pair quadrature order overridden' in text]
        # The run-level union lists the repeated warning once ...
        self.assertEqual(len(override), 1)
        self.assertEqual(metadata['warning_count'], len(metadata['warnings']))
        for record in metadata['frequency_metadata']:
            for channel in ('VV', 'HH'):
                channel_metadata = record['metadata']['channel_metadata'][channel]
                # ... every solve still reports it, and nothing from other frequencies.
                self.assertEqual(channel_metadata['warnings'], override)
                self.assertEqual(channel_metadata['quality_gate']['values']['warnings_count'], 1)
                notes = [n for n in channel_metadata['information'] if n.startswith('Estimated peak memory')]
                self.assertEqual(len(notes), 1)
                self.assertIn('GiB', notes[0])

    def test_notices_written_directly_to_the_shared_library_are_not_lost(self):
        geometry = dict(segments=[circle(0.07, 48)], ibcs=[], dielectrics=[])
        original = rcs._mesh_wavelength_for_snapshot

        def library_writer(snapshot, materials, frequency):
            materials.warn_once('direct library warning')
            materials.inform_once('direct library note')
            return original(snapshot, materials, frequency)

        with mock.patch.object(rcs, '_mesh_wavelength_for_snapshot', library_writer):
            result = rcs.solve_monostatic_rcs_2d(geometry, [1.0, 1.1], [0.0], geometry_units='meters',
                                                 solver_method='direct', execution_options=GLOBAL_DENSE)
        metadata = result['metadata']
        self.assertEqual(metadata['warnings'].count('direct library warning'), 1)
        self.assertEqual(metadata['information'].count('direct library note'), 1)
        first = metadata['frequency_metadata'][0]['metadata']['channel_metadata']
        self.assertIn('direct library warning', first['VV']['warnings'])
        # Raised once per run by the library: later solves neither repeat nor accumulate it.
        later = metadata['frequency_metadata'][1]['metadata']['channel_metadata']
        self.assertEqual(later['VV']['warnings'], [])
        self.assertTrue(metadata['quality_gate']['passed'])

    def test_repeated_solves_in_one_preparation_scope_do_not_accumulate(self):
        from ghost_backend.twod.preparation import preparation_scope
        geometry = dict(segments=[circle(0.07, 48)], ibcs=[], dielectrics=[])
        counts = []
        with preparation_scope(), mock.patch.object(rcs, '_estimate_memory_gb', side_effect=lambda *a, **k: 9.0 + len(counts)), \
                mock.patch.object(rcs, '_solve_memory_limit_gb', lambda *a, **k: 1e6):
            for index in range(12):
                result = rcs.solve_monostatic_rcs_2d_single_polarization(
                    geometry, [1.0 + 0.05 * index], [0.0], 'TM', geometry_units='meters', solver_method='direct')
                counts.append(result['metadata']['warning_count'])
        self.assertEqual(counts, [0] * 12)


class MeshConvergenceFloorTests(unittest.TestCase):
    """Fix 2: dB/phase are judged relative to a level floor; complex change stays primary."""

    k = 62.8
    angles = np.arange(0.0, 90.01, 0.1)

    def pattern(self, stretch=0.0):
        x = 20 * (1 + stretch) * np.sin(np.radians(self.angles)) + 1e-9
        return 100 * np.sinc(x / np.pi) * np.exp(0.3j * x)

    def test_policy_carries_the_level_floors(self):
        for target in ('standard', 'tight'):
            policy = accuracy_target_policy(target)
            self.assertEqual(policy['db_floor_relative'], 3e-2)
            self.assertEqual(policy['phase_floor_relative'], 3e-2)
        self.assertEqual(PRODUCTION_MESH_CONVERGENCE_DEFAULTS['db_floor_relative'], 3e-2)
        with self.assertRaisesRegex(ValueError, 'db_floor_relative'):
            gate([], [], db_floor_relative=1.0)

    def test_converged_pattern_with_deep_nulls_passes(self):
        truth = self.pattern()
        self.assertLess(20 * np.log10(np.min(np.abs(truth)) / 100), -70)
        rng = np.random.default_rng(0)
        for change in (1e-4, 1e-3):
            perturbed = truth + change * 100 * np.exp(2j * np.pi * rng.random(truth.size))
            base, fine = rows(3.0, self.angles, truth, self.k), rows(3.0, self.angles, perturbed, self.k)
            for target in ('standard', 'tight'):
                result = gate(base, fine, target)
                self.assertTrue(result['passed'], (change, target, result['violations']))
            # Unfloored dB swings at the nulls are still reported for information.
            self.assertGreater(result['max_abs_db_unfloored'], result['max_abs_db'])
            # Without the floor the same converged result fails (the audited defect).
            self.assertFalse(gate(base, fine, 'tight', db_floor_relative=0.0, phase_floor_relative=1e-6)['passed'])

    def test_unconverged_patterns_still_fail(self):
        truth = self.pattern()
        base = rows(3.0, self.angles, truth, self.k)
        # A 2% electrical-size error keeps its complex change below the standard
        # limit; the floored dB check above the floor must reject it.
        stretched = gate(rows(3.0, self.angles, self.pattern(0.02), self.k), base)
        self.assertFalse(stretched['passed'])
        self.assertTrue(any(v.startswith('Max |dB|') for v in stretched['violations']))
        self.assertLess(stretched['complex_max_normalized'], 0.05)
        # A 1% electrical-size error fails the tight target.
        self.assertFalse(gate(rows(3.0, self.angles, self.pattern(0.01), self.k), base, 'tight')['passed'])
        # A large random complex change fails through the primary complex metric.
        rng = np.random.default_rng(1)
        noisy = truth + 0.08 * 100 * np.exp(2j * np.pi * rng.random(truth.size))
        result = gate(rows(3.0, self.angles, noisy, self.k), base)
        self.assertFalse(result['passed'])
        self.assertTrue(any('complex-field' in v for v in result['violations']))
        # A phase rotation of the main lobe fails the phase check.
        rotated = gate(rows(3.0, self.angles, truth * np.exp(0.5j), self.k), base)
        self.assertFalse(rotated['passed'])
        self.assertTrue(any('phase' in v for v in rotated['violations']))

    def test_real_plate_with_deep_nulls_certifies_tight_and_coarse_mesh_fails(self):
        angles = list(np.arange(0.0, 90.01, 0.25))
        tight = accuracy_target_policy('tight')
        result = rcs.solve_monostatic_rcs_2d_certified(
            plate(), [3.0], angles, geometry_units='meters', mesh_convergence_policy=tight,
            solver_method='auto', execution_options=GLOBAL_DENSE)
        vv = result['metadata']['mesh_convergence']['channels']['VV']
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertGreater(vv['max_abs_db_unfloored'], tight['max_abs_limit_db'])
        amplitude = np.abs(fields(result, 'VV'))
        self.assertLess(20 * np.log10(np.min(amplitude) / np.max(amplitude)), -60)
        with self.assertRaisesRegex(ValueError, 'Certified 2-D mesh convergence failed'):
            rcs.solve_monostatic_rcs_2d_certified(
                plate('-4'), [3.0], angles, geometry_units='meters', mesh_convergence_policy=tight,
                solver_method='auto', execution_options=GLOBAL_DENSE)


class KernelTablePathTests(unittest.TestCase):
    """Fix 3/10: direct, mixed-precision and density paths use the kernel tables."""

    angles = [0.0, 37.0, 90.0, 180.0]

    def test_direct_uses_tables_and_matches_experimental(self):
        body = mixed_body()
        direct = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters',
                                             solver_method='direct')
        experimental = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters',
                                                   solver_method='experimental_cpu')
        report = direct['metadata']['cpu_kernel_execution']
        self.assertTrue(report['kernel_tables'] and all(t['used'] for t in report['kernel_tables']))
        self.assertEqual((report['scope'], report['precision']), ('kernel_tables', 'double'))
        # Labels and the diagnostic defaults of the plain direct API are unchanged.
        self.assertEqual(direct['metadata']['solver_method'], 'dense_lu')
        self.assertNotIn('experimental_cpu', direct['metadata'])
        self.assertNotIn('execution_threads', direct['metadata'])
        self.assertIsNone(cpu.current_state())
        for pol in ('VV', 'HH'):
            self.assertLess(relative(fields(direct, pol), fields(experimental, pol)), 1e-12)

    def test_direct_tables_match_the_reference_hankel_path(self):
        from ghost_backend.twod.assembly import kernels
        import ghost_backend.twod.operators as ops
        body = mixed_body()
        tables = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters',
                                             solver_method='direct')
        with mock.patch.object(kernels, 'select_far_kernels', side_effect=lambda mesh, k, g, h, **kw: (g, h)), \
                mock.patch.object(ops, '_NATIVE_FAR', False):
            reference = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters',
                                                    solver_method='direct')
        for pol in ('VV', 'HH'):
            self.assertLess(relative(fields(tables, pol), fields(reference, pol)), 1e-10)

    def test_mixed_precision_keeps_mixed_lu_with_tables(self):
        body = mixed_body()
        double = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters')
        with linear_precision('mixed'):
            mixed = rcs.solve_monostatic_rcs_2d(body, [3.0], self.angles, geometry_units='meters')
        report = mixed['metadata']['cpu_kernel_execution']
        self.assertEqual(report['precision'], 'mixed')
        self.assertTrue(all(t['used'] for t in report['kernel_tables']))
        for pol in ('VV', 'HH'):
            channel = mixed['metadata']['channel_metadata'][pol]
            self.assertEqual(channel['linear_backend'], 'cpu_mixed_lu')
            self.assertEqual(channel['dense_mixed_precision_solve_count'], 1)
            self.assertLess(relative(fields(mixed, pol), fields(double, pol)), 1e-10)
        self.assertIsNone(cpu.current_state())

    def test_bistatic_mixed_precision_keeps_mixed_lu_with_tables(self):
        body = mixed_body()
        args = (body, [3.0], [0.0, 40.0], [0.0, 90.0, 200.0])
        double = rcs.solve_bistatic_rcs_2d(*args, geometry_units='meters')
        with linear_precision('mixed'):
            mixed = rcs.solve_bistatic_rcs_2d(*args, geometry_units='meters')
        self.assertEqual(mixed['metadata']['cpu_kernel_execution']['precision'], 'mixed')
        self.assertTrue(all(t['used'] for t in mixed['metadata']['cpu_kernel_execution']['kernel_tables']))
        for pol in ('VV', 'HH'):
            self.assertEqual(mixed['metadata']['channel_metadata'][pol]['linear_backend'], 'cpu_mixed_lu')
            self.assertLess(relative(fields(mixed, pol), fields(double, pol)), 1e-10)
        self.assertIsNone(cpu.current_state())

    def test_gpu_or_auto_dense_backend_keeps_its_path(self):
        with mock.patch.dict(os.environ, {'GHOST_DENSE_BACKEND': 'auto'}):
            result = rcs.solve_monostatic_rcs_2d(mixed_body(), [3.0], [0.0], geometry_units='meters',
                                                 solver_method='direct')
        self.assertNotIn('cpu_kernel_execution', result['metadata'])

    def test_boundary_densities_use_tables_and_match(self):
        from ghost_backend.twod.assembly import kernels
        import ghost_backend.twod.operators as ops
        body = mixed_body()
        for pol in ('TE', 'TM'):
            actual = rcs.compute_boundary_densities(body, 3.0, 31.0, pol, geometry_units='meters')
            self.assertTrue(all(t['used'] for t in actual['metadata']['cpu_kernel_execution']['kernel_tables']))
            with mock.patch.object(kernels, 'select_far_kernels', side_effect=lambda mesh, k, g, h, **kw: (g, h)), \
                    mock.patch.object(ops, '_NATIVE_FAR', False):
                expected = rcs.compute_boundary_densities(body, 3.0, 31.0, pol, geometry_units='meters')
            value = lambda r: np.asarray(r['density_real']) + 1j * np.asarray(r['density_imag'])
            self.assertLess(relative(value(actual), value(expected)), 1e-10)
        self.assertIsNone(cpu.current_state())

    def test_plain_api_uses_host_threads_like_the_automatic_profile(self):
        # October 2026: a bare automatic request resolves its threads on the
        # host (physical cores), as the GUI and batch profiles do; an explicit
        # GHOST_ASSEMBLY_THREADS launch override still pins the count.
        from ghost_backend.execution.options import host_assembly_threads
        with mock.patch.dict(os.environ, {'GHOST_ASSEMBLY_THREADS': ''}):
            result = rcs.solve_monostatic_rcs_2d(mixed_body(), [3.0], [0.0], geometry_units='meters')
        self.assertEqual(result['metadata']['execution_threads']['assembly'], host_assembly_threads())
        self.assertEqual(result['metadata']['solver_method_requested'], 'auto')
        with mock.patch.dict(os.environ, {'GHOST_ASSEMBLY_THREADS': '1'}):
            pinned = rcs.solve_monostatic_rcs_2d(mixed_body(), [3.0], [0.0], geometry_units='meters')
        self.assertEqual(pinned['metadata']['execution_threads']['assembly'], 1)


class ConditionEstimateTests(unittest.TestCase):
    """Fix 4: the estimate is a function of the matrix only."""

    def test_estimate_ignores_and_preserves_global_rng(self):
        from scipy import linalg
        rng = np.random.default_rng(5)
        matrix = rng.standard_normal((80, 80)) + 1j * rng.standard_normal((80, 80))
        matrix[:, 3] = matrix[:, 7] + 1e-4 * matrix[:, 3]
        lu, piv = linalg.lu_factor(matrix)
        estimates = []
        for seed in range(6):
            np.random.seed(seed)
            before = np.random.get_state()
            estimates.append(rcs._equilibrated_condition_from_lu(matrix, lu, piv))
            after = np.random.get_state()
            np.testing.assert_array_equal(after[1], before[1])
            self.assertEqual(after[2], before[2])
        self.assertEqual(len(set(estimates)), 1)

    def test_lossless_dielectric_gate_value_is_seed_independent(self):
        radius = 0.07
        frequency = 2.66572 * rcs.C0 / (2 * math.pi * radius) / 1e9
        geometry = dict(segments=[circle(radius, 104, 3, material=1)], ibcs=[],
                        dielectrics=[['1', '4', '0', '1', '0']])
        values = set()
        for seed in (0, 1, 2, 5):
            np.random.seed(seed)
            result = rcs.solve_monostatic_rcs_2d_single_polarization(
                geometry, [frequency], [0.0], 'TM', geometry_units='meters', solver_method='direct',
                strict_quality_gate=False, compute_condition_number=True)
            values.add(result['metadata']['condition_est_max'])
        self.assertEqual(len(values), 1)


class MergedMetadataTests(unittest.TestCase):
    """Fix 5: merged maxima/minima propagate NaN and infinity."""

    def test_channel_merge_propagates_nonfinite(self):
        channels = dict(VV=dict(condition_est_max=12.0, residual_norm_max=1e-14, panel_count=10),
                        HH=dict(condition_est_max=float('nan'), residual_norm_max=float('inf')))
        self.assertTrue(math.isnan(rcs._finite_metadata_max(channels, 'condition_est_max')))
        self.assertEqual(rcs._finite_metadata_max(channels, 'residual_norm_max'), float('inf'))
        self.assertEqual(rcs._finite_metadata_max(channels, 'panel_count'), 10.0)  # absent in HH: skipped
        self.assertEqual(rcs._finite_metadata_max(channels, 'missing', default=3.0), 3.0)

    def test_frequency_merge_propagates_nonfinite(self):
        def one(frequency, condition, residual_min):
            sample = dict(frequency_ghz=frequency, theta_inc_deg=0.0, theta_scat_deg=0.0, polarization='VV')
            return dict(samples=[sample], co_solved_samples=dict(VV=[sample], HH=[]),
                        metadata=dict(condition_est_max=condition, residual_norm_min=residual_min,
                                      condition_est_mean=condition, warnings=['w']))
        merged = rcs._merge_frequency_results(iter([one(2.0, 5.0, 1e-15), one(1.0, float('nan'), float('-inf'))]),
                                              [2.0, 1.0])['metadata']
        self.assertTrue(math.isnan(merged['condition_est_max']))
        self.assertTrue(math.isnan(merged['condition_est_mean']))
        self.assertEqual(merged['residual_norm_min'], float('-inf'))
        self.assertEqual((merged['warnings'], merged['warning_count']), (['w'], 1))


class RequestOrderTests(unittest.TestCase):
    """Fix 7: samples follow the request order (no consumer relies on sorting)."""

    def test_monostatic_and_bistatic_samples_keep_request_order(self):
        geometry = dict(segments=[circle(0.05, 48)], ibcs=[], dielectrics=[])
        angles, frequencies = [90.0, 0.0, 45.0, -30.0], [0.9, 0.6]
        result = rcs.solve_monostatic_rcs_2d(geometry, frequencies, angles, geometry_units='meters',
                                             solver_method='direct')
        expected = [(f, a) for f in frequencies for a in angles]
        for pol in ('VV', 'HH'):
            self.assertEqual([(r['frequency_ghz'], r['theta_inc_deg']) for r in result['co_solved_samples'][pol]],
                             expected)
        self.assertEqual([(r['frequency_ghz'], r['theta_inc_deg'], r['polarization']) for r in result['samples']],
                         [(f, a, p) for f, a in expected for p in ('VV', 'HH')])
        bistatic = rcs.solve_bistatic_rcs_2d(geometry, [0.6], [40.0, 0.0], [180.0, 90.0, 270.0],
                                             geometry_units='meters')
        self.assertEqual([(r['theta_inc_deg'], r['theta_scat_deg']) for r in bistatic['co_solved_samples']['HH']],
                         [(i, o) for i in (40.0, 0.0) for o in (180.0, 90.0, 270.0)])


class MemoryDetectionTests(unittest.TestCase):
    """Fix 6: cgroup/SLURM edge cases and GiB labels."""

    def test_cgroup_limit_with_unreadable_usage_uses_own_rss(self):
        with mock.patch.object(rcs, '_read_cgroup_int', side_effect=[4 * GIB, None]), \
                mock.patch.object(rcs, '_process_rss_bytes', return_value=GIB):
            self.assertEqual(rcs._cgroup_available_bytes(), 3 * GIB)

    def test_slurm_subtracts_every_local_process_of_the_allocation(self):
        environ = {1: dict(SLURM_JOB_ID='7', SLURM_STEP_ID='0', SLURM_PROCID='0'),
                   2: dict(SLURM_JOB_ID='7', SLURM_STEP_ID='0', SLURM_PROCID='1'),
                   3: dict(SLURM_JOB_ID='8'), 4: None}
        rss = {1: GIB, 2: 2 * GIB, 3: 4 * GIB, 4: 8 * GIB}
        base = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        common = [mock.patch.object(rcs, '_process_ids', return_value=[1, 2, 3, 4, os.getpid()]),
                  mock.patch.object(rcs, '_environ_of', side_effect=lambda pid: environ.get(pid)),
                  mock.patch.object(rcs, '_rss_of', side_effect=lambda pid: rss[pid]),
                  mock.patch.object(rcs, '_process_rss_bytes', return_value=GIB)]
        for patch in common:
            patch.start()
        try:
            node = dict(base, SLURM_MEM_PER_NODE='16384', SLURM_JOB_ID='7', SLURM_STEP_ID='0', SLURM_PROCID='0')
            with mock.patch.dict(os.environ, node, clear=True):
                # Job share of the node: this process plus both tasks of job 7.
                self.assertEqual(rcs._slurm_available_bytes(), (16 - 1 - 1 - 2) * GIB)
            task = dict(base, SLURM_MEM_PER_CPU='4096', SLURM_CPUS_PER_TASK='2', SLURM_JOB_ID='7',
                        SLURM_STEP_ID='0', SLURM_PROCID='0')
            with mock.patch.dict(os.environ, task, clear=True):
                # Per-task share: only processes of this task (procid 0) count.
                self.assertEqual(rcs._slurm_available_bytes(), (8 - 1 - 1) * GIB)
        finally:
            for patch in common:
                patch.stop()

    def test_two_d_memory_messages_are_labelled_gib(self):
        with mock.patch.object(rcs, '_detect_available_gb', return_value=3.25):
            message = rcs._memory_gate_message(7.5, 2.925, 'The test solve', unit='GiB')
            legacy = rcs._memory_gate_message(7.5, 2.925, 'The test solve')
        self.assertIn('requires an estimated 7.50 GiB', message)
        self.assertIn('3.25 GiB is currently available', message)
        self.assertIn('requires an estimated 7.50 GB', legacy)
        with mock.patch.object(rcs, '_solve_memory_limit_gb', return_value=1e-9):
            with self.assertRaisesRegex(MemoryError, r'GiB'):
                rcs.solve_monostatic_rcs_2d(dict(segments=[circle(0.05, 24)], ibcs=[], dielectrics=[]), [0.6],
                                            [0.0], geometry_units='meters', solver_method='direct')


class AutomaticThreadTests(unittest.TestCase):
    """Fix 8: the automatic profile resolves threads on the executing host."""

    def test_profile_is_host_independent_and_resolves_per_host(self):
        profile = options.efficient_defaults()
        self.assertEqual(profile['assembly_threads'], 'auto')
        self.assertEqual(options.automatic_options()['assembly_threads'], 'auto')
        expected = max(1, min(options.physical_core_count(), os.cpu_count() or 1,
                              options.AUTOMATIC_ASSEMBLY_THREAD_LIMIT))
        with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '', 'SLURM_CPUS_ON_NODE': ''}):
            with options.execution_scope(profile):
                self.assertEqual(options.effective_assembly_threads(), expected)
            # A scheduler/driver allocation bounds it, so batch units never
            # oversubscribe; a larger allocation still uses physical cores only.
            with options.execution_scope(profile, assembly_threads=3):
                self.assertEqual(options.effective_assembly_threads(), min(3, expected))
            with options.execution_scope(profile, assembly_threads=4 * (os.cpu_count() or 1)):
                self.assertEqual(options.effective_assembly_threads(), expected)
            with options.execution_scope(dict(profile, assembly_threads=6), assembly_threads=2):
                self.assertEqual(options.effective_assembly_threads(), 2)
        with mock.patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}):
            with options.execution_scope(profile):
                self.assertEqual(options.effective_assembly_threads(), 1)
        # The plain API's diagnostic defaults are unchanged.
        self.assertEqual(options.DEFAULTS['assembly_threads'], 1)


class OperatorCacheKeyTests(unittest.TestCase):
    """Fix 9: cached operators are keyed by the far-quadrature and tile settings."""

    def test_cache_distinguishes_assembly_rules(self):
        calls = []

        @cpu.cached_operator('D')
        def operator(mesh, k0, order=8):
            calls.append(options.current_options())
            return np.full((3, 3), float(len(calls)), dtype=complex)

        state = cpu.CPUState()
        from ghost_backend.twod.assembly import kernels
        with mock.patch.object(kernels, 'mesh_key', return_value=b'mesh'), cpu._STATE.override(state):
            with options.execution_scope({}):
                first = operator(object(), 2.0)
                self.assertIs(operator(object(), 2.0), first)
            for changed in (dict(far_quadrature_order=12), dict(far_grading=False), dict(assembly_tile=64)):
                with options.execution_scope(changed):
                    self.assertIsNot(operator(object(), 2.0), first)
        self.assertEqual(len(calls), 4)
        self.assertEqual(state.cache_stats['hits'], 1)


if __name__ == '__main__':
    unittest.main()
