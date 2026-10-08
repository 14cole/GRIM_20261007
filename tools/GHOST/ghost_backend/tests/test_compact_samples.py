"""Compact application results preserve all solver values and export contracts."""
import sys
import copy
import json
import pickle
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver
from ghost_backend.twod.samples import (FIELDS, SampleTable, SampleChunks,
    SampleSelection, compact_samples, sample_buffer, merge_tables, sample_column,
    sorted_samples)
from ghost_backend.twod.checkpoints import FrequencyCheckpoints, run_checkpointed
from ghost_backend.io import grim
from ghost_backend.io.grim import _build_grid_for_co_solved_samples
from test_experimental_cpu import fixture


def table(coordinates):
    with compact_samples():
        result = sample_buffer(len(coordinates))
    for frequency, incidence, observation in coordinates:
        row = dict.fromkeys(FIELDS, 1.)
        row.update(frequency_ghz=frequency, theta_inc_deg=incidence, theta_scat_deg=observation)
        result.append(row)
    return result


class CompactSampleTests(unittest.TestCase):
    def test_copy_and_pickle_preserve_values_and_shared_channel_storage(self):
        channels, rows = merge_tables(dict(VV=dict(samples=table([(1., 0., 0.), (1., 90., 90.)])),
                                          HH=dict(samples=table([(1., 90., 90.), (1., 0., 0.)]))))
        result = dict(samples=rows, co_solved_samples=channels)
        for cloned in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
            self.assertEqual(list(cloned['samples']), list(rows))
            self.assertIs(cloned['samples'].first, cloned['co_solved_samples']['VV'])
            self.assertIs(cloned['samples'].second, cloned['co_solved_samples']['HH'])
            self.assertFalse(np.shares_memory(cloned['samples'].first.data, rows.first.data))
        shallow = copy.copy(rows)
        self.assertIs(shallow.first.data, rows.first.data)

    def test_column_sort_and_incidence_views_do_not_expand_rows(self):
        rows = table([(2., 90., 20.), (1., 0., 20.), (2., 90., 0.)])
        expected = sorted(list(rows), key=lambda row: tuple(row[k] for k in FIELDS[:3]))
        with mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('rows expanded')):
            ordered = sorted_samples(rows)
            grouped = grim._group_incidence_samples(rows)
            np.testing.assert_array_equal(sample_column(ordered, 'theta_scat_deg'), [20., 0., 20.])
        self.assertIsInstance(ordered, SampleSelection)
        self.assertEqual(list(ordered), expected)
        self.assertIs(grouped[90.].rows, rows)
        self.assertEqual(list(grouped[90.]), [rows[0], rows[2]])

    def test_storage_is_lazy_bounded_float64_and_opt_in(self):
        self.assertIsInstance(sample_buffer(3), list)
        with compact_samples():
            rows = sample_buffer(20000)
        self.assertEqual(rows.nbytes, 0)
        rows.append(dict.fromkeys(FIELDS, 1.))
        self.assertEqual(rows.nbytes, 20000 * 9 * 8)
        rows[0]['rcs_amp_real'] = 5.
        self.assertEqual(rows[0]['rcs_amp_real'], 1.)
        self.assertIsInstance(sample_buffer(3), list)

    def test_pair_validation_and_arbitrary_order(self):
        first = table([(2., 90., 0.), (1., 0., 20.), (2., 90., 20.)])
        second = table([(2., 90., 20.), (2., 90., 0.), (1., 0., 20.)])
        channels, rows = merge_tables(dict(VV=dict(samples=first), HH=dict(samples=second)))
        self.assertIs(channels['VV'].data, first.data)
        self.assertEqual([r['polarization'] for r in rows], ['VV', 'HH'] * 3)
        for i in range(0, len(rows), 2):
            self.assertEqual(tuple(rows[i][k] for k in FIELDS[:3]), tuple(rows[i+1][k] for k in FIELDS[:3]))
        self.assertEqual(rows[-1], rows[5])
        self.assertEqual(rows[1:4], list(rows)[1:4])
        with self.assertRaises(IndexError):
            rows[6]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            duplicate = table([(1., 0., 0.), (1., 0., 0.)])
            merge_tables(dict(VV=dict(samples=duplicate), HH=dict(samples=duplicate)))
        with self.assertRaisesRegex(ValueError, 'same physical grid'):
            merge_tables(dict(VV=dict(samples=first), HH=dict(samples=table([(1., 0., 0.)]))))

    def test_frequency_chunks_retain_arrays_without_row_expansion(self):
        chunks = SampleChunks()
        a, b = table([(1., 0., 0.)]), table([(2., 0., 0.), (3., 0., 0.)])
        chunks.extend(a)
        chunks.extend([])
        chunks.extend(b)
        self.assertIs(chunks.chunks[0], a)
        self.assertEqual([row['frequency_ghz'] for row in chunks], [1., 2., 3.])
        self.assertEqual(chunks[-1], b[1])
        self.assertEqual(chunks[::-1], list(chunks)[::-1])

    def test_real_sweeps_match_lists_including_certification_and_export(self):
        geometry = fixture('pec', 24)
        cases = [
            (solver.solve_monostatic_rcs_2d, dict(frequencies_ghz=[.8, .6], elevations_deg=[90., 0., -30.], solver_method='direct')),
            (solver.solve_bistatic_rcs_2d, dict(frequencies_ghz=[.6], incidence_angles_deg=[40., 0.], observation_angles_deg=[180., 90., 270.])),
            (solver.solve_monostatic_rcs_2d_certified, dict(frequencies_ghz=[.6], elevations_deg=[90., 0., -30.], solver_method='direct')),
        ]
        for solve, arguments in cases:
            with self.subTest(solver=solve.__name__):
                kwargs = dict(geometry_snapshot=geometry, geometry_units='meters', **arguments)
                ordinary = solve(**kwargs)
                with compact_samples():
                    packed = solve(**kwargs)
                self.assertNotIsInstance(packed['samples'], list)
                self.assertEqual(list(packed['samples']), ordinary['samples'])
                for pol in ('VV', 'HH'):
                    self.assertEqual(list(packed['co_solved_samples'][pol]), ordinary['co_solved_samples'][pol])
                if 'bistatic' not in solve.__name__:
                    expected = _build_grid_for_co_solved_samples(ordinary['co_solved_samples'], ordinary['polarization_mapping'])
                    actual = _build_grid_for_co_solved_samples(packed['co_solved_samples'], packed['polarization_mapping'])
                    for key in expected:
                        if isinstance(expected[key], np.ndarray):
                            np.testing.assert_array_equal(actual[key], expected[key])
                        else:
                            self.assertEqual(actual[key], expected[key])
                with tempfile.TemporaryDirectory() as directory:
                    store = FrequencyCheckpoints(directory, 'compact-checkpoint', False)
                    # Checkpoint contract is one frequency at a time.
                    if len(arguments['frequencies_ghz']) == 1:
                        store.save(.6, packed)
                        loaded = store.load(.6)
                        self.assertIsNotNone(loaded)
                        self.assertEqual(loaded['samples'], list(packed['samples']))

    def test_compact_checkpoint_reload_preserves_shared_columns_and_extensions(self):
        coordinates = [(1., 40., 90.), (1., 0., 180.)]
        channels, rows = merge_tables(dict(VV=dict(samples=table(coordinates)), HH=dict(samples=table(coordinates[::-1]))))
        result = dict(samples=rows, co_solved_samples=channels, metadata={'mesh_convergence_certified': True})
        with tempfile.TemporaryDirectory() as directory:
            store = FrequencyCheckpoints(directory, 'columns', True)
            store.save(1., result)
            with compact_samples(), mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('rows expanded')):
                loaded = store.load(1.)
            self.assertIsInstance(loaded['samples'], SampleTable)
            self.assertEqual(list(loaded['samples']), list(rows))
            for channel in ('VV', 'HH'):
                self.assertIs(loaded['co_solved_samples'][channel].rows, loaded['samples'])
                self.assertEqual(list(loaded['co_solved_samples'][channel]),
                                 [row for row in rows if row['polarization'] == channel])
            self.assertIsInstance(store.load(1.)['samples'], list)
            extended = dict(result, samples=list(rows))
            extended['samples'][0]['extension'] = dict(a=[1, 2])
            store.save(1., extended)
            with compact_samples():
                loaded = store.load(1.)
            self.assertIsInstance(loaded['samples'], list)
            self.assertEqual(loaded['samples'], extended['samples'])

    def test_interrupted_sweep_resumes_with_compact_frequency_views(self):
        arguments = dict(geometry_snapshot=fixture('pec', 24), frequencies_ghz=[.8, .6],
                         elevations_deg=[90., 0.], geometry_units='meters', solver_method='direct')
        original = solver.solve_monostatic_rcs_2d
        def interrupt_second(**kwargs):
            if kwargs['frequencies_ghz'] == [.6]:
                raise InterruptedError('test cancellation')
            return original(**kwargs)
        with tempfile.TemporaryDirectory() as directory, compact_samples(), \
                mock.patch('ghost_backend.twod.checkpoints.input_identity', return_value='resume-test'):
            with self.assertRaises(InterruptedError):
                run_checkpointed(interrupt_second, arguments, directory, {}, 'double', False)
            with mock.patch.object(solver, 'solve_monostatic_rcs_2d', wraps=original) as solve:
                resumed = run_checkpointed(solve, arguments, directory, {}, 'double', False)
                self.assertEqual(solve.call_count, 1)
                self.assertEqual(solve.call_args.kwargs['frequencies_ghz'], [.6])
            expected = original(**arguments)
            self.assertIsInstance(resumed['samples'], SampleChunks)
            self.assertEqual(list(resumed['samples']), list(expected['samples']))
            self.assertEqual(resumed['metadata']['frequency_checkpoints']['reused'], 1)
            for channel in ('VV', 'HH'):
                self.assertEqual(list(resumed['co_solved_samples'][channel]), list(expected['co_solved_samples'][channel]))

    def test_gui_scope_covers_both_solvers_and_restores_after_exception(self):
        from ghost_backend.ui import solver as ui
        worker = ui._SolveWorker(fixture('pec', 24), '', '', [.6], [0., 90.], 'meters', {},
                                 mesh_certification=False)
        def compact_route(*args):
            self.assertIsInstance(sample_buffer(1), SampleTable)
            raise InterruptedError('scope test')
        with mock.patch.object(worker, '_run_2d_with_precision', side_effect=compact_route):
            with self.assertRaises(InterruptedError):
                worker._run_2d(worker.snapshot, None)
        self.assertIsInstance(sample_buffer(1), list)
        def bor_route(**kwargs):
            self.assertIsInstance(sample_buffer(1), SampleTable)
            return {'samples': [{'frequency_ghz': .6}], 'solver': 'bor_mom_rcs'}
        with mock.patch.object(ui, 'solve_monostatic_rcs_bor_survey', side_effect=bor_route):
            self.assertIsInstance(worker._run_bor()['samples'], list)
        self.assertIsInstance(sample_buffer(1), list)

    def test_all_exports_and_metadata_match_list_results_exactly(self):
        with compact_samples():
            packed = solver.solve_bistatic_rcs_2d(geometry_snapshot=fixture('pec', 24), geometry_units='meters',
                frequencies_ghz=[.6], incidence_angles_deg=[40., 0.], observation_angles_deg=[180., 90., 270.])
            mono = solver.solve_monostatic_rcs_2d(geometry_snapshot=fixture('pec', 24), geometry_units='meters',
                frequencies_ghz=[.6], elevations_deg=[180., 90., 270.])
        for dual in (True, False):
            for mode in ('monostatic', 'bistatic'):
                with self.subTest(dual=dual, mode=mode), tempfile.TemporaryDirectory() as directory:
                    compact = mono if mode == 'monostatic' else packed
                    ordinary = dict(compact, samples=list(compact['samples']),
                                    co_solved_samples={key: list(rows) for key, rows in compact['co_solved_samples'].items()})
                    self.assertEqual(grim._solver_metadata_json(compact), grim._solver_metadata_json(ordinary))
                    pair = [dict(item) for item in (ordinary, compact)]
                    if not dual:
                        for result in pair:
                            result['samples'] = result.pop('co_solved_samples')['VV']
                            result.pop('polarizations')
                            result.pop('polarization_mapping')
                            result['polarization'] = 'TE'
                    outputs = []
                    for index, result in enumerate(pair):
                        with mock.patch.object(grim, '_solver_metadata_json', wraps=grim._solver_metadata_json) as metadata:
                            files = grim.export_result_to_grim(result, str(Path(directory) / ('result' + str(index))))
                            self.assertEqual(metadata.call_count, 1)
                        outputs.append(files)
                    for left, right in zip(*outputs):
                        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
                            self.assertEqual(set(a.files), set(b.files))
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                    csv_a = grim.export_result_to_dbke_csv(pair[0], str(Path(directory) / 'a.csv'))
                    csv_b = grim.export_result_to_dbke_csv(pair[1], str(Path(directory) / 'b.csv'))
                    self.assertEqual(Path(csv_a).read_bytes(), Path(csv_b).read_bytes())

    def test_metadata_json_matches_legacy_schema_order_and_extensions(self):
        rows = [dict(frequency_ghz=2., theta_inc_deg=0., theta_scat_deg=10., linear_residual=float('inf')),
                dict(frequency_ghz=1., theta_inc_deg=90., theta_scat_deg=10., condition_est=np.float64(3.)),
                dict(ignored_extension=True)]
        result = dict(samples=rows, metadata={'complex': complex(1., 2.), 'array': np.arange(3)})
        actual = grim._solver_metadata_json(result)
        parsed = json.loads(actual)
        self.assertEqual(parsed['sample_diagnostics'], [
            dict(frequency_ghz=1., theta_inc_deg=90., theta_scat_deg=10., polarization='', condition_est=3.),
            dict(frequency_ghz=2., theta_inc_deg=0., theta_scat_deg=10., polarization='',
                 linear_residual={'__nonfinite_float__': 'infinity'})])
        self.assertEqual(actual, json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False))

    def test_bor_export_keeps_three_dimensional_normalization(self):
        row = dict(frequency_ghz=1., theta_inc_deg=20., theta_scat_deg=20., rcs_amp_real=1.,
                   rcs_amp_imag=.5, rcs_linear=4. * np.pi * 1.25, linear_backward_error=1e-14)
        result = dict(solver='bor_mom_rcs', scattering_mode='monostatic', polarization='VV',
                      rcs_linear_quantity='sigma_3d', rcs_log_unit='dBsm', samples=[row])
        with tempfile.TemporaryDirectory() as directory, compact_samples():
            files = grim.export_result_to_grim(result, str(Path(directory) / 'bor'))
            with np.load(files[0], allow_pickle=False) as stored:
                units = json.loads(stored['units'].item())
                self.assertEqual(units['rcs_linear_quantity'], 'sigma_3d')
                self.assertEqual(units['rcs_log_unit'], 'dBsm')
                np.testing.assert_allclose(stored['rcs_power'], row['rcs_linear'], rtol=1e-7)
                np.testing.assert_array_equal(stored['rcs_amp_real'], np.ones((1, 1, 1, 1)))
                metadata = json.loads(stored['solver_metadata_json'].item())
                self.assertEqual(metadata['sample_diagnostics'][0]['linear_backward_error'], 1e-14)
            with self.assertRaisesRegex(ValueError, 'only accepts 2-D'):
                grim.export_result_to_dbke_csv(result, str(Path(directory) / 'bor.csv'))
        self.assertIsInstance(result['samples'], list)


if __name__ == '__main__':
    unittest.main()
