"""BoR compact rows preserve channel order, diagnostics, certification and exports."""
import copy
from contextlib import nullcontext
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor.samples import channel_buffers, finish_channels
from ghost_backend.bor.options import estimate_output_gb
from ghost_backend.twod.samples import (
    BOR_FIELDS, SampleTable, SampleSelection, compact_samples, sample_columns,
    sample_column,
)
from ghost_backend.bor import dispatch
from ghost_backend.io import grim
from ghost_backend.twod.checkpoints import FrequencyCheckpoints
from test_bor_physics_regression import _pec_sphere_snapshot


def rows(compact, expand=True):
    aspects = [90., 0., 180., 90.]
    with compact_samples() if compact else nullcontext():
        channels = channel_buffers(2, aspects, expand)
    for frequency in (2., 1.):
        for channel in ('VV', 'HH'):
            for index, angle in enumerate(aspects):
                row = {key: float(index + 1) for key in BOR_FIELDS}
                row.update(frequency_ghz=frequency, theta_inc_deg=angle, theta_scat_deg=angle,
                           linear_backward_error=1e-15, rcs_amp_real=1e-10 * index)
                channels[channel].append(row)
    return channels, finish_channels(channels, expand)


class BorCompactSampleTests(unittest.TestCase):
    def test_legacy_order_duplicate_inputs_and_mirroring_are_preserved(self):
        for expand in (False, True):
            expected_channels, expected = rows(False, expand)
            channels, actual = rows(True, expand)
            self.assertIsInstance(actual, SampleSelection)
            self.assertEqual(list(actual), expected)
            for channel in channels:
                self.assertEqual(list(channels[channel]), expected_channels[channel])
            for cloned in (copy.deepcopy((channels, actual)), pickle.loads(pickle.dumps((channels, actual)))):
                self.assertEqual(list(cloned[1]), expected)
            actual[0]['rcs_amp_real'] = 99.
            self.assertEqual(actual[0], expected[0])

    def test_columns_and_certification_views_keep_backward_errors_and_shared_data(self):
        channels, actual = rows(True)
        with mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('expanded rows')):
            columns = sample_columns(actual)
            np.testing.assert_array_equal(columns['linear_backward_error'], 1e-15)
            self.assertEqual(len(columns), 11)
            self.assertIs(dispatch._bor_channel_result({'co_solved_samples': channels}, 'VV')['samples'], channels['VV'])
            np.testing.assert_array_equal(sample_column(actual, 'polarization'), columns['polarization'])
        # Both labeled combined views point into the existing per-channel stores.
        for labeled in actual.rows.chunks:
            self.assertIs(labeled.rows, channels[labeled.labels['polarization']])
        table = channels['VV'].rows
        self.assertEqual(table.data.dtype, np.float64)
        self.assertEqual(table.nbytes, len(table) * 10 * 8)

    def test_checkpoint_roundtrip_preserves_bor_schema(self):
        channels, actual = rows(True)
        actual = SampleSelection(actual, np.flatnonzero(sample_column(actual, 'frequency_ghz') == 1.))
        with tempfile.TemporaryDirectory() as directory, compact_samples():
            store = FrequencyCheckpoints(directory, 'bor-compact-schema', False)
            store.save(1., {'solver': 'bor_mom_rcs', 'samples': actual, 'co_solved_samples': channels, 'metadata': {}})
            restored = store.load(1.)
            self.assertIsInstance(restored['samples'], SampleTable)
            self.assertEqual(list(restored['samples']), list(actual))
            np.testing.assert_array_equal(sample_column(restored['samples'], 'linear_backward_error'), 1e-15)
            for pol in ('VV', 'HH'):
                self.assertEqual(list(restored['co_solved_samples'][pol]),
                                 [row for row in channels[pol] if row['frequency_ghz'] == 1.])

    def test_output_reservation_accounts_for_compact_sorting_and_certification(self):
        legacy = estimate_output_gb(10, 100, True, True)
        with compact_samples():
            compact = estimate_output_gb(10, 100, True, True)
        self.assertEqual(compact * 8, legacy)
        self.assertGreater(compact * 1e9, 10 * 100 * 2 * 2 * 2 * 80)

    def test_real_bor_dispatch_and_grim_export_match_lists(self):
        kwargs = dict(geometry_snapshot=_pec_sphere_snapshot(explicit_elements=-20),
                      frequencies_ghz=[.6], elevations_deg=[90., 0., 180.],
                      geometry_units='meters', workers=1, expand_to_360=True,
                      bor_options={'factorization': 'dense'})
        ordinary = dispatch.solve_monostatic_rcs_bor_survey(**kwargs)
        with compact_samples():
            compact = dispatch.solve_monostatic_rcs_bor_survey(**kwargs)
        self.assertEqual(list(compact['samples']), ordinary['samples'])
        for pol in ('VV', 'HH'):
            self.assertEqual(list(compact['co_solved_samples'][pol]), ordinary['co_solved_samples'][pol])
        self.assertEqual(compact['metadata']['quality_gate'], ordinary['metadata']['quality_gate'])
        from ghost_backend.assembly.fields import bodies_from_bor_solver_result
        expected_bodies = bodies_from_bor_solver_result(ordinary)
        with mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('expanded rows')):
            actual_bodies = bodies_from_bor_solver_result(compact)
        for frequency in expected_bodies:
            for key in expected_bodies[frequency]:
                np.testing.assert_array_equal(actual_bodies[frequency][key], expected_bodies[frequency][key])
        # Timings differ between runs; serialize the same metadata to isolate storage.
        compact['metadata'] = ordinary['metadata']
        with tempfile.TemporaryDirectory() as directory:
            before = grim.export_result_to_grim(ordinary, str(Path(directory) / 'list'))
            after = grim.export_result_to_grim(compact, str(Path(directory) / 'compact'))
            for a, b in zip(before, after):
                with np.load(a, allow_pickle=False) as expected, np.load(b, allow_pickle=False) as actual:
                    self.assertEqual(actual.files, expected.files)
                    for key in actual.files:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
