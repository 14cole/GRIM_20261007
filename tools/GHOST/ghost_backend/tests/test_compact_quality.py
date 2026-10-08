"""Compact matching preserves rounding, validation and convergence decisions."""
import sys
from pathlib import Path
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod.samples import FIELDS, SampleTable
from ghost_backend.runs.quality import evaluate_mesh_convergence


def compare(base, fine):
    return evaluate_mesh_convergence(dict(samples=base), dict(samples=fine), 1., 3.)


class CompactQualityTests(unittest.TestCase):
    def tables(self):
        rng = np.random.default_rng(189)
        data = rng.standard_normal((40, len(FIELDS)))
        data[:, :3] = np.column_stack((np.repeat([.6, .8], 20), np.zeros(40), np.tile(np.arange(20), 2)))
        # Distinct exact points may intentionally share a rounded key.
        data[1, 2], data[2, 2] = 1.0000000001, 1.0000000002
        fine = data.copy()
        fine[:, 5:7] *= 1.001
        fine[:, 4] += .005
        return SampleTable(data[rng.permutation(40)]), SampleTable(fine[rng.permutation(40)])

    def test_numeric_path_matches_every_report_field_without_row_expansion(self):
        base, fine = self.tables()
        expected = compare(list(base), list(fine))
        with mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('expanded rows')):
            actual = compare(base, fine)
        self.assertEqual(actual, expected)

    def test_invalid_data_retains_existing_validation(self):
        for mutation, message in (
                (lambda data: data.__setitem__((0, 5), np.nan), 'non-finite'),
                (lambda data: data.__setitem__((0, 4), np.inf), 'non-finite'),
                (lambda data: data.__setitem__((0, slice(0, 3)), data[1, :3]), 'duplicate exact'),
                (lambda data: data.__setitem__((0, 0), 8.), 'sample grids differ')):
            base, fine = self.tables()
            mutation(fine.data)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                compare(base, fine)

    def test_nulls_and_decimal_rounding_match_row_path(self):
        base, fine = self.tables()
        base.data[:, 5:7] = 0.
        fine.data[:, 5:7] = 0.
        for rows in (base, fine):
            rows.data[:, 4] = -120.
            rows.data[:, 1] = 1.2345678905
        self.assertEqual(compare(base, fine), compare(list(base), list(fine)))


if __name__ == '__main__':
    unittest.main()
