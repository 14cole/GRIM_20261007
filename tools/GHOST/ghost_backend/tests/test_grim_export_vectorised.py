"""The vectorised GRIM export grid matches the historical per-row builder exactly."""
import cmath
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.io import grim
from ghost_backend.io.grim import C0, EPS, _build_grid_for_samples
from ghost_backend.twod.samples import SampleTable, compact_samples, sample_buffer


def _legacy_grid(samples, polarization, linear_quantity='sigma_2d', preserve_raw=True):
    """The row loop that built export grids before vectorisation (reference)."""
    if not samples:
        raise ValueError('No samples available to export.')
    validated_rows = np.empty((len(samples), 5), dtype=np.float64)
    for row_index, row in enumerate(samples):
        try:
            az = grim._required_finite_sample_value(row, 'theta_scat_deg')
            freq = grim._required_finite_sample_value(row, 'frequency_ghz')
            lin = grim._required_finite_sample_value(row, 'rcs_linear')
            amp_real = grim._required_finite_sample_value(row, 'rcs_amp_real')
            amp_imag = grim._required_finite_sample_value(row, 'rcs_amp_imag')
        except ValueError as exc:
            raise ValueError(f"Invalid solver sample at index {row_index}: {exc}") from exc
        if freq <= 0.0:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: frequency_ghz "
                f"must be positive; got {freq!r}.")
        if lin < 0.0:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: rcs_linear "
                f"must be non-negative; got {lin!r}.")
        with np.errstate(over="ignore", invalid="ignore"):
            amp_abs2 = amp_real * amp_real + amp_imag * amp_imag
        if not math.isfinite(amp_abs2):
            raise ValueError(
                f"Invalid solver sample at index {row_index}: the finite "
                "complex amplitude is too large to form finite physical power.")
        if linear_quantity == 'sigma_2d':
            k0 = 2.0 * math.pi * freq * 1.0e9 / C0
            expected_linear = amp_abs2 / (4.0 * k0)
        else:
            expected_linear = 4.0 * math.pi * amp_abs2
        if not math.isfinite(expected_linear):
            raise ValueError(
                f"Invalid solver sample at index {row_index}: normalized "
                "linear RCS is not finite.")
        consistency_tol = 2.0e-8 * max(expected_linear, lin) + np.finfo(float).tiny
        if abs(lin - expected_linear) > consistency_tol:
            raise ValueError(
                f"Invalid solver sample at index {row_index}: rcs_linear "
                f"{lin:.12g} is inconsistent with the complex amplitude under "
                f"{linear_quantity} normalization (expected "
                f"{expected_linear:.12g}).")
        validated_rows[row_index] = (az, freq, lin, amp_real, amp_imag)
    azimuths = np.asarray(sorted({row[0] for row in validated_rows}), dtype=float)
    frequencies = np.asarray(sorted({row[1] for row in validated_rows}), dtype=float)
    shape = (len(azimuths), 1, len(frequencies), 1)
    rcs_phase = np.full(shape, np.nan, dtype=np.float32)
    rcs_power = np.full(shape, np.nan, dtype=np.float32)
    rcs_amp_real = np.full(shape, np.nan, dtype=np.float64) if preserve_raw else None
    rcs_amp_imag = np.full(shape, np.nan, dtype=np.float64) if preserve_raw else None
    az_index = {value: i for i, value in enumerate(azimuths)}
    f_index = {value: i for i, value in enumerate(frequencies)}
    for az, freq, lin, amp_real, amp_imag in validated_rows:
        idx = (az_index[az], 0, f_index[freq], 0)
        amp_value_raw = complex(amp_real, amp_imag)
        phase_value = 0.0 if amp_value_raw == 0.0j else float(cmath.phase(amp_value_raw))
        existing_power = rcs_power[idx]
        if not np.isnan(existing_power):
            if abs(existing_power - lin) > EPS:
                raise ValueError(
                    f'Duplicate sample conflict at az={az}, el=0, f={freq}, pol={polarization}.')
            existing_phase = rcs_phase[idx]
            if np.isfinite(existing_phase):
                two_pi = 2.0 * math.pi
                diff_angle = (phase_value - float(existing_phase) + math.pi) % two_pi - math.pi
                if abs(diff_angle) > 1e-5:
                    raise ValueError(
                        f'Duplicate amplitude conflict at az={az}, el=0, f={freq}, pol={polarization}.')
            continue
        rcs_power[idx] = 0.0 if lin == 0.0 else float(lin)
        rcs_phase[idx] = phase_value
        if preserve_raw:
            rcs_amp_real[idx] = float(amp_value_raw.real)
            rcs_amp_imag[idx] = float(amp_value_raw.imag)
    missing = np.argwhere(~np.isfinite(rcs_power))
    if missing.size:
        first = tuple(int(value) for value in missing[0])
        raise ValueError(
            "Solver samples do not form a complete rectangular "
            f"(angle x frequency x polarization) grid: {len(missing)} cell(s) "
            f"are missing; first missing index is {first}.")
    return dict(azimuths=azimuths, frequencies=frequencies, rcs_power=rcs_power,
                rcs_phase=rcs_phase, rcs_amp_real=rcs_amp_real, rcs_amp_imag=rcs_amp_imag)


def _rows(rng, n_az, n_f, quantity, repeats=0):
    az = np.round(np.linspace(0.0, 180.0, n_az), 6)
    fr = np.linspace(8.0, 12.0, n_f)
    grid_az, grid_f = np.meshgrid(az, fr, indexing='ij')
    # Small amplitudes keep float32-rounded power within the absolute EPS
    # duplicate tolerance, so exact repeats are accepted as they always were.
    amp = (rng.standard_normal(grid_az.shape) + 1j * rng.standard_normal(grid_az.shape)) * 1e-4
    amp.flat[0] = 0.0                       # a zero field keeps phase 0
    amp.flat[1] = complex(-0.0, 0.0)        # a negative-zero real part is still a zero field
    k0 = 2.0 * np.pi * grid_f * 1e9 / C0
    linear = np.abs(amp) ** 2 / (4.0 * k0) if quantity == 'sigma_2d' else 4.0 * np.pi * np.abs(amp) ** 2
    rows = [
        dict(frequency_ghz=float(f), theta_inc_deg=float(a), theta_scat_deg=float(a),
             rcs_linear=float(l), rcs_db=0.0, rcs_amp_real=float(c.real),
             rcs_amp_imag=float(c.imag), rcs_amp_phase_deg=0.0, linear_residual=0.0)
        for a, f, l, c in zip(grid_az.ravel(), grid_f.ravel(), linear.ravel(), amp.ravel())
    ]
    rows += [dict(rows[index]) for index in range(repeats)]  # exact repeats are accepted
    order = rng.permutation(len(rows))
    return [rows[index] for index in order]


def _table(rows):
    with compact_samples():
        table = sample_buffer(len(rows))
    for row in rows:
        table.append(row)
    return table


class VectorisedExportGridTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2026)

    def assert_same_grid(self, actual, reference):
        for key, value in reference.items():
            if value is None:
                self.assertNotIn(key, actual)
                continue
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
            self.assertEqual(actual[key].dtype, value.dtype, key)

    def test_rows_and_compact_tables_match_the_row_loop_bit_for_bit(self):
        for quantity in ('sigma_2d', 'sigma_3d'):
            rows = _rows(self.rng, 61, 23, quantity, repeats=40)
            unit = 'dBke' if quantity == 'sigma_2d' else 'dBsm'
            reference = _legacy_grid(rows, 'TM', quantity)
            for samples in (rows, _table(rows)):
                payload = _build_grid_for_samples(
                    samples, 'TM', rcs_log_unit=unit, rcs_linear_quantity=quantity)
                self.assert_same_grid(payload, reference)
                self.assertEqual(payload['rcs_power'].shape, (61, 1, 23, 1))
                self.assertEqual(payload['polarizations'].tolist(), ['HH'])
                self.assertTrue(payload['raw_complex_amplitude_preserved'])
            without_raw = _build_grid_for_samples(
                rows, 'TM', preserve_raw_complex_amplitude=False,
                rcs_log_unit=unit, rcs_linear_quantity=quantity)
            self.assertNotIn('rcs_amp_real', without_raw)
            self.assertNotIn('complex_field_domain', without_raw)
            np.testing.assert_array_equal(without_raw['rcs_power'], reference['rcs_power'])

    def test_compact_tables_are_read_as_columns_without_row_dictionaries(self):
        table = _table(_rows(self.rng, 19, 7, 'sigma_2d'))
        with mock.patch.object(SampleTable, '__getitem__', side_effect=AssertionError('row materialised')), \
                mock.patch.object(SampleTable, '__iter__', side_effect=AssertionError('rows iterated')):
            payload = _build_grid_for_samples(table, 'TE')
        self.assertEqual(payload['rcs_power'].shape, (19, 1, 7, 1))
        self.assertEqual(payload['polarizations'].tolist(), ['VV'])

    def test_zero_fields_keep_zero_phase_and_power(self):
        rows = _rows(self.rng, 3, 2, 'sigma_3d')
        payload = _build_grid_for_samples(rows, 'VV', rcs_log_unit='dBsm', rcs_linear_quantity='sigma_3d')
        zero_cells = (payload['rcs_amp_real'] == 0.0) & (payload['rcs_amp_imag'] == 0.0)
        self.assertEqual(int(np.count_nonzero(zero_cells)), 2)
        self.assertTrue(np.all(payload['rcs_phase'][zero_cells] == 0.0))
        self.assertTrue(np.all(payload['rcs_power'][zero_cells] == 0.0))

    def test_invalid_rows_and_duplicates_raise_the_historical_messages(self):
        base = _rows(self.rng, 9, 5, 'sigma_2d')

        def missing_field(rows):
            del rows[7]['rcs_amp_real']

        def text_value(rows):
            rows[3]['frequency_ghz'] = 'ten'

        def non_finite(rows):
            rows[5]['rcs_linear'] = float('nan')

        def negative_linear(rows):
            rows[2]['rcs_linear'] = -1.0

        def zero_frequency(rows):
            rows[4]['frequency_ghz'] = 0.0

        def inconsistent_linear(rows):
            rows[6]['rcs_linear'] *= 1.001

        def duplicate_power_conflict(rows):
            repeat = dict(rows[0])
            repeat['rcs_linear'] *= 1.5
            repeat['rcs_amp_real'] *= math.sqrt(1.5)
            repeat['rcs_amp_imag'] *= math.sqrt(1.5)
            rows.append(repeat)

        def duplicate_phase_conflict(rows):
            repeat = dict(rows[0])
            rotated = complex(repeat['rcs_amp_real'], repeat['rcs_amp_imag']) * cmath.exp(1j * 1e-3)
            repeat['rcs_amp_real'], repeat['rcs_amp_imag'] = rotated.real, rotated.imag
            rows.append(repeat)

        def incomplete_grid(rows):
            del rows[11]

        def two_bad_rows_report_the_first(rows):
            rows[8]['rcs_linear'] = -2.0
            rows[1]['frequency_ghz'] = -3.0

        for mutate in (missing_field, text_value, non_finite, negative_linear, zero_frequency,
                       inconsistent_linear, duplicate_power_conflict, duplicate_phase_conflict,
                       incomplete_grid, two_bad_rows_report_the_first):
            rows = [dict(row) for row in base]
            mutate(rows)
            with self.assertRaises(ValueError) as expected:
                _legacy_grid(rows, 'TM')
            for samples in (rows, _table(rows) if mutate not in (missing_field, text_value) else None):
                if samples is None:
                    continue
                with self.assertRaises(ValueError) as actual:
                    _build_grid_for_samples(samples, 'TM')
                self.assertEqual(str(actual.exception), str(expected.exception), mutate.__name__)
        with self.assertRaisesRegex(ValueError, 'No samples available'):
            _build_grid_for_samples([], 'TM')
        with self.assertRaisesRegex(ValueError, "must be 'sigma_2d' or 'sigma_3d'"):
            _build_grid_for_samples(base, 'TM', rcs_linear_quantity='power')

    def test_first_repeat_wins_and_later_consistent_repeats_are_ignored(self):
        rows = _rows(self.rng, 4, 3, 'sigma_2d')
        rows = sorted(rows, key=lambda row: (row['theta_scat_deg'], row['frequency_ghz']))
        shifted = dict(rows[0])
        shifted['rcs_amp_real'] *= 1.0 + 4e-9   # within both the power and phase tolerances
        rows.append(shifted)
        payload = _build_grid_for_samples(rows, 'TM')
        self.assertEqual(payload['rcs_amp_real'][0, 0, 0, 0], rows[0]['rcs_amp_real'])
        self.assertEqual(payload['rcs_power'][0, 0, 0, 0], np.float32(rows[0]['rcs_linear']))


if __name__ == '__main__':
    unittest.main()
