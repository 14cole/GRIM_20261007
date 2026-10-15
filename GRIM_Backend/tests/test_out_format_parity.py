"""The array-based ``.out`` reader reproduces the former row loop exactly."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.samples import _cst_samples_equivalent

UNITS = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_log_unit": "dBke"}


def _former_load_out(path):
    """The row-by-row implementation the reader replaced, kept as the reference."""
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"line {line_no}: expected exactly 4 columns "
                    "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                )
            try:
                freq_ghz = float(parts[0])
                azimuth_deg = float(parts[1])
                rcs_dbke = float(parts[2])
                phase_deg = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc
            if not np.isfinite(freq_ghz) or freq_ghz <= 0.0:
                raise ValueError(f"line {line_no}: frequency_ghz must be positive and finite")
            if not np.isfinite(azimuth_deg):
                raise ValueError(f"line {line_no}: azimuth_deg must be finite")
            if np.isnan(rcs_dbke) or np.isposinf(rcs_dbke):
                raise ValueError(f"line {line_no}: rcs_dbke must be finite or -Inf")
            if np.isinf(phase_deg):
                raise ValueError(f"line {line_no}: phase_deg must be finite or NaN")
            records.append((freq_ghz, azimuth_deg, rcs_dbke, phase_deg))
    if not records:
        raise ValueError("OUT contains no data rows")
    frequencies = np.asarray(sorted({r[0] for r in records}), dtype=float)
    azimuths = np.asarray(sorted({r[1] for r in records}), dtype=float)
    f_idx = {float(v): i for i, v in enumerate(frequencies.tolist())}
    az_idx = {float(v): i for i, v in enumerate(azimuths.tolist())}
    shape = (len(azimuths), 1, len(frequencies), 1)
    power = np.full(shape, np.nan, dtype=np.float64)
    phase = np.full(shape, np.nan, dtype=np.float64)
    for freq_ghz, azimuth_deg, rcs_dbke, phase_deg in records:
        ai = az_idx[float(azimuth_deg)]
        fi = f_idx[float(freq_ghz)]
        lambda_m = C0 / (float(freq_ghz) * 1.0e9)
        if np.isneginf(rcs_dbke):
            sigma_2d = 0.0
        else:
            with np.errstate(over="raise", invalid="raise"):
                try:
                    sigma_2d = (lambda_m / (2.0 * np.pi)) * (10.0 ** (rcs_dbke / 10.0))
                except (FloatingPointError, OverflowError) as exc:
                    raise ValueError(
                        "OUT dBke magnitude overflows finite linear power at "
                        f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                    ) from exc
        incoming_power = float(sigma_2d)
        if not np.isfinite(incoming_power):
            raise ValueError(
                "OUT dBke magnitude does not produce finite linear power at "
                f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
            )
        incoming_phase = float(np.deg2rad(phase_deg)) if np.isfinite(phase_deg) else np.nan
        existing_power = float(power[ai, 0, fi, 0])
        if np.isfinite(existing_power):
            existing_phase = float(phase[ai, 0, fi, 0])
            if not _cst_samples_equivalent(existing_power, existing_phase, incoming_power, incoming_phase):
                raise ValueError(
                    "conflicting duplicate OUT sample at "
                    f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                )
            if not np.isfinite(existing_phase) and np.isfinite(incoming_phase):
                phase[ai, 0, fi, 0] = incoming_phase
            continue
        power[ai, 0, fi, 0] = incoming_power
        phase[ai, 0, fi, 0] = incoming_phase
    return azimuths, frequencies, power, phase


def _former_grid(path, polarization):
    azimuths, frequencies, power, phase = _former_load_out(path)
    return RcsGrid(
        azimuths, [0.0], frequencies, [polarization],
        rcs_power=power, rcs_phase=phase, rcs_domain="power_phase", units=dict(UNITS),
    )


class OutReaderParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def _write(self, name, text):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        return path

    def _assert_same_grid(self, path, polarization):
        loaded = RcsGrid.load_out(path)
        expected = _former_grid(path, polarization)
        np.testing.assert_array_equal(loaded.azimuths, expected.azimuths)
        np.testing.assert_array_equal(loaded.frequencies, expected.frequencies)
        self.assertEqual(list(loaded.polarizations), [polarization])
        np.testing.assert_array_equal(np.asarray(loaded.power), np.asarray(expected.power))
        np.testing.assert_array_equal(np.asarray(loaded.phase), np.asarray(expected.phase))
        return loaded

    def _assert_same_error(self, path, message):
        with self.assertRaises(ValueError) as new:
            RcsGrid.load_out(path)
        with self.assertRaises(ValueError) as former:
            _former_load_out(path)
        self.assertEqual(str(new.exception), str(former.exception))
        self.assertIn(message, str(new.exception))

    def test_random_table_with_comments_gaps_and_repeats_matches_the_former_reader(self):
        rng = np.random.default_rng(7)
        frequencies = [9.0, 9.25, 10.0, 11.5, 12.0]
        azimuths = np.round(np.linspace(-45.0, 45.0, 31), 3)
        rows = []
        for frequency in frequencies:
            for azimuth in azimuths:
                level = float(rng.uniform(-60.0, 40.0))
                phase = float(rng.uniform(-180.0, 180.0))
                if rng.uniform() < 0.05:
                    level = -np.inf
                if rng.uniform() < 0.1:
                    phase = np.nan
                rows.append([frequency, azimuth, level, phase])
        rng.shuffle(rows)
        for index in range(0, len(rows), 23):
            rows[index][2:] = [-np.inf, np.nan]  # zero width without a phase
        lines = ["﻿# frequency_GHz azimuth_deg rcs_dBke phase_deg\n", "\n"]
        for index, (frequency, azimuth, level, phase) in enumerate(rows):
            lines.append(f"{frequency:.17g}\t {azimuth:.17g}  {level:.17g} {phase:.17g}  # row {index}\n")
            if index % 17 == 0:
                # An exact repeat.
                lines.append(f"{frequency:.17g} {azimuth:.17g} {level:.17g} {phase:.17g}\n")
            if index % 23 == 0:
                # A repeat of a zero-width sample that supplies its phase.
                lines.append(f"{frequency:.17g} {azimuth:.17g} -inf 12.5\n")
            if index % 40 == 0:
                lines.append("   \n# a comment line\n")
        path = self._write("sweep_VV.out", "".join(lines))
        loaded = self._assert_same_grid(path, "VV")
        self.assertEqual(loaded.azimuths.size, 31)
        self.assertEqual(loaded.frequencies.size, 5)
        self.assertTrue(np.all(np.isfinite(np.asarray(loaded.power))))

    def test_polarization_comes_from_the_filename(self):
        for name, polarization in (("a_HH.out", "HH"), ("b-vv.out", "VV"), ("plain.out", "NA")):
            path = self._write(name, "1 0 10 0\n1 1 11 5\n")
            self._assert_same_grid(path, polarization)

    def test_duplicate_phase_fill_then_conflict_is_reported_like_before(self):
        # A finite-width sample without a phase cannot be completed by a
        # later phase (the row loop rejected that too); a zero-width one can.
        path = self._write("dup_HH.out", "1 0 10 nan\n1 0 10 30\n")
        self._assert_same_error(path, "conflicting duplicate OUT sample at frequency=1 GHz, azimuth=0 deg")
        path = self._write("fill_HH.out", "1 0 -inf nan\n1 1 10 nan\n1 0 -inf 30\n1 0 -inf 31\n")
        loaded = self._assert_same_grid(path, "HH")
        self.assertAlmostEqual(float(np.asarray(loaded.phase)[0, 0, 0, 0]), np.deg2rad(30.0))
        self.assertEqual(float(np.asarray(loaded.power)[0, 0, 0, 0]), 0.0)

    def test_first_offending_line_wins_whatever_its_problem(self):
        cases = (
            ("1 0 10 0\n-1 0 10 0\n1 0 10\n", "line 2: frequency_ghz must be positive and finite"),
            ("1 0 10 0\n1 0 10\n-1 0 10 0\n", "line 2: expected exactly 4 columns"),
            ("1 0 10 0\n1 nan 10 0\n", "line 2: azimuth_deg must be finite"),
            ("1 0 10 0\n1 0 inf 0\n", "line 2: rcs_dbke must be finite or -Inf"),
            ("1 0 nan 0\n", "line 1: rcs_dbke must be finite or -Inf"),
            ("1 0 10 inf\n", "line 1: phase_deg must be finite or NaN"),
            ("1 0 abc 0\n", "line 1: invalid numeric value"),
            ("# only comments\n\n", "OUT contains no data rows"),
            ("1 0 10 0\n1 0 5000 0\n", "OUT dBke magnitude overflows finite linear power at frequency=1 GHz, azimuth=0 deg"),
            # A conflicting repeat on an earlier line is reported before a
            # later overflow, as the row loop did.
            ("1 0 10 0\n1 1 10 0\n1 0 20 0\n1 2 5000 0\n", "conflicting duplicate OUT sample at frequency=1 GHz, azimuth=0 deg"),
            ("1 0 10 0\n1 1 5000 0\n1 0 20 0\n", "OUT dBke magnitude overflows finite linear power at frequency=1 GHz, azimuth=1 deg"),
        )
        for text, message in cases:
            with self.subTest(message=message):
                self._assert_same_error(self._write("case_HH.out", text), message)

    def test_minus_infinity_is_a_zero_width_sample(self):
        path = self._write("zero_VV.out", "2 0 -inf 0\n2 1 -inf nan\n2 2 3 0\n")
        loaded = self._assert_same_grid(path, "VV")
        power = np.asarray(loaded.power)[:, 0, 0, 0]
        self.assertEqual(power[0], 0.0)
        self.assertEqual(power[1], 0.0)
        self.assertGreater(power[2], 0.0)


if __name__ == "__main__":
    unittest.main()
