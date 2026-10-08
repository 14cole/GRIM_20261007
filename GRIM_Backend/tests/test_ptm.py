import os
import struct
import tempfile
import unittest

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
import GRIM_Backend.io.ptm as ptm_io


def _fixed_ascii(value, width):
    return str(value).encode("ascii")[:width].ljust(width, b"*")


def _independent_fixture(
    path,
    *,
    byte_order="little",
    embedded_count=True,
    num_aspects=3,
    num_frequencies=40,
    start_aspect=179.0,
    aspect_increment=2.0,
):
    """Pack a fixture without calling the production PTM writer."""
    prefix = "<" if byte_order == "little" else ">"
    ints = (num_aspects, 11, 12, 13, 14, 15, 16, 17, 18)
    floats = (
        start_aspect, aspect_increment, 10.0, 3.9, 1.25, 7.5, -2.5, -12.0, 3.0
    )
    header = bytearray(struct.pack(f"{prefix}9i9f", *ints, *floats))
    header.extend(b"HH")
    header.extend(_fixed_ascii("fixture subject", 50))
    header.extend(_fixed_ascii("fixture configuration", 50))
    if embedded_count:
        header.extend(_fixed_ascii("fixture operator", 46))
        header.extend(struct.pack(f"{prefix}i", num_frequencies))
    else:
        header.extend(_fixed_ascii("fixture operator", 50))
    header.extend(_fixed_ascii("known_HH.ptm", 50))
    header.extend(_fixed_ascii("23AUG2026", 9))
    header.extend(_fixed_ascii("12:34:56", 8))
    if len(header) != ptm_io.PTM_HEADER_SIZE:
        raise AssertionError(len(header))

    block_size = 8 * num_frequencies
    sample_numbers = np.arange(num_aspects * num_frequencies, dtype=np.float32).reshape(
        num_aspects, num_frequencies
    )
    iq = (sample_numbers + 0.25) + 1j * (-sample_numbers - 0.75)
    iq = np.asarray(iq, dtype=np.complex64)
    with open(path, "wb") as stream:
        stream.write(header)
        stream.write(b"\x00" * (block_size - len(header)))
        stream.write(np.asarray(iq, dtype=np.dtype(f"{prefix}c8")).tobytes())
    return iq


class PtmReadTests(unittest.TestCase):
    def test_loads_independent_little_endian_fixture_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "known.ptm")
            expected_iq = _independent_fixture(path)
            grid = RcsGrid.load_ptm(path)

            np.testing.assert_allclose(grid.azimuths, [-179.0, -177.0, 179.0])
            np.testing.assert_allclose(grid.elevations, [7.5])
            np.testing.assert_allclose(grid.frequencies, np.linspace(8.05, 11.95, 40))
            self.assertEqual(grid.polarizations.tolist(), ["HH"])
            np.testing.assert_allclose(
                grid.rcs[:, 0, :, 0], expected_iq[[1, 2, 0], :],
                rtol=2.0e-6, atol=2.0e-6
            )
            self.assertEqual(grid.linear_quantity(), "sigma_3d")
            # PTM aspect/pitch load as plain azimuth/elevation.
            self.assertNotIn("angular_coordinate_system", grid.units)
            self.assertNotIn("angular_coordinate_system", grid.extra)
            self.assertNotIn("ptm_cut_type", grid.extra)
            self.assertEqual(grid.extra["ptm_corecell"], 11)
            self.assertEqual(grid.extra["ptm_subject"], "fixture subject")
            self.assertTrue(grid.extra["ptm_embedded_num_frequencies"])
            self.assertEqual(
                (grid.extra["ptm_roll"], grid.extra["ptm_tilt"]), (1.25, -2.5)
            )

            conic = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                rcs=grid.rcs,
                units={
                    "azimuth": "deg",
                    "elevation": "deg",
                    "frequency": "GHz",
                    "rcs_log_unit": "dBsm",
                    "rcs_linear_quantity": "sigma_3d",
                },
            )
            # Coordinate systems no longer gate arithmetic with az/el data.
            grid.coherent_add(conic, metadata_attested=True)

            derived = grid.coherent_add(grid, metadata_attested=True)
            derived_path = derived.save_ptm(os.path.join(tmp, "derived.ptm"))
            derived_header = ptm_io.read_ptm(derived_path).header
            self.assertAlmostEqual(derived_header.roll, 1.25)
            self.assertAlmostEqual(derived_header.tilt, -2.5)

    def test_reads_big_endian_and_legacy_operator_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            big_path = os.path.join(tmp, "big.ptm")
            expected_iq = _independent_fixture(big_path, byte_order="big")
            parsed = ptm_io.read_ptm(big_path)
            self.assertEqual(parsed.header.byte_order, "big")
            self.assertTrue(parsed.header.embedded_num_frequencies)
            np.testing.assert_array_equal(parsed.iq, expected_iq[[1, 2, 0], :])

            legacy_path = os.path.join(tmp, "legacy.ptm")
            _independent_fixture(legacy_path, embedded_count=False)
            legacy = ptm_io.read_ptm(legacy_path)
            self.assertFalse(legacy.header.embedded_num_frequencies)
            self.assertEqual(legacy.header.operator, "fixture operator")

    def test_rejects_inexact_framing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.ptm")
            _independent_fixture(path)
            with open(path, "rb+") as stream:
                stream.truncate(os.path.getsize(path) - 1)
            with self.assertRaisesRegex(ValueError, "invalid PTM framing/header"):
                ptm_io.read_ptm(path)

    def test_rejects_duplicate_or_seam_alias_aspects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "alias.ptm")
            _independent_fixture(path, start_aspect=-180.0, aspect_increment=360.0)
            with self.assertRaisesRegex(ValueError, "duplicate/seam-alias"):
                ptm_io.read_ptm(path)

    def test_rejects_nonfinite_iq_without_inventing_missing_sample_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nan.ptm")
            _independent_fixture(path)
            with open(path, "rb+") as stream:
                stream.seek(8 * 40)
                stream.write(struct.pack("<ff", float("nan"), 0.0))
            with self.assertRaisesRegex(ValueError, "missing-sample marker"):
                ptm_io.read_ptm(path)

    def test_rejects_duplicate_or_nonpositive_reconstructed_frequencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            zero_bandwidth = os.path.join(tmp, "zero_bandwidth.ptm")
            _independent_fixture(zero_bandwidth)
            with open(zero_bandwidth, "rb+") as stream:
                stream.seek(48)
                stream.write(struct.pack("<f", 0.0))
            with self.assertRaisesRegex(ValueError, "bandwidth must be positive"):
                ptm_io.read_ptm(zero_bandwidth)

            nonpositive_start = os.path.join(tmp, "nonpositive_start.ptm")
            _independent_fixture(nonpositive_start)
            with open(nonpositive_start, "rb+") as stream:
                stream.seek(44)
                stream.write(struct.pack("<f", 1.0))
                stream.seek(48)
                stream.write(struct.pack("<f", 2.0))
            with self.assertRaisesRegex(ValueError, "lowest frequency"):
                ptm_io.read_ptm(nonpositive_start)


class PtmWriteTests(unittest.TestCase):
    @staticmethod
    def _grid(*, pitch=0.0):
        aspects = np.asarray([-179.0, -177.0, 179.0])
        frequencies = np.linspace(8.05, 11.95, 40)
        indices = np.arange(aspects.size * frequencies.size, dtype=float).reshape(
            aspects.size, frequencies.size
        )
        iq = (1.0 + indices / 100.0) * np.exp(1j * (indices / 37.0))
        grid = RcsGrid(
            aspects,
            [pitch],
            frequencies,
            ["VV"],
            rcs=iq[:, None, :, None],
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
        )
        return grid, np.asarray(iq, dtype=np.complex64)

    def test_writes_exact_little_endian_framing_and_round_trips_iq(self):
        grid, expected_iq = self._grid()
        with tempfile.TemporaryDirectory() as tmp:
            path = grid.save_ptm(os.path.join(tmp, "written"))
            self.assertTrue(path.endswith(".ptm"))
            block_size = 8 * 40
            self.assertEqual(os.path.getsize(path), (3 + 1) * block_size)
            with open(path, "rb") as stream:
                raw = stream.read()
            self.assertEqual(struct.unpack_from("<i", raw, 0)[0], 3)
            self.assertEqual(raw[72:74], b"VV")
            self.assertEqual(struct.unpack_from("<i", raw, 220)[0], 40)
            self.assertNotIn(
                "GRIM_GC_V1", ptm_io.read_ptm(path).header.configuration
            )
            disk_iq = np.frombuffer(raw, dtype="<c8", count=3 * 40, offset=block_size)
            np.testing.assert_array_equal(
                disk_iq.reshape(3, 40), expected_iq[[2, 0, 1], :]
            )

            restored = RcsGrid.load_ptm(path)
            np.testing.assert_allclose(restored.azimuths, grid.azimuths)
            np.testing.assert_allclose(restored.frequencies, grid.frequencies, atol=2.0e-6)
            np.testing.assert_allclose(
                restored.rcs[:, 0, :, 0], expected_iq, rtol=2.0e-6, atol=2.0e-6
            )

    def test_preserves_opaque_header_metadata_but_writes_little_endian(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.ptm")
            _independent_fixture(source, byte_order="big")
            grid = RcsGrid.load_ptm(source)
            output = grid.save_ptm(os.path.join(tmp, "copy.ptm"))
            parsed = ptm_io.read_ptm(output)
            self.assertEqual(parsed.header.byte_order, "little")
            self.assertEqual(parsed.header.corecell, 11)
            self.assertEqual(parsed.header.phase_flag, 18)
            self.assertEqual(parsed.header.start_aspect, 179.0)
            self.assertEqual(parsed.header.aspect_increment, 2.0)
            self.assertEqual(parsed.header.roll, 1.25)
            self.assertEqual(parsed.header.tilt, -2.5)
            self.assertEqual(parsed.header.cal_dbsm, -12.0)
            self.assertEqual(parsed.header.subject, "fixture subject")
            self.assertEqual(parsed.header.configuration, "fixture configuration")
            self.assertEqual(parsed.header.operator, "fixture operator")
            self.assertEqual(parsed.header.date, "23AUG2026")
            self.assertEqual(parsed.header.time, "12:34:56")
            self.assertEqual(parsed.header.filename, "copy.ptm")

    def test_long_header_filename_preserves_legacy_suffix(self):
        grid, _ = self._grid()
        with tempfile.TemporaryDirectory() as tmp:
            basename = f"{'x' * 60}.ptm"
            output = grid.save_ptm(os.path.join(tmp, basename))
            stored = ptm_io.read_ptm(output).header.filename

        self.assertEqual(len(stored), 50)
        self.assertEqual(stored[:45], "x" * 45)
        self.assertTrue(stored.endswith("x.ptm"))

    def test_rejects_lossy_or_unrepresentable_exports(self):
        grid, _ = self._grid()
        with tempfile.TemporaryDirectory() as tmp:
            too_few = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies[:36],
                grid.polarizations,
                rcs=grid.rcs[:, :, :36, :],
                units=dict(grid.units),
            )
            with self.assertRaisesRegex(ValueError, "at least 37 frequencies"):
                too_few.save_ptm(os.path.join(tmp, "short.ptm"))

            nonuniform_freq = RcsGrid(
                grid.azimuths,
                grid.elevations,
                np.where(np.arange(40) == 20, grid.frequencies + 0.01, grid.frequencies),
                grid.polarizations,
                rcs=grid.rcs,
                units=dict(grid.units),
            )
            with self.assertRaisesRegex(ValueError, "uniformly spaced"):
                nonuniform_freq.save_ptm(os.path.join(tmp, "nonuniform.ptm"))

            power_only = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                rcs_power=grid.rcs_power,
                units=dict(grid.units),
            )
            with self.assertRaisesRegex(ValueError, "requires phase"):
                power_only.save_ptm(os.path.join(tmp, "power.ptm"))

            sparse_power = np.array(grid.rcs_power, copy=True)
            sparse_phase = np.array(grid.rcs_phase, copy=True)
            sparse_power[0, 0, 0, 0] = np.nan
            sparse_phase[0, 0, 0, 0] = np.nan
            sparse = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                rcs_power=sparse_power,
                rcs_phase=sparse_phase,
                units=dict(grid.units),
            )
            existing = os.path.join(tmp, "sparse.ptm")
            with open(existing, "wb") as stream:
                stream.write(b"existing file remains intact")
            with self.assertRaisesRegex(ValueError, "missing-sample marker"):
                sparse.save_ptm(existing)
            with open(existing, "rb") as stream:
                self.assertEqual(stream.read(), b"existing file remains intact")

            zero_power = np.array(grid.rcs_power, copy=True)
            zero_phase = np.array(grid.rcs_phase, copy=True)
            zero_power[0, 0, 0, 0] = 0.0
            zero_phase[0, 0, 0, 0] = np.nan
            zero = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                rcs_power=zero_power,
                rcs_phase=zero_phase,
                units=dict(grid.units),
            )
            zero_path = zero.save_ptm(os.path.join(tmp, "zero.ptm"))
            zero_loaded = RcsGrid.load_ptm(zero_path)
            self.assertEqual(float(zero_loaded.rcs_power[0, 0, 0, 0]), 0.0)
            self.assertTrue(np.isfinite(zero_loaded.rcs[0, 0, 0, 0]))

            sigma_2d = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                grid.polarizations,
                rcs=grid.rcs,
                units={**grid.units, "rcs_linear_quantity": "sigma_2d"},
            )
            with self.assertRaisesRegex(ValueError, "sigma_3d"):
                sigma_2d.save_ptm(os.path.join(tmp, "2d.ptm"))

            unsupported_polarity = RcsGrid(
                grid.azimuths,
                grid.elevations,
                grid.frequencies,
                ["TE"],
                rcs=grid.rcs,
                units=dict(grid.units),
            )
            with self.assertRaisesRegex(ValueError, "documented polarizations"):
                unsupported_polarity.save_ptm(os.path.join(tmp, "te.ptm"))

    def test_any_elevation_and_polarization_exports_without_coordinate_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonzero, _ = self._grid(pitch=7.5)
            output = nonzero.save_ptm(os.path.join(tmp, "nonzero.ptm"))
            self.assertAlmostEqual(ptm_io.read_ptm(output).header.pitch, 7.5)
            nonzero.save_pio(os.path.join(tmp, "nonzero.pio"))

            # GRIM no longer claims a coordinate convention, so an inherited
            # GRIM_GC_V1 marker is removed while other provenance is kept.
            nonzero.extra["ptm_configuration"] = "GRIM_GC_V1;keep provenance"
            marked = nonzero.save_ptm(os.path.join(tmp, "marked.ptm"))
            self.assertEqual(
                ptm_io.read_ptm(marked).header.configuration, "keep provenance"
            )

            cross, _ = self._grid(pitch=5.0)
            cross.polarizations = np.asarray(["VH"])
            cross.extra["ptm_roll"] = 1.0
            cross_path = cross.save_ptm(os.path.join(tmp, "vh.ptm"))
            cross_header = ptm_io.read_ptm(cross_path).header
            self.assertEqual(cross_header.polarity, "VH")
            self.assertAlmostEqual(cross_header.roll, 1.0)

    def test_multiple_axes_require_explicit_slice_selection(self):
        aspects = [0.0, 1.0]
        frequencies = np.linspace(1.0, 2.0, 40)
        iq = np.ones((2, 2, 40, 2), dtype=np.complex64)
        grid = RcsGrid(
            aspects,
            [0.0, 5.0],
            frequencies,
            ["HH", "VV"],
            rcs=iq,
            units={"frequency": "GHz", "rcs_linear_quantity": "sigma_3d"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "el_idx required"):
                grid.save_ptm(os.path.join(tmp, "multi.ptm"))
            with self.assertRaisesRegex(ValueError, "pol_idx required"):
                grid.save_ptm(os.path.join(tmp, "multi.ptm"), el_idx=0)
            output = grid.save_ptm(
                os.path.join(tmp, "selected.ptm"), el_idx=1, pol_idx=1
            )
            loaded = RcsGrid.load_ptm(output)
            self.assertEqual(loaded.polarizations.tolist(), ["VV"])
            np.testing.assert_allclose(loaded.elevations, [5.0])


if __name__ == "__main__":
    unittest.main()
