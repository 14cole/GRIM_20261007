"""SS imports preserve the source I/Q through coherent operations and export."""

import struct

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import load_dataset
from GRIM_Backend.io.xpatch import read_ss
from GRIM_Backend.tests.test_ss import build_ss


def _write_samples(path, fields, coordinates, *, mode="incident", ifreq=1):
    """Write the supplied float32 components into actual SS records."""
    fields = np.asarray(fields, dtype=np.complex64)
    n_signals, n_frequencies, n_polarizations = fields.shape
    assert n_polarizations == 4
    metadata = build_ss(
        path, n_signals, n_frequencies, ifreq, 9.0, 9.0 + n_frequencies - 1,
        dict(edge_diff=True, iqmatrix=True, ibspsave=3), mode,
        sweep_values=[azimuth for azimuth, _ in coordinates],
        pre_frequency_padding=37,
    )
    record_size = metadata["nbytesb"] + metadata["nbytesd"]
    angle_offset = 280 if mode == "incident" else 288
    with open(path, "r+b") as stream:
        for signal, (azimuth, elevation) in enumerate(coordinates):
            header_d = signal * record_size + metadata["nbytesb"]
            stream.seek(header_d + angle_offset)
            stream.write(struct.pack(">ff", azimuth, elevation))
            stream.seek(header_d + 408)
            components = np.empty((n_frequencies, 4, 2), dtype=">f4")
            components[..., 0] = fields[signal].real
            components[..., 1] = fields[signal].imag
            stream.write(components.tobytes())


@pytest.mark.parametrize("mode", ["incident", "observation"])
@pytest.mark.parametrize("ifreq", [1, 2])
def test_public_ss_load_and_native_export_preserve_original_quadratures(
    tmp_path, mode, ifreq,
):
    samples = np.empty((2, 2, 4), dtype=np.complex64)
    samples.real = -1.0
    samples.imag = [[1e-8, -1e-8, 1e-12, -1e-12],
                    [1e-6, -1e-6, 1e-20, -1e-20]]
    path = tmp_path / "precision.ss"
    _write_samples(
        path, samples, [(30.0, 5.0), (40.0, 5.0)], mode=mode, ifreq=ifreq,
    )
    original_bytes = path.read_bytes()

    parsed = read_ss(path, verbose=False)
    assert parsed["angle_source"] == mode
    for index, name in enumerate(("vv", "vh", "hv", "hh")):
        np.testing.assert_array_equal(parsed[name], samples[..., index])

    loaded = load_dataset(path)
    expected = samples.reshape(2, 1, 2, 4)
    assert loaded.storage_kind == "complex"
    assert loaded.rcs.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(loaded.rcs, expected)
    np.testing.assert_array_equal(loaded.azimuths, [30.0, 40.0])
    np.testing.assert_array_equal(loaded.elevations, [5.0])
    np.testing.assert_array_equal(loaded.polarizations, ["VV", "VH", "HV", "HH"])
    assert loaded.linear_quantity() == "sigma_3d"
    assert loaded.units["rcs_log_unit"] == "dBsm"
    assert loaded.units["frequency"] == "GHz"

    restored = load_dataset(loaded.save(str(tmp_path / "roundtrip.grim")))
    np.testing.assert_array_equal(restored.rcs, expected)
    assert restored.rcs.dtype == np.dtype(np.complex64)
    assert path.read_bytes() == original_bytes


def test_ss_import_retains_the_known_coherent_cancellation_residual(tmp_path):
    samples = np.empty((1, 1, 4), dtype=np.complex64)
    samples.real = -1.0
    samples.imag = [1e-8, -1e-8, 1e-6, -1e-6]
    path = tmp_path / "cancel.ss"
    _write_samples(path, samples, [(0.0, 0.0)])
    loaded = load_dataset(path)
    reference = RcsGrid(
        loaded.azimuths, loaded.elevations, loaded.frequencies,
        loaded.polarizations, rcs=np.ones(loaded.shape, dtype=np.complex64),
        units=loaded.units,
    )
    residual = loaded.coherent_add(reference)
    expected = np.zeros(loaded.shape, dtype=np.complex64)
    expected.imag = samples.imag.reshape(loaded.shape)
    np.testing.assert_array_equal(residual.rcs, expected)
    np.testing.assert_allclose(
        residual.power, expected.imag.astype(np.float64) ** 2, rtol=2e-7,
        atol=0.0,
    )
    np.testing.assert_array_equal(loaded.rcs, samples.reshape(loaded.shape))


@pytest.mark.parametrize("mode", ["incident", "observation"])
def test_ss_coordinate_scatter_preserves_channels_zeros_and_missing_cells(
    tmp_path, mode,
):
    samples = np.empty((3, 2, 4), dtype=np.complex64)
    samples.real = np.arange(24, dtype=np.float32).reshape(samples.shape) - 12
    samples.imag = np.resize([1e-8, -1e-8], samples.shape)
    samples[0, 0, 0] = 3 + 4j
    samples[1, 1, 1] = 0j
    samples[2, 0, 3] = complex(np.nan, np.nan)
    coordinates = [(30.0, 5.0), (10.0, 0.0), (10.0, 5.0)]
    path = tmp_path / "sparse.ss"
    _write_samples(path, samples, coordinates, mode=mode, ifreq=2)

    loaded = load_dataset(path)
    expected = np.full((2, 2, 2, 4), complex(np.nan, np.nan), dtype=np.complex64)
    expected[1, 1] = samples[0]
    expected[0, 0] = samples[1]
    expected[0, 1] = samples[2]
    np.testing.assert_array_equal(loaded.azimuths, [10.0, 30.0])
    np.testing.assert_array_equal(loaded.elevations, [0.0, 5.0])
    np.testing.assert_array_equal(loaded.frequencies, [9.0, 10.0])
    np.testing.assert_array_equal(loaded.rcs, expected)
    assert loaded.power[1, 1, 0, 0] == 25.0
    assert loaded.power[0, 0, 1, 1] == 0.0
    assert np.isnan(loaded.phase[0, 1, 0, 3])
    assert np.isnan(loaded.power[1, 0]).all()
    assert np.isnan(loaded.phase[1, 0]).all()
    assert loaded.extra["source_format"] == "Xpatch SS"
    if mode == "observation":
        assert loaded.extra["fixed_incident_azimuth_deg"] == 0.0
        assert loaded.extra["fixed_incident_elevation_deg"] == 0.0

    restored = load_dataset(loaded.save(str(tmp_path / "sparse.grim")))
    np.testing.assert_array_equal(restored.rcs, expected)
    assert restored.units == loaded.units
