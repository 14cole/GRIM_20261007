"""Binary I/O remains exact with bounded decoding and disk-backed outputs."""
import gc
import hashlib
import tracemalloc
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.binary_storage import estimate_binary_import
from GRIM_Backend.io.loaders import load_dataset
from GRIM_Backend.tests import test_dataset_core_repairs as pio_fixtures
from GRIM_Backend.tests.test_ss_complex_precision import _write_samples


@pytest.mark.parametrize("precision,real_dtype", [("Single", "f4"), ("Double", "f8")])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("real_only", [False, True])
@pytest.mark.parametrize("closed_descending", [False, True])
def test_pio_disk_precision_axes_missing_values_and_source_independence(
        tmp_path, precision, real_dtype, endian, real_only, closed_descending):
    x = np.array([360., 270., 180., 90., 0.] if closed_descending else [0., 2., 4.])
    y = np.array([10., 9., 8.])
    field = (np.arange(x.size * y.size).reshape(x.size, y.size) - 5 + 1e-8j)
    field[1, 1] = np.nan + 1j * np.nan
    source = tmp_path / "input.pio"
    if real_only:
        data = field.real.astype(endian + real_dtype).reshape(-1, order="F")
    else:
        ordered = field.reshape(-1, order="F")
        data = np.empty((ordered.size, 2), dtype=endian + real_dtype)
        data[:, 0], data[:, 1] = ordered.real, ordered.imag
    pio_fixtures.PioStreamingTests._write_pio_fixture(source,
        xsize=str(x.size), ysize=str(y.size), xvals=":".join(map(str, x)),
        yvals=":".join(map(str, y)), precision=precision,
        data_type="Real" if real_only else "Complex", payload=data.tobytes())
    if endian == ">":
        source.write_bytes(source.read_bytes().replace(b"Little Endian", b"Big    Endian"))
    original = source.read_bytes()
    expected = load_dataset(source)
    cache = tmp_path / "cache"
    cache.mkdir()
    actual = load_dataset(source, storage="disk", cache_dir=cache)
    assert actual.sample_storage == "disk" and actual.load_storage == "disk"
    assert actual._complex.dtype == expected._complex.dtype
    for name in ("azimuths", "elevations", "frequencies", "polarizations", "power", "phase", "rcs"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    expected_field = np.array(field.real if real_only else field,
                              dtype=np.complex64 if real_dtype == "f4" else np.complex128)
    if closed_descending:
        expected_field = expected_field[:-1][::-1]
    expected_field = expected_field[:, ::-1]
    expected_field[~np.isfinite(expected_field)] = np.nan + 1j * np.nan
    np.testing.assert_array_equal(actual.rcs[:, 0, :, 0], expected_field)
    assert source.read_bytes() == original
    source.unlink()  # No source-file map remains attached to the output.
    restored = RcsGrid.load(actual.save(tmp_path / "roundtrip.grim"))
    np.testing.assert_array_equal(restored.rcs, actual.rcs)
    del actual
    gc.collect()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("mode", ["incident", "observation"])
@pytest.mark.parametrize("ifreq", [1, 2])
def test_ss_disk_preserves_sparse_looks_and_quadratures(tmp_path, mode, ifreq):
    samples = np.empty((3, 7, 4), dtype=np.complex64)
    samples.real = -1
    samples.imag = np.arange(84).reshape(3, 7, 4) * 1e-9
    samples[1, 3, 2] = np.nan + 1j * np.nan
    samples[2, 4, 1] = 0
    source = tmp_path / "input.ss"
    _write_samples(source, samples, [(0., 0.), (1., 0.), (0., 1.)], mode=mode, ifreq=ifreq)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = load_dataset(source)
    cache = tmp_path / "cache"
    cache.mkdir()
    with mock.patch("GRIM_Backend.io.xpatch.np.fromfile", side_effect=AssertionError("whole file read")):
        actual = load_dataset(source, storage="disk", cache_dir=cache)
    assert actual.sample_storage == "disk" and actual.load_storage == "disk"
    for name in ("azimuths", "elevations", "frequencies", "polarizations", "power", "phase", "rcs"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.units == expected.units
    assert np.isnan(actual.power[1, 1]).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    source.unlink()
    restored = RcsGrid.load(actual.save(tmp_path / "roundtrip.grim"))
    np.testing.assert_array_equal(restored.rcs, actual.rcs)
    view = np.asarray(actual._complex)[0]
    del actual
    gc.collect()
    assert list(cache.iterdir())
    np.testing.assert_array_equal(view, expected.rcs[0])
    del view
    gc.collect()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("extension", ["pio", "ss"])
def test_binary_memory_budget_estimate_and_bounded_disk_decode(tmp_path, extension):
    cache = tmp_path / "cache"
    cache.mkdir()
    source = tmp_path / ("large." + extension)
    if extension == "pio":
        # 16 MiB of complex output; loader scratch is bounded at 32K cells.
        field = np.ones((512, 1, 2048, 1), dtype=np.complex128) * (1 + 2j)
        grid = RcsGrid(np.arange(512) * .1, [0.], np.arange(2048) * .01 + 8., ["VV"], rcs=field)
        grid.save_pio(source, precision="double")
    else:
        fields = np.ones((16, 32_768, 4), dtype=np.complex64) * (1 + 2j)
        _write_samples(source, fields, [(float(i), 0.) for i in range(16)])
    memory = estimate_binary_import(source)
    disk = estimate_binary_import(source, storage="disk")
    assert disk["estimated_peak_bytes"] < memory["retained_bytes"]
    budget = disk["estimated_peak_bytes"] + 4096
    with pytest.raises(MemoryError):
        load_dataset(source, max_output_bytes=budget)
    tracemalloc.start()
    actual = load_dataset(source, storage="disk", cache_dir=cache, max_output_bytes=budget)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert actual.sample_storage == "disk"
    assert peak < memory["retained_bytes"] * .5
    np.testing.assert_array_equal(actual.rcs_slice((0, 0, slice(0, 8), 0)), 1 + 2j)
    del actual
    gc.collect()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("extension", ["pio", "ss"])
def test_invalid_binary_samples_cleanup_private_outputs(tmp_path, extension):
    source = tmp_path / ("bad." + extension)
    if extension == "pio":
        pio_fixtures.PioStreamingTests._write_pio_fixture(source, payload=np.asarray([np.inf, 1], dtype="<f4").tobytes())
    else:
        _write_samples(source, np.full((1, 2, 4), np.inf + 1j, np.complex64), [(0., 0.)])
    cache = tmp_path / "cache"
    cache.mkdir()
    try:
        load_dataset(source, storage="disk", cache_dir=cache)
    except ValueError as error:
        assert "infinite" in str(error)
    else:
        raise AssertionError("invalid samples were accepted")
    gc.collect()
    assert not list(cache.iterdir())


def test_binary_selection_is_still_explicitly_native_only(tmp_path):
    with pytest.raises(ValueError, match="supported for .grim"):
        load_dataset(tmp_path / "example.pio", selection={"frequency": (8, 9)})


@pytest.mark.parametrize("component", ["real", "imag"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_ss_rejects_either_one_sided_nan_without_losing_other_component(tmp_path, component, storage):
    source = tmp_path / "one-sided.ss"
    values = np.ones((1, 2, 4), dtype=np.complex64) * (1 + 2j)
    getattr(values, component)[0, 0, 0] = np.nan
    _write_samples(source, values, [(0., 0.)])
    with pytest.raises(ValueError, match="one-sided missing"):
        load_dataset(source, storage=storage, cache_dir=tmp_path)
