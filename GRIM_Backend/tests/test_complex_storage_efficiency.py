"""Storage-layer efficiency contracts for canonical complex samples.

These pin the allocation and validation behaviour that the complex-storage
migration enables: power reads that never copy the field, single-copy
advanced-index reads, exact rearrangements that adopt their fresh samples,
raw solver subsets that stay raw, and raw-pair checks that cost O(1) per read.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import duplicate_dataset


def _field(shape, dtype=np.complex128, seed=0):
    rng = np.random.default_rng(seed)
    field = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    field[0, 0, 0, 0] = np.nan + 1j * np.nan  # a missing sample
    field[1, 0, 0, 0] = 0.0  # a known zero field
    return field


def _axes(shape):
    return (
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        np.linspace(8.0, 12.0, shape[2]),
        ["VV", "HH"][: shape[3]],
    )


def _sigma_units(quantity):
    return {
        "frequency": "GHz",
        "rcs_linear_quantity": quantity,
        "rcs_log_unit": "dBke" if quantity == "sigma_2d" else "dBsm",
    }


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_power_reads_match_squared_magnitude_without_copying_the_field(dtype):
    shape = (256, 4, 64, 2)
    field = _field(shape, dtype)
    ds = RcsGrid(*_axes(shape), rcs=field)
    real_dtype = np.dtype(np.float32 if dtype == np.complex64 else np.float64)
    tolerance = 4 * np.finfo(real_dtype).eps
    expected = field.real.astype(np.float64) ** 2 + field.imag.astype(np.float64) ** 2

    full = np.asarray(ds.power)
    assert full.dtype == real_dtype
    assert not full.flags.writeable
    np.testing.assert_allclose(full, expected, rtol=tolerance, atol=0)
    assert np.isnan(full[0, 0, 0, 0]) and full[1, 0, 0, 0] == 0.0

    basic = ds.power[2:5, 0, :, 1]
    np.testing.assert_allclose(basic, expected[2:5, 0, :, 1], rtol=tolerance, atol=0)
    advanced = np.ix_([1, 3], [0], [0, 7], [1])
    np.testing.assert_allclose(ds.power[advanced], expected[advanced], rtol=tolerance, atol=0)
    assert ds.power[3, 1, 2, 0] == pytest.approx(expected[3, 1, 2, 0], rel=tolerance)
    np.testing.assert_allclose(ds.power.flat[3:5], expected.reshape(-1)[3:5], rtol=tolerance, atol=0)

    # The result plus one scratch array; the former path also copied the
    # complex field and took hypot before squaring (about 4x the result).
    result_bytes = field.size * real_dtype.itemsize
    assert _peak_bytes(lambda: np.asarray(ds.power)) < 2.5 * result_bytes


def test_phase_reads_do_not_copy_the_field():
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field)
    phase = np.asarray(ds.phase)
    expected = np.where(np.isnan(field.real), np.nan, np.angle(field))
    np.testing.assert_array_equal(phase, expected)
    # Previously a full copy of the complex field preceded ``angle``.
    assert _peak_bytes(lambda: np.asarray(ds.phase)) < 4 * field.size * 8


def test_advanced_index_field_reads_allocate_once_and_basic_slices_still_copy():
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field)
    selection = np.ix_(np.arange(0, 256, 2), [0, 1, 2, 3], np.arange(64), [0, 1])
    result = ds.rcs_slice(selection)
    assert result.flags.owndata and result.flags.writeable
    np.testing.assert_array_equal(result, field[selection])
    # One owned result plus NumPy's own advanced-indexing scratch; the former
    # extra copy made this at least 2x the result size.
    assert _peak_bytes(lambda: ds.rcs_slice(selection)) < 1.75 * result.nbytes

    view_read = ds.rcs_slice((slice(0, 10), 0, slice(None), 0))
    assert view_read.flags.owndata and view_read.flags.writeable
    view_read[...] = 123.0
    np.testing.assert_array_equal(ds.rcs_slice((slice(0, 10), 0, slice(None), 0)), field[0:10, 0, :, 0])

    scalar = ds.rcs_slice((2, 0, 0, 0))
    assert np.ndim(scalar) == 0 and scalar == field[2, 0, 0, 0]


def test_sample_kwargs_return_owned_arrays_and_tag_raw_fields():
    shape = (16, 2, 4, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field)
    samples = ds._sample_kwargs((slice(0, 4), 0, slice(None), 0))
    assert set(samples) == {"rcs"}
    assert samples["rcs"].flags.owndata and samples["rcs"].flags.writeable
    samples["rcs"][...] = 0
    np.testing.assert_array_equal(ds.rcs_slice((slice(0, 4), 0, slice(None), 0)), field[0:4, 0, :, 0])
    swapped = ds._sample_kwargs(transform=lambda value: np.swapaxes(value, 0, 1))["rcs"]
    assert swapped.flags.owndata and swapped.flags.writeable
    np.testing.assert_array_equal(swapped, np.swapaxes(field, 0, 1))

    raw = RcsGrid(*_axes(shape), rcs=field, rcs_storage_domain="raw_field", units=_sigma_units("sigma_3d"))
    raw_samples = raw._sample_kwargs()
    assert raw_samples["rcs_storage_domain"] == "raw_field"
    np.testing.assert_array_equal(raw_samples["rcs"], field)  # unnormalized solver field
    assert raw_samples["rcs"].flags.owndata

    power_only = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2)
    power_samples = power_only._sample_kwargs()
    assert set(power_samples) == {"rcs_power"}
    assert power_samples["rcs_power"].flags.owndata and power_samples["rcs_power"].flags.writeable


def test_exact_rearrangements_adopt_their_fresh_samples():
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field, units={"frequency": "GHz"})

    copy = duplicate_dataset(ds)
    assert copy.storage_kind == "complex"
    np.testing.assert_array_equal(copy.rcs, field)
    assert not np.shares_memory(copy._complex, ds._complex)
    assert _peak_bytes(lambda: duplicate_dataset(ds)) < 1.5 * field.nbytes  # was about 2.2x

    half_azimuths = np.arange(128, dtype=float)
    half = ds.axis_crop(azimuths=half_azimuths)
    np.testing.assert_array_equal(half.rcs, field[:128])
    assert _peak_bytes(lambda: ds.axis_crop(azimuths=half_azimuths)) < 1.5 * field.nbytes / 2

    wrapped = ds.wrap_phase("0_360")
    assert wrapped.units["phase_wrap"] == "0_360"
    np.testing.assert_array_equal(wrapped.rcs, field)
    phases = np.asarray(wrapped.phase)
    finite = np.isfinite(phases)
    assert np.all((phases[finite] >= 0.0) & (phases[finite] < 2 * np.pi))
    # No grid-sized phase array is built and discarded for complex storage.
    assert _peak_bytes(lambda: ds.wrap_phase("0_360")) < 1.5 * field.nbytes  # was about 2.7x


def test_power_phase_wrap_still_rewraps_the_stored_phase():
    shape = (8, 1, 2, 1)
    power = np.full(shape, 4.0)
    phase = np.linspace(-3 * np.pi, 3 * np.pi, power.size).reshape(shape)  # unwrapped turns
    ds = RcsGrid(*_axes(shape), rcs_power=power, rcs_phase=phase)
    assert ds.storage_kind == "power_phase"
    wrapped = ds.wrap_phase("-180_180")
    expected = np.remainder(phase + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(np.asarray(wrapped.phase), expected, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(wrapped.power), power)
    np.testing.assert_array_equal(np.asarray(ds.phase), phase)  # source untouched


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_exact_subsets_of_raw_fields_keep_raw_authority_without_normalizing(quantity):
    shape = (256, 4, 64, 2)
    raw = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=raw, rcs_storage_domain="raw_field", units=_sigma_units(quantity))
    assert ds.storage_domain == "raw_field"
    physical = ds.rcs

    half_azimuths = np.arange(128, dtype=float)
    half = ds.axis_crop(azimuths=half_azimuths)
    assert half.storage_kind == "complex" and half.storage_domain == "raw_field"
    assert half.real_dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(half.extra["rcs_amp_real"], raw.real[:128])
    np.testing.assert_array_equal(half.extra["rcs_amp_imag"], raw.imag[:128])
    np.testing.assert_array_equal(half.rcs, physical[:128])
    # One owned copy of the raw slice: no normalized field, no derived
    # power/phase pair and no consistency validation pass (about 5x before).
    assert _peak_bytes(lambda: ds.axis_crop(azimuths=half_azimuths)) < 1.5 * raw.nbytes / 2

    copy = duplicate_dataset(ds)
    assert copy.storage_domain == "raw_field"
    np.testing.assert_array_equal(copy.rcs, physical)
    np.testing.assert_array_equal(copy.extra["rcs_amp_imag"], raw.imag)

    wrapped = ds.wrap_phase("0_360")
    assert wrapped.storage_domain == "raw_field" and wrapped.units["phase_wrap"] == "0_360"
    np.testing.assert_array_equal(wrapped.rcs, physical)

    swapped = ds.swap_elevation_azimuth()
    assert swapped.storage_domain == "raw_field"
    np.testing.assert_array_equal(swapped.rcs, np.swapaxes(physical, 0, 1))

    shifted = ds.shift_azimuth(10.0)
    assert shifted.storage_domain == "raw_field"
    np.testing.assert_array_equal(shifted.rcs, physical)


def test_frequency_relabel_rescales_raw_samples_and_keeps_the_physical_response():
    shape = (8, 1, 3, 1)
    raw = _field(shape)
    frequencies = np.array([10.04, 10.16, 10.27])
    ds = RcsGrid(np.arange(8.0), [0.0], frequencies, ["VV"], rcs=raw,
                 rcs_storage_domain="raw_field", units=_sigma_units("sigma_2d"))
    before = ds.rcs.copy()

    rounded = ds.round_frequencies(1)
    assert rounded.storage_domain == "raw_field"
    np.testing.assert_allclose(rounded.rcs, before, rtol=8e-15, atol=0)
    scale = np.sqrt(np.round(frequencies, 1) / frequencies)[None, None, :, None]
    np.testing.assert_allclose(rounded.extra["rcs_amp_real"], raw.real * scale, rtol=8e-15, atol=0)
    np.testing.assert_allclose(rounded.extra["rcs_amp_imag"], raw.imag * scale, rtol=8e-15, atol=0)
    assert "rescaled" in rounded.extra["frequency_relabel_normalization"]
    assert rounded.audit()["errors"] == []

    edited = ds.edit_axis_value("frequency", 0, 10.3)
    order = np.argsort([10.3, 10.16, 10.27], kind="stable")
    assert edited.storage_domain == "raw_field"
    np.testing.assert_allclose(edited.rcs, before[:, :, order, :], rtol=8e-15, atol=0)

    # A relabel to a nonpositive frequency keeps the physical response and
    # drops the raw encoding, which that frequency cannot normalise.
    small = RcsGrid(np.arange(8.0), [0.0], np.array([0.04]), ["VV"], rcs=raw[:, :, :1, :],
                    rcs_storage_domain="raw_field", units=_sigma_units("sigma_2d"))
    result = small.round_frequencies(1)
    assert result.frequencies.item() == 0
    assert result.storage_kind == "complex" and result.storage_domain == "physical"
    assert "rcs_amp_real" not in result.extra and "rcs_amp_imag" not in result.extra
    np.testing.assert_allclose(result.rcs, small.rcs, rtol=8e-15, atol=0)


def test_overlap_of_raw_fields_keeps_raw_authority_and_masks_missing_cells():
    shape = (16, 1, 4, 1)
    raw_a = _field(shape, seed=1)
    raw_b = _field(shape, seed=2)
    frequencies = np.linspace(8.0, 12.0, 4)
    a = RcsGrid(np.arange(16.0), [0.0], frequencies, ["VV"], rcs=raw_a,
                rcs_storage_domain="raw_field", units=_sigma_units("sigma_3d"))
    b = RcsGrid(np.arange(4.0, 20.0), [0.0], frequencies, ["VV"], rcs=raw_b,
                rcs_storage_domain="raw_field", units=_sigma_units("sigma_3d"))
    overlap_a, overlap_b = RcsGrid.overlap_many(a, b)
    assert overlap_a.storage_domain == overlap_b.storage_domain == "raw_field"
    np.testing.assert_array_equal(overlap_a.azimuths, np.arange(4.0, 16.0))
    expected_a = a.rcs[4:16].copy()
    expected_a[0, 0, 0, 0] = np.nan + 1j * np.nan  # b has no sample at azimuth 4, 8 GHz
    np.testing.assert_array_equal(overlap_a.rcs, expected_a)
    expected_b = b.rcs[:12].copy()
    expected_b[0, 0, 0, 0] = np.nan + 1j * np.nan
    np.testing.assert_array_equal(overlap_b.rcs, expected_b)
    assert np.isnan(overlap_a.extra["rcs_amp_real"][0, 0, 0, 0])
    np.testing.assert_array_equal(overlap_a.extra["rcs_amp_real"][1:], raw_a.real[5:16])


def test_unusable_raw_pair_does_not_rescan_the_grid_on_every_read(monkeypatch):
    shape = (64, 1, 4, 1)
    field = _field(shape)
    # A power ratio has no raw normalization, so the pair stays metadata and
    # every read used to re-run a grid-sized coverage scan before ignoring it.
    ds = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2, rcs_phase=np.angle(field),
                 units={"rcs_linear_quantity": "power_ratio", "rcs_log_unit": "dB", "frequency": "GHz"},
                 extra={"rcs_amp_real": field.real.copy(), "rcs_amp_imag": field.imag.copy(),
                        "raw_complex_amplitude_preserved": True})
    assert ds.storage_kind == "complex" and ds.storage_domain == "physical"
    assert "rcs_amp_real" in ds.extra
    scans = []
    original = RcsGrid._raw_pair_coverage_matches

    def counting(self, real, imag):
        scans.append(1)
        return original(self, real, imag)

    monkeypatch.setattr(RcsGrid, "_raw_pair_coverage_matches", counting)
    for _ in range(3):
        ds.power[0, 0, 0, 0]
        ds.phase[:, 0, 0, 0]
        ds.rcs_slice((1, 0, 0, 0))
        assert ds.real_dtype == np.dtype(np.float64)
        ds.native_sample_payload()
        ds._bounded_complex_slice_reader()
    assert scans == []
    np.testing.assert_allclose(ds.power[2, 0, 1, 0], np.abs(field[2, 0, 1, 0]) ** 2, rtol=1e-12)


def test_coverage_scan_for_an_authoritative_pair_is_memoised_until_arrays_change(monkeypatch):
    shape = (32, 1, 2, 1)
    raw = _field(shape)
    # Power deliberately disagrees with the pair, so GRIM keeps power/phase for
    # audit while reads still prefer the normalized raw field.
    ds = RcsGrid(*_axes(shape), rcs_power=2.0 * 4 * np.pi * np.abs(raw) ** 2, rcs_phase=np.angle(raw),
                 units=_sigma_units("sigma_3d"),
                 extra={"rcs_amp_real": raw.real.copy(), "rcs_amp_imag": raw.imag.copy(),
                        "raw_complex_amplitude_preserved": True})
    assert ds.storage_kind == "power_phase"
    scans = []
    original = RcsGrid._scan_raw_pair_coverage

    def counting(real, imag, source):
        scans.append(1)
        return original(real, imag, source)

    monkeypatch.setattr(RcsGrid, "_scan_raw_pair_coverage", staticmethod(counting))
    first = ds.rcs_slice((3, 0, 0, 0))
    ds.rcs_slice((4, 0, 1, 0))
    ds._bounded_complex_slice_reader()
    assert scans == []  # the construction-time verdict is reused
    np.testing.assert_allclose(first, raw[3, 0, 0, 0] * np.sqrt(4 * np.pi), rtol=1e-15)

    ds.extra["rcs_amp_real"] = ds.extra["rcs_amp_real"].copy()  # a replaced array is re-validated once
    ds.rcs_slice((3, 0, 0, 0))
    ds.rcs_slice((5, 0, 0, 0))
    assert scans == [1]


def test_coherent_subtract_adopts_its_sanitized_output():
    shape = (64, 1, 8, 1)
    left = RcsGrid(*_axes(shape), rcs=_field(shape, seed=3), units={"frequency": "GHz"})
    right = RcsGrid(*_axes(shape), rcs=_field(shape, seed=4), units={"frequency": "GHz"})
    result = left.coherent_subtract(right)
    expected = left.rcs - right.rcs
    assert result.storage_kind == "complex"
    np.testing.assert_array_equal(result.rcs, expected)
    assert np.isnan(result.rcs[0, 0, 0, 0])


def test_raw_normalization_predicate_mirrors_the_pair_normalizer():
    shape = (4, 1, 2, 1)
    raw = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=raw, rcs_storage_domain="raw_field", units=_sigma_units("sigma_2d"))
    assert ds._raw_normalization_defined()
    k0 = 2 * np.pi * ds.frequencies * 1e9 / C0
    np.testing.assert_allclose(ds.rcs, raw / (2 * np.sqrt(k0))[None, None, :, None], rtol=4e-16)
    ratio = RcsGrid(*_axes(shape), rcs=raw, units={"rcs_linear_quantity": "power_ratio", "rcs_log_unit": "dB"})
    assert not ratio._raw_normalization_defined()
    with pytest.raises(ValueError, match="normalization is not defined"):
        RcsGrid(*_axes(shape), rcs=raw, rcs_storage_domain="raw_field",
                units={"rcs_linear_quantity": "power_ratio", "rcs_log_unit": "dB"})
