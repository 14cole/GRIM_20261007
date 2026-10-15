"""Storage-layer efficiency contracts for canonical complex samples.

These pin the allocation and validation behaviour that the complex-storage
migration enables: power reads that never copy the field, single-copy
advanced-index reads, exact rearrangements that adopt their fresh samples,
raw solver subsets that stay raw, raw-pair checks that cost O(1) per read,
phase-availability answers that read no sample of canonical storage, native
loads that adopt the decoded field, and phase reads with a single scratch mask.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.transforms import (
    convert_extrusion,
    duplicate_dataset,
    translate_phase_center,
)
from GRIM_Backend.reports.plot_data import NamedGrid, _phase_capability


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


def test_phase_coverage_answers_from_storage_kind_and_scans_only_polar_pairs(monkeypatch):
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field)
    assert ds.storage_kind == "complex" and ds.phase_coverage() == "complete"
    assert ds._unknown_phase_count() == 0
    power_only = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2)
    assert power_only.storage_kind == "power" and power_only.phase_coverage() == "none"
    assert power_only._unknown_phase_count() == field.size - 1  # every finite sample
    # Neither canonical kind reads a sample (about 2 MiB of full-grid masks before).
    assert _peak_bytes(ds.phase_coverage) < 16 * 1024
    assert _peak_bytes(power_only.phase_coverage) < 16 * 1024

    power = np.abs(field) ** 2
    phase = np.angle(field)
    phase[5:9, 0, :, 0] = np.nan  # finite power without a phase
    mixed = RcsGrid(*_axes(shape), rcs_power=power, rcs_phase=phase)
    assert mixed.storage_kind == "power_phase" and mixed.phase_coverage() == "partial"
    assert mixed._unknown_phase_count() == 4 * 64
    # The scan is bounded by the validation block, not by the grid.
    monkeypatch.setattr("GRIM_Backend.datasets.storage._RAW_COMPLEX_VALIDATION_BLOCK_CELLS", 4096)
    assert mixed.phase_coverage() == "partial" and mixed._unknown_phase_count() == 4 * 64
    assert _peak_bytes(mixed.phase_coverage) < 64 * 1024
    assert _peak_bytes(mixed._unknown_phase_count) < 64 * 1024

    exceptional_power, exceptional_phase = power.copy(), np.angle(field)
    exceptional_power[3, 0, 0, 0] = 0.0
    exceptional_phase[3, 0, 0, 0] = 1.0  # a nonzero phase at zero power keeps polar storage
    exceptional = RcsGrid(*_axes(shape), rcs_power=exceptional_power, rcs_phase=exceptional_phase)
    assert exceptional.storage_kind == "power_phase" and exceptional.phase_coverage() == "complete"

    small = (4, 1, 2, 1)
    empty = RcsGrid(*_axes(small), rcs_power=np.full(small, np.nan))
    assert empty.phase_coverage() == "none" and empty._unknown_phase_count() == 0
    legacy = RcsGrid(*_axes(small), rcs=_field(small))
    legacy.rcs_phase[2, 0, 0, 0] = np.nan  # a legacy edit removes one phase
    assert legacy.storage_kind == "legacy_power_phase"
    assert legacy.phase_coverage() == "partial" and legacy._unknown_phase_count() == 1


def test_phase_availability_checks_read_no_samples_of_complex_grids(monkeypatch):
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field, units=_sigma_units("sigma_3d"))
    other = RcsGrid(*_axes(shape), rcs=_field(shape, seed=1), units=_sigma_units("sigma_3d"))
    reads = []
    for name in ("power_slice", "phase_slice"):
        original = getattr(RcsGrid, name)

        def counting(self, selection=Ellipsis, *, _name=name, _original=original):
            reads.append((_name, selection))
            return _original(self, selection)

        monkeypatch.setattr(RcsGrid, name, counting)

    ds._assert_compatible(other, coherent=True)
    assert _phase_capability((NamedGrid("a", ds), NamedGrid("b", other)))[0]
    moved = translate_phase_center(ds, x_m=0.1, y_m=0.0, z_m=0.0)
    assert moved.storage_kind == "complex"
    assert reads == []
    # Each check used to build full power and phase arrays (about 2.4 MiB here).
    assert _peak_bytes(lambda: ds._assert_compatible(other, coherent=True)) < 16 * 1024
    assert _peak_bytes(lambda: _phase_capability((NamedGrid("a", ds),))) < 16 * 1024

    power_only = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2, units=_sigma_units("sigma_3d"))
    with pytest.raises(ValueError, match=r"requires phase; right grid has \d+ finite-power sample"):
        ds._assert_compatible(power_only, coherent=True)
    assert _phase_capability((NamedGrid("p", power_only),))[2].startswith(
        "No finite stored phase is available for: p"
    )
    with pytest.raises(ValueError, match="needs complex"):
        translate_phase_center(power_only, x_m=0.1, y_m=0.0, z_m=0.0)

    phase = np.angle(field)
    phase[5:9, 0, :, 0] = np.nan
    mixed = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2, rcs_phase=phase,
                    units=_sigma_units("sigma_3d"))
    assert _phase_capability((NamedGrid("m", mixed),))[0]  # partial phase still plots
    with pytest.raises(ValueError, match=r"left grid has 256 finite-power sample\(s\) with unknown phase"):
        mixed._assert_compatible(ds, coherent=True)
    shifted = translate_phase_center(mixed, x_m=0.1, y_m=0.0, z_m=0.0)
    assert shifted.storage_kind == "power_phase"
    np.testing.assert_array_equal(np.isnan(np.asarray(shifted.phase)), np.isnan(phase))


def test_convert_extrusion_scales_the_complex_field_without_a_power_grid(monkeypatch):
    shape = (256, 4, 64, 2)
    field = _field(shape)
    ds = RcsGrid(*_axes(shape), rcs=field, units=_sigma_units("sigma_3d"))
    reads = []
    original = RcsGrid.power_slice
    monkeypatch.setattr(RcsGrid, "power_slice",
                        lambda self, selection=Ellipsis: reads.append(selection) or original(self, selection))
    width = convert_extrusion(ds, to="dbke", length_m=2.0)
    assert reads == []
    assert width.storage_kind == "complex" and width.linear_quantity() == "sigma_2d"
    frequency_hz = np.asarray(ds.frequencies, dtype=float) * 1.0e9
    scale = (C0 / (2.0 * 2.0 * 2.0 * frequency_hz)).reshape(1, 1, -1, 1)
    np.testing.assert_array_equal(width.rcs, field * np.sqrt(scale))
    # Field copy, scaled field and the constructor's copy; the unused power
    # grid added half a field (about 2.7x before).
    assert _peak_bytes(lambda: convert_extrusion(ds, to="dbke", length_m=2.0)) < 2.5 * field.nbytes

    phase = np.angle(field)
    phase[5:9, 0, :, 0] = np.nan
    polar = RcsGrid(*_axes(shape), rcs_power=np.abs(field) ** 2, rcs_phase=phase,
                    units=_sigma_units("sigma_3d"))
    assert polar.storage_kind == "power_phase"
    converted = convert_extrusion(polar, to="dbke", length_m=2.0)
    np.testing.assert_array_equal(np.asarray(converted.power), np.asarray(polar.power) * scale)
    np.testing.assert_array_equal(np.asarray(converted.phase), np.asarray(polar.phase))


@pytest.mark.parametrize("compressed", [False, True])
def test_native_load_adopts_the_decoded_complex_field(tmp_path, monkeypatch, compressed):
    shape = (512, 4, 64, 2)  # 4 MiB of complex128 samples
    field = _field(shape)
    # Validation blocks are a constant, not a fraction of the grid; shrink
    # them so the bounds below reflect a large archive.
    monkeypatch.setattr("GRIM_Backend.io.native._RAW_COMPLEX_VALIDATION_BLOCK_CELLS", 8192)
    physical = RcsGrid(*_axes(shape), rcs=field, units=_sigma_units("sigma_3d"))
    path = physical.save(tmp_path / "physical", compressed=compressed)
    restored = RcsGrid.load(path)
    assert restored.storage_kind == "complex" and restored.storage_domain == "physical"
    np.testing.assert_array_equal(restored.rcs, field)
    assert restored._complex.flags.owndata and not restored._complex.flags.writeable
    # The decoded member becomes the stored field: no constructor copy and no
    # grid-sized masks (about 2.2x before).  Archive decoding itself keeps a
    # few hundred KiB of transient buffers, hence the minimum of two runs.
    assert min(_peak_bytes(lambda: RcsGrid.load(path)) for _ in range(2)) < 1.6 * field.nbytes

    raw = RcsGrid(*_axes(shape), rcs=field, rcs_storage_domain="raw_field", units=_sigma_units("sigma_3d"))
    raw_path = raw.save(tmp_path / "raw", compressed=compressed)
    restored_raw = RcsGrid.load(raw_path)
    assert restored_raw.storage_kind == "complex" and restored_raw.storage_domain == "raw_field"
    np.testing.assert_array_equal(restored_raw.rcs, raw.rcs)
    np.testing.assert_array_equal(restored_raw.extra["rcs_amp_imag"], field.imag)
    assert np.shares_memory(restored_raw.extra["rcs_amp_real"], restored_raw._complex)
    assert min(_peak_bytes(lambda: RcsGrid.load(raw_path)) for _ in range(2)) < 1.6 * field.nbytes


def test_native_arrays_from_a_mapping_are_copied_not_adopted():
    shape = (16, 2, 4, 2)
    field = _field(shape)
    grid = RcsGrid(*_axes(shape), rcs=field)
    payload = grid._native_archive_payload()
    assert payload["rcs_complex"] is grid._complex  # the mapping holds the live array
    restored = RcsGrid._from_native_arrays(payload)
    np.testing.assert_array_equal(restored.rcs, field)
    assert not np.shares_memory(restored._complex, grid._complex)


def _former_phase(field, wrap, real_dtype):
    """The phase formula before the in-place rewrite, kept to pin identical values."""
    result = np.angle(field)
    result = np.where(np.isfinite(field.real) & np.isfinite(field.imag), result, np.nan)
    if wrap == "0_360":
        result = np.where(result < 0, result + 2 * np.pi, result)
        result = np.where(result >= 2 * np.pi, 0., result)
    elif wrap == "-180_180":
        result = np.where(result >= np.pi, -np.pi, result)
    return result.astype(real_dtype, copy=False)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("wrap", ["", "0_360", "-180_180"])
def test_phase_reads_are_bit_identical_with_one_scratch_mask(dtype, wrap):
    shape = (256, 4, 64, 2)
    field = _field(shape, dtype)
    field[2, 0, 0, 0] = -1.0 + 0.0j            # exactly pi
    field[3, 0, 0, 0] = complex(-1.0, -0.0)    # exactly -pi
    field[4, 0, 0, 0] = complex(1.0, -0.0)     # negative zero
    field[5, 0, 0, 0] = complex(1.0, -1e-17)   # wraps up to exactly 2*pi
    real_dtype = np.dtype(np.float32 if dtype == np.complex64 else np.float64)
    ds = RcsGrid(*_axes(shape), rcs=field, units={"phase_wrap": wrap} if wrap else {})
    expected = _former_phase(ds.rcs, wrap, real_dtype)
    phase = np.asarray(ds.phase)
    assert phase.dtype == real_dtype and not phase.flags.writeable
    np.testing.assert_array_equal(phase, expected)
    assert np.isnan(phase[0, 0, 0, 0])
    if wrap == "0_360":
        assert phase[5, 0, 0, 0] == 0.0 and phase[3, 0, 0, 0] == expected[3, 0, 0, 0]
    elif wrap == "-180_180":
        assert phase[2, 0, 0, 0] == -np.pi and phase[3, 0, 0, 0] == -np.pi
    np.testing.assert_array_equal(ds.phase[2:6, 0, :, 1], expected[2:6, 0, :, 1])
    selection = np.ix_([1, 3], [0], [0, 7], [1])
    np.testing.assert_array_equal(ds.phase[selection], expected[selection])
    scalar = ds.phase[3, 0, 0, 0]
    assert np.ndim(scalar) == 0 and scalar == expected[3, 0, 0, 0]
    # One result plus one boolean scratch array (about 1.13x the result for
    # float64, 1.25x for float32); the former angle/where chain peaked at
    # 2.1x to 3.3x.
    assert _peak_bytes(lambda: np.asarray(ds.phase)) < 1.5 * field.size * real_dtype.itemsize


@pytest.mark.parametrize("kind", ["complex", "raw_field", "power_phase", "legacy"])
def test_power_phase_slice_matches_the_separate_reads_from_one_field_read(kind, monkeypatch):
    shape = (64, 2, 16, 2)
    field = _field(shape)
    if kind == "raw_field":
        ds = RcsGrid(*_axes(shape), rcs=field, rcs_storage_domain="raw_field",
                     units=_sigma_units("sigma_2d"))
    elif kind == "power_phase":
        power, phase = np.abs(field) ** 2, np.angle(field)
        phase[2, 0, :, 0] = np.nan  # finite power without a phase
        power[0, 0, 0, 0] = phase[0, 0, 0, 0] = np.nan
        ds = RcsGrid(*_axes(shape), rcs_power=power, rcs_phase=phase)
        assert ds.storage_kind == "power_phase"
    else:
        ds = RcsGrid(*_axes(shape), rcs=field)
        if kind == "legacy":
            ds.rcs_power  # expose the writable arrays
            assert ds._legacy_samples
    for selection in (Ellipsis, (slice(None), 1, slice(None), 0),
                      np.ix_([1, 5], [0], [0, 3], [1]), (3, 0, 2, 1)):
        power, phase = ds.power_phase_slice(selection)
        np.testing.assert_array_equal(power, ds.power_slice(selection))
        np.testing.assert_array_equal(phase, ds.phase_slice(selection))
        assert not np.asarray(power).flags.writeable and not np.asarray(phase).flags.writeable
    if kind in ("complex", "raw_field"):
        reads = []
        original = RcsGrid._complex_samples

        def counting(self, selection, *, owned):
            reads.append(owned)
            return original(self, selection, owned=owned)

        monkeypatch.setattr(RcsGrid, "_complex_samples", counting)
        ds.power_phase_slice((slice(None), 0, slice(None), 0))
        assert reads == [False]


def _overlapping_pair(seed, *, raw_field, scale=1.0):
    """Two 2-D width grids sharing 16 azimuths with identical samples there."""
    shape = (96, 1, 32, 2)
    field = _field(shape, seed=seed)
    azimuths = np.arange(shape[0], dtype=float)
    frequencies = np.linspace(8.0, 12.0, shape[2])
    units = _sigma_units("sigma_2d")
    kwargs = {"rcs_storage_domain": "raw_field"} if raw_field else {}
    left = RcsGrid(azimuths[:64], [0.0], frequencies, ["VV", "HH"],
                   rcs=field[:64].copy(), units=units, **kwargs)
    right = RcsGrid(azimuths[48:], [0.0], frequencies, ["VV", "HH"],
                    rcs=(field[48:] * scale).copy(), units=units, **kwargs)
    return left, right


def _former_join(left, right, **kwargs):
    """The power/phase merge, reached by exposing the legacy arrays first."""
    left.rcs_power
    right.rcs_power
    return RcsGrid.join_many(left, right, **kwargs)


@pytest.mark.parametrize("raw_field", [True, False])
def test_join_of_complex_inputs_merges_the_stored_field(raw_field):
    joined = RcsGrid.join_many(*_overlapping_pair(1, raw_field=raw_field))
    former = _former_join(*_overlapping_pair(1, raw_field=raw_field))
    assert joined.storage_kind == "complex"
    assert joined.storage_domain == ("raw_field" if raw_field else "physical")
    assert joined.azimuths.size == 96
    np.testing.assert_allclose(joined.rcs, former.rcs, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.isnan(joined.rcs.real), np.isnan(former.rcs.real))
    if raw_field:
        # The raw solver field itself is merged, so the raw pair is exact.
        np.testing.assert_array_equal(joined.extra["rcs_amp_real"], former.extra["rcs_amp_real"])
        np.testing.assert_array_equal(joined.extra["rcs_amp_imag"], former.extra["rcs_amp_imag"])
    else:
        assert "rcs_amp_real" not in joined.extra
    for overlap in ("first", "last"):
        left, right = _overlapping_pair(2, raw_field=raw_field, scale=2.0)
        policy = RcsGrid.join_many(left, right, overlap=overlap)
        source = left if overlap == "first" else right
        np.testing.assert_allclose(policy.rcs[48:64], source.rcs[-16:] if overlap == "first" else source.rcs[:16], rtol=1e-12)
        former_policy = _former_join(*_overlapping_pair(2, raw_field=raw_field, scale=2.0), overlap=overlap)
        np.testing.assert_allclose(policy.rcs, former_policy.rcs, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="conflicting finite samples overlap during join"):
        RcsGrid.join_many(*_overlapping_pair(3, raw_field=raw_field, scale=1.0 + 1e-3))


def test_join_of_physical_fields_keeps_the_former_tolerances():
    left, right = _overlapping_pair(4, raw_field=False)
    nudged = right.rcs
    nudged[:16] *= np.exp(1j * 5e-6)  # inside the 1e-5 rad phase rule
    RcsGrid.join_many(left, RcsGrid(right.azimuths, right.elevations, right.frequencies,
                                    right.polarizations, rcs=nudged, units=right.units))
    nudged = right.rcs
    nudged[:16] *= np.exp(1j * 2e-5)  # outside it
    with pytest.raises(ValueError, match="conflicting finite samples overlap during join"):
        RcsGrid.join_many(left, RcsGrid(right.azimuths, right.elevations, right.frequencies,
                                        right.polarizations, rcs=nudged, units=right.units))
    nudged = right.rcs
    nudged[:16] *= 1.0 + 2e-6  # outside the 1e-6 relative power rule (power scales twice)
    with pytest.raises(ValueError, match="conflicting finite samples overlap during join"):
        RcsGrid.join_many(left, RcsGrid(right.azimuths, right.elevations, right.frequencies,
                                        right.polarizations, rcs=nudged, units=right.units))


def test_mixed_storage_join_still_uses_the_power_and_phase_merge():
    def pair():
        left, _ = _overlapping_pair(5, raw_field=True)
        whole = RcsGrid(np.arange(96, dtype=float), [0.0], left.frequencies, ["VV", "HH"],
                        rcs=_field((96, 1, 32, 2), seed=5), rcs_storage_domain="raw_field",
                        units=left.units)
        # The same physical response, stored normalized rather than raw.
        right = RcsGrid(whole.azimuths[48:], [0.0], left.frequencies, ["VV", "HH"],
                        rcs=whole.rcs[48:], units=left.units)
        return left, right

    left, right = pair()
    assert RcsGrid._complex_join_domain([left, right]) is None
    joined = RcsGrid.join_many(left, right)
    former = _former_join(*pair())
    np.testing.assert_allclose(joined.rcs, former.rcs, rtol=1e-12, atol=0)
    assert "rcs_amp_real" not in joined.extra


def test_complex_join_retains_one_field_instead_of_power_phase_and_raw_arrays():
    shape = (256, 1, 256, 2)
    field = _field(shape, seed=6)
    azimuths = np.arange(shape[0], dtype=float)
    frequencies = np.linspace(8.0, 12.0, shape[2])
    units = _sigma_units("sigma_2d")

    def pair():
        return (
            RcsGrid(azimuths[:160], [0.0], frequencies, ["VV", "HH"], rcs=field[:160].copy(),
                    rcs_storage_domain="raw_field", units=units),
            RcsGrid(azimuths[128:], [0.0], frequencies, ["VV", "HH"], rcs=field[128:].copy(),
                    rcs_storage_domain="raw_field", units=units),
        )

    left, right = pair()
    peak = _peak_bytes(lambda: RcsGrid.join_many(left, right))
    former_left, former_right = pair()
    former_left.rcs_power
    former_right.rcs_power
    former_peak = _peak_bytes(lambda: RcsGrid.join_many(former_left, former_right))
    # One complex128 output plus bounded block scratch, against power, phase
    # and two raw float64 arrays plus the constructor's complex conversion.
    assert peak < 0.6 * former_peak
    assert peak < 2.5 * field.nbytes


def test_overlap_reads_each_input_once(monkeypatch):
    shape = (40, 1, 8, 2)
    field = _field(shape, seed=8)
    axes = _axes(shape)
    grids = [
        RcsGrid(*axes, rcs=field),
        RcsGrid(*axes, rcs=field, rcs_storage_domain="raw_field", units=_sigma_units("sigma_2d")),
    ]
    power, phase = np.abs(field) ** 2, np.angle(field)
    phase[3, 0, :, 1] = np.nan
    grids.append(RcsGrid(*axes, rcs_power=power, rcs_phase=phase))
    reads = []
    for name in ("_sample_kwargs", "power_phase_slice", "rcs_slice", "power_slice", "phase_slice"):
        original = getattr(RcsGrid, name)

        def counting(self, *args, _name=name, _original=original, **kwargs):
            reads.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(RcsGrid, name, counting)
    results = RcsGrid.overlap_many(*grids)
    # One owned read per complex input (the physical one goes through
    # rcs_slice inside _sample_kwargs) and one paired read for the polar
    # input, which copies its two stored arrays once each.
    assert reads.count("_sample_kwargs") == 2 and reads.count("power_phase_slice") == 1
    assert reads.count("rcs_slice") == 1
    assert reads.count("power_slice") == 1 and reads.count("phase_slice") == 1
    assert results[1].storage_domain == "raw_field"
    missing = np.isnan(np.asarray(results[2].power))
    for result in results:
        np.testing.assert_array_equal(np.isnan(np.asarray(result.power)), missing)
    # The common polarization axis is sorted; compare in that order.
    order = [list(axes[3]).index(label) for label in results[0].polarizations]
    np.testing.assert_allclose(results[0].rcs, field[..., order], rtol=0, atol=0, equal_nan=True)


def test_elevation_pair_relabel_merges_the_stored_field():
    shape = (180, 2, 4, 2)
    field = _field(shape, seed=9)
    azimuths = np.arange(shape[0], dtype=float)
    units = _sigma_units("sigma_2d")

    def grid():
        return RcsGrid(azimuths, [-5.0, 5.0], np.linspace(8.0, 12.0, 4), ["VV", "HH"],
                       rcs=field.copy(), rcs_storage_domain="raw_field", units=units)

    merged = grid().combine_elevation_pair_to_azimuth_360(-5.0, 5.0, assumptions_attested=True)
    former_source = grid()
    former_source.rcs_power
    former = former_source.combine_elevation_pair_to_azimuth_360(-5.0, 5.0, assumptions_attested=True)
    assert merged.storage_kind == "complex" and merged.storage_domain == "raw_field"
    np.testing.assert_array_equal(merged.azimuths, former.azimuths)
    np.testing.assert_allclose(merged.rcs, former.rcs, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(merged.extra["rcs_amp_real"], former.extra["rcs_amp_real"])
    np.testing.assert_array_equal(merged.extra["rcs_amp_imag"], former.extra["rcs_amp_imag"])
    assert "elevation_pair_to_azimuth_json" in merged.extra


def _former_grid_content_sha256(grid):
    """The whole-array hash the range-calibration provenance used to compute."""
    import hashlib
    import json

    digest = hashlib.sha256()
    digest.update(b"grim.range-calibration-grid-id.v2\0")

    def _update_array(label, values):
        contiguous = np.ascontiguousarray(values)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(str(contiguous.shape).encode("ascii") + b"\0")
        digest.update(contiguous.tobytes(order="C"))

    for values in (
        np.asarray(grid.azimuths, dtype=np.float64),
        np.asarray(grid.elevations, dtype=np.float64),
        np.asarray(grid.frequencies, dtype=np.float64),
        np.asarray(grid.power, dtype=np.float64),
        np.asarray(grid.phase, dtype=np.float64),
    ):
        _update_array("modeled-array", values)
    cells_per_azimuth = int(np.prod(grid.power.shape[1:]))
    azimuth_block = max(1, 262_144 // max(1, cells_per_azimuth))
    digest.update(b"authoritative-complex-field\0")
    digest.update(str(grid.power.shape).encode("ascii") + b"\0")
    for start in range(0, len(grid.azimuths), azimuth_block):
        field = np.asarray(grid.rcs_slice((slice(start, start + azimuth_block), slice(None),
                                           slice(None), slice(None))), dtype=np.complex128)
        _update_array("field-real", field.real)
        _update_array("field-imag", field.imag)
    digest.update(json.dumps([str(value) for value in grid.polarizations.tolist()],
                             separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps(dict(grid.units or {}), sort_keys=True, separators=(",", ":"),
                             default=str).encode("utf-8"))
    convention_metadata = {
        key: grid._declared_scalar_metadata(key)
        for key in ("phase_reference", "time_convention", "polarization_basis", "amplitude_convention")
    }
    digest.update(json.dumps(convention_metadata, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def test_range_calibration_hash_streams_the_same_bytes_without_whole_arrays():
    import json

    shape = (70_000, 1, 2, 2)  # 4 cells per azimuth: the hash walks several slabs
    units = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz",
             "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
             "angular_coordinate_system": "conic"}

    def grid(seed):
        return RcsGrid(np.linspace(-180.0, 180.0, shape[0], endpoint=False), [0.0], [9.0, 10.0],
                       ["VV", "HH"], rcs=_field(shape, seed=seed), units=units)

    exact = grid(11)
    expected = _former_grid_content_sha256(exact)
    result = grid(12).range_calibrate(grid(13), exact, 0.0, convention_attested=True)
    provenance = json.loads(result.extra["range_calibration_json"])
    assert provenance["exact_reference_content_sha256"] == expected
    assert provenance["measured_calibration_content_sha256"] == _former_grid_content_sha256(grid(13))


def test_native_preflight_charges_no_constructor_copy_for_adoptable_archives(tmp_path):
    from GRIM_Backend.datasets import memory as memory_module

    shape = (64, 1, 64, 2)
    field = _field(shape, seed=14)
    cells = field.size
    scratch = min(cells, memory_module._RAW_COMPLEX_VALIDATION_BLOCK_CELLS) * 128
    adoptable = RcsGrid(*_axes(shape), rcs=field).save(str(tmp_path / "adoptable.grim"))
    report = memory_module._preflight_native_archive_allocation(adoptable)
    assert report["estimated_peak_bytes"] <= report["retained_bytes"] + scratch
    assert RcsGrid.load(adoptable, max_output_bytes=report["estimated_peak_bytes"]).rcs.shape == shape
    # A physical field that keeps a raw pair in extra is still copied and masked.
    retained = RcsGrid(*_axes(shape), rcs=field, units=_sigma_units("sigma_3d"),
                       extra={"rcs_amp_real": (field / np.sqrt(4 * np.pi)).real,
                              "rcs_amp_imag": (field / np.sqrt(4 * np.pi)).imag})
    copied = retained.save(str(tmp_path / "retained.grim"))
    copied_report = memory_module._preflight_native_archive_allocation(copied)
    assert copied_report["estimated_peak_bytes"] - copied_report["retained_bytes"] >= field.nbytes + 3 * cells
