"""Storage contracts: exact fields, bounded derived reads and legacy edits."""
import copy
import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid


def grid(field=None, **kwargs):
    if field is not None:
        field = np.asarray(field).reshape(-1, 1, 1, 1)
        count = field.shape[0]
    else:
        count = np.asarray(kwargs["rcs_power"]).size
        kwargs["rcs_power"] = np.asarray(kwargs["rcs_power"]).reshape(count, 1, 1, 1)
        if "rcs_phase" in kwargs:
            kwargs["rcs_phase"] = np.asarray(kwargs["rcs_phase"]).reshape(count, 1, 1, 1)
    return RcsGrid(np.arange(count), [0.], [10.], ["HH"], rcs=field, **kwargs)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_is_single_exact_source_and_input_is_not_aliased(dtype):
    field = np.array([1. + 1.e-18j, -1. + 1.e-18j, 0.j], dtype=dtype)
    original = field.copy()
    ds = grid(field)
    assert ds.storage_kind == "complex"
    assert ds._power is None and ds._phase is None
    assert ds.real_dtype == field.real.dtype
    np.testing.assert_array_equal(ds.rcs.ravel(), original)
    field[:] = 0
    np.testing.assert_array_equal(ds.rcs.ravel(), original)
    assert not ds.native_sample_payload()["rcs_complex"].flags.writeable


def test_complex_constructor_does_not_convert_to_power_or_phase(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("complex input must remain complex")
    monkeypatch.setattr(RcsGrid, "_complex_from_power_phase", fail)
    monkeypatch.setattr(RcsGrid, "_clean_power", fail)
    monkeypatch.setattr(RcsGrid, "_clean_phase", fail)
    ds = grid([1.e200 + 2.e199j, 1. + 1.e-18j])
    assert ds.rcs[0, 0, 0, 0] == 1.e200 + 2.e199j


def test_power_and_phase_match_physical_complex_response():
    ds = grid([3. + 4.j, -2. + 0.j, 0.j])
    np.testing.assert_allclose(np.asarray(ds.power).ravel(), [25., 4., 0.])
    np.testing.assert_allclose(np.asarray(ds.phase).ravel(), [np.arctan2(4., 3.), np.pi, 0.])
    assert ds.storage_kind == "complex"
    assert ds._power is None and ds._phase is None


def test_lazy_components_bound_slices_and_flat_reads(monkeypatch):
    ds = grid(np.arange(1, 101) * (1 + 2j))
    calls = []
    original = ds._complex_samples
    def observe(selection, *, owned):
        result = original(selection, owned=owned)
        calls.append(result.size)
        return result
    monkeypatch.setattr(ds, "_complex_samples", observe)
    assert ds.power.shape == (100, 1, 1, 1)
    assert ds.power.size == 100
    assert ds.power.dtype == np.dtype(np.float64)
    assert calls == []
    np.testing.assert_allclose(ds.power[3:5].ravel(), [80., 125.])
    np.testing.assert_allclose(ds.power.flat[3:5], [80., 125.])
    assert calls == [2, 2]
    assert ds.phase.flat[-1] == pytest.approx(np.arctan2(2., 1.))


def test_derived_arrays_readonly_but_explicit_copies_are_writable():
    ds = grid([1 + 2j])
    with pytest.raises(ValueError, match="read-only"):
        ds.power[...] = 4
    with pytest.raises(ValueError):
        np.asarray(ds.phase)[...] = 0
    np.array(ds.power, copy=True)[...] = 99
    ds.power.copy()[...] = 99
    assert ds.rcs.item() == 1 + 2j


def test_legacy_read_keeps_exact_field_and_saved_numpy_view_mutation_is_observed():
    original = np.array([-1 + 1.e-18j, 3 + 4j])
    ds = grid(original)
    legacy = ds.rcs_power
    np.testing.assert_array_equal(ds.rcs.ravel(), original)
    np.asarray(legacy).reshape(-1)[1] = 100
    assert abs(ds.rcs[1, 0, 0, 0]) == pytest.approx(10.)
    assert ds._complex is None
    assert ds.native_sample_payload()["storage_kind"] == "power_phase"


def test_legacy_phase_ufunc_out_invalidates_complex_authority():
    ds = grid([1 + 0j, 1 + 0j])
    phase = ds.rcs_phase
    np.add(np.asarray(phase), np.pi / 2, out=np.asarray(phase))
    np.testing.assert_allclose(ds.rcs, 1j, atol=1.e-15)


def test_shallow_copies_share_later_legacy_power_and_phase_edits():
    original = grid([1 + 2j, 3 + 4j])
    row_copy = copy.copy(original)
    assert original._power is None and row_copy._power is None
    original.rcs_power[:] *= 10
    assert row_copy.power is original.power
    np.testing.assert_allclose(row_copy.power.ravel(), [50, 250])
    row_copy.rcs_phase[:] = np.pi / 2
    assert original.phase is row_copy.phase
    np.testing.assert_allclose(original.rcs.real, 0, atol=1e-13)
    np.testing.assert_array_equal(original.rcs, row_copy.rcs)


def test_shallow_copy_explicit_replacement_keeps_other_dataset_unchanged():
    original = grid([-2 + 1.e-18j, 1 + 2j])
    row_copy = copy.copy(original)
    row_copy.set_complex((1, 0, 0, 0), 5 + 6j)
    original.rcs_power[:] *= 4
    assert row_copy.rcs[1, 0, 0, 0] == 5 + 6j
    assert row_copy.rcs[0, 0, 0, 0] == -2 + 1.e-18j


def test_legacy_replaced_arrays_keep_guard_subclass_and_auditable_shape():
    class Guard(np.ndarray):
        pass
    ds = grid([1 + 2j])
    replacement = ds.rcs_power.view(Guard)
    ds.rcs_power = replacement
    assert ds.power is replacement
    ds.rcs_phase = np.zeros((2,))
    assert ds.phase.shape == (2,)
    assert ds.audit()["status"] == "error"


def test_power_only_has_no_phase_array_and_zero_is_valid_complex():
    ds = grid(rcs_power=[4., 0., np.nan])
    assert ds.storage_kind == "power"
    assert ds._phase is None and ds._complex is None
    assert np.isnan(np.asarray(ds.phase)).all()
    assert np.isnan(ds.rcs[0, 0, 0, 0])
    assert ds.rcs[1, 0, 0, 0] == 0j
    assert "rcs_phase" not in ds.native_sample_payload()


def test_mixed_unknown_and_zero_unknown_phase_are_retained():
    ds = grid(rcs_power=[4., 9., 0.], rcs_phase=[0., np.nan, np.nan])
    assert ds.storage_kind == "power_phase"
    assert ds.rcs[0, 0, 0, 0] == 2
    assert np.isnan(ds.rcs[1, 0, 0, 0])
    assert ds.rcs[2, 0, 0, 0] == 0
    assert np.isnan(ds.phase[2, 0, 0, 0])


def test_unwrapped_turn_count_is_retained():
    ds = grid(rcs_power=[1., 1.], rcs_phase=[0., 5 * np.pi])
    assert ds.storage_kind == "power_phase"
    np.testing.assert_array_equal(np.asarray(ds.phase).ravel(), [0., 5 * np.pi])


def test_declared_wrap_controls_derived_phase():
    ds = grid([1 - 1j], units={"phase_wrap": "0_360"})
    assert ds.phase.item() == pytest.approx(7 * np.pi / 4)
    np.testing.assert_array_equal(ds.rcs.ravel(), [1 - 1j])


def test_direct_complex_edit_preserves_tiny_quadrature_exactly():
    ds = grid([1 + 2j, 2 + 3j])
    value = -1 + 1.e-18j
    ds.set_complex((0, 0, 0, 0), value)
    assert ds.rcs[0, 0, 0, 0] == value
    assert ds.rcs[1, 0, 0, 0] == 2 + 3j
    assert ds._power is None and ds._phase is None


def test_complex_edit_promotes_to_precision_of_replacement():
    ds = grid(np.array([1 + 2j], dtype=np.complex64))
    value = np.complex128(1.0000000001 + 2j)
    ds.set_complex((0, 0, 0, 0), value)
    assert ds.rcs.dtype == np.dtype(np.complex128)
    assert ds.rcs.item() == value


@pytest.mark.parametrize("change", [{"phase": .2}, {"power": 9.}])
def test_polar_edit_leaves_other_complex_samples_exact(change):
    original = np.array([-2 + 1.e-18j, 1 + 0j])
    ds = grid(original)
    ds.set_power_phase((1, 0, 0, 0), **change)
    assert ds.rcs[0, 0, 0, 0] == original[0]
    assert ds.storage_kind == "complex"


def test_power_edit_preserves_selected_tiny_quadrature_direction():
    ds = grid([-2 + 1.e-18j])
    ds.set_power_phase((0, 0, 0, 0), power=16.)
    assert ds.rcs.item() == -4 + 2.e-18j


def test_promoted_power_edit_computes_amplitude_in_promoted_precision():
    original = np.array([1 + 2j, -2 + 1.e-18j], dtype=np.complex64)
    ds = grid(original)
    requested = np.float64(4.0000000001)
    ds.set_power_phase((0, 0, 0, 0), power=requested)
    assert ds.rcs.dtype == np.dtype(np.complex128)
    assert ds.power[0, 0, 0, 0] == pytest.approx(requested, rel=8 * np.finfo(np.float64).eps, abs=0)
    assert ds.rcs[1, 0, 0, 0] == np.complex128(original[1])


@pytest.mark.parametrize("phase", [np.nan, 1.7])
def test_zero_phase_information_is_retained_when_supplied(phase):
    ds = grid(rcs_power=[0., 1.], rcs_phase=[phase, 0.])
    assert ds.storage_kind == "power_phase"
    np.testing.assert_equal(ds.phase[0, 0, 0, 0], phase)
    assert ds.rcs[0, 0, 0, 0] == 0


@pytest.mark.parametrize("phase", [np.nan, 1.7])
def test_zero_raw_phase_information_is_retained_when_supplied(phase):
    ds = grid(rcs_power=[0., 4 * np.pi], rcs_phase=[phase, 0.],
              extra={"rcs_amp_real": np.array([0., 1.]).reshape(2, 1, 1, 1),
                     "rcs_amp_imag": np.zeros((2, 1, 1, 1))})
    assert ds.storage_kind == "power_phase"
    np.testing.assert_equal(ds.phase[0, 0, 0, 0], phase)
    assert ds.rcs[0, 0, 0, 0] == 0
    assert ds.native_sample_payload()["storage_kind"] == "power_phase"


def test_explicit_phase_removal_preserves_magnitude_without_inventing_phase():
    ds = grid([3 + 4j, 1 + 2j])
    ds.set_power_phase((0, 0, 0, 0), phase=np.nan)
    assert ds.power[0, 0, 0, 0] == 25
    assert np.isnan(ds.rcs[0, 0, 0, 0])
    assert ds.power[1, 0, 0, 0] == pytest.approx(5)


def test_raw_storage_owns_one_field_and_exposes_views():
    raw = np.array([1 + 1.e-18j, 2 + 3j])
    ds = grid(raw, rcs_storage_domain="raw_field", units={"rcs_linear_quantity": "sigma_3d"})
    assert ds.storage_domain == "raw_field"
    assert ds._power is None and ds._phase is None
    assert np.shares_memory(ds.extra["rcs_amp_real"], ds._complex)
    assert np.shares_memory(ds.extra["rcs_amp_imag"], ds._complex)
    np.testing.assert_allclose(ds.rcs.ravel(), raw * np.sqrt(4 * np.pi))
    np.testing.assert_array_equal(ds.native_sample_payload()["rcs_complex"].ravel(), raw)


def test_raw_legacy_read_preserves_precision_then_edit_removes_raw_authority():
    raw = np.array([-1 + 1.e-18j])
    ds = grid(raw, rcs_storage_domain="raw_field", units={"rcs_linear_quantity": "sigma_3d"})
    expected = ds.rcs.copy()
    power = ds.rcs_power
    np.testing.assert_array_equal(ds.rcs, expected)
    power[...] *= 4
    assert abs(ds.rcs.item()) == pytest.approx(2 * abs(expected.item()))
    assert ds.storage_domain == "physical"
    assert "rcs_amp_real" not in ds.extra


@pytest.mark.parametrize("legacy", [False, True])
def test_late_raw_override_and_inplace_edits_remain_authoritative(legacy):
    ds = grid(np.array([1 + 2j, 3 + 4j], dtype=np.complex64))
    if legacy:
        assert ds.rcs_power.shape == ds.shape
    raw = np.array([2 + 3j, 4 + 5j]).reshape(ds.shape)
    ds.extra.update(rcs_amp_real=raw.real.copy(), rcs_amp_imag=raw.imag.copy())
    np.testing.assert_allclose(ds.rcs, raw * np.sqrt(4 * np.pi))
    reader, dtype, authoritative = ds._bounded_complex_slice_reader(include_authority=True)
    assert authoritative and dtype == np.dtype(np.float64)
    np.testing.assert_allclose(reader((0, 0, 0, 0)), raw[0, 0, 0, 0] * np.sqrt(4 * np.pi))
    before = ds.native_sample_payload()["rcs_complex"].copy()
    ds.extra["rcs_amp_real"][0, 0, 0, 0] = 7
    assert ds.rcs[0, 0, 0, 0] == (7 + 3j) * np.sqrt(4 * np.pi)
    after = ds.native_sample_payload()["rcs_complex"]
    assert before[0, 0, 0, 0] != after[0, 0, 0, 0]


def test_inconsistent_raw_inputs_remain_visible_to_audit():
    ds = grid(rcs_power=[1.], rcs_phase=[0.], extra={
        "rcs_amp_real": np.full((1, 1, 1, 1), 100.),
        "rcs_amp_imag": np.zeros((1, 1, 1, 1)),
        "raw_complex_amplitude_preserved": True,
    })
    codes = {entry["code"] for entry in ds.audit()["errors"]}
    assert "raw_complex_power_mismatch" in codes


def test_sample_transfer_preserves_complex_components_and_does_not_alias():
    ds = grid([-1 + 1.e-18j, 1 + 2j])
    samples = ds._sample_kwargs(selection=(slice(None, None, -1), slice(None), slice(None), slice(None)))
    assert "rcs" in samples and "rcs_phase" not in samples
    np.testing.assert_array_equal(samples["rcs"].ravel(), [1 + 2j, -1 + 1.e-18j])
    samples["rcs"][...] = 0
    assert ds.rcs[0, 0, 0, 0] == -1 + 1.e-18j
