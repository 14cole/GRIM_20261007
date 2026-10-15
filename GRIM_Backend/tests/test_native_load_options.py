"""Bounded native loading preserves physical samples and cache lifetimes."""
import gc
import json
import zipfile
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.memory import estimate_native_archive_allocation
from GRIM_Backend.io.native_cache import _retry_cleanup


def write(path, payload, compressed=True):
    with path.open("wb") as stream:
        (np.savez_compressed if compressed else np.savez)(stream, **payload)
    return path


def payload(shape=(8, 2, 5, 2), kind="complex", domain="physical", version=2):
    field = (np.arange(np.prod(shape)).reshape(shape) / 10 + 1 + 2j).astype(np.complex64)
    values = dict(azimuths=np.arange(shape[0]) * .1, elevations=np.arange(shape[1]) * .2,
                  frequencies=8e9 + np.arange(shape[2]) * 1e9,
                  polarizations=np.asarray(["HH", "VV"][:shape[3]]),
                  units=json.dumps({"azimuth": "rad", "elevation": "rad", "frequency": "Hz",
                                    "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"}),
                  history="test source", ancillary=np.arange(np.prod(shape)*3).reshape(shape+(3,)))
    if version == 2:
        values.update(grim_format_version=2, storage_kind=kind, storage_domain=domain)
    if kind == "complex":
        values["rcs_complex"] = field
    else:
        values["rcs_power"] = np.square(field.real) + np.square(field.imag)
        if kind == "power_phase" or version == 1:
            phase = np.angle(field)
            phase[1, 0, 0, 0] = np.nan
            phase[2, 0, 0, 0] = 8.0  # preserve unwrapped phase
            values["rcs_phase"] = phase
    return values


@pytest.mark.parametrize("kind,version", [("complex", 2), ("power", 2), ("power_phase", 2), ("power_phase", 1)])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_selected_native_samples_match_full_load(tmp_path, kind, version, storage):
    source = write(tmp_path / "source.grim", payload(kind=kind, version=version))
    original = source.read_bytes()
    full = RcsGrid.load(source)
    selected = RcsGrid.load(source, storage=storage, cache_dir=tmp_path,
                            selection={"azimuth": (.2, .6), "frequency": (9e9, 11e9), "polarization": ["vv"]})
    chosen = np.ix_(np.flatnonzero((full.azimuths >= .2) & (full.azimuths <= .6)), [0, 1], [1, 2, 3], [1])
    np.testing.assert_array_equal(selected.power, full.power[chosen])
    np.testing.assert_array_equal(selected.phase, full.phase[chosen])
    np.testing.assert_array_equal(selected.extra["ancillary"], full.extra["ancillary"][chosen])
    assert selected.units == full.units
    assert selected.load_storage == storage
    assert "Native load selection" in selected.history
    assert source.read_bytes() == original
    saved = selected.save(tmp_path / "saved")
    restored = RcsGrid.load(saved)
    # Eager loading may compact a complete float32 power/phase selection to
    # complex64; its sqrt/exp/square round trip has float32 rounding error.
    np.testing.assert_allclose(restored.power, selected.power, rtol=4*np.finfo(np.float32).eps)
    np.testing.assert_allclose(restored.phase, selected.phase, rtol=4*np.finfo(np.float32).eps)
    del selected
    gc.collect()


@pytest.mark.parametrize("domain", ["physical", "raw_field"])
@pytest.mark.parametrize("order,dtype", [("C", "complex64"), ("F", "complex128"), ("F", ">c16")])
def test_disk_complex_layout_and_raw_normalization(tmp_path, domain, order, dtype):
    values = payload(domain=domain)
    values["rcs_complex"] = np.array(values["rcs_complex"], order=order, dtype=dtype)
    if domain == "raw_field":
        units = json.loads(values["units"])
        units.update(rcs_linear_quantity="sigma_2d", rcs_log_unit="dBke")
        values["units"] = json.dumps(units)
    source = write(tmp_path / "source.grim", values)
    full = RcsGrid.load(source)
    selected = RcsGrid.load(source, storage="disk", cache_dir=tmp_path,
                            selection={"frequency": (9e9, 11e9)})
    np.testing.assert_array_equal(selected.rcs_slice(...), full.rcs_slice((slice(None), slice(None), slice(1, 4), slice(None))))
    np.testing.assert_array_equal(selected.power, full.power[:, :, 1:4, :])
    assert selected.storage_domain == domain
    assert not selected._complex.flags.owndata
    del selected
    gc.collect()


def test_disk_and_selected_load_fit_below_full_eager_budget(tmp_path):
    values = payload(shape=(150_000, 1, 2, 1))
    values.pop("ancillary")
    source = write(tmp_path / "large.grim", values)
    eager = estimate_native_archive_allocation(source)
    disk = estimate_native_archive_allocation(source, storage="disk")
    # For larger grids the fixed validation block stays bounded; a narrow
    # selection does not pay for all of the original sample arrays in RAM.
    selection = {"azimuth": (0, .9)}
    selected = estimate_native_archive_allocation(source, selection=selection)
    assert selected["estimated_peak_bytes"] < eager["estimated_peak_bytes"]
    with pytest.raises(MemoryError):
        RcsGrid.load(source, max_output_bytes=selected["estimated_peak_bytes"])
    bounded = RcsGrid.load(source, selection=selection, max_output_bytes=selected["estimated_peak_bytes"], cache_dir=tmp_path)
    np.testing.assert_array_equal(bounded.rcs_slice(...), values["rcs_complex"][:10])
    assert disk["disk_bytes"] >= source.stat().st_size
    assert eager["axis_units"]["frequency"] == "Hz"


@pytest.mark.parametrize("kind", ["complex", "power", "power_phase"])
def test_cache_lives_with_plain_numpy_views_then_is_removed(tmp_path, kind):
    source = write(tmp_path / "source.grim", payload(kind=kind))
    loaded = RcsGrid.load(source, storage="disk", cache_dir=tmp_path)
    samples = loaded._complex if loaded._complex is not None else loaded._power
    view = np.asarray(samples)[1:, ..., 0]
    expected = view.copy()
    paths = list(tmp_path.glob("grim-native-*"))
    assert len(paths) == 1
    del loaded, samples
    gc.collect()
    assert paths[0].exists()
    np.testing.assert_array_equal(view, expected)
    del view
    gc.collect()
    _retry_cleanup()
    assert not paths[0].exists()


def test_disk_load_can_exceed_the_eager_memory_limit(tmp_path):
    values = dict(azimuths=np.arange(512), elevations=[0], frequencies=np.arange(4096)+1., polarizations=["VV"],
                  grim_format_version=2, storage_kind="complex", storage_domain="physical",
                  rcs_complex=np.full((512, 1, 4096, 1), 1+2j, dtype=np.complex64))
    source = write(tmp_path / "large.grim", values)
    disk = estimate_native_archive_allocation(source, storage="disk")
    eager = estimate_native_archive_allocation(source)
    assert disk["estimated_peak_bytes"] < eager["estimated_peak_bytes"]
    assert disk["retained_bytes"] < disk["logical_dataset_bytes"]
    assert disk["estimated_peak_bytes"] >= disk["retained_bytes"]
    assert disk["mapped_bytes"] > 0
    with pytest.raises(MemoryError):
        RcsGrid.load(source, max_output_bytes=disk["estimated_peak_bytes"])
    loaded = RcsGrid.load(source, storage="disk", max_output_bytes=disk["estimated_peak_bytes"], cache_dir=tmp_path)
    assert loaded.power[511, 0, 4095, 0] == 5
    del loaded
    gc.collect()


def test_selected_legacy_raw_pair_keeps_exact_complex_authority(tmp_path):
    values = payload(kind="power_phase", version=1)
    field = (1 + np.arange(np.prod(values["rcs_power"].shape)).reshape(values["rcs_power"].shape) / 10 + 1e-12j)
    values["rcs_power"] = (np.abs(field)**2 * (4*np.pi)).astype(np.float32)
    values["rcs_phase"] = np.angle(field).astype(np.float32)
    values.update(rcs_amp_real=field.real, rcs_amp_imag=field.imag, raw_complex_amplitude_preserved=True)
    source = write(tmp_path / "raw.grim", values)
    selected = RcsGrid.load(source, storage="disk", cache_dir=tmp_path, selection={"frequency": (9e9, 11e9)})
    np.testing.assert_array_equal(selected.rcs_slice(...), field[:, :, 1:4, :] * np.sqrt(4*np.pi))
    saved = selected.save(tmp_path / "raw-subset")
    restored = RcsGrid.load(saved)
    np.testing.assert_allclose(restored.rcs_slice(...), selected.rcs_slice(...), rtol=2e-15)
    del selected
    gc.collect()


@pytest.mark.parametrize("storage", ["memory", "disk"])
@pytest.mark.parametrize("change,match", [
    ({"rcs_complex": np.ones((8, 2, 5, 2))}, "invalid rcs_complex"),
    ({"rcs_complex": np.full((8, 2, 5, 2), complex(np.nan, 0))}, "one-sided NaN"),
    ({"rcs_complex": np.full((8, 2, 5, 2), complex(1e200, 0))}, "infinite rcs_power"),
])
def test_bounded_native_loader_retains_validation(tmp_path, storage, change, match):
    values = payload()
    values.update(change)
    source = write(tmp_path / "bad.grim", values)
    with pytest.raises(ValueError, match=match):
        RcsGrid.load(source, storage=storage, selection={"frequency": (8e9, 9e9)}, cache_dir=tmp_path)
    gc.collect()
    _retry_cleanup()
    assert not list(tmp_path.glob("grim-native-*"))


@pytest.mark.parametrize("selection", [{"frequency": (20e9, 21e9)}, {"azimuth": (2, 1)},
                                        {"elevation": (np.nan, 1)}, {"polarization": ["HV"]}, {"bad": (0, 1)}])
def test_invalid_selections_are_explicit(tmp_path, selection):
    source = write(tmp_path / "source.grim", payload())
    with pytest.raises(ValueError):
        RcsGrid.load(source, storage="disk", selection=selection, cache_dir=tmp_path)


def test_header_framing_is_checked_before_private_extraction(tmp_path):
    source = tmp_path / "bad.grim"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("rcs_complex.npy", b"not an npy header")
    with mock.patch("GRIM_Backend.io.native_cache.load_native_arrays") as extraction:
        with pytest.raises(ValueError):
            RcsGrid.load(source, storage="disk")
        extraction.assert_not_called()


def test_private_legacy_phase_wrapping_and_unknown_power_match_eager(tmp_path):
    values = payload(kind="power_phase", version=1)
    values["rcs_power"][0, 0, 0, 0] = np.nan
    units = json.loads(values["units"])
    units["phase_wrap"] = "0_360"
    values["units"] = json.dumps(units)
    source = write(tmp_path / "source.grim", values)
    full = RcsGrid.load(source)
    mapped = RcsGrid.load(source, storage="disk", cache_dir=tmp_path)
    np.testing.assert_array_equal(mapped.power, full.power)
    np.testing.assert_array_equal(mapped.phase, full.phase)
    del mapped
    gc.collect()


@pytest.mark.parametrize("storage", ["memory", "disk"])
@pytest.mark.parametrize("case", ["integer_power", "large_phase"])
def test_mixed_component_precision_matches_eager_before_conversion(tmp_path, storage, case):
    shape = (2, 1, 1, 1)
    if case == "integer_power":
        power = np.asarray([2**40+1, 2**40+129], dtype=np.int64).reshape(shape)
        phase = np.zeros(shape, dtype=np.float64)
    else:
        power = np.ones(shape, dtype=np.float64)
        phase = np.full(shape, 1e8, dtype=np.float32)
    source = write(tmp_path / "mixed.grim", dict(
        azimuths=[0., 1.], elevations=[0.], frequencies=[1.], polarizations=["VV"],
        rcs_power=power, rcs_phase=phase, units=json.dumps({"phase_wrap": "0_360"})))
    eager = RcsGrid.load(source)
    loaded = RcsGrid.load(source, storage=storage, selection={"azimuth": (0., 1.)}, cache_dir=tmp_path)
    assert loaded._power.dtype == np.float64
    assert loaded._phase.dtype == np.float64
    # Ordinary complete-phase loading reconstructs complex128, which may
    # introduce one final rounding unit in power/phase. Promotion itself is
    # exact, including the integer values that float32 cannot represent.
    np.testing.assert_array_equal(loaded.power, power.astype(np.float64))
    np.testing.assert_allclose(loaded.power, eager.power, rtol=2e-15)
    np.testing.assert_allclose(loaded.phase, eager.phase, rtol=2e-15)
    report = estimate_native_archive_allocation(source, storage=storage, selection={"azimuth": (0., 1.)})
    assert report["logical_dataset_bytes"] >= power.size * 16
    del loaded
    gc.collect()


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_selected_native_load_preserves_empty_ancillary_arrays(tmp_path, storage):
    values = payload()
    values["empty_auxiliary"] = np.empty((0, 3), dtype=np.float64)
    source = write(tmp_path / "empty-extra.grim", values)
    loaded = RcsGrid.load(source, storage=storage, selection={"azimuth": (0., 0.)}, cache_dir=tmp_path)
    assert loaded.extra["empty_auxiliary"].shape == (0, 3)
    assert loaded.extra["empty_auxiliary"].dtype == np.float64
    saved = loaded.save(tmp_path / "saved-empty")
    assert RcsGrid.load(saved).extra["empty_auxiliary"].shape == (0, 3)
    del loaded
    gc.collect()


@pytest.mark.parametrize("selection", [None, {"azimuth": (0., .3)}])
@pytest.mark.parametrize("consistent", [False, True])
def test_physical_complex_field_validated_before_retained_raw_pair_authority(tmp_path, selection, consistent):
    values = payload()
    physical = values["rcs_complex"]
    raw = physical.astype(np.complex128) / np.sqrt(4*np.pi)
    if not consistent:
        raw *= 10
    values.update(rcs_amp_real=raw.real, rcs_amp_imag=raw.imag, raw_complex_amplitude_preserved=True)
    source = write(tmp_path / "physical-with-raw.grim", values)
    if not consistent:
        for storage in ("memory", "disk"):
            with pytest.raises(ValueError, match="inconsistent"):
                RcsGrid.load(source, storage=storage, selection=selection, cache_dir=tmp_path)
        gc.collect()

        _retry_cleanup()
        assert not list(tmp_path.glob("grim-native-*"))
    else:
        eager = RcsGrid.load(source)
        loaded = RcsGrid.load(source, storage="disk", selection=selection, cache_dir=tmp_path)
        count = loaded.shape[0]
        np.testing.assert_allclose(loaded.rcs_slice(...), eager.rcs_slice(...)[:count], rtol=2e-15)
        np.testing.assert_allclose(loaded.power, eager.power[:count], rtol=2e-15)
        del loaded
        gc.collect()


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_opted_in_legacy_object_axes_and_ancillary_follow_selection(tmp_path, storage):
    values = payload(kind="power_phase", version=1)
    values["polarizations"] = np.asarray(["HH", "VV"], dtype=object)
    values["object_auxiliary"] = values["ancillary"].astype(object)
    values["empty_object_auxiliary"] = np.empty((0,), dtype=object)
    source = write(tmp_path / "legacy-object.grim", values)
    with pytest.raises(ValueError, match="object-typed"):
        RcsGrid.load(source, storage=storage, selection={"polarization": ["VV"]}, cache_dir=tmp_path)
    loaded = RcsGrid.load(source, storage=storage, selection={"polarization": ["VV"]},
                         allow_legacy_pickle=True, cache_dir=tmp_path)
    np.testing.assert_array_equal(loaded.polarizations, ["VV"])
    np.testing.assert_array_equal(loaded.power, values["rcs_power"][..., 1:2])
    np.testing.assert_array_equal(loaded.extra["object_auxiliary"], values["object_auxiliary"][..., 1:2, :])
    assert loaded.extra["empty_object_auxiliary"].shape == (0,)
    del loaded
    gc.collect()


@pytest.mark.parametrize("storage", ["memory", "disk"])
@pytest.mark.parametrize("component", ["power", "phase"])
def test_mutable_legacy_components_invalidate_raw_metadata_after_bounded_load(tmp_path, storage, component):
    import copy
    values = payload(kind="power_phase", version=1)
    field = np.full(values["rcs_power"].shape, 1 + 1e-12j, dtype=np.complex128)
    values.update(rcs_power=(np.abs(field)**2 * 4*np.pi).astype(np.float32),
                  rcs_phase=np.angle(field).astype(np.float32),
                  rcs_amp_real=field.real, rcs_amp_imag=field.imag, raw_complex_amplitude_preserved=True)
    source = write(tmp_path / "mutable-raw.grim", values)
    loaded = RcsGrid.load(source, storage=storage, selection={"polarization": ["VV"]}, cache_dir=tmp_path)
    sibling = copy.copy(loaded)
    original = loaded.rcs_slice(...)
    writable = loaded.rcs_power if component == "power" else loaded.rcs_phase
    # Merely reading writable components must keep the high precision raw I/Q.
    np.testing.assert_array_equal(loaded.rcs_slice(...), original)
    if component == "power":
        np.asarray(writable)[...] *= 2
    else:
        np.asarray(writable)[...] += .25
    expected = np.sqrt(loaded.rcs_power) * np.exp(1j*loaded.rcs_phase)
    np.testing.assert_allclose(loaded.rcs_slice(...), expected, rtol=2e-7)
    np.testing.assert_allclose(sibling.rcs_slice(...), expected, rtol=2e-7)
    assert "rcs_amp_real" not in loaded.extra
    restored = RcsGrid.load(loaded.save(tmp_path / "edited"))
    np.testing.assert_allclose(restored.power, loaded.power, rtol=4*np.finfo(np.float32).eps)
    np.testing.assert_allclose(restored.phase, loaded.phase, rtol=4*np.finfo(np.float32).eps)
    del writable, sibling, loaded
    gc.collect()
