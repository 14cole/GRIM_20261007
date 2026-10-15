"""Numerical and lifetime guarantees for managed operation destinations."""
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.memory import _checked_dense_import_allocation
from GRIM_Backend.datasets.result_storage import (
    allocate_result, array_storage, copy_result, operation_result_storage,
    resident_result_bytes, result_storage_info, result_storage_mode,
    select_result, take_result,
)


def grid(**kwargs):
    return RcsGrid([0., 1.], [0., 2.], [1., 2., 3.], ["HH"], **kwargs)


def test_auto_cumulative_budget_and_nested_thread_isolation(tmp_path):
    assert result_storage_info() is None
    with operation_result_storage("auto", ram_budget_bytes=16, cache_dir=tmp_path):
        a = allocate_result((2,), np.float64)
        assert array_storage(a) == "memory"
        assert resident_result_bytes(1) == 0
        b = allocate_result((2,), np.float64)
        assert array_storage(b) == "disk"
        with operation_result_storage("memory"):
            assert result_storage_mode() == "memory"
        assert result_storage_mode() == "auto"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(result_storage_mode).result() == "memory"
    assert result_storage_info() is None
    del a, b
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_auto_grouped_projection_accounts_for_partial_ram_admission(tmp_path):
    with operation_result_storage("auto", ram_budget_bytes=10, cache_dir=tmp_path):
        assert resident_result_bytes(16) == 0
        assert resident_result_bytes(16, aggregate=True) == 10
        first = allocate_result((8,), np.uint8)
        second = allocate_result((8,), np.uint8)
        assert array_storage(first) == "memory"
        assert array_storage(second) == "disk"
        assert result_storage_info()["reserved_ram_bytes"] == 8
        assert resident_result_bytes(16, aggregate=True) == 2


def test_map_lifetime_follows_plain_array_views(tmp_path):
    with operation_result_storage("disk", cache_dir=tmp_path):
        a = allocate_result((20,), np.float64, fill=3)
        view = np.asarray(a)[3:12:2]
        copied = np.array(a)
    assert array_storage(view) == "disk"
    assert array_storage(copied) == "memory"
    del a
    gc.collect()
    assert list(tmp_path.iterdir())
    view[0] = 11
    np.testing.assert_array_equal(view, [11, 3, 3, 3, 3])
    del view, copied
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_failed_operation_releases_cache(tmp_path):
    def fail():
        with operation_result_storage("disk", cache_dir=tmp_path):
            target = allocate_result((100,), np.complex128)
            target[:] = 1
            raise RuntimeError("cancel operation")
    try:
        fail()
    except RuntimeError:
        pass
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["complex", "power", "phase", "partial", "zero", "unwrapped"])
def test_constructor_disk_matches_memory_exactly(tmp_path, dtype, case):
    p = np.linspace(0.2, 12, 12, dtype=dtype).reshape(2, 2, 3, 1)
    phase = np.linspace(-1, 1, 12, dtype=dtype).reshape(p.shape)
    kwargs = {"rcs_power": p, "rcs_phase": phase}
    if case == "complex":
        kwargs = {"rcs": (np.sqrt(p) * np.exp(1j * phase)).astype(
            np.complex64 if dtype == np.float32 else np.complex128)}
    elif case == "power":
        kwargs.pop("rcs_phase")
    elif case == "partial":
        phase[0, 0, 0, 0] = np.nan
    elif case == "zero":
        p[0, 0, 0, 0] = 0
    elif case == "unwrapped":
        phase[0, 0, 0, 0] = 4 * np.pi
    expected = grid(**kwargs)
    with operation_result_storage("disk", cache_dir=tmp_path):
        actual = grid(**kwargs)
    assert actual.sample_storage == "disk"
    assert expected.storage_kind == actual.storage_kind
    assert expected.real_dtype == actual.real_dtype
    np.testing.assert_array_equal(actual.power, expected.power)
    np.testing.assert_array_equal(actual.phase, expected.phase)
    np.testing.assert_array_equal(actual.rcs, expected.rcs)
    assert not hasattr(expected, "_operation_storage_options")
    assert actual._operation_storage_options["mode"] == "disk"
    del actual
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_owned_mapping_adoption_and_selection(tmp_path):
    values = (np.arange(12).reshape(2, 2, 3, 1) + 1j).astype(complex)
    with operation_result_storage("disk", cache_dir=tmp_path):
        source = copy_result(values)
        actual = grid(rcs=source, _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN)
        assert np.shares_memory(source, actual._complex)
        selected = actual._sample_kwargs(np.ix_([1, 0], [1], [2, 0], [0]))["rcs"]
        np.testing.assert_array_equal(selected, values[np.ix_([1, 0], [1], [2, 0], [0])])
        assert array_storage(selected) == "disk"
        assert not np.shares_memory(selected, source)
        selected[:] = 9
        np.testing.assert_array_equal(actual.rcs, values)


@pytest.mark.parametrize("selection", [Ellipsis, (slice(None, None, -1), 1, slice(0, 3, 2), 0),
    (np.array([1, 0]), slice(None), slice(None), slice(None)),
    (0, 0, 1, 0)])
def test_selection_matches_numpy(tmp_path, selection):
    values = np.arange(12).reshape(2, 2, 3, 1)
    with operation_result_storage("disk", cache_dir=tmp_path):
        actual = select_result(values, selection)
    np.testing.assert_array_equal(actual, values[selection])


def test_take_matches_numpy_and_lazy_copy_is_bounded(tmp_path):
    class Lazy:
        shape = (5, 200, 150, 1)
        dtype = np.dtype(float)
        def __array__(self, *args, **kwargs):
            raise AssertionError("must not materialize full source")
        def __getitem__(self, key):
            shape = np.broadcast_shapes(*(np.shape(x) for x in key)) if isinstance(key[0], np.ndarray) else tuple(
                len(range(*part.indices(size))) for part, size in zip(key, self.shape))
            assert np.prod(shape) <= 65_536
            return np.full(shape, 3.)
    with operation_result_storage("disk", cache_dir=tmp_path):
        copied = copy_result(Lazy())
        taken = take_result(copied, [4, 1, 3], axis=0)
    np.testing.assert_array_equal(taken, np.take(copied, [4, 1, 3], axis=0))


def test_output_projection_excludes_disk_but_imports_keep_their_ram_guard():
    with operation_result_storage("disk"):
        assert resident_result_bytes(800) == 0
        with pytest.raises(MemoryError):
            _checked_dense_import_allocation((100,), [np.float64], source="import", max_output_bytes=4)
        with pytest.raises(ValueError, match="positive"):
            _checked_dense_import_allocation((-1,), [np.float64], source="operation")


def test_disk_space_failure_is_clear_and_leaves_no_cache(tmp_path):
    with operation_result_storage("disk", cache_dir=tmp_path), mock.patch(
            "GRIM_Backend.datasets.result_storage.shutil.disk_usage", return_value=mock.Mock(free=1)):
        with pytest.raises(OSError, match="temporary disk space"):
            allocate_result((200,), np.float64)
    assert not list(tmp_path.iterdir())


def test_zero_size_and_object_metadata_remain_usable(tmp_path):
    with operation_result_storage("disk", cache_dir=tmp_path):
        empty = allocate_result((0,), float)
        objects = copy_result(np.array([{"meaning": 2}], dtype=object))
    assert empty.shape == (0,)
    assert objects[0] == {"meaning": 2}
    assert array_storage(objects) == "memory"


@pytest.mark.parametrize("quantity", ["sigma_2d", "sigma_3d"])
def test_raw_authority_and_native_round_trip(tmp_path, quantity):
    values = np.linspace(1, 12, 12).reshape(2, 2, 3, 1) * (1 + 2j)
    units = {"linear_quantity": quantity, "frequency": "GHz", "phase_wrap": "0_360"}
    expected = grid(rcs=values, rcs_storage_domain="raw_field", units=units)
    cache = tmp_path / "cache"
    cache.mkdir()
    with operation_result_storage("disk", cache_dir=cache):
        actual = grid(rcs=values, rcs_storage_domain="raw_field", units=units)
        duplicate = grid(**actual._sample_kwargs(), units=units, _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN)
    assert actual.sample_storage == duplicate.sample_storage == "disk"
    np.testing.assert_array_equal(duplicate.rcs, expected.rcs)
    np.testing.assert_array_equal(duplicate.extra["rcs_amp_real"], values.real)
    path = tmp_path / "result.grim"
    duplicate.save(path)
    restored = RcsGrid.load(path)
    np.testing.assert_array_equal(restored.rcs, expected.rcs)
    del actual, duplicate
    gc.collect()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("method", ["complex", "power", "phase", "legacy"])
def test_mutable_samples_remain_compatible_in_disk_scope(tmp_path, method):
    values = np.full((2, 2, 3, 1), 1 + 2j)
    expected, actual = grid(rcs=values), grid(rcs=values)
    def update(target):
        if method == "complex":
            target.set_complex((1, 1, 2, 0), 3 + 4j)
        elif method == "power":
            target.set_power_phase((1, 1, 2, 0), power=25.)
        elif method == "phase":
            target.set_power_phase((1, 1, 2, 0), phase=np.nan)
        else:
            target.rcs_phase[1, 1, 2, 0] = .25
    update(expected)
    with operation_result_storage("disk", cache_dir=tmp_path):
        update(actual)
    assert actual.sample_storage == "disk"
    np.testing.assert_array_equal(actual.power, expected.power)
    np.testing.assert_array_equal(actual.phase, expected.phase)
    np.testing.assert_array_equal(actual.rcs, expected.rcs)


def test_transformed_writable_disk_source_is_never_adopted(tmp_path):
    with operation_result_storage("disk", cache_dir=tmp_path):
        source = grid(rcs_power=np.ones((2, 2, 3, 1)))
        exposed = source.rcs_power
        exposed[0, 1, 0, 0] = 10
        transformed = source.swap_elevation_azimuth()
        assert not np.shares_memory(exposed, transformed._power)
        exposed[0, 1, 0, 0] = 99
    assert transformed.power[1, 0, 0, 0] == 10
