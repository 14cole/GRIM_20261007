"""Selected caches retain selected samples, and adaptive saves sample any layout."""
import gc
import zipfile

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.batch import _grim_save_compression_decision, _representative_contiguous_bytes, _stage_and_publish_grim_batch


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_selected_cache_discards_extracted_full_members(tmp_path, order, storage):
    shape = (128, 2, 1024, 2)
    field = np.full(shape, 1 + 2j, dtype=np.complex128, order=order)
    grid = RcsGrid(np.arange(128), [0, 1], np.arange(1024) + 1, ["HH", "VV"],
                   rcs=field, extra={"cell_values": np.ones(shape + (2,), dtype=np.float32)})
    source = grid.save(tmp_path / "source.grim")
    cache = tmp_path / "cache"
    cache.mkdir()
    selected = RcsGrid.load(source, storage=storage, cache_dir=cache,
        selection={"azimuth": (0, 0), "elevation": (0, 0), "frequency": (1, 16), "polarization": ["HH"]})
    cache_bytes = sum(item.stat().st_size for item in cache.rglob("*") if item.is_file())
    assert cache_bytes < 16_384  # Includes numeric metadata and ownership marker.
    np.testing.assert_array_equal(selected.rcs, field[:1, :1, :16, :1])
    assert selected.extra["cell_values"].shape == (1, 1, 16, 1, 2)
    if storage == "memory":
        assert not list(cache.iterdir())
    else:
        view = np.asarray(selected._complex)
        del selected
        gc.collect()
        assert list(cache.iterdir())
        np.testing.assert_array_equal(view, field[:1, :1, :16, :1])
        del view
    gc.collect()
    assert not list(cache.iterdir())


@pytest.mark.parametrize("layout", ["C", "F", "transpose", "reverse"])
def test_compression_sampling_is_bounded_for_every_layout(tmp_path, layout):
    field = np.ones((128, 2, 1024, 2), dtype=np.complex128) * (1 + 2j)
    if layout == "F":
        field = np.asfortranarray(field)
    elif layout == "transpose":
        field = field.swapaxes(0, 2)
    elif layout == "reverse":
        field = field[::-1]
    sample = _representative_contiguous_bytes(field, 4096)
    assert 0 < len(sample) <= 4096
    grid = RcsGrid(np.arange(field.shape[0]), [0, 1], np.arange(field.shape[2]) + 1, ["HH", "VV"], rcs=field)
    decision = _grim_save_compression_decision(grid)
    assert decision["compressed"]
    assert 0 < decision["sample_bytes"] <= 1024**2
    target = tmp_path / "saved.grim"
    _stage_and_publish_grim_batch([(grid, str(target), "history")])
    with zipfile.ZipFile(target) as archive:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
    restored = RcsGrid.load(target)
    np.testing.assert_array_equal(restored.rcs, grid.rcs)
