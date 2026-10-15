"""Managed axis/merge results preserve samples and bound numerical scratch."""
import gc
import tracemalloc
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.result_storage import operation_result_storage, array_storage


def _source(kind="complex128"):
    shape = (4, 2, 5, 2)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    field = np.sqrt(values) * np.exp(1j * (values / 9))
    field[1, 0, 2, 0] = np.nan + 1j * np.nan
    units = {"frequency": "GHz", "azimuth": "deg", "elevation": "deg"}
    args = {}
    if kind.startswith("complex"):
        args["rcs"] = field.astype(kind)
    elif kind == "power":
        args["rcs_power"] = values
    elif kind == "partial":
        phase = np.angle(field)
        phase[0, 1, 3, 1] = np.nan
        phase[2, 1, 2, 0] = 4 * np.pi
        args.update(rcs_power=values, rcs_phase=phase)
    else:
        units.update(rcs_linear_quantity="sigma_2d", rcs_log_unit="dBke")
        args.update(rcs=field, rcs_storage_domain="raw_field")
    return RcsGrid([-180., -90., 0., 90.], [-10., 10.], [1., 2., 3., 4., 5.], ["HH", "VV"],
                   units=units, extra={"note": "preserve", "producer_array": values[..., None]}, **args)


def _assert_same(actual, expected, *, archive=False):
    assert actual.storage_kind == expected.storage_kind
    assert actual.storage_domain == expected.storage_domain
    for name in ("azimuths", "elevations", "frequencies", "polarizations", "power", "phase", "rcs"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.units == expected.units
    # Storage-dependent working-set estimates are intentionally different.
    for key in (set(expected.extra) if archive else set(actual.extra) | set(expected.extra)):
        if key in {"operation_provenance_json", "stitch_provenance_json", "rcs_domain", "load_storage"}:
            continue
        np.testing.assert_array_equal(actual.extra[key], expected.extra[key])


def _operations():
    return {
        "crop": lambda g: g.axis_crop(azimuths=[-90., 90.], frequencies=[2., 5.]),
        "swap": lambda g: g.swap_elevation_azimuth(),
        "edit": lambda g: g.edit_axis_value("azimuth", 0, 120.),
        "frequency_edit": lambda g: g.edit_axis_value("frequency", 0, 1.5),
        "mirror": lambda g: g.mirror_about_azimuth(0.),
        "wrap": lambda g: g.wrap_azimuth("0_360"),
        "phase_wrap": lambda g: g.wrap_phase("0_360"),
        "units": lambda g: g.convert_axis_units(frequency="Hz", azimuth="rad"),
        "interpolate": lambda g: g.interpolate_axis("frequency", [1., 1.5, 2., 3.5, 5.]),
        "measured_only": lambda g: g.interpolate_axis("frequency", [1., 1.5, 2., 3.5, 5.], support="measured-only"),
        "align": lambda g: g.align_to(g.axis_crop(azimuths=[-90., 0.], frequencies=[1., 3., 5.]), mode="interp"),
        "join": lambda g: RcsGrid.join_many(g.axis_crop(azimuths=[-180., -90.]), g.axis_crop(azimuths=[0., 90.])),
        "stitch": lambda g: RcsGrid.stitch_many(g, g, policy="priority-first"),
        "power_mean": lambda g: RcsGrid.stitch_many(g, g, policy="power-mean"),
        "overlap": lambda g: RcsGrid.overlap_many(g, g.axis_crop(frequencies=[2., 4.]))[0],
    }


@pytest.mark.parametrize("kind", ["complex64", "complex128", "power", "partial", "raw"])
@pytest.mark.parametrize("mode", ["memory", "disk", "auto"])
def test_axis_and_merge_result_storage_parity_and_roundtrip(tmp_path, kind, mode):
    source = _source(kind)
    original = np.array(source.power, copy=True)
    for name, operation in _operations().items():
        expected = operation(source)
        with operation_result_storage(mode, ram_budget_bytes=0, cache_dir=tmp_path):
            actual = operation(source)
        _assert_same(actual, expected)
        assert actual.sample_storage == ("memory" if mode == "memory" else "disk"), name
        restored = RcsGrid.load(actual.save(tmp_path / name))
        _assert_same(restored, actual, archive=True)
    np.testing.assert_array_equal(source.power, original)


@pytest.mark.parametrize("mode", ["disk", "auto"])
def test_coordinate_results_and_grid_ancillary_arrays_use_selected_storage(tmp_path, mode):
    source = _source()
    sentri = RcsGrid(source.azimuths, [80., 100.], source.frequencies, source.polarizations,
                     rcs=source.rcs, units={**source.units, "elevation_coordinate_convention": "sentri_theta_top_zero"},
                     extra={"producer_array": np.arange(source.power.size).reshape(source.power.shape)})
    waterline = RcsGrid(source.azimuths, [0.], source.frequencies, source.polarizations,
                       rcs=source.rcs[:, :1], extra={"producer_array": np.ones(source.rcs[:, :1].shape)})
    pair = source.axis_crop(azimuths=[-90., 0.])
    for grid, operation in (
        (sentri, lambda g: g.convert_sentri_elevation_to_grim()),
        (waterline, lambda g: g.convert_wedge_to_conic()),
        (pair, lambda g: g.combine_elevation_pair_to_azimuth_360(assumptions_attested=True)),
    ):
        expected = operation(grid)
        with operation_result_storage(mode, ram_budget_bytes=0, cache_dir=tmp_path):
            actual = operation(grid)
        _assert_same(actual, expected)
        assert actual.sample_storage == "disk"
        if "producer_array" in actual.extra:
            assert array_storage(actual.extra["producer_array"]) == "disk"


def test_wedge_regrid_disk_matches_memory_across_frequency_blocks(tmp_path):
    azimuth = np.arange(-180., 180., 30.)
    elevation = np.array([-30., -15., 0., 15., 30.])
    field = np.broadcast_to(np.array([2.+.4j, .8+.1j, .35-.2j]), (12, 5, 40, 3)).copy()
    field *= np.linspace(1., 2., 40)[None, None, :, None]
    source = RcsGrid(azimuth, elevation, np.linspace(1., 2., 40), ["VV", "HH", "VH"], rcs=field)
    expected = source.convert_wedge_to_conic()
    with operation_result_storage("disk", cache_dir=tmp_path):
        actual = source.convert_wedge_to_conic()
    _assert_same(actual, expected)
    assert actual.sample_storage == "disk"


def test_join_conflict_still_rejected_and_failed_files_cleaned(tmp_path):
    source = _source()
    other = RcsGrid(source.azimuths, source.elevations, source.frequencies, source.polarizations,
                    rcs=source.rcs * 2, units=source.units)
    with operation_result_storage("disk", cache_dir=tmp_path):
        with pytest.raises(ValueError, match="conflicting finite"):
            RcsGrid.join_many(source, other)
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


def test_disk_interpolation_never_materializes_full_complex_source(tmp_path):
    source = _source("partial")
    with mock.patch.object(RcsGrid, "rcs", property(lambda self: (_ for _ in ()).throw(AssertionError("full rcs read")))):
        with operation_result_storage("disk", cache_dir=tmp_path):
            result = source.interpolate_axis("frequency", np.linspace(1., 5., 17))
    assert result.sample_storage == "disk"


def test_disk_interpolation_temporary_allocations_are_bounded(tmp_path):
    field = np.ones((64, 2, 2048, 2), dtype=np.complex128)
    source = RcsGrid(np.arange(64.), [-1., 1.], np.arange(2048.) + 1., ["HH", "VV"], rcs=field)
    tracemalloc.start()
    with operation_result_storage("disk", cache_dir=tmp_path):
        result = source.interpolate_axis("frequency", np.linspace(1., 2048., 4095))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.sample_storage == "disk"
    assert peak < 12 * 1024**2

@pytest.mark.parametrize("operation", ["join", "stitch"])
def test_disk_merge_preflight_counts_resident_scratch_not_mapped_output(tmp_path, monkeypatch, operation):
    import GRIM_Backend.datasets.combine as combine
    monkeypatch.setattr(combine, "_JOIN_MERGE_BLOCK_CELLS", 32)
    source = _source()
    combine_op = (lambda: RcsGrid.join_many(source, source, max_output_bytes=4000)) if operation == "join" else (
        lambda: RcsGrid.stitch_many(source, source, max_output_bytes=6000)
    )
    with operation_result_storage("memory"):
        with pytest.raises(MemoryError):
            combine_op()
    with operation_result_storage("disk", cache_dir=tmp_path):
        result = combine_op()
    assert result.sample_storage == "disk"
