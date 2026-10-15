"""Numerical parity and bounded workspace for managed operation outputs."""
from __future__ import annotations

import gc
import json
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.result_storage import operation_result_storage
from GRIM_Backend.datasets import transforms as transforms


def _grid(kind="complex", dtype=np.float64, *, shape=(7, 3, 8, 2)):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    power = (1.0 + values % 19).astype(dtype)
    phase = ((values % 17 - 8) * 0.17).astype(dtype)
    if kind == "partial":
        phase[1, 0, 0, 0] = np.nan
        power[0, 0, 0, 0] = 0.0
        power[2, 0, 0, 0] = np.nan
    elif kind == "unknown":
        phase[:] = np.nan
    extra = {}
    if kind == "raw":
        raw = np.sqrt(power.astype(np.float64) / (4 * np.pi)) * np.exp(1j * phase)
        extra.update(rcs_amp_real=raw.real, rcs_amp_imag=raw.imag)
    return RcsGrid(np.arange(shape[0]) * 10., np.arange(shape[1]) * 5.,
                   8. + np.arange(shape[2]) * .1, ["HH", "VV"][:shape[3]],
                   rcs_power=power, rcs_phase=phase, extra=extra,
                   units={"frequency": "GHz", "azimuth": "deg", "elevation": "deg",
                          "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"})


OPERATIONS = {
    "duplicate": lambda g: transforms.duplicate_dataset(g),
    "phase_shift": lambda g: transforms.shift_dataset(g, phase_degrees=37.5),
    "offset": lambda g: transforms.offset_db(g, 4.25),
    "extrusion": lambda g: transforms.convert_extrusion(g, to="dbke", length_m=2.),
    "decimate_power": lambda g: transforms.decimate_axis(g, axis="frequency", factor=3),
    "decimate_complex": lambda g: transforms.decimate_axis(g, axis="elevation", factor=2, mode="coherent"),
    "medianize": lambda g: transforms.medianize_azimuth(g, window_degrees=25., slide_degrees=15.),
    "gate": lambda g: transforms.time_gate(g, start_m=-.5, stop_m=.5),
    "phase_center": lambda g: transforms.translate_phase_center(g, x_m=.1, y_m=.2, z_m=.3),
    "coherent_add": lambda g: g.coherent_add(g),
    "coherent_subtract": lambda g: g.coherent_subtract(g),
    "incoherent_add": lambda g: g.incoherent_add(g),
    "incoherent_subtract": lambda g: g.incoherent_subtract(g),
    "db_subtract": lambda g: g.arithmetic_db_subtract(g),
    "divide": lambda g: transforms.coherent_divide(g, g),
    "calibrate": lambda g: g.range_calibrate(g, g, range_offset_m=.1),
    "mean": lambda g: g.statistics_dataset("mean", ["azimuth", "elevation"]),
    "median": lambda g: g.statistics_dataset("median", ["frequency"], domain="dbsm", broadcast_reduced=True),
    "percentile": lambda g: g.statistics_dataset("p80", ["elevation", "frequency"]),
    "std": lambda g: g.statistics_dataset("std", ["azimuth", "frequency"]),
    "complex_mean": lambda g: g.statistics_dataset("mean", ["frequency"], domain="complex"),
}


def _assert_equal(left, right):
    assert left.storage_kind == right.storage_kind
    assert left.real_dtype == right.real_dtype
    for name in ("azimuths", "elevations", "frequencies", "polarizations", "power", "phase", "rcs"):
        np.testing.assert_array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)), err_msg=name)
    assert left.units == right.units
    for key in ("coherent_sample_qa_json", "range_calibration_json", "statistics_reduction_json", "time_gate_json"):
        if key in left.extra:
            assert json.loads(str(np.asarray(left.extra[key]).item())) == json.loads(str(np.asarray(right.extra[key]).item()))


@pytest.mark.parametrize("mode", ["disk", "auto"])
@pytest.mark.parametrize("kind,dtype", [("complex", np.float32), ("complex", np.float64),
                                        ("partial", np.float64), ("raw", np.float32)])
@pytest.mark.parametrize("name", list(OPERATIONS))
def test_numeric_result_storage_parity(tmp_path, mode, kind, dtype, name):
    source = _grid(kind, dtype)
    operation = OPERATIONS[name]
    expected = operation(source)
    with operation_result_storage(mode, ram_budget_bytes=0, cache_dir=tmp_path):
        result = operation(source)
    assert result.sample_storage == "disk"
    _assert_equal(result, expected)
    path = tmp_path / "result.grim"
    result.save(path)
    loaded = RcsGrid.load(path)
    _assert_equal(loaded, expected)
    del result
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


@pytest.mark.parametrize("name", ["offset", "medianize", "mean", "percentile", "incoherent_add", "extrusion"])
def test_missing_phase_remains_unknown(tmp_path, name):
    source = _grid("unknown")
    expected = OPERATIONS[name](source)
    with operation_result_storage("disk", cache_dir=tmp_path):
        result = OPERATIONS[name](source)
    _assert_equal(result, expected)
    assert result.phase_coverage() == "none"
    assert result.sample_storage == "disk"


def test_calibration_slices_reference_and_dut(tmp_path):
    source = _grid(shape=(2, 2, 70_000, 2))
    original = source.rcs_slice
    largest = 0
    def read(selection):
        nonlocal largest
        values = original(selection)
        largest = max(largest, values.size)
        return values
    # Reference/DUT computations are bounded. The compatibility content hash
    # intentionally retains its historical azimuth slab framing.
    with operation_result_storage("disk", cache_dir=tmp_path), mock.patch.object(source, "rcs_slice", side_effect=read):
        result = source.range_calibrate(source, source, range_offset_m=0.)
    assert result.sample_storage == "disk"
    assert largest <= 280_000
    np.testing.assert_allclose(result.rcs, source.rcs, rtol=1e-14, atol=1e-14)


def test_disk_coherent_subtract_counts_only_resident_destination(tmp_path):
    source = _grid(shape=(4, 3, 8, 2))
    # The same scratch allowance fits disk output, while RAM also needs the
    # retained field. Explicit bounds still govern actual arithmetic scratch.
    cells = int(np.prod(source.shape))
    scratch = cells * (12 * 16 + 8 * 8)
    with pytest.raises(MemoryError):
        source.coherent_subtract(source, maximum_working_bytes=scratch)
    with operation_result_storage("disk", cache_dir=tmp_path):
        result = source.coherent_subtract(source, maximum_working_bytes=scratch)
    assert result.sample_storage == "disk"
    np.testing.assert_array_equal(result.rcs, 0)
