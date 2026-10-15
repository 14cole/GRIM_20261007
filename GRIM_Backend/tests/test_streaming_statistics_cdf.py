"""Exact population semantics with bounded statistics/CDF workspaces."""
import gc
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets import reduction_workspace as reduction
from GRIM_Backend.datasets.result_storage import operation_result_storage
from GRIM_Backend.plotting import distribution_analysis as distribution
from GRIM_Backend.plotting.modes import cdf_mode
from GRIM_Backend.plotting.requests import PlotCancelled


def _grid(dtype=np.float64, kind="complex", shape=(9, 3, 11, 2)):
    rng = np.random.default_rng(8304)
    power = rng.uniform(0.1, 100., shape).astype(dtype)
    phase = rng.uniform(-np.pi, np.pi, shape).astype(dtype)
    if kind == "partial":
        phase.ravel()[::13] = np.nan
        power.ravel()[::17] = np.nan
    elif kind == "unknown":
        phase[:] = np.nan
    extra = {}
    if kind == "raw":
        raw = np.sqrt(power.astype(float) / (4 * np.pi)) * np.exp(1j * phase)
        extra = {"rcs_amp_real": raw.real, "rcs_amp_imag": raw.imag}
    return RcsGrid(np.arange(shape[0]), np.arange(shape[1]), np.linspace(8., 12., shape[2]),
                   ["HH", "VV"][:shape[3]], rcs_power=power, rcs_phase=phase,
                   extra=extra, units={"frequency": "GHz", "rcs_linear_quantity": "sigma_3d"})


@pytest.mark.parametrize("dtype,kind", [(np.float32, "complex"), (np.float64, "complex"),
                                        (np.float64, "partial"), (np.float32, "raw"), (np.float64, "unknown")])
@pytest.mark.parametrize("axes", [(0,), (1, 2), (0, 2, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("statistic", ["mean", "min", "max", "std", "median", "percentile"])
@pytest.mark.parametrize("domain", ["magnitude", "complex", "dbsm", "dbke"])
def test_streamed_statistics_match_numpy(tmp_path, monkeypatch, dtype, kind, axes, statistic, domain):
    if statistic == "percentile" and domain == "complex":
        pytest.skip("Complex percentiles have no defined ordering contract")
    if statistic == "std" and domain.startswith("db"):
        pytest.skip("Log spread is not an absolute RCS dataset")
    monkeypatch.setattr(reduction, "BLOCK_CELLS", 7)
    source = _grid(dtype, kind)
    values = reduction.domain_values(source, (slice(None),) * 4, domain)
    reducer = getattr(np, "nan" + statistic)
    args = (73.5,) if statistic == "percentile" else ()
    expected = reducer(values, *args, axis=axes, keepdims=True)
    with operation_result_storage("disk", cache_dir=tmp_path):
        actual = reduction.reduce_dataset(source, domain, axes, statistic, 73.5)
    assert actual.dtype == expected.dtype
    eps = np.finfo(expected.real.dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=64 * eps, atol=64 * eps, equal_nan=True)


@pytest.mark.parametrize("percentile", [0., 5.5, 50., 73.5, 100.])
def test_exact_disk_percentile_and_cleanup(tmp_path, monkeypatch, percentile):
    values = np.random.default_rng(61).normal(size=4001)
    values[::13] = np.nan
    original = reduction.workspace_array
    monkeypatch.setattr(reduction, "workspace_array", lambda shape, dtype: original(shape, dtype, ram_limit_bytes=0))
    with operation_result_storage("memory", cache_dir=tmp_path):
        actual = reduction._exact_rank((values[i:i+19] for i in range(0, len(values), 19)), len(values), values.dtype, "percentile", percentile)
    assert actual == np.nanpercentile(values, percentile)
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


@pytest.mark.parametrize("values", [[1., np.inf], [-np.inf, 1.], [0., 1., 2., np.inf],
                                     [np.nan, -np.inf, 0., np.inf], [np.inf]])
@pytest.mark.parametrize("percentile", [0., 25., 50., 200. / 3., 100.])
def test_exact_rank_matches_numpy_infinity_neighbors(values, percentile):
    values = np.asarray(values)
    with np.errstate(invalid="ignore"):
        actual = reduction._exact_rank((values[:2], values[2:]), len(values), values.dtype, "percentile", percentile)
        expected = np.nanpercentile(values, percentile)
    np.testing.assert_array_equal(actual, expected)


def test_streaming_std_handles_large_offset():
    values = 1e9 + np.random.default_rng(52).normal(size=1_000_003)
    actual = reduction._stream_scalar((values[i:i + 65536] for i in range(0, len(values), 65536)), "std", values.dtype)
    assert actual == pytest.approx(np.std(values), rel=1e-8)


@pytest.mark.parametrize("kind,statistic,domain", [("complex", "mean", "complex"),
    ("partial", "median", "complex"), ("raw", "mean", "magnitude"),
    ("unknown", "median", "magnitude"), ("unknown", "percentile", "dbke"),
    ("partial", "std", "magnitude")])
def test_streamed_dataset_result_preserves_phase_metadata_and_native_roundtrip(tmp_path, monkeypatch, kind, statistic, domain):
    source = _grid(np.float64, kind)
    expected = source.statistics_dataset(statistic, domain=domain, percentile=73.5)
    monkeypatch.setattr(reduction, "BLOCK_CELLS", 7)
    original = reduction.workspace_array
    monkeypatch.setattr(reduction, "workspace_array", lambda shape, dtype: original(shape, dtype, ram_limit_bytes=0))
    with operation_result_storage("disk", cache_dir=tmp_path):
        actual = source.statistics_dataset(statistic, domain=domain, percentile=73.5)
    for name in ("power", "phase", "rcs"):
        np.testing.assert_allclose(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)),
                                   rtol=1e-13, atol=1e-13, equal_nan=True)
    if domain != "complex":
        assert np.all(np.isnan(actual.phase))
    assert actual.sample_storage == "disk"
    assert actual.extra["statistics_reduction_json"] == expected.extra["statistics_reduction_json"]
    filename = tmp_path / "statistics.grim"
    actual.save(filename)
    loaded = RcsGrid.load(filename)
    for name in ("power", "phase", "rcs"):
        np.testing.assert_array_equal(np.asarray(getattr(loaded, name)), np.asarray(getattr(actual, name)))
    assert loaded.units == actual.units
    del actual
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


def _selection(dataset):
    return tuple(np.arange(n) for n in dataset.shape)


def _baseline_population(dataset, selection, scale, weighting):
    ai, ei, fi, pi = selection
    power = np.asarray(dataset.power[np.ix_(ai, ei, fi, [pi[0]])][..., 0], dtype=float)
    valid = np.isfinite(power) & (power >= 0.)
    zero = valid & (power == 0.)
    display = power if scale == "linear" else dataset.linear_to_default_db(np.where(valid, power, np.nan), frequency_value=dataset.frequencies[fi][None, None, :], eps=0.)
    if scale != "linear":
        display[zero] = -np.inf
    included = valid & (np.isfinite(display) | zero)
    weighted_axes = {"samples": (), "azimuth": (0,), "frequency": (2,), "azimuth_frequency": (0, 2), "all_axes": (0, 1, 2)}[weighting]
    coordinates = (dataset.azimuths[ai], dataset.elevations[ei], dataset.frequencies[fi])
    weights = [distribution.quadrature_weights(axis) if i in weighted_axes else np.full(len(axis), 1. / len(axis)) for i, axis in enumerate(coordinates)]
    grid_weights = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
    values, weights = display[included], grid_weights[included]
    order = np.argsort(values, kind="stable")
    return values[order], weights[order], dict(selected=power.size, valid=values.size, zero=np.count_nonzero(zero), excluded=power.size-values.size)


@pytest.mark.parametrize("scale", ["linear", "dbsm"])
@pytest.mark.parametrize("weighting", ["samples", "azimuth", "frequency", "azimuth_frequency", "all_axes"])
def test_bounded_cdf_preserves_population_weights_ties_and_thresholds(tmp_path, monkeypatch, scale, weighting):
    monkeypatch.setattr(distribution, "BLOCK_CELLS", 7)
    source = _grid(np.float64, "partial")
    power = source.rcs_power
    power.ravel()[::4] = 0.
    power.ravel()[1::5] = 1.
    power.ravel()[3::11] = -1.
    source.azimuths = np.array([0., 0., 1., 5., 6., 20., 30., 31., 60.])
    source.units.update(rcs_log_unit="dBke", rcs_linear_quantity="sigma_2d")
    selection = (np.array([8, 3, 0, 1, 7]), np.arange(3), np.array([10, 0, 4, 8]), [0])
    expected, weights, counts = _baseline_population(source, selection, scale, weighting)
    config = distribution.options({"weighting": weighting, "threshold": 0.})
    with operation_result_storage("disk", cache_dir=tmp_path):
        pop = distribution.population(source, selection, scale=scale, weighting=weighting)
        curve = distribution.prepare_curve("Run", source, selection, "All", config, scale=scale)
    np.testing.assert_array_equal(pop["values"], expected)
    np.testing.assert_array_equal(pop["weights"], weights)
    assert pop["counts"] == counts
    for threshold in [-1e4, 0., 1., 10., 1e4]:
        wanted = (100. * weights[expected <= threshold].sum() / weights.sum(),
                  100. * weights[expected >= threshold].sum() / weights.sum())
        np.testing.assert_allclose(distribution.threshold_lookup(curve, threshold), wanted, rtol=1e-13, atol=1e-13)


def test_cdf_cancel_releases_external_workspace(tmp_path, monkeypatch):
    source = _grid()
    original = distribution.workspace_array
    monkeypatch.setattr(distribution, "workspace_array", lambda shape, dtype: original(shape, dtype, ram_limit_bytes=0))
    monkeypatch.setattr(distribution, "BLOCK_CELLS", 7)
    calls = 0
    def canceled():
        nonlocal calls
        calls += 1
        return calls > 3
    with operation_result_storage("memory", cache_dir=tmp_path), pytest.raises(PlotCancelled):
        distribution.prepare_curve("Run", source, _selection(source), "All", distribution.options(), scale="dbsm", cancel=canceled)
    gc.collect()
    assert not list(tmp_path.glob("grim-result-*"))


def test_threshold_update_reuses_exact_population_and_invalidates(monkeypatch):
    from test_sweep_distribution_expansion import grid, owner
    source = grid([0., 1., 1., 10.])
    config = {"threshold_enabled": True, "threshold": 0.}
    harness = owner(source, cdf=config)
    cdf_mode.render(harness)
    from GRIM_Backend.plotting.modes.common import NativeSamples
    line = harness.plot_ax.lines[0]
    line._grim_native_samples = NativeSamples.curve(line.get_xdata(), line.get_ydata())
    config["threshold"] = 50.
    with mock.patch.object(distribution, "population", side_effect=AssertionError("reread population")):
        assert cdf_mode.update_threshold(harness)
    assert harness.plot_ax.lines[0]._grim_cdf_threshold_lookup == pytest.approx((100., 0.))
    assert harness.plot_ax.lines[0].get_xdata()[-1] > 50.
    assert line._grim_native_samples.values[0, -1] == line.get_xdata()[-1]
    config["weighting"] = "azimuth"
    assert not cdf_mode.update_threshold(harness)
    cdf_mode.clear_state(harness.plot_figure)
    assert not cdf_mode.update_threshold(harness)
