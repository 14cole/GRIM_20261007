"""Finite-layer cutoff references independent of characteristic impedance.

The oracle exponentiates the first-order tangential Maxwell system directly.
It does not use the solver's kz, branch selection, sinc, or impedance cascade.
"""
from __future__ import annotations

import cmath
import copy
from dataclasses import replace
import math
import pickle
import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from ibc import compute as c


def slab(thickness=0.003, eps=0.5, mu=1.0):
    return c.LoadedLayer(thickness, False, 0.0,
                         c.ConstantMaterial(complex(eps), complex(mu)), None)


def sheet(resistance=240.0):
    return c.LoadedLayer(0.0, False, 0.0, None, None, True, resistance)


def reference(frequency, angle, layers, pol, *, thickness_scale=1.0,
              eps_scale=1.0, mu_scale=1.0):
    omega = 2.0 * math.pi * frequency * 1e9
    kx = omega * math.sqrt(c.MU0 * c.EPS0) * math.sin(math.radians(angle))
    z0 = c.ETA0 * (1.0 / math.cos(math.radians(angle)) if pol == "te"
                   else math.cos(math.radians(angle)))
    chain = np.eye(2, dtype=complex)
    for layer in layers:
        if layer.is_sheet:
            matrix = np.array([[1.0, 0.0], [1.0 / layer.sheet_resistance, 1.0]])
        else:
            material = layer.table_0deg
            if isinstance(material, c.ConstantMaterial):
                eps_r, mu_r = material.eps_r, material.mu_r
            else:
                # Test tables use exact sample frequencies, no shared interpolator.
                index = material.freq_ghz.index(frequency)
                eps_r, mu_r = material.eps_r[index], material.mu_r[index]
            eps = c.EPS0 * eps_r * eps_scale
            mu = c.MU0 * mu_r * mu_scale
            if pol == "te":
                upper, lower = omega * mu, omega * eps - kx ** 2 / (omega * mu)
            else:
                upper, lower = omega * mu - kx ** 2 / (omega * eps), omega * eps
            matrix = expm(1j * np.array([[0.0, upper], [lower, 0.0]])
                          * layer.thickness_m * thickness_scale)
        chain = chain @ matrix
    a, b, cc, d = chain.ravel()
    den = a + b / z0 + cc * z0 + d
    return chain, {
        "air": (a + b / z0 - cc * z0 - d) / den,
        "metal": (b - z0 * d) / (b + z0 * d),
        "insertion": 2.0 / den,
    }


def coefficient(metrics, kind):
    return (10.0 ** (metrics[kind + "_loss_db"] / 20.0)
            * cmath.exp(1j * math.radians(metrics[kind + "_phase_deg"])))


def assert_response(metrics, expected, atol=2e-11):
    assert all(math.isfinite(value) for value in metrics.values())
    for kind, value in expected.items():
        assert abs(coefficient(metrics, kind) - value) < atol
    air_absorption = 1.0 - abs(expected["air"]) ** 2 - abs(expected["insertion"]) ** 2
    metal_absorption = 1.0 - abs(expected["metal"]) ** 2
    assert 10.0 ** (metrics["air_absorption_db"] / 10.0) == pytest.approx(
        max(0.0, air_absorption), abs=atol)
    assert 10.0 ** (metrics["metal_absorption_db"] / 10.0) == pytest.approx(
        max(0.0, metal_absorption), abs=atol)


def check_paths(freqs, angle, layers, pol, **scales):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        properties = c.prepare_layer_properties_many(freqs, layers)
        waves = c.prepare_layer_wave_terms_many(
            freqs, angle, layers, pol,
            eps_scale=scales.get("eps_scale", 1.0),
            mu_scale=scales.get("mu_scale", 1.0), prepared_properties=properties)
        paths = [
            c.compute_angle_metrics_many(freqs, angle, layers, pol, **scales),
            c.compute_angle_metrics_many(freqs, angle, layers, pol,
                                         prepared_properties=properties, **scales),
            c.compute_angle_metrics_many(freqs, angle, layers, pol,
                                         prepared_wave_terms=waves, return_arrays=True, **scales),
        ]
        for index, f in enumerate(freqs):
            matrix, expected = reference(f, angle, layers, pol, **scales)
            scalar = c.compute_angle_metrics(f, angle, layers, pol, **scales)
            assert_response(scalar, expected)
            for path in paths:
                assert_response({key: values[index] for key, values in path.items()}, expected)
            np.testing.assert_allclose(
                np.asarray(c.cascade_abcd(f, angle, layers, pol, **scales)).reshape(2, 2),
                matrix, rtol=3e-12, atol=2e-11)
    return waves


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("angle,eps,mu", [
    (30.0, math.sin(math.radians(30.0)) ** 2, 1.0),
    (30.0, 0.25, 1.0), (45.0, 0.5, 1.0), (60.0, 0.75, 1.0),
    (45.0, 0.25, 2.0), (45.0, -0.5, -1.0),
])
def test_cutoff_all_response_paths_match_maxwell(angle, eps, mu, pol):
    check_paths([1.0, 10.0, 18.0], angle, [slab(eps=eps, mu=mu)], pol)


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_exact_zero_and_analytic_cutoff_matrix(pol):
    layer = slab()
    _zc, kz = c.layer_wave_params(10e9, 45.0, 0.5 + 0j, 1.0 + 0j, pol)
    assert kz == 0.0  # Exercise the exact, formerly failing branch explicitly.
    omega = 2 * math.pi * 10e9
    matrix = (np.array([[1, 1j * omega * c.MU0 * layer.thickness_m], [0, 1]])
              if pol == "te" else
              np.array([[1, 0], [1j * omega * c.EPS0 * 0.5 * layer.thickness_m, 1]]))
    np.testing.assert_allclose(np.asarray(c.cascade_abcd(10, 45, [layer], pol)).reshape(2, 2),
                               matrix, rtol=2e-14, atol=2e-14)
    check_paths([10.0], 45.0, [layer], pol)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("offset", [-1e-8, -1e-12, -1e-15, 0.0, 1e-15, 1e-12, 1e-8])
def test_cutoff_is_continuous_from_propagating_and_evanescent_sides(pol, offset):
    layers = [slab(eps=0.5 + offset)]
    check_paths([10.0], 45.0, layers, pol)
    center = reference(10.0, 45.0, [slab()], pol)[1]
    actual = c.compute_angle_metrics(10.0, 45.0, layers, pol)
    for kind in center:
        assert abs(coefficient(actual, kind) - center[kind]) < 4e-8


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("loss", [1e-14, 1e-9, 1e-4])
def test_small_passive_loss_through_cutoff(pol, loss):
    check_paths([2.0, 10.0, 18.0], 45.0, [slab(eps=0.5 - 1j * loss)], pol)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_cutoff_in_multilayer_with_resistive_sheets(pol, position):
    layers = [slab(0.001, 3.1 - 0.1j, 1.2 - 0.02j), slab(0.002, 1.3)]
    layers.insert(position, slab())
    layers.insert(1, sheet())
    layers.append(sheet(400.0))
    check_paths([3.0, 10.0, 17.0], 45.0, layers, pol)


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_dispersive_vector_crosses_cutoff_and_reuses_prepared_geometry(pol, monkeypatch):
    freqs = [8.0, 10.0, 12.0]
    layer = slab()
    layer.table_0deg = c.MaterialTable(freqs, [0.5 - 1e-6, 0.5, 0.5 + 1e-6], [1.0] * 3)
    layers = [sheet(), layer, slab(0.002, 2.0 - 0.1j)]
    waves = check_paths(freqs, 45.0, layers, pol)
    # Plain tuple caches remain consumable, including cutoff.
    legacy = [tuple(term) if term is not None else None for term in waves]
    legacy_result = c.compute_angle_metrics_many(freqs, 45.0, layers, pol,
                                                prepared_wave_terms=legacy)
    for i, f in enumerate(freqs):
        assert_response({k: v[i] for k, v in legacy_result.items()}, reference(f, 45, layers, pol)[1])
    varied = [replace(layers[0], sheet_resistance=110),
              replace(layer, thickness_m=0.007), layers[2]]
    # The cached finite terms must support thickness/resistance search without
    # a material lookup, as in the inverse/thickness workflows.
    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("Prepared terms unexpectedly re-interpolated material")
    monkeypatch.setattr(c, "layer_properties_many", unexpected_lookup)
    actual = c.compute_angle_metrics_many(freqs, 45, varied, pol, thickness_scale=1.3,
                                          prepared_wave_terms=waves)
    for i, f in enumerate(freqs):
        assert_response({k: v[i] for k, v in actual.items()},
                        reference(f, 45, varied, pol, thickness_scale=1.3)[1])


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_material_and_thickness_scale_cutoff(pol):
    check_paths([10.0], 45.0, [slab(eps=2.0, mu=2.0)], pol,
                eps_scale=0.5, mu_scale=0.25, thickness_scale=1.7)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("copy_cache", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_prepared_cache_copy_preserves_cutoff_without_material_lookup(pol, copy_cache, monkeypatch):
    layers = [slab()]
    waves = copy_cache(c.prepare_layer_wave_terms_many([10.0], 45.0, layers, pol))
    zc, kz = waves[0]
    assert len(waves[0]) == 2
    assert kz[0] == 0.0
    assert math.isinf(zc[0].real) if pol == "te" else zc[0] == 0.0

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("Copied prepared terms lost finite material couplings")
    monkeypatch.setattr(c, "layer_properties_many", unexpected_lookup)
    result = c.compute_angle_metrics_many([10.0], 45.0, layers, pol, prepared_wave_terms=waves)
    assert_response({key: values[0] for key, values in result.items()},
                    reference(10.0, 45.0, layers, pol)[1])


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_numpy_unavailable_uses_finite_scalar_cutoff(pol, monkeypatch):
    monkeypatch.setattr(c, "NUMPY_AVAILABLE", False)
    result = c.compute_angle_metrics_many([10.0], 45.0, [slab()], pol)
    assert_response({key: values[0] for key, values in result.items()},
                    reference(10.0, 45.0, [slab()], pol)[1])


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_thick_loss_scaling_after_cutoff_and_sheet(pol):
    layers = [slab(), sheet(), slab(2.0, 10.0 - 10.0j)]
    freq, angle = 18.0, 45.0
    # A two-metre absorbing layer is an independent half-space limit at 18 GHz.
    omega = 2 * math.pi * freq * 1e9
    kz = cmath.sqrt(omega ** 2 * c.MU0 * c.EPS0 * (10 - 10j - 0.5))
    load = omega * c.MU0 / kz if pol == "te" else kz / (omega * c.EPS0 * (10 - 10j))
    chain, _ = reference(freq, angle, layers[:2], pol)
    a, b, cc, d = chain.ravel()
    zin = (a * load + b) / (cc * load + d)
    z0 = c.ETA0 / math.cos(math.radians(angle)) if pol == "te" else c.ETA0 * math.cos(math.radians(angle))
    gamma = (zin - z0) / (zin + z0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scalar = c.compute_angle_metrics(freq, angle, layers, pol)
        vector = c.compute_angle_metrics_many([freq], angle, layers, pol)
    for result in (scalar, {k: v[0] for k, v in vector.items()}):
        assert all(math.isfinite(v) for v in result.values())
        assert abs(coefficient(result, "air") - gamma) < 3e-12
        assert abs(coefficient(result, "metal") - gamma) < 3e-12
        assert result["insertion_loss_db"] == pytest.approx(-300.0, abs=1e-12)
        assert 10 ** (result["air_absorption_db"] / 10) == pytest.approx(1 - abs(gamma) ** 2, abs=3e-12)


def test_seeded_ordinary_stacks_match_independent_maxwell():
    rng = np.random.default_rng(20261010)
    for case in range(80):
        layers = [slab(float(rng.uniform(1e-5, 0.004)),
                       complex(rng.uniform(0.2, 15), -rng.uniform(0, 2)),
                       complex(rng.uniform(0.5, 3), -rng.uniform(0, 0.6)))
                  for _ in range(1 + case % 4)]
        if case % 3 == 0:
            layers.insert(case % len(layers), sheet(float(rng.uniform(30, 1000))))
        check_paths([float(rng.uniform(0.1, 18))], float(rng.uniform(0, 85)),
                    layers, ("te", "tm")[case % 2])
