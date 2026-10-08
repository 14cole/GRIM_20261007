"""October 2026 experiments ported into the project, and the drivers' overhead items.

- BoR graded near rules: the coarse level is evaluated on every NEAR_CHECK_STRIDE-th point of a
  chunk (the fine level for every point, first); a chunk whose probes all pass is accepted at the
  fine level, a probed failure falls back to the complete check.  GHOST_BOR_NEAR_CHECK_STRIDE=0
  restores the complete check.
- BoR disjoint near pairs: the coarse meridian level on every stride-th pair of a batch, likewise.
- Drivers: a unit is verified before its solve and before its export, not after publication; the
  BoR deliverable is built in memory and written once; the forecast cache key excludes the CPU/RAM
  allocation and a reused forecast is repriced under the current one.
"""
import inspect
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.bor import kernels, solver as bor  # noqa: E402
from general_fixtures import fixture  # noqa: E402


# ---------------------------------------------------------------- BoR near rules

def _near_points(n=40, seed=3):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.05, 0.2, n)
    gap = 10 ** rng.uniform(-7, -1, n)
    angle = rng.uniform(0, np.pi, n)
    return (rho, np.zeros(n), np.cos(angle), np.sin(angle), rho + gap * np.cos(angle), gap * np.sin(angle),
            np.cos(angle + 0.3), np.sin(angle + 0.3))


def _parts(result):
    return list(result) if isinstance(result, (tuple, list)) else [result]


def _near_results(args, stride):
    with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': str(stride)}):
        green = kernels.modal_kernels_near(*[args[i] for i in (0, 1, 4, 5)], 40.0 - 2.0j, 9)
        mfie = kernels.mfie_kernels_near(*args, 40.0 - 2.0j, 9)
        ibc = kernels.ibc_kernels_near(*args, 40.0 - 2.0j, 9)
    return _parts(green) + _parts(mfie) + _parts(ibc)


class _FakeLevels:
    """``evaluate``/``next_orders`` stand-ins for _refine_near_chunk (kind 'g'):
    every point's value moves by 1e-10 per order step (accepted at 2e-8), except
    ``hard`` points whose coarse value is far off."""

    def __init__(self, count=5, hard=()):
        self.count, self.hard, self.calls = count, set(hard), []

    def evaluate(self, ids, orders, stable):
        self.calls.append((tuple(int(o) for o in orders), [int(i) for i in ids]))
        values = np.empty((len(ids), 2, self.count))
        for row, index in enumerate(ids):
            base = 2.0 if (index in self.hard and int(orders[0]) == 8) else 1.0 + 1e-10 * int(orders[0])
            values[row, 0] = base
            values[row, 1] = 0.5 * base
        return values

    @staticmethod
    def next_orders(orders):
        grown = np.asarray(orders) * 2
        return None if grown[0] > 64 else grown

    def run(self, ids, stride):
        out = np.zeros((max(ids) + 1, self.count), complex)
        with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': str(stride)}):
            kernels._refine_near_chunk('g', tuple(np.zeros(max(ids) + 1) for _ in range(4)), np.asarray(ids),
                                       False, np.array([8, 8]), self.evaluate, self.next_orders, out)
        return out


def test_default_stride_is_four_and_the_environment_overrides_it():
    assert kernels.NEAR_CHECK_STRIDE == 4
    with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': ''}):
        assert kernels._near_check_stride() == 4
    with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': '0'}):
        assert kernels._near_check_stride() == 0
    with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': '-3'}):
        assert kernels._near_check_stride() == 0
    with mock.patch.dict(os.environ, {'GHOST_BOR_NEAR_CHECK_STRIDE': 'x'}):
        assert kernels._near_check_stride() == 4


def test_passing_probes_accept_the_chunk_at_the_fine_level_with_a_quarter_of_the_coarse_work():
    levels = _FakeLevels()
    out = levels.run(list(range(12)), stride=4)
    assert levels.calls == [((16, 16), list(range(12))), ((8, 8), [0, 4, 8])]
    expected = 2.0 * ((1.0 + 1.6e-9) + 0.5j * (1.0 + 1.6e-9))
    np.testing.assert_array_equal(out, np.full((12, 5), expected))


def test_a_failing_probe_falls_back_to_the_complete_check_bitwise():
    sampled = _FakeLevels(hard=(4,))
    out = sampled.run(list(range(12)), stride=4)
    complete = _FakeLevels(hard=(4,))
    reference = complete.run(list(range(12)), stride=0)
    np.testing.assert_array_equal(out, reference)
    # fine for all, the probe, then the complete coarse level and the hard point's next level
    assert sampled.calls[:3] == [((16, 16), list(range(12))), ((8, 8), [0, 4, 8]), ((8, 8), list(range(12)))]
    assert sampled.calls[3] == ((32, 32), [4])
    assert complete.calls == [((16, 16), list(range(12))), ((8, 8), list(range(12))), ((32, 32), [4])]


def test_small_chunks_and_stride_zero_run_the_complete_check():
    levels = _FakeLevels()
    levels.run(list(range(7)), stride=4)            # fewer than two strides: no probe
    assert levels.calls == [((16, 16), list(range(7))), ((8, 8), list(range(7)))]
    levels = _FakeLevels()
    levels.run(list(range(12)), stride=0)
    assert levels.calls == [((16, 16), list(range(12))), ((8, 8), list(range(12)))]


def test_sampled_near_rules_match_the_complete_check_bitwise_on_a_mixed_batch():
    args = _near_points()
    for sampled, complete in zip(_near_results(args, 4), _near_results(args, 0)):
        np.testing.assert_array_equal(sampled, complete)


# ---------------------------------------------------------------- BoR disjoint meridian pairs

def _disjoint_batch(stride):
    gen = bor.Generatrix(bor.sphere_generatrix(0.05, 36))
    pairs = [(e, f) for e in range(0, 36, 4) for f in range(e + 3, 36, 5)]
    assert len(pairs) >= 8
    with mock.patch.object(bor, 'NEAR_MERIDIAN_CHECK_STRIDE', stride), \
            mock.patch.object(bor, '_contract_near_batch', wraps=bor._contract_near_batch) as contracted:
        results = bor._converged_disjoint_batch(gen, gen, pairs, 21.0, 4, ('efie', 'mfie'), signed=False)
    evaluated = sum(len(call.args[0]) for call in contracted.call_args_list)
    return results, evaluated, len(pairs)


def test_sampled_meridian_check_matches_the_complete_one_and_evaluates_fewer_pairs():
    assert bor.NEAR_MERIDIAN_CHECK_STRIDE == 4
    sampled, sampled_count, n = _disjoint_batch(4)
    complete, complete_count, _ = _disjoint_batch(0)
    for (blocks, order, _), (reference, reference_order, _) in zip(sampled, complete):
        assert order == reference_order
        for kind in ('efie', 'mfie'):
            np.testing.assert_array_equal(blocks[kind], reference[kind])
    assert complete_count >= 2 * n
    assert sampled_count <= n + (n + 3) // 4 + (complete_count - 2 * n)


# ---------------------------------------------------------------- drivers

def test_units_are_verified_before_the_solve_and_before_the_export_only():
    from ghost_backend import run_local_monostatic, run_local_bor, run_hpc_monostatic, run_hpc_bor_monostatic
    for driver in (run_local_monostatic, run_local_bor, run_hpc_monostatic, run_hpc_bor_monostatic):
        source = inspect.getsource(driver._solve_and_export)
        assert source.count('_verify_run_provenance(') == 2, driver.__name__
        tail = source[source.index('export_result_to_grim('):]
        assert '_verify_' not in tail, driver.__name__


def _bodies():
    theta = np.linspace(0.0, 180.0, 19)
    rng = np.random.default_rng(7)
    out = {}
    for frequency in (1.0, 1.5):
        out[frequency] = dict(theta_deg=theta,
                              amp_vv=rng.normal(size=19) + 1j * rng.normal(size=19),
                              amp_hh=rng.normal(size=19) + 1j * rng.normal(size=19))
    return out


def test_bor_deliverable_is_written_once(tmp_path):
    from ghost_backend.assembly import fields
    from ghost_backend.io import grim
    profile = 0.1 * np.column_stack((np.sin(np.linspace(0, np.pi, 7)), np.cos(np.linspace(0, np.pi, 7))))
    bodies = _bodies()
    with mock.patch.object(grim, '_save_grim_npz', wraps=grim._save_grim_npz) as writes:
        saved = fields.save_monostatic_grim(bodies, profile, str(tmp_path / 'body'),
                                            azimuths_deg=[0.0, 30.0], elevations_deg=[0.0, 90.0],
                                            artifact_metadata={'geometry_input_sha256': 'abc'})
    assert writes.call_count == 1
    assert Path(saved) == tmp_path / 'body.grim'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['body.grim']
    loaded = fields.load_body_grim(saved)
    for frequency, body in bodies.items():
        np.testing.assert_allclose(loaded[frequency]['amp_vv'], body['amp_vv'])
        np.testing.assert_allclose(loaded[frequency]['amp_hh'], body['amp_hh'])
    with np.load(saved, allow_pickle=False) as payload:
        assert payload['polarizations'].tolist() == ['VV', 'HH', 'VH']
        assert str(payload['geometry_input_sha256']) == 'abc'
        assert bool(payload['raw_complex_amplitude_preserved'])
        np.testing.assert_allclose(payload['body_profile_rho_m'], profile[:, 0])


def test_reused_forecast_is_repriced_under_the_current_allocation():
    from ghost_backend.execution.options import execution_scope, validate_options
    from ghost_backend.execution.selection import select_backend
    from ghost_backend.twod.preparation import preparation_scope
    arguments = dict(geometry_snapshot=fixture('reentrant', 48), frequencies_ghz=[1.0], elevations_deg=[0.0, 90.0],
                     geometry_units='meters', solver_method='auto', max_panels=10000)
    options = validate_options(dict(factorization='adaptive'))
    with preparation_scope():
        first = select_backend(arguments, options, certified=True)
        with execution_scope(options, assembly_threads=2, memory_budget_gib=1.0):
            reused = select_backend(arguments, options, certified=True)
    with preparation_scope():
        with execution_scope(options, assembly_threads=2, memory_budget_gib=1.0):
            fresh = select_backend(arguments, options, certified=True)
    assert not first.get('forecast_reused') and reused['forecast_reused'] and not fresh.get('forecast_reused')
    assert reused['selected'] == fresh['selected'] and reused['retry_order'] == fresh['retry_order']
    assert reused['admission_budget_gib'] == pytest.approx(fresh['admission_budget_gib'])
    for mode, candidate in fresh['candidates'].items():
        assert reused['candidates'][mode]['cost'] == pytest.approx(candidate['cost'], rel=1e-12)
        assert reused['candidates'][mode]['peak_gb'] == pytest.approx(candidate['peak_gb'], rel=1e-12)
    assert len(reused['meshes']) == len(fresh['meshes'])
    for a, b in zip(reused['meshes'], fresh['meshes']):
        assert a['unknowns'] == b['unknowns'] and a['dense_peak_gib'] == pytest.approx(b['dense_peak_gib'])
