"""Cost-proportional CPU reservations, per-unit BLAS teams and memory evidence in the 2-D drivers (October 2026).

- ``cpu_reservations``: every unit gets the larger of the fill rule and its cost-proportional share of the
  node, capped at the physical cores; uniform shares fill the cores, heavy units of a long sweep get the
  threads they need, an explicit assembly-thread setting wins.
- The drivers apply the reservation to assembly threads, the CPU allocation and the BLAS team of each unit,
  start pool workers with the node's BLAS cap, and record forecast versus measured peak memory per unit.
- The submit summary reports which mesh path each unit will take.
"""
import inspect
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
import ghost_backend.hpc.scheduler as scheduler  # noqa: E402

CORES, BUDGET = 96, 554.0


def _wave(reservations, costs, peaks, cores=CORES, budget=BUDGET, pool=None):
    """The first admission wave in cost order under the dispatcher's rules."""
    order = sorted(costs, key=lambda name: -costs[name])
    pool = len(order) if pool is None else pool
    used_cpu = used_ram = 0.0
    wave = []
    for name in order:
        if len(wave) < pool and (not wave or (used_ram + peaks[name] <= budget and used_cpu + reservations[name] <= cores)):
            wave.append(name)
            used_cpu += reservations[name]
            used_ram += peaks[name]
    return wave, used_cpu


def test_uniform_units_fill_the_cores_instead_of_losing_the_remainder():
    costs = {'u%d' % i: 1.0 for i in range(36)}
    peaks = {name: 5.0 for name in costs}
    old = {name: max(2, scheduler.assembly_threads_for_unit(CORES, 36, BUDGET, 5.0)) for name in costs}
    new = scheduler.cpu_reservations([(n, costs[n], peaks[n]) for n in costs], CORES, 36, BUDGET, cap=48)
    assert set(old.values()) == {2} and _wave(old, costs, peaks)[1] == 72          # the former rule: 24 cores idle
    assert set(new.values()) == {4} and _wave(new, costs, peaks)[1] == CORES       # now 24 units x 4 = 96


def test_heavy_units_of_a_sweep_get_their_cost_share_and_light_ones_one_cpu():
    costs = {'heavy': 100.0}
    costs.update({'light%d' % i: 1.0 for i in range(70)})
    peaks = {name: (12.0 if name == 'heavy' else 1.5) for name in costs}
    new = scheduler.cpu_reservations([(n, costs[n], peaks[n]) for n in costs], CORES, 71, BUDGET, cap=48)
    assert new['heavy'] == 48                     # ceil(1.5 * 100/170 * 96) = 85, capped at the physical cores
    assert all(new[n] == 1 for n in costs if n != 'heavy')
    wave, used = _wave(new, costs, peaks)
    assert wave[0] == 'heavy' and used == CORES and len(wave) == 49


def test_memory_bound_units_keep_the_fill_rule_and_explicit_settings_win():
    units = [('big', 1.0, 100.0)] + [('small%d' % i, 1.0, 1.0) for i in range(199)]
    new = scheduler.cpu_reservations(units, CORES, 71, BUDGET, cap=48)
    # a 0.5% cost share asks for one thread; the fill rule (five 100 GiB copies fit) grants 19
    assert new['big'] == scheduler.assembly_threads_for_unit(CORES, 71, BUDGET, 100.0) == 19
    assert all(new[name] == 1 for name in new if name != 'big')
    explicit = scheduler.cpu_reservations(units[:2], CORES, 71, BUDGET, configured=4, cap=48)
    assert explicit == {'big': 4, 'small0': 4}
    assert scheduler.cpu_reservations([], CORES, 8, BUDGET) == {}
    assert scheduler.cpu_reservations([('only', 0.0, 0.0)], CORES, 1, BUDGET, cap=48)['only'] == 48


def test_blas_cap_and_memory_probes():
    cap = scheduler.blas_thread_cap(CORES)
    assert 1 <= cap <= CORES
    assert scheduler.blas_thread_cap(1) == 1
    assert isinstance(scheduler.reset_peak_rss(), bool)
    peak = scheduler.peak_rss_gib()
    assert peak is None or peak > 0.0


def test_drivers_apply_the_reservation_to_blas_and_allocation_and_record_memory():
    from ghost_backend import run_local_monostatic, run_hpc_monostatic
    for driver in (run_local_monostatic, run_hpc_monostatic):
        star = inspect.getsource(driver._solve_and_export_star)
        assert 'threadpool_limits(limits=cpus, user_api="blas")' in star, driver.__name__
        assert 'cpu_allocation_scope(cpus)' in star, driver.__name__
        solve = inspect.getsource(driver._solve_and_export)
        assert '_record_memory_evidence(' in solve, driver.__name__
        worker = inspect.getsource(driver.worker if hasattr(driver, 'worker') else driver.main)
        assert 'cpu_reservations(' in worker and 'blas_thread_cap(' in worker, driver.__name__
    assert run_hpc_monostatic._TASKS_PER_CHILD == 8


def test_memory_evidence_lands_in_the_result_metadata(capsys):
    from ghost_backend import run_hpc_monostatic
    result = {'metadata': {}}
    context = {'forecast_peak_gib': 2.0}
    unit = {'geometry_stem': 'case', 'frequency_ghz': 1.0}
    with mock.patch.object(scheduler, 'peak_rss_gib', return_value=1.5):
        run_hpc_monostatic._record_memory_evidence(result, context, unit, True)
    evidence = result['metadata']['execution_memory']
    assert evidence['forecast_peak_gib'] == 2.0 and evidence['measured_peak_gib'] == 1.5
    assert evidence['scope'] == 'worker_process_since_unit_start'
    assert '0.75 of forecast' in capsys.readouterr().out


def test_schedule_records_carry_the_mesh_path():
    from ghost_backend import run_hpc_monostatic
    source = inspect.getsource(run_hpc_monostatic._plan_schedule)
    assert '"fine_polynomial_degree"' in source and '"base_polynomial_degree"' in source
    summary = inspect.getsource(run_hpc_monostatic.submit)
    assert 'Mesh path' in summary
