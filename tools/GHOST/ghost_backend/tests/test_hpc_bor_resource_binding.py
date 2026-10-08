"""Compute-node forecasts and solver admission share resource limits."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest import mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend import run_hpc_bor_monostatic as driver
from ghost_backend.bor import dispatch, memory, solver
from ghost_backend.bor.options import validate_options
from ghost_backend.execution import options as execution


def manifest():
    return {"aspects_deg": [0., 90.], "solver_config": dict(geometry_units="meters",
        workers_per_unit=8, mesh_certification=False, assembly="auto", table_precision="auto",
        bor_execution_options=dict(factorization="auto", near_backend="threads"))}


def resource(peak, backend="dense", assembly="tables"):
    return dict(estimated_peak_gb=peak, stream_spill_candidate_gb=.25,
        assembly_estimate=assembly, table_precision_estimate="double",
        bor_execution_options=validate_options(dict(factorization=backend, near_backend="threads")))


def test_reprice_stale_login_estimate_and_bind_compute_choices():
    pair = dict(frequency_ghz=1., estimated_peak_gb=1.77)
    saved = manifest()
    before_pair, before_saved = deepcopy(pair), deepcopy(saved)
    profile = execution.current_options()
    calls = []
    def estimate(*args, **kwargs):
        calls.append(deepcopy(kwargs))
        assert execution.allocated_cpu_budget() == 4
        assert execution.allocated_memory_budget() == 8.
        assert execution.current_options() == profile
        return resource(3.13)
    with mock.patch.object(execution, "_usable_logical_cpus", return_value=32), \
            mock.patch.object(dispatch, "estimate_bor_resources", side_effect=estimate):
        plan = driver._compute_resource_plan(pair, {}, ".", saved, 4, 8.)
    assert len(calls) == 2
    assert calls[0]["assembly"] == "auto"
    assert calls[1]["assembly"] == "tables"
    assert calls[1]["table_precision"] == "double"
    assert calls[1]["bor_options"]["factorization"] == "dense"
    assert calls[1]["workers"] == 4
    assert plan["estimated_peak_gb"] == 3.13
    assert plan["memory_reservation_gib"] * 1024**3 / 1e9 >= 3.13
    assert pair == before_pair and saved == before_saved


def test_compressed_capacity_is_frozen_and_repriced():
    calls = []
    def estimate(*args, **kwargs):
        calls.append(deepcopy(kwargs))
        return resource(3. if len(calls) == 1 else 3.2, "compressed", "compressed")
    with mock.patch.object(dispatch, "estimate_bor_resources", side_effect=estimate), \
            mock.patch("ghost_backend.compressed.runtime.automatic_storage_bytes", return_value=64*1024**2+1):
        plan = driver._compute_resource_plan(dict(frequency_ghz=1.), {}, ".", manifest(), 4, 8.)
    assert calls[1]["assembly"] == "tables"
    assert calls[1]["bor_options"]["compressed_storage_mib"] == 65
    assert plan["bor_execution_options"] == calls[1]["bor_options"]
    assert plan["estimated_peak_gb"] == 3.2


@pytest.mark.parametrize("peak", [float("nan"), float("inf"), 0., -1.])
def test_invalid_peak_rejected(peak):
    with mock.patch.object(dispatch, "estimate_bor_resources", return_value=resource(peak)):
        with pytest.raises(ValueError, match="positive and finite"):
            driver._compute_resource_plan(dict(frequency_ghz=1.), {}, ".", manifest(), 1, 8.)


@pytest.mark.parametrize("fail", [False, True])
def test_pool_scopes_hold_and_restore(fail):
    before = execution.allocated_memory_budget(), execution.current_options()
    def solve(*args):
        assert execution.allocated_cpu_budget() == 3
        assert execution.allocated_memory_budget() == 1.25
        assert execution.current_options() == before[1]
        if fail:
            raise RuntimeError("deliberate solver failure")
        return "written", "result.grim"
    with mock.patch.object(execution, "_usable_logical_cpus", return_value=32), \
            mock.patch.object(driver, "_solve_and_export", side_effect=solve):
        result = driver._solve_and_export_star(({}, {}, ".", ".", 3, 1.25))
    assert result[0] == ("err" if fail else "ok")
    assert (execution.allocated_memory_budget(), execution.current_options()) == before


def test_nested_scope_and_decimal_budget_guard():
    with execution.memory_allocation_scope(1.):
        with execution.memory_allocation_scope(8.):
            assert execution.allocated_memory_budget() == 1.
        with mock.patch("ghost_backend.twod.solver._configured_solve_memory_limit_gb", return_value=8.), \
                mock.patch.object(solver, "estimate_bor_dense_peak_gb", return_value=0.), \
                mock.patch.object(solver, "estimate_bor_total_peak_gb", return_value=1.05) as total:
            assert memory.solve_memory_limit_gb() == 1024**3 / 1e9
            assert solver._guard_bor_dense_memory(2, 2, 1, 1) == 1.05
            total.return_value = 1.08
            with pytest.raises(MemoryError):
                solver._guard_bor_dense_memory(2, 2, 1, 1)


def test_oversized_unit_cannot_enlarge_node_budget():
    with mock.patch.object(dispatch, "estimate_bor_resources", return_value=resource(10.)):
        plan = driver._compute_resource_plan(dict(frequency_ghz=1.), {}, ".", manifest(), 1, 2.)
        assert plan["estimated_peak_gb"] == 10.
        assert plan["memory_reservation_gib"] == 2.
        with execution.memory_allocation_scope(1.):
            nested = driver._compute_resource_plan(dict(frequency_ghz=1.), {}, ".", manifest(), 1, 2.)
        assert nested["memory_reservation_gib"] == 1.
