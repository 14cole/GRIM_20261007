"""The desktop BoR sweep shares CPU/RAM without changing result semantics."""
from unittest import mock
import tempfile
import numpy as np

from ghost_backend.execution import frequency_sweep as fs
from ghost_backend.execution.options import execution_scope
from ghost_backend.bor import dispatch, checkpoints
from ghost_backend.twod.samples import compact_samples
from test_bor_physics_regression import _pec_sphere_snapshot


def arguments():
    return dict(geometry_snapshot=_pec_sphere_snapshot(explicit_elements=-20),
                frequencies_ghz=[.7, .6, .7], elevations_deg=[90., 0., 180.],
                geometry_units="meters", workers=2,
                bor_options=dict(factorization="dense", near_backend="threads"))


def test_real_spawned_bor_sweep_preserves_fields_duplicates_and_resume():
    args = arguments()
    outputs = []
    with compact_samples(), execution_scope({}, assembly_threads=4, memory_budget_gib=8.), \
            mock.patch("ghost_backend.execution.options.blas_core_budget", return_value=4):
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                result = checkpoints.run_checkpointed(dispatch.solve_monostatic_rcs_bor_survey,
                    args, directory, args["bor_options"], False, frequency_workers=workers)
                outputs.append(result)
                resumed = checkpoints.run_checkpointed(dispatch.solve_monostatic_rcs_bor_survey,
                    args, directory, args["bor_options"], False, frequency_workers=workers)
                # Frequency .7 is deliberately requested twice.
                assert resumed["metadata"]["frequency_checkpoints"]["completed"] == 3
                assert len(resumed["samples"]) == len(result["samples"])
    evidence = outputs[1]["metadata"]["frequency_execution"]
    assert evidence["maximum_active_workers"] == 2
    assert evidence["peak_reserved_gib"] <= evidence["memory_budget_gib"]
    for pol in ("VV", "HH"):
        fields = [np.array([[r["rcs_amp_real"], r["rcs_amp_imag"]]
                           for r in value["co_solved_samples"][pol]]) for value in outputs]
        np.testing.assert_allclose(*fields, rtol=5e-12, atol=2e-14)
    gates = [value["metadata"]["quality_gate"] for value in outputs]
    assert gates[0]["passed"] == gates[1]["passed"]
    assert gates[0]["thresholds"] == gates[1]["thresholds"]
    for key in gates[0]:
        if key != "values":
            assert gates[0][key] == gates[1][key]
    for key in gates[0]["values"]:
        np.testing.assert_allclose(gates[0]["values"][key], gates[1]["values"][key],
                                   rtol=5e-12, atol=2e-14, err_msg=key)


def test_bor_preview_reserves_gib_and_preserves_explicit_modes():
    args = dict(arguments(), n_modes=12)
    with mock.patch.object(dispatch, "estimate_bor_resources", return_value=dict(
            estimated_peak_gb=2., n_unknowns_estimate=500, mode_cap_estimate=12)) as preview:
        rows = fs._plan_bor(args, args["bor_options"], False, [.6], 2, 8., 8)
    assert rows[0]["memory_gib"] == 2e9/1024**3*1.25
    assert rows[0]["cpus"] == 4
    assert preview.call_args.kwargs["workers"] == 2
    assert preview.call_args.kwargs["n_modes"] == 12
    assert preview.call_args.kwargs["frequency_count"] == 3


def test_bor_scheduler_recognizes_only_public_entry_points():
    assert fs._solver_name(dispatch.solve_monostatic_rcs_bor_survey) == "solve_monostatic_rcs_bor_survey"
    def custom(**kwargs):
        return None
    custom.__name__ = "solve_monostatic_rcs_bor_survey"
    assert fs._solver_name(custom) is None
