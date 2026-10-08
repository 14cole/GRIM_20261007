"""Supported sheets publish complete profiles without implying opaque surfaces."""
import copy
from pathlib import Path
import sys
from unittest import mock
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.assembly.fields import (bodies_from_bor_solver_result, bor_output_profile,
    bor_output_profile_metadata, load_body_grim, load_body_profile_grim,
    outer_generatrix, save_monostatic_grim)
from ghost_backend.bor.dispatch import solve_monostatic_rcs_bor_survey
from ghost_backend.bor.kernels import C0
from ghost_backend.runs.bor_setup import resource_summary


def segment(name, kind, points, impedance=0):
    return dict(name=name, seg_type=kind,
        properties=[str(kind), "1", str(impedance), "1" if kind == 3 else "0", "0"],
        point_pairs=[dict(x1=a[0], y1=a[1], x2=b[0], y2=b[1]) for a,b in zip(points[:-1], points[1:])])


def sphere(mixed=False):
    radius = .8*C0/(2*np.pi*1e9)
    angle = np.linspace(0, np.pi, 19)
    points = radius*np.column_stack((np.sin(angle), np.cos(angle)))
    points[[0,-1],0] = 0.
    segments = ([segment("film",1,points[:10],1), segment("pec",2,points[9:])] if mixed
                else [segment("film",1,points,1)])
    return dict(segments=segments, ibcs=[["1","constant","100","20","0","0"]], dielectrics=[]), points


@pytest.mark.parametrize("mixed", [False, True])
def test_real_sheet_solve_publish_and_load(tmp_path, mixed):
    snapshot, expected = sphere(mixed)
    result = solve_monostatic_rcs_bor_survey(snapshot, [1.], [0.,90.],
        geometry_units="meters", assembly="streaming", workers=1)
    profile = bor_output_profile(snapshot)
    np.testing.assert_allclose(profile, expected)
    saved = save_monostatic_grim(bodies_from_bor_solver_result(result), profile,
        str(tmp_path/"sheet.grim"), azimuths_deg=[0.], elevations_deg=[0.,90.],
        artifact_metadata=bor_output_profile_metadata(snapshot))
    bodies = load_body_grim(saved)
    assert np.all(np.isfinite(bodies[1.]["amp_vv"]))
    np.testing.assert_allclose(load_body_profile_grim(saved), expected)
    with np.load(saved, allow_pickle=False) as payload:
        assert str(payload["body_profile_kind"]) == "transmitting_sheet"
        assert payload["polarizations"].tolist() == ["VV","HH","VH"]
    with pytest.raises(ValueError, match="automatic feature placement"):
        load_body_profile_grim(saved, require_feature_surface=True)
    with pytest.raises(ValueError, match="TYPE 1"):
        outer_generatrix(snapshot)


def test_open_sheet_profile_stays_open():
    points = np.array([[.1,.2],[.1,0.],[.1,-.2]])
    np.testing.assert_allclose(bor_output_profile(dict(segments=[segment("film",1,points,1)])), points)


@pytest.mark.parametrize("kinds", [(2,2),(3,3),(2,3)])
def test_opaque_outer_spans_and_units_preserved(kinds):
    points = np.array([[0.,1.],[1.,0.],[0.,-1.]])
    snapshot = dict(segments=[segment("upper",kinds[0],points[:2]),segment("lower",kinds[1],points[1:])])
    np.testing.assert_array_equal(bor_output_profile(snapshot), points)
    np.testing.assert_allclose(bor_output_profile(snapshot,"inches"), points*.0254)
    assert bor_output_profile_metadata(snapshot) == dict(body_profile_kind="outer_boundary")


def test_disconnected_sheet_rejected_before_planning():
    snapshot, _ = sphere(True)
    snapshot = copy.deepcopy(snapshot)
    for pair in snapshot["segments"][1]["point_pairs"]:
        pair["y1"] -= 1.
        pair["y2"] -= 1.
    setup = dict(schema="grim.bor-run-setup",version=1,frequencies_ghz=[1.],aspects_deg=[0.,90.],
        units="meters",mesh_certification=False,accuracy="standard",cfie_alpha=.5,bor_options={})
    with mock.patch("ghost_backend.bor.dispatch.estimate_bor_resources") as estimate, \
            mock.patch("ghost_backend.bor.dispatch.resolve_automatic_plan") as plan:
        with pytest.raises(ValueError,match="chain|generatrix"):
            resource_summary(snapshot,".",setup)
        estimate.assert_not_called()
        plan.assert_not_called()


@pytest.mark.parametrize("launch", ["local", "hpc"])
@pytest.mark.parametrize("mixed", [False, True])
def test_sheet_driver_end_to_end(tmp_path, launch, mixed):
    import json, os, subprocess
    from ghost_backend.hpc.common import configure_driver, latest_run_dir, run_status
    from ghost_backend.hpc.runtime_snapshot import snapshot_backend_runtime
    original_backend = Path(__file__).resolve().parents[1]
    import_root = snapshot_backend_runtime(original_backend, tmp_path/"checkout")
    backend = import_root/"ghost_backend"
    snapshot, expected = sphere(mixed)
    geometry = tmp_path/"geometry"
    geometry.mkdir()
    rows = ["Title: sheet driver regression"]
    for item in snapshot["segments"]:
        rows += ["Segment: {} {}".format(item["name"],item["seg_type"]),
                 "properties: " + " ".join(item["properties"])]
        rows += ["{:.17g} {:.17g} {:.17g} {:.17g}".format(p["x1"],p["y1"],p["x2"],p["y2"])
                 for p in item["point_pairs"]]
    rows += ["IBCS_Resistances:", "1 constant 100 20 0 0", "Dielectrics:"]
    (geometry/"sheet.geo").write_text("\n".join(rows)+"\n")
    settings = dict(GEOMETRY_DIRS=[str(geometry)], FREQUENCIES_GHZ=[1.],
        AZIMUTHS_DEG=[0.,90.], ELEVATIONS_DEG=[0.], GEOMETRY_UNITS="meters",
        MESH_CERTIFICATION=False, WORKERS_PER_UNIT=1, OUTPUT_DIR=str(tmp_path/"runs"))
    name = "run_local_bor.py" if launch == "local" else "run_hpc_bor_monostatic.py"
    settings.update(dict(WORKERS=1) if launch == "local" else
                    dict(SUBMIT=False, CORES_PER_NODE=1, MAX_WORKERS_PER_NODE=1))
    driver = configure_driver(backend/name, tmp_path/"driver.py", settings)
    env = dict(os.environ, PYTHONPATH=str(import_root), PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    def invoke(path, *args):
        result = subprocess.run([sys.executable,"-B",str(path),*args], cwd=tmp_path,
            env=env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout+result.stderr
    invoke(driver)
    run = latest_run_dir(tmp_path/"runs")
    if launch == "hpc":
        # Editing the original recipe and backend after submission cannot
        # affect the saved run, even with its old checkout on PYTHONPATH.
        config = driver.with_suffix(".config.json")
        payload = json.loads(config.read_text())
        payload["settings"]["FREQUENCIES_GHZ"] = [2.]
        config.write_text(json.dumps(payload))
        solver = backend/"bor/solver.py"
        solver.write_text(solver.read_text()+"\n# later source checkout edit\n")
        invoke(run/"driver_configured.py", "--worker", str(run), "0", "0")
        status = run_status(run)
        assert status["complete"] and status["attestation_verified"], status
        for unit in (run/"results/by_frequency").glob("*.grim"):
            with np.load(unit, allow_pickle=False) as stored:
                metadata = json.loads(str(stored["solver_metadata_json"]))["metadata"]
                plan = metadata["hpc_execution_plan"]
                assert plan["bor_execution_options"]["factorization"] != "auto"
                assert plan["memory_reservation_gib"] > 0
    output = run/"results/sheet.grim"
    assert output.is_file()
    np.testing.assert_allclose(load_body_profile_grim(str(output)), expected)
    with np.load(output, allow_pickle=False) as stored:
        assert str(stored["body_profile_kind"]) == "transmitting_sheet"
        np.testing.assert_array_equal(stored["frequencies"], [1.])
