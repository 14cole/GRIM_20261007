"""Submit-time 2-D resource planning and the runtime-environment check (October 2026).

- The hp certification pair (degrees 2 and 3 on one coarsened snapshot) is priced from one panel and
  linear-mesh build, enriched per degree, with records identical to per-degree planning.
- The geometric near-pair count of a mesh is computed once and shared by the polarizations and degrees.
- Geometries are planned on worker processes with records identical to the serial plan.
- The runtime fingerprint ignores the CPU features NumPy detects at import (a login node and a compute
  node of another generation differ there); a real mismatch names its fields and
  GHOST_RUNTIME_ENVIRONMENT_CHECK=warn downgrades it to a warning.
"""
import copy
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.execution import provenance  # noqa: E402
from ghost_backend.execution.options import automatic_options, execution_scope  # noqa: E402
import ghost_backend.hpc.scheduler as scheduler  # noqa: E402
import ghost_backend.twod.solver as solver  # noqa: E402
from ghost_backend.twod.formulations import regions  # noqa: E402
from ghost_backend.twod.geometry import copy_linear_mesh  # noqa: E402
from test_experimental_cpu import fixture  # noqa: E402

PLANNING = dict(polarizations=['TM', 'TE'], geometry_units='meters', max_panels=10000,
                fine_factor=1.5, n_angles=3, solver_method='experimental_cpu')


def _geometry_file(directory, name='case.geo', scale=1.0, automatic=False):
    """The PEC rectangle fixture as a .geo file; ``automatic`` drops its explicit
    panel count (the hp pair needs an automatic density and, at ``scale`` 10 and
    6 GHz, a reference mesh above the 512-panel floor)."""
    from ghost_backend.geometry.io import Segment, build_geometry_text
    snapshot = fixture('pec', 24)
    segment = snapshot['segments'][0]
    x = [p[k] * scale for p in segment['point_pairs'] for k in ('x1', 'x2')]
    y = [p[k] * scale for p in segment['point_pairs'] for k in ('y1', 'y2')]
    properties = list(segment['properties'])
    if automatic:
        properties[1] = '0'
    path = Path(directory) / name
    path.write_text(build_geometry_text('Test', [Segment('PEC', '2', properties, x, y)], [], []))
    return str(path)


# A fixed compressed storage budget keeps the compressed forecast deterministic
# (the automatic budget follows free memory at the time of the call).
FIXED_STORAGE = dict(compressed_storage_mib=256)


def _per_degree_reference(rcs_solver, snapshot, materials, freq, pols, scale, max_panels, scopes):
    """The former planner: one complete panel/mesh build per degree."""
    out = {}
    for degree, scope in scopes.items():
        with execution_scope(scope):
            out[degree] = scheduler._resource_records_for_frequency(
                rcs_solver, snapshot, materials, freq, pols, scale, max_panels)
    return out


def test_hp_pair_records_equal_per_degree_planning_from_one_mesh_build():
    with tempfile.TemporaryDirectory() as directory:
        path = _geometry_file(directory, scale=10.0, automatic=True)
        with execution_scope(dict(automatic_options(), **FIXED_STORAGE)):
            with mock.patch.object(solver, '_build_linear_mesh_interface_aware',
                                   wraps=solver._build_linear_mesh_interface_aware) as meshes, \
                    mock.patch.object(solver, '_build_panels', wraps=solver._build_panels) as panels, \
                    mock.patch.object(regions, 'geometric_near_pair_count',
                                      wraps=regions.geometric_near_pair_count) as near:
                plan = scheduler.predict_2d_resources_many(path, [6.0], **PLANNING)
            with mock.patch.object(scheduler, '_resource_records_for_degrees', _per_degree_reference):
                reference = scheduler.predict_2d_resources_many(path, [6.0], **PLANNING)
    assert plan == reference
    record = plan[(6.0, 'TM')]
    assert (record['base_polynomial_degree'], record['fine_polynomial_degree']) == (2, 3)   # the hp pair
    assert record['fine_nodes'] > record['nodes']
    # one panel build, one linear mesh (TM and TE share the topology), one near-pair count for both degrees
    assert (panels.call_count, meshes.call_count, near.call_count) == (1, 1, 1)


def test_near_pair_count_is_memoized_and_carried_by_mesh_copies():
    snapshot = fixture('pec', 24)
    materials = solver.MaterialLibrary.from_entries([], [], base_dir='')
    scale = solver._unit_scale_to_meters('meters')
    lambda_min, _, _ = solver._mesh_wavelength_for_snapshot(snapshot, materials, .6)
    panels = solver._build_panels(snapshot, scale, lambda_min, max_panels=10000,
                                  segment_wavelengths=solver.segment_wavelengths(snapshot, materials, [.6], scale, lambda_min),
                                  materials=materials, frequencies_ghz=[.6])
    k0 = 2 * np.pi * .6e9 / solver.C0
    infos = solver._build_coupled_panel_info(panels, materials, .6, 'TM', k0)
    mesh, _ = solver._build_linear_mesh_interface_aware(panels, infos)
    centers = np.asarray([e.center for e in mesh.elements])
    lengths = np.asarray([e.length for e in mesh.elements])
    direct = regions.geometric_near_pair_count(centers, lengths)
    with mock.patch.object(regions, 'geometric_near_pair_count', wraps=regions.geometric_near_pair_count) as counted:
        assert regions.mesh_near_pair_count(mesh) == direct
        assert regions.mesh_near_pair_count(mesh) == direct
        clone = copy_linear_mesh(mesh)
        assert regions.mesh_near_pair_count(clone) == direct
    assert counted.call_count == 1
    assert clone.nodes is not mesh.nodes and clone.elements[0] is not mesh.elements[0]
    assert clone.elements[0].node_ids == mesh.elements[0].node_ids


def test_geometries_plan_on_worker_processes_identically():
    with tempfile.TemporaryDirectory() as directory:
        requests = {_geometry_file(directory, 'case%d.geo' % index): [.6, .8] for index in range(4)}
        with execution_scope(dict(factorization='adaptive', **FIXED_STORAGE)):
            serial = scheduler.predict_2d_resources_for_geometries(requests, workers=1, **PLANNING)
            seen = []
            parallel = scheduler.predict_2d_resources_for_geometries(
                requests, workers=2, progress=lambda done, total, path: seen.append((done, total)), **PLANNING)
    assert list(parallel) == list(requests) and parallel == serial
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert all(len(batch) == 4 for batch in serial.values())


def test_a_broken_planning_pool_falls_back_to_serial_planning(capsys):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    class BrokenPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            future = Future()
            future.set_exception(BrokenProcessPool('worker died'))
            return future

    with tempfile.TemporaryDirectory() as directory:
        requests = {_geometry_file(directory, 'case%d.geo' % index): [.6] for index in range(4)}
        with execution_scope(dict(factorization='adaptive', **FIXED_STORAGE)):
            serial = scheduler.predict_2d_resources_for_geometries(requests, workers=1, **PLANNING)
            seen = []
            with mock.patch('concurrent.futures.ProcessPoolExecutor', BrokenPool):
                recovered = scheduler.predict_2d_resources_for_geometries(
                    requests, workers=2, progress=lambda done, total, path: seen.append(done), **PLANNING)
    assert recovered == serial and seen == [1, 2, 3, 4]
    assert '[warn] planning worker processes failed' in capsys.readouterr().out


def test_planning_worker_count_rules():
    with mock.patch.dict(os.environ, {'GHOST_PLANNING_WORKERS': ''}):
        assert scheduler.planning_worker_count(None, 3) == 1          # below four geometries: serial
        assert scheduler.planning_worker_count(16, 4) == 4            # never more than the geometries
        assert 1 <= scheduler.planning_worker_count(None, 100) <= scheduler.PLANNING_WORKERS_MAX
        assert scheduler.planning_worker_count(1, 100) == 1
    with mock.patch.dict(os.environ, {'GHOST_PLANNING_WORKERS': '3'}):
        assert scheduler.planning_worker_count(None, 100) == 3
        assert scheduler.planning_worker_count(5, 100) == 5           # the driver's setting wins


# ---------------------------------------------------------------- runtime-environment check

def test_cpu_features_detected_at_import_are_informational():
    before = provenance.runtime_environment_fingerprint()
    payload = provenance.runtime_environment_payload()
    assert 'SIMD Extensions' in payload['numpy_config']
    with mock.patch.dict(np.__config__.CONFIG, {'SIMD Extensions': {'baseline': ['X86_V2'], 'found': ['OTHER_CPU']},
                                                'Machine Information': {'host': {'cpu': 'other'}}}):
        changed = provenance.runtime_environment_payload()
        assert changed['numpy_config']['SIMD Extensions'] != payload['numpy_config']['SIMD Extensions']
        assert provenance.runtime_environment_fingerprint() == before
        assert provenance.describe_runtime_mismatch(payload) == ''
        provenance.verify_runtime_environment(before, payload)


def test_a_real_mismatch_names_its_field_and_the_switch_downgrades_it(capsys):
    recorded = copy.deepcopy(provenance.runtime_environment_payload())
    recorded['numpy_version'] = '0.0.0'
    recorded['numpy_config'] = dict(recorded['numpy_config'], **{'Build Dependencies': {'blas': {'name': 'other-blas'}}})
    detail = provenance.describe_runtime_mismatch(recorded)
    assert 'numpy_version: recorded 0.0.0' in detail and 'Build Dependencies' in detail
    with mock.patch.dict(os.environ, {'GHOST_RUNTIME_ENVIRONMENT_CHECK': ''}):
        with pytest.raises(RuntimeError) as failure:
            provenance.verify_runtime_environment('0' * 64, recorded, origin='the HPC run manifest')
        assert 'numpy_version' in str(failure.value) and 'GHOST_RUNTIME_ENVIRONMENT_CHECK=warn' in str(failure.value)
        with pytest.raises(RuntimeError, match='does not record its submission environment'):
            provenance.verify_runtime_environment('0' * 64, None)
    with mock.patch.dict(os.environ, {'GHOST_RUNTIME_ENVIRONMENT_CHECK': 'warn'}):
        provenance._runtime_mismatches_warned.clear()
        provenance.verify_runtime_environment('0' * 64, recorded, origin='the HPC run manifest')
        provenance.verify_runtime_environment('0' * 64, recorded, origin='the HPC run manifest')
    out = capsys.readouterr().out
    assert out.count('[warn]') == 1 and 'numpy_version' in out


def test_drivers_record_the_submission_environment_and_verify_through_the_shared_check():
    import inspect
    from ghost_backend import run_local_monostatic, run_local_bor, run_hpc_monostatic, run_hpc_bor_monostatic
    for driver in (run_local_monostatic, run_local_bor, run_hpc_monostatic, run_hpc_bor_monostatic):
        source = inspect.getsource(driver._verify_run_provenance)
        assert 'verify_runtime_environment(' in source and 'submission_runtime_environment' in source, driver.__name__
        assert 'runtime differs' not in source, driver.__name__
