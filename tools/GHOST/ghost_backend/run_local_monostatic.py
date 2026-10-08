#!/usr/bin/env python3
"""Local monostatic RCS sweep -- run_hpc_monostatic.py without SLURM.

One .grim per (geometry, frequency) unit is written to
<OUTPUT_DIR>/run_YYYYMMDD_HHMMSS/results/{FRD,OPN}/ as soon as that unit
finishes, named "<FREQ:.3f>GHz_<geometry_stem>.grim". Every file contains the
canonical VV and HH channels. The role folders can be passed directly to the
downstream subtraction tool.
each file carries its own source/runtime/input attestation inside the artifact,
so a resumed run verifies what it reuses without a sidecar per output.

Scheduling matches the HPC path. Units are costed from the mesh the solver will
actually build and run dearest-first, and concurrent solves are admitted
against a memory budget rather than filling every core regardless of unit size
-- one 40 GB geometry does not get eight copies of itself started on a 32 GB
laptop.

Edit the CONFIG block and run:

    python run_local_monostatic.py
"""
import platform as _platform
import sys as _sys

# Pool workers inherit this interpreter, so a driver started with the wrong python
# fails inside a worker instead of here. NumPy 2 and SciPy 1.14 (and this package)
# need 3.10; say so before importing the backend.
if _sys.version_info < (3, 10):
    raise SystemExit(
        "GHOST requires Python 3.10 or newer; this is {} at {}.\n"
        "Activate the environment that was used to build the backend, then rerun "
        "(ghost_backend/hpc/check_environment.py reports what a host provides). "
        "Run the driver with that interpreter.".format(
            _platform.python_version(), _sys.executable))
if not __package__:
    import sys
    from pathlib import Path
    if Path(__file__).resolve().parent.name == "ghost_backend":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ghost_backend.execution.paths import backend_root as _backend_root
from ghost_backend.execution.options import (
    blas_thread_reservation,
    AUTOMATIC_LU_PRECISION, AUTOMATIC_SOLVER_METHOD, current_options,
)
from ghost_backend.runs.execution import driver_execution, unit_execution

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ghost_backend.hpc.scheduler as hpc_scheduler
from ghost_backend.runs.quality import accuracy_target_policy
from ghost_backend.execution.provenance import (
    backend_source_fingerprint,
    backend_source_inventory,
    describe_source_mismatch,
    embed_output_attestation,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    runtime_environment_payload,
    verify_runtime_environment,
    stable_json_fingerprint,
    unit_solve_spec_fingerprint,
    verify_embedded_attestation,
)

from ghost_backend.runs.inputs import verify_local_unit_input as _verify_unit_input
from ghost_backend.runs.inputs import load_geometry_snapshot


# ===============================================================================
# CONFIG
# ===============================================================================

# Geometry folders. Every *.geo file under these (recursively) is included.
FRD_DIR = "ghost_backend/geometry/geometries/FRD"
OPN_DIR = "ghost_backend/geometry/geometries/OPN"

# Sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "ghost_backend/results/rcs_runs"

GEOMETRY_UNITS = "inches"         # "inches" | "meters"
MESH_CERTIFICATION = True        # True: compare base/fine meshes. False: one uncertified mesh.
ACCURACY_TARGET = "standard"     # "standard" | "tight"

# Optional resource caps. The backend, mesh, threads and memory admission are
# chosen automatically for each solve.
WORKERS = None                    # max concurrent solves; None = cpu_count() - 1
PLANNING_WORKERS = None           # processes for the mesh/storage planning; None = min(8, CPUs), 1 = serial
MAX_SOLVE_GB = None               # per-solve RAM ceiling in GiB; None = from available RAM

# ===============================================================================

# Internal scheduling constants.
# Fraction of detected RAM the scheduler may reserve for solves; a workstation
# keeps room for its desktop and page cache.
_MEMORY_HEADROOM = 0.75
# Dense: whole-estimate margin. Compressed: sampled operator margin only.
_MEMORY_SAFETY = 1.35
_MAX_PANELS = 100_000
# Pool workers are replaced after this many units so allocator growth from a
# big solve cannot accumulate across a long sweep (a respawn costs about half
# a second of imports and cold hashing, 60 ms per unit at this interval).
_TASKS_PER_CHILD = 8
_GEOMETRY_EXTS = (".geo",)

MANIFEST_SCHEMA = "ghost.local.2d-run.v3"
OUTPUT_POLARIZATIONS = ("VV", "HH")

# Parsed geometry snapshots reused by successive units in each spawned worker.
_SNAPSHOT_CACHE = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]


def _solver_source_records() -> 'Tuple[str, Dict[str, str]]':
    backend_dir = str(_backend_root())
    return backend_dir, {'driver_configured.py': str(Path(__file__).resolve())}


def _solver_source_fingerprint() -> 'str':
    backend_dir, extra = _solver_source_records()
    return backend_source_fingerprint(backend_dir, extra)


def _solver_source_inventory() -> 'Dict[str, str]':
    backend_dir, extra = _solver_source_records()
    return backend_source_inventory(backend_dir, extra)


def _write_json_atomic(path: 'Path', payload: 'Dict[str, Any]') -> 'None':
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _verify_run_provenance(context: 'Dict[str, Any]') -> 'None':
    """Re-check that the solver source and numerical runtime still match the run.

    Called around every unit. The file hashes underneath come from
    `hpc_scheduler.install_fingerprint_cache`, so a repeat check is a stat per
    backend file rather than a full re-read, and the cache expires on a timer
    so full re-reads keep happening inside long-lived workers.
    """

    if _solver_source_fingerprint() != context.get("solver_source_sha256"):
        detail = describe_source_mismatch(
            context.get("solver_source_inventory") or {},
            _solver_source_inventory(),
        )
        raise RuntimeError(
            "Local-run solver source/native artifacts changed; no mixed-state "
            f"field will be written or reused. ({detail})"
        )
    verify_runtime_environment(
        str(context.get("runtime_environment_sha256", "")),
        context.get("submission_runtime_environment"), origin="the local run manifest",
    )


def _unit_attestation_fields(
    context: 'Dict[str, Any]',
    unit: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    return {
        "run_id": str(context["run_id"]),
        "solver_source_sha256": str(context["solver_source_sha256"]),
        "runtime_environment_sha256":
            str(context["runtime_environment_sha256"]),
        "geometry_input_sha256": str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256": str(context["run_solve_spec_sha256"]),
        "unit_solve_spec_sha256": unit_solve_spec_fingerprint(unit),
        "solver_config_sha256": str(context["solver_config_sha256"]),
        "angular_grid_kind": "azimuths_deg",
        # The grid is a run-level property, so it is bound by hash here and
        # stored once in the manifest instead of being repeated in every unit
        # record and every attestation.
        "angular_grid_sha256": str(context["angular_grid_sha256"]),
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _discover_geometries() -> 'List[Path]':
    found: 'List[Path]' = []
    seen: 'set' = set()
    for d in (FRD_DIR, OPN_DIR):
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for ext in _GEOMETRY_EXTS:
            for p in sorted(root.rglob(f"*{ext}")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                found.append(p)
    return found


def _geometry_role(path: 'Path') -> 'str':
    resolved = Path(path).resolve()
    for role, directory in (("FRD", FRD_DIR), ("OPN", OPN_DIR)):
        try:
            resolved.relative_to(Path(directory).resolve())
            return role
        except ValueError:
            continue
    raise ValueError(f"Geometry is outside configured input folders: {path}")


def _unit_name(unit: 'Dict[str, Any]') -> 'str':
    return (f"{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}.grim")


def _unit_output_path(results_dir: 'Path', unit: 'Dict[str, Any]') -> 'Path':
    role = str(unit.get("role", "")).strip().upper()
    folder = results_dir / role if role else results_dir
    return folder / _unit_name(unit)


def _load_snapshot(geometry_path: 'str') -> 'Tuple[Dict[str, Any], str]':
    """Load through the shared reader using this driver's process-local cache."""
    return load_geometry_snapshot(geometry_path, _SNAPSHOT_CACHE)


def _pool_initializer(blas_threads: 'int') -> 'None':
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()
    import ghost_backend.twod.solver as rcs_solver

    rcs_solver.set_assembly_threads(1)


@unit_execution
def _solve_and_export(
    unit: 'Dict[str, Any]',
    context: 'Dict[str, Any]',
    results_dir_str: 'str',
) -> 'Tuple[str, str]':
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""

    results_dir = Path(results_dir_str)
    out_path = _unit_output_path(results_dir, unit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    attestation = _unit_attestation_fields(context, unit)
    if out_path.exists():
        verify_embedded_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    snapshot, material_base = _load_snapshot(str(unit["geometry"]))

    solve_kwargs = dict(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in context["azimuths_deg"]],
        geometry_units=context["geometry_units"],
        material_base_dir=material_base,
        max_panels=context["max_panels"],
        solver_method=context.get("solver_method", "direct"),
    )
    # Select precision inside each worker; context variables are process-local.
    from ghost_backend.linalg.refined_lu import linear_precision
    from ghost_backend.twod.samples import compact_samples
    peak_reset = hpc_scheduler.reset_peak_rss()
    with compact_samples(), linear_precision(context.get("lu_precision", "double")):
        if context["mesh_certification"]:
            from ghost_backend.twod.solver import solve_monostatic_rcs_2d_certified
            result = solve_monostatic_rcs_2d_certified(
                mesh_convergence_policy=context["mesh_convergence_policy"],
                **solve_kwargs
            )
        else:
            from ghost_backend.twod.solver import solve_monostatic_rcs_2d_survey
            result = solve_monostatic_rcs_2d_survey(**solve_kwargs)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    _record_memory_evidence(result, context, unit, peak_reset)

    # Bind the result to its run state inside the artifact, before export, so
    # results/ holds one file per unit instead of a .grim and a sidecar.
    embed_output_attestation(result, attestation)

    from ghost_backend.io.grim import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_local_monostatic.py pols=VV,HH "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    # The run state was verified before the solve and again before the
    # export; a third check after the artifact is published could only
    # report a change it can no longer prevent (October 2026 audit, R-D-1).
    return ("written", actual_path)


def _record_memory_evidence(result, context, unit, peak_reset):
    """Measured peak resident size of this worker against the planner's
    forecast, in the artifact metadata and the run log."""
    measured = hpc_scheduler.peak_rss_gib()
    forecast = context.get("forecast_peak_gib")
    evidence = {
        "forecast_peak_gib": (float(forecast) if forecast is not None else None),
        "measured_peak_gib": measured,
        "scope": ("worker_process_since_unit_start" if peak_reset
                  else "worker_process_lifetime"),
        "note": "Compressed tile processes are children of the worker and are not "
                "included; the forecast covers the whole unit.",
    }
    metadata = result.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["execution_memory"] = evidence
    if measured is not None:
        ratio = ("" if not forecast else f", {measured / float(forecast):.2f} of forecast")
        print(f"      memory {_unit_name(unit)}: forecast "
              f"{(float(forecast) if forecast else 0.0):.2f} GiB, measured peak "
              f"{measured:.2f} GiB ({evidence['scope']}{ratio})", flush=True)


def _solve_and_export_star(args: 'tuple') -> 'tuple':
    """Pool entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so a failure
    names the line it happened on rather than only its message.  The unit's
    CPU reservation bounds its assembly threads, its CPU allocation and its
    BLAS team alike.
    """

    unit, context, results_dir_str, assembly_threads = args
    try:
        import ghost_backend.twod.solver as rcs_solver
        from ghost_backend.execution.thread_control import threadpool_limits
        cpus = max(1, int(assembly_threads))
        rcs_solver.set_assembly_threads(cpus)
        context = dict(context, execution_assembly_threads=cpus)
        with hpc_scheduler.cpu_allocation_scope(cpus), \
                threadpool_limits(limits=cpus, user_api="blas"):
            status, path = _solve_and_export(unit, context, results_dir_str)
        return ("ok", status, path)
    except Exception:
        return ("err", traceback.format_exc(), "")


def _plan(units, fine_factor, n_angles, records_out=None):
    """Share preparation across a geometry's frequencies and polarizations."""
    from ghost_backend.runs.batch import combine_channels
    grouped = {}
    for unit in units:
        grouped.setdefault(str(unit['geometry']), []).append(unit)
    costs, peaks = {}, {}
    batches = hpc_scheduler.predict_2d_resources_for_geometries(
        {geometry: sorted({float(u['frequency_ghz']) for u in group}) for geometry, group in grouped.items()},
        ['TM', 'TE'], GEOMETRY_UNITS, _MAX_PANELS, fine_factor=fine_factor,
        n_angles=n_angles, safety=_MEMORY_SAFETY, solver_method=AUTOMATIC_SOLVER_METHOD,
        workers=PLANNING_WORKERS)
    for geometry, group in grouped.items():
        batch = batches[geometry]
        for unit in group:
            name = _unit_name(unit)
            plans = [batch[(float(unit['frequency_ghz']), pol)] for pol in ('TM', 'TE')]
            record = dict(unit=name, **combine_channels(plans, n_angles, fine_factor))
            costs[name], peaks[name] = record['cost'], record['peak_gb']
            if records_out is not None:
                records_out.append(record)
    return costs, peaks


def _unit_assembly_threads(
    cores: 'int', pool_size: 'int', budget_gb: 'float', peak_gb: 'float'
) -> 'int':
    """Thread count/CPU reservation derived from this unit's own footprint."""

    return hpc_scheduler.assembly_threads_for_unit(
        cores, pool_size, budget_gb, peak_gb, configured=current_options()['assembly_threads']
    )


def _validate_config() -> 'Tuple[List[float], List[float]]':
    if ACCURACY_TARGET not in ("standard", "tight"):
        sys.exit("ERROR: ACCURACY_TARGET must be 'standard' or 'tight'.")
    if type(MESH_CERTIFICATION) is not bool:
        sys.exit("ERROR: MESH_CERTIFICATION must be True or False.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    if (
        not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies}) != len(frequencies)
    ):
        sys.exit(
            "ERROR: frequencies must be finite, positive, unique, and "
            "distinct at the 0.001 GHz output-name precision."
        )
    azimuths = [float(value) for value in AZIMUTHS_DEG]
    if (
        not all(math.isfinite(value) for value in azimuths)
        or len(set(azimuths)) != len(azimuths)
    ):
        sys.exit("ERROR: AZIMUTHS_DEG must be finite and unique.")
    if str(GEOMETRY_UNITS).strip().lower() not in {"inches", "meters"}:
        sys.exit("ERROR: GEOMETRY_UNITS must be 'inches' or 'meters'.")
    if MAX_SOLVE_GB is not None and float(MAX_SOLVE_GB) <= 0.0:
        sys.exit("ERROR: MAX_SOLVE_GB must be positive or None.")
    if WORKERS is not None and int(WORKERS) < 1:
        sys.exit("ERROR: WORKERS must be a positive integer or None.")
    return frequencies, azimuths


@driver_execution
def main() -> 'None':
    frequencies, azimuths = _validate_config()
    if MAX_SOLVE_GB:
        # Read by the solver's own memory gate, in this process and every
        # forked worker.
        os.environ["GHOST_MAX_SOLVE_GB"] = f"{float(MAX_SOLVE_GB):g}"
    blas_threads = blas_thread_reservation()
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()

    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")
    stems = [geometry.stem for geometry in geometries]
    if len(stems) != len(set(stems)):
        sys.exit(
            "ERROR: geometry stems must be unique; output names would "
            "otherwise collide."
        )

    units: 'List[Dict[str, Any]]' = []
    from ghost_backend.assembly.fields import geometry_input_fingerprint
    for geom in geometries:
        input_fingerprint = geometry_input_fingerprint(
            str(geom), GEOMETRY_UNITS
        )
        for f in frequencies:
            # The azimuth grid is deliberately NOT repeated per unit: it is
            # identical for every unit in the run and is bound by hash.
            units.append({
                "geometry":      str(geom.resolve()),
                "geometry_stem": geom.stem,
                "geometry_input_sha256": input_fingerprint,
                "role":          _geometry_role(geom),
                "polarizations": list(OUTPUT_POLARIZATIONS),
                "frequency_ghz": float(f),
            })

    mesh_policy = accuracy_target_policy(ACCURACY_TARGET)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir     = Path(OUTPUT_DIR).resolve() / run_id
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir()
    (results_dir / "FRD").mkdir()
    (results_dir / "OPN").mkdir()
    solver_config = {
        "geometry_units": GEOMETRY_UNITS,
        "linear_solver": "dense_lu",
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "max_panels": _MAX_PANELS,
        "blas_threads_per_worker": blas_threads,
        "mesh_convergence_policy": mesh_policy,
        "accuracy_target": ACCURACY_TARGET,
        "lu_precision": AUTOMATIC_LU_PRECISION,
        "solver_method": AUTOMATIC_SOLVER_METHOD,
        "execution_options": current_options(),
        "mesh_certification": bool(MESH_CERTIFICATION),
    }
    manifest: 'Dict[str, Any]' = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "frequencies_ghz": frequencies,
        "azimuths_deg": azimuths,
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "n_units": len(units),
        "solver_source_sha256": _solver_source_fingerprint(),
        # Per-file hashes behind that fingerprint, so a later mismatch can say
        # which file moved instead of only that one did.
        "solver_source_inventory": _solver_source_inventory(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "submission_runtime_environment": runtime_environment_payload(),
        "solver_config": solver_config,
        "units": units,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    # Resolve shared manifest fields once in the parent process.
    context = {
        "run_id": run_id,
        "solver_source_sha256": manifest["solver_source_sha256"],
        "solver_source_inventory": manifest["solver_source_inventory"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "submission_runtime_environment": manifest.get("submission_runtime_environment"),
        "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
        "solver_config_sha256": stable_json_fingerprint(solver_config),
        "geometry_units": GEOMETRY_UNITS,
        "max_panels": _MAX_PANELS,
        "mesh_convergence_policy": mesh_policy,
        "lu_precision": AUTOMATIC_LU_PRECISION,
        "solver_method": AUTOMATIC_SOLVER_METHOD,
        "execution_options": current_options(),
        "mesh_certification": bool(MESH_CERTIFICATION),
        "azimuths_deg": azimuths,
        "angular_grid_sha256": stable_json_fingerprint(
            azimuths
        ),
    }

    # Cost every unit from the real mesh, then run dearest-first: a frequency
    # sweep's cost spread is large (it grows like the square of the node
    # count), and starting with the cheap end leaves the expensive tail with
    # nothing to overlap against.
    fine_factor = (
        float(mesh_policy["fine_factor"]) if MESH_CERTIFICATION else 1.0
    )
    planning_records = []
    costs, peaks = _plan(units, fine_factor, len(azimuths), planning_records)
    # Sorted into a new list, never in place: `units` is the same object the
    # manifest holds, and reordering it after the run fingerprint was taken
    # would make the manifest on disk hash differently from what every
    # attestation recorded.
    ordered = sorted(
        units, key=lambda u: (-costs.get(_unit_name(u), 1.0), _unit_name(u))
    )

    cores = hpc_scheduler.detect_cores()
    blas_cap = hpc_scheduler.blas_thread_cap(cores)
    if blas_threads > cores:
        raise ValueError("BLAS threads per solve exceed the available CPU allocation ({}).".format(cores))
    # GiB throughout, like the planner's peaks; capped by what the desktop
    # leaves available when the sweep starts.
    budget_gb, memory_gb, available_gb = hpc_scheduler.local_memory_budget_gb(
        _MEMORY_HEADROOM
    )
    worker_cap = max(1, cores - 1) if WORKERS is None else int(WORKERS)
    pool_size = max(1, min(cores, worker_cap, len(ordered)))
    from ghost_backend.runs.batch import apply_batch_choices
    resolved, batch_selections, batch_summary = apply_batch_choices(
        planning_records, cores, pool_size, budget_gb, current_options())
    for record in resolved:
        costs[record['unit']], peaks[record['unit']] = record['cost'], record['peak_gb']
    ordered = sorted(units, key=lambda u: (-costs[_unit_name(u)], _unit_name(u)))
    _write_json_atomic(run_dir / 'schedule.json', dict(units=resolved, batch_selection=batch_summary))
    heaviest = max((peaks.get(_unit_name(u), 0.0) for u in ordered), default=0.0)
    heaviest_concurrency = (
        pool_size if heaviest <= 0.0
        else max(1, min(pool_size, int(budget_gb // heaviest)))
    )
    unit_thread_counts = [
        _unit_assembly_threads(
            cores, pool_size, budget_gb, peaks.get(_unit_name(unit), 0.0)
        )
        for unit in ordered
    ] or [1]
    min_threads = min(unit_thread_counts)
    max_threads = max(unit_thread_counts)
    thread_label = (
        str(min_threads) if min_threads == max_threads
        else f"{min_threads}-{max_threads} dynamic"
    )

    print("=" * 70)
    print("Local monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print("  Polarizations : VV, HH (co-solved)")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(ordered)}  (geometry x frequency)")
    print(f"  Mesh check    : {'base + fine comparison' if MESH_CERTIFICATION else 'base only (no mesh comparison)'}")
    print(f"  Workers       : {pool_size} of {cores} cpus  "
          f"(CPU reservation per solve, assembly and BLAS alike; BLAS team at most {blas_cap}, "
          f"assembly threads/solve: {thread_label})")
    if heaviest_concurrency < pool_size:
        print(f"  Heaviest units: {heaviest_concurrency} concurrent at "
              f"{heaviest:.1f} GB each; smaller units expand dynamically")
    available_text = (f", {available_gb:.1f} GiB available"
                      if available_gb is not None else "")
    print(f"  Memory        : {memory_gb:.1f} GiB installed{available_text}, "
          f"{budget_gb:.1f} GiB schedulable")
    if not MESH_CERTIFICATION:
        print("  Mesh comparison is off: downstream use remains enabled; the")
        print("  user owns the mesh-resolution decision.")
    print("=" * 70, flush=True)

    counters = {"written": 0, "skipped": 0, "failed": 0}
    started = time.time()
    total = len(ordered)

    def _prepare(unit):
        name = _unit_name(unit)
        peak_gb = peaks.get(name, 0.0)
        assembly_threads = reservations.get(name) or _unit_assembly_threads(
            cores, pool_size, budget_gb, peak_gb
        )
        return (
            name, peak_gb,
            (_solve_and_export_star,
             ((unit, dict(context, batch_backend_selection=batch_selections.get(name),
                          forecast_peak_gib=peak_gb),
               str(results_dir), assembly_threads),)),
        )

    def _finished():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(name, payload):
        kind, first, _second = payload
        if kind == "ok":
            counters["skipped" if first == "skipped" else "written"] += 1
            print(f"  [{_finished():4d}/{total}] {first:7s}  {name}", flush=True)
        else:
            counters["failed"] += 1
            print(f"  [{_finished():4d}/{total}] FAILED   {name}", flush=True)
            for line in str(first).rstrip().splitlines():
                print(f"      {line}", flush=True)

    def _on_error(name, exc):
        counters["failed"] += 1
        print(f"  [{_finished():4d}/{total}] FAILED (dispatch) {name}: {exc!r}",
              flush=True)

    # Ordinary spawned workers can run compressed tile processes; unlike a
    # daemon Pool they also report native worker death to the dispatcher.
    # Import after the launch environment has pinned native thread pools.
    from ghost_backend.hpc.common import ExecutorPool
    # One CPU reservation per unit (fill rule or cost-proportional share,
    # whichever is larger); workers inherit the node's BLAS pool size and each
    # unit limits its own BLAS team to its reservation.
    reservations = hpc_scheduler.cpu_reservations(
        [(_unit_name(u), costs.get(_unit_name(u), 1.0), peaks.get(_unit_name(u), 0.0)) for u in ordered],
        cores, pool_size, budget_gb, configured=current_options()["assembly_threads"])
    hpc_scheduler.pin_blas_threads(blas_cap)
    try:
        with ExecutorPool(
            processes=pool_size,
            initializer=_pool_initializer,
            initargs=(blas_cap,),
            max_tasks_per_child=_TASKS_PER_CHILD,
        ) as pool:
            dispatcher = hpc_scheduler.MemoryAwareDispatcher(
                pool, budget_gb=budget_gb, max_concurrent=pool_size,
                cpu_budget=cores,
            )

            def _resources(unit):
                name = _unit_name(unit)
                peak_gb = peaks.get(name, 0.0)
                return (
                    peak_gb,
                    reservations.get(name) or max(blas_threads, _unit_assembly_threads(
                        cores, pool_size, budget_gb, peak_gb
                    )),
                )

            dispatcher.run(
                ordered, _prepare, _on_result, _on_error, _resources
            )
    except BaseException as exc:
        # The manifest must not claim a stopped sweep is still running.
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        _write_json_atomic(manifest_path, manifest)
        raise

    elapsed = time.time() - started
    print(f"\n  Done. wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}.  "
          f"{elapsed:.1f} s elapsed.")
    print(f"  Outputs: {results_dir}/")
    manifest["status"] = "failed" if counters["failed"] else "complete"
    _write_json_atomic(manifest_path, manifest)
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
