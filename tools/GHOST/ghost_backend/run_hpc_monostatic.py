#!/usr/bin/env python3
"""
HPC monostatic RCS sweep driver (SLURM).

Edit the CONFIG block below and run:

    python run_hpc_monostatic.py

Workflow:
- Discover geometry files under FRD_DIR + OPN_DIR.
- Expand into a (geometry x frequency) unit list. Both physical channels and
  all azimuths for a unit are solved in one production call.
- Cost every unit from the mesh the solver will actually build, deal the units
  out longest-processing-time-first, and write that plan beside the manifest.
- Write N_JOBS sbatch job arrays and submit them. Every array task runs the
  same worker: each takes its planned share, then steals whatever is still
  unclaimed, so the tail of a run rebalances itself.
- As each unit finishes, its VV/HH result is exported immediately to
  "<FREQ:.3f>GHz_<geometry_stem>.grim" in
  <run_dir>/results/{FRD,OPN}/.  The role-preserving layout can be passed
  directly to the downstream concatenate/subtract tools.

Scheduling notes -- what makes a big sweep finish sooner:

- COST-AWARE PLANNING. Units are costed at submit time from the realized mesh
  and assigned longest-first.
- WORK STEALING.  Array tasks are interchangeable and coordinate only through
  atomic claim files, so an early-finishing node picks up someone else's
  backlog, a preempted or requeued task loses only its in-flight units, and a
  second submission on another partition can join a sweep already in progress.
- MEMORY-AWARE ADMISSION.  Solves start while the sum of their estimated peaks
  fits the node's memory allocation, instead of filling every core regardless.
  On a 96-core / 750 GB node that runs many small units at once and narrows to
  a few when the expensive ones come up, rather than OOM-killing pool workers
  (a cgroup kill, unlike a Python MemoryError, can wedge the pool).
- CHEAP PER-UNIT PROVENANCE.  The before/after source and input checks stay,
  but repeat hashes come from a stat-keyed cache that expires on a timer, so a
  sweep of thousands of units does not spend its time re-reading the backend
  tree over a shared filesystem.

Restartable: a unit whose .grim already exists is skipped once its attestation
verifies, so cancelling and resubmitting is always safe.

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_monostatic.py --worker <run_dir> <submission_index> <task_index>
"""
import platform as _platform
import sys as _sys

# SLURM array tasks exec the interpreter that submitted them, so a driver started
# with the wrong python fails inside a worker instead of here. NumPy 2 and SciPy
# 1.14 (and this package) need 3.10; say so before importing the backend.
if _sys.version_info < (3, 10):
    raise SystemExit(
        "GHOST requires Python 3.10 or newer; this is {} at {}.\n"
        "Activate the environment that was used to build the backend, then rerun "
        "(ghost_backend/hpc/check_environment.py reports what a host provides). "
        "Set PYTHON_EXE in the CONFIG block, or load modules in JOB_PROLOGUE, to "
        "pin the interpreter a submitted job runs.".format(
            _platform.python_version(), _sys.executable))
if not __package__:
    import sys
    from pathlib import Path
    _driver_directory = Path(__file__).resolve().parent
    _runtime_directory = _driver_directory / "runtime"
    if "--worker" in sys.argv:
        _worker_index = sys.argv.index("--worker")
        if len(sys.argv) > _worker_index + 1:
            _runtime_directory = Path(sys.argv[_worker_index + 1]).resolve() / "runtime"
    if (_runtime_directory / "ghost_backend" / "execution" / "paths.py").is_file():
        sys.path.insert(0, str(_runtime_directory))
    elif _driver_directory.name == "ghost_backend":
        sys.path.insert(0, str(_driver_directory.parent))
from ghost_backend.execution.paths import backend_root as _backend_root
from ghost_backend.execution.options import (
    blas_thread_reservation,
    AUTOMATIC_LU_PRECISION, AUTOMATIC_SOLVER_METHOD, current_options,
)
from ghost_backend.runs.execution import driver_execution, unit_execution

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ghost_backend.hpc.scheduler as hpc_scheduler
from ghost_backend.geometry.io import material_sidecar_paths
from ghost_backend.runs.quality import accuracy_target_policy
from ghost_backend.execution.provenance import (
    backend_source_fingerprint,
    backend_source_inventory,
    describe_source_mismatch,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    runtime_environment_payload,
    stable_json_fingerprint,
    verify_runtime_environment,
    embed_output_attestation,
    unit_solve_spec_fingerprint,
    verify_embedded_attestation,
)

# Compatibility imports retain the established module entrypoints.
from ghost_backend.runs.inputs import (
    publish_submission_journal as _durably_publish_submitted_jobs,
)


# ===============================================================================
# CONFIG -- the only section most users need to edit
# ===============================================================================

# Input geometry folders. Every *.geo file found under these paths
# (recursively) is added to the sweep. Source folder is NOT injected into
# output filenames -- the geometry filename is preserved verbatim.
FRD_DIR = "ghost_backend/geometry/geometries/FRD"
OPN_DIR = "ghost_backend/geometry/geometries/OPN"

# Requested sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

GEOMETRY_UNITS = "inches"         # "inches" | "meters"
MESH_CERTIFICATION = True        # True: compare base/fine meshes. False: one uncertified mesh.
ACCURACY_TARGET = "standard"     # "standard" | "tight"

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "ghost_backend/results/rcs_runs"

# --- How much of the cluster to use ----------------------------------------
# N_NODES is the size of each sbatch job array, N_JOBS the number of separate
# submissions (use more than one to spread across partitions or accounts).
# Total parallelism is N_NODES x N_JOBS nodes.
#
# Unlike the previous round-robin scheme these are pure throughput knobs. Array
# tasks are interchangeable and pull work from a shared claim directory, so
# nothing is stranded if a task never starts, is cancelled, or is preempted,
# and raising N_NODES on a later submission simply adds capacity to a run that
# is already going.
N_NODES = 1
N_JOBS  = 1

# Cap on array tasks running at once (SLURM's `--array=...%N`). None = no cap.
ARRAY_THROTTLE = None

# Worker processes for the submit-time mesh/storage planning (one geometry per
# task; the plan is identical whatever the count). None = GHOST_PLANNING_WORKERS
# or min(8, usable CPUs); 1 = serial. Sweeps below four geometries plan serially.
PLANNING_WORKERS = None

# ===============================================================================
# SLURM allocation and job environment. The backend, mesh, threads and memory
# admission are chosen automatically for each solve.
# ===============================================================================

# --- SLURM resources (per array task = one node) ---------------------------
SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None            # e.g. "my_project"; None to omit
SLURM_QOS       = None
SLURM_TIME      = None            # None = no walltime limit; or "HH:MM:SS"
CORES_PER_NODE  = None            # None = request whole node via --exclusive
                                  # (pool size auto-detected from SLURM env).
                                  # Or set an integer, e.g. 32.
MEM_PER_NODE    = "0"             # "0" = ALL memory of the node (SLURM idiom;
                                  # recommended with --exclusive). None = omit
                                  # the directive -> cluster default applies
                                  # (often DefMemPerCPU ~3.5G x CPUs, which can
                                  # be far less than node RAM). Or e.g. "64G".
MAX_WORKERS_PER_NODE = None       # Hard ceiling on concurrent solves per node.
                                  # None = one per allocated core. The memory
                                  # budget below is usually the binding
                                  # constraint, so this rarely needs setting.
MAX_SOLVE_GB    = None            # Hard ceiling on ONE solve's estimated
                                  # footprint, exported to the job as
                                  # GHOST_MAX_SOLVE_GB. None = derive it from
                                  # current available RAM (0.9 x available,
                                  # with no minimum floor).
                                  #
                                  # Set it when you intend to run something
                                  # very large. Detection needs a number it
                                  # can trust: with MEM_PER_NODE = "0" SLURM
                                  # reports 0, so the limit falls back to the
                                  # cgroup or /proc/meminfo, and if neither is
                                  # meaningful the ceiling collapses to 32 GB
                                  # and a big solve is refused on a big node.
                                  # Either set this, or give MEM_PER_NODE an
                                  # explicit size like "750G".
SLURM_MAIL_TYPE = None            # e.g. "END,FAIL"
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]  # raw extra lines, e.g. "--constraint=intel"

JOB_PROLOGUE = []  # type: List[str]

PYTHON_EXE    = sys.executable           # interpreter used inside the job
SUBMIT        = True                     # False -> write .slurm files but don't sbatch

# ===============================================================================

# Internal scheduling constants.
# Fraction of the node's memory allocation the scheduler may reserve for
# solves; the rest covers the parent process, page cache and estimate error.
_MEMORY_HEADROOM = 0.85
# Dense: whole-estimate margin. Compressed: sampled operator margin only.
_MEMORY_SAFETY = 1.35
_MAX_PANELS = 100_000
# Pool workers are replaced after this many units so allocator growth from a
# big solve cannot accumulate across a long sweep.
_TASKS_PER_CHILD = 8
# A claim whose heartbeat has been quiet this long is treated as abandoned.
_CLAIM_STALE_SECONDS = 3600
_GEOMETRY_EXTS = (".geo",)

_SBATCH = shutil.which("sbatch") or "sbatch"
MANIFEST_SCHEMA = "ghost.hpc.2d-run.v2"
SCHEDULE_SCHEMA = "ghost.hpc.2d-schedule.v2"
OUTPUT_POLARIZATIONS = ("VV", "HH")

# Parsed geometry snapshots reused by successive units in each spawned worker.
_SNAPSHOT_CACHE = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]


# --- shared helpers --------------------------------------------------------

def _solver_source_records():
    # type: () -> Tuple[str, Dict[str, str]]
    """(backend directory, extra logical records) the fingerprint is built from.

    The running driver is recorded under a fixed logical name so that submit
    (which runs the configured copy) and the worker (which runs the copy SLURM
    execs out of the run directory) hash the same bytes under the same name.
    """

    backend_dir = str(_backend_root())
    return backend_dir, {'driver_configured.py': str(Path(__file__).resolve())}


def _solver_source_fingerprint():
    # type: () -> str
    backend_dir, extra = _solver_source_records()
    return backend_source_fingerprint(backend_dir, extra)


def _solver_source_inventory():
    # type: () -> Dict[str, str]
    backend_dir, extra = _solver_source_records()
    return backend_source_inventory(backend_dir, extra)


def _verify_run_provenance(context):
    # type: (Dict[str, Any]) -> None
    """Re-check that the solver source and numerical runtime still match the run.

    Called around every unit, exactly as before. What changed is the cost: the
    file hashes underneath come from `hpc_scheduler.install_fingerprint_cache`,
    so a repeat check is a stat per backend file rather than a full re-read,
    and the cache expires on a timer so full re-reads keep happening inside
    long-lived workers.
    """

    expected_source = str(context.get("solver_source_sha256", ""))
    expected_runtime = str(context.get("runtime_environment_sha256", ""))
    if not expected_source or not expected_runtime:
        raise RuntimeError(
            "HPC run manifest lacks exact solver-source/runtime provenance; "
            "legacy runs must be regenerated before reuse."
        )
    if _solver_source_fingerprint() != expected_source:
        # Name the files. "Something under ghost_backend/ differs" is not actionable,
        # and the usual cause is a tree that was only partly updated.
        detail = describe_source_mismatch(
            context.get("solver_source_inventory") or {},
            _solver_source_inventory(),
        )
        if not context.get("solver_source_inventory"):
            detail = ("this run predates per-file inventories, so the "
                      "differing file cannot be named -- run "
                      "tests/diagnose_provenance.py for what is checked")
        raise RuntimeError(
            "Solver source/native artifacts differ from the HPC run manifest; "
            "no cached or new field will be used from this mixed source state. "
            f"({detail}). Either restore the recorded source or submit a new "
            "run with the code you actually want to execute."
        )
    verify_runtime_environment(
        expected_runtime, context.get("submission_runtime_environment"),
        origin="the HPC run manifest",
    )


def _unit_attestation_fields(context, unit):
    # type: (Dict[str, Any], Dict[str, Any]) -> Dict[str, Any]
    return {
        "run_id": str(context["run_id"]),
        "solver_source_sha256": str(context["solver_source_sha256"]),
        "runtime_environment_sha256": str(context["runtime_environment_sha256"]),
        "geometry_input_sha256": str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256": str(context["run_solve_spec_sha256"]),
        "unit_solve_spec_sha256": unit_solve_spec_fingerprint(unit),
        "solver_config_sha256": str(context["solver_config_sha256"]),
        "angular_grid_kind": "azimuths_deg",
        "angular_grid_sha256": str(context["angular_grid_sha256"]),
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(unit, context):
    # type: (Dict[str, Any], Dict[str, Any]) -> None
    from ghost_backend.assembly.fields import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]), str(context["geometry_units"])
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Frozen geometry/material input changed during the HPC unit: "
            f"{unit['geometry']}"
        )


def _discover_geometries():
    # type: () -> List[Path]
    """Return every geometry file under FRD_DIR/OPN_DIR (deduplicated)."""
    found = []   # type: List[Path]
    seen = set()  # type: set
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


def _geometry_role(path):
    # type: (Path) -> str
    """Return the configured FRD/OPN role for a discovered geometry."""

    resolved = Path(path).resolve()
    for role, directory in (("FRD", FRD_DIR), ("OPN", OPN_DIR)):
        try:
            resolved.relative_to(Path(directory).resolve())
            return role
        except ValueError:
            continue
    raise ValueError(f"Geometry is outside configured input folders: {path}")


def _unit_name(unit):
    # type: (Dict[str, Any]) -> str
    return (f"{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}.grim")


def _unit_output_path(run_dir, unit):
    # type: (Path, Dict[str, Any]) -> Path
    role = str(unit.get("role", "")).strip().upper()
    folder = run_dir / "results" / role if role else run_dir / "results"
    return folder / _unit_name(unit)


def _load_snapshot(geometry_path):
    # type: (str) -> Tuple[Dict[str, Any], str]
    """Parsed snapshot for one geometry, built at most once per process.

    Spawned workers parse a geometry on first use and reuse it for later units.
    """

    cached = _SNAPSHOT_CACHE.get(geometry_path)
    if cached is not None:
        return cached
    from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot

    path = Path(geometry_path)
    title, segments, ibcs, dielectrics = parse_geometry(path.read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = str(path)
    entry = (snapshot, str(path.parent))
    _SNAPSHOT_CACHE[geometry_path] = entry
    return entry


def _pool_initializer(blas_threads):
    # type: (int) -> None
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()
    import ghost_backend.twod.solver as rcs_solver

    rcs_solver.set_assembly_threads(1)


@unit_execution
def _solve_and_export(unit, context, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str) -> Tuple[str, str]
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""

    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
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
        history=(f"run_hpc_monostatic.py pols=VV,HH "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    # The run state was verified before the solve and again before the
    # export; a third check after the artifact is published could only
    # report a change it can no longer prevent (October 2026 audit, R-D-1).
    return ("written", actual_path)


def _record_memory_evidence(result, context, unit, peak_reset):
    """Measured peak resident size of this worker against the planner's
    forecast, in the artifact metadata and the task log: the evidence for
    tightening the 1.35x / 0.85 memory factors on a given cluster."""
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


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    """Pool entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so the SLURM log
    shows where the failure happened, not just the message.  The unit's CPU
    reservation bounds its assembly threads, its CPU allocation and its BLAS
    team alike (the pool worker's BLAS pool was started at the node's physical
    core count so the per-unit limit can grow to the reservation).
    """

    unit, context, run_dir_str, assembly_threads = args
    try:
        import ghost_backend.twod.solver as rcs_solver
        from ghost_backend.execution.thread_control import threadpool_limits
        cpus = max(1, int(assembly_threads))
        rcs_solver.set_assembly_threads(cpus)
        context = dict(context, execution_assembly_threads=cpus)
        with hpc_scheduler.cpu_allocation_scope(cpus), \
                threadpool_limits(limits=cpus, user_api="blas"):
            status, path = _solve_and_export(unit, context, run_dir_str)
        return ("ok", status, path)
    except Exception:
        return ("err", traceback.format_exc(), "")


# --- submit mode (user-invoked) --------------------------------------------


def _plan_schedule(units, n_slots, fine_factor, n_angles):
    # type: (List[Dict[str, Any]], int, float, int) -> Dict[str, Any]
    """Cost every unit, size its memory, and deal the units out to slots.

    Geometry validation/material loading is batched once per geometry.  TM
    and TE receive exact formulation-specific records, then their costs are
    summed and their sequential peak memory is reserved as one output unit.
    """
    from ghost_backend.runs.batch import combine_channels

    resource_cache = {}  # type: Dict[Tuple[str, float, str], Dict[str, Any]]
    grouped = {}  # type: Dict[str, Dict[str, List[Any]]]
    for unit in units:
        geometry = str(unit["geometry"])
        group = grouped.setdefault(geometry, {"frequencies": []})
        frequency = float(unit["frequency_ghz"])
        if frequency not in group["frequencies"]:
            group["frequencies"].append(frequency)

    total = len(grouped)
    records_total = sum(len(group["frequencies"]) * 2 for group in grouped.values())
    workers = hpc_scheduler.planning_worker_count(PLANNING_WORKERS, total)
    started = time.monotonic()
    report_step = max(1, total // 20)
    state = {"next_report": report_step, "last_report": started}
    print(
        f"  Planning mesh/storage bounds for {records_total} channel record(s) across "
        f"{total} geometry file(s) on {workers} process(es); no coefficient sampling...",
        flush=True,
    )

    def planning_progress(completed, _total, _geometry):
        # type: (int, int, str) -> None
        now = time.monotonic()
        if completed < total and completed < state["next_report"] and now - state["last_report"] < 10.0:
            return
        elapsed = max(now - started, 1.0e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0.0 else 0.0
        print(
            f"    planned {completed}/{total} geometries ({100.0 * completed / total:.0f}%) "
            f"in {elapsed:.1f}s, ETA {eta:.1f}s",
            flush=True,
        )
        while state["next_report"] <= completed:
            state["next_report"] += report_step
        state["last_report"] = now

    batches = hpc_scheduler.predict_2d_resources_for_geometries(
        {geometry: list(group["frequencies"]) for geometry, group in grouped.items()},
        ["TM", "TE"],
        GEOMETRY_UNITS,
        _MAX_PANELS,
        fine_factor=fine_factor,
        n_angles=n_angles,
        safety=_MEMORY_SAFETY,
        solver_method=AUTOMATIC_SOLVER_METHOD,
        workers=workers,
        progress=planning_progress,
    )
    for geometry, batch in batches.items():
        for (frequency, polarization), planned in batch.items():
            resource_cache[(geometry, frequency, polarization)] = planned

    records = []     # type: List[Dict[str, Any]]
    for unit in units:
        geometry = str(unit["geometry"])
        frequency = float(unit["frequency_ghz"])
        plans = {
            polarization: resource_cache[(geometry, frequency, polarization)]
            for polarization in ("TM", "TE")
        }
        records.append({
            "unit": _unit_name(unit),
            "base_polynomial_degree": max(int(p.get("base_polynomial_degree", 1)) for p in plans.values()),
            "fine_polynomial_degree": max(int(p.get("fine_polynomial_degree", 1)) for p in plans.values()),
            "nodes": max(int(p["nodes"]) for p in plans.values()),
            "fine_nodes": max(
                int(p["fine_nodes"]) for p in plans.values()
            ),
            "base_system_dofs": max(
                int(p["base_system_dofs"]) for p in plans.values()
            ),
            "system_dofs": max(
                int(p["system_dofs"]) for p in plans.values()
            ),
            "n_regions": max(int(p["n_regions"]) for p in plans.values()),
            "formulations": {
                "HH": str(plans["TM"]["formulation"]),
                "VV": str(plans["TE"]["formulation"]),
            },
        })
        records[-1].update(combine_channels(list(plans.values()), n_angles, fine_factor))
    assignment = hpc_scheduler.balance_units(records, n_slots)
    for record, slot in zip(records, assignment):
        record["slot"] = int(slot)
    elapsed = time.monotonic() - started
    print(f"  Resource plan ready in {elapsed:.1f}s.", flush=True)
    return {
        "schema": SCHEDULE_SCHEMA,
        "n_slots": int(n_slots),
        "fine_factor": float(fine_factor),
        "planning": {
            "method": "batched_exact",
            "storage_forecast": "dimensions_and_payload_ceiling",
            "coefficient_sampling": False,
            "elapsed_seconds": float(elapsed),
            "geometry_preflights": int(len(grouped)),
            "worker_processes": int(workers),
            "frequency_mesh_groups": int(
                sum(len(group["frequencies"]) for group in grouped.values())
            ),
            "unit_records": int(len(records)),
        },
        "units": records,
        "summary": hpc_scheduler.slot_plan_summary(records, assignment, n_slots),
    }


def _validate_config():
    # type: () -> Tuple[List[float], List[float]]
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
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        sys.exit("ERROR: N_NODES and N_JOBS must be >= 1.")
    if ARRAY_THROTTLE is not None and int(ARRAY_THROTTLE) < 1:
        sys.exit("ERROR: ARRAY_THROTTLE must be None or >= 1.")
    if MAX_SOLVE_GB is not None and float(MAX_SOLVE_GB) <= 0.0:
        sys.exit("ERROR: MAX_SOLVE_GB must be positive or None.")
    return frequencies, azimuths


@driver_execution
def submit():
    # type: () -> None
    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    frequencies, azimuths = _validate_config()

    stems = [g.stem for g in geometries]
    if len(stems) != len(set(stems)):
        sys.exit("ERROR: geometry stems must be unique; per-unit result names "
                 "would otherwise overwrite one another.")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "results").mkdir()
    (run_dir / "results" / "FRD").mkdir()
    (run_dir / "results" / "OPN").mkdir()
    (run_dir / "claims").mkdir()

    # Freeze every geometry and its referenced material sidecars before
    # any worker can start.  A later submission may replace its staging bundle;
    # queued/archived runs must remain immutable.
    frozen_geometries = []
    for index, geom in enumerate(geometries):
        inp = run_dir / "inputs" / f"{index:04d}_{geom.stem}"
        inp.mkdir(parents=True, exist_ok=False)
        frozen = inp / geom.name
        shutil.copy2(str(geom), str(frozen))
        for table_name in material_sidecar_paths(str(geom)):
            table = Path(table_name)
            shutil.copy2(str(table), str(inp / table.name))
        frozen_geometries.append((geom, frozen))

    units = []  # type: List[Dict[str, Any]]
    from ghost_backend.assembly.fields import geometry_input_fingerprint
    for original, geom in frozen_geometries:
        input_fingerprint = geometry_input_fingerprint(str(geom), GEOMETRY_UNITS)
        for f in frequencies:
            # The angular grid is identical for every unit and is recorded
            # once at manifest level.
            units.append({
                "geometry":      str(geom.resolve()),
                "geometry_stem": geom.stem,
                "geometry_original": str(original.resolve()),
                "geometry_input_sha256": input_fingerprint,
                "role":          _geometry_role(original),
                "polarizations": list(OUTPUT_POLARIZATIONS),
                "frequency_ghz": float(f),
            })

    mesh_policy = accuracy_target_policy(ACCURACY_TARGET)
    # Workers import their own immutable backend, including native kernels.
    from ghost_backend.hpc.runtime_snapshot import snapshot_backend_runtime
    source_driver = Path(__file__).resolve()
    runtime_parent = snapshot_backend_runtime(_backend_root(), run_dir)
    script_path = run_dir / "driver_configured.py"
    shutil.copy2(str(source_driver), str(script_path))
    source_records = {'driver_configured.py': str(script_path)}
    frozen_backend = str(runtime_parent / "ghost_backend")
    manifest = {
        "schema":          MANIFEST_SCHEMA,
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": frequencies,
        "azimuths_deg":    azimuths,
        "polarizations":   list(OUTPUT_POLARIZATIONS),
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "solver_source_sha256": backend_source_fingerprint(frozen_backend, source_records),
        # Per-file hashes behind that fingerprint, so a later mismatch can say
        # which file moved instead of only that one did.
        "solver_source_inventory": backend_source_inventory(frozen_backend, source_records),
        "runtime_pythonpath": "runtime",
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "submission_runtime_environment": runtime_environment_payload(),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "linear_solver":           "dense_lu",
            "polarizations":           list(OUTPUT_POLARIZATIONS),
            "max_panels":              _MAX_PANELS,
            "blas_threads_per_worker": blas_thread_reservation(),
            "cores_per_node":          CORES_PER_NODE,
            "mesh_convergence_policy": mesh_policy,
            "accuracy_target":         ACCURACY_TARGET,
            "lu_precision":            AUTOMATIC_LU_PRECISION,
            "solver_method":           AUTOMATIC_SOLVER_METHOD,
            "execution_options": current_options(),
            "mesh_certification": bool(MESH_CERTIFICATION),
        },
        "units": units,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # The schedule lives beside the manifest rather than inside it: it says
    # where work should run, not what is solved, and the manifest fingerprint
    # that per-unit attestations bind to must cover only the latter.
    n_slots = int(N_NODES) * int(N_JOBS)
    # fine_factor <= 1 tells the cost model there is only one mesh to solve.
    schedule = _plan_schedule(
        units, n_slots,
        float(mesh_policy["fine_factor"]) if MESH_CERTIFICATION else 1.0,
        len(azimuths),
    )
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))

    slurm_paths = []  # type: List[Path]
    for j in range(int(N_JOBS)):
        sp = run_dir / f"submit_job{j}.slurm"
        sp.write_text(hpc_scheduler.build_sbatch_script(
            job_name=f"rcs_{run_dir.name}_j{j}",
            run_dir=run_dir,
            script_path=script_path,
            array_size=int(N_NODES),
            array_throttle=ARRAY_THROTTLE,
            partition=SLURM_PARTITION,
            cpus_per_node=CORES_PER_NODE,
            mem_per_node=MEM_PER_NODE,
            walltime=SLURM_TIME,
            account=SLURM_ACCOUNT,
            qos=SLURM_QOS,
            mail_type=SLURM_MAIL_TYPE,
            mail_user=SLURM_MAIL_USER,
            extra_sbatch=SLURM_EXTRA_SBATCH,
            # Module setup cannot replace the run's frozen backend.
            prologue=[
                *JOB_PROLOGUE,
                ("export PYTHONPATH="
                 f"{shlex.quote(str(runtime_parent))}"
                 ":${PYTHONPATH:-}"),
            ],
            python_exe=PYTHON_EXE,
            worker_args=(f"--worker {shlex.quote(str(run_dir))} {j} "
                         "${SLURM_ARRAY_TASK_ID}"),
            submission_index=j,
            blas_threads=blas_thread_reservation(),
            extra_env=(
                {"GHOST_MAX_SOLVE_GB": f"{float(MAX_SOLVE_GB):g}"}
                if MAX_SOLVE_GB else {}
            ),
        ))
        sp.chmod(0o755)
        slurm_paths.append(sp)

    peaks = [float(r["peak_gb"]) for r in schedule["units"] if r["peak_gb"] > 0]
    summary = schedule["summary"]
    print("=" * 70)
    print("HPC monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print("  Polarizations : VV, HH (co-solved)")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(units)}  (geometry x frequency)")
    print(f"  Slots         : {N_JOBS} job(s) x {N_NODES} node(s) "
          f"= {n_slots} parallel nodes")
    cores_str = str(CORES_PER_NODE) if CORES_PER_NODE is not None else "auto (--exclusive)"
    mem_str   = str(MEM_PER_NODE) if MEM_PER_NODE else "unlimited"
    time_str  = str(SLURM_TIME) if SLURM_TIME else "unlimited"
    print(f"  Per node      : {cores_str} cores, {mem_str} RAM, "
          f"{time_str} walltime")
    if peaks:
        print(f"  Unit peak RAM : {min(peaks):.2f}-{max(peaks):.2f} GB "
              f"estimated (incl. {_MEMORY_SAFETY:g}x safety)")
    hp_units = sum(int(r.get("fine_polynomial_degree", 1)) > 1 for r in schedule["units"])
    if MESH_CERTIFICATION:
        print(f"  Mesh path     : {hp_units} unit(s) on the hp pair (P2/P3, one coarsened "
              f"mesh), {len(schedule['units']) - hp_units} on the linear pair "
              f"(P1 and P1 x{float(mesh_policy['fine_factor']):g})")
    else:
        print(f"  Mesh path     : {len(schedule['units'])} single linear mesh(es), "
              "uncertified (MESH_CERTIFICATION = False)")
        import ghost_backend.twod.solver as _solver
        ceiling = (
            float(MAX_SOLVE_GB) if MAX_SOLVE_GB
            else _solver._solve_memory_limit_gb()
        )
        source = ("MAX_SOLVE_GB" if MAX_SOLVE_GB
                  else f"{_solver._detect_available_gb():.0f} GB detected here")
        print(f"  Solve ceiling : {ceiling:.1f} GB per solve ({source})")
        if max(peaks) > ceiling:
            print("  [warn] the heaviest planned unit exceeds that ceiling. It "
                  "is evaluated on the COMPUTE node, not here, so this may be "
                  "fine -- but if the node cannot report its own memory the "
                  "ceiling falls back to 32 GB. Set MAX_SOLVE_GB, or give "
                  "MEM_PER_NODE an explicit size, to be sure.")
    idle = int(summary.get("idle_slots", 0))
    print(f"  Plan balance  : {summary['imbalance']:.2f}x the best any schedule "
          f"could do (1.00 = optimal; stealing absorbs the rest)")
    if idle:
        print(f"                  {idle} of {n_slots} slot(s) have no planned "
              "work -- fewer units than nodes, so those tasks exit at once")
    if not MESH_CERTIFICATION:
        print("  Certification : OFF -- base mesh only, typically ~3x faster")
        print("                  Downstream use remains enabled; the user owns the")
        print("                  mesh-resolution decision.")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT:
        print("\n  SUBMIT=False -- submit manually with:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    if shutil.which("sbatch") is None:
        print("\n  [warn] sbatch not on PATH. Submit manually:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    submitted_job_ids = []
    submitted_path = run_dir / "submitted_jobs.json"
    for sp in slurm_paths:
        print(f"\n  Submitting: sbatch {sp.name}")
        res = subprocess.run(
            [_SBATCH, "--parsable", str(sp)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            sys.exit(f"sbatch failed (exit {res.returncode}):\n"
                     f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        job_id = res.stdout.strip().split(";", 1)[0].strip()
        if not job_id.isdigit():
            sys.exit(f"sbatch returned an invalid job ID: {res.stdout!r}")
        submitted_job_ids.append(job_id)
        # Emit immediately as a second recovery channel if journal I/O fails.
        print(f"  Submitted batch job {job_id}", flush=True)
        submitted_document = {
            "schema": "ghost.hpc.submitted-jobs.v1",
            "job_ids": submitted_job_ids,
            "updated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        _durably_publish_submitted_jobs(submitted_path, submitted_document)

    print(f"\nMonitor with:  squeue -u $USER")
    print(f"Outputs in:    {run_dir}/results/")


# --- worker mode (invoked by SLURM) ----------------------------------------

def _read_schedule(run_dir):
    # type: (Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]
    """Return (cost, peak_gb, slot), failing closed without a valid plan."""

    path = run_dir / "schedule.json"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; refusing to run without per-unit memory "
            "reservations. Regenerate the submission."
        )
    try:
        schedule = json.loads(path.read_text())
        records = schedule["units"]
        if not isinstance(records, list) or not records:
            raise ValueError("unit list is empty")
        for record in records:
            if float(record["peak_gb"]) <= 0.0:
                raise ValueError(
                    f"unit {record.get('unit', '<unknown>')} has no positive "
                    "memory reservation"
                )
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unreadable or incomplete {path}; refusing to run without "
            f"per-unit memory reservations: {exc}"
        ) from exc
    costs = {str(r["unit"]): float(r.get("cost", 1.0)) for r in records}
    peaks = {str(r["unit"]): float(r.get("peak_gb", 0.0)) for r in records}
    slots = {str(r["unit"]): int(r.get("slot", 0)) for r in records}
    return costs, peaks, slots


def _planned_names(units, slots, slot, n_slots):
    # type: (List[Dict[str, Any]], Dict[str, int], int, int) -> set
    """Names of the units this slot owns in the submit-time plan.

    The worker refuses a missing or incomplete schedule before reaching here.
    """

    return {
        _unit_name(u) for u in units if slots.get(_unit_name(u), -1) == slot
    }


def _ordered_candidates(units, costs, slots, slot, n_slots):
    # type: (List[Dict[str, Any]], Dict[str, float], Dict[str, int], int, int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """(this slot's planned units, everyone else's), each dearest first.

    Return separate planned and peer lists. Dispatch the peer list only after
    the task finishes its planned share.
    """

    def _key(unit):
        name = _unit_name(unit)
        return (-costs.get(name, 1.0), name)

    mine_names = _planned_names(units, slots, slot, n_slots)
    mine = [u for u in units if _unit_name(u) in mine_names]
    others = [u for u in units if _unit_name(u) not in mine_names]
    return sorted(mine, key=_key), sorted(others, key=_key)


def _unit_assembly_threads(cores, pool_size, budget_gb, peak_gb):
    # type: (int, int, float, float) -> int
    """Thread count/CPU reservation derived from this unit's own footprint."""

    return hpc_scheduler.assembly_threads_for_unit(
        cores, pool_size, budget_gb, peak_gb, configured=current_options()["assembly_threads"]
    )


@driver_execution
def worker(run_dir_str, submission_index, task_index):
    # type: (str, int, int) -> None
    blas_threads = blas_thread_reservation()
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    solver_config = manifest["solver_config"]
    # Resolve shared manifest fields once for the worker context.
    context = {
        "run_id": manifest["run_id"],
        "solver_source_sha256": manifest["solver_source_sha256"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "submission_runtime_environment": manifest.get("submission_runtime_environment"),
        "solver_source_inventory": manifest.get("solver_source_inventory") or {},
        "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
        "solver_config_sha256": stable_json_fingerprint(solver_config),
        "geometry_units": solver_config["geometry_units"],
        "max_panels": solver_config["max_panels"],
        "mesh_convergence_policy": solver_config["mesh_convergence_policy"],
        "lu_precision": solver_config.get("lu_precision", "double"),
        "solver_method": solver_config.get("solver_method", "direct"),
        "execution_options": solver_config.get("execution_options", current_options()),
        "azimuths_deg": list(manifest["azimuths_deg"]),
        "angular_grid_sha256": stable_json_fingerprint(
            [float(value) for value in manifest["azimuths_deg"]]
        ),
        # Older manifests predate the switch and were always certified.
        "mesh_certification": bool(solver_config.get("mesh_certification", True)),
    }
    _verify_run_provenance(context)

    units   = manifest["units"]
    n_nodes = int(manifest.get("n_nodes", 1))
    n_jobs  = int(manifest.get("n_jobs", 1))
    n_slots = max(1, n_nodes * n_jobs)
    slot    = int(submission_index) * n_nodes + int(task_index)

    costs, peaks, slots = _read_schedule(run_dir)
    planned_units, steal_units = _ordered_candidates(
        units, costs, slots, slot, n_slots
    )
    candidates = planned_units + steal_units
    mine = _planned_names(units, slots, slot, n_slots)
    planned = len(planned_units)

    cores = hpc_scheduler.detect_cores()
    if blas_threads > cores:
        raise ValueError("BLAS threads per solve exceed the available CPU allocation ({}).".format(cores))
    blas_cap = hpc_scheduler.blas_thread_cap(cores)
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * _MEMORY_HEADROOM)
    worker_cap = cores if MAX_WORKERS_PER_NODE is None else max(1, int(MAX_WORKERS_PER_NODE))
    # Concurrency is sized from this task's OWN share, not from the whole
    # sweep. Sizing it from the total let one task claim every unit before the
    # others had started; it also left each solve with a sliver of the node
    # when the run was smaller than the cluster.
    pool_size = max(1, min(cores, worker_cap, max(1, planned)))
    batch_selections = {}
    if context['execution_options']['factorization'] == 'adaptive':
        from ghost_backend.runs.batch import apply_batch_choices
        records = json.loads((run_dir / 'schedule.json').read_text())['units']
        if any('backend_candidates' not in r for r in records):
            raise ValueError('Automatic batch selection requires backend forecasts; regenerate the run.')
        for share in (planned_units, steal_units):
            names = {_unit_name(u) for u in share if not _unit_output_path(run_dir, u).is_file()}
            resolved, selections, summary = apply_batch_choices(
                [r for r in records if r['unit'] in names], cores, pool_size,
                budget_gb, context['execution_options'])
            batch_selections.update(selections)
            for record in resolved:
                costs[record['unit']], peaks[record['unit']] = record['cost'], record['peak_gb']
        planned_units, steal_units = _ordered_candidates(units, costs, slots, slot, n_slots)
        candidates = planned_units + steal_units
    heaviest = max(
        (peaks.get(_unit_name(u), 0.0) for u in planned_units), default=0.0
    )
    heaviest_concurrency = (
        pool_size if heaviest <= 0.0
        else max(1, min(pool_size, int(budget_gb // heaviest)))
    )
    # One CPU reservation per unit: the larger of the fill rule and the unit's
    # cost-proportional share of its phase (planned, then stealable).
    reservations = {}
    for share in (planned_units, steal_units):
        reservations.update(hpc_scheduler.cpu_reservations(
            [(_unit_name(u), costs.get(_unit_name(u), 1.0), peaks.get(_unit_name(u), 0.0)) for u in share],
            cores, pool_size, budget_gb, configured=current_options()["assembly_threads"]))
    planned_thread_counts = [reservations[_unit_name(unit)] for unit in planned_units] or [1]
    min_threads = min(planned_thread_counts)
    max_threads = max(planned_thread_counts)
    thread_label = (
        str(min_threads) if min_threads == max_threads
        else f"{min_threads}-{max_threads} dynamic"
    )

    print("=" * 70)
    print(f"  Slot {slot}/{n_slots - 1}  "
          f"(submission={submission_index}, task={task_index})")
    print(f"  Units in run   : {len(units)}   planned for this slot: {planned}"
          f"   (then {len(steal_units)} stealable)")
    print(f"  Cores detected : {cores}   pool size: {pool_size}   "
          f"(CPU reservation per solve, assembly and BLAS alike: "
          f"{thread_label}; BLAS team at most {blas_cap})")
    if heaviest_concurrency < pool_size:
        print(f"  Heaviest units : {heaviest_concurrency} concurrent at "
              f"{heaviest:.1f} GB each; smaller units expand dynamically")
    print(f"  Memory         : {memory_gb:.1f} GB allocated, "
          f"{budget_gb:.1f} GB schedulable")
    print("=" * 70, flush=True)

    if not candidates:
        print("  Nothing to do.")
        return

    # Workers load each geometry into their process-local snapshot cache.
    for unit in candidates:
        path = Path(unit["geometry"])
        if not path.is_file():
            sys.exit(f"Geometry missing on compute node: {path}")

    broker = hpc_scheduler.ClaimBroker(
        run_dir / "claims", stale_seconds=float(_CLAIM_STALE_SECONDS)
    )
    broker.start_heartbeat()

    counters = {"written": 0, "skipped": 0, "failed": 0, "passed": 0}
    peer_units = set()
    started = time.time()
    total = len(candidates)

    def _prepare(unit):
        name = _unit_name(unit)
        peak_gb = peaks.get(name, 0.0)
        assembly_threads = reservations.get(name) or _unit_assembly_threads(
            cores, pool_size, budget_gb, peak_gb
        )
        dispatch = (
            name, peak_gb,
            (_solve_and_export_star,
             ((unit, dict(context, batch_backend_selection=batch_selections.get(name),
                          forecast_peak_gib=peak_gb),
               str(run_dir), assembly_threads),)),
        )
        if _unit_output_path(run_dir, unit).is_file():
            # An already-written result is dispatched, not skipped outright, so
            # its attestation is verified before the run is called complete --
            # that check is what makes reusing an interrupted run safe. Only
            # the slot that owns the unit does it, so across the whole run each
            # output is verified exactly once and no claim is needed (the check
            # is read-only and the result is already final).
            if name not in mine:
                peer_units.add(name)
                counters["passed"] = len(peer_units)
                return None
            return dispatch
        if not broker.try_claim(name):
            peer_units.add(name)
            counters["passed"] = len(peer_units)
            return None
        peer_units.discard(name)
        counters["passed"] = len(peer_units)
        return dispatch

    def _finished():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(name, payload):
        kind, first, _second = payload
        if kind == "ok":
            if first == "skipped":
                counters["skipped"] += 1
            else:
                counters["written"] += 1
            broker.release(name)
            print(f"  [{_finished():4d}/{total}] {first:7s}  {name}", flush=True)
        else:
            counters["failed"] += 1
            # Hand the unit back so another task (or a later resubmission) can
            # retry it; a claim left behind would look busy until it went stale.
            broker.abandon(name)
            print(f"  [{_finished():4d}/{total}] FAILED   {name}", flush=True)
            for line in str(first).rstrip().splitlines():
                print(f"      {line}", flush=True)

    def _on_error(name, exc):
        counters["failed"] += 1
        broker.abandon(name)
        print(f"  [{_finished():4d}/{total}] FAILED (dispatch) {name}: {exc!r}",
              flush=True)

    from ghost_backend.hpc.common import ExecutorPool
    # Workers inherit this BLAS pool size at spawn; each unit then limits its
    # own BLAS team to its reservation (_solve_and_export_star).
    hpc_scheduler.pin_blas_threads(blas_cap)
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
        try:
            # Own share first. Only when it is finished does this task reach
            # for anyone else's, so a fast starter cannot swallow the run.
            dispatcher.run(
                planned_units, _prepare, _on_result, _on_error, _resources
            )
            if steal_units:
                dispatcher.run(
                    steal_units, _prepare, _on_result, _on_error, _resources
                )
            def _output_ready(unit):
                return _unit_output_path(run_dir, unit).is_file()

            def _waiting_for_outputs(missing_count, round_index):
                if round_index == 1 or round_index % 60 == 0:
                    print(
                        f"  Waiting for {missing_count} peer-owned output(s); "
                        "live claims remain fenced and orphan claims will be retried.",
                        flush=True,
                    )

            remaining = dispatcher.run_until_outputs_complete(
                candidates,
                _prepare,
                _on_result,
                _on_error,
                _output_ready,
                _resources,
                should_stop=lambda: counters["failed"] > 0,
                retry_seconds=1.0,
                on_wait=_waiting_for_outputs,
            )
        finally:
            broker.stop_heartbeat()

    elapsed = time.time() - started
    print(f"\n  Slot complete. wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}, "
          f"observed on peer tasks={counters['passed']}.  {elapsed:.1f} s elapsed.")
    if counters["failed"]:
        raise SystemExit(1)
    if remaining:
        raise SystemExit(
            f"ERROR: worker stopped with {len(remaining)} expected output(s) missing."
        )
    from ghost_backend.hpc.common import run_status
    completion = run_status(run_dir)
    if not completion["complete"]:
        integrity = (
            completion["attestation_error"]
            or f"missing={len(completion['missing'])}, "
               f"unexpected={len(completion['unexpected'])}"
        )
        raise SystemExit(
            "ERROR: worker cannot report success because the manifest-exact "
            f"attested result set is incomplete ({integrity})."
        )


# --- entry point -----------------------------------------------------------

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3,
        metavar=("RUN_DIR", "SUBMISSION_INDEX", "TASK_INDEX"),
        help="Internal: join a run as one array task. Invoked by SLURM.",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else:
        submit()


if __name__ == "__main__":
    main()
