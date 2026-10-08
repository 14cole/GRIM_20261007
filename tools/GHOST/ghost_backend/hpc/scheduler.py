#!/usr/bin/env python3
"""Shared scheduling core for the HPC sweep drivers."""

import errno
import json
import math
import os
import shutil
import socket
import stat
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - exercised by Windows installations
    fcntl = None

try:
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple


_POLARIZATION_ALIASES = {
    "TM": "TM", "HH": "TM", "H": "TM", "HORIZONTAL": "TM",
    "TE": "TE", "VV": "TE", "V": "TE", "VERTICAL": "TE",
}


def canonical_polarization(label: 'str') -> 'str':
    """Canonical TM/TE channel for a user-facing polarization label."""

    text = str(label or "").strip().upper()
    try:
        return _POLARIZATION_ALIASES[text]
    except KeyError:
        raise ValueError(
            f"Unsupported polarization {label!r}. Use TM/TE or the radar "
            "aliases VV/HH (VV = TE, HH = TM)."
        ) from None


def distinct_polarization_channels(
    labels: 'Sequence[str]',
    canonical: 'Optional[Callable[[str], str]]' = None,
) -> 'List[str]':
    """Validate 2-D resource-planning labels and return one per channel.

    Rejects both an unknown label and two spellings of the *same* channel. That
    second case is the one worth catching: ``["VV", "TE"]`` looks like two
    polarizations and is one, so without this a cost planner could reserve the
    same physical channel twice.

    Labels are returned as written by default so caller-owned record keys stay
    stable. Pass ``canonical`` to return a fixed spelling instead.
    """

    resolved = []  # type: List[str]
    seen = {}  # type: Dict[str, str]
    for label in labels:
        channel = canonical_polarization(label)
        if channel in seen:
            raise ValueError(
                f"{label!r} and {seen[channel]!r} are the same physical "
                f"channel ({channel}); list each channel once."
            )
        seen[channel] = str(label)
        resolved.append(str(label) if canonical is None else canonical(label))
    if not resolved:
        raise ValueError("no polarizations given.")
    return resolved


_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def pin_blas_threads(count: 'int') -> 'None':
    """Pin every BLAS/OpenMP backend to ``count`` threads.

    Must run before numpy/scipy import to take effect on all backends, so the
    drivers call it at module import and again in each pool worker.
    """

    value = str(max(1, int(count)))
    for name in _BLAS_THREAD_VARS:
        os.environ[name] = value


def detect_cores() -> 'int':
    """CPUs usable by this process, respecting both SLURM and affinity."""
    from ghost_backend.execution.options import _usable_logical_cpus
    return _usable_logical_cpus()


# The scheduler's memory unit.  Every budget and every reservation handed to
# MemoryAwareDispatcher is in GiB (2**30 bytes): detect_memory_gb reports GiB
# (SLURM's "MB" are MiB, cgroup and /proc/meminfo are bytes/KiB) and the 2-D
# planner prices in GiB.  The BoR planner prices in decimal GB (1e9 bytes), so
# the BoR drivers convert with decimal_gb_to_gib before admission; comparing
# the two units directly made BoR admission ~7% more conservative than planned.
BYTES_PER_GIB = 1024 ** 3

# Last resort when no probe can see the host's memory.
FALLBACK_MEMORY_GIB = 8.0


def decimal_gb_to_gib(value: 'float') -> 'float':
    """Convert a decimal-GB (1e9-byte) estimate to the scheduler's GiB."""

    return float(value) * 1.0e9 / BYTES_PER_GIB


def _psutil_total_bytes() -> 'Optional[int]':
    try:
        import psutil  # type: ignore
        total = int(psutil.virtual_memory().total)
    except Exception:
        return None
    return total if total > 0 else None


def _windows_memory_status() -> 'Optional[Tuple[int, int]]':
    """(installed, available) physical bytes from GlobalMemoryStatusEx."""

    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:
        return None


def _windows_total_bytes() -> 'Optional[int]':
    """Installed physical memory from GlobalMemoryStatusEx (Windows only)."""

    status = _windows_memory_status()
    return status[0] if status is not None and status[0] > 0 else None


def _sysconf_total_bytes() -> 'Optional[int]':
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None
    return total if total > 0 else None


def _host_memory_bytes() -> 'Optional[int]':
    """Installed memory of this host (psutil, then native probes), or None."""

    for probe in (_psutil_total_bytes, _windows_total_bytes, _sysconf_total_bytes):
        total = probe()
        if total is not None:
            return total
    return None


def detect_memory_gb() -> 'float':
    """Memory this task may use, in GiB (see BYTES_PER_GIB).

    SLURM's allocation is authoritative when present: a node with 750 GB
    installed may still have been given a 64 GB cgroup, and /proc/meminfo
    reports the machine, not the cgroup.  Without SLURM, a cgroup limit or
    /proc/meminfo, the host's installed memory comes from psutil, then
    GlobalMemoryStatusEx on Windows or sysconf elsewhere.  The fixed fallback
    is used only when every probe fails: it used to be what every Windows
    workstation got, so a 32 GB machine scheduled units against ~6 GB.
    """

    raw = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return float(int(raw)) / 1024.0
    raw = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
    if raw.isdigit() and int(raw) > 0:
        # Memory belongs to the full SLURM allocation even when process
        # affinity narrows the CPUs used for computation.
        cpus = detect_cores()
        for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            allocated = os.environ.get(name, "").strip()
            if allocated.isdigit() and int(allocated) > 0:
                cpus = int(allocated)
                break
        return float(int(raw)) * float(cpus) / 1024.0
    for path, scale in (
        ("/sys/fs/cgroup/memory.max", 1.0),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", 1.0),
    ):
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit():
            limit = float(text) * scale / (1024.0 ** 3)

            if 0.5 < limit < 1.0e6:
                return limit
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / (1024.0 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    total = _host_memory_bytes()
    if total is not None:
        return float(total) / BYTES_PER_GIB
    return FALLBACK_MEMORY_GIB


def detect_available_memory_gb() -> 'Optional[float]':
    """Memory the host can give new work right now, in GiB, or None.

    For the workstation drivers: a desktop keeps a large and varying share of
    installed RAM, so their schedulable budget is also capped by what is
    available when the sweep starts.  Cluster workers own their allocation
    and use :func:`detect_memory_gb` alone.
    """

    try:
        import psutil  # type: ignore
        available = int(psutil.virtual_memory().available)
    except Exception:
        available = None
    if available is None:
        status = _windows_memory_status()
        if status is not None:
            available = status[1]
    if available is None:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            available = None
    if available is None or available <= 0:
        return None
    return float(available) / BYTES_PER_GIB


def local_memory_budget_gb(headroom: 'float') -> 'Tuple[float, float, Optional[float]]':
    """(budget, installed, available) GiB for a workstation sweep.

    The budget is ``headroom`` x installed memory, capped by 90% of what is
    available at start (the same fraction a solve's own memory gate allows),
    and never below 1 GiB.
    """

    installed = detect_memory_gb()
    available = detect_available_memory_gb()
    budget = installed * float(headroom)
    if available is not None:
        budget = min(budget, 0.9 * available)
    return max(1.0, budget), installed, available


@contextmanager
def cpu_allocation_scope(cpus: 'Optional[int]'):
    """Bound the solver CPU allocation of one scheduled unit, and nothing else.

    ``ghost_backend.execution.options.allocated_cpu_budget()`` sizes native
    thread teams, near-preparation workers and BLAS limits.  Outside a
    scheduled solve it is the host's CPU count, so concurrent units of one
    sweep each size themselves to the whole machine.  This sets only that
    allocation.  It deliberately does not activate a 2-D execution profile:
    the BoR solvers carry their own options, the runtime fingerprint the
    drivers verify includes the active 2-D profile, and a profile would also
    change how GHOST_* launch overrides are read.  ``None`` leaves the current
    allocation unchanged.
    """

    if cpus is None:
        yield None
        return
    count = max(1, int(cpus))
    import ghost_backend.execution.options as execution_options
    public = getattr(execution_options, "cpu_allocation_scope", None)
    if callable(public):
        with public(count) as allocated:
            yield allocated
        return
    allocation = getattr(execution_options, "_ASSEMBLY_ALLOCATION", None)
    if allocation is None:
        yield count
        return
    with allocation.override(count):
        yield count


def free_disk_gib(path: 'os.PathLike') -> 'Optional[float]':
    """Free space of the filesystem holding ``path`` in GiB, or None."""

    try:
        return float(shutil.disk_usage(str(path)).free) / BYTES_PER_GIB
    except (OSError, ValueError):
        return None


def sweep_stale_bor_spill(log: 'Optional[Callable[[str], None]]' = print) -> 'Any':
    """Remove spill directories left behind by killed BoR solves.

    Streamed BoR solves spill far blocks to ``ghost-bor-far-*`` temporary
    directories that a hard exit (a crashed or killed worker) cannot remove;
    at 10 GHz one solve spills tens of GB.  The sweep itself lives in
    ``ghost_backend.bor.streaming``; it is feature-detected so a backend
    without it still runs, and any failure is reported, never raised.
    """

    try:
        import ghost_backend.bor.streaming as bor_streaming
    except Exception as exc:  # noqa: BLE001 - optional housekeeping
        if log is not None:
            log(f"  [warn] stale spill sweep unavailable: {exc!r}")
        return None
    sweep = getattr(bor_streaming, "remove_stale_spill_directories", None)
    if not callable(sweep):
        return None
    try:
        removed = sweep()
    except Exception as exc:  # noqa: BLE001 - optional housekeeping
        if log is not None:
            log(f"  [warn] stale spill sweep failed: {exc!r}")
        return None
    if log is not None:
        if isinstance(removed, dict):
            # {'removed': directories, 'bytes': ..., 'failed': ...}
            if removed.get("removed") or removed.get("failed"):
                log(f"  Stale BoR spill removed: {int(removed.get('removed') or 0)} "
                    f"director{'y' if removed.get('removed') == 1 else 'ies'}, "
                    f"{float(removed.get('bytes') or 0) / BYTES_PER_GIB:.2f} GiB"
                    + (f"; {int(removed['failed'])} could not be removed"
                       if removed.get("failed") else ""))
        elif removed:
            log(f"  Stale BoR spill removed: {removed!r}")
    return removed


class FingerprintCache:
    """Memoized file hashing for the per-unit provenance checks.

    The drivers verify the solver source and the frozen geometry inputs before
    and after every unit, which is the right check to make -- a run whose
    source changed underneath it must not publish fields.  Recomputing it from
    scratch each time means re-reading several MB of backend sources per unit
    per worker, which on a shared filesystem costs more than the check is
    worth once a sweep has thousands of units.

    Repeat hashes are served from an (inode identity, size, mtime) key, and the
    whole cache is dropped every ``full_recheck_seconds`` so a full re-read
    still happens regularly inside a long-lived worker.  A fresh worker process
    always starts with an empty cache.
    """

    def __init__(self, full_recheck_seconds: 'float' = 300.0) -> 'None':
        self._entries: 'Dict[str, Tuple[Tuple[int, int, int, int], str]]' = {}
        self._full_recheck_seconds = float(full_recheck_seconds)
        self._last_flush = time.monotonic()
        self._lock = threading.Lock()

    def sha256_file(self, path: 'str') -> 'str':
        import hashlib

        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path)
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        now = time.monotonic()
        with self._lock:
            if now - self._last_flush >= self._full_recheck_seconds:
                self._entries.clear()
                self._last_flush = now
            cached = self._entries.get(abs_path)
            if cached is not None and cached[0] == key:
                return cached[1]
        digest = hashlib.sha256()
        with open(abs_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        with self._lock:
            self._entries[abs_path] = (key, value)
        return value


_FINGERPRINT_CACHE: 'Optional[FingerprintCache]' = None


def install_fingerprint_cache(full_recheck_seconds: 'float' = 300.0) -> 'None':
    """Route ``workflow_provenance.sha256_file`` through the cache.

    Called by the drivers in the worker process.  Submission stays uncached so
    the manifest is always written from freshly read bytes.
    """

    global _FINGERPRINT_CACHE
    import ghost_backend.execution.provenance as workflow_provenance

    if _FINGERPRINT_CACHE is None:
        _FINGERPRINT_CACHE = FingerprintCache(full_recheck_seconds)
    workflow_provenance.sha256_file = _FINGERPRINT_CACHE.sha256_file


def _same_interface_topology(
    rcs_solver: 'Any',
    panels: 'Sequence[Any]',
    left: 'Sequence[Any]',
    right: 'Sequence[Any]',
) -> 'bool':
    """Whether two polarization previews produce the same nodal topology.

    Today the signature depends only on geometry/interface flags, not material
    values or polarization. Keeping the comparison explicit makes the reuse
    fail-safe if a future formulation introduces a polarization-dependent
    interface: that channel simply receives its own mesh.
    """

    if len(left) != len(right) or len(left) != len(panels):
        return False
    return all(
        rcs_solver._linear_panel_signature_from_info(panel, l_info)
        == rcs_solver._linear_panel_signature_from_info(panel, r_info)
        for panel, l_info, r_info in zip(panels, left, right)
    )


def _resource_records_for_frequency(
    rcs_solver: 'Any',
    snapshot: 'Dict[str, Any]',
    materials: 'Any',
    frequency_ghz: 'float',
    polarizations: 'Sequence[Tuple[str, str]]',
    unit_scale: 'float',
    max_panels: 'int',
) -> 'Dict[str, Dict[str, Any]]':
    """Build one exact mesh per distinct interface topology at a frequency."""

    freq_ghz = float(frequency_ghz)
    k0 = 2.0 * math.pi * freq_ghz * 1.0e9 / rcs_solver.C0
    lambda_min, _, _ = rcs_solver._mesh_wavelength_for_snapshot(
        snapshot, materials, freq_ghz
    )
    panels = rcs_solver._build_panels(
        snapshot, unit_scale, lambda_min, max_panels=int(max_panels),
        segment_wavelengths=rcs_solver.segment_wavelengths(snapshot, materials, [freq_ghz], unit_scale, lambda_min),
        materials=materials, frequencies_ghz=[freq_ghz],
    )


    topology_groups = []  # type: List[Tuple[List[Any], Any]]
    records = {}  # type: Dict[str, Dict[str, Any]]
    for requested_pol, canonical_pol in polarizations:
        preview = rcs_solver._build_coupled_panel_info(
            panels, materials, freq_ghz, canonical_pol, k0
        )
        mesh = None
        for representative, candidate_mesh in topology_groups:
            if _same_interface_topology(
                rcs_solver, panels, representative, preview
            ):
                mesh = candidate_mesh
                break
        if mesh is None:
            mesh, _stats = rcs_solver._build_linear_mesh_interface_aware(
                panels, preview
            )
            topology_groups.append((preview, mesh))


        infos = preview
        rcs_solver._assert_no_type1_sheet_for_mixed(infos)
        rcs_solver._assert_air_exterior(infos)
        rcs_solver._assert_supported_te_type2_contours(
            mesh, infos, canonical_pol
        )
        resources = rcs_solver._dense_formulation_resources(
            mesh, infos, canonical_pol,
            rcs_solver.layer_for_mesh(mesh, materials, freq_ghz)
            if any(i.bc_kind == 'thin_layer' for i in infos) else None,
            sample_compression=False,
        )
        records[requested_pol] = {
            "panels": int(len(panels)),
            **resources,
        }
    return records


def _resource_records_for_degrees(
    rcs_solver: 'Any',
    snapshot: 'Dict[str, Any]',
    materials: 'Any',
    frequency_ghz: 'float',
    polarizations: 'Sequence[Tuple[str, str]]',
    unit_scale: 'float',
    max_panels: 'int',
    scopes: 'Dict[int, Dict[str, Any]]',
) -> 'Dict[int, Dict[str, Dict[str, Any]]]':
    """``_resource_records_for_frequency`` for several basis degrees of one snapshot.

    ``scopes`` maps each degree to the execution options it is priced under
    (they differ only in ``basis_order``).  Panels, material coefficients and
    the interface-aware linear meshes do not depend on the degree (only
    ``basis.enrich`` reads it), so they are built once, under the first scope
    at degree 1; each degree then enriches its own copy of every topology
    mesh and prices the formulation, which reproduces the single-degree
    records exactly.  The hp certification pair (degrees 2 and 3 on one
    coarsened snapshot) formerly built its panels and meshes twice.
    """
    from ghost_backend.execution.options import execution_scope
    from ghost_backend.twod.basis import enrich
    from ghost_backend.twod.geometry import copy_linear_mesh

    degrees = list(scopes)
    freq_ghz = float(frequency_ghz)
    k0 = 2.0 * math.pi * freq_ghz * 1.0e9 / rcs_solver.C0
    with execution_scope(dict(scopes[degrees[0]], basis_order=1)):
        lambda_min, _, _ = rcs_solver._mesh_wavelength_for_snapshot(
            snapshot, materials, freq_ghz
        )
        panels = rcs_solver._build_panels(
            snapshot, unit_scale, lambda_min, max_panels=int(max_panels),
            segment_wavelengths=rcs_solver.segment_wavelengths(snapshot, materials, [freq_ghz], unit_scale, lambda_min),
            materials=materials, frequencies_ghz=[freq_ghz],
        )
        previews = []  # type: List[Tuple[str, str, Any, int]]
        topology_groups = []  # type: List[Tuple[Any, Any]]
        for requested_pol, canonical_pol in polarizations:
            preview = rcs_solver._build_coupled_panel_info(
                panels, materials, freq_ghz, canonical_pol, k0
            )
            index = next((i for i, (representative, _) in enumerate(topology_groups)
                          if _same_interface_topology(rcs_solver, panels, representative, preview)), None)
            if index is None:
                mesh, _stats = rcs_solver._build_linear_mesh_interface_aware(panels, preview)
                topology_groups.append((preview, mesh))
                index = len(topology_groups) - 1
            previews.append((requested_pol, canonical_pol, preview, index))
    records = {}  # type: Dict[int, Dict[str, Dict[str, Any]]]
    for degree in degrees:
        with execution_scope(scopes[degree]):
            meshes = [enrich(copy_linear_mesh(mesh))[0] for _, mesh in topology_groups]
            records[degree] = {}
            for requested_pol, canonical_pol, infos, index in previews:
                mesh = meshes[index]
                rcs_solver._assert_no_type1_sheet_for_mixed(infos)
                rcs_solver._assert_air_exterior(infos)
                rcs_solver._assert_supported_te_type2_contours(
                    mesh, infos, canonical_pol
                )
                resources = rcs_solver._dense_formulation_resources(
                    mesh, infos, canonical_pol,
                    rcs_solver.layer_for_mesh(mesh, materials, freq_ghz)
                    if any(i.bc_kind == 'thin_layer' for i in infos) else None,
                    sample_compression=False,
                )
                records[degree][requested_pol] = {
                    "panels": int(len(panels)),
                    **resources,
                }
    return records


def predict_2d_resources_many(
    geometry_path: 'str',
    frequencies_ghz: 'Sequence[float]',
    polarizations: 'Sequence[str]',
    geometry_units: 'str',
    max_panels: 'int',
    fine_factor: 'float' = 1.0,
    n_angles: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
    progress: 'Optional[Callable[[float, str], None]]' = None,
    solver_method: 'str' = 'direct',
) -> 'Dict[Tuple[float, str], Dict[str, Any]]':
    """Plan a geometry sweep using the exact solver mesh and formulation.

    Geometry validation, material-table loading, and certification refinement
    are performed once per geometry. Panels are built once per frequency and
    the interface-aware mesh is shared across polarizations only after their
    complete topology signatures compare equal. Material coefficients,
    formulation checks, DOF counts, and memory estimates remain specific to
    every frequency/polarization unit. Errors still propagate so an
    unsupported unit cannot enter a sweep with a zero-GB reservation.
    """

    from ghost_backend.execution.options import current_options, execution_scope
    settings = current_options()
    if solver_method == 'auto':
        solver_method = 'experimental_cpu'
    import ghost_backend.twod.solver as rcs_solver
    from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot
    from ghost_backend.runs.quality import scale_snapshot_panel_density

    frequencies = [float(value) for value in frequencies_ghz]
    if not frequencies:
        raise ValueError("2-D resource planning requires at least one frequency.")
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
    ):
        raise ValueError("Planning frequencies must be finite, positive, and unique.")
    requested_pols = distinct_polarization_channels(
        [str(value) for value in polarizations]
    )
    normalized_pols = [
        (label, rcs_solver._normalize_polarization(label))
        for label in requested_pols
    ]

    path = Path(geometry_path)
    title, segments, ibcs, dielectrics = parse_geometry(path.read_text())
    base_snapshot = build_geometry_snapshot(
        title, segments, ibcs, dielectrics
    )
    base_dir = str(path.parent)
    unit_scale = rcs_solver._unit_scale_to_meters(geometry_units)
    materials = rcs_solver.MaterialLibrary.from_entries(
        base_snapshot.get("ibcs", []) or [],
        base_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    rcs_solver.validate_geometry_snapshot_for_solver(
        base_snapshot,
        base_dir=base_dir,
        meters_scale=unit_scale,
        material_library=materials,
    )

    from ghost_backend.twod.adaptive_geometry import candidate_meshes
    original_snapshot = base_snapshot
    def resource_records(snapshot, degrees, freq):
        """{degree: records} for one snapshot: panels, coefficients and the
        linear topology meshes are built once and enriched per degree."""
        if settings is None:
            shared = _resource_records_for_frequency(rcs_solver,snapshot,materials,freq,normalized_pols,unit_scale,max_panels)
            return {degree: shared for degree in degrees}
        strategy = 'local' if '_2d_hp_coarsening' in snapshot else settings['mesh_strategy']
        scopes = {degree: dict(settings, basis_order=degree, mesh_strategy=strategy) for degree in degrees}
        return _resource_records_for_degrees(rcs_solver,snapshot,materials,freq,normalized_pols,unit_scale,max_panels,scopes)
    planned = {}
    for freq_ghz in frequencies:
        candidates = candidate_meshes(original_snapshot, materials, float(fine_factor),
            settings is not None and settings['mesh_strategy']=='adaptive', [freq_ghz], unit_scale)
        base_snapshot, fine_snapshot = candidates[0][1], candidates[-1][1]
        base_degree, fine_degree = candidates[0][2], candidates[-1][2]
        if settings is not None and settings['mesh_strategy']!='adaptive':
            base_degree = fine_degree = settings['basis_order']
        if len(candidates) == 1:
            base_records = fine_records = resource_records(base_snapshot, [base_degree], freq_ghz)[base_degree]
        elif fine_snapshot is base_snapshot:
            # The hp pair: one snapshot, two degrees, one panel and mesh build.
            shared = resource_records(base_snapshot, sorted({base_degree, fine_degree}), freq_ghz)
            base_records, fine_records = shared[base_degree], shared[fine_degree]
        else:
            base_records = resource_records(base_snapshot, [base_degree], freq_ghz)[base_degree]
            fine_records = resource_records(fine_snapshot, [fine_degree], freq_ghz)[fine_degree]
        for requested_pol in requested_pols:
            base = base_records[requested_pol]
            fine = fine_records[requested_pol]
            if fine["formulation"] != base["formulation"]:
                raise RuntimeError(
                    "base/fine resource planning selected different formulations"
                )
            modes = ('dense','compressed') if settings is not None and settings['factorization'] == 'adaptive' else (None,)
            estimates = {}
            for mode in modes:
                if mode is None:
                    estimates[mode] = _mesh_peak_estimates(rcs_solver, base, fine, n_angles,
                                                         solver_method, safety, floor_gb)
                else:
                    with execution_scope(dict(settings, factorization=mode)):
                        estimates[mode] = _mesh_peak_estimates(rcs_solver, base, fine, n_angles,
                                                             solver_method, safety, floor_gb)
            peak_gb, memory_estimate = max(estimates.values(), key=lambda value: value[0])
            planned[(freq_ghz, requested_pol)] = {
                "nodes": int(base["nodes"]),
                "base_polynomial_degree": base_degree,
                "fine_polynomial_degree": fine_degree,
                "adaptive_refinements_readmitted": bool(settings and settings["mesh_strategy"]=="adaptive"),
                "panels": int(base["panels"]),
                "base_system_dofs": int(base["system_dofs"]),
                "base_operator_matrices": int(base["operator_matrices"]),
                "fine_nodes": int(fine["nodes"]),
                "fine_panels": int(fine["panels"]),
                "fine_system_dofs": int(fine["system_dofs"]),
                "fine_operator_matrices": int(fine["operator_matrices"]),
                "n_regions": int(fine["n_regions"]),
                "formulation": str(fine["formulation"]),

                "system_dofs": int(fine["system_dofs"]),
                "operator_matrices": int(fine["operator_matrices"]),
                "peak_gb": peak_gb,
            }
            if memory_estimate is not None:
                planned[(freq_ghz, requested_pol)]['memory_estimate'] = memory_estimate
            if modes[0] is not None:
                from ghost_backend.execution.policy import relative_cost, MODEL
                planned[(freq_ghz, requested_pol)]['backend_candidates'] = {
                    mode: dict(peak_gb=peak, memory_estimate=estimate, model=MODEL,
                               cost=relative_cost(base,n_angles,mode)+(relative_cost(fine,n_angles,mode) if len(candidates)>1 else 0.))
                    for mode, (peak, estimate) in estimates.items()
                }
            if progress is not None:
                progress(freq_ghz, requested_pol)
    return planned


def _mesh_peak_estimates(solver, base, fine, n_angles, method, safety, floor):
    """Bound both meshes without evaluating numerical coefficient tiles.

    The compressed forecast reserves the configured retained-payload ceiling
    plus structural inverse and workspace bounds. Actual solves still sample
    and enforce their normal memory, storage and numerical-quality gates.
    """
    from ghost_backend.compressed.memory import forecast
    from ghost_backend.compressed.runtime import storage_budget
    from ghost_backend.execution.cpu import configured_batch_size
    from ghost_backend.twod.operators import get_assembly_threads
    results = []
    for original in (base, fine):
        resources = dict(original)
        estimate = solver._estimate_memory_gb(resources['nodes'], False,
            n_regions=max(1, resources['n_regions']), system_dofs=resources['system_dofs'],
            operator_matrices=resources['operator_matrices'], dense_resources=resources,
            n_rhs=max(1, int(n_angles)), solver_method=method, formulation=resources['formulation'])
        memory = None
        if 'memory_estimate' in resources:
            memory = forecast(resources['nodes'], resources['system_dofs'], max(1, int(n_angles)),
                min(configured_batch_size(), max(1, int(n_angles))), get_assembly_threads(),
                storage_budget(), resources, float(safety), float(floor))
        results.append((memory['peak_bytes']/1024**3 if memory else floor+safety*estimate, memory))
    peak, memory = max(results, key=lambda value: value[0])
    if memory is not None:
        memory['certification_mesh_peak_bytes'] = dict(base=int(results[0][0]*1024**3),
                                                     fine=int(results[1][0]*1024**3))
    return peak, memory


def predict_2d_resources(
    geometry_path: 'str',
    frequency_ghz: 'float',
    polarization: 'str',
    geometry_units: 'str',
    max_panels: 'int',
    fine_factor: 'float' = 1.0,
    n_angles: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
    solver_method: 'str' = 'direct',
) -> 'Dict[str, Any]':
    """Predict resources for one 2-D unit."""

    key = (float(frequency_ghz), str(polarization))
    return predict_2d_resources_many(
        geometry_path,
        [key[0]],
        [key[1]],
        geometry_units,
        max_panels,
        fine_factor=fine_factor,
        n_angles=n_angles,
        safety=safety,
        floor_gb=floor_gb,
        solver_method=solver_method,
    )[key]


# Submit-time planning runs one geometry per task on worker processes once a
# sweep has this many geometries; below it the process start-up (the backend
# imports) costs more than it saves.
PLANNING_WORKERS_MAX = 8
PLANNING_SERIAL_BELOW = 4


def planning_worker_count(requested: 'Any' = None, geometries: 'int' = 1) -> 'int':
    """Worker processes for resource planning: ``requested`` (a driver's
    PLANNING_WORKERS), else GHOST_PLANNING_WORKERS, else min(8, usable CPUs);
    never more than the geometries, and 1 below four geometries."""
    value = requested
    if value is None:
        raw = os.environ.get("GHOST_PLANNING_WORKERS", "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = None
    if value is None:
        value = min(PLANNING_WORKERS_MAX, detect_cores())
    geometries = max(1, int(geometries))
    if geometries < PLANNING_SERIAL_BELOW:
        return 1
    return max(1, min(int(value), geometries))


def _predict_batch_task(path, frequencies, options=None, **planning):
    """One geometry's plan under the submitting process's execution options."""
    from ghost_backend.execution.options import execution_scope
    if options is None:
        return predict_2d_resources_many(path, frequencies, **planning)
    with execution_scope(options):
        return predict_2d_resources_many(path, frequencies, **planning)


def predict_2d_resources_for_geometries(
    requests: 'Dict[str, Sequence[float]]',
    polarizations: 'Sequence[str]',
    geometry_units: 'str',
    max_panels: 'int',
    *,
    fine_factor: 'float' = 1.0,
    n_angles: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
    solver_method: 'str' = 'direct',
    workers: 'Any' = None,
    progress: 'Optional[Callable[[int, int, str], None]]' = None,
) -> 'Dict[str, Dict[Tuple[float, str], Dict[str, Any]]]':
    """``predict_2d_resources_many`` for every geometry of a sweep, on worker
    processes when the sweep has enough geometries (``planning_worker_count``).

    ``requests`` maps geometry paths to their frequencies; the result maps each
    path to its batch, in the order of ``requests``.  Each geometry is planned
    exactly as the serial call would be, under the calling process's execution
    options (reapplied in the workers), so the records are identical whatever
    the worker count.  ``progress(done, total, path)`` is called in the calling
    process as each geometry completes.  A failing geometry is named on stdout
    and its error re-raised after the other tasks are cancelled; a pool whose
    workers die (a launcher without an importable main module, a memory kill)
    is abandoned and the remaining geometries are planned in this process.
    """
    from ghost_backend.execution.options import current_options
    paths = list(requests)
    workers = planning_worker_count(workers, len(paths))
    options = current_options()
    planning = dict(polarizations=list(polarizations), geometry_units=geometry_units,
                    max_panels=max_panels, fine_factor=fine_factor, n_angles=n_angles,
                    safety=safety, floor_gb=floor_gb, solver_method=solver_method)
    results = {}  # type: Dict[str, Any]
    if workers <= 1:
        for index, path in enumerate(paths):
            results[path] = _predict_batch_task(path, list(requests[path]), options=options, **planning)
            if progress is not None:
                progress(index + 1, len(paths), path)
        return results
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_predict_batch_task, path, list(requests[path]), options=options, **planning): path
                       for path in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results[path] = future.result()
                except BrokenProcessPool:
                    raise
                except BaseException:
                    for other in futures:
                        other.cancel()
                    print(f"  Resource planning failed for {path}", flush=True)
                    raise
                if progress is not None:
                    progress(len(results), len(paths), path)
    except BrokenProcessPool as exc:
        print(f"  [warn] planning worker processes failed ({exc}); planning the remaining "
              f"{len(paths) - len(results)} geometries in this process", flush=True)
        for path in paths:
            if path in results:
                continue
            results[path] = _predict_batch_task(path, list(requests[path]), options=options, **planning)
            if progress is not None:
                progress(len(results), len(paths), path)
    return {path: results[path] for path in paths}


def unit_cost(
    nodes: 'int',
    n_angles: 'int',
    fine_factor: 'float' = 2.0,
    fine_nodes: 'Optional[int]' = None,
    system_dofs: 'Optional[int]' = None,
    fine_system_dofs: 'Optional[int]' = None,
    operator_matrices: 'int' = 3,
    fine_operator_matrices: 'Optional[int]' = None,
) -> 'float':
    """Relative wall-clock cost of one certified unit.

    Only ratios matter -- this feeds bin packing, not a walltime request.  The
    Terms are the three that actually scale: retained operator assembly
    (operator count times N^2 element pairs), LU factorization (system DOFs
    cubed), and multi-RHS solve/residual work (system DOFs squared per angle).
    This distinction matters for coated bodies whose interface-side system can
    be much larger than the boundary-node count. A certified solve runs the
    base mesh and a refined one, so both are counted.
    """

    angles = max(1.0, float(n_angles))

    def _one(n_nodes: 'float', n_dofs: 'float', n_operators: 'int') -> 'float':
        n = max(1.0, float(n_nodes))
        d = max(1.0, float(n_dofs))


        assembly = n * n * max(1.0, float(n_operators) / 3.0)
        rhs_solve = d * d * angles / 1500.0
        factorization = (d ** 3) / 13000.0
        return assembly + rhs_solve + factorization

    base_dofs = nodes if system_dofs is None else int(system_dofs)
    base = _one(nodes, base_dofs, operator_matrices)
    if float(fine_factor) <= 1.0:

        return base
    refined_nodes = (
        max(1.0, float(nodes) * float(fine_factor))
        if fine_nodes is None else max(1.0, float(fine_nodes))
    )
    refined_dofs = (
        max(1.0, float(base_dofs) * float(fine_factor))
        if fine_system_dofs is None else max(1.0, float(fine_system_dofs))
    )
    refined_operators = (
        int(operator_matrices)
        if fine_operator_matrices is None else int(fine_operator_matrices)
    )
    return base + _one(refined_nodes, refined_dofs, refined_operators)


def unit_peak_gb(
    nodes: 'int',
    fine_factor: 'float' = 2.0,
    n_regions: 'int' = 1,
    safety: 'float' = 1.35,
    floor_gb: 'float' = 0.6,
) -> 'float':
    """Peak resident memory for one certified unit, in GB.

    Built on the solver's own dense-storage estimate for the *fine* mesh (the
    larger of the two solves) so the scheduler and the solver's internal memory
    gate cannot disagree about what a unit costs.  ``floor_gb`` covers the
    interpreter, numpy/scipy, and the forked snapshot; ``safety`` covers
    allocator slack and the transient copies a factorization makes.
    """

    fine_nodes = max(1, int(math.ceil(float(nodes) * float(fine_factor))))
    try:
        import ghost_backend.twod.solver as rcs_solver

        dense_gb = float(
            rcs_solver._estimate_memory_gb(
                fine_nodes, use_cfie=False, n_regions=max(1, int(n_regions))
            )
        )
    except Exception:
        dense_gb = (128.0 * fine_nodes * fine_nodes) / (1024.0 ** 3)
    from ghost_backend.compressed.runtime import enabled as compressed_enabled, storage_budget
    if compressed_enabled():
        from ghost_backend.compressed.memory import forecast
        from ghost_backend.execution.cpu import configured_batch_size
        from ghost_backend.twod.operators import get_assembly_threads
        return forecast(fine_nodes,2*fine_nodes,1000,configured_batch_size(),
            get_assembly_threads(),storage_budget(),safety=float(safety),floor_gb=float(floor_gb))['peak_bytes']/1024**3
    return float(floor_gb) + float(safety) * dense_gb


def assembly_threads_for_unit(
    cores: 'int',
    max_concurrent: 'int',
    budget_gb: 'float',
    peak_gb: 'float',
    configured: 'Any' = "auto",
) -> 'int':
    """Choose a CPU reservation/thread count for one memory-sized solve.

    Homogeneous units fill the node without oversubscription: if only four
    copies of a 70 GB unit fit, each gets one quarter of the cores; if many
    cheap units fit, each gets correspondingly fewer.  The dispatcher reserves
    these CPU counts as well as memory, so mixed heavy/light backfill cannot
    multiply the heavy-unit thread count by every cheap process admitted.
    """

    cores = max(1, int(cores))
    max_concurrent = max(1, int(max_concurrent))
    if str(configured).strip().lower() != "auto":
        return max(1, min(cores, int(configured)))
    peak_gb = max(0.0, float(peak_gb))
    if peak_gb <= 0.0:
        concurrency = max_concurrent
    else:
        concurrency = max(
            1,
            min(max_concurrent, int(max(0.0, float(budget_gb)) // peak_gb)),
        )
    return max(1, cores // concurrency)


# Thread-efficiency margin of the cost-proportional CPU share: a unit that
# would take the balanced per-core work of its share on one thread gets this
# many times the threads that perfect scaling would need.
CPU_RESERVATION_MARGIN = 1.5


def cpu_reservations(units, cores, max_concurrent, budget_gb, configured='auto',
                     margin=CPU_RESERVATION_MARGIN, cap=None):
    """CPU reservation (assembly and BLAS threads) of every unit of one task's share.

    ``units`` are ``(name, cost, peak_gb)`` records.  Each unit gets the larger
    of the fill rule (``assembly_threads_for_unit``: the cores divided among as
    many copies of the unit as memory and the pool admit, which alone left a
    15 GHz unit on two threads while a 96-core node idled) and its
    cost-proportional share, ``ceil(margin * cost / total * cores)``, so that
    the heaviest units of a long sweep finish within the balanced per-core
    work of the share.  ``cap`` (default: the host's physical cores, where
    BLAS stops scaling) bounds both.  An explicit ``configured`` thread count
    wins for every unit, as before.  Returns ``{name: cpus}``.
    """
    from ghost_backend.execution.options import physical_core_count
    units = list(units)
    cores = max(1, int(cores))
    cap = max(1, min(cores, int(cap) if cap else physical_core_count()))
    total = sum(max(0.0, float(cost)) for _, cost, _ in units)
    explicit = str(configured).strip().lower() != "auto"
    reservations = {}
    for name, cost, peak in units:
        fill = assembly_threads_for_unit(cores, max_concurrent, budget_gb, peak, configured)
        if explicit:
            reservations[name] = fill
            continue
        share = (int(math.ceil(float(margin) * max(0.0, float(cost)) / total * cores))
                 if total > 0.0 else 1)
        reservations[name] = max(1, min(cap, max(fill, share)))
    return reservations


def blas_thread_cap(cores=None) -> 'int':
    """Largest BLAS team a unit may be granted: the physical cores of this host,
    bounded by the CPU allocation.  Pool workers are started with this as their
    BLAS pool size so a unit's per-unit limit can grow to its reservation."""
    from ghost_backend.execution.options import physical_core_count
    cores = detect_cores() if cores is None else max(1, int(cores))
    return max(1, min(cores, physical_core_count()))


def reset_peak_rss() -> 'bool':
    """Reset this process's peak resident-size counter (Linux); False elsewhere,
    where ``peak_rss_gib`` then reports the process lifetime peak."""
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def peak_rss_gib() -> 'Optional[float]':
    """Peak resident size of this process in GiB (since the last reset on
    Linux), or None when no probe is available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) * 1024.0 / BYTES_PER_GIB
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # type: ignore
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return float(peak) / BYTES_PER_GIB
    except Exception:
        pass
    try:
        import resource  # type: ignore
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak > 0:
            scale = 1.0 if sys.platform == "darwin" else 1024.0
            return peak * scale / BYTES_PER_GIB
    except Exception:
        pass
    return None


def predict_bor_extent(
    geometry_path: 'str',
    geometry_units: 'str',
) -> 'Tuple[float, float]':
    """(generatrix arc length, maximum radius) of a BoR geometry, in meters.

    Read straight off the .geo point pairs (x = rho, y = z), so this costs a
    file parse rather than a mesh build.  Returns (0, 0) when the geometry
    cannot be read, which the caller treats as "unknown".
    """

    try:
        import ghost_backend.twod.solver as rcs_solver
        from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot

        scale = rcs_solver._unit_scale_to_meters(geometry_units)
        title, segments, ibcs, dielectrics = parse_geometry(
            Path(geometry_path).read_text()
        )
        snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        arc = 0.0
        radius = 0.0
        for segment in snapshot.get("segments", []) or []:
            for pair in segment.get("point_pairs", []) or []:
                x1 = float(pair.get("x1", 0.0)) * scale
                y1 = float(pair.get("y1", 0.0)) * scale
                x2 = float(pair.get("x2", 0.0)) * scale
                y2 = float(pair.get("y2", 0.0)) * scale
                arc += math.hypot(x2 - x1, y2 - y1)
                radius = max(radius, abs(x1), abs(x2))
        return float(arc), float(radius)
    except Exception:
        return 0.0, 0.0


def bor_unit_cost(
    arc_length_m: 'float',
    radius_m: 'float',
    frequency_ghz: 'float',
    n_aspects: 'int',
) -> 'float':
    """Relative wall-clock cost of one body-of-revolution unit.

    A BoR solve builds its far blocks for every element pair and mode and
    factors one dense system per azimuthal mode.  Generatrix elements scale
    with arc length in wavelengths and the kept modes with the circumference
    in wavelengths.  The far build dominates in practice, so cost is modelled
    as (elements^2) x (modes): the 120 x 5 in ogive took 5.7x longer at 10 GHz
    than at 4 GHz (2,160 against 960 elements, 27 against 19 modes), which
    this model puts at 7.2x and the former (elements^3) x (modes) at 16x.
    Only the ratio matters here; it feeds bin packing, not a walltime request.

    Deliberately coarser than the 2-D model, which builds the real mesh: there
    is no equally cheap way to predict a BoR discretization exactly, and an
    approximate plan plus run-time stealing beats an exact plan that costs a
    solve to compute.  A geometry that cannot be read falls back to unit cost.
    """

    if arc_length_m <= 0.0 or frequency_ghz <= 0.0:
        return 1.0
    wavelength = 299_792_458.0 / (float(frequency_ghz) * 1e9)
    elements = max(1.0, 20.0 * float(arc_length_m) / wavelength)
    modes = max(1.0, 2.0 * math.pi * max(radius_m, wavelength / 20.0) / wavelength + 6.0)
    return (elements ** 2) * modes * (1.0 + max(1, int(n_aspects)) / 500.0)


def balance_units(
    units: 'Sequence[Dict[str, Any]]',
    n_slots: 'int',
    cost_key: 'str' = "cost",
) -> 'List[int]':
    """Assign each unit a slot, longest-processing-time-first.

    Returns a list of slot indices parallel to ``units``.  LPT is the standard
    greedy makespan heuristic (never worse than 4/3 of optimal), and on a
    frequency sweep it is dramatically better than round-robin because it puts
    the handful of expensive high-frequency units on different slots first and
    fills the gaps with cheap ones.
    """

    slots = max(1, int(n_slots))
    order = sorted(
        range(len(units)),
        key=lambda i: (-float(units[i].get(cost_key, 1.0)), i),
    )
    loads = [0.0] * slots
    assignment = [0] * len(units)
    for index in order:
        target = min(range(slots), key=lambda s: (loads[s], s))
        assignment[index] = target
        loads[target] += float(units[index].get(cost_key, 1.0))
    return assignment


def slot_plan_summary(
    units: 'Sequence[Dict[str, Any]]',
    assignment: 'Sequence[int]',
    n_slots: 'int',
) -> 'Dict[str, Any]':
    """Predicted per-slot load, for the submit-time report.

    ``imbalance`` is the plan's makespan against the best any schedule could
    do, not against the mean load.  The mean is the wrong yardstick whenever a
    single unit costs more than an even share -- with 6 units across 50 slots,
    or with one dominant high-frequency unit, a perfect plan still shows a
    large max/mean ratio, and reporting that as imbalance would send you
    tuning a scheduler that is already optimal.  Against the lower bound
    max(total/slots, dearest unit), 1.00 means "nothing left to gain".
    """

    slots = max(1, int(n_slots))
    loads = [0.0] * slots
    counts = [0] * slots
    for unit, slot in zip(units, assignment):
        loads[slot] += float(unit.get("cost", 1.0))
        counts[slot] += 1
    total = sum(loads)
    dearest = max((float(u.get("cost", 1.0)) for u in units), default=0.0)
    lower_bound = max(total / slots, dearest)
    makespan = max(loads) if loads else 0.0
    return {
        "slot_units": counts,
        "idle_slots": sum(1 for count in counts if count == 0),
        "makespan": makespan,
        "lower_bound": lower_bound,
        "imbalance": (makespan / lower_bound) if lower_bound > 0 else 1.0,


        "max_over_mean": (makespan / (total / slots)) if total > 0 else 1.0,
    }


class ClaimBroker:
    """Fenced cross-node unit claiming, with steal-on-stale recovery.

    A stable advisory operation lock serializes claim creation/replacement;
    owner tokens and generations fence former holders.  The shared run
    filesystem must provide cluster-coherent advisory locks and atomic
    same-directory replacement.  No scheduler process or message passing is
    needed, and array tasks can join or die freely.

    Claims carry a heartbeat (the file's mtime, refreshed by
    :meth:`start_heartbeat`).  A claim whose heartbeat has gone quiet for
    ``stale_seconds`` belongs to a task that was killed or preempted, and any
    other task may take it over.  The exact Slurm requeue successor may use the
    scheduler restart generation to recover its own fresh orphan immediately;
    unrelated live peers remain fenced.
    """

    def __init__(
        self,
        claims_dir: 'os.PathLike',
        stale_seconds: 'float' = 3600.0,
        heartbeat_seconds: 'float' = 60.0,
    ) -> 'None':
        self.dir = Path(claims_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stale_seconds = float(stale_seconds)
        self.heartbeat_seconds = float(heartbeat_seconds)
        if not math.isfinite(self.stale_seconds) or self.stale_seconds <= 0.0:
            raise ValueError("stale_seconds must be positive and finite.")
        if not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds <= 0.0:
            raise ValueError("heartbeat_seconds must be positive and finite.")
        self.owner_token = uuid.uuid4().hex
        self._slurm_job_id = str(os.environ.get("SLURM_JOB_ID", "")).strip()
        self._slurm_array_job_id = str(
            os.environ.get("SLURM_ARRAY_JOB_ID", "")
        ).strip()
        self._slurm_array_task_id = str(
            os.environ.get("SLURM_ARRAY_TASK_ID", "")
        ).strip()
        self._slurm_cluster_name = str(
            os.environ.get("SLURM_CLUSTER_NAME", "")
        ).strip()
        try:
            restart_count = int(str(
                os.environ.get("SLURM_RESTART_COUNT", "0")
            ).strip())
        except (TypeError, ValueError, OverflowError):
            restart_count = -1
        self._slurm_restart_count = restart_count
        self._held: 'Dict[str, Tuple[Path, str, int]]' = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: 'Optional[threading.Thread]' = None

    def _path(self, key: 'str') -> 'Path':
        value = str(key)
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError("claim key must be one nonempty filename component.")
        return self.dir / f"{value}.claim"

    def _operation_path(self, key: 'str') -> 'Path':
        self._path(key)
        return self.dir / f".{key}.claim-operation"

    @staticmethod
    def _try_lock_file(path: 'Path', *, blocking: 'bool' = False) -> 'Optional[int]':
        """Open and exclusively lock a stable one-byte coordination file.

        Kernel-owned locks are released when a process exits and, unlike an
        mtime-expired ``O_EXCL`` sentinel, cannot be stolen from a process that
        is merely paused.  That property is the fencing guarantee needed
        around claim replacement.
        """

        if path.is_symlink():
            raise RuntimeError(f"Claim operation path is a symbolic link: {path}")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(str(path), flags, 0o600)
        except OSError as exc:
            if exc.errno in {getattr(errno, "ELOOP", -1), getattr(errno, "EMLINK", -1)}:
                raise RuntimeError(f"Claim operation path is a symbolic link: {path}") from exc
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RuntimeError(f"Claim operation path is not a regular file: {path}")
            if fcntl is not None:
                operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(fd, operation)
                except OSError as exc:
                    if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                        os.close(fd)
                        return None
                    raise
            elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                os.lseek(fd, 0, os.SEEK_SET)
                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                try:
                    msvcrt.locking(fd, mode, 1)
                except OSError as exc:
                    if not blocking and exc.errno in {
                        errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLK", -1)
                    }:
                        os.close(fd)
                        return None
                    raise


                if os.fstat(fd).st_size == 0:
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.write(fd, b"\0")
                    os.fsync(fd)
            else:  # pragma: no cover - every supported platform has one API
                raise RuntimeError("No supported advisory file-lock API is available.")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _unlock_file(fd: 'int') -> 'None':
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows-only branch
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)

    @staticmethod
    def _write_fd(fd: 'int', payload: 'bytes') -> 'None':
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _read_document(path: 'Path') -> 'Optional[Dict[str, Any]]':
        try:
            if path.is_symlink() or path.stat().st_size > 64 * 1024:
                return None
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return document if isinstance(document, dict) else None

    @staticmethod
    def _identity(document: 'Optional[Dict[str, Any]]') -> 'Tuple[str, int]':
        if not document:
            return "", 0
        token = str(document.get("owner_token") or "")
        generation = document.get("generation", 0)
        if isinstance(generation, bool):
            generation = 0
        try:
            generation = int(generation)
        except (TypeError, ValueError, OverflowError):
            generation = 0
        return token, max(generation, 0)

    def _payload(self, generation: 'int') -> 'bytes':
        return json.dumps({
            "schema": "ghost.hpc.claim.v2",
            "owner_token": self.owner_token,
            "generation": int(generation),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": self._slurm_job_id,
            "array_job": self._slurm_array_job_id,
            "array_task": self._slurm_array_task_id,
            "cluster": self._slurm_cluster_name,
            "restart_count": max(0, self._slurm_restart_count),
            "claimed_at": time.time(),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _is_requeued_successor(
        self, document: 'Optional[Dict[str, Any]]'
    ) -> 'bool':
        """Return whether this process is Slurm's fenced restart of the owner.

        A fresh claim normally belongs to a live peer and must never be stolen.
        Slurm requeues are the one safe exception: the scheduler does not start
        restart N+1 for an array task until restart N has ended, and exposes the
        monotonically increasing ``SLURM_RESTART_COUNT`` as a fencing value.
        Requiring the exact job/task/cluster identity plus a strictly newer
        restart count prevents an ordinary peer (or two local processes with
        blank Slurm variables) from using this fast takeover path.
        """

        if not document or not self._slurm_job_id or self._slurm_restart_count <= 0:
            return False
        try:
            prior_restart = int(document.get("restart_count", 0))
        except (TypeError, ValueError, OverflowError):
            return False
        if prior_restart < 0 or self._slurm_restart_count <= prior_restart:
            return False
        return (
            str(document.get("job", "")).strip() == self._slurm_job_id
            and str(document.get("array_job", "")).strip()
            == self._slurm_array_job_id
            and str(document.get("array_task", "")).strip()
            == self._slurm_array_task_id
            and str(document.get("cluster", "")).strip()
            == self._slurm_cluster_name
        )

    def _matches(self, path: 'Path', token: 'str', generation: 'int') -> 'bool':
        return self._identity(self._read_document(path)) == (token, generation)

    def _acquire_operation(
        self, key: 'str', *, blocking: 'bool' = False
    ) -> 'Optional[Tuple[int, str]]':
        """Serialize one short claim mutation.

        The operation file is not the long-lived work claim.  It closes the
        read/replace window between competing stale takers and between takeover
        and an old owner's heartbeat/abandon.  The kernel releases this lock
        on process exit; it is never stolen from a merely paused owner.
        """

        path = self._operation_path(key)
        token = uuid.uuid4().hex
        fd = self._try_lock_file(path, blocking=blocking)
        if fd is None:
            return None
        payload = json.dumps({
            "schema": "ghost.hpc.claim-operation.v2",
            "owner_token": token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": time.time(),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            os.fsync(fd)
            return fd, token
        except BaseException:
            self._unlock_file(fd)
            raise

    def _release_operation(self, operation: 'Optional[Tuple[int, str]]') -> 'None':
        if operation is None:
            return
        fd, _token = operation
        try:
            self._unlock_file(fd)
        except OSError:
            pass

    def try_claim(self, key: 'str') -> 'bool':
        """Take ``key`` if it is unclaimed or its holder has gone quiet."""

        path = self._path(key)
        operation = self._acquire_operation(key)
        if operation is None:
            return False
        temporary: 'Optional[Path]' = None
        try:
            existing = self._read_document(path)
            if (
                path.exists()
                and not self._is_stale(path)
                and not self._is_requeued_successor(existing)
            ):
                return False
            _old_token, old_generation = self._identity(existing)
            generation = old_generation + 1 if path.exists() else 1
            payload = self._payload(generation)
            if path.exists():
                temporary = self.dir / f".claim-write.{uuid.uuid4().hex}.tmp"
                fd = os.open(
                    str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
                self._write_fd(fd, payload)
                os.replace(str(temporary), str(path))
                temporary = None
            else:
                try:
                    fd = os.open(
                        str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                    )
                except OSError as exc:
                    if exc.errno == errno.EEXIST:
                        return False
                    raise
                self._write_fd(fd, payload)
            if not self._matches(path, self.owner_token, generation):
                return False
            held = (path, self.owner_token, generation)
            with self._lock:
                self._held[key] = held
            return True
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            self._release_operation(operation)

    def _is_stale(self, path: 'Path') -> 'bool':
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return True
        return age > self.stale_seconds

    def release(self, key: 'str') -> 'None':
        """Drop the heartbeat for a finished unit (the claim file stays as a
        record that it was done here)."""

        with self._lock:
            self._held.pop(key, None)

    def abandon(self, key: 'str') -> 'None':
        """Give a unit back after a failure, so another task can retry it."""

        with self._lock:
            held = self._held.get(key)
        if held is None:
            return
        path, token, generation = held
        try:
            operation = self._acquire_operation(key, blocking=True)
        except OSError:


            return
        if operation is None:
            return
        forget = False
        try:
            try:
                still_owned = self._matches(path, token, generation)
            except OSError:


                return
            if still_owned:
                try:
                    path.unlink()
                except FileNotFoundError:
                    forget = True
                except OSError:
                    return
                else:
                    forget = True
            else:


                forget = True
        finally:
            self._release_operation(operation)
        if forget:
            with self._lock:
                if self._held.get(key) == held:
                    self._held.pop(key, None)

    def _heartbeat_once(self) -> 'None':
        now = time.time()
        with self._lock:
            held_items = list(self._held.items())
        for key, held in held_items:
            path, token, generation = held
            try:
                operation = self._acquire_operation(key)
            except OSError:
                continue
            if operation is None:
                continue
            try:
                if self._matches(path, token, generation):
                    try:
                        os.utime(str(path), (now, now))
                    except OSError:
                        pass
                else:
                    with self._lock:
                        if self._held.get(key) == held:
                            self._held.pop(key, None)
            finally:
                self._release_operation(operation)

    def start_heartbeat(self) -> 'None':
        if self._thread is not None:
            return

        def _beat() -> 'None':
            while not self._stop.wait(self.heartbeat_seconds):
                try:
                    self._heartbeat_once()
                except OSError:


                    continue

        self._stop.clear()
        self._thread = threading.Thread(target=_beat, name="claim-heartbeat", daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> 'None':
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class WorkerCrashError(RuntimeError):
    """A unit's worker process died (native crash, kill or out-of-memory).

    Recorded for a unit whose crash retry is used up.  The dispatcher has
    already rebuilt its pool and carries on with the other units.
    """


class WorkerPoolFailure(RuntimeError):
    """The process pool kept dying without completing any unit.

    Raised only after every prepared unit has been reported, so a pool that
    cannot run anything (a broken native library, an initializer that kills
    its worker) stops the sweep loudly instead of respawning forever.
    """


class _Unit:
    """One prepared unit: its reservations, dispatch arguments and crash record."""

    __slots__ = ("key", "gb", "cpus", "disk_gb", "args", "handle",
                 "crashes", "isolated")

    def __init__(self, key, gb, cpus, disk_gb, args):
        self.key = key
        self.gb = float(gb)
        self.cpus = int(cpus)
        self.disk_gb = float(disk_gb)
        self.args = args
        self.handle = None
        self.crashes = 0
        self.isolated = False


class _DispatchState:
    """In-flight work and reservations of one ``MemoryAwareDispatcher.run``."""

    def __init__(self, disk_capacity_gb: 'Optional[float]') -> 'None':
        self.inflight = []  # type: List[_Unit]
        # Prepared (claimed) units waiting to run again: a submission the
        # broken pool refused, or a crash casualty awaiting its retry.
        self.requeued = deque()  # type: Deque[_Unit]
        self.casualties = []  # type: List[_Unit]
        self.reserved_gb = 0.0
        self.reserved_cpus = 0
        self.reserved_disk_gb = 0.0
        self.disk_capacity_gb = disk_capacity_gb
        self.broken = None  # type: Optional[BaseException]
        # Pool breaks since a unit last completed; bounds the respawning.
        self.pool_failures = 0

    def isolating(self) -> 'bool':
        return any(unit.isolated for unit in self.inflight)


class MemoryAwareDispatcher:
    """Run units across a process pool under a node memory budget.

    A fixed pool size is the wrong control for a sweep whose units differ by
    two orders of magnitude in footprint: sized for the big units it wastes the
    node on the small ones, and sized for the small ones it OOM-kills on the
    big ones.  This admits work while ``sum(estimated peak) <= budget`` and,
    when CPU requests are supplied, ``sum(assembly threads) <= cores``.  A
    node therefore runs many cheap units concurrently, narrows for expensive
    ones, and can backfill spare resources without oversubscribing either.
    Memory is in GiB throughout (see ``BYTES_PER_GIB``).

    ``disk_budget_gb`` (GiB, or a callable returning the GiB free now) adds a
    scratch-disk reservation: a unit whose resource request carries a disk
    size (BoR far-block spill) starts only when that size fits beside the
    running units' reservations.  A callable is re-read whenever nothing is
    running, so space a crashed unit left behind is not counted as free.

    One unit is always admitted when nothing is running, so a unit larger than
    the whole budget still runs (and fails loudly with a MemoryError from the
    solver's own gate) instead of deadlocking the node.

    A worker process that dies breaks a ``concurrent.futures`` pool: every
    unit in flight fails with ``BrokenProcessPool`` and the pool refuses new
    work.  The dispatcher then rebuilds the pool (``pool.rebuild()``) and
    retries each casualty once, alone, so an innocent sibling completes and a
    unit that crashes again by itself is reported through ``on_error`` as a
    :class:`WorkerCrashError`.  The remaining units continue.  Only a pool
    that breaks ``max_pool_failures`` times in a row without completing any
    unit stops the run, with :class:`WorkerPoolFailure`.
    """

    def __init__(
        self,
        pool: 'Any',
        budget_gb: 'float',
        max_concurrent: 'int',
        cpu_budget: 'Optional[int]' = None,
        poll_seconds: 'float' = 0.05,
        disk_budget_gb: 'Any' = None,
        crash_retries: 'int' = 1,
        max_pool_failures: 'int' = 4,
    ) -> 'None':
        self.pool = pool
        self.budget_gb = float(budget_gb)
        self.max_concurrent = max(1, int(max_concurrent))
        self.cpu_budget = (
            None if cpu_budget is None else max(1, int(cpu_budget))
        )
        self.poll_seconds = float(poll_seconds)
        self.disk_budget_gb = disk_budget_gb
        self.crash_retries = max(0, int(crash_retries))
        self.max_pool_failures = max(1, int(max_pool_failures))
        self.pool_rebuilds = 0

    def run(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
        resource_request: 'Optional[Callable[[Dict[str, Any]], Tuple[float, ...]]]' = None,
    ) -> 'None':
        """Work through ``candidates`` in order and drain the pool.

        ``prepare(unit)`` claims the unit and returns
        (key, estimated_gb, apply_async_arguments), or None when the unit is
        already finished or held by another task.  It runs only when there is
        room to start something, so claims are taken at the moment work
        actually begins.

        When ``resource_request`` is supplied, it returns ``(GiB, CPUs)`` or
        ``(GiB, CPUs, disk GiB)`` before a unit is claimed.  The dispatcher
        scans past a temporarily blocked large unit and backfills the first
        smaller unit that fits every budget.  Deferred units keep their
        original order and are reconsidered whenever work completes.  Without
        the callback, admission uses memory alone and preserves candidate
        order for callers with fixed-size work.
        """

        if resource_request is not None:
            self._run_with_backfill(
                candidates, prepare, on_result, on_error, resource_request
            )
            return

        state = _DispatchState(self._disk_capacity())
        held = None  # type: Optional[_Unit]
        cursor = 0

        while True:
            while (len(state.inflight) < self.max_concurrent
                   and not state.isolating()):
                if state.requeued:
                    unit = state.requeued[0]
                    if not self._admissible(state, unit):
                        break
                    state.requeued.popleft()
                else:
                    if held is None:
                        while cursor < len(candidates):
                            prepared = prepare(candidates[cursor])
                            cursor += 1
                            if prepared is not None:
                                key, gb, args = prepared
                                held = _Unit(key, max(0.0, float(gb)), 0, 0.0, args)
                                break
                        if held is None:
                            break
                    if not self._admissible(state, held):
                        break
                    unit, held = held, None
                if not self._launch(state, unit):
                    break

            if state.broken is not None:
                self._recover(state, on_result, on_error)
                continue
            if not state.inflight:
                if held is None and cursor >= len(candidates) and not state.requeued:
                    return
                time.sleep(self.poll_seconds)
                continue

            progressed = self._collect(state, on_result, on_error)
            if state.broken is not None:
                self._recover(state, on_result, on_error)
                continue
            if not progressed:
                time.sleep(self.poll_seconds)

    def run_until_outputs_complete(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
        output_ready: 'Callable[[Dict[str, Any]], bool]',
        resource_request: 'Optional[Callable[[Dict[str, Any]], Tuple[float, ...]]]' = None,
        should_stop: 'Optional[Callable[[], bool]]' = None,
        retry_seconds: 'float' = 5.0,
        on_wait: 'Optional[Callable[[int, int], None]]' = None,
        sleep: 'Callable[[float], None]' = time.sleep,
    ) -> 'Tuple[Dict[str, Any], ...]':
        """Keep revisiting deferred claims until every expected output exists.

        ``run`` intentionally treats a unit held by another task as deferred.
        A single pass is not enough after a task dies: its fresh lease can make
        every peer skip the unit and exit before stale takeover becomes legal.
        This completion loop keeps workers alive, without stealing from a live
        owner, until the peer publishes the output or the lease becomes
        reclaimable.  A Slurm restart of the same task is fenced separately by
        :meth:`ClaimBroker._is_requeued_successor` and can reclaim immediately.

        The returned tuple is empty on success.  ``should_stop`` lets a driver
        stop retrying after a real solve failure and exit nonzero with the
        remaining units instead of looping forever on deterministic bad input.
        """

        delay = float(retry_seconds)
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError("retry_seconds must be positive and finite.")
        round_index = 0
        while True:
            missing = tuple(unit for unit in candidates if not output_ready(unit))
            if not missing:
                return ()
            if should_stop is not None and should_stop():
                return missing

            self.run(
                missing,
                prepare,
                on_result,
                on_error,
                resource_request,
            )
            missing = tuple(unit for unit in candidates if not output_ready(unit))
            if not missing:
                return ()
            if should_stop is not None and should_stop():
                return missing

            round_index += 1
            if on_wait is not None:
                on_wait(len(missing), round_index)
            sleep(delay)

    def _run_with_backfill(
        self,
        candidates: 'Sequence[Dict[str, Any]]',
        prepare: 'Callable[[Dict[str, Any]], Optional[Tuple[str, float, Any]]]',
        on_result: 'Callable[[str, Any], None]',
        on_error: 'Callable[[str, BaseException], None]',
        resource_request: 'Callable[[Dict[str, Any]], Tuple[float, ...]]',
    ) -> 'None':
        """Resource-aware admission with claim-safe, order-preserving backfill."""

        pending: 'Deque[Dict[str, Any]]' = deque(candidates)
        deferred: 'List[Dict[str, Any]]' = []
        state = _DispatchState(self._disk_capacity())

        while True:
            while (len(state.inflight) < self.max_concurrent
                   and not state.isolating()):
                if state.requeued:
                    # Already prepared (and claimed) work goes first; a crash
                    # casualty waits until it can run alone.
                    unit = state.requeued[0]
                    if not self._admissible(state, unit):
                        break
                    state.requeued.popleft()
                    if not self._launch(state, unit):
                        break
                    continue

                selected = None
                request = (0.0, 1, 0.0)
                while pending:
                    candidate = pending.popleft()
                    request = self._parse_request(resource_request(candidate))
                    if self._admissible(state, _Unit(None, *request, None)):
                        selected = candidate
                        break
                    deferred.append(candidate)
                if selected is None:
                    break

                prepared = prepare(selected)
                if prepared is None:
                    continue
                key, prepared_gb, args = prepared
                selected_gb, selected_cpus, selected_disk = request
                if not math.isclose(
                    max(0.0, float(prepared_gb)), selected_gb,
                    rel_tol=1.0e-12, abs_tol=1.0e-12,
                ):
                    on_error(
                        key,
                        ValueError(
                            "resource_request and prepare returned different "
                            f"memory estimates ({selected_gb:g} vs "
                            f"{float(prepared_gb):g} GB)"
                        ),
                    )
                    continue
                unit = _Unit(key, selected_gb, selected_cpus, selected_disk, args)
                if not self._launch(state, unit):
                    break

            if state.broken is not None:
                self._recover(state, on_result, on_error)
                if deferred:
                    pending.extendleft(reversed(deferred))
                    deferred = []
                continue
            if not state.inflight:
                if state.requeued:
                    continue
                if deferred:
                    pending.extendleft(reversed(deferred))
                    deferred = []
                    continue
                if not pending:
                    return

            progressed = self._collect(state, on_result, on_error)
            if state.broken is not None:
                self._recover(state, on_result, on_error)
                progressed = True
            if progressed and deferred:
                pending.extendleft(reversed(deferred))
                deferred = []
            if not progressed:
                time.sleep(self.poll_seconds)

    # -- admission ---------------------------------------------------------

    @staticmethod
    def _parse_request(request: 'Any') -> 'Tuple[float, int, float]':
        values = tuple(request)
        if len(values) == 2:
            gb, cpus = values
            disk = 0.0
        elif len(values) == 3:
            gb, cpus, disk = values
        else:
            raise ValueError(
                "resource_request must return (GiB, CPUs) or "
                "(GiB, CPUs, disk GiB)."
            )
        disk = float(disk or 0.0)
        return (
            max(0.0, float(gb)),
            max(1, int(cpus)),
            disk if math.isfinite(disk) and disk > 0.0 else 0.0,
        )

    def _disk_capacity(self) -> 'Optional[float]':
        budget = self.disk_budget_gb
        if callable(budget):
            try:
                budget = budget()
            except Exception:  # noqa: BLE001 - no disk figure, no disk gate
                budget = None
        if budget is None:
            return None
        value = float(budget)
        return max(0.0, value) if math.isfinite(value) else None

    def _admissible(self, state: '_DispatchState', unit: '_Unit') -> 'bool':
        """Whether ``unit`` may start beside what is already running."""

        if not state.inflight:
            return True
        if unit.isolated:
            return False
        if state.reserved_gb + unit.gb > self.budget_gb:
            return False
        if (self.cpu_budget is not None
                and state.reserved_cpus + unit.cpus > self.cpu_budget):
            return False
        if (state.disk_capacity_gb is not None and unit.disk_gb > 0.0
                and state.reserved_disk_gb + unit.disk_gb
                > state.disk_capacity_gb):
            return False
        return True

    def _launch(self, state: '_DispatchState', unit: '_Unit') -> 'bool':
        """Submit ``unit``; on a broken pool requeue it first and return False."""

        try:
            handle = self.pool.apply_async(*unit.args)
        except BrokenProcessPool as exc:
            state.requeued.appendleft(unit)
            if state.broken is None:
                state.broken = exc
            return False
        unit.handle = handle
        state.inflight.append(unit)
        state.reserved_gb += unit.gb
        state.reserved_cpus += unit.cpus
        state.reserved_disk_gb += unit.disk_gb
        return True

    @staticmethod
    def _release(state: '_DispatchState', unit: '_Unit') -> 'None':
        state.inflight.remove(unit)
        if state.inflight:
            state.reserved_gb -= unit.gb
            state.reserved_cpus -= unit.cpus
            state.reserved_disk_gb -= unit.disk_gb
        else:
            # Exact zero, so rounding cannot accumulate across a long sweep.
            state.reserved_gb = 0.0
            state.reserved_cpus = 0
            state.reserved_disk_gb = 0.0

    # -- completion and crash recovery -------------------------------------

    def _collect(self, state, on_result, on_error) -> 'bool':
        progressed = False
        for unit in list(reversed(state.inflight)):
            if not unit.handle.ready():
                continue
            self._release(state, unit)
            progressed = True
            self._settle(state, unit, on_result, on_error)
        return progressed

    @staticmethod
    def _settle(state, unit, on_result, on_error) -> 'None':
        try:
            value = unit.handle.get()
        except BrokenProcessPool as exc:
            state.casualties.append(unit)
            if state.broken is None:
                state.broken = exc
            return
        except BaseException as exc:  # noqa: BLE001 - reported, not raised
            state.pool_failures = 0
            on_error(unit.key, exc)
            return
        state.pool_failures = 0
        try:
            on_result(unit.key, value)
        except BaseException as exc:  # noqa: BLE001 - reported, not raised
            on_error(unit.key, exc)

    def _recover(self, state, on_result, on_error) -> 'None':
        """Rebuild a broken pool and decide the fate of every unit it held."""

        cause = state.broken
        state.broken = None
        rebuild = getattr(self.pool, "rebuild", None)
        detail = ""
        rebuild_error = None  # type: Optional[BaseException]
        if callable(rebuild):
            try:
                detail = str(rebuild() or "")
            except Exception as exc:  # noqa: BLE001 - reported below
                rebuild_error = exc
        else:
            rebuild_error = RuntimeError("the pool cannot be rebuilt")
        # Once rebuilt, every future of the dead pool has settled: a unit that
        # finished just before the break keeps its result, the rest are
        # casualties of the break.
        for unit in list(state.inflight):
            self._release(state, unit)
            if unit.handle is not None and unit.handle.ready():
                self._settle(state, unit, on_result, on_error)
            else:
                state.casualties.append(unit)
        state.broken = None
        casualties, state.casualties = state.casualties, []
        state.pool_failures += 1
        shared = len(casualties) > 1
        if rebuild_error is not None:
            message = (
                "The worker pool broke and could not be restarted "
                f"({rebuild_error}); no further units can run."
            )
            stranded = casualties + list(state.requeued)
            state.requeued.clear()
            for unit in stranded:
                on_error(unit.key, WorkerCrashError(message))
            raise WorkerPoolFailure(message) from cause
        self.pool_rebuilds += 1
        for unit in casualties:
            unit.crashes += 1
            if unit.crashes > self.crash_retries:
                on_error(unit.key, WorkerCrashError(
                    self._crash_message(unit, detail, shared)))
            else:
                unit.isolated = True
                state.requeued.append(unit)
        if state.pool_failures >= self.max_pool_failures:
            message = (
                f"The worker pool broke {state.pool_failures} times in a row "
                "without completing a unit"
                + (f" ({detail})" if detail else "")
                + "; the sweep stops here instead of respawning forever. "
                "Completed outputs are kept, so a rerun resumes."
            )
            stranded = list(state.requeued)
            state.requeued.clear()
            for unit in stranded:
                on_error(unit.key, WorkerCrashError(message))
            raise WorkerPoolFailure(message) from cause
        # Nothing is running now: re-read the free scratch space, which no
        # longer counts spill that a killed unit left behind.
        state.disk_capacity_gb = self._disk_capacity()

    def _crash_message(self, unit: '_Unit', detail: 'str', shared: 'bool') -> 'str':
        where = f" ({detail})" if detail else ""
        if unit.crashes > 1:
            history = "it crashed again when retried alone"
        elif shared:
            history = ("other units were in flight, so it may not be the "
                       "cause (crash retries are off)")
        else:
            history = ("it was the only unit in flight (crash retries are "
                       "off)")
        return (f"worker process died while solving this unit{where}; "
                f"{history}. The pool was rebuilt and the sweep continued.")


def build_sbatch_script(
    *,
    job_name: 'str',
    run_dir: 'os.PathLike',
    script_path: 'os.PathLike',
    array_size: 'int',
    array_throttle: 'Optional[int]',
    partition: 'str',
    cpus_per_node: 'Optional[int]',
    mem_per_node: 'Optional[str]',
    walltime: 'Optional[str]',
    account: 'Optional[str]',
    qos: 'Optional[str]',
    mail_type: 'Optional[str]',
    mail_user: 'Optional[str]',
    extra_sbatch: 'Sequence[str]',
    prologue: 'Sequence[str]',
    python_exe: 'str',
    worker_args: 'str',
    submission_index: 'int',
    blas_threads: 'int',
    extra_env: 'Optional[Dict[str, str]]' = None,
) -> 'str':
    """Write one array-job script.

    Array tasks are interchangeable: every task runs the same worker, which
    pulls units from the shared claim directory.  That means the array size is
    a *parallelism* knob rather than a partitioning key -- oversubscribing or
    cancelling part of an array cannot strand work, and a second submission on
    a different partition can join the same run at any time.
    """

    import shlex
    from pathlib import PurePosixPath

    run_dir_text = str(run_dir).replace("\\", "/")
    script_path_text = str(script_path).replace("\\", "/")
    for label, value in (("run_dir", run_dir_text), ("script_path", script_path_text)):
        if any(char in value for char in ('"', "\r", "\n")):
            raise ValueError(f"{label} contains characters unsafe for an sbatch script")
    run_dir = PurePosixPath(run_dir_text)
    script_path = PurePosixPath(script_path_text)
    array = f"0-{max(1, int(array_size)) - 1}"
    if array_throttle and int(array_throttle) > 0:
        array += f"%{int(array_throttle)}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={partition}",
        f'#SBATCH --output="{run_dir}/logs/sub{submission_index}_%A_%a.out"',
        f'#SBATCH --error="{run_dir}/logs/sub{submission_index}_%A_%a.err"',


        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
    ]
    if cpus_per_node is not None:
        lines.append(f"#SBATCH --cpus-per-task={int(cpus_per_node)}")
    else:
        lines.append("#SBATCH --exclusive")
    if mem_per_node:
        lines.append(f"#SBATCH --mem={mem_per_node}")
    if walltime:
        lines.append(f"#SBATCH --time={walltime}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if mail_type:
        lines.append(f"#SBATCH --mail-type={mail_type}")
    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")
    for extra in extra_sbatch:
        text = str(extra).strip()
        if not text:
            continue
        lines.append(text if text.startswith("#SBATCH") else f"#SBATCH {text}")

    lines += ["", "set -euo pipefail", f"cd {shlex.quote(str(script_path.parent))}"]


    for name in _BLAS_THREAD_VARS:
        lines.append(f"export {name}={max(1, int(blas_threads))}")
    for name, value in sorted((extra_env or {}).items()):
        lines.append(f"export {name}={shlex.quote(str(value))}")
    lines += list(prologue)
    lines += [
        (f"exec {shlex.quote(python_exe)} {shlex.quote(str(script_path))} "
         f"{worker_args}"),
        "",
    ]
    return "\n".join(lines)
