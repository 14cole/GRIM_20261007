"""Validated, serializable execution settings scoped to one solve."""
from contextlib import contextmanager
from functools import wraps
import inspect
import math
import ntpath
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

from ghost_backend.execution.runtime import ScopedValue

DEFAULTS = {
    'version': 1,
    'factorization': 'dense',
    'mesh_strategy': 'global',
    'basis_order': 1,
    'compressed_storage_mib': 2048,
    'ram_budget_gib': None,
    'temporary_directory': '',
    'assembly_threads': 1,
    'blas_threads': 1,
    'rhs_compression': 'auto',
    'angle_batch_size': 256,
    'assembly_tile': 0,
    'far_quadrature_order': 0,
    'far_grading': True,
    'dense_residual_storage': 'auto',
    'compressed_far_method': 'full',
    'frequency_preconditioner': 'off',
}
# compressed_storage_mib 0 sizes compressed storage from the solve's RAM limit at run time.
AUTOMATIC_STORAGE_MIB = 0
# The automatic profile resolves assembly threads on the executing host (see
# effective_assembly_threads): the physical cores available to this process,
# bounded by the scheduler's per-solve CPU allocation when one is set (batch
# workers, whose per-unit counts the drivers compute).
EFFICIENT_DEFAULTS = dict(DEFAULTS, factorization='adaptive', mesh_strategy='adaptive',
                          compressed_storage_mib=AUTOMATIC_STORAGE_MIB, assembly_threads='auto', blas_threads='auto')
# SMT siblings add no assembly throughput, tile parallelism saturates, and
# every thread is priced 128 MiB of workspace in the memory forecasts, so
# 'auto' never exceeds this many threads.
AUTOMATIC_ASSEMBLY_THREAD_LIMIT = 16
_ACTIVE = ScopedValue('ghost_execution_options', default=None)
_ASSEMBLY_ALLOCATION = ScopedValue('ghost_assembly_allocation', default=None)
_MEMORY_ALLOCATION = ScopedValue('ghost_memory_allocation', default=None)
_BLAS_LOCK = threading.RLock()
_BLAS_EVENTS = ScopedValue('ghost_blas_events', default=None)
_PUBLIC_DEPTH = ScopedValue('ghost_public_execution_depth',default=0)
_AUTOMATIC_REQUEST = ScopedValue('ghost_automatic_backend_request', default=False)
_FACTOR_EVIDENCE = ScopedValue('ghost_measured_factor_evidence', default=None)


def automatic_backend_requested():
    return _AUTOMATIC_REQUEST.get()

_ENV_FIELDS = {
    'GHOST_CPU_FACTORIZATION': 'factorization',
    'GHOST_COMPRESSED_STORAGE_MIB': 'compressed_storage_mib',
    'GHOST_CPU_RHS_COMPRESSION': 'rhs_compression',
    'GHOST_CPU_ANGLE_BATCH_SIZE': 'angle_batch_size',
    'GHOST_ASSEMBLY_THREADS': 'assembly_threads',
    'GHOST_ASSEMBLY_TILE': 'assembly_tile',
    'GHOST_FAR_QUAD_ORDER': 'far_quadrature_order',
    'GHOST_MAX_SOLVE_GB': 'ram_budget_gib',
}


def validate_options(value):
    """Return a complete independent JSON record; reject unsupported values."""
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Execution settings must contain supported fields only.')
    result = dict(DEFAULTS)
    result.update(value)
    if type(result['version']) is not int or result['version'] != 1:
        raise ValueError('Unsupported execution settings version.')
    if result['factorization'] not in ('dense', 'hierarchical', 'auto', 'compressed', 'adaptive'):
        raise ValueError('Choose dense, hierarchical, auto, compressed, or adaptive factorization.')
    if type(result['basis_order']) is not int or result['basis_order'] not in (1, 2, 3):
        raise ValueError('Boundary polynomial degree must be 1, 2 or 3.')
    if result['mesh_strategy'] not in ('global', 'local', 'adaptive'):
        raise ValueError('Mesh strategy must be global, local or adaptive.')
    if result['rhs_compression'] not in ('off', 'auto', 'on'):
        raise ValueError('RHS compression must be off, auto, or on.')
    if result['dense_residual_storage'] not in ('auto', 'memory', 'disk'):
        raise ValueError('Dense residual storage must be auto, memory, or disk.')
    if result['compressed_far_method'] not in ('full', 'verified_cur'):
        raise ValueError('Compressed far method must be full or verified_cur (experimental).')
    if result['frequency_preconditioner'] not in ('off', 'reuse'):
        raise ValueError('Frequency preconditioner must be off or reuse (experimental).')
    storage = result['compressed_storage_mib']
    if type(storage) is not int or not (storage == AUTOMATIC_STORAGE_MIB or 16 <= storage <= 1048576):
        raise ValueError('compressed_storage_mib must be 0 for automatic or an integer from 16 to 1048576.')
    for key, lower, upper in [('angle_batch_size', 1, 256),
                              ('assembly_tile', 0, 65536), ('far_quadrature_order', 0, 64)]:
        number = result[key]
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError('{} must be an integer from {} to {}.'.format(key, lower, upper))
    threads = result['assembly_threads']
    if threads != 'auto' and (type(threads) is not int or not 1 <= threads <= 1024):
        raise ValueError('Assembly threads must be auto or an integer from 1 to 1024.')
    threads = result['blas_threads']
    if threads != 'auto' and (type(threads) is not int or not 1 <= threads <= 1024):
        raise ValueError('BLAS threads must be auto or an integer from 1 to 1024.')
    ram = result['ram_budget_gib']
    if ram is not None and (type(ram) not in (int, float) or not math.isfinite(ram) or ram <= 0):
        raise ValueError('RAM budget must be positive GiB or null for available memory.')
    if type(result['far_grading']) is not bool:
        raise ValueError('Far grading must be true or false.')
    directory = result['temporary_directory']
    if not isinstance(directory, str) or any(c in directory for c in '\r\n\x00'):
        raise ValueError('Temporary directory must be a path string.')
    if directory and not (os.path.isabs(directory) or ntpath.isabs(directory)):
        raise ValueError('Temporary directory must be absolute or empty for the system temporary directory.')
    return result


def efficient_defaults():
    """Return the automatic run profile.

    Thread counts are ``'auto'`` so that a serialized profile (a GUI request,
    a batch manifest built on a login node) is resolved on the host that runs
    the solve; an integer count is still bounded by this host's CPU count.
    """
    values = dict(EFFICIENT_DEFAULTS)
    cores = max(1, os.cpu_count() or 1)
    for key in ('assembly_threads',):
        if values[key] != 'auto':
            values[key] = min(values[key], cores)
    return values


_PSUTIL_MISSING = False
_PSUTIL_IMPORT_LOCK = threading.Lock()


def _optional_psutil():
    """Avoid repeatedly searching for an absent optional package.

    Only module absence is cached, never CPU counts or affinity. An already
    loaded module (including one loaded after an earlier miss) takes priority.
    Other import failures remain retryable and keep the native fallback.
    """
    global _PSUTIL_MISSING
    if 'psutil' in sys.modules:
        return sys.modules['psutil']
    if _PSUTIL_MISSING:
        return None
    with _PSUTIL_IMPORT_LOCK:
        if 'psutil' in sys.modules:
            return sys.modules['psutil']
        if _PSUTIL_MISSING:
            return None
        try:
            import psutil
        except ModuleNotFoundError as exc:
            if exc.name == 'psutil':
                _PSUTIL_MISSING = True
            return None
        except Exception:
            return None
        return psutil


def _usable_logical_cpus():
    """Logical CPUs this process may use: affinity mask and SLURM task allocation."""
    count = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    else:
        # Windows has no sched_getaffinity. Respect a restricted desktop/job
        # affinity too, while keeping psutil optional for headless installs.
        affinity = None
        try:
            psutil = _optional_psutil()
            if psutil is not None:
                affinity = psutil.Process().cpu_affinity()
        except Exception:
            pass
        if affinity:
            count = min(count, len(affinity))
        elif os.name == 'nt':
            try:
                import ctypes
                kernel = ctypes.windll.kernel32
                kernel.GetCurrentProcess.restype = ctypes.c_void_p
                kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
                process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
                if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                        ctypes.byref(process_mask), ctypes.byref(system_mask)) and process_mask.value:
                    count = min(count, bin(process_mask.value).count('1'))
            except (AttributeError, OSError):
                pass
    for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        raw = os.environ.get(name, '').strip()
        if raw.isdigit() and int(raw) > 0:
            count = min(count, int(raw))
            break
    return max(1, int(count))


_PHYSICAL_CORES = []


def physical_core_count():
    """Physical cores of this host (psutil), else the logical CPU count."""
    if not _PHYSICAL_CORES:
        count = None
        try:
            psutil = _optional_psutil()
            if psutil is not None:
                count = psutil.cpu_count(logical=False)
        except Exception:
            count = None
        _PHYSICAL_CORES.append(max(1, int(count or os.cpu_count() or 1)))
    return _PHYSICAL_CORES[0]


def host_assembly_threads():
    """'auto' assembly threads on this host (before any scheduler allocation).

    The physical core count, bounded by the logical CPUs this process may use
    (affinity mask, SLURM task allocation) and by
    ``AUTOMATIC_ASSEMBLY_THREAD_LIMIT``.  The solve's BLAS pool stays at its
    reservation (``blas_thread_reservation``) while assembly threads run.
    """
    return max(1, min(physical_core_count(), _usable_logical_cpus(), AUTOMATIC_ASSEMBLY_THREAD_LIMIT))


# Every 2D GUI and batch run uses the automatic profile: the backend, mesh and
# polynomial degree are chosen per solve, and double-precision LU is fixed.
AUTOMATIC_SOLVER_METHOD = 'auto'
AUTOMATIC_LU_PRECISION = 'double'


def automatic_options(ram_budget_gib=None):
    """The single 2D run profile, optionally bounded by a per-solve RAM budget."""
    values = efficient_defaults()
    if ram_budget_gib is not None:
        values['ram_budget_gib'] = float(ram_budget_gib)
    return validate_options(values)


def automatic_run(scattering='monostatic', ram_budget_gib=None):
    """Solver method, LU precision and execution settings for a 2D run.

    Adaptive meshing and backend selection are monostatic capabilities, so a
    bistatic run uses dense factorization on the reference mesh.
    """
    options = automatic_options(ram_budget_gib)
    if scattering == 'monostatic':
        return dict(solver_method=AUTOMATIC_SOLVER_METHOD, lu_precision=AUTOMATIC_LU_PRECISION,
                    execution_options=options)
    if scattering != 'bistatic':
        raise ValueError('Scattering must be monostatic or bistatic.')
    options = validate_options(dict(options, factorization='dense', mesh_strategy='global'))
    return dict(solver_method='direct', lu_precision=AUTOMATIC_LU_PRECISION, execution_options=options)


def from_environment(defaults=None):
    """Capture launch defaults once, for callers without an explicit profile."""
    values = validate_options(defaults if defaults is not None else {})
    for name, key in _ENV_FIELDS.items():
        raw = os.environ.get(name, '').strip()
        if not raw:
            continue
        if key == 'ram_budget_gib':
            values[key] = float(raw)
        elif key in ('factorization', 'rhs_compression') or (key == 'assembly_threads' and raw == 'auto'):
            values[key] = raw.lower()
        else:
            values[key] = int(raw)
    explicit_blas = os.environ.get('OPENBLAS_NUM_THREADS', '') or os.environ.get('MKL_NUM_THREADS', '')
    if explicit_blas:
        values['blas_threads'] = int(explicit_blas)
    values['far_grading'] = os.environ.get('GHOST_FAR_GRADED', '1') != '0'
    return validate_options(values)


def current_options():
    """Return a copy of the active settings, or None outside a configured run."""
    value = _ACTIVE.get()
    return dict(value) if value is not None else None


def option(name, fallback=None):
    active = _ACTIVE.get()
    return active[name] if active is not None else fallback


def effective_assembly_threads(fallback=1):
    """Resolve requested concurrency against the scheduler's CPU allocation.

    ``'auto'`` is this host's physical cores (``host_assembly_threads``),
    bounded by the allocation when one is set: a batch worker runs with at
    most the per-unit count its driver computed, so concurrent units never
    oversubscribe the node, and a lone unit handed every logical CPU still
    uses no more threads than physical cores.  An integer request is capped
    by the allocation.
    """
    active = _ACTIVE.get()
    if active is None:
        return min(max(1, int(fallback)), allocated_cpu_budget())
    requested = active['assembly_threads']
    allocation = allocated_cpu_budget()
    if requested == 'auto':
        host = host_assembly_threads()
        return min(allocation, host)
    return min(requested, allocation)


def allocated_memory_budget():
    return _MEMORY_ALLOCATION.get()


def allocated_cpu_budget():
    """Usable affinity/SLURM CPUs, bounded by this solve's reservation."""
    usable = _usable_logical_cpus()
    return max(1, min(usable, int(_ASSEMBLY_ALLOCATION.get() or usable)))


@contextmanager
def cpu_allocation_scope(cpus):
    """Temporarily lend part of a CPU reservation without changing the profile."""
    count = min(allocated_cpu_budget(), max(1, int(cpus)))
    with _ASSEMBLY_ALLOCATION.override(count):
        yield count


@contextmanager
def memory_allocation_scope(memory_gib):
    """Bound one solve's RAM without changing its execution profile."""
    if memory_gib is None:
        yield allocated_memory_budget()
        return
    memory = float(memory_gib)
    if not math.isfinite(memory) or memory <= 0.0:
        raise ValueError('Allocated solve memory must be positive and finite.')
    inherited = allocated_memory_budget()
    if inherited is not None:
        memory = min(memory, inherited)
    with _MEMORY_ALLOCATION.override(memory):
        yield memory


def blas_core_budget():
    """Threads for dense linear algebra: the CPU allocation, at most the physical cores.

    OpenBLAS on the SMT threads of an 8-core, 16-thread host was slower for
    LU (complex N = 8,000: 3.06 s on 16 threads against 2.19 s on 8) and
    pathological for products of fewer than about 256 rows: 0.5 to 2.1 s
    against 1.4 to 3.7 ms on 8 threads (a disk-spooled residual, 47-row
    blocks, took over 20 minutes instead of seconds).
    """
    return max(1, min(allocated_cpu_budget(), physical_core_count()))


_POOL_BLAS_LOCK = threading.Lock()
_POOL_BLAS_USERS = 0
_POOL_BLAS_LIMITS = None


@contextmanager
def single_thread_blas():
    """BLAS on one thread while a pool of Python threads issues BLAS calls.

    OpenBLAS's threaded server (0.3.31, pthreads layer, Windows) faulted with
    an access violation in its own threads when eight far-tile threads issued
    multithreaded products at once: a streamed BoR range rebuilt inside a mode
    worker, whose BLAS share was two threads, crashed the process in two of
    three 5 GHz dielectric solves and in the first of a series of smaller ones,
    and never with one BLAS thread.  Such pools already occupy the cores, so
    their calls lose nothing on one thread.  Sections may nest and overlap
    across threads: the first to start sets the process-wide limit, the last
    to finish restores the limits it found.
    """
    global _POOL_BLAS_USERS, _POOL_BLAS_LIMITS
    with _POOL_BLAS_LOCK:
        if _POOL_BLAS_USERS == 0:
            from ghost_backend.execution.thread_control import threadpool_info, threadpool_limits
            threaded = any(int(pool.get('num_threads') or 1) > 1 for pool in threadpool_info()
                           if pool.get('user_api') == 'blas')
            _POOL_BLAS_LIMITS = threadpool_limits(limits=1, user_api='blas') if threaded else None
        _POOL_BLAS_USERS += 1
    try:
        yield
    finally:
        with _POOL_BLAS_LOCK:
            _POOL_BLAS_USERS -= 1
            if _POOL_BLAS_USERS == 0:
                limits, _POOL_BLAS_LIMITS = _POOL_BLAS_LIMITS, None
                if limits is not None:
                    limits.restore_original_limits()


def blas_thread_reservation(options=None):
    """Initial BLAS pool and batch CPU reservation; auto may borrow idle assembly cores."""
    value = (options or current_options() or DEFAULTS)['blas_threads']
    return min(2 if value == 'auto' else int(value), allocated_cpu_budget())


def environment_value(name, default=''):
    """Read a captured solver setting, with launch-environment compatibility."""
    active = _ACTIVE.get()
    if active is not None:
        if name == 'GHOST_DENSE_BACKEND':
            return 'cpu'
        key = _ENV_FIELDS.get(name)
        if key is not None:
            value = active[key]
            return '' if value is None else str(value)
    return os.environ.get(name, default)


def validate_for_run(options, method='direct', precision='double', scattering='monostatic', kind='2d'):
    value = validate_options(options)
    mode = value['factorization']
    if (value['mesh_strategy']=='adaptive' or value['basis_order']>1) and (kind!='2d' or scattering!='monostatic'):
        raise ValueError('Adaptive polynomial meshing supports 2D monostatic runs only.')
    if value['mesh_strategy'] == 'local' and (kind != '2d' or scattering != 'monostatic'):
        raise ValueError('Local material meshing supports 2D monostatic runs only.')
    if kind != '2d' and mode != 'dense':
        raise ValueError('Hierarchical and compressed selections apply to the 2D solver only.')
    if mode != 'dense' and (precision != 'double' or scattering != 'monostatic'):
        raise ValueError('Hierarchical and compressed runs require monostatic scattering and double precision.')
    if mode in ('compressed', 'adaptive') and method not in ('auto','experimental_cpu'):
        raise ValueError('Compressed assembly requires CPU streaming kernel evaluation.')
    if method == 'experimental_cpu' and (precision != 'double' or scattering != 'monostatic'):
        raise ValueError('CPU streaming requires monostatic scattering and double precision.')
    return value


def temporary_directory():
    """Return and check the configured local directory for compressed spooling."""
    directory = option('temporary_directory', '') or tempfile.gettempdir()
    path = Path(directory)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('Temporary directory is not available on this host: {}'.format(directory))
    return str(path)


@contextmanager
def execution_scope(value, limit_blas=False, assembly_threads=None, memory_budget_gib=None):
    """Restore settings after completion or failure; optionally control native BLAS."""
    checked = validate_options(value)
    allocation = assembly_threads if assembly_threads is not None else _ASSEMBLY_ALLOCATION.get()
    if allocation is not None:
        allocation = max(1, int(allocation))
        inherited_allocation = _ASSEMBLY_ALLOCATION.get()
        if inherited_allocation is not None:
            allocation = min(allocation, inherited_allocation)
    memory=memory_budget_gib if memory_budget_gib is not None else _MEMORY_ALLOCATION.get()
    if memory is not None and (not math.isfinite(memory) or memory <= 0):
        raise ValueError('Allocated solve memory must be positive and finite.')
    inherited_memory=_MEMORY_ALLOCATION.get()
    if inherited_memory is not None:
        memory=min(memory,inherited_memory)
    with _ACTIVE.override(checked), _ASSEMBLY_ALLOCATION.override(allocation), _MEMORY_ALLOCATION.override(memory):
        if not limit_blas:
            yield checked
            return
        import numpy
        import scipy.linalg
        from ghost_backend.execution.thread_control import threadpool_limits
        with _BLAS_LOCK:
            with threadpool_limits(limits=blas_thread_reservation(checked), user_api='blas'):
                yield checked


@contextmanager
def linear_algebra_threads(matrix_size=None):
    """Honor explicit caps; auto uses matrix size within the solve's CPU allocation."""
    active = _ACTIVE.get()
    if active is None:
        yield None
        return
    setting = active['blas_threads']
    if setting == 'auto':
        n = max(0, int(matrix_size or 0))
        desired = 1 if n <= 256 else 4 if n <= 1024 else 8 if n <= 2048 else blas_core_budget()
        desired = min(desired, blas_core_budget())
    else:
        desired = int(setting)
    target = min(desired, allocated_cpu_budget())
    from ghost_backend.execution.thread_control import threadpool_info, threadpool_limits
    with _BLAS_LOCK:
        with threadpool_limits(limits=target, user_api='blas'):
            events = _BLAS_EVENTS.get()
            if events is not None:
                events.append(dict(matrix_size=matrix_size, policy=setting, requested_threads=target,
                    pools=[{key: row.get(key) for key in ('internal_api', 'prefix', 'num_threads')}
                           for row in threadpool_info() if row.get('user_api') == 'blas']))
            yield target


def configured_execution(function):
    """Accept execution_options at public solver entry points."""
    signature = inspect.signature(function)
    def invoke_configured(*args, **kwargs):
        execution_start=time.perf_counter()
        requested = kwargs.pop('execution_options', None)
        inherited = current_options()
        # Public convenience spelling; internally retain the existing CPU
        # kernel/field lifecycle, with an explicitly matrix-free factorization.
        supplied=signature.bind(*args,**kwargs)
        # Normalize at the public boundary before planners or nested solves use
        # sequence truth tests. Keep dimensionality and finite-value validation.
        import numpy as np
        for name in ('frequencies_ghz','elevations_deg','incidence_angles_deg','observation_angles_deg'):
            if name not in supplied.arguments or supplied.arguments[name] is None:
                continue
            raw=np.asarray(supplied.arguments[name])
            if raw.ndim != 1 or raw.dtype.kind not in 'iuf':
                raise ValueError('{} must be a one-dimensional real numeric sequence.'.format(name))
            supplied.arguments[name]=raw.tolist()
        args,kwargs=supplied.args,supplied.kwargs
        method_parameter=signature.parameters.get('solver_method')
        public_method=str(supplied.arguments.get('solver_method',method_parameter.default if method_parameter else '')).strip().lower()
        automatic=public_method=='auto' and 'monostatic' in function.__name__
        if automatic:
            from ghost_backend.linalg.refined_lu import requested_precision
            if requested is not None and not isinstance(requested,dict):
                raise ValueError('Execution settings must be an object.')
            if inherited is None:
                # Explicit old profiles and launch overrides retain their meaning.
                base=from_environment()
                if not os.environ.get('GHOST_CPU_FACTORIZATION','').strip():
                    base['factorization']='adaptive'
                if requested is None and not os.environ.get('GHOST_ASSEMBLY_THREADS','').strip():
                    # A bare automatic request runs on the host's physical cores,
                    # as the GUI and batch profiles do (the former one-thread
                    # default made a direct API solve 4x slower than the GUI's).
                    base['assembly_threads']='auto'
                    base['blas_threads']='auto'
                if requested is None or 'mesh_strategy' not in requested:
                    base['mesh_strategy']='adaptive'
                requested=dict(base,**(requested or {}))
            supplied.arguments['solver_method']='experimental_cpu' if requested_precision()=='double' else 'direct'
            if requested_precision()!='double' and inherited is None and requested['factorization']=='adaptive':
                requested['factorization']='dense'
            if 'max_panels' in signature.parameters and 'max_panels' not in supplied.arguments:
                supplied.arguments['max_panels']=100_000
            args,kwargs=supplied.args,supplied.kwargs
        if requested is not None and inherited is not None and validate_options(requested) != inherited:
            raise ValueError('A nested solve must use the active execution settings.')
        value = requested if requested is not None else inherited
        if value is None and os.environ.get('GHOST_CPU_FACTORIZATION', '').strip().lower() == 'adaptive':
            value = from_environment()
        if value is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        from ghost_backend.linalg.refined_lu import requested_precision
        value = validate_for_run(value, method=bound.arguments.get('solver_method', 'direct'),
                         precision=requested_precision(),
                         scattering='bistatic' if 'bistatic' in function.__name__ else 'monostatic')
        timing_key=None
        if _PUBLIC_DEPTH.get()==1 and 'monostatic' in function.__name__:
            from ghost_backend.execution.timing_history import request_key
            try:
                with execution_scope(value):
                    timing_key=request_key(bound.arguments,value,function.__name__)
            except (OSError,TypeError,ValueError):
                pass  # Optional timing evidence never replaces input validation.
        timing_entries=None
        if timing_key is not None and _PUBLIC_DEPTH.get()==1:
            from ghost_backend.execution.timing_history import read, MAX_AGE, with_factor_choices
            from ghost_backend.execution.factor_timing import choices
            from ghost_backend.linalg.crossover import install
            from ghost_backend.execution.selection import current_batch_selection
            timing_entries=read()
            factor_choices,factor_evidence=choices(timing_key,timing_entries,MAX_AGE)
            # A batch reservation was priced outside this request scope. Keep
            # its original factor policy rather than introduce a larger LU
            # workspace without repricing that already admitted worker plan.
            if current_batch_selection() is None and not os.environ.get('GHOST_HIERARCHICAL_MIN_UNKNOWNS'):
                install(factor_choices)
                _FACTOR_EVIDENCE.get().update(factor_evidence)
                timing_key=with_factor_choices(timing_key,factor_choices)
        selection = None
        requested_value = value
        if value['factorization'] == 'adaptive':
            if 'cfie_alpha' in bound.arguments:
                from ghost_backend.twod.solver import _validate_disabled_2d_cfie_alpha
                _validate_disabled_2d_cfie_alpha(bound.arguments['cfie_alpha'])
            from ghost_backend.execution.selection import select_backend, current_batch_selection
            selection = current_batch_selection()
            batch_planned=selection is not None
            if selection is None:
                planning_start=time.perf_counter()
                selection = select_backend(bound.arguments, value, certified='certified' in function.__name__)
                selection['planning_seconds']=time.perf_counter()-planning_start
            if timing_key is not None:
                from ghost_backend.execution.timing_history import adjust
                selection=adjust(selection,timing_key,batch=batch_planned,entries=timing_entries)
            value = dict(value, factorization=selection['selected'])
        if bound.arguments.get('solver_method') == 'auto' and value['factorization'] == 'compressed':
            bound.arguments['solver_method']='experimental_cpu'
            args,kwargs=bound.args,bound.kwargs
        def invoke():
            nonlocal value,selection
            from ghost_backend.execution.errors import BackendNumericalError
            from ghost_backend.execution.selection import request_selection_scope
            from numpy.linalg import LinAlgError
            failures=[]
            modes=[value['factorization']]+(selection.get('retry_order',[]) if selection else [])
            # The forecast of this request (its candidate meshes, priced and
            # admitted) is visible to the adaptive controller, which would
            # otherwise rebuild and re-price the same meshes per candidate step.
            forecast=selection if selection is not None and selection.get('meshes') else None
            for index,mode in enumerate(modes):
                value=dict(value,factorization=mode)
                attempt_start=time.perf_counter()
                try:
                    with execution_scope(value), request_selection_scope(forecast):
                        result=function(*args,**kwargs)
                except (MemoryError,BackendNumericalError,LinAlgError) as exc:
                    if selection is None or index == len(modes)-1:
                        raise
                    failures.append(dict(backend=mode,error=type(exc).__name__,message=str(exc),
                                         wall_seconds=time.perf_counter()-attempt_start))
                else:
                    if failures:
                        selection=dict(selection,initial_selection=selection['selected'],selected=mode,
                                       failed_attempts=failures,reason='Selected after an admitted numerical/resource retry.')
                    return result
                # Exception tracebacks no longer own failed native workspaces.
                import gc
                gc.collect()
            raise RuntimeError('Automatic backend retry exhausted.')

        events = _BLAS_EVENTS.get()
        if events is None:
            events = []
        event_start = len(events)
        with _BLAS_EVENTS.override(events), _AUTOMATIC_REQUEST.override(selection is not None or _AUTOMATIC_REQUEST.get()), execution_scope(value, limit_blas=requested is not None and inherited is None):
            result = invoke()
            if isinstance(result, dict):
                metadata = result.setdefault('metadata', {})
                metadata['linear_algebra_execution'] = list(events[event_start:])
                if _FACTOR_EVIDENCE.get():
                    metadata['measured_factor_choices']=dict(_FACTOR_EVIDENCE.get())
                adaptation = metadata.get('adaptive_mesh', {})
                actual = dict(value, basis_order=metadata.get('polynomial_degree', value['basis_order']))
                per_frequency = metadata.get('frequency_metadata', [])
                if per_frequency:
                    actual = dict(value)
                    metadata['execution_options_scope'] = 'request; effective settings are in frequency_metadata'
                if adaptation.get('final_backend') and not per_frequency:
                    actual['factorization'] = adaptation['final_backend']
                    if selection is not None and selection['selected'] != actual['factorization']:
                        selection = dict(selection, initial_selection=selection['selected'],
                            selected=actual['factorization'],
                            reason='Backend reselected on the adaptive mesh; see adaptive_mesh steps for admission evidence.')
                metadata['execution_options'] = actual
                result['metadata'].setdefault('polynomial_degree', value['basis_order'])
                result['metadata']['execution_wall_seconds']=time.perf_counter()-execution_start
                if allocated_memory_budget() is not None:
                    result['metadata']['execution_memory_reservation_gib']=allocated_memory_budget()
                if selection is not None:
                    if per_frequency and metadata.get('backend_selection'):
                        metadata['request_backend_selection'] = selection
                    else:
                        metadata['backend_selection'] = selection
                    result['metadata']['requested_execution_options'] = requested_value
                if automatic:
                    result['metadata']['solver_method_requested']='auto'
                result['metadata']['execution_threads'] = dict(
                    assembly=effective_assembly_threads(), blas=current_options()['blas_threads'])
                if timing_key is not None:
                    from ghost_backend.execution.timing_history import record
                    record(timing_key,actual['factorization'],result['metadata']['execution_wall_seconds'],result['metadata'])
            return result
    @wraps(function)
    def call(*args,**kwargs):
        if _PUBLIC_DEPTH.get():
            with _PUBLIC_DEPTH.override(_PUBLIC_DEPTH.get()+1):
                return invoke_configured(*args,**kwargs)
        from ghost_backend.linalg.crossover import scope
        with _PUBLIC_DEPTH.override(1), scope({}), _FACTOR_EVIDENCE.override({}):
            return invoke_configured(*args,**kwargs)
    parameters = list(signature.parameters.values())
    parameters.append(inspect.Parameter('execution_options', inspect.Parameter.KEYWORD_ONLY, default=None))
    call.__signature__ = signature.replace(parameters=parameters)
    return call
