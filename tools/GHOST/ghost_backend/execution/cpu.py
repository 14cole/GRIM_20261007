"""Opt-in CPU execution state, isolated per synchronous solve/thread.

Two kinds of state exist.  The experimental CPU method owns the full state
(kernel tables, batched solves, evidence and labels).  Every other 2-D path
(explicit ``direct``, the mixed-precision automatic run, bistatic solves and
boundary densities) owns a *kernel-only* state: validated kernel tables and
native far assembly for operator evaluation, with the requested LU precision
and linear method left unchanged.
"""
from ghost_backend.execution.options import environment_value
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import inspect
import pickle
import os
from ghost_backend.execution.runtime import ScopedValue

EXPERIMENTAL_METHOD = "experimental_cpu"
BATCH_SIZE = 256
CACHE_BYTES = 64 * 1024 ** 2
TABLE_BYTES = 32 * 1024 ** 2
STREAMED_FORMULATIONS = frozenset(("te_robin", "robin", "single_dielectric", "multi_region",
                                  "sheet", "mixed_sheet_pec", "thin_dielectric_layer"))
KERNEL_REPORT_KEY = 'cpu_kernel_execution'
_STATE = ScopedValue("ghost_cpu_execution", default=None)
# True while a kernel-only state is hidden from a mixed-precision linear solve.
_HIDDEN = ScopedValue("ghost_cpu_execution_hidden", default=False)


def configured_batch_size():
    """Same explicit angle-batch limit for execution and resource planning."""
    value = str(environment_value('GHOST_CPU_ANGLE_BATCH_SIZE', str(BATCH_SIZE))).strip()
    try:
        size = int(value)
    except ValueError:
        raise ValueError('GHOST_CPU_ANGLE_BATCH_SIZE must be a positive integer.')
    if size < 1 or size > BATCH_SIZE:
        raise ValueError('GHOST_CPU_ANGLE_BATCH_SIZE must be between 1 and {}.'.format(BATCH_SIZE))
    return size


def current_state():
    state = _STATE.get()
    if state is None or not state.active or _HIDDEN.get():
        return None
    return state


def requested_cpu():
    return _STATE.get() is not None


@contextmanager
def requested_precision_solve():
    """Keep the caller's LU precision for a linear solve under kernel tables.

    The shared field solve forces double-precision LU whenever a CPU state is
    visible, which is the experimental method's contract.  Kernel tables only
    change how operators are evaluated, so under a kernel-only state with
    mixed precision requested the state is hidden for the duration of the
    solve: the factorization then follows ``linear_precision`` exactly as it
    did before tables were enabled.  Operators assembled before this scope
    already used the tables.
    """
    state = _STATE.get()
    if state is None or not getattr(state, 'kernel_only', False):
        yield
        return
    from ghost_backend.linalg.refined_lu import requested_precision
    if requested_precision() == 'double':
        yield
        return
    with _HIDDEN.override(True):
        yield


class CPUState:
    def __init__(self, abort_event=None, progress_callback=None, kernel_only=False):
        self.active = True
        self.kernel_only = bool(kernel_only)
        self.reuse_operators = True
        self.abort_event = abort_event
        self.progress_callback = None
        self.batch_size = configured_batch_size()
        self.cache = OrderedDict()
        self.tables = OrderedDict()
        self.table_bytes = 0
        self.table_events = []
        self.systems = []
        self.formulations = []
        self.memory_estimates = []
        self.stage_cost_meshes = []
        self.cache_stats = dict(hits=0, stores=0, evictions=0, bytes=0,
                                peak_bytes=0, budget_bytes=CACHE_BYTES)

    def checkpoint(self, completed=None, total=None):
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")


        if completed is not None and self.progress_callback is not None:
            try:
                self.progress_callback(completed, total)
            except Exception:
                pass
        if self.abort_event is not None and self.abort_event.is_set():
            raise InterruptedError("Solve canceled by user.")

    def select(self, resources):
        self.active = resources["formulation"] in STREAMED_FORMULATIONS


        self.reuse_operators = False
        self.formulations.append(dict(formulation=resources["formulation"],
            streamed=self.active, reason="" if self.active else
            "This formulation uses the reference CPU implementation."))
        self.checkpoint()

    def report(self):
        from ghost_backend.linalg.refined_lu import requested_precision
        return dict(version=1, precision=requested_precision() if self.kernel_only else "double",
                    device="cpu", scope="kernel_tables" if self.kernel_only else "experimental_cpu",
                    batch_size=self.batch_size, cache=dict(self.cache_stats),
                    kernel_tables=list(self.table_events),
                    table_budget_bytes=TABLE_BYTES, systems=list(self.systems),
                    formulations=list(self.formulations), memory_estimates=list(self.memory_estimates),
                    stage_cost_meshes=list(self.stage_cost_meshes))


def select_formulation(resources, progress_callback=None, frequency_ghz=None, polarization=None):
    state = _STATE.get()
    if state is not None:
        state.progress_callback = progress_callback
        state.select(resources)
        if frequency_ghz is not None:
            state.stage_cost_meshes.append(dict(frequency_ghz=float(frequency_ghz), phase='actual',
                polarization=polarization, polynomial_degree=resources.get('basis_width',2)-1,
                panels=resources.get('panels'), unknowns=resources['system_dofs'],
                formulation=resources['formulation'], resources=dict(resources)))


def _kernel_tables_unavailable():
    """Paths that keep their previous execution unchanged.

    A nested call reuses the enclosing state.  An explicit GPU/auto dense
    backend keeps its device choice (a visible CPU state pins solves to the
    CPU).  LU precision is not a reason: ``requested_precision_solve`` keeps a
    mixed-precision factorization under a kernel-only state.
    """
    if _STATE.get() is not None:
        return True
    from ghost_backend.twod.solver import _requested_dense_backend
    return _requested_dense_backend()[0] in ('gpu', 'auto')


def _run_with_kernel_tables(function, signature, args, kwargs):
    """Run ``function`` owning a kernel-only state released on every exit."""
    bound = signature.bind(*args, **kwargs)
    state = CPUState(bound.arguments.get('abort_event'), bound.arguments.get('progress_callback'),
                     kernel_only=True)
    state.reuse_operators = False
    with _STATE.override(state):
        try:
            state.checkpoint()
            result = function(*args, **kwargs)
            state.checkpoint()
            if isinstance(result, dict):
                result.setdefault('metadata', {})[KERNEL_REPORT_KEY] = state.report()
            return result
        finally:
            state.cache.clear()
            state.tables.clear()


def kernel_tables(function):
    """Validated CPU kernel tables for a direct-method 2-D computation.

    Operators use the tables and the native far assembly; the LU keeps the
    requested precision (see ``requested_precision_solve``).  Explicit GPU/auto
    dense-backend requests keep their existing path.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        if _kernel_tables_unavailable():
            return function(*args, **kwargs)
        return _run_with_kernel_tables(function, signature, args, kwargs)
    return call


def experimental_monostatic(function):
    """Own one bounded cache across channels and certification mesh pairs.

    Other monostatic methods (``direct``, including the mixed-precision
    automatic run) own a kernel-only state: tables for operator evaluation,
    with the method's LU, labels and metadata otherwise unchanged.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        method = str(bound.arguments.get("solver_method", "direct")).strip().lower()
        if method != EXPERIMENTAL_METHOD:
            if _kernel_tables_unavailable():
                return function(*args, **kwargs)
            return _run_with_kernel_tables(function, signature, args, kwargs)
        from ghost_backend.linalg.refined_lu import requested_precision
        if requested_precision() != "double":
            raise ValueError("Experimental CPU requires double LU precision.")
        if _STATE.get() is not None:
            return function(*args, **kwargs)
        state = CPUState(bound.arguments.get("abort_event"), bound.arguments.get("progress_callback"))
        with _STATE.override(state):
            state.checkpoint()
            result = function(*args, **kwargs)
            state.checkpoint()
            metadata = result.setdefault("metadata", {})
            metadata["solver_method_requested"] = EXPERIMENTAL_METHOD
            metadata["solver_method"] = "dense_lu_experimental_cpu" if state.systems else "dense_lu"
            from ghost_backend.compressed.runtime import enabled
            if enabled() and state.systems:metadata['solver_method']='compressed_experimental_cpu'
            adaptation=metadata.get('adaptive_mesh',{})
            if adaptation.get('final_backend'):
                selected=adaptation['final_backend']
                metadata['solver_method']={'dense':'dense_lu_experimental_cpu','compressed':'compressed_experimental_cpu'}.get(selected,selected)
            if metadata.get('frequency_metadata'):
                methods = {row['metadata'].get('solver_method', '') for row in metadata['frequency_metadata']}
                metadata['solver_method'] = next(iter(methods)) if len(methods)==1 else 'mixed (see frequency metadata)'
            metadata["experimental_cpu"] = state.report()
            return result
    return call


def bistatic_kernels(function):
    """Own validated CPU tables across bistatic channels and mesh checks.

    This enables kernel evaluation only; it does not opt a bistatic request
    into monostatic adaptation, compressed assembly or a different LU method.
    Explicit GPU/auto dense-backend callers retain their existing execution
    path; mixed-precision callers keep mixed LU (``requested_precision_solve``).
    """
    signature = inspect.signature(function)

    @wraps(function)
    def call(*args, **kwargs):
        if _kernel_tables_unavailable():
            return function(*args, **kwargs)
        return _run_with_kernel_tables(function, signature, args, kwargs)
    return call


def _assembly_rule_key():
    """Scoped/global assembly settings that change operator values.

    The far-pair quadrature order and grading change the computed operator;
    the tile edge changes the floating-point summation order.  They are read
    exactly as the operator code resolves them (active execution option, else
    the module/launch default).
    """
    from ghost_backend.execution.options import option
    from ghost_backend.twod import operators as ops
    return (int(option('far_quadrature_order', ops._FAR_QUAD_ORDER)),
            bool(option('far_grading', ops._FAR_GRADED)),
            int(option('assembly_tile', ops._ASSEMBLY_TILE)))


def cached_operator(label):
    """Cache immutable operator references within one bounded solve scope."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def call(*args, **kwargs):
            state = current_state()
            from ghost_backend.twod.assembly.session import current_session
            if (state is None or not state.reuse_operators or current_session() is not None
                    or kwargs.get('operator_outputs') is not None or kwargs.get('destination') is not None
                    or kwargs.get('prepared_geometry') is not None):
                return function(*args, **kwargs)
            state.checkpoint()
            from ghost_backend.twod.assembly.kernels import mesh_key
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)


            if (params.get('destination') is not None or params.get('operator_outputs') is not None
                    or params.get('prepared_geometry') is not None):
                return function(*args, **kwargs)
            mesh = params.pop("mesh")
            params["k0"] = complex(params["k0"])
            key = (label, mesh_key(mesh), pickle.dumps(params, protocol=4), _assembly_rule_key())
            stats = state.cache_stats
            if key in state.cache:
                value, size = state.cache.pop(key)
                state.cache[key] = (value, size)
                stats["hits"] += 1
                return value
            value = function(*args, **kwargs)
            arrays = [value] if label == "D" else [a for pair in value for a in pair]
            size = sum(a.nbytes for a in arrays if any(a.strides) or hasattr(a, 'row_map'))
            if size <= CACHE_BYTES:
                for a in arrays:
                    a.flags.writeable = False
                while state.cache and stats["bytes"] + size > CACHE_BYTES:
                    _, (_, old) = state.cache.popitem(last=False)
                    stats["bytes"] -= old
                    stats["evictions"] += 1
                state.cache[key] = (value, size)
                stats["bytes"] += size
                stats["stores"] += 1
                stats["peak_bytes"] = max(stats["peak_bytes"], stats["bytes"])
            return value
        return call
    return decorate
