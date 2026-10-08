"""Bounded process preparation of independent, contracted BoR near blocks.

Workers receive geometry and quadrature parameters, never solver objects,
checkpoints, dense matrices, or mutable caches. Results are consumed in pair
order by the owning process. One lazy pool serves the entire preparation.
"""
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys
from pathlib import Path
from functools import lru_cache
from ghost_backend.execution.runtime import (ScopedValue, pin_worker_environment,
                                             release_worker_environment)

_POOL = ScopedValue('ghost_bor_near_process_pool', default=None)
PROCESS_OVERHEAD_BYTES = 256 * 1024**2


@lru_cache(maxsize=16)
def _guarded_main(filename):
    # Spawn imports the entry script. An unguarded top-level solve would run
    # again in every child; keep those direct/interactive callers on threads.
    import ast
    try:
        statements = ast.parse(Path(filename).read_text(encoding='utf-8-sig')).body
    except (OSError,UnicodeError,SyntaxError):
        return False
    for statement in statements:
        test = statement.test if isinstance(statement,ast.If) else None
        if isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Eq):
            left,right = test.left,test.comparators[0]
            if isinstance(left,ast.Constant): left,right = right,left
            if (isinstance(left,ast.Name) and left.id=='__name__' and
                isinstance(right,ast.Constant) and right.value=='__main__'):
                return True
    return False


def process_capable():
    main = sys.modules.get('__main__')
    filename = getattr(main, '__file__', '')
    # A daemonic process (a multiprocessing.Pool worker) may not start
    # children; a concurrent.futures process worker is not daemonic and may,
    # so a driver that runs its units in such workers keeps process
    # preparation inside them.
    return (sys.version_info >= (3, 9) and not getattr(sys, 'frozen', False)
            and not multiprocessing.current_process().daemon
            and bool(filename) and _guarded_main(filename))


def _initialize():
    # Each process does quadrature; nested BLAS teams waste the CPU allocation.
    import numpy
    import scipy.linalg
    from ghost_backend.execution.thread_control import threadpool_limits
    global _limits
    _limits = threadpool_limits(limits=1)


# Spawn/import costs dominate small solves: automatic selection keeps the
# thread path below this many (near pairs) x (prepared modes).
AUTO_PROCESS_WORK_THRESHOLD = 8000


def process_backend_possible(workers):
    """Whether this host and option set can use process workers at all."""
    from ghost_backend.bor.options import current_options
    return (int(workers) > 1 and current_options()['near_backend'] != 'threads'
            and process_capable())


def processes_selected(pair_count=None, mode_tasks=None, workers=None):
    """The one workload policy shared by memory planning and the executor.

    ``mode_tasks`` is the number of prepared modes (cap + 1); ``workers`` is
    accepted for interface compatibility and does not change the policy. An
    unknown workload is not selected in advance: the planner then keeps the
    thread reservation and records the process pool size that was verified
    to fit, which bounds the executor if it meets a large job later.
    """
    from ghost_backend.bor.options import current_options
    backend = current_options()['near_backend']
    if backend == 'processes':
        return True
    if backend != 'auto' or pair_count is None or mode_tasks is None:
        return False
    return int(pair_count) * int(mode_tasks) >= AUTO_PROCESS_WORK_THRESHOLD


@contextmanager
def process_scope(workers, process_workers=None):
    """``process_workers`` is the pool size admitted by the memory plan."""
    admitted = int(workers) if process_workers is None else int(process_workers)
    state = dict(workers=int(workers), process_workers=admitted, executor=None, jobs=0,
                 environment_pinned=False)
    with _POOL.override(state):
        try:
            yield state
        finally:
            try:
                if state['executor'] is not None:
                    state['executor'].shutdown(wait=True, cancel_futures=True)
            finally:
                if state['environment_pinned']:
                    release_worker_environment()


def executor_for(pair_count, mode_cap):
    state = _POOL.get()
    if state is None:
        return None
    if state['executor'] is None:
        if (state['process_workers'] <= 1
                or not process_backend_possible(state['process_workers'])
                or not processes_selected(pair_count, mode_cap+1, state['process_workers'])):
            return None
        # The pool spawns its workers on demand while this scope lasts; they
        # must start with one BLAS thread (see SINGLE_THREAD_WORKER_ENVIRONMENT).
        pin_worker_environment()
        state['environment_pinned'] = True
        state['executor'] = ProcessPoolExecutor(max_workers=state['process_workers'],
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize)
    # Once started, the pool serves every later call of this preparation, small
    # ones included. Its idle workers stay resident until the scope ends; a
    # thread team sized without them would overlap that memory, which the plan
    # prices as alternatives (threads OR processes), never as a sum.
    state['jobs'] += pair_count
    return state['executor']


class NearTask:
    """One near element pair, integrated for the nonnegative modes only.

    Negative modes follow by exact tangential-block parity (near_storage),
    and every consumer retains only ``0..m_max``, so the signed half is never
    contracted here.
    """

    def __init__(self, gp, gq, k, m_max, kinds, depth=4, pair_kind=None,
                 near_order=12, near_rtol=2e-5, near_max_order=192,
                 mode_start=0, return_families=False):
        self.gp, self.gq, self.k = gp, gq, k
        self.m_max, self.kinds, self.depth = m_max, kinds, depth
        self.mode_start, self.return_families = int(mode_start), bool(return_families)
        self.pair_kind = pair_kind
        from ghost_backend.bor.options import current_options
        self.junction_refinement = current_options()['near_refinement']
        self.near_order, self.near_rtol, self.near_max_order = near_order, near_rtol, near_max_order

    def __call__(self, pair):
        return self.run_batch([pair])[0]

    def run_batch(self, pairs):
        """Results of every pair, in order, as separate calls would return them.

        Touching pairs are integrated together, and so are the disjoint pairs
        at each refinement level (``_contract_near_batch`` and
        ``_converged_disjoint_batch``): one graded-rule call per batch instead
        of several per pair, bitwise the same blocks.
        """
        from ghost_backend.bor.solver import (_same_surface_points, _junction_cell_points,
            _contract_near_batch, _converged_disjoint_batch)
        results = [None] * len(pairs)
        touching, disjoint = [], []
        for index, pair in enumerate(pairs):
            e, f = pair
            if self.pair_kind is None:
                cell = ('diag' if e == f else 'corner10' if f == e+1 else 'corner01') if abs(e-f) <= 1 else None
            else:
                cell = self.pair_kind.get(tuple(pair))
            if cell is None:
                disjoint.append((index, pair))
                continue
            # Cross-surface pairs (pair_kind given) touch at a junction and use
            # the refined junction rule, exactly as the in-process path does.
            points = (_same_surface_points(self.gp, e, f, self.kinds, self.depth)
                      if self.pair_kind is None else _junction_cell_points(cell, self.junction_refinement))
            touching.append((index, (self.gp, e, self.gq, f, points)))
        pick = ((lambda blocks: blocks[self.kinds[0]])
                if self.pair_kind is None and not self.return_families else (lambda blocks: blocks))
        if touching:
            outs = _contract_near_batch([job for _, job in touching], self.k, self.m_max,
                                        self.kinds, signed=False, mode_start=self.mode_start)
            for (index, _), blocks in zip(touching, outs):
                results[index] = (pick(blocks), None)
        if disjoint:
            outs = _converged_disjoint_batch(self.gp, self.gq, [pair for _, pair in disjoint],
                self.k, self.m_max, self.kinds, self.near_order, self.near_rtol, self.near_max_order,
                signed=False, mode_start=self.mode_start)
            for (index, _), (blocks, order, error) in zip(disjoint, outs):
                results[index] = (pick(blocks), (order, error))
        return results


def run_near_batch(task, batch):
    """Process-pool entry: one pickled task serves a small batch of pairs."""
    return task.run_batch(batch)
