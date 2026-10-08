"""Bounded packed block products for an inverse tree's repeated probes.

The accurate coefficient operator is unchanged. One scoped thread team owns
disjoint output groups; each group preserves the serial source-tile order.
Inverse construction enables this only for the explicitly selected randomized
builder. The default ACA builder keeps its original coefficient-access path.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import numpy as np
from ghost_backend.execution.runtime import ScopedValue

WORKSPACE_BYTES = 32 * 1024**2
MIN_ENTRIES = 1024**2
MAX_WORKERS = 4
_ACTIVE = ScopedValue('ghost_inverse_block_products', None)


class ProductTeam:
    def __init__(self):
        self.pool = None
        self.workers = 0
        self.evidence = dict(products=0, chunks=0, teams=0, peak_workspace_bytes=0,
                             workspace_budget_bytes=WORKSPACE_BYTES)

    def close(self):
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def apply(self, operator, rows, cols, x, row_plan, col_plan, trans):
        """Return None when the original partial-tile path is more appropriate."""
        if trans not in (0, 1, 2):
            raise ValueError('Invalid transpose mode')
        x = np.asarray(x)
        expected = len(cols) if trans == 0 else len(rows)
        if x.ndim != 2 or x.shape[0] != expected or not x.shape[1]:
            raise ValueError('Block RHS does not match the requested operation.')
        if min(len(rows), len(cols)) < 512 or len(rows)*len(cols) < MIN_ENTRIES:
            return None
        rp = operator.plan(rows) if row_plan is None else row_plan
        cp = operator.plan(cols) if col_plan is None else col_plan
        incoming, outgoing = (cp, rp) if trans == 0 else (rp, cp)
        from ghost_backend.execution.options import allocated_cpu_budget, effective_assembly_threads, single_thread_blas
        workers = min(MAX_WORKERS, effective_assembly_threads(allocated_cpu_budget()), len(outgoing))
        if workers < 2:
            return None
        input_rows = sum(len(operator.groups[group]) for group, _, _ in incoming)
        largest = max(len(operator.groups[group]) for group, _, _ in incoming + outgoing)
        # Input packing plus each worker's output, product, rank intermediate,
        # and final indexed gather. Dense and compressed tile ranks never exceed
        # their edge. Include one extra gather while packing on the main thread.
        bytes_per_column = 16*(input_rows + (4*workers+1)*largest)
        width = min(x.shape[1], WORKSPACE_BYTES // max(1, bytes_per_column))
        if width < min(8, x.shape[1]):
            return None
        operator.checkpoint()
        if self.pool is None:
            self.workers = workers
            self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ghost-inverse-products')
            self.evidence['teams'] += 1
        # A smaller subsequent allocation must not inherit the larger team.
        if workers < self.workers:
            return None
        result = np.empty((len(rows) if trans == 0 else len(cols), x.shape[1]), complex)
        self.evidence['products'] += 1
        self.evidence['peak_workspace_bytes'] = max(self.evidence['peak_workspace_bytes'], bytes_per_column*width)
        with single_thread_blas():
            for start in range(0, x.shape[1], width):
                stop = min(start+width, x.shape[1])
                rhs = {}
                for group, positions, local in incoming:
                    operator.checkpoint()
                    block = np.zeros((len(operator.groups[group]), stop-start), complex)
                    block[local] = x[positions, start:stop].conj() if trans == 2 else x[positions, start:stop]
                    rhs[group] = block

                def work(plan):
                    group, positions, local = plan
                    out = np.zeros((len(operator.groups[group]), stop-start), complex)
                    for source, _, _ in incoming:
                        operator.checkpoint()
                        if trans == 0:
                            u, v = operator.tiles[group, source]
                            out += u @ rhs[source] if v is None else u @ (v @ rhs[source])
                        else:
                            u, v = operator.tiles[source, group]
                            out += u.T @ rhs[source] if v is None else v.T @ (u.T @ rhs[source])
                    # A.H x = conjugate(A.T conjugate(x)): conjugate only the
                    # RHS/output once, never every retained coefficient factor.
                    result[positions, start:stop] = out[local].conj() if trans == 2 else out[local]

                futures = []
                try:
                    for plan in outgoing:
                        futures.append(self.pool.submit(work, plan))
                    for future in futures:
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    # Complete running tasks before their buffers and BLAS
                    # reservation leave scope, even on cancellation or error.
                    for future in futures:
                        if not future.cancelled():
                            try:
                                future.result()
                            except BaseException:
                                pass
                    raise
                self.evidence['chunks'] += 1
        return result


@contextmanager
def product_scope(enabled=True, blas_threads=None):
    team = ProductTeam()
    with ExitStack() as stack, _ACTIVE.override(team if enabled else None):
        if blas_threads == 1:
            # A refcounted scope allows simultaneous BoR mode workers to
            # overlap safely. A check-and-skip on the current pool count could
            # restore another worker's BLAS setting before that worker exits.
            from ghost_backend.execution.options import single_thread_blas
            stack.enter_context(single_thread_blas())
        elif blas_threads is not None:
            from ghost_backend.execution.thread_control import threadpool_info
            pools = [row for row in threadpool_info() if row.get('user_api') == 'blas']
            if any(int(row.get('num_threads') or 1) != blas_threads for row in pools):
                from ghost_backend.execution.options import current_options, execution_scope, linear_algebra_threads
                stack.enter_context(execution_scope(dict(current_options() or {}, blas_threads=blas_threads)))
                stack.enter_context(linear_algebra_threads())
        try:
            yield team
        finally:
            team.close()


def prepared_product(operator):
    original = operator.block_matmul
    # Preserve arbitrary/custom block-product semantics instead of reaching
    # inside an unrelated operator's storage.
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.polarization_cache import SpooledOperator
    standard = type(operator) in (StreamedOperator, SpooledOperator)
    def apply(rows, cols, x, row_plan=None, col_plan=None, trans=0):
        team = _ACTIVE.get()
        if team is not None and standard and (not isinstance(operator, SpooledOperator) or operator.loaded):
            result = team.apply(operator, rows, cols, x, row_plan, col_plan, trans)
            if result is not None:
                return result
        return original(rows, cols, x, row_plan, col_plan, trans=trans)
    return apply
