"""Galerkin projection of an already qualified compressed operator.

The caller supplies compatible trial/test spaces and retains ownership of the
fine operator. This module neither assumes nested interpolation nodes nor
discards the fine operator's coefficient-error evidence.
"""
import hashlib
import collections
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse

from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.execution.metrics import timed_stage


class ProjectedOracle:
    dropped_routes=0

    def __init__(self,fine,prolongation):
        self.fine=fine
        self.p=sparse.csc_matrix(prolongation,dtype=np.complex128,copy=True)
        self.p.sum_duplicates();self.p.eliminate_zeros();self.p.sort_indices()
        if (self.p.shape[0]!=len(fine) or not self.p.shape[1] or
                not np.all(np.isfinite(self.p.data)) or np.any(np.diff(self.p.indptr)==0)):
            raise ValueError('Prolongation must be finite with one nonempty column per coarse unknown.')
        self.n=self.p.shape[1];self.calls=self.entries=0
        self.plans={};self.lock=threading.Lock()
        digest=hashlib.sha256()
        for value in (self.p.indptr,self.p.indices,self.p.data):digest.update(value.tobytes())
        previous=getattr(fine,'recycling_identity',None)
        self.recycling_identity=(previous,digest.digest()) if previous is not None else None

    def prepare_groups(self,groups):
        # Each sparse embedding is shared by an entire tile row/column.
        # Precompute it once, before concurrent read-only contractions begin.
        self.plans={id(group):(self.p[:,group],None) for group in groups}
        self.plans={key:(mapping,self.fine.projection_plan(mapping))
                    for key,(mapping,_) in self.plans.items()}

    def get_with_error(self,rows,cols):
        self.fine.checkpoint()
        left,left_plan=self.plans.get(id(rows),(None,None))
        right,right_plan=self.plans.get(id(cols),(None,None))
        if left is None:left=self.p[:,rows]
        if right is None:right=self.p[:,cols]
        # Small sparse supports map each coarse trial/test basis into the
        # actual fine basis. Contract directly into the retained tile factors;
        # neither a fine submatrix nor a dense prolongation is materialized.
        result=self.fine.project_sparse(left,right,left_plan,right_plan)
        with self.lock:
            self.calls+=1;self.entries+=result.size
        # Incoming coefficient errors are propagated at operator level below.
        # Complete-tile checks here account for the new storage compression.
        return result,np.zeros(result.shape,float)


def projection_workers():
    from ghost_backend.compressed.operator import _thread_budget
    return min(4,max(1,_thread_budget()))


def projection_workspace_bytes(tile=512):
    # One bounded tile task per worker: sparse contraction, raw/error arrays,
    # QR/reconstruction scratch, and the completed payload awaiting its store.
    # This covers the <=512 tile used by the polynomial-pair runtime.
    return int((32+48*projection_workers()*max(1.,(tile/512)**2))*1024**2)


def _assemble_projection(operator,oracle):
    from ghost_backend.execution.options import single_thread_blas
    oracle.prepare_groups(operator.groups)
    jobs=((i,j) for j in range(len(operator.groups)) for i in range(len(operator.groups)))
    workers=projection_workers()
    def work(i,j):
        raw,tail=oracle.get_with_error(operator.groups[i],operator.groups[j])
        return operator.compress_tile(i,j,raw,tail)
    if workers==1:
        for i,j in jobs:operator.store_tile(work(i,j))
    else:
        pending=collections.deque()
        with single_thread_blas(),ThreadPoolExecutor(max_workers=workers,thread_name_prefix='ghost-project') as pool:
            try:
                for i,j in jobs:
                    if len(pending)>=workers:operator.store_tile(pending.popleft().result())
                    pending.append(pool.submit(contextvars.copy_context().run,work,i,j))
                while pending:operator.store_tile(pending.popleft().result())
            finally:
                for future in pending:future.cancel()
    operator.finalize(oracle)
    operator.evidence.update(projection_workers=workers,
        projection_workspace_bytes=projection_workspace_bytes(max(map(len,operator.groups))))


@timed_stage('compressed_order_projection')
def project_operator(fine,prolongation,coordinates,budget,checkpoint=None,tile=512,spool_directory=None):
    """Build P.T A P and carry conservative incoming row/column error sums.

    For E bounding the fine coefficient error, row sums of |P.T| E |P|
    are bounded by |P.T| row_sums(E) * ||P||_inf. Column sums follow the
    same formula with column_sums(E). This does not treat sampled errors as
    deterministic bounds; it propagates the fine operator's existing contract.
    """
    oracle=ProjectedOracle(fine,prolongation)
    absolute=abs(oracle.p)
    amplification=float(np.max(np.asarray(absolute.sum(axis=1))))
    incoming_rows=np.asarray(absolute.T @ fine.row_error).reshape(-1)*amplification
    incoming_columns=np.asarray(absolute.T @ fine.column_error).reshape(-1)*amplification
    if not (np.all(np.isfinite(incoming_rows)) and np.all(np.isfinite(incoming_columns))):
        raise ValueError('Projected coefficient-error bounds overflowed.')
    cls=StreamedOperator;extra={}
    if spool_directory is not None:
        from ghost_backend.compressed.polarization_cache import SpooledOperator
        cls=SpooledOperator;extra['directory']=spool_directory
    operator=cls(oracle,coordinates,tile=tile,budget=budget,assemble=False,
                 checkpoint=checkpoint or fine.checkpoint,**extra)
    try:
        _assemble_projection(operator,oracle)
    except BaseException:
        if hasattr(operator,'close'):operator.close()
        raise
    operator.row_error+=incoming_rows;operator.column_error+=incoming_columns
    if not (np.all(np.isfinite(operator.row_error)) and np.all(np.isfinite(operator.column_error))):
        if hasattr(operator,'close'):operator.close()
        raise ValueError('Projected coefficient-error bounds overflowed.')
    operator.evidence.update(geometry_queries=0,geometry_coefficients=0,
        projected_coefficient_queries=oracle.calls,projected_coefficients=oracle.entries,
        projected_from_unknowns=len(fine),incoming_error_amplification=amplification,
        incoming_row_error_bound=float(np.max(incoming_rows)),
        incoming_column_error_bound=float(np.max(incoming_columns)),
        row_error_bound=float(np.max(operator.row_error)),
        construction='qualified_fine_galerkin_projection')
    return operator
