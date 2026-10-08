"""One-pass tile assembly and compressed operator storage."""
import collections
import contextvars
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ghost_backend.twod.assembly.compact import CompactOperator
import scipy.linalg as la
from ghost_backend.linalg.sweep import _qr_basis
from ghost_backend.linalg.hierarchical import spatial_order
from ghost_backend.execution.metrics import timed_stage

# A ceiling, resolved against the active allocation at each operation.
MATMUL_WORKERS=4
MATMUL_THREADED_COLUMNS=8


@timed_stage('compressed_coefficients')
def tile_values(oracle, rows, cols, missing=None):
    """Query only missing polarizations, optionally proposing a verified basis."""
    from ghost_backend.execution.options import option
    sources = list(getattr(oracle, 'oracles', [oracle]))
    missing = list(range(len(sources))) if missing is None else list(missing)
    proposals = {}
    experimental = option('compressed_far_method', 'full') == 'verified_cur'
    if len(missing) == len(sources):
        if experimental and hasattr(oracle, 'propose_fast_far'):
            proposed = oracle.propose_fast_far(rows, cols)
            if proposed is not None:
                proposals = dict(enumerate(proposed if isinstance(proposed, list) else [proposed]))
        values = oracle.get_with_error(rows, cols)
        values = [values] if len(sources) == 1 else values
        return {i: (*values[i], proposals.get(i)) for i in missing}
    result = {}
    for i in missing:
        source = sources[i]
        proposed = source.propose_fast_far(rows, cols) if experimental and hasattr(source, 'propose_fast_far') else None
        result[i] = (*source.get_with_error(rows, cols), proposed)
    return result


def reciprocal_enabled(oracle):
    from ghost_backend.execution.options import environment_value,option
    setting=environment_value('GHOST_COMPRESSED_RECIPROCAL','auto').strip().lower()
    if setting not in ('auto','off'):
        raise ValueError('GHOST_COMPRESSED_RECIPROCAL must be auto or off.')
    if setting=='off' or option('compressed_far_method','full')!='full':return False
    from ghost_backend.compressed.regional_coefficients import PreparedOracle,PairedOracle
    # An overridden query can have additional semantics (including cancellation
    # or specialized coefficients); never bypass a subclass's public method.
    return (type(oracle) is PreparedOracle or
            type(oracle) is PairedOracle and all(type(source) is PreparedOracle for source in oracle.oracles))


def assembly_tasks(oracle,operators):
    """At most two reciprocal tiles per task, with no deferred opposite grid."""
    paired=reciprocal_enabled(oracle)
    groups=operators[0].groups
    for j in range(len(groups)):
        for i in range(j if paired else 0,len(groups)):
            positions=((i,j),(j,i)) if paired and i!=j else ((i,j),)
            yield tuple((r,c,[index for index,op in enumerate(operators) if (r,c) not in op.pilot_tiles])
                        for r,c in positions)


@timed_stage('compressed_coefficients')
def _reciprocal_values(oracle,requests):
    from ghost_backend.compressed.regional_coefficients import reciprocal_values
    return reciprocal_values(oracle,requests)


def tile_batch_values(oracle,groups,task):
    if len(task)>1:
        return _reciprocal_values(oracle,[(groups[i],groups[j],missing) for i,j,missing in task])
    i,j,missing=task[0]
    return [tile_values(oracle,groups[i],groups[j],missing) if missing else {}]


def take_pilot(operator, i, j):
    value = operator.pilot_tiles.pop((i, j), None)
    if value is not None:
        operator.pilot_reuses += 1
    return value


def _thread_budget():
    from ghost_backend.execution.options import allocated_cpu_budget, effective_assembly_threads
    return effective_assembly_threads(allocated_cpu_budget())


def _run_groups(groups, operation, workers):
    """Disjoint output groups; each retains its original tile summation order."""
    workers = min(max(1, int(workers)), _thread_budget(), len(groups))
    if workers < 2:
        for group in groups:
            operation(group)
        return
    from ghost_backend.execution.options import single_thread_blas
    # The outer team owns the cores. In particular, do not keep the widened
    # BLAS team from solve_fields inside each concurrent tile product.
    with single_thread_blas(), ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ghost-groups') as pool:
        futures = [pool.submit(contextvars.copy_context().run, operation, group) for group in groups]
        for future in futures:
            future.result()


def tile_payload(raw, tail, tolerance, method, probe=True):
    """Local scope releases decomposition work before assembling the next tile."""
    if not np.any(raw):
        return (np.empty((len(raw),0),complex), np.empty((0,raw.shape[1]),complex)), raw, tail, True
    norm = max(float(np.linalg.norm(raw)), 1e-300)
    # Strict storage break-even; reject before generating a useless full Q.
    rank_limit = (raw.size - 1) // sum(raw.shape)
    reconstructed = difference = None
    if method == 'qr':
        left = right = None
        if probe and min(raw.shape) >= 128:
            # Distant smooth blocks usually fit a small sampled column space.
            # This only proposes a basis: the full original tile is verified.
            # Larger spatial tiles often need more than sixteen directions.
            # A still-small proposal avoids factoring every column when that
            # first basis is too narrow. Its stricter sampled threshold keeps
            # weak directions until the authoritative complete-tile check.
            width = 64 if min(raw.shape) >= 384 else (48 if min(raw.shape) >= 256 else 16)
            ids = np.linspace(0, raw.shape[1]-1, width).astype(int)
            threshold = tolerance * norm
            if width > 16:
                threshold *= .1 * np.sqrt(width / raw.shape[1])
            candidate, _ = _qr_basis(raw[:, ids], threshold)
            recovery = candidate.conj().T @ raw
            if width > 16:
                # Trim weak directions in the small projected matrix so a
                # broader proposal does not unnecessarily grow retained tiles
                # and downstream factors. The complete tile is still checked.
                reduced, recovery = _qr_basis(recovery, .75 * tolerance * norm)
                candidate = candidate @ reduced
            proposed = candidate @ recovery
            proposal_error = abs(raw - proposed)
            if np.linalg.norm(proposal_error) <= tolerance * norm:
                left, right = candidate, recovery
                reconstructed, difference = proposed, proposal_error
            candidate = recovery = proposed = proposal_error = None
        if left is None:
            left, right = _qr_basis(raw, tolerance * norm, max_rank=rank_limit)
        if left is None:
            return (raw,None), raw, tail, False
    else:
        u,s,v=la.svd(raw,full_matrices=False,check_finite=False)
        energy=np.sqrt(np.cumsum(s[::-1]**2)[::-1])
        rank=int(np.count_nonzero(energy>tolerance*max(np.linalg.norm(s),1e-300)))
        if rank > rank_limit:
            return (raw,None), raw, tail, False
        left=u[:,:rank].copy();right=(s[:rank,None]*v[:rank]).copy()
    if left.nbytes+right.nbytes >= raw.nbytes:
        return (raw,None), raw, tail, False
    if reconstructed is None:
        # The sampled proposal above was already verified against the whole tile.
        reconstructed=left@right
        difference=abs(raw-reconstructed)
        if np.linalg.norm(difference)>tolerance*norm:
            return (raw,None), raw, tail, False
    tail+=difference
    return (left,right), reconstructed, tail, True


class TileWriter:
    """Compress finished tiles on worker threads while the next tile assembles.

    Workers only run pure compression; each result is stored on the caller's
    thread in submission order, so tile order, byte accounting and results are
    identical to the serial loop. At most `depth` tiles wait in memory. Leaving
    the context waits for the workers, so owners may close spools afterwards.
    """
    def __init__(self, workers=2, depth=4):
        # The caller is assembling the next tile while compression runs.
        # Reserve its CPU too; a one-CPU solve stays entirely inline.
        self.budget = _thread_budget()
        self.workers = min(max(0, int(workers)), max(0, self.budget-1))
        self.depth, self.pending, self.pool = max(1, int(depth)), collections.deque(), None
        self._scope = ExitStack()

    def __enter__(self):
        if self.workers:
            from ghost_backend.execution.options import single_thread_blas, cpu_allocation_scope
            try:
                self._scope.enter_context(single_thread_blas())
                # The caller's native far-assembly team must share the same
                # reservation with compression, rather than use every CPU.
                self._scope.enter_context(cpu_allocation_scope(self.budget-self.workers))
                self.pool = self._scope.enter_context(ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='ghost-tiles'))
            except BaseException:
                self._scope.close()
                raise
        return self

    def submit(self, compress, store, *args):
        if self.pool is None:
            store(compress(*args))
            return
        while self.pending and (len(self.pending) >= self.depth or self.pending[0][0].done()):
            future, previous_store = self.pending.popleft()
            previous_store(future.result())
        self.pending.append((self.pool.submit(contextvars.copy_context().run, compress, *args), store))

    def submit_prepared(self, value, store):
        """Keep reused tiles in the same summation order as newly built tiles."""
        if self.pool is None:
            store(value)
            return
        while self.pending and (len(self.pending) >= self.depth or self.pending[0][0].done()):
            future, previous_store = self.pending.popleft()
            previous_store(future.result())
        from concurrent.futures import Future
        future = Future();future.set_result(value)
        self.pending.append((future,store))

    def __exit__(self, kind, value, traceback):
        try:
            while kind is None and self.pending:
                future, store = self.pending.popleft()
                store(future.result())
        finally:
            for future, _ in self.pending:
                future.cancel()
            self.pending.clear()
            self._scope.close()
        return False


class StreamedOperator:
    def __init__(self,oracle,coordinates,tile=128,tolerance=1e-14,budget=512*1024**2,checkpoint=None,compression='qr',assemble=True):
        if not isinstance(tile,(int,np.integer)) or not 1 <= tile <= 1024:
            raise ValueError('Tile size must be an integer in 1..1024.')
        if not np.isfinite(tolerance) or not 0<tolerance<1 or not np.isfinite(budget) or budget<=0:
            raise ValueError('Invalid tolerance or storage budget.')
        if compression not in ('qr','svd'):
            raise ValueError('Compression must be qr or svd.')
        coordinates=np.asarray(coordinates,float)
        if oracle.n<=0 or coordinates.ndim!=2 or coordinates.shape[0]!=oracle.n or not coordinates.shape[1] or not np.all(np.isfinite(coordinates)):
            raise ValueError('Coordinates must be finite and match the nonempty system.')
        self.n=oracle.n;self.bytes=0;self.tiles={};self.checkpoint=checkpoint or (lambda:None)
        self.recycling_identity = getattr(oracle,'recycling_identity',None)
        self.coordinates=coordinates.copy();self.shape=(self.n,self.n)
        self.entries=self.calls=self.max_entries=0
        self.row_error=np.zeros(self.n);self.row_norm=np.zeros(self.n)
        self.column_error=np.zeros(self.n);self.column_norm=np.zeros(self.n)
        self.row_max=np.zeros(self.n)
        order=spatial_order(coordinates,np.arange(self.n),tile)
        self.order=order

        pending=[order];self.groups=[]
        while pending:
            ids=pending.pop()
            if len(ids)<=tile:self.groups.append(ids)
            else:pending.extend((ids[len(ids)//2:],ids[:len(ids)//2]))
        self.group_bounds = np.asarray([(self.coordinates[ids].min(axis=0), self.coordinates[ids].max(axis=0))
                                        for ids in self.groups])
        self.group_id=np.empty(self.n,int);self.local_id=np.empty(self.n,int)
        for i,ids in enumerate(self.groups):self.group_id[ids]=i;self.local_id[ids]=np.arange(len(ids))
        ends=np.cumsum([len(ids) for ids in self.groups])
        self.group_slices=tuple(slice(int(stop-len(ids)),int(stop)) for ids,stop in zip(self.groups,ends))
        self.bytes=sum(a.nbytes for a in (order,self.group_id,self.local_id,self.row_error,self.row_norm,
                                        self.column_error,self.column_norm,self.row_max,self.coordinates,self.group_bounds))
        if self.bytes>budget:raise MemoryError('Compressed operator exceeded its retained-storage cap.')
        self.compressed=0;self.peak_tile=0;self.tolerance=tolerance;self.compression=compression;self.budget=budget
        self.pilot_reuses = 0
        from ghost_backend.compressed.pilots import take
        self.pilot_tiles = take(getattr(oracle, 'pilot_identity', None), self) if assemble else {}
        if not assemble:return
        self.assemble_tiles(oracle)

    def assemble_tiles(self,oracle):
        from ghost_backend.compressed.tile_processes import prepare, compressed_tiles
        workers,payload=prepare(oracle,[self])
        if workers:
            for (result,) in compressed_tiles(oracle,[self],workers,payload,self.checkpoint):
                self.store_tile(result)
            self.finalize(oracle)
            return
        with TileWriter() as writer:
            for task in assembly_tasks(oracle,[self]):
                if hasattr(oracle,'prepare_columns'):oracle.prepare_columns(self.groups[task[0][1]])
                values=tile_batch_values(oracle,self.groups,task)
                for (i,j,_),value in zip(task,values):
                    self.checkpoint()
                    previous=take_pilot(self,i,j)
                    if previous is not None:writer.submit_prepared(previous,self.store_tile)
                    else:writer.submit(self.compress_tile,self.store_tile,i,j,*value[0])
                values=None
        self.finalize(oracle)

    def separated(self, i, j):
        """Attempt a sampled basis only for spatially separated groups."""
        a, b = self.group_bounds[i], self.group_bounds[j]
        gap = np.maximum(0., np.maximum(a[0]-b[1], b[0]-a[1]))
        diameter = max(np.linalg.norm(a[1]-a[0]), np.linalg.norm(b[1]-b[0]))
        return np.linalg.norm(gap) > .5 * diameter

    def add_tile(self,i,j,raw,tail):
        self.store_tile(self.compress_tile(i,j,raw,tail))

    @timed_stage('compressed_tile_compression')
    def compress_tile(self,i,j,raw,tail,proposal=None):
        """Validate and compress one tile without touching shared operator state."""
        self.checkpoint()
        rows,cols=self.groups[i],self.groups[j]
        if raw.shape!=(len(rows),len(cols)) or tail.shape!=raw.shape:
            raise ValueError('Duplicate or incorrectly shaped tile.')
        size=raw.nbytes+tail.nbytes
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(tail)) or np.any(tail<0):
            raise ValueError('Oracle returned invalid coefficients or error bounds.')
        payload,accepted=(raw,None),False
        if i!=j and proposal is not None:
            payload,raw,tail,accepted=proposal.validate(raw,tail,self.tolerance)
        if i!=j and not accepted:
            payload,raw,tail,accepted=tile_payload(raw,tail,self.tolerance,self.compression,probe=self.separated(i,j))
        magnitude=abs(raw)
        sums=(np.sum(magnitude,axis=1),np.max(magnitude,axis=1),np.sum(tail,axis=1),
              np.sum(magnitude,axis=0),np.sum(tail,axis=0))
        return i,j,size,payload,accepted,sums

    def store_tile(self,compressed):
        """Account one compressed tile; callers store tiles in assembly order."""
        i,j,size,payload,accepted,(row_norm,row_max,row_error,column_norm,column_error)=compressed
        if (i,j) in self.tiles:raise ValueError('Duplicate or incorrectly shaped tile.')
        rows,cols=self.groups[i],self.groups[j]
        self.peak_tile=max(self.peak_tile,size)
        self.compressed+=int(accepted)
        self.row_norm[rows]+=row_norm
        self.row_max[rows]=np.maximum(self.row_max[rows],row_max)
        self.row_error[rows]+=row_error
        self.column_norm[cols]+=column_norm
        self.column_error[cols]+=column_error
        self.bytes+=sum(a.nbytes for a in payload if a is not None)
        if self.bytes>self.budget:raise MemoryError('Compressed operator exceeded its retained-storage cap.')
        self.tiles[i,j]=payload

    def finalize(self,oracle):
        if len(self.tiles)!=len(self.groups)**2:raise ValueError('Operator has missing tiles.')
        if not all(np.all(np.isfinite(a)) for a in (self.row_norm,self.row_error,self.column_norm,self.column_error,self.row_max)):
            raise ValueError('Operator norms or coefficient error bounds overflowed.')
        # Reciprocal tasks finish both directions together. Keep every later
        # multiplication and equilibration in canonical column-major tile order.
        self.tiles=dict(sorted(self.tiles.items(),key=lambda item:(item[0][1],item[0][0])))
        self.evidence=dict(unknowns=self.n,tiles=len(self.tiles),compressed_tiles=self.compressed,
            retained_bytes=self.bytes,max_query_bytes=self.peak_tile,geometry_queries=oracle.calls,
            geometry_coefficients=oracle.entries,dropped_routes=oracle.dropped_routes,
            row_error_bound=float(self.row_error.max()),storage_budget=self.budget,compression=self.compression)
        self.evidence['reused_admission_tiles'] = self.pilot_reuses
        if hasattr(self, 'worker_moment_cache'):
            self.evidence['worker_moment_cache'] = self.worker_moment_cache
        if hasattr(self, 'assembly_components'):
            self.evidence['assembly_components'] = self.assembly_components

    def __len__(self):return self.n
    def __matmul__(self,value):return self.matmul(value)
    @property
    def nbytes(self):return self.bytes

    def iter_tiles(self):
        for (i,j),(left,right) in self.tiles.items():
            self.checkpoint()
            yield self.groups[i],self.groups[j],left if right is None else left@right

    def equilibrate(self):
        """One tile pass; row maxima were accumulated during final assembly.

        Column groups run on threads and sum their tiles in the serial order.
        """
        self.iter_tiles().close()  # spooled operators must be loaded first
        row=np.where(self.row_max>0,self.row_max,1.)
        column=np.zeros(self.n);sums=np.zeros(self.n)
        by_column={}
        for i,j in self.tiles:by_column.setdefault(j,[]).append(i)
        def group(j):
            cols=self.groups[j];largest=np.zeros(len(cols));total=np.zeros(len(cols))
            for i in by_column[j]:
                self.checkpoint()
                left,right=self.tiles[i,j]
                magnitude=abs(left if right is None else left@right)/row[self.groups[i],None]
                largest=np.maximum(largest,np.max(magnitude,axis=0))
                total+=np.sum(magnitude,axis=0)
            column[cols]=largest
            sums[cols]=total
        _run_groups(by_column, group, MATMUL_WORKERS)
        column=np.where(column>0,column,1.)
        return row,column,float(np.max(sums/column))

    def get(self,rows,cols):
        rows,cols=CompactOperator._ids(rows,self.n),CompactOperator._ids(cols,self.n)
        return self._get(rows,cols)

    def plan(self,ids):
        """Tile groups of a validated index subset: (group, positions in ids, local tile ids)."""
        if len(ids)==1:
            return [(int(self.group_id[ids[0]]),np.zeros(1,int),self.local_id[ids])]
        group=self.group_id[ids]
        order=np.argsort(group,kind='stable')
        return [(int(group[positions[0]]),positions,self.local_id[ids[positions]])
                for positions in np.split(order,np.flatnonzero(np.diff(group[order]))+1) if len(positions)]

    def _get(self,rows,cols,row_plan=None,col_plan=None):
        """Internal access for index subsets of a validated spatial permutation."""
        if len(rows)*len(cols)*16>16*1024**2:
            raise MemoryError('Coefficient query exceeds the 16 MiB workspace limit.')
        self.entries+=len(rows)*len(cols);self.calls+=1;self.max_entries=max(self.max_entries,len(rows)*len(cols))
        result=np.empty((len(rows),len(cols)),complex)
        row_plan=self.plan(rows) if row_plan is None else row_plan
        col_plan=self.plan(cols) if col_plan is None else col_plan
        for i,ri,local_r in row_plan:
            self.checkpoint()
            single=len(ri)==1
            for j,ci,local_c in col_plan:
                left,right=self.tiles[i,j]
                if single:
                    result[ri[0],ci]=left[local_r[0],local_c] if right is None else left[local_r[0]] @ right[:,local_c]
                else:
                    value=left[np.ix_(local_r,local_c)] if right is None else left[local_r] @ right[:,local_c]
                    result[np.ix_(ri,ci)]=value
        return result

    def block_matmul(self,rows,cols,x,row_plan=None,col_plan=None,trans=0):
        """Apply a subblock or its transpose/adjoint without reconstructing it.

        The inverse builder supplies validated unique tree subsets and reuses
        their plans. Products retain the tile factors; workspace is bounded by
        a tile times the number of probe columns, rather than the whole block.
        """
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode')
        x=np.asarray(x)
        expected=len(cols) if trans==0 else len(rows)
        if x.ndim!=2 or x.shape[0]!=expected or not x.shape[1]:
            raise ValueError('Block RHS does not match the requested operation.')
        row_plan=self.plan(rows) if row_plan is None else row_plan
        col_plan=self.plan(cols) if col_plan is None else col_plan
        if trans:
            result=np.zeros((len(cols),x.shape[1]),complex)
            def adj(a):return a.T if trans==1 else a.conj().T
            for i,ri,local_r in row_plan:
                self.checkpoint()
                local=np.zeros((len(self.groups[i]),x.shape[1]),complex);local[local_r]=x[ri]
                for j,ci,local_c in col_plan:
                    left,right=self.tiles[i,j]
                    value=adj(left) @ local
                    result[ci]+=value[local_c] if right is None else adj(right[:,local_c]) @ value
            return result
        result=np.zeros((len(rows),x.shape[1]),complex)
        for j,ci,local_c in col_plan:
            self.checkpoint()
            local=np.zeros((len(self.groups[j]),x.shape[1]),complex);local[local_c]=x[ci]
            for i,ri,local_r in row_plan:
                left,right=self.tiles[i,j]
                result[ri]+=left[local_r] @ (local if right is None else right @ local)
        return result

    def projection_plan(self,mapping):
        mapping=mapping.tocsr()
        active=np.flatnonzero(np.diff(mapping.indptr))
        result=[]
        for group in np.unique(self.group_id[active]):
            part=mapping[self.groups[int(group)]]
            columns=np.unique(part.indices)
            result.append((int(group),columns,part[:,columns]))
        return result

    def project_sparse(self,left_map,right_map,left_plan=None,right_plan=None):
        """left_map.T A right_map, contracting sparse supports into tile factors.

        In particular, polynomial embeddings have only a few nonzeros per
        fine row. Densifying them into hundreds of RHS columns would discard
        that structure and repeat mostly-zero matrix products.
        """
        self.iter_tiles().close()
        if left_map.shape[0]!=self.n or right_map.shape[0]!=self.n:
            raise ValueError('Sparse projection maps must match the operator.')
        left_plan=self.projection_plan(left_map) if left_plan is None else left_plan
        right_plan=self.projection_plan(right_map) if right_plan is None else right_plan
        result=np.zeros((left_map.shape[1],right_map.shape[1]),complex)
        for i,ri,p in left_plan:
            self.checkpoint()
            for j,ci,q in right_plan:
                u,v=self.tiles[i,j]
                projected=np.asarray(p.T @ u)
                if v is None:
                    value=np.asarray(q.T @ projected.T).T
                else:
                    value=projected @ np.asarray(q.T @ v.T).T
                result[np.ix_(ri,ci)]+=value
        return result

    def matmul(self,b,trans=0):
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode')
        b=np.asarray(b);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=self.n or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid operator RHS.')
        # Gather each RHS once, then use contiguous group views for every
        # product. Repeated b[group] indexing used to copy each tile's RHS.
        rhs=b[self.order]
        ordered=np.zeros_like(rhs,dtype=complex)
        slices=self.group_slices
        def adj(a):return a.T if trans==1 else a.conj().T
        workers=min(MATMUL_WORKERS,_thread_budget()) if b.shape[1]>=MATMUL_THREADED_COLUMNS else 1
        if workers<2:
            for (i,j),(left,right) in self.tiles.items():
                self.checkpoint()
                if trans==0:
                    ordered[slices[i]]+=left@rhs[slices[j]] if right is None else left@(right@rhs[slices[j]])
                else:
                    ordered[slices[j]]+=adj(left)@rhs[slices[i]] if right is None else adj(right)@(adj(left)@rhs[slices[i]])
        else:
            outputs={}
            for i,j in self.tiles:outputs.setdefault(i if trans==0 else j,[]).append((i,j))
            def group(first):
                value=ordered[slices[first]]
                for i,j in outputs[first]:
                    self.checkpoint()
                    left,right=self.tiles[i,j]
                    if trans==0:
                        value+=left@rhs[slices[j]] if right is None else left@(right@rhs[slices[j]])
                    else:
                        value+=adj(left)@rhs[slices[i]] if right is None else adj(right)@(adj(left)@rhs[slices[i]])
            # Futures return no array: finished groups live only in ordered.
            _run_groups(outputs,group,workers)
        # Reuse the private gathered RHS buffer when its dtype permits, so the
        # common complex128 solve needs only two full RHS-sized work buffers.
        result=rhs if rhs.dtype==ordered.dtype else np.empty_like(ordered)
        result[self.order]=ordered
        return result[:,0] if vector else result
