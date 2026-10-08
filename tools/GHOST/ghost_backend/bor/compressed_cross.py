"""Bounded rectangular far storage with full original-coefficient checks.

Every tile is sampled by the existing cross-surface quadrature. Only
well-separated tiles without near pairs are eligible for compression, and
every coefficient is checked against its original before retaining factors.
The accepted per-tile relative Frobenius bound also bounds the disjoint full
far matrix. Near integrations are neither approximated nor stored here.
"""
import math
import tempfile
import threading
import weakref

import numpy as np
import scipy.linalg as la

from ghost_backend.bor import streaming as st
from ghost_backend.bor.compressed_far import FAR_COMPRESSION_TOLERANCE, FAR_COMPRESSION_ETA, _Family
from ghost_backend.bor.near_storage import mode_sign


CROSS_TILE_NODES = 32
CROSS_STORAGE_CHUNK_BYTES = 1024**2


class CrossCompressionBudgetError(ValueError):
    """Use the original stream if compact indexing cannot fit its allowance."""


def _verified_component(raw, tolerance, eligible):
    """An owned exact tile or factors checked against all original entries."""
    if not eligible or min(raw.shape) < 8:
        return ('dense', np.array(raw, order='C', copy=True)), 0.
    norm = float(la.norm(raw))
    if not math.isfinite(norm) or not np.all(np.isfinite(raw)):
        raise RuntimeError('Non-finite rectangular BoR far coefficients.')
    if norm == 0.:
        return ('lr', (np.zeros((raw.shape[0], 0), complex),
                       np.zeros((0, raw.shape[1]), complex))), 0.
    try:
        u, singular, vh = la.svd(raw, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        return ('dense', np.array(raw, order='C', copy=True)), 0.
    tail = np.sqrt(np.cumsum((singular**2)[::-1]))[::-1]
    rank = int(np.count_nonzero(tail > tolerance * norm))
    if rank * sum(raw.shape) >= raw.size:
        return ('dense', np.array(raw, order='C', copy=True)), 0.
    left = np.ascontiguousarray(u[:, :rank] * singular[:rank])
    right = np.ascontiguousarray(vh[:rank])
    # SVD's tail estimate proposes a rank; direct comparison independently
    # accepts it. No random-row probe or matrix-solve residual substitutes.
    error = float(la.norm(raw - left @ right) / norm)
    if not math.isfinite(error) or error > tolerance:
        return ('dense', np.array(raw, order='C', copy=True)), 0.
    return ('lr', (left, right)), error


def _close_spool(file, owner):
    try:
        file.close()
    finally:
        owner.release()


class CompressedCrossFarBlocks:
    """Cross-stream interface, preserving admitted ranges and disk spilling.

    A range never retains more numerical payload than its dense counterpart.
    Spilled ranges use one delete-on-close file, so thousands of low-rank
    arrays do not require thousands of files or resident mappings on Windows.
    """
    def __init__(self, cross, m_max, dtype=np.complex128, tile_budget_gb=1.,
                 workers=1, mode_block=None, spill=None, **_):
        if np.dtype(dtype) != np.dtype(np.complex128):
            raise ValueError('Verified rectangular compression requires double precision.')
        self.cross, self.sp, self.sq = cross, cross.sp, cross.sq
        self.families = ('efie', 'ibc') if cross.need_p else ('efie',)
        self.m_max, self.k = int(m_max), complex(cross.k)
        self.Np, self.Nq = self.sp.Nn, self.sq.Nn
        self.go_p, self.go_q = self.sp.gauss_order, self.sq.gauss_order
        self.mode_block = st._aligned_stream_mode_block(self.m_max, mode_block, workers)
        self._near_sources = {e: [] for e in range(self.sp.gen.n_elems)}
        for e, f in cross.near_pairs:
            self._near_sources[e].append(f)
        self._workers = 1
        self._native_threads = st.streaming_tile_threads()
        budget = float(tile_budget_gb) * 1e9
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('Streaming tile budget must be positive and finite.')
        self._work_bytes = min(st.modal_kernels.FFT_BUILD_BUDGET, budget/8)
        rho = max(np.max(self.sp.gen.nodes[:,0]), np.max(self.sq.gen.nodes[:,0]))
        nx = st._n_xi_efie(self.k,rho,self.m_max,cross._far_gap)
        if cross.need_p:
            nx = max(nx, st._n_xi_bracket(self.k,rho,self.m_max,cross._far_gap))
        self._tile_nodes = CROSS_TILE_NODES

        def tile_cost(nodes, modes):
            # Each nodal tile needs at most one extra supporting element per
            # direction. Charge both original contraction buffers and the
            # bounded SVD/reconstruction while the sampled band is retained.
            te,fe = min(nodes+1,self.sp.gen.n_elems),min(nodes+1,self.sq.gen.n_elems)
            return (st.BOR_STREAM_TILE_SLACK * st._banded_tile_bytes(
                te,fe,self.go_p,self.go_q,modes+2,modes,True,cross.need_p,self._work_bytes,nx)
                + 16*nodes*nodes*(4*modes+12))

        while self._tile_nodes > 1 and tile_cost(self._tile_nodes,1) > budget/2:
            self._tile_nodes //= 2
        if tile_cost(self._tile_nodes,1) > budget/2:
            raise CrossCompressionBudgetError('Rectangular compression tile exceeds its work allowance.')
        self._modal_batch = st._largest_fitting(
            lambda modes: tile_cost(self._tile_nodes,modes) <= budget/2,
            min(self.mode_block,self.m_max+1))
        held_modes = self.m_max+1 if spill is not None else self.mode_block
        tiles = math.ceil(self.Np/self._tile_nodes)*math.ceil(self.Nq/self._tile_nodes)
        # Four components per requested family, three int64 values each.
        # The fixed allowance conservatively covers their keys and containers.
        index_upper = tiles*(held_modes*4*len(self.families)*24+2048)
        if index_upper+CROSS_STORAGE_CHUNK_BYTES > budget/4:
            raise CrossCompressionBudgetError('Rectangular compression indexes exceed their work allowance.')
        self._index_upper = index_upper
        self._range_lock, self._file_lock = threading.RLock(), threading.Lock()
        self._file = self._spill = self._finalizer = None
        self._closed = False
        self._blocks = {}
        self._arena = []
        self._payload = self._disk_bytes = self.n_sweeps = 0
        self.lo, self.hi = 1, 0
        self.Z, self.B = _Family('efie'), _Family('ibc') if cross.need_p else None
        self.evidence = dict(backend='verified_rectangular_tiles', tolerance=FAR_COMPRESSION_TOLERANCE,
                             coefficient_check='all_original_coefficients',
                             max_relative_block_error=0., lowrank_components=0, dense_components=0,
                             index_upper_gb=index_upper/1e9, tile_nodes=self._tile_nodes)
        from ghost_backend.bor.options import current_checkpoint
        self._checkpoint = current_checkpoint() or (lambda: None)
        wp = st.far_weights(self.sp.g, self.sp.gen.n_elems, self.go_p, pmchwt=True)
        wq = st.far_weights(self.sq.g, self.sq.gen.n_elems, self.go_q, pmchwt=True)
        self._left_all, self._left_one = wp['left_all'], wp['lv']['1']
        self._right_groups, self._right_one = wq['right_groups'], wq['right_one']
        try:
            if spill is not None:
                self._spill = st._SpillFiles(spill, 'ghost-bor-cross-compressed-')
                self._file = tempfile.TemporaryFile(mode='w+b', dir=self._spill.path)
                self._finalizer = weakref.finalize(self, _close_spool, self._file, self._spill)
                self.mode_block = self.m_max + 1
            self._ensure(0)
        except BaseException:
            self.close()
            raise

    def _eligible(self, I, J):
        e0, e1 = max(0, I[0]-1), min(I[1], self.sp.gen.n_elems)
        f0, f1 = max(0, J[0]-1), min(J[1], self.sq.gen.n_elems)
        if any(f0 <= f < f1 for e in range(e0, e1) for f in self._near_sources[e]):
            return False
        a, b = self.sp.gen.nodes[e0:e1+1], self.sq.gen.nodes[f0:f1+1]
        gap = np.maximum(0., np.maximum(a.min(0)-b.max(0), b.min(0)-a.max(0)))
        diameter = min(float(la.norm(np.ptp(a, axis=0))), float(la.norm(np.ptp(b, axis=0))))
        return float(la.norm(gap)) > 0 and float(la.norm(gap)) >= FAR_COMPRESSION_ETA * diameter

    def _sample(self, family, I, J, lo, hi, eligible):
        e0, e1 = max(0, I[0]-1), min(I[1], self.sp.gen.n_elems)
        f0, f1 = max(0, J[0]-1), min(J[1], self.sq.gen.n_elems)
        rows, re = slice(e0*self.go_p, e1*self.go_p), e1-e0
        modes = np.arange(lo, hi+1)
        if family == 'efie':
            start = max(0,lo-1)
            sampled = st._banded_stream(self, rows, 'g', np.arange(start,hi+2),
                                        (f0,f1), near_free=eligible)
            left = self._left_all[:,:,rows].reshape(2*len(st._LEFT_KINDS), re, self.go_p)
            band = st._efie_band(sampled,left,self._right_groups,modes,start,self.k,
                                 f0,f1,re,self.go_p)
        else:
            sampled = st._banded_stream(self,rows,'ibc',modes,(f0,f1),near_free=eligible)
            band = st._bracket_band(sampled,self._left_one[:,rows].reshape(2,re,self.go_p),
                                    self._right_one,f0,f1,re,self.go_p)
        return band[:,J[0]-f0:J[1]-f0,I[0]-e0:I[1]-e0,:].transpose(3,0,2,1)

    def _store_array(self, array):
        array = np.ascontiguousarray(array)
        offset = self._payload
        self._payload += array.nbytes
        if self._file is None:
            flat = array.reshape(-1)
            position = offset//16
            while flat.size:
                chunk, start = divmod(position,self._chunk_items)
                if chunk == len(self._arena):
                    self._arena.append(np.empty(self._chunk_items,complex))
                count = min(flat.size,self._chunk_items-start)
                self._arena[chunk][start:start+count] = flat[:count]
                flat,position = flat[count:],position+count
            return offset
        self._spill.reserve(array.nbytes)
        offset = self._file.tell()
        try:
            if array.nbytes:
                self._file.write(memoryview(array).cast('B'))
        except OSError as exc:
            raise st.StreamingSpillError('Cannot write compressed cross far blocks: '+str(exc)) from exc
        self._disk_bytes += array.nbytes
        return offset

    def _load_array(self, offset, shape):
        count = math.prod(shape)
        if self._file is None:
            chunk,start = divmod(int(offset)//16,self._chunk_items)
            if count == 0:
                return np.empty(shape,complex)
            if start+count <= self._chunk_items:
                return self._arena[chunk][start:start+count].reshape(shape)
            out = np.empty(count,complex)
            copied = 0
            while copied < count:
                size = min(count-copied,self._chunk_items-start)
                out[copied:copied+size] = self._arena[chunk][start:start+size]
                copied,chunk,start = copied+size,chunk+1,0
            return out.reshape(shape)
        with self._file_lock:
            self._file.seek(offset)
            data = self._file.read(count*16)
        if len(data) != count*16:
            raise st.StreamingSpillError('A compressed cross far-block file was truncated.')
        return np.frombuffer(data, dtype=np.complex128).reshape(shape)

    def _build_range(self, lo, hi):
        from ghost_backend.execution.options import single_thread_blas
        self._blocks = {family: {} for family in self.families}
        self._arena = []
        self._payload = 0
        self._chunk_items = max(1,min(CROSS_STORAGE_CHUNK_BYTES//16,
                                     4*len(self.families)*self.Np*self.Nq*(hi-lo+1)))
        with single_thread_blas():
            for i in range(0,self.Np,self._tile_nodes):
                I = (i,min(i+self._tile_nodes,self.Np))
                for j in range(0,self.Nq,self._tile_nodes):
                    self._checkpoint()
                    J = (j,min(j+self._tile_nodes,self.Nq))
                    eligible = self._eligible(I,J)
                    batch = self._modal_batch
                    for family in self.families:
                        entries = np.empty((hi-lo+1,4,3),np.int64)
                        for start in range(lo,hi+1,batch):
                            self._checkpoint()
                            raw = self._sample(family,I,J,start,min(hi,start+batch-1),eligible)
                            for mode_index,mode in enumerate(raw,start-lo):
                                for uv,component in enumerate(mode):
                                    (kind,value),error = _verified_component(component,FAR_COMPRESSION_TOLERANCE,eligible)
                                    self.evidence['max_relative_block_error'] = max(self.evidence['max_relative_block_error'],error)
                                    self.evidence['lowrank_components' if kind=='lr' else 'dense_components'] += 1
                                    entries[mode_index,uv] = (
                                        (self._store_array(value[0]),self._store_array(value[1]),value[0].shape[1])
                                        if kind=='lr' else (self._store_array(value),0,-1))
                            raw = None
                        self._blocks[family][(I,J)] = entries
        if self._file is not None:
            self._file.flush()
        self.lo,self.hi = lo,hi
        self.n_sweeps += 1
        self.evidence.update(stored_gb=self._payload/1e9,spilled_gb=self._disk_bytes/1e9,
                             resident_numeric_gb=self.memory_gb())

    def _ensure(self, mode):
        if self._closed:
            raise RuntimeError('Compressed cross far blocks were released.')
        if not self.lo <= mode <= self.hi:
            lo = (mode//self.mode_block)*self.mode_block
            self._build_range(lo,min(self.m_max,lo+self.mode_block-1))

    def write_blocks(self, which, m, targets):
        # The production sweep has a mode-range barrier. Keep standalone
        # concurrent callers safe too: descriptors and their arena must belong
        # to the same range for the whole reconstruction/copy operation.
        if which not in self.families:
            raise ValueError('This cross stream was prepared for EFIE only.')
        with self._range_lock:
            self._ensure(abs(m))
            blocks, index = self._blocks[which],abs(m)-self.lo
            for (I,J), entries in blocks.items():
                self._checkpoint()
                rows,cols = I[1]-I[0],J[1]-J[0]
                for uv,(left,right,rank) in enumerate(entries[index]):
                    raw = (self._load_array(left,(rows,cols)) if rank < 0 else
                           self._load_array(left,(rows,int(rank))) @ self._load_array(right,(int(rank),cols)))
                    np.multiply(raw,mode_sign(uv,m),out=targets[uv][slice(*I),slice(*J)])

    def efie_blocks(self,m):
        out = tuple(np.empty((self.Np,self.Nq),complex) for _ in range(4))
        self.write_blocks('efie',m,out)
        return out

    def bracket_blocks(self,m):
        out = tuple(np.empty((self.Np,self.Nq),complex) for _ in range(4))
        self.write_blocks('ibc',m,out)
        return out

    def spilled_gb(self):
        return self._disk_bytes/1e9

    def memory_gb(self):
        indexes = sum(a.nbytes for family in self._blocks.values() for a in family.values())
        return (indexes+sum(a.nbytes for a in self._arena))/1e9

    def close(self):
        self._closed = True
        self._blocks = {}
        self._arena = []
        self.Z = self.B = None
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        elif self._spill is not None:
            self._spill.release()
        self._file = self._spill = None
