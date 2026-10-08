"""One exact far build on disk; bounded ordinary-array modal read windows.

Temporary maps are used only during construction, where StreamingFarBlocks
already flushes and releases completed node rows. They are all closed before
the first modal operator is published. Solves use seek/readinto, so touching
new modes cannot accumulate resident pages of an all-mode memory mapping.
"""
import tempfile
import weakref
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby

import numpy as np

from ghost_backend.bor.streaming import StreamingFarBlocks, StreamingSpillError, _aligned_stream_mode_block


def _close_files(files, owner):
    for handle, _shape, _dtype in list(files.values()):
        handle.close()
    files.clear()
    if owner is not None:
        owner.release()


class SpooledFarBlocks(StreamingFarBlocks):
    def __init__(self, solver, m_max, *, mode_block=None, workers=1, spill=None, **kwargs):
        if spill is None:
            raise ValueError('Exact far spooling requires an admitted disk directory.')
        self._files = {}
        self._building = True
        self._file_finalizer = None
        self._reloads = 0
        block = _aligned_stream_mode_block(m_max, mode_block, workers)
        self._window_modes = block
        super().__init__(solver,m_max,mode_block=mode_block,workers=workers,spill=spill,**kwargs)
        try:
            # Every worker has joined, and no caller has yet received a view.
            # Closing construction maps leaves only the owning ordinary files.
            for array in (self.Z,self.K,self.B):
                if array is not None:
                    array.flush()
                    array._mmap.close()
            self.Z = self.K = self.B = None
            self._building = False
            self.mode_block = block
            self.lo,self.hi = 1,0
            self._file_finalizer = weakref.finalize(self,_close_files,self._files,self._spill)
            self._ensure(0)
        except BaseException:
            self.close()
            raise

    def _allocate(self,name,shape):
        handle = tempfile.TemporaryFile(mode='w+b',dir=self._spill.path)
        dtype = np.dtype(self.dtype)
        shape = tuple(map(int,shape))
        try:
            value = np.memmap(handle,dtype=dtype,mode='w+',shape=shape)
        except BaseException:
            handle.close()
            raise
        self._files[name] = (handle,shape,dtype)
        self._spilled_bytes += int(np.prod(shape))*dtype.itemsize
        return value

    def _build_range(self,lo,hi):
        # Build only one test-row group at a time. An arbitrarily slow tile
        # must not leave many later completed row groups resident in maps.
        # The mapped live rows share the same allowance as the read window;
        # their lifetimes do not overlap.
        entries = ((4*self._efie_entries() if self._efie else 0)
                   +4*self.Nn*self.Nn*(int(self._mfie)+int(self._has_ibc)))
        budget = entries*self._window_modes*np.dtype(self.dtype).itemsize
        row_bytes = 4*self.Nn*(hi-lo+1)*np.dtype(self.dtype).itemsize*sum(
            (self._efie,self._mfie,self._has_ibc))
        rows = budget//max(1,row_bytes)-1  # one shared node row
        if rows < 1:
            raise StreamingSpillError('Exact far spool row workspace exceeds the admitted modal window.')
        self._tile_rows = min(self._tile_rows,int(rows))
        return super()._build_range(lo,hi)

    def _run_build_tiles(self,tiles,operation):
        from ghost_backend.execution.options import single_thread_blas
        if self._workers <= 1:
            for tile in tiles:
                self._checkpoint_spool()
                operation(tile)
            return
        with single_thread_blas(), ThreadPoolExecutor(max_workers=self._workers) as pool:
            for _row, group in groupby(tiles,key=lambda tile:tile[0]):
                self._checkpoint_spool()
                for _ in pool.map(operation,group):
                    pass

    def _read_family(self,name,lo,hi):
        if name not in self._files:
            return None
        handle,shape,dtype = self._files[name]
        block = np.empty((shape[0],hi-lo+1)+shape[2:],dtype=dtype)
        per_mode = int(np.prod(shape[2:]))*dtype.itemsize
        for uv in range(shape[0]):
            self._checkpoint_spool()
            handle.seek((uv*shape[1]+lo)*per_mode)
            destination = memoryview(block[uv]).cast('B')
            offset = 0
            # Bound cancellation latency even for a very large one-mode cache.
            while offset < len(destination):
                self._checkpoint_spool()
                count = handle.readinto(destination[offset:offset+8*1024**2])
                if not count:
                    raise IOError('Exact BoR far-coefficient spool ended unexpectedly.')
                offset += count
        return block

    def _checkpoint_spool(self):
        checkpoint = getattr(self.solver,'_checkpoint',None)
        if checkpoint is not None:
            checkpoint()

    def _ensure(self,mode):
        if self._building:
            return super()._ensure(mode)
        with self._range_lock:
            if self._closed:
                raise RuntimeError('Exact BoR far spool was closed.')
            mode = abs(int(mode))
            if not 0 <= mode <= self.m_max:
                raise ValueError('Requested mode exceeds the exact far spool cap.')
            if self.lo <= mode <= self.hi:
                return
            lo = mode//self.mode_block*self.mode_block
            hi = min(lo+self.mode_block-1,self.m_max)
            self.lo,self.hi = 1,0
            self.Z = self.K = self.B = None
            try:
                self.Z = self._read_family('efie',lo,hi)
                self.K = self._read_family('mfie',lo,hi)
                self.B = self._read_family('ibc',lo,hi)
            except BaseException:
                self.close()
                raise
            self._sidx = {m:m-lo for m in range(lo,hi+1)}
            self._positive_modes = np.arange(lo,hi+1)
            self.lo,self.hi = lo,hi
            self._reloads += 1
            self.evidence = dict(backend='exact_spooled_modal_windows',
                coefficient_check='unchanged_production_coefficients',
                mode_block=self.mode_block,mode_range=[lo,hi],reloads=self._reloads,
                stored_gb=self.memory_gb(),spilled_gb=self.spilled_gb())

    def close(self):
        arrays = (getattr(self,'Z',None),getattr(self,'K',None),getattr(self,'B',None))
        self.Z = self.K = self.B = None
        for array in arrays:
            mapping = getattr(array,'_mmap',None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        finalizer,self._file_finalizer = self._file_finalizer,None
        if finalizer is not None:
            finalizer()
        else:
            _close_files(self._files,getattr(self,'_spill',None))
        super().close()
