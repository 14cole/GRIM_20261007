"""Disk-backed original coefficients for full, bounded-memory residuals."""
import shutil
import tempfile
import weakref
import threading
from contextlib import contextmanager
import numpy as np


class ResidualSpool:
    block_bytes = 16 * 1024**2

    def __init__(self, matrix, directory, checkpoint):
        self.shape, self.dtype, self.ndim = matrix.shape, matrix.dtype, 2
        self.checkpoint = checkpoint
        if shutil.disk_usage(directory).free < matrix.nbytes + 64*1024**2:
            raise OSError('Insufficient temporary disk space for full original-matrix residuals.')
        self.file = tempfile.TemporaryFile(prefix='ghost-residual-', suffix='.bin', dir=directory)
        self._cleanup = weakref.finalize(self, self.file.close)
        try:
            for start, stop in self._rows():
                self.checkpoint()
                block = np.ascontiguousarray(matrix[start:stop])
                self.file.write(memoryview(block).cast('B'))
            self.file.flush()
        except BaseException:
            self.close()
            raise

    # Row blocks of at least this many rows: OpenBLAS products of fewer rows
    # can take seconds on many threads (see execution.options.blas_core_budget).
    min_block_rows = 256

    def _block_rows(self):
        return max(self.min_block_rows, self.block_bytes // (16*self.shape[1]))

    @property
    def buffer_bytes(self):
        """Bytes of one row block read back from the spool."""
        return 16 * self.shape[1] * min(self.shape[0], self._block_rows())

    def _rows(self):
        count = self._block_rows()
        for start in range(0, self.shape[0], count):
            yield start, min(start+count, self.shape[0])

    def __len__(self):
        return self.shape[0]

    def __matmul__(self, rhs):
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.shape[1]:
            raise ValueError('Residual RHS does not match the original matrix.')
        result = np.empty((self.shape[0],rhs.shape[1]),complex)
        self.file.seek(0)
        for start, stop in self._rows():
            self.checkpoint()
            block = np.fromfile(self.file,dtype=np.complex128,count=(stop-start)*self.shape[1])
            if block.size != (stop-start)*self.shape[1]:
                raise OSError('Original-matrix residual spool is incomplete.')
            result[start:stop] = block.reshape(stop-start,self.shape[1]) @ rhs
        return result[:,0] if vector else result

    def close(self):
        self._cleanup()

    def restore_into(self, matrix):
        """Restore an owned assembly buffer needed by the next polarization."""
        self.file.seek(0)
        for start, stop in self._rows():
            self.checkpoint()
            block = np.fromfile(self.file,dtype=np.complex128,count=(stop-start)*self.shape[1])
            if block.size != (stop-start)*self.shape[1]:
                raise OSError('Original-matrix residual spool is incomplete.')
            matrix[start:stop] = block.reshape(stop-start,self.shape[1])


# 'auto' keeps the original coefficients next to their LU whenever that copy
# fits within host headroom and the solve reservation. A hierarchical plan
# may not price a full LU copy, so fallback must check again. Each residual
# product of a spooled matrix re-reads the whole file: the certified 10 GHz
# airfoil (7.8 GB system)
# spent 64 s of 207 s re-reading it about 20 times, and ran in 145 s kept in
# memory within its 16.7 GB estimate. A copy that fits stays in memory; an
# explicit reservation also applies to matrices below the disk-I/O threshold.
AUTO_SPOOL_MIN_BYTES = 512 * 1024**2
COPY_MARGIN_BYTES = 256 * 1024**2
_COPY_LOCK = threading.Lock()
_PENDING_WORKSPACE_BYTES = 0


@contextmanager
def reserve_factor_workspace(nbytes):
    """Account for unfinished hierarchical builds during matrix-copy admission.

    Retaining the full reservation until a build finishes is conservative:
    resident partial factors may also appear in RSS. It prevents simultaneous
    builders from spending each other's remaining factor workspace on copies.
    """
    global _PENDING_WORKSPACE_BYTES
    with _COPY_LOCK:
        _PENDING_WORKSPACE_BYTES += int(nbytes)
    try:
        yield
    finally:
        with _COPY_LOCK:
            _PENDING_WORKSPACE_BYTES -= int(nbytes)


def copy_fits(nbytes, extra_bytes=0):
    """Check host headroom and the remaining solve/scheduler reservation.

    The original matrix is already resident. Include current process residency
    (conservatively including concurrent work) before admitting an additional
    LU copy; a host with free RAM does not enlarge a solve's reservation.
    Unknown availability counts as not fitting.
    """
    from ghost_backend.twod.solver import _process_rss_bytes, _solve_memory_limit_gb
    try:
        resident = max(int(nbytes), _process_rss_bytes())
        from ghost_backend.compressed.worker_pool import retained_bytes
        resident += retained_bytes()
        available = float(_solve_memory_limit_gb(resident / 1024**3)) * 1024**3 - resident
    except Exception:
        return False
    required = nbytes + max(COPY_MARGIN_BYTES, nbytes // 8)
    required += int(extra_bytes) + _PENDING_WORKSPACE_BYTES
    return available > 0 and available >= required


def require_copy_capacity(nbytes):
    """Fail before allocating an unadmitted fallback matrix copy."""
    if not copy_fits(nbytes):
        raise MemoryError('Dense LU and its original-matrix residual copy do not fit the '
                          'remaining solve RAM budget. Use an owned matrix with automatic '
                          'or disk residual storage, or increase the RAM reservation.')


def copy_for_lu(matrix):
    """Admit and materialize one LU copy before another worker can admit its own.

    A check alone races across BoR mode workers: several can all observe the
    same free RAM before any copy is resident. Only the bandwidth-bound copy
    holds this lock; the much longer LAPACK factorization remains parallel.
    """
    with _COPY_LOCK:
        require_copy_capacity(matrix.nbytes)
        return np.array(matrix, dtype=np.complex128, order='F', copy=True)


def auto_spooled(nbytes):
    """The 'auto' residual-storage decision for an eligible owned matrix."""
    from ghost_backend.execution.options import allocated_memory_budget, environment_value
    # Explicit reservations apply even below the usual disk-I/O threshold.
    reserved = allocated_memory_budget() is not None or bool(environment_value('GHOST_MAX_SOLVE_GB', '').strip())
    return (reserved or nbytes >= AUTO_SPOOL_MIN_BYTES) and not copy_fits(nbytes)


def selected(matrix, owned, factor_mode):
    from ghost_backend.execution.options import option
    policy = option('dense_residual_storage','auto')
    # 'dense' and 'auto' factor large systems hierarchically; an LU fallback
    # that finds no room for its copy spools the original like any other.
    eligible = (owned and factor_mode in ('dense', 'auto') and matrix.flags.owndata
                and matrix.flags.f_contiguous and matrix.flags.writeable)
    return eligible and (policy == 'disk' or policy == 'auto' and auto_spooled(matrix.nbytes))
