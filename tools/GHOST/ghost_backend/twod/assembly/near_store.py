"""Bounded near-block storage preserving the original output accumulation order."""
import shutil
import tempfile
import numpy as np

NEAR_STORAGE_BYTES = 16 * 1024**2
# Enough pairs to keep eight bounded integration batches busy. Coefficient
# records remain streamed through NearStore; this is not an all-pairs cache.
NEAR_INTEGRATION_PAIRS = 8192


class NearStore:
    """Keep small near operators in RAM, otherwise use a private temporary file.

    The near-pair geometry is indexed separately. Keeping coefficient records on
    disk allows each destination to consume all pairs in its original order,
    including reverse K' lookup for D, without retaining all complex blocks.
    Explicit reads (not mmap) keep the process's buffers bounded on both OSes.
    """
    def __init__(self, count, width, kinds, checkpoint=None, budget_bytes=None):
        self.count, self.width, self.kinds = int(count), int(width), tuple(kinds)
        self.checkpoint = checkpoint or (lambda: None)
        self.record_bytes = 16*self.width*self.width
        self.bytes = self.count*self.record_bytes*len(self.kinds)
        budget = NEAR_STORAGE_BYTES if budget_bytes is None else max(0, int(budget_bytes))
        self.file, self.arrays = None, {}
        if self.bytes <= budget:
            self.arrays = {kind: np.zeros((count, width, width), complex) for kind in self.kinds}
        else:
            from ghost_backend.execution.options import temporary_directory
            directory = temporary_directory()
            if shutil.disk_usage(directory).free < self.bytes + 64*1024**2:
                raise OSError('Insufficient temporary disk space for bounded near-operator storage.')
            self.file = tempfile.TemporaryFile(prefix='ghost-near-', suffix='.bin', dir=directory)

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
        self.arrays.clear()

    def _offset(self, kind, start):
        return (self.kinds.index(kind)*self.count + int(start))*self.record_bytes

    def write(self, kind, start, values):
        self.checkpoint()
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim != 3 or values.shape[1:] != (self.width, self.width) or not 0 <= start <= self.count-len(values):
            raise ValueError('Near-block write does not match its storage.')
        if self.file is None:
            target = self.arrays[kind][start:start+len(values)]
            if (values.__array_interface__['data'][0] == target.__array_interface__['data'][0]
                    and values.strides == target.strides):
                return  # the batch integrator already wrote this slice in place
            target[...] = values
        else:
            self.file.seek(self._offset(kind, start))
            self.file.write(memoryview(np.ascontiguousarray(values)).cast('B'))

    def read(self, kind, positions):
        self.checkpoint()
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        if np.any(positions < 0) or np.any(positions >= self.count):
            raise ValueError('Near-block index is out of bounds.')
        if self.file is None:
            return self.arrays[kind][positions]
        result = np.empty((len(positions), self.width, self.width), complex)
        # Sorted contiguous reads are important for reverse-pair queries. Return
        # in request order, including duplicates, without mapping the whole file.
        order = np.argsort(positions, kind='stable')
        sorted_positions = positions[order]
        breaks = np.r_[0, np.flatnonzero(np.diff(sorted_positions) != 1)+1, len(order)]
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            if hi == lo:
                continue
            self.checkpoint()
            self.file.seek(self._offset(kind, sorted_positions[lo]))
            block = np.fromfile(self.file, dtype=np.complex128, count=(hi-lo)*self.width*self.width)
            if block.size != (hi-lo)*self.width*self.width:
                raise OSError('Near-operator temporary storage is incomplete.')
            result[order[lo:hi]] = block.reshape(hi-lo, self.width, self.width)
        return result
