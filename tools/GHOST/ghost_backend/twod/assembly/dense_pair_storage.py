"""Bounded temporary storage for an optional shared cubic Galerkin matrix."""
import numpy as np
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.residual_spool import ResidualSpool


class DensePairSpool(ResidualSpool):
    # There are no resident matrix coefficients between polynomial solves.
    # Four MiB source blocks bound source, projected, and gathered row buffers
    # below the pair's 32 MiB workspace reservation for this <=3-entry map.
    bytes = 0
    block_bytes = 4 * 1024**2
    min_block_rows = 1

    @timed_stage('polynomial_spool_write')
    def __init__(self, matrix, directory, checkpoint):
        self.disk_bytes = int(matrix.nbytes)
        super().__init__(matrix, directory, checkpoint)

    def read_rows(self):
        self.file.seek(0)
        for start, stop in self._rows():
            self.checkpoint()
            count = (stop-start)*self.shape[1]
            block = np.fromfile(self.file, dtype=np.complex128, count=count)
            if block.size != count:
                raise OSError('Shared cubic-matrix spool is incomplete.')
            yield start, stop, block.reshape(stop-start, self.shape[1])

    @timed_stage('polynomial_spool_restore')
    def restore(self):
        self.checkpoint()
        matrix = np.empty(self.shape, dtype=np.complex128, order='F')
        self.restore_into(matrix)
        return matrix


@timed_stage('polynomial_projection')
def project_spooled(spool, p, checkpoint=lambda: None):
    """One sequential read for P.T A P, without retaining the dense fine A."""
    n, m = p.shape
    if spool.shape != (n, n) or np.max(np.diff(p.indptr), initial=0) > 3:
        raise ValueError('Shared cubic projection has an incompatible sparse map.')
    result = np.zeros((m, m), dtype=np.complex128, order='F')
    pt = p.T.tocsr()
    for start, stop, block in spool.read_rows():
        checkpoint()
        local = p[start:stop]
        rows = np.unique(local.indices)
        right = (pt @ block.T).T
        # Only supported coarse rows are materialized. A full M-by-M temporary
        # from P.T[:, start:stop] @ right would defeat bounded storage.
        contribution = local[:, rows].T @ right
        result[rows] += contribution
        block = right = contribution = None
    return result
