"""Run-owned spawned tile workers, with bounded retained residency."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

_GENERATION = None


def run_tile(path, task):
    global _GENERATION
    prepared=0.
    from ghost_backend.compressed import tile_processes as tiles
    if _GENERATION != path:
        started=time.perf_counter()
        # Drop mesh/oracle/CPU tables, but retain the separately bounded near
        # moments. Their keys include geometry, wavenumber, direction and rule;
        # compatible P2/P3 generations can therefore reuse the same integrals.
        tiles._WORKER = None
        import gc
        gc.collect()
        tiles._initialize(Path(path).read_bytes())
        _GENERATION = path
        prepared=time.perf_counter()-started
    result,counters=tiles._tile(task)
    if counters.component_seconds is not None:
        counters.component_seconds['prepare']=prepared
    return result,counters


class WorkerPool:
    def __init__(self, workers):
        self.workers = workers
        self.executor = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'))
        self.directory = tempfile.TemporaryDirectory(prefix='ghost-tile-workers-')
        self.generation = 0
        self.payload_bytes = 0
        self.path = None
        self.closed = False

    def prepare(self, payload):
        if self.path is not None:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.generation += 1
        self.path = Path(self.directory.name) / ('oracle-{}.bin'.format(self.generation))
        self.path.write_bytes(payload)
        self.payload_bytes = len(payload)
        return str(self.path)

    @property
    def reserved_bytes(self):
        from ghost_backend.compressed.tile_processes import WORKER_BYTES
        return self.workers * (WORKER_BYTES + 2*self.payload_bytes)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for process in list((getattr(self.executor, '_processes', None) or {}).values()):
                if process.is_alive():
                    process.terminate()
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # Python versions before cancel_futures was added.
                self.executor.shutdown(wait=True)
        finally:
            self.directory.cleanup()


def acquire(workers):
    from ghost_backend.twod.preparation import run_resources
    resources = run_resources()
    if resources is None:
        return None
    previous = resources.get('compressed_tile_workers')
    if previous is not None and (previous.closed or previous.workers != workers):
        previous.close()
        previous = None
    if previous is None:
        previous = WorkerPool(workers)
        resources['compressed_tile_workers'] = previous
    return previous


def retained_bytes():
    from ghost_backend.twod.preparation import run_resources
    resources = run_resources()
    pool = resources.get('compressed_tile_workers') if resources else None
    return pool.reserved_bytes if pool is not None and not pool.closed else 0


def forecast_bytes(dofs, threads, groups=None):
    """Price workers in every phase, including when idle during factorization."""
    from ghost_backend.compressed.tile_processes import MIN_TILES, MAX_WORKERS, WORKER_BYTES
    from ghost_backend.execution.options import environment_value
    import math
    if groups is None:
        groups = 2**int(math.ceil(math.log(max(1., dofs/512.), 2)))
    if mp.current_process().daemon or groups**2 < MIN_TILES:
        return retained_bytes()
    configured = environment_value('GHOST_TILE_PROCESSES', '').strip()
    count = min(MAX_WORKERS, int(threads), int(configured) if configured else int(threads))
    anticipated = max(0, count)*WORKER_BYTES if count >= 2 else 0
    return max(retained_bytes(), anticipated)
