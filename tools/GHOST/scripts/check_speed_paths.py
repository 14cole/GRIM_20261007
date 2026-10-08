"""Report which GHOST speed paths this machine can use, and which it would fall back from.

Run with the Python that runs GHOST, from anywhere:

    python scripts/check_speed_paths.py

It imports the checkout this script belongs to.  Each line is OK or SLOW; the
exit status is 1 when anything is SLOW.  What a given solve actually used is
recorded in its result metadata (see BOR_PERFORMANCE.md, "Which paths ran").
"""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def checks():
    """``[(label, ok, detail)]`` for every speed path that has a slower fallback."""
    sys.path.insert(0, str(ROOT))
    rows = []

    from ghost_backend.bor import streaming, kernels
    backend = streaming.sampling_backend_name()
    rows.append(('BoR far sampler', backend == 'banded_native_pairs',
                 backend if backend == 'banded_native_pairs' else
                 backend + ': NumPy fallback, rebuild with bor/native/build_kernel.py'))
    names = ('near_green', 'near_brackets_stable', 'parity_moments',
             'near_green_rule', 'near_brackets_rule')
    missing = [name for name in names if kernels._native_entry(name) is None]
    rows.append(('BoR near kernels', not missing,
                 'native' if not missing else 'NumPy fallback for ' + ', '.join(missing)))

    from ghost_backend.twod.assembly.native import far, table
    for label, library in (('2D far library', far._dll()), ('2D table library', table.library())):
        rows.append((label, library is not None,
                     'loaded' if library is not None else 'not loadable here (NumPy fallback)'))

    from ghost_backend.execution.options import blas_core_budget, physical_core_count
    from ghost_backend.execution.thread_control import threadpool_info
    pools = [pool for pool in threadpool_info() if pool.get('user_api') == 'blas']
    optimized = [pool for pool in pools if pool.get('internal_api') in ('openblas', 'mkl', 'blis')]
    budget = blas_core_budget()
    threads = max([int(pool.get('num_threads') or 0) for pool in optimized] or [0])
    rows.append(('BLAS', bool(optimized) and threads >= budget,
                 ('{} with {} threads ({} physical cores, dense algebra uses {})'.format(
                     optimized[0]['internal_api'], threads, physical_core_count(), budget)
                  if optimized else 'no optimized BLAS: dense algebra runs on reference BLAS')))

    # Settings that switch a fast path off for every solve of this environment.
    settings = []
    if os.environ.get('GHOST_HIERARCHICAL_MIN_UNKNOWNS', '').strip() == '0':
        settings.append('GHOST_HIERARCHICAL_MIN_UNKNOWNS=0 keeps large dense systems on LU')
    if os.environ.get('GHOST_CPU_FACTORIZATION', '').strip().lower() == 'compressed':
        settings.append('GHOST_CPU_FACTORIZATION=compressed forces the compressed 2-D backend')
    rows.append(('Environment', not settings, '; '.join(settings) or 'no fast path disabled'))

    present = importlib.util.find_spec('ghost_backend.bor.compressed_far') is not None
    rows.append(('September 25 speedups', present,
                 'present' if present else 'this checkout predates them: sync the current main'))
    return rows


def main():
    rows = checks()
    print('GHOST at {} (Python {})'.format(ROOT, sys.version.split()[0]))
    for label, ok, detail in rows:
        print('{:<22} {:<5} {}'.format(label, 'OK' if ok else 'SLOW', detail))
    return 0 if all(ok for _, ok, _ in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
