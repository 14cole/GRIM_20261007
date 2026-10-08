"""Capture execution settings for batch planning, manifests, and workers."""
from functools import wraps
import json
import time
from pathlib import Path

from ghost_backend.execution.options import (
    automatic_options, current_options, execution_scope, validate_for_run,
)


def driver_options(namespace):
    """The automatic run profile, bounded by the driver's optional MAX_SOLVE_GB.

    Without one, a GHOST_MAX_SOLVE_GB launch environment still bounds the run.
    """
    budget = namespace.get('MAX_SOLVE_GB')
    if budget is None:
        import os
        raw = os.environ.get('GHOST_MAX_SOLVE_GB', '').strip()
        budget = float(raw) if raw else None
    return automatic_options(budget)


def driver_execution(function):
    """Scope driver planning and worker management to a captured profile."""
    @wraps(function)
    def call(*args, **kwargs):
        profile = None
        if function.__name__ == 'worker':
            directory = args[0] if args else kwargs['run_dir_str']
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            config = manifest['solver_config']
            if config.get('execution_options') is not None:
                profile = validate_for_run(config['execution_options'], config.get('solver_method', 'auto'),
                                           config.get('lu_precision', 'double'))
        if profile is None:
            profile = driver_options(function.__globals__)
        with execution_scope(profile):
            return function(*args, **kwargs)
    return call


def unit_execution(function):
    """Apply manifest settings inside a fresh or reused pool worker."""
    @wraps(function)
    def call(unit, context, destination):
        profile = context.get('execution_options') or current_options() or driver_options(function.__globals__)
        from ghost_backend.execution.metrics import progress_listener
        last = [0.0]
        def progress(event):
            now = time.monotonic()
            if now - last[0] < 10.0:
                return
            last[0] = now
            rss = event.get('process_rss_bytes')
            memory = '; RSS {:.2f} GiB'.format(rss / 1024**3) if rss is not None else ''
            print('  {} | {}{} | {:.1f}s{}'.format(unit.get('name', unit.get('geometry', 'solve')),
                event['phase'] + ': ' if event['phase'] else '', event['stage'],
                event['elapsed_seconds'], memory), flush=True)
        from ghost_backend.execution.selection import batch_selection_scope
        selection=context.get('batch_backend_selection')
        reservation=(selection.get('candidates',{}).get(selection['selected'],{}).get('peak_gb')
                     if selection else None)
        with execution_scope(profile, limit_blas=True,
                             assembly_threads=context.get('execution_assembly_threads'),memory_budget_gib=reservation), \
                batch_selection_scope(selection):
            with progress_listener(progress):
                return function(unit, context, destination)
    return call
