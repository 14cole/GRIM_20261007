"""Bounded spawned frequency workers for desktop 2-D and BoR solves.

Each worker owns one numerical solve. Desktop recovery runs save completed
fields directly in their unique run folder and return only completion metadata.
The API also supports in-memory results and legacy explicit checkpoints.
CPU reservations exclude nested tile pools; RAM includes idle interpreters,
active solves and any retained results.
"""
import multiprocessing as mp
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


WORKER_GIB = .5
AUTO_MAX_WORKERS = 2
AUTO_MIN_COST = 8.
# The BoR score is N^2 times the modal cap, in millions, rather than the
# 2-D selector's factorization cost. Four 128-panel frequencies score ~4.
AUTO_MIN_BOR_COST = 1.
_STOP = None
_PROGRESS = None
_ALLOWED_SOLVERS = ('solve_monostatic_rcs_2d', 'solve_monostatic_rcs_2d_survey',
                    'solve_monostatic_rcs_2d_certified')
_BOR_SOLVERS = ('solve_monostatic_rcs_bor', 'solve_monostatic_rcs_bor_survey',
                'solve_monostatic_rcs_bor_certified')


def _initialize(stop, progress):
    global _STOP, _PROGRESS
    _STOP, _PROGRESS = stop, progress
    # Frequency workers own the CPU share. Do not multiply it by another pool.
    os.environ['GHOST_TILE_PROCESSES'] = '0'


def _compute(payload):
    recovery = None
    if payload.get('recovery_directory'):
        from ghost_backend.execution.recovery import RecoveryRun
        recovery = RecoveryRun.open(payload['recovery_directory'])
        recovery.verify_inputs()
    is_bor = payload['solver'] in _BOR_SOLVERS
    if is_bor:
        from ghost_backend.bor import dispatch as solver
    else:
        from ghost_backend.twod import solver
    from ghost_backend.twod.checkpoints import FrequencyCheckpoints
    from ghost_backend.twod.preparation import preparation_scope, sweep_mesh_scope
    from ghost_backend.twod.samples import compact_samples
    from ghost_backend.execution.options import execution_scope
    from ghost_backend.execution.selection import batch_selection_scope
    from ghost_backend.linalg.refined_lu import linear_precision
    if payload['solver'] not in _ALLOWED_SOLVERS + _BOR_SOLVERS:
        raise ValueError('Unsupported frequency worker entry point.')
    frequency = payload['frequency']
    last_report = [0.]
    def report(done, total, message):
        now = time.monotonic()
        if now - last_report[0] >= .15 or done == total:
            try:
                _PROGRESS.put_nowait((frequency, int(done), int(total), str(message)))
            except queue.Full:
                pass
            last_report[0] = now
    captured = recovery.arguments(payload['arguments']) if recovery is not None else payload['arguments']
    arguments = dict(captured, frequencies_ghz=[frequency],
                     abort_event=_STOP, progress_callback=report)
    if is_bor:
        # Each frequency's modal/near workers share this process's CPU grant.
        arguments['workers'] = min(int(arguments.get('workers') or payload['cpus']), payload['cpus'])
        arguments['bor_options'] = payload['options']
    with preparation_scope(), sweep_mesh_scope(payload['frequencies']), compact_samples(), \
            linear_precision(payload['precision']), \
            execution_scope({} if is_bor else payload['options'], limit_blas=True,
                            assembly_threads=payload['cpus'], memory_budget_gib=payload['memory_gib']), \
            batch_selection_scope(payload['selection']):
        result = getattr(solver, payload['solver'])(**arguments)
    profile = result.get('metadata', {}).get('runtime_profile')
    result.setdefault('metadata', {})['frequency_worker'] = dict(
        cpus=payload['cpus'], memory_reservation_gib=payload['memory_gib'])
    if recovery is not None:
        recovery.save(frequency, result)
        return dict(frequency=frequency, profile=profile, result=None, warning=None)
    if payload.get('directory') is None:
        return dict(frequency=frequency, profile=profile, result=result, warning=None)
    try:
        store = FrequencyCheckpoints(payload['directory'], payload['identity'], payload['certified'])
        store.save(frequency, result)
    except OSError as exc:
        return dict(frequency=frequency, profile=profile, result=result,
                    warning='Frequency {:g} GHz computed but checkpoint could not be saved: {}'.format(frequency, exc))
    return dict(frequency=frequency, profile=profile, result=None, warning=None)


def _solver_name(solve):
    from ghost_backend.twod import solver
    name = getattr(solve, '__name__', '')
    if name in _ALLOWED_SOLVERS and getattr(solver, name) is solve:
        return name
    if name in _BOR_SOLVERS:
        from ghost_backend.bor import dispatch
        if getattr(dispatch, name) is solve:
            return name
    return None


def _plan_bor(arguments, options, certified, frequencies, workers, budget, cores, output_frequency_count=None):
    """Use the solve's resource preview; convert decimal BoR GB to scheduler GiB."""
    from ghost_backend.bor.dispatch import estimate_bor_resources
    from ghost_backend.runs.quality import validate_mesh_convergence_policy
    from ghost_backend.execution.options import execution_scope
    cpus = max(1, cores // workers)
    mode_workers = min(int(arguments.get('workers') or cpus), cpus)
    keys = ('geometry_units', 'material_base_dir', 'n_modes', 'max_elements',
            'table_precision', 'assembly', 'stream_budget_gb', 'expand_to_360')
    controls = {key: arguments[key] for key in keys if key in arguments}
    controls['fine_factor'] = validate_mesh_convergence_policy(
        arguments.get('mesh_convergence_policy'))['fine_factor']
    records = []
    for frequency in frequencies:
        try:
            with execution_scope({}, assembly_threads=cpus, memory_budget_gib=budget):
                candidate = estimate_bor_resources(arguments['geometry_snapshot'], frequency,
                    arguments['elevations_deg'], workers=mode_workers, mesh_certification=certified,
                    frequency_count=(len(arguments['frequencies_ghz']) if output_frequency_count is None
                                     else output_frequency_count), bor_options=options, **controls)
        except MemoryError:
            continue
        # Keep the preview's safety margin, plus room for run-time cap extension.
        # If extension cannot fit this reservation, the parent retries after
        # the pool drains; explicit n_modes and all convergence gates are intact.
        memory = float(candidate['estimated_peak_gb']) * 1e9 / 1024**3 * 1.25
        cost = float(candidate['n_unknowns_estimate'])**2 * (candidate['mode_cap_estimate']+1) / 1e6
        records.append(dict(frequency=frequency, cpus=cpus, memory_gib=memory,
                            cost=cost, selection=None))
    return sorted(records, key=lambda r: (-r['cost'], r['frequency']))


def _plan(arguments, options, certified, frequencies, workers, budget, cores):
    from ghost_backend.execution.options import execution_scope
    from ghost_backend.execution.selection import select_backend
    cpus = max(1, cores // workers)
    records = []
    for frequency in frequencies:
        try:
            with execution_scope(options, assembly_threads=cpus, memory_budget_gib=budget):
                selection = select_backend(dict(arguments, frequencies_ghz=[frequency]), options, certified)
        except MemoryError:
            # A large high-frequency unit must not prevent two smaller ones
            # from sharing the machine. The parent retries omitted units with
            # its full reservation after the worker pool has been released.
            continue
        mode = options['factorization']
        mode = selection['selected'] if mode == 'adaptive' else ('compressed' if mode == 'compressed' else 'dense')
        candidate = selection['candidates'][mode]
        # The ordinary memory gate uses an admission margin; leave that margin
        # inside the worker reservation too, and price the worker separately.
        memory = float(candidate['peak_gb']) * 1.25
        selection = dict(selection, retry_order=[m for m in selection.get('retry_order', [])
                         if selection['candidates'][m]['peak_gb'] <= candidate['peak_gb']])
        records.append(dict(frequency=frequency, cpus=cpus, memory_gib=memory,
                            cost=float(candidate['cost']),
                            selection=selection if options['factorization'] == 'adaptive' else None))
    return sorted(records, key=lambda r: (-r['cost'], r['frequency']))


def _close(executor, stop, failed):
    if failed:
        stop.set()
        processes = list((getattr(executor, '_processes', None) or {}).values())
        # Give cancellation checkpoints a short chance to leave native work.
        deadline = time.monotonic() + 2.
        for process in processes:
            process.join(max(0., deadline-time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        # A killed reader can otherwise strand a queued argument pipe on Windows.
        from ghost_backend.hpc.common import ExecutorPool
        ExecutorPool._release_feeder(executor, limit_seconds=5.)
    try:
        executor.shutdown(wait=True, cancel_futures=failed)
    except TypeError:  # Python 3.6-3.8
        executor.shutdown(wait=True)


def compute_parallel(solve, arguments, directory, store, options, precision, certified,
                     requested_workers, budget, recovery_directory=None):
    """Return completed work, or None when the sequential path is preferable."""
    if requested_workers != 'auto' and (type(requested_workers) is not int or requested_workers < 1):
        raise ValueError('Frequency workers must be auto or a positive integer.')
    if requested_workers == 1 or len(arguments['frequencies_ghz']) < 2 or mp.current_process().daemon:
        return None
    name = _solver_name(solve)
    if name is None:
        return None  # Custom API callables retain their existing in-process contract.
    from ghost_backend.execution.options import blas_core_budget, validate_options
    is_bor = name in _BOR_SOLVERS
    if is_bor:
        from ghost_backend.bor.options import validate_options
    from ghost_backend.twod.checkpoints import _retained_bytes
    options = validate_options(options)
    cores = blas_core_budget()
    # BoR permits duplicate requested frequencies: solve each once, then the
    # checkpoint merger restores every occurrence in the requested grid.
    frequencies = list(dict.fromkeys(arguments['frequencies_ghz']))
    fresh = store is None
    existing = [] if fresh else [f for f in frequencies if store.available(f)]
    missing = [f for f in frequencies if f not in existing]
    workers = min(cores, len(missing), AUTO_MAX_WORKERS if requested_workers == 'auto' else requested_workers)
    if workers < 2:
        return None
    fallback_limit = (max(0., budget)*1024**3 if fresh
                      else min(64*1024**2, max(0., budget)*1024**3*.02))
    # API memory results reserve room for their transfer copy as they complete.
    # Recovery workers return only metadata, so their retained field cost is zero.
    available = budget - workers*WORKER_GIB - (0. if fresh else fallback_limit/1024**3)
    if available <= 0:
        return None
    try:
        planner = _plan_bor if is_bor else _plan
        planning = {'output_frequency_count':1} if is_bor and recovery_directory is not None else {}
        records = planner(arguments, options, certified, missing, workers, available, cores, **planning)
    except MemoryError:
        # The unpartitioned parent may admit a case that cannot reserve two
        # worker interpreters alongside its numerical workspace.
        return None
    if len(records) < 2:
        return None
    smallest = sorted(r['memory_gib'] for r in records)[:2]
    minimum_cost = AUTO_MIN_BOR_COST if is_bor else AUTO_MIN_COST
    if sum(smallest) > available or (requested_workers == 'auto' and sum(r['cost'] for r in records) < minimum_cost):
        return None
    progress, abort = arguments.get('progress_callback'), arguments.get('abort_event')
    clean = {k: v for k, v in arguments.items() if k not in ('progress_callback', 'abort_event')}
    context = mp.get_context('spawn')
    stop, messages = context.Event(), context.Queue(maxsize=256)
    pending = list(records)
    running, fractions = {}, {}
    deferred = set()
    completed = set(existing)
    unsaved, warnings, profiles = {}, [], []
    peak_workers, peak_reserved, retained = 0, 0., 0
    failed = True
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_initialize, initargs=(stop, messages))
    try:
        while pending or running:
            if abort is not None and abort.is_set():
                raise InterruptedError('Solve canceled; no result was published.' if fresh else
                                       'Solve canceled; completed frequency checkpoints were retained.')
            used = sum(r['memory_gib'] for r in running.values())
            used_cpus = sum(r['cpus'] for r in running.values())
            for record in list(pending):
                if abort is not None and abort.is_set():
                    raise InterruptedError('Solve canceled; no result was published.' if fresh else
                                           'Solve canceled; completed frequency checkpoints were retained.')
                if retained > fallback_limit:
                    pending.clear()
                    break
                if (len(running) >= workers or used_cpus+record['cpus'] > cores or
                        used+record['memory_gib']+retained/1024**3 > available):
                    continue
                payload = dict(record, solver=name, arguments=clean, frequencies=frequencies,
                               options=options, precision=precision,
                               directory=None if fresh else str(directory),
                               identity=None if fresh else store.identity, certified=certified)
                if recovery_directory is not None:
                    payload['recovery_directory'] = str(recovery_directory)
                # Workers import only the backend, never replay a GUI __main__.
                from ghost_backend.compressed.tile_processes import _without_main_module
                from ghost_backend.execution.runtime import single_thread_worker_environment
                with _without_main_module(), single_thread_worker_environment():
                    future = executor.submit(_compute, payload)
                running[future] = record
                pending.remove(record)
                used += record['memory_gib']
                used_cpus += record['cpus']
                peak_workers = max(peak_workers, len(running))
                peak_reserved = max(peak_reserved, used + workers*WORKER_GIB + retained/1024**3)
            if not running:
                if pending:
                    # A unit too large for a spawned share keeps the normal
                    # in-process path, after completed checkpoints are retained.
                    break
                continue
            done, _ = wait(running, timeout=.05, return_when=FIRST_COMPLETED)
            for future in done:
                record = running.pop(future)
                try:
                    value = future.result()
                except MemoryError:
                    # A later adaptive candidate can exceed the initial
                    # forecast. Retry this frequency in the parent with its
                    # full reservation after every worker has drained.
                    fractions.pop(record['frequency'], None)
                    deferred.add(record['frequency'])
                    continue
                frequency = value['frequency']
                completed.add(frequency)
                fractions.pop(frequency, None)
                if value['profile']:
                    profiles.append(value['profile'])
                if value['result'] is not None:
                    unsaved[frequency] = value['result']
                    if value['warning']:
                        warnings.append(value['warning'])
                    retained = _retained_bytes(unsaved) * (2 if fresh else 1)
                # Future objects retain their result even after it has been
                # transferred into the run's output collection.
                del value
            done.clear()
            while True:
                try:
                    frequency, done_count, total, message = messages.get_nowait()
                except queue.Empty:
                    break
                if frequency not in completed and frequency not in deferred:
                    fractions[frequency] = min(1., max(0., done_count/max(total, 1)))
            if progress:
                progress(int(1000*(len(completed)+sum(fractions.values()))), 1000*len(frequencies),
                         'Frequency sweep: {} of {} complete; {} workers active.'.format(len(completed), len(frequencies), len(running)))
        failed = False
    finally:
        _close(executor, stop, failed)
        messages.close()
        messages.join_thread()
    return dict(completed=[f for f in frequencies if f in completed], unsaved=unsaved,
                warnings=warnings, profiles=profiles, reused=len(existing),
                worker_peaks={key: max((p[key] for p in profiles if p.get(key) is not None), default=None)
                              for key in ('sampled_peak_process_rss_bytes', 'sampled_peak_process_tree_rss_bytes',
                                          'sampled_peak_process_tree_private_bytes')},
                pending=[f for f in frequencies if f not in completed],
                details=dict(mode='parallel', workers=workers, maximum_active_workers=peak_workers,
                             cpu_budget=cores, memory_budget_gib=budget,
                             peak_reserved_gib=peak_reserved, interpreter_reservation_gib=workers*WORKER_GIB))
