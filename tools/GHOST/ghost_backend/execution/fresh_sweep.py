"""Fresh sweeps with optional run-owned recovery and bounded output memory."""
import time
from contextlib import ExitStack


def run_fresh(solve, arguments, options, precision, certified, *, solver_kind='2d',
              frequency_workers='auto', recovery=None):
    """Compute every requested frequency without reusing earlier runs.

    Matrix and geometry sharing within this invocation is still allowed. A
    later invocation always calls the numerical solver again, including after
    cancellation. With recovery, completed fields are saved in this run's
    unique folder; otherwise the API returns its normal in-memory result.
    Existing disk checkpoints are never consulted.
    """
    from ghost_backend.twod.preparation import preparation_scope, sweep_mesh_scope
    if recovery is not None:
        arguments = recovery.arguments(arguments)
    with ExitStack() as scope:
        scope.enter_context(preparation_scope())
        scope.enter_context(sweep_mesh_scope(arguments['frequencies_ghz']))
        if solver_kind == 'bor':
            from ghost_backend.bor.options import _OUTPUT_GB, estimate_output_gb, output_reserved_gb
            reserved = estimate_output_gb(1 if recovery is not None else len(arguments['frequencies_ghz']),
                len(arguments['elevations_deg']), certified, bool(arguments.get('expand_to_360', False)))
            scope.enter_context(_OUTPUT_GB.override(max(reserved, output_reserved_gb())))
        try:
            result = _run(solve, arguments, options, precision, certified, solver_kind, frequency_workers, recovery)
            if recovery is not None:
                recovery.status('completed')
            return result
        except BaseException as exc:
            if recovery is not None:
                try:
                    recovery.status('interrupted' if isinstance(exc, InterruptedError) else 'failed', exc)
                except OSError:
                    pass
            raise


def _run(solve, arguments, options, precision, certified, solver_kind, frequency_workers, recovery=None):
    from ghost_backend.twod.checkpoints import _retained_bytes
    from ghost_backend.twod.solver import _merge_frequency_results, _solve_memory_limit_gb
    from ghost_backend.execution.options import memory_allocation_scope
    from ghost_backend.execution.frequency_sweep import compute_parallel
    started = time.perf_counter()
    frequencies = list(arguments['frequencies_ghz'])
    if not frequencies:
        raise ValueError('At least one frequency is required.')
    if solver_kind != 'bor' and len(set(frequencies)) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported in a co-polarized result grid.')
    abort, progress = arguments.get('abort_event'), arguments.get('progress_callback')
    def check_abort():
        if abort is not None and abort.is_set():
            raise InterruptedError('Solve canceled; no result was published.')
    check_abort()
    initial_budget = _solve_memory_limit_gb()
    extra = {} if recovery is None else {'recovery_directory':str(recovery.directory)}
    parallel = compute_parallel(solve, arguments, None, None, options, precision,
                                certified, frequency_workers, initial_budget, **extra)
    completed = {} if parallel is None else parallel['unsaved']
    completed_frequencies = set() if parallel is None else set(parallel['completed'])
    profiles = [] if parallel is None else parallel['profiles']
    unique = list(dict.fromkeys(frequencies))
    retained = _retained_bytes(completed)
    for frequency in unique:
        check_abort()
        if frequency in completed_frequencies:
            continue
        def report(done, total, message):
            if progress:
                progress(1000*len(completed_frequencies) + int(1000*done/max(total, 1)),
                         1000*len(unique), message)
        # Allow space for the result merge as well as the retained samples.
        remaining = min(initial_budget - 2*retained/1024**3, _solve_memory_limit_gb())
        if remaining <= 0:
            raise MemoryError('Completed results leave insufficient RAM for the remaining frequencies. '
                              'Run a smaller frequency or angle range.')
        with memory_allocation_scope(remaining):
            result = solve(**dict(arguments, frequencies_ghz=[frequency], progress_callback=report))
        if recovery is None:
            completed[frequency] = result
        else:
            recovery.save(frequency, result)
        completed_frequencies.add(frequency)
        profile = result.get('metadata', {}).get('runtime_profile')
        if profile:
            profiles.append(profile)
        if recovery is None:
            retained += _retained_bytes(result)
        if progress:
            progress(1000*len(completed_frequencies), 1000*len(unique),
                     'Frequency {:g} GHz {}; {} of {} complete.'.format(
                         frequency, 'saved for this run' if recovery is not None else 'computed',
                         len(completed_frequencies), len(unique)))
        del result
    check_abort()
    merge = _merge_frequency_results
    if solver_kind == 'bor':
        from ghost_backend.bor.checkpoints import merge_frequency_results as merge
    result = (recovery.result() if recovery is not None else
              completed[frequencies[0]] if len(frequencies) == 1 else
              merge((completed[f] for f in frequencies), frequencies))
    metadata = result.setdefault('metadata', {})
    metadata.pop('frequency_checkpoints', None)
    execution = dict(mode='sequential') if parallel is None else dict(parallel['details'])
    execution.update(computed_frequencies=len(unique), reused_frequencies=0,
                     persistent_solve_cache=False)
    metadata['frequency_execution'] = execution
    keys = ('sampled_peak_process_rss_bytes', 'sampled_peak_process_tree_rss_bytes',
            'sampled_peak_process_tree_private_bytes')
    profile = dict(wall_seconds=time.perf_counter()-started,
        stage_seconds={key:sum(p.get('stage_seconds', {}).get(key, 0.) for p in profiles)
                       for key in set(key for p in profiles for key in p.get('stage_seconds', {}))},
        stage_calls={key:sum(p.get('stage_calls', {}).get(key, 0) for p in profiles)
                     for key in set(key for p in profiles for key in p.get('stage_calls', {}))},
        process_tree_incomplete_samples=sum(p.get('process_tree_incomplete_samples', 0) for p in profiles),
        stage_semantics='Fresh execution only. Nested stages may overlap.',
        memory_semantics='Process samples from this run; includes other work in the process.',
        process_tree_memory_semantics='Process-tree RSS may count shared pages more than once.')
    for key in keys:
        values = [p[key] for p in profiles if p.get(key) is not None]
        profile[key] = max(values) if values else None
    if parallel is not None:
        # Neither independent worker maxima nor parent-only samples measure
        # the total simultaneous RAM of a parallel sweep.
        for key, value in parallel['worker_peaks'].items():
            profile[key.replace('sampled_peak_', 'sampled_peak_frequency_worker_')] = value
            profile[key] = None
        profile['stage_semantics'] = ('Fresh execution only. Worker times are summed; '
            'concurrent and nested stages may overlap and their sum may exceed wall time.')
        profile['memory_semantics'] = profile['process_tree_memory_semantics'] = (
            'Total concurrent RSS was not sampled. frequency_worker fields are maxima '
            'of individual worker samples, not aggregate sweep memory.')
    metadata['runtime_profile'] = profile
    return result
