"""BOR execution controls, independent of the 2D execution profile."""
from functools import wraps
import inspect
from ghost_backend.execution.runtime import ScopedValue


# factorization='auto' sizes the backend from the memory estimate, as the 2-D
# entry points do: dense/streaming while it fits, compressed once it does not.
# compressed_storage_mib=0 sizes the compressed cap the same way, instead of a
# fixed cap that an electrically large body silently outgrows.
# stream_spill='auto' lets a streamed conductor solve whose far blocks exceed
# the retained-block budget accumulate them once, for every mode, into
# memory-mapped temporary files instead of rebuilding a block per mode range
# (each range re-samples every far pair); 'off' keeps the per-range rebuild.
# far_compression='auto' keeps the streamed far blocks of a surface with at
# least bor.compressed_far.FAR_COMPRESSION_MIN_NODES nodes as a hierarchical
# (compressed) store built by cross approximation; 'on' compresses every
# streamed surface and 'off' keeps the dense streamed blocks.
DEFAULTS = dict(version=1, angle_batch_size=64, rhs_compression='auto',
                factorization='auto', compressed_storage_mib=0,
                compression_tile='auto', tile_cache_mib=16, near_backend='auto',
                stream_spill='auto', far_compression='auto',
                quadrature_check='off', near_refinement=0)
_ACTIVE = ScopedValue('ghost_bor_options', default=None)
_ABORT = ScopedValue('ghost_bor_abort', default=None)
_OUTPUT_GB = ScopedValue('ghost_bor_output_gb', default=0.)
_RESUME = ScopedValue('ghost_bor_mode_resume', default=None)
_RECYCLING = ScopedValue('ghost_bor_frequency_identity', default=(None, None))


def current_recycling_parameters():
    return _RECYCLING.get()


def _recycling_parameters(function, arguments, options):
    """Run-local identity of a modal equation, independent of frequency/RHS.

    Include material values, geometry, formulation, basis and quadrature
    controls. Dispersive values that change therefore conservatively miss.
    Callbacks, output grids and resource controls do not change the operator.
    """
    if 'freq_hz' not in arguments or options['factorization'] != 'compressed':
        return None, None
    from ghost_backend.compressed.recycling import capacity_bytes
    if not capacity_bytes():
        return None, None
    import hashlib
    import pickle
    excluded = {'freq_hz', 'thetas_deg', 'n_modes', 'workers', 'progress',
                'check_abort', 'mode_tol', 'assembly', 'stream_budget_gb'}
    values = {name: value for name, value in arguments.items() if name not in excluded}
    controls = (options['near_refinement'], options['quadrature_check'])
    identity = hashlib.sha256(pickle.dumps(
        (function.__module__, function.__qualname__, values, controls), protocol=5)).digest()
    return ('bor_modal_v1', identity), float(arguments['freq_hz'])


def current_mode_resume():
    return _RESUME.get()


def output_reserved_gb():
    return float(_OUTPUT_GB.get())


def estimate_output_gb(frequencies, aspects, certified=False, expanded=False):
    """Two channels plus sorting/column work and base/fine overlap.

    Compact rows retain ten float64 fields, shared labels and order indices;
    allow 256 bytes each for their column/sort temporaries. Public list calls
    retain their original conservative dictionary allowance.
    """
    from ghost_backend.twod.samples import compact_samples_enabled
    row_bytes = 256. if compact_samples_enabled() else 2048.
    return row_bytes * int(frequencies) * int(aspects) * 2 * (2 if expanded else 1) * (2 if certified else 1) / 1e9


def reserve_output(function):
    signature = inspect.signature(function)
    @wraps(function)
    def call(*args, **kwargs):
        bind_kwargs = dict(kwargs)
        bind_kwargs.pop('bor_options', None)
        values = signature.bind_partial(*args, **bind_kwargs).arguments
        reserved = estimate_output_gb(len(values.get('frequencies_ghz', ())),
            len(values.get('elevations_deg', ())), 'certified' in function.__name__,
            bool(values.get('expand_to_360', False)))
        with _OUTPUT_GB.override(max(output_reserved_gb(), reserved)):
            return function(*args, **kwargs)
    return call


class ModalConvergenceError(RuntimeError):
    def __init__(self, message, mode_cap, tail=None, tolerance=None):
        super().__init__(message)
        self.mode_cap = int(mode_cap)
        self.tail, self.tolerance = tail, tolerance


class BorAdmissionError(MemoryError):
    """A solve plan rejected by the memory gate, before any operator preparation.

    Only this rejection lets an automatic call move on to a smaller plan; an
    allocation failure while a solve runs stays a plain ``MemoryError``.
    ``streaming`` tells whether the rejected plan already streamed its far blocks,
    ``required_gb`` is the gate's estimate for it and ``mode_cap`` the azimuthal
    cap it was priced at, when known. ``expanded_caps`` are the automatic cap
    extensions in force: a requirement priced before a later extension is
    obsolete, because that cap is known not to converge.
    """
    def __init__(self, message, streaming=False, required_gb=None, mode_cap=None):
        super().__init__(message)
        self.streaming = bool(streaming)
        self.required_gb = None if required_gb is None else float(required_gb)
        self.mode_cap = None if mode_cap is None else int(mode_cap)
        self.expanded_caps = []


def next_automatic_plan(plan, rejection, caller_assembly, single_precision):
    """Next smaller plan after an admission rejection of an automatic direct call.

    Plans are ``(factorization, assembly)``; ``assembly`` None keeps the caller's
    request. The order is by speed, not by size (fixed workspaces make a streamed
    or compressed plan the larger one on a small body): dense streaming comes
    before compression because it is several times faster when it fits.
    ``caller_assembly`` is the caller's own request (None when the entry has no
    such argument), never the argument a streamed plan has overwritten. An
    explicit assembly is not exchanged for the other dense assembly; as it
    always has, ``factorization='auto'`` still authorizes compression, which has
    its own assembly. Compression needs double precision, so a single-precision
    request never reaches it and keeps its memory diagnostic instead.
    """
    factorization, assembly = plan
    if factorization != 'dense':
        return None
    if assembly is None and caller_assembly == 'auto' and not getattr(rejection, 'streaming', False):
        return 'dense', 'streaming'
    return None if single_precision else ('compressed', None)


def _rejected_plans_error(rejections):
    """``(error, cause)`` for an automatic call whose every plan was rejected by admission.

    Only plans priced under the final mode-cap extensions count: a smaller
    requirement from an earlier cap would send the caller to a limit that fails
    again. The smallest of those is reported (enough to admit that plan, which
    need not be the least any plan would need: plans rejected before the last
    extension are not priced again) and the other rejections stay visible.
    """
    def label(item):
        cap = '' if item['mode_cap'] is None else ' at mode cap {}'.format(item['mode_cap'])
        return '{}/{}{}'.format(item['factorization'], item['assembly'], cap)
    final = rejections[-1]['mode_cap_extensions']
    current = [item for item in rejections if item['mode_cap_extensions'] == final]
    smallest = min(current, key=lambda item: (item['required_gb'] is None, item['required_gb'] or 0.))
    others = [item for item in rejections if item is not smallest]
    error = BorAdmissionError(
        '{} No automatic plan was admitted: this requirement is for {}; also rejected: {}.'.format(
            smallest['reason'], label(smallest), ', '.join(label(item) for item in others)),
        smallest['assembly'] == 'streaming', smallest['required_gb'], smallest['mode_cap'])
    error.expanded_caps = list(final)
    return error, MemoryError(' | '.join('{}: {}'.format(label(item), item['reason']) for item in others))


def next_mode_cap(error, attempt):
    """Enlarged cap after an unconverged automatic modal truncation.

    Operator preparation is repeated at the new cap and its cost grows with the
    cap, so the first extension is sized from the measured tail: beyond the
    incident bandwidth modal increments fall faster than geometrically, and
    assuming only a halving per mode gives ``log2(tail/tolerance)`` further
    modes, plus the two quiet modes the test needs and a margin. It never
    exceeds the former 1.5x rule, which remains the second, last attempt.
    """
    import math
    ceiling = max(error.mode_cap+12, (3*error.mode_cap+1)//2)
    tail, tolerance = error.tail, error.tolerance
    if attempt or not tail or not tolerance or not math.isfinite(tail) or tail <= tolerance:
        return ceiling
    extra = int(math.ceil(math.log2(tail/tolerance))) + 8
    return min(ceiling, error.mode_cap + max(12, extra))


def validate_options(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('BOR options must contain supported fields only.')
    result = dict(DEFAULTS)
    result.update(value)
    for name, lower, upper in (('version', 1, 1), ('angle_batch_size', 1, 256)):
        number = result[name]
        if type(number) is not int or not lower <= number <= upper:
            raise ValueError('BOR {} must be an integer in {}..{}.'.format(name, lower, upper))
    tile = result['compression_tile']
    if tile != 'auto' and (type(tile) is not int or not 8 <= tile <= 128):
        raise ValueError('BOR compression_tile must be auto or an integer in 8..128.')
    storage = result['compressed_storage_mib']
    if type(storage) is not int or not (storage == 0 or 16 <= storage <= 1048576):
        raise ValueError('BOR compressed_storage_mib must be 0 for automatic, '
                         'or an integer in 16..1048576.')
    if type(result['tile_cache_mib']) is not int or not 0 <= result['tile_cache_mib'] <= 4096:
        raise ValueError('BOR tile_cache_mib must be an integer in 0..4096.')
    if result['rhs_compression'] not in ('off', 'auto', 'on'):
        raise ValueError('BOR rhs_compression must be off, auto, or on.')
    if result['factorization'] not in ('auto', 'dense', 'compressed'):
        raise ValueError('BOR factorization must be auto, dense, or compressed.')
    if result['near_backend'] not in ('auto', 'threads', 'processes'):
        raise ValueError('BOR near_backend must be auto, threads, or processes.')
    if result['stream_spill'] not in ('auto', 'off'):
        raise ValueError('BOR stream_spill must be auto or off.')
    if result['far_compression'] not in ('auto', 'on', 'off'):
        raise ValueError('BOR far_compression must be auto, on, or off.')
    if result['quadrature_check'] not in ('off', 'refine'):
        raise ValueError('BOR quadrature_check must be off or refine.')
    if type(result['near_refinement']) is not int or not 0 <= result['near_refinement'] <= 2:
        raise ValueError('BOR near_refinement must be an integer in 0..2.')
    if result['quadrature_check'] == 'refine' and result['near_refinement'] == 2:
        raise ValueError('BOR quadrature comparison needs near_refinement below 2.')
    return result


def current_options():
    return dict(_ACTIVE.get() or DEFAULTS)


def resolved_compression_tile(options, exact_far_cache=False):
    """Honor explicit sizes; larger automatic tiles require cheap exact slices."""
    tile = options['compression_tile']
    return (128 if exact_far_cache else 32) if tile == 'auto' else int(tile)


def compressed_requested():
    return current_options()['factorization'] == 'compressed'


def option_scope(value):
    return _ACTIVE.override(value)


def current_checkpoint():
    return _ABORT.get()


def _keyword_catchall(signature):
    return next((name for name, parameter in signature.parameters.items()
                 if parameter.kind is inspect.Parameter.VAR_KEYWORD), None)


def _caller_assembly(signature, bound):
    """The caller's own assembly request, or None when the entry cannot take one."""
    if 'assembly' in signature.parameters:
        return str(bound.arguments['assembly']).strip().lower()
    catchall = _keyword_catchall(signature)
    if catchall is None:
        return None
    return str((bound.arguments.get(catchall) or {}).get('assembly', 'auto')).strip().lower()


def _impose_assembly(signature, bound, assembly):
    if 'assembly' in signature.parameters:
        bound.arguments['assembly'] = assembly
    else:
        catchall = _keyword_catchall(signature)  # the survey entry forwards **kwargs
        bound.arguments[catchall] = dict(bound.arguments.get(catchall) or {}, assembly=assembly)


def _run_configured(function, signature, bound, options, supplied, checkpoint, expanded_caps=()):
    """One execution of a configured public call under a resolved factorization.

    ``expanded_caps`` are automatic mode-cap extensions already found necessary
    by a plan that was rejected afterwards; this plan starts from the last one.
    """
    automatic_modes = ('freq_hz' in signature.parameters
                       and 'n_modes' in signature.parameters
                       and bound.arguments.get('n_modes') is None)
    expanded_caps = list(expanded_caps) if automatic_modes else []
    if expanded_caps:
        bound.arguments['n_modes'] = expanded_caps[-1]
    if options['factorization'] == 'compressed':
        precision = str(bound.arguments.get('table_precision', 'auto')).strip().lower()
        if precision not in ('auto', 'double', 'single'):
            raise ValueError('BOR table_precision must be auto, single, or double.')
        if str(bound.arguments.get('assembly', 'auto')).strip().lower() not in ('auto', 'tables', 'streaming'):
            raise ValueError('BOR assembly must be auto, tables, or streaming.')
        if precision == 'single':
            raise ValueError('Compressed BOR assembly requires double precision.')
        for name, value in (('assembly', 'tables'), ('table_precision', 'double')):
            if name in signature.parameters:
                bound.arguments[name] = value
    args, kwargs = bound.args, bound.kwargs
    from ghost_backend.bor.cache import TileCache, current_cache, cache_scope
    cache = current_cache() if options['factorization'] == 'compressed' else None
    if (cache is None or supplied is not None) and options['factorization'] == 'compressed':
        cache = TileCache(options['tile_cache_mib'] * 1024**2)
    direct_grid = bound.arguments.get('thetas_deg')
    import numpy as np
    direct_output = 0. if direct_grid is None else estimate_output_gb(1, np.size(direct_grid))
    recycling = _recycling_parameters(function, bound.arguments, options)
    from ghost_backend.bor.preparation import numerical_preparation
    with _ACTIVE.override(options), _RECYCLING.override(recycling), numerical_preparation() as prepared:
        with _ABORT.override(checkpoint), _OUTPUT_GB.override(max(output_reserved_gb(),direct_output)):
            with cache_scope(cache):
                resume_state = {} if automatic_modes else None
                while True:
                    try:
                        prepared.begin_attempt()
                        with _RESUME.override(resume_state):
                            result = function(*args, **kwargs)
                        break
                    except ModalConvergenceError as exc:
                        if not automatic_modes or len(expanded_caps) >= 2:
                            raise
                        if checkpoint is not None:
                            checkpoint()
                        cap = next_mode_cap(exc, len(expanded_caps))
                        expanded_caps.append(cap)
                        bound.arguments['n_modes'] = cap
                        args, kwargs = bound.args, bound.kwargs
                    except BorAdmissionError as exc:
                        if expanded_caps and not exc.expanded_caps:
                            exc.expanded_caps = list(expanded_caps)
                        raise
                if expanded_caps and isinstance(result, dict):
                    result['automatic_mode_cap_extensions'] = expanded_caps
                if isinstance(result, dict) and prepared.surfaces:
                    result['numerical_preparation'] = prepared.evidence()
    return result, cache


def configured(function):
    """Accept bor_options= on public APIs, restoring nested calls on failure."""
    from ghost_backend.twod.preparation import preparation_scope
    signature = inspect.signature(function)
    @wraps(function)
    @preparation_scope()
    def wrapped(*args, **kwargs):
        supplied = kwargs.pop('bor_options', None)
        options = current_options() if supplied is None else validate_options(supplied)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        # This decorator also serves resource previews, which have no fields
        # to compare and must never launch numerical solves during setup.
        field_solve = 'thetas_deg' in signature.parameters or 'elevations_deg' in signature.parameters
        if options['quadrature_check'] == 'refine' and field_solve:
            from ghost_backend.bor.quadrature import checked_solve
            import numpy as np
            values = dict(bound.arguments)
            catchall = _keyword_catchall(signature)
            if catchall is not None:
                values.update(values.get(catchall) or {})
            reserved = estimate_output_gb(len(values.get('frequencies_ghz', (1,))),
                np.size(values.get('thetas_deg', values.get('elevations_deg', ()))),
                'certified' in function.__name__, bool(values.get('expand_to_360', False)))
            # Direct and survey entries have no outer reserve_output wrapper.
            # Establish their normal allowance before checked_solve adds the
            # compact base fields retained during the refined execution.
            with _OUTPUT_GB.override(max(output_reserved_gb(), reserved)):
                return checked_solve(wrapped, args, kwargs, options)
        checkpoint = bound.arguments.get('check_abort', current_checkpoint())
        # Preserve the public contract: reject CFIE endpoint values before
        # trying to inspect geometry (including intentionally invalid inputs).
        if 'cfie_alpha' in bound.arguments and bound.arguments.get('formulation','cfie') in ('cfie','auto'):
            import math
            alpha = float(bound.arguments['cfie_alpha'])
            if not math.isfinite(alpha) or not 0 < alpha < 1:
                raise ValueError('BoR CFIE alpha must be finite and satisfy 0 < alpha < 1.')
        automatic = options['factorization'] == 'auto'
        plan = options['factorization'], None
        if automatic:
            from ghost_backend.bor.dispatch import resolve_automatic_plan
            with _ACTIVE.override(options):
                plan = resolve_automatic_plan(bound.arguments, certified='certified' in function.__name__)
        # The direct chooser cannot construct the solvers whose admission it
        # predicts. An automatic plan rejected by that admission gate, which
        # precedes operator preparation, is replaced by the next smaller one
        # (dense tables, dense streaming, compression). Nothing else is retried:
        # an allocation failure while a solve runs propagates.
        direct = 'freq_hz' in signature.parameters and 'geometry_snapshot' not in signature.parameters
        caller_assembly = _caller_assembly(signature, bound)
        single_precision = str(bound.arguments.get('table_precision', 'auto')).strip().lower() == 'single'
        if plan[1] is not None and caller_assembly is None:
            plan = 'compressed', None  # an entry that cannot take the streamed assembly
        pristine_args, pristine_kwargs = args, dict(kwargs)
        rejections, expanded_caps = [], []
        while True:
            options = dict(options, factorization=plan[0])
            bound = signature.bind_partial(*pristine_args, **pristine_kwargs)
            bound.apply_defaults()
            if plan[1] is not None:
                _impose_assembly(signature, bound, plan[1])
            try:
                result, cache = _run_configured(function, signature, bound, options, supplied,
                                                checkpoint, expanded_caps)
                break
            except BorAdmissionError as exc:
                following = (next_automatic_plan(plan, exc, caller_assembly, single_precision)
                             if automatic and direct else None)
                # Keep only text: the traceback would pin the rejected solver objects.
                rejections.append(dict(factorization=plan[0],
                    assembly='streaming' if exc.streaming else (plan[1] or 'tables'),
                    mode_cap=exc.mode_cap if exc.mode_cap is not None else (exc.expanded_caps or [None])[-1],
                    mode_cap_extensions=list(exc.expanded_caps),
                    required_gb=exc.required_gb, reason=str(exc)))
                if following is None:
                    if len(rejections) > 1:
                        error, others = _rejected_plans_error(rejections)
                        raise error from others
                    raise
                if checkpoint is not None:
                    checkpoint()
                plan, expanded_caps = following, exc.expanded_caps
        if isinstance(result, dict):
            if plan[1] is not None:
                result['automatic_assembly'] = plan[1]
                if isinstance(result.get('metadata'), dict):
                    result['metadata']['automatic_assembly'] = plan[1]
                    if 'assembly_requested' in result['metadata']:
                        result['metadata']['assembly_requested'] = caller_assembly  # the caller's, not the imposed one
            if rejections:
                result['automatic_factorization_fallback'] = dict(
                    rejected=rejections[0]['factorization'], used=plan[0], assembly=plan[1],
                    reason=rejections[0]['reason'], rejections=rejections)
        if isinstance(result, dict):
            if 'modes_used' in result:
                result.setdefault('near_quadrature', {})['self_and_junction_convergence_checked'] = False
                result['near_quadrature']['refinement_depth_increment'] = options['near_refinement']
                result['modal_convergence_scope'] = 'requested_angles_and_polarizations'
            result['bor_execution_options'] = dict(options)
            if cache is not None:
                result['bor_tile_cache'] = cache.evidence()
                if isinstance(result.get('metadata'), dict):
                    result['metadata']['bor_tile_cache'] = cache.evidence()
            if isinstance(result.get('metadata'), dict):
                result['metadata']['bor_execution_options'] = dict(options)
            if options['factorization'] == 'compressed' and 'assembly' in result:
                result['assembly'] = 'compressed'
        return result
    parameters = list(signature.parameters.values())
    position = next((i for i, parameter in enumerate(parameters)
                     if parameter.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
    parameters.insert(position, inspect.Parameter('bor_options', inspect.Parameter.KEYWORD_ONLY, default=None))
    wrapped.__signature__ = signature.replace(parameters=parameters)
    return wrapped


def bounded_rhs_count(count, polarizations=2):
    return min(int(count), polarizations * current_options()['angle_batch_size'])
