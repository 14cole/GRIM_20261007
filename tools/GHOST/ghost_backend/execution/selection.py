"""Shared, capability-aware planning before allocating solver operators."""
import math
import copy
import json
import hashlib
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.execution.runtime import ScopedValue
from ghost_backend.execution.policy import MODEL, BACKENDS, relative_cost, rank_candidates, work_threads

_BATCH_SELECTION = ScopedValue('ghost_batch_backend_selection', default=None)
_REQUEST_SELECTION = ScopedValue('ghost_request_backend_selection', default=None)


def request_selection_scope(value):
    """The forecast made at the public boundary of the running request (or None)."""
    return _REQUEST_SELECTION.override(value)


def current_request_selection():
    value = _REQUEST_SELECTION.get()
    return dict(value) if value is not None else None


def batch_selection_scope(value):
    if value is not None and (value.get('requested') != 'adaptive' or
                              value.get('selected') not in BACKENDS):
        raise ValueError('Invalid batch backend selection.')
    return _BATCH_SELECTION.override(value)


def current_batch_selection():
    value = _BATCH_SELECTION.get()
    return dict(value) if value is not None else None


def select_backend(arguments, options, certified=False, checkpoint=None):
    from ghost_backend.twod.preparation import sweep_mesh_scope
    with sweep_mesh_scope(arguments['frequencies_ghz']):
        return _select_backend(arguments, options, certified, checkpoint)


def _select_backend(arguments, options, certified=False, checkpoint=None):
    """Reuse immutable run forecasts, but make every admission against live RAM."""
    from ghost_backend.twod.preparation import forecast_cache, mesh_frequencies
    from ghost_backend.twod import solver as s
    from ghost_backend.runs.quality import validate_mesh_convergence_policy
    cache = forecast_cache()
    if checkpoint:
        checkpoint()
    event = arguments.get('abort_event')
    if event is not None and event.is_set():
        raise InterruptedError('Backend planning canceled.')
    fields = ('geometry_snapshot', 'frequencies_ghz', 'elevations_deg', 'polarization')
    inputs = {key: arguments[key] for key in fields if key in arguments}
    inputs.update(geometry_units=arguments.get('geometry_units', 'inches'),
                  material_base_dir=arguments.get('material_base_dir'),
                  max_panels=arguments.get('max_panels', s.MAX_PANELS_DEFAULT),
                  mesh_reference_ghz=arguments.get('mesh_reference_ghz'),
                  fine_factor=validate_mesh_convergence_policy(arguments.get('mesh_convergence_policy'))['fine_factor'] if certified else 1.)
    if arguments.get('mesh_reference_ghz') is not None:
        inputs['mesh_frequencies_ghz'] = mesh_frequencies(arguments['frequencies_ghz'])
    normalized = validate_options(options)
    from ghost_backend.linalg.refined_lu import requested_precision
    # Retain only a digest: large geometry snapshots and angle grids must not be
    # repeated in every frequency's cache key.  The CPU/RAM allocation is not
    # part of the key: a reused forecast is repriced under the current
    # allocation (_refresh_memory_forecast), so the sweep planner, which
    # selects under each worker's share, reuses the preview's records instead
    # of building every candidate mesh again (October 2026 audit, R-2D-7).
    key = hashlib.sha256(json.dumps([inputs, normalized, requested_precision()], sort_keys=True).encode('utf-8')).digest() if cache is not None else None
    if cache is not None and key in cache:
        cached, resource_records = cache[key]
        result = copy.deepcopy(cached)
        _refresh_memory_forecast(result, resource_records, arguments, normalized, checkpoint)
        result['forecast_reused'] = True
        return result
    frequencies = list(arguments['frequencies_ghz'])
    if cache is not None and len(frequencies) > 1 and 'polarization' not in arguments:
        if len(set(frequencies)) != len(frequencies):
            raise ValueError('Duplicate frequencies are not supported.')
        # Setup previews each frequency independently. An uncheckpointed API
        # sweep may choose one backend for the whole sweep; combine the same
        # forecasts instead of constructing every candidate mesh again.
        parts = [select_backend(dict(arguments, frequencies_ghz=[frequency]),
                                normalized, certified, checkpoint) for frequency in frequencies]
        result = copy.deepcopy(parts[0])
        result['meshes'] = [row for part in parts for row in part['meshes']]
        result['candidates'] = {
            mode: dict(cost=sum(part['candidates'][mode]['cost'] for part in parts),
                       peak_gb=max(part['candidates'][mode]['peak_gb'] for part in parts))
            for mode in result['candidates']}
        with execution_scope(dict(normalized, factorization='dense')):
            budget = s._solve_memory_limit_gb()
        ranked = rank_candidates(result['candidates'], budget)
        result.update(selected=ranked[0], retry_order=ranked[1:], admission_budget_gib=budget,
                      dense_peak_gib=max(part['dense_peak_gib'] for part in parts),
                      forecast_reused=all(part.get('forecast_reused', False) for part in parts))
        return result
    resource_records = []
    result = _forecast_backend(arguments, normalized, certified, checkpoint, resource_records)
    if cache is not None:
        cache[key] = (copy.deepcopy(result), resource_records)
    return result


def _refresh_memory_forecast(result, resource_records, arguments, options, checkpoint=None):
    """Reprice saved geometry resources with current RAM, storage and CPU limits.

    Automatic compressed storage and dense residual storage depend on live
    memory, so retaining the old peak and merely re-ranking is insufficient;
    the work prior depends on the thread allocation, so the cost is summed
    again under the current one (same records, same formula as the forecast).
    Only small geometry resource counts are cached; no meshes or operators.
    """
    from ghost_backend.twod import solver as s
    for candidate in result['candidates'].values():
        candidate['peak_gb'] = 0.
        candidate['cost'] = 0.
    dense = dict(options, factorization='dense')
    n_rhs = len(arguments['elevations_deg'])
    with execution_scope(dense):
        budget = s._solve_memory_limit_gb()
        for record, (resources, mesh_options) in zip(result['meshes'], resource_records):
            if checkpoint is not None:
                checkpoint()
            event = arguments.get('abort_event')
            if event is not None and event.is_set():
                raise InterruptedError('Backend planning canceled.')
            peaks = {}
            for mode in result['candidates']:
                with execution_scope(dict(mesh_options, factorization=mode)):
                    peaks[mode] = s._estimate_memory_gb(resources['nodes'], False,
                        n_regions=resources['n_regions'], system_dofs=resources['system_dofs'],
                        operator_matrices=resources['operator_matrices'], dense_resources=dict(resources),
                        n_rhs=len(arguments['elevations_deg']), solver_method='experimental_cpu')
                result['candidates'][mode]['peak_gb'] = max(result['candidates'][mode]['peak_gb'], peaks[mode])
            with execution_scope(mesh_options):
                threads = work_threads(resources['system_dofs'], mesh_options)
            for mode in result['candidates']:
                result['candidates'][mode]['cost'] += relative_cost(resources, n_rhs, mode, *threads)
            record.update(dense_peak_gib=peaks['dense'], backend_peak_gib=peaks)
    ranked = rank_candidates(result['candidates'], budget)
    result.update(selected=ranked[0], retry_order=ranked[1:], admission_budget_gib=budget,
                  dense_peak_gib=max(record['dense_peak_gib'] for record in result['meshes']))


def _forecast_backend(arguments, options, certified=False, checkpoint=None, resource_records=None):
    """Forecast both polarizations and certification meshes before allocating A.

    Rank compatible backends using mesh-specific work and memory forecasts.
    This is a deterministic resource heuristic, not a promise of minimum time.
    Explicit backend choices never call this function.
    """
    from ghost_backend.twod import solver as s
    from ghost_backend.twod.preparation import prepare_geometry, mesh_frequencies as sizing_frequencies
    from ghost_backend.runs.quality import validate_mesh_convergence_policy, scale_snapshot_panel_density
    snapshot = arguments['geometry_snapshot']
    frequencies = list(arguments['frequencies_ghz'])
    if not frequencies or any(not math.isfinite(f) or f <= 0 for f in frequencies):
        raise ValueError('Frequencies must be a nonempty list of finite positive GHz values.')
    if len(set(frequencies)) != len(frequencies):
        raise ValueError('Duplicate frequencies are not supported.')
    if not arguments['elevations_deg'] or any(not math.isfinite(a) for a in arguments['elevations_deg']):
        raise ValueError('Angles must be a nonempty list of finite values.')
    units = arguments.get('geometry_units', 'inches')
    _, _, materials, scale = prepare_geometry(snapshot, arguments.get('material_base_dir'), units)
    from ghost_backend.twod.adaptive_geometry import candidate_meshes
    factor = validate_mesh_convergence_policy(arguments.get('mesh_convergence_policy'))['fine_factor'] if certified else 1.
    records = []
    candidates={m:dict(cost=0.,peak_gb=0.) for m in BACKENDS}
    exclusions={}
    dense = dict(validate_options(options), factorization='dense')
    with execution_scope(dense):
        budget = s._solve_memory_limit_gb()
        for freq in arguments['frequencies_ghz']:
            if checkpoint:
                checkpoint()
            event = arguments.get('abort_event')
            if event is not None and event.is_set():
                raise InterruptedError('Backend planning canceled.')
            ref = arguments.get('mesh_reference_ghz') or freq
            # Fixed-reference requests use the whole sweep for conservative
            # material sizing, even though operators remain frequency-local.
            mesh_frequencies = (sizing_frequencies(frequencies) if arguments.get('mesh_reference_ghz') is not None
                                else frequencies if 'polarization' in arguments else [freq])
            geometries = candidate_meshes(snapshot, materials, factor, options['mesh_strategy']=='adaptive',
                                         mesh_frequencies, scale, arguments.get('mesh_reference_ghz'))
            if options['mesh_strategy']!='adaptive':
                geometries = [(phase, geometry, options['basis_order']) for phase, geometry, _ in geometries]
            for phase, geometry, degree in geometries:
                mesh_options=dict(dense,basis_order=degree,mesh_strategy='local' if '_2d_hp_coarsening' in geometry else dense['mesh_strategy'])
                with execution_scope(mesh_options):
                    wavelength, _, _ = s._conservative_mesh_wavelength_for_frequencies(
                        geometry, materials, set(mesh_frequencies) | {ref}) if arguments.get('mesh_reference_ghz') else s._mesh_wavelength_for_snapshot(geometry, materials, ref)
                    # Conservative global mesh bounds local-material candidate sizes.
                    served = set(mesh_frequencies) | {ref} if arguments.get('mesh_reference_ghz') else [ref]
                    panels = s._build_panels(geometry, scale, wavelength, max_panels=arguments.get('max_panels', s.MAX_PANELS_DEFAULT),
                        segment_wavelengths=s.segment_wavelengths(geometry,materials,served,scale,wavelength),
                        materials=materials, frequencies_ghz=served)
                    k0 = 2 * math.pi * freq * 1e9 / s.C0
                    pols=(s._normalize_polarization(arguments['polarization']),) if 'polarization' in arguments else ('TE','TM')
                    for pol in pols:
                        infos = s._build_coupled_panel_info(panels, materials, freq, pol, k0)
                        mesh, _ = s._build_linear_mesh_interface_aware(panels, infos)
                        coupled = s._build_linear_coupled_infos(mesh, materials, freq, pol, k0)
                        layer = s.layer_for_mesh(mesh, materials, freq) if any(i.bc_kind == 'thin_layer' for i in coupled) else None
                        resources = s._dense_formulation_resources(mesh, coupled, pol, layer, sample_compression=False)
                        if resource_records is not None:
                            resource_records.append((dict(resources), dict(mesh_options)))
                        peaks={}
                        for mode in BACKENDS:
                            with execution_scope(dict(dense,factorization=mode)):
                                measured=dict(resources)
                                peaks[mode]=s._estimate_memory_gb(resources['nodes'], False,
                                    n_regions=resources['n_regions'], system_dofs=resources['system_dofs'],
                                    operator_matrices=resources['operator_matrices'], dense_resources=measured,
                                    n_rhs=len(arguments['elevations_deg']), solver_method='experimental_cpu')
                            threads=work_threads(resources['system_dofs'],mesh_options)
                            candidates[mode]['cost']+=relative_cost(resources,len(arguments['elevations_deg']),mode,*threads)
                            candidates[mode]['peak_gb']=max(candidates[mode]['peak_gb'],peaks[mode])
                        records.append(dict(frequency_ghz=float(freq), phase=phase, polarization=pol,
                            polynomial_degree=degree, panels=len(panels), unknowns=resources['system_dofs'], dense_peak_gib=peaks['dense'],
                            formulation=resources['formulation'],backend_peak_gib=peaks, resources=dict(resources)))
    peak = max(r['dense_peak_gib'] for r in records)
    candidates={m:c for m,c in candidates.items() if m not in exclusions}
    try:
        ranked=rank_candidates(candidates,budget)
    except MemoryError as exc:
        kinds=', '.join(sorted({r['formulation'] for r in records}))
        raise MemoryError('{} Formulations: {}.'.format(exc,kinds)) from exc
    selected=ranked[0]
    return dict(requested='adaptive', selected=selected, dense_peak_gib=peak,
        model=MODEL,objective='predicted_solve_completion',candidates=candidates,
        retry_order=ranked[1:],exclusions=exclusions,optimality_guaranteed=False,
        admission_budget_gib=budget, dense_margin_fraction=.2,
        reason='Lowest predicted cost among compatible backends admitted against available RAM.',
        meshes=records)
