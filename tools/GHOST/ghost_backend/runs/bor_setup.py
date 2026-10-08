"""Portable BOR physics settings, with explicit aspect-angle semantics."""
import math
from ghost_backend.bor.options import option_scope, validate_options
from ghost_backend.twod.preparation import preparation_scope


def validate_bor_setup(value):
    fields = {'schema', 'version', 'frequencies_ghz', 'aspects_deg', 'units',
              'mesh_certification', 'accuracy', 'cfie_alpha', 'bor_options'}
    if (not isinstance(value, dict) or set(value) - {'radar_grid'} != fields or
            value.get('schema') != 'grim.bor-run-setup' or
            type(value.get('version')) is not int or value['version'] != 1):
        raise ValueError('Choose a supported GRIM BOR run setup (version 1).')
    result = dict(value)
    for key in ('frequencies_ghz', 'aspects_deg'):
        samples = value[key]
        if not isinstance(samples, list) or not 0 < len(samples) <= 100000:
            raise ValueError('{}: supply 1 to 100,000 samples.'.format(key))
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in samples):
            raise ValueError('{}: every sample must be finite and numeric.'.format(key))
        if len(samples) != len(set(samples)):
            raise ValueError('{}: duplicate samples are not supported.'.format(key))
        if key == 'frequencies_ghz' and any(x <= 0 for x in samples):
            raise ValueError('Frequencies must be positive GHz values.')
        if key == 'aspects_deg' and any(x < 0 or x > 180 for x in samples):
            raise ValueError('BOR aspects must lie in [0, 180] degrees from +z.')
        result[key] = list(samples)
    if value['units'] not in ('inches', 'meters'):
        raise ValueError('BOR geometry units must be inches or meters.')
    if type(value['mesh_certification']) is not bool:
        raise ValueError('mesh_certification must be true or false.')
    if value['accuracy'] not in ('standard', 'tight'):
        raise ValueError('BOR accuracy must be standard or tight.')
    alpha = value['cfie_alpha']
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('BOR CFIE alpha must lie strictly between 0 and 1.')
    result['bor_options'] = validate_options(value['bor_options'])
    grid = value.get('radar_grid')
    if grid is not None:
        import numpy as np
        from ghost_backend.assembly.fields import radar_grid_aspects, validate_radar_grid
        keys = {'azimuths_deg', 'elevations_deg', 'axis_az_deg', 'axis_el_deg', 'roll_deg'}
        if not isinstance(grid, dict) or set(grid) != keys:
            raise ValueError('Invalid BOR radar-grid fields.')
        for key in ('azimuths_deg', 'elevations_deg'):
            if (not isinstance(grid[key], list) or not 0 < len(grid[key]) <= 100000 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in grid[key])):
                raise ValueError('BOR radar axes must be finite numeric lists.')
        for key in ('axis_az_deg', 'axis_el_deg', 'roll_deg'):
            if type(grid[key]) not in (int, float) or not math.isfinite(grid[key]):
                raise ValueError('BOR body attitude must be finite and numeric.')
        validate_radar_grid(grid['azimuths_deg'], grid['elevations_deg'])
        if len(grid['azimuths_deg'])*len(grid['elevations_deg']) > 100000:
            raise ValueError('Saved BOR radar grids support at most 100,000 looks.')
        aspects = radar_grid_aspects(grid['azimuths_deg'], grid['elevations_deg'], grid['axis_az_deg'], grid['axis_el_deg'])
        if len(aspects) != len(result['aspects_deg']) or not np.allclose(aspects, result['aspects_deg'], rtol=0., atol=1e-10):
            raise ValueError('BOR body aspects do not match the saved radar grid.')
        grid = dict(grid, azimuths_deg=list(grid['azimuths_deg']), elevations_deg=list(grid['elevations_deg']))
    result['radar_grid'] = grid
    return result


def driver_settings(value):
    value = validate_bor_setup(value)
    grid = value['radar_grid'] or dict(azimuths_deg=[0.],
        elevations_deg=[90.-x for x in value['aspects_deg']], axis_az_deg=0., axis_el_deg=90., roll_deg=0.)
    return dict(FREQUENCIES_GHZ=value['frequencies_ghz'], AZIMUTHS_DEG=grid['azimuths_deg'],
                ELEVATIONS_DEG=grid['elevations_deg'], BODY_AXIS_AZ_DEG=grid['axis_az_deg'],
                BODY_AXIS_EL_DEG=grid['axis_el_deg'], BODY_ROLL_DEG=grid['roll_deg'],
                GEOMETRY_UNITS=value['units'], MESH_CERTIFICATION=value['mesh_certification'],
                ACCURACY_TARGET=value['accuracy'], CFIE_ALPHA=value['cfie_alpha'],
                BOR_EXECUTION_OPTIONS=value['bor_options'])


@preparation_scope()
def resource_summary(snapshot, base_dir, value, checkpoint=None, forecast_frequencies=None):
    import os
    from ghost_backend.bor.dispatch import estimate_bor_resources, resolve_automatic_plan
    from ghost_backend.runs.quality import accuracy_target_policy
    from ghost_backend.runs.setup import geometry_dimensions, validate_material_coverage
    value = validate_bor_setup(value)
    from ghost_backend.assembly.fields import bor_output_profile
    bor_output_profile(snapshot, value["units"])
    policy = accuracy_target_policy(value['accuracy'])
    options = dict(value['bor_options'])
    workers = max(1, (os.cpu_count() or 2)-1)
    if checkpoint is not None:
        checkpoint()
    frequencies = (value['frequencies_ghz'] if forecast_frequencies is None
                   else list(forecast_frequencies))
    if forecast_frequencies is not None:
        # Cached fields remove the need for a numerical resource forecast,
        # but input validation still covers the entire requested sweep.
        from ghost_backend.twod.preparation import prepare_geometry
        _, _, library, _ = prepare_geometry(snapshot, base_dir, value['units'])
        validate_material_coverage(snapshot, library, value['frequencies_ghz'], checkpoint)
    assembly = {}
    if frequencies and options['factorization'] == 'auto':
        # The same plan as the solve: backend, and streamed far blocks when it imposes
        # them. The resolver prices under the active options (they size the aspect
        # batches), so scope the caller's, as `configured` does for a public call.
        with option_scope(dict(options)):
            chosen, imposed = resolve_automatic_plan(dict(
                geometry_snapshot=snapshot, frequencies_ghz=frequencies,
                elevations_deg=value['aspects_deg'], geometry_units=value['units'],
                material_base_dir=base_dir, workers=workers,
                mesh_certification=value['mesh_certification'], fine_factor=policy['fine_factor'],
                check_abort=checkpoint))
        options['factorization'] = chosen
        if imposed is not None:
            assembly = dict(assembly=imposed)
    peak, elements = 0., 0
    mode_workers = set()
    for frequency in frequencies:
        if checkpoint is not None:
            checkpoint()
        estimate = estimate_bor_resources(snapshot, frequency, value['aspects_deg'],
            geometry_units=value['units'], material_base_dir=base_dir,
            workers=workers,
            frequency_count=len(value['frequencies_ghz']),
            mesh_certification=value['mesh_certification'], fine_factor=policy['fine_factor'],
            bor_options=options, **assembly)
        peak = max(peak, estimate['estimated_peak_gb'])
        elements = max(elements, estimate['mesh_elements'])
        mode_workers.add(estimate['active_mode_workers'])
    worker_note = (str(min(mode_workers)) if len(mode_workers) == 1 else
                   '{}\u2013{}'.format(min(mode_workers), max(mode_workers)) if mode_workers else '')
    backend_note = options['factorization'] + (' (streamed far blocks)' if assembly else '')
    if frequencies and value['bor_options']['factorization'] == 'auto':
        backend_note = 'auto \u2192 ' + backend_note
    grid = value['radar_grid']
    grid_note = ("Output: {} azimuths \u00d7 {} elevations; radar VV, HH and VH.\n".format(
        len(grid['azimuths_deg']), len(grid['elevations_deg'])) if grid is not None else '')
    resource_note = ('Up to {} mesh elements; estimated peak {:.2f} GB with {} simultaneous mode workers (memory limited). '.format(elements, peak, worker_note)
        + 'This is an allocation forecast, not measured RAM. Numerical quality and convergence are checked during the solve.'
        if frequencies else 'Matching checkpoints found for every frequency; no new solve forecast needed. '
        + 'Checkpoint integrity is verified again when loading the results.')
    return ('BOR geometry and material checks passed. ' + geometry_dimensions(snapshot, value['units']) + '\n'
        + '{} frequencies; {} aspects from +z; VV + HH. '.format(len(value['frequencies_ghz']), len(value['aspects_deg']))
        + ('Base/fine mesh comparison' if value['mesh_certification'] else 'Survey; no mesh certificate')
        + '; {} accuracy; CFIE alpha {:g}.\n'.format(value['accuracy'], value['cfie_alpha'])
        + grid_note
        + '{} factorization; {} aspects per batch; {} incident basis reuse.\n'.format(
            backend_note, options['angle_batch_size'], options['rhs_compression'])
        + resource_note)
