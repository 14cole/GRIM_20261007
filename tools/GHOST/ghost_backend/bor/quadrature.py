"""Optional same-mesh spatial-quadrature comparison for BoR solves.

Only compact complex fields survive between the two solves. The returned
result uses the refined rule; agreement is evidence at the requested samples,
not a proof of continuous-field or geometric-model accuracy.
"""
import time
import numpy as np

MAX_NORMALIZED_CHANGE = 2e-3
RMS_NORMALIZED_CHANGE = 1e-3


def _fields(result):
    from ghost_backend.twod.samples import sample_column
    channels = result.get('co_solved_samples')
    records = {}
    if channels:
        for pol in ('VV', 'HH'):
            rows = channels[pol]
            def column(key):
                values = sample_column(rows, key)
                return np.asarray(values if values is not None else [row[key] for row in rows])
            frequency = column('frequency_ghz')
            angles = np.column_stack((column('theta_inc_deg'), column('theta_scat_deg')))
            amplitude = column('rcs_amp_real') + 1j * column('rcs_amp_imag')
            for value in np.unique(frequency):
                mask = frequency == value
                records[(pol, float(value))] = (angles[mask].copy(), amplitude[mask].copy())
    else:
        angles = np.asarray(result['theta_deg'], float).copy()
        for pol in ('VV', 'HH'):
            records[(pol, None)] = (angles, np.asarray(result['amp_' + pol.lower()], complex).copy())
    if any(not np.all(np.isfinite(amplitude)) for _, amplitude in records.values()):
        raise RuntimeError('BoR quadrature comparison received non-finite fields.')
    return records


def compare_fields(base, refined):
    if set(base) != set(refined):
        raise RuntimeError('BoR quadrature comparison changed the frequency/channel grid.')
    records = []
    for key, (angles, before) in base.items():
        new_angles, after = refined[key]
        if before.shape != after.shape or not np.array_equal(angles, new_angles):
            raise RuntimeError('BoR quadrature comparison changed the angular grid.')
        scale = max(float(np.max(abs(after))), float(np.max(abs(before))), 1e-300)
        difference = abs(after - before) / scale
        maximum = float(np.max(difference))
        rms = float(np.sqrt(np.mean(difference**2)))
        records.append(dict(polarization=key[0], frequency_ghz=key[1],
            complex_max_normalized=maximum, complex_rms_normalized=rms,
            passed=maximum <= MAX_NORMALIZED_CHANGE and rms <= RMS_NORMALIZED_CHANGE))
    return dict(passed=all(row['passed'] for row in records), samples=records,
        complex_max_limit=MAX_NORMALIZED_CHANGE, complex_rms_limit=RMS_NORMALIZED_CHANGE,
        scope='same_mesh_self_adjacent_and_junction_rules_at_requested_samples',
        geometry_approximation_certified=False, published_rule='refined')


def checked_solve(solve, args, kwargs, options):
    from ghost_backend.bor.options import _OUTPUT_GB, output_reserved_gb
    started = time.perf_counter()
    base_options = dict(options, quadrature_check='off')
    fine_options = dict(base_options, near_refinement=options['near_refinement'] + 1)
    def arguments(phase):
        out = dict(kwargs)
        for name in ('progress_callback', 'progress'):
            callback = kwargs.get(name)
            if callback is not None:
                def report(done, total, message, callback=callback):
                    callback(phase*1000 + int(1000*done/max(total, 1)), 2000,
                             ('Base' if phase == 0 else 'Refined') + ' quadrature: ' + str(message))
                out[name] = report
        return out
    base_result = solve(*args, bor_options=base_options, **arguments(0))
    base = _fields(base_result)
    del base_result
    retained = sum(grid.nbytes + field.nbytes for grid, field in base.values())
    with _OUTPUT_GB.override(output_reserved_gb() + retained / 1e9):
        result = solve(*args, bor_options=fine_options, **arguments(1))
    evidence = compare_fields(base, _fields(result))
    evidence.update(base_refinement=options['near_refinement'],
                    refined_refinement=fine_options['near_refinement'],
                    comparison_field_bytes=retained, wall_seconds=time.perf_counter()-started)
    if not evidence['passed']:
        worst = max(row['complex_max_normalized'] for row in evidence['samples'])
        raise RuntimeError('BoR spatial quadrature comparison failed (maximum normalized field change '
                           '{:.3g}; limit {:.3g}). Refine the geometry or investigate close surfaces/junctions.'
                           .format(worst, MAX_NORMALIZED_CHANGE))
    result['quadrature_comparison'] = evidence
    # The configured solve may have resolved automatic factorization or used
    # an admitted fallback. Keep those actual options in the returned evidence.
    result.setdefault('bor_execution_options', dict(fine_options))
    result['requested_bor_execution_options'] = dict(options)
    if 'modes_used' in result:
        result.setdefault('near_quadrature', {})['self_and_junction_convergence_checked'] = True
    metadata = result.get('metadata')
    if isinstance(metadata, dict):
        metadata['quadrature_comparison'] = evidence
        metadata.setdefault('bor_execution_options', dict(result['bor_execution_options']))
        metadata['requested_bor_execution_options'] = dict(options)
        for row in metadata.get('per_frequency', []):
            row.setdefault('near_quadrature', {})['self_and_junction_convergence_checked'] = True
    return result
