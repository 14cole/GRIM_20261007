"""Persistent, input-verified BoR frequency resume with BoR metadata semantics."""
import json
import math


def _combine(items, field=''):
    """Conservative diagnostic summary; individual evidence remains available."""
    if all(isinstance(value, dict) for value in items):
        return {key: _combine([value[key] for value in items if key in value], key)
                for key in dict.fromkeys(key for value in items for key in value)}
    if all(isinstance(value, bool) for value in items):
        return all(items)
    if all(isinstance(value, (int, float)) for value in items):
        if field.endswith('_count'):
            return sum(items)
        if any(not math.isfinite(value) for value in items):
            return next(value for value in items if not math.isfinite(value))
        if field == 'mesh_wavelength_m' or field.endswith('_min'):
            return min(items)
        return max(items)
    if all(isinstance(value, list) for value in items):
        result, seen = [], set()
        for value in items:
            for entry in value:
                identity = json.dumps(entry, sort_keys=True, default=lambda value: value.item())
                if identity not in seen:
                    seen.add(identity)
                    result.append(entry)
        return result
    return items[0]


def merge_frequency_results(results, frequencies):
    from ghost_backend.twod.samples import frequency_buffer, sorted_samples, sample_column
    result, records = None, []
    for value in results:
        if value.get('solver') != 'bor_mom_rcs':
            raise ValueError('Expected a BoR frequency checkpoint.')
        if result is None:
            result = {key: entry for key, entry in value.items()
                      if key not in ('samples', 'co_solved_samples', 'metadata')}
            result['samples'] = frequency_buffer()
            result['co_solved_samples'] = {pol: frequency_buffer() for pol in ('VV', 'HH')}
        result['samples'].extend(value['samples'])
        for pol in ('VV', 'HH'):
            result['co_solved_samples'][pol].extend(value['co_solved_samples'][pol])
        records.append(value['metadata'])
    if result is None or len(records) != len(frequencies):
        raise ValueError('Expected one completed BoR result per requested frequency.')
    metadata = _combine(records)
    # These describe the unchanged angular request, rather than accumulating
    # diagnostics. BoR combined rows use numeric frequency/channel/angle order;
    # per-channel rows retain requested order unless 360-degree expansion sorts.
    for key in ('aspect_count', 'elevation_count', 'output_aspect_count'):
        if key in records[0]:
            metadata[key] = records[0][key]
    metadata['frequency_count'] = len(frequencies)
    metadata['frequency_metadata'] = [dict(frequency_ghz=float(frequency), metadata=record)
                                      for frequency, record in zip(frequencies, records)]
    metadata['per_frequency'] = [entry for record in records for entry in record.get('per_frequency', [])]
    if metadata.get('mesh_convergence'):
        metadata['mesh_convergence']['aggregation'] = (
            'Each frequency independently certified; summary errors are worst-frequency values, '
            'not a recomputed sweep RMS. See frequency_metadata for complete evidence.')
    metadata['warning_count'] = len(metadata.get('warnings', []))
    compact = all(sample_column(result['samples'], key) is not None
                  for key in ('frequency_ghz', 'polarization', 'theta_inc_deg'))
    result['samples'] = (sorted(result['samples'], key=lambda row: (
        row['frequency_ghz'], row['polarization'], row['theta_inc_deg']))
        if not compact else sorted_samples(
            result['samples'], keys=('frequency_ghz', 'polarization', 'theta_inc_deg')))
    if metadata.get('expanded_to_360'):
        for pol, rows in result['co_solved_samples'].items():
            result['co_solved_samples'][pol] = sorted_samples(rows, keys=('frequency_ghz', 'theta_inc_deg'))
    result['metadata'] = metadata
    return result


def run_checkpointed(solve, arguments, directory, options, certified, *, frequency_workers=1):
    from ghost_backend.twod.checkpoints import run_checkpointed as run
    from ghost_backend.bor.options import _OUTPUT_GB, estimate_output_gb, output_reserved_gb
    reserved = estimate_output_gb(len(arguments['frequencies_ghz']),
        len(arguments['elevations_deg']), certified, bool(arguments.get('expand_to_360', False)))
    with _OUTPUT_GB.override(max(reserved, output_reserved_gb())):
        return run(solve, arguments, directory, options, 'double', certified,
                   solver_kind='bor', merge=merge_frequency_results,
                   frequency_workers=frequency_workers)
