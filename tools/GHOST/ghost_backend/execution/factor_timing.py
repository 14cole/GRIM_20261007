"""Exact-workload evidence for the LU versus HODLR crossover.

Only complete successful runs count as samples. Polarizations of one run
are summed, never misrepresented as independent repeats. These choices
change the existing factor selection; they do not bypass memory admission.
"""
import math
import os
import statistics
import time

MIN_SAMPLES = 2
MAX_SAMPLES = 5
MAX_SPREAD = 1.15
WIN_MARGIN = .05


def descriptor(payload, options):
    from ghost_backend.execution.timing_history import _digest
    from ghost_backend.execution.options import allocated_cpu_budget, allocated_memory_budget
    family = dict(payload)
    family['options'] = {k: v for k, v in options.items()
                         if k not in ('factorization', 'temporary_directory')}
    family['allocation'] = (allocated_cpu_budget(), allocated_memory_budget())
    # A deliberate threshold override is how a paired measurement is made.
    # Every other algorithm/runtime override remains part of its identity.
    family['runtime_overrides'] = {k: v for k, v in os.environ.items()
        if k.startswith('GHOST_') and k not in
        ('GHOST_TIMING_CACHE_DIR', 'GHOST_HIERARCHICAL_MIN_UNKNOWNS', 'GHOST_CPU_FACTORIZATION')}
    return _digest(family)


def observation(key, metadata):
    family = getattr(key, 'factor', None)
    systems = metadata.get('experimental_cpu', {}).get('systems', [])
    if not family or not isinstance(systems, list) or not systems:
        return None
    groups = {}
    for event in systems:
        if not isinstance(event, dict):
            return None
        variant = event.get('factor_variant')
        seconds = event.get('factor_work_seconds')
        n = event.get('unknowns')
        if (variant not in ('lu', 'hodlr') or type(n) is not int or n <= 0
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
                or event.get('factor_work_failed') or event.get('factor_fallback')
                or event.get('factor_rebuilds') or event.get('factorizations') != 1):
            return None
        group = groups.setdefault(n, dict(variant=variant, seconds=0., systems=0))
        if group['variant'] != variant:
            return None
        group['seconds'] += seconds
        group['systems'] += 1
    return dict(family=family, time=time.time(), groups={str(k): v for k, v in groups.items()})


def choices(key, entries, max_age):
    family = getattr(key, 'factor', None)
    if not family:
        return {}, {}
    groups = {}
    now = time.time()
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        rows = entry.get('_factorizations', [])
        if not isinstance(rows, list):
            continue
        for row in rows[-MAX_SAMPLES:]:
            if not isinstance(row, dict) or row.get('family') != family:
                continue
            stamp = row.get('time')
            if (not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    or not 0 <= now-stamp <= max_age or not isinstance(row.get('groups'), dict)):
                continue
            for size, sample in row['groups'].items():
                try:
                    n = int(size)
                except (TypeError, ValueError):
                    continue
                if not isinstance(sample, dict):
                    continue
                variant, seconds, count = sample.get('variant'), sample.get('seconds'), sample.get('systems')
                if (n <= 0 or variant not in ('lu', 'hodlr') or type(count) is not int or count <= 0
                        or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0):
                    continue
                groups.setdefault(n, {}).setdefault(count, {}).setdefault(variant, []).append((stamp, seconds))
    selected, evidence = {}, {}
    for n, counts in groups.items():
        # Different counts imply a changed adaptive/system workload.
        if len(counts) != 1:
            continue
        count, samples = next(iter(counts.items()))
        values = {v: [seconds for _, seconds in sorted(samples.get(v, []))[-MAX_SAMPLES:]]
                  for v in ('lu', 'hodlr')}
        if any(len(rows) < MIN_SAMPLES or max(rows)/min(rows) > MAX_SPREAD for rows in values.values()):
            continue
        medians = {v: statistics.median(rows) for v, rows in values.items()}
        winner = min(medians, key=medians.get)
        other = 'lu' if winner == 'hodlr' else 'hodlr'
        # Require separated repeated samples, not just a favorable median.
        if max(values[winner]) > (1.-WIN_MARGIN)*min(values[other]):
            continue
        selected[n] = winner
        evidence[str(n)] = dict(selected=winner, systems_per_run=count,
            samples={v: len(rows) for v, rows in values.items()}, median_seconds=medians)
    return selected, evidence
