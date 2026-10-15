"""Explicit empirical populations shared by GUI and headless CDF plots.

Quadrature represents uniform measure over the selected coordinate hull,
including gaps. It is not dwell-time or spherical solid-angle weighting.
"""
from __future__ import annotations

import numpy as np
from .modes import common
from GRIM_Backend.datasets.memory import _bounded_grid_selections
from GRIM_Backend.datasets.reduction_workspace import workspace_array, BLOCK_CELLS


def _cancel(cancel):
    from .requests import check_cancel
    check_cancel(cancel)


def options(values=None):
    result = dict(exceedance=False, grouping="run", band_count=4,
                  sector_width_deg=30., weighting="samples", threshold_enabled=False, threshold=0.)
    result.update(dict(values or {}))
    if result["grouping"] not in {"run", "band", "sector", "band_sector"}:
        raise ValueError("Unknown CDF grouping")
    if result["weighting"] not in {"samples", "azimuth", "frequency", "azimuth_frequency", "all_axes"}:
        raise ValueError("Unknown CDF weighting")
    result["band_count"] = int(result["band_count"])
    result["sector_width_deg"] = float(result["sector_width_deg"])
    result["threshold"] = float(result["threshold"])
    if result["band_count"] < 1 or not np.isfinite(result["sector_width_deg"]) or result["sector_width_deg"] <= 0:
        raise ValueError("CDF band count and sector width must be positive")
    if not np.isfinite(result["threshold"]):
        raise ValueError("CDF threshold must be finite")
    return result


def owner_options(owner):
    controls = getattr(owner, "analysis_controls", None)
    getter = getattr(controls, "cdf_options", None)
    if callable(getter):
        return options(getter())
    return options({"exceedance": bool(controls is not None and controls.cdf_exceedance())})


def quadrature_weights(coordinates):
    """Normalized trapezoidal weights; coincident coordinates share weight."""
    x = np.asarray(coordinates, dtype=float)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) <= 1:
        return np.full(len(x), 1. / len(x))
    gaps = np.diff(unique)
    weights = np.r_[gaps[0] / 2., (gaps[:-1] + gaps[1:]) / 2., gaps[-1] / 2.]
    weights /= np.sum(weights)
    return weights[inverse] / counts[inverse]


def population(dataset, selection, *, scale="dbsm", weighting="samples", cancel=None):
    ai, ei, fi, pi = selection
    pi = int(np.asarray(pi).ravel()[0])
    selected_axes = {"samples": (), "azimuth": (0,), "frequency": (2,),
                     "azimuth_frequency": (0, 2), "all_axes": (0, 1, 2)}[weighting]
    coords = (np.asarray(dataset.azimuths)[ai], np.asarray(dataset.elevations)[ei], np.asarray(dataset.frequencies)[fi])
    # Each axis is normalized to prevent unit-dependent numerical overflow.
    ws = [quadrature_weights(v) if i in selected_axes else np.full(len(v), 1. / len(v))
          for i, v in enumerate(coords)]
    shape = (len(ai), len(ei), len(fi))
    selected = int(np.prod(shape))
    # Ordinals preserve stable ordering of equal levels without an O(N)
    # argsort index or a second whole-population gather. Large workspaces are
    # private mappings; quicksort needs only bounded stack space.
    records = (workspace_array(selected, np.dtype([("value", "f8"), ("ordinal", "i8"), ("weight", "f8")]))
               if weighting != "samples" else None)
    sample_values = workspace_array(selected, np.float64) if records is None else records["value"]
    count = zero_count = 0
    coverage = zero_weight = 0.
    indices = tuple(np.asarray(v) for v in (ai, ei, fi))
    for block in _bounded_grid_selections(shape, BLOCK_CELLS):
        _cancel(cancel)
        selected_indices = tuple(v[part] for v, part in zip(indices, block))
        power = np.asarray(dataset.power[np.ix_(*selected_indices, [pi])][..., 0], dtype=float)
        valid = np.isfinite(power) & (power >= 0.)
        zero = valid & (power == 0.)
        if scale == "linear":
            display = power
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                display = np.asarray(dataset.linear_to_default_db(
                    np.where(valid, power, np.nan),
                    frequency_value=np.asarray(dataset.frequencies)[selected_indices[2]][None, None, :], eps=0.), dtype=float)
            display[zero] = -np.inf
        included = valid & (np.isfinite(display) | zero)
        grid_weights = ws[0][block[0]][:, None, None] * ws[1][block[1]][None, :, None] * ws[2][block[2]][None, None, :]
        values, weights = display[included], grid_weights[included]
        stop = count + values.size
        sample_values[count:stop] = values
        # Traversal uses C-order contiguous logical tiles. Compute native
        # flattened order explicitly for shapes split on more than one axis.
        if records is not None:
            ordinal = ((np.arange(block[0].start, block[0].stop)[:, None, None] * shape[1]
                        + np.arange(block[1].start, block[1].stop)[None, :, None]) * shape[2]
                       + np.arange(block[2].start, block[2].stop)[None, None, :])
            records["ordinal"][count:stop] = ordinal[included]
            records["weight"][count:stop] = weights
        count = stop
        zero_count += int(np.count_nonzero(zero))
        coverage += float(np.sum(weights))
        zero_weight += float(np.sum(grid_weights[zero & included]))
    _cancel(cancel)
    if records is None:
        values = sample_values[:count]
        values.sort(kind="quicksort")
        weights = np.broadcast_to(ws[0][0] * ws[1][0] * ws[2][0], (count,))
    else:
        records = records[:count]
        records.sort(order=("value", "ordinal"), kind="quicksort")
        values, weights = records["value"], records["weight"]
    _cancel(cancel)
    counts = dict(selected=selected, valid=count, zero=zero_count, excluded=selected-count)
    return {"values": values, "weights": weights, "counts": counts,
            "coverage": coverage, "zero_weight": zero_weight}


def groups(dataset, reference, selection, config, *, azimuths, frequencies):
    ai, ei, fi, pi = selection
    az = common.values_for_display(reference, dataset, "azimuth", np.asarray(dataset.azimuths)[ai])
    fr = common.values_for_display(reference, dataset, "frequency", np.asarray(dataset.frequencies)[fi])
    ag = np.zeros(len(ai), dtype=int)
    fg = np.zeros(len(fi), dtype=int)
    grouping = config["grouping"]
    if grouping in {"sector", "band_sector"}:
        az_deg = common.convert_axis_values(az, "azimuth", common.axis_unit(reference, "azimuth"), "deg")
        selected_deg = common.convert_axis_values(azimuths, "azimuth", common.axis_unit(reference, "azimuth"), "deg")
        origin = float(np.min(selected_deg))
        ag = np.floor((az_deg - origin) / config["sector_width_deg"] + 1e-10).astype(int)
    if grouping in {"band", "band_sector"}:
        lo, hi = float(np.min(frequencies)), float(np.max(frequencies))
        edges = np.linspace(lo, hi, config["band_count"] + 1)
        fg = np.clip(np.searchsorted(edges, fr, side="right") - 1, 0, config["band_count"] - 1)
    for a in np.unique(ag):
        for f in np.unique(fg):
            parts = []
            if grouping in {"band", "band_sector"}:
                parts.append(f"Band {f+1}: {edges[f]:.6g}–{edges[f+1]:.6g} {common.axis_unit(reference, 'frequency')}")
            if grouping in {"sector", "band_sector"}:
                low = origin + a * config["sector_width_deg"]
                parts.append(f"Sector {low:g}–{low+config['sector_width_deg']:g} deg")
            yield (np.asarray(ai)[ag == a], ei, np.asarray(fi)[fg == f], pi), "; ".join(parts) or "All selected coordinates"


def prepare_curve(name, dataset, selection, group, config, *, scale, cancel=None):
    pop = population(dataset, selection, scale=scale, weighting=config["weighting"], cancel=cancel)
    values, weights = pop["values"], pop["weights"]
    if not len(values) or pop["coverage"] <= 0:
        return None
    levels = workspace_array(len(values), np.float64)
    masses = workspace_array(len(values), np.float64)
    distinct = 0
    for first in range(0, len(values), BLOCK_CELLS):
        _cancel(cancel)
        part = values[first:first + BLOCK_CELLS]
        starts = np.r_[0, np.flatnonzero(part[1:] != part[:-1]) + 1]
        block_levels = part[starts]
        block_masses = np.add.reduceat(weights[first:first + BLOCK_CELLS], starts)
        if distinct and block_levels[0] == levels[distinct - 1]:
            masses[distinct - 1] += block_masses[0]
            block_levels, block_masses = block_levels[1:], block_masses[1:]
        stop = distinct + len(block_levels)
        levels[distinct:stop], masses[distinct:stop] = block_levels, block_masses
        distinct = stop
    levels, masses = levels[:distinct], masses[:distinct]
    cdf, exceed = workspace_array(distinct, np.float64), workspace_array(distinct, np.float64)
    total = 0.
    for first in range(0, distinct, BLOCK_CELLS):
        _cancel(cancel)
        part = np.cumsum(masses[first:first + BLOCK_CELLS]) + total
        total = float(part[-1])
        cdf[first:first + BLOCK_CELLS] = part / pop["coverage"] * 100.
    total = 0.
    for stop in range(distinct, 0, -BLOCK_CELLS):
        _cancel(cancel)
        first = max(0, stop - BLOCK_CELLS)
        part = np.cumsum(masses[first:stop][::-1]) + total
        total = float(part[-1])
        exceed[first:stop] = part[::-1] / pop["coverage"] * 100.
    pi = int(np.asarray(selection[3]).ravel()[0])
    pol = dataset.polarizations[pi]
    unit = ({"sigma_3d": "m²", "sigma_2d": "m", "power_ratio": "dimensionless", "ratio": "dimensionless"}.get(str(dataset.linear_quantity()).lower(), "linear") if scale == "linear" else dataset.default_log_unit())
    weighted = config["weighting"] != "samples"
    if weighted:
        median = levels[min(int(np.searchsorted(cdf, 50.)), len(levels)-1)]
    else:
        middle = len(values) // 2
        median = float(values[middle]) if len(values) % 2 else float(np.mean([values[middle - 1], values[middle]]))
    counts = pop["counts"]
    label = (f"{name} | Pol {pol}, {counts['valid']:,} samples, "
             f"{'weighted P50' if weighted else 'median'} {median:.4g} {unit}")
    details = []
    if group != "All selected coordinates":
        details.append(group)
    if counts["excluded"] or counts["zero"]:
        details.append(f"valid {counts['valid']:,}/{counts['selected']:,}; {counts['zero']:,} zero;")
    if weighted:
        measure = {"azimuth":"az", "frequency":"freq", "azimuth_frequency":"az x freq", "all_axes":"az x el x freq"}[config["weighting"]]
        details.append(f"{measure} hull quadrature; {100*pop['coverage']:.4g}% valid weight")
    if details:
        label += "\n" + "\n".join(details)
    key = ("cdf", bool(config["exceedance"]), config["grouping"], group, config["weighting"],
           tuple(float(v) for v in dataset.azimuths[selection[0]]),
           tuple(float(v) for v in dataset.elevations[selection[1]]),
           tuple(float(v) for v in dataset.frequencies[selection[2]]), str(pol))
    # Sorted population is transient: retain only the distinct levels and mass.
    pop.pop("values"); pop.pop("weights")
    curve = dict(dataset=dataset, name=name, pol=str(pol), group=group, unit=unit,
                 levels=levels, cdf=cdf, exceed=exceed, population=pop, label=label, key=key)
    curve["lookup"] = threshold_lookup(curve, config["threshold"])
    return curve


def threshold_lookup(curve, threshold):
    """Exact inclusive tails from retained distinct levels, in O(log N)."""
    levels = curve["levels"]
    right = int(np.searchsorted(levels, threshold, side="right"))
    left = int(np.searchsorted(levels, threshold, side="left"))
    below = float(curve["cdf"][right - 1]) if right else 0.
    above = float(curve["exceed"][left]) if left < len(levels) else 0.
    return below, above


def bounds(curves, threshold=None):
    finite = [c["levels"][1:] if np.isneginf(c["levels"][0]) else c["levels"] for c in curves if len(c["levels"])]
    finite = [v for v in finite if v.size]
    low, high = (min(v[0] for v in finite), max(v[-1] for v in finite)) if finite else (-1., 1.)
    if threshold is not None:
        low, high = min(low, threshold), max(high, threshold)
    pad = max(1e-9, .02 * (high-low))
    return float(low-pad), float(high+pad)


def draw_values(curve, limits, *, exceedance=False):
    levels, pop = curve["levels"], curve["population"]
    first = int(bool(len(levels) and np.isneginf(levels[0])))
    zero_percent = 100. * pop["zero_weight"] / pop["coverage"]
    left = (100.-zero_percent if exceedance else zero_percent) if first else (100. if exceedance else 0.)
    size = len(levels) - first + 2
    x, y = workspace_array(size, np.float64), workspace_array(size, np.float64)
    x[0], x[-1], y[0], y[-1] = limits[0], limits[1], left, 0. if exceedance else 100.
    cumulative = curve["exceed" if exceedance else "cdf"]
    for start in range(first, len(levels), BLOCK_CELLS):
        stop = min(len(levels), start + BLOCK_CELLS)
        target = slice(start - first + 1, stop - first + 1)
        x[target], y[target] = levels[start:stop], cumulative[start:stop]
    return x, y


def y_label(config):
    subject = "Samples" if config["weighting"] == "samples" else "Selected coordinate measure"
    return f"{subject} at or {'above' if config['exceedance'] else 'below'} level (%)"


def table_text(curves, config):
    definition = ("Equal observation weights." if config["weighting"] == "samples" else
                  "Normalized trapezoidal quadrature over each selected group coordinate hull, including gaps; singleton axes have unit weight. Not dwell-time or solid angle.")
    header = "Run\tPolarization\tGroup\tUnit\tValid\tSelected\tZero\tValid weight (%)"
    if config["threshold_enabled"]:
        header += "\tThreshold\tP(X <= threshold) (%)\tP(X >= threshold) (%)"
    rows = [definition, "Runs and polarizations remain separate; missing data excluded; zero power retained. Inclusive tails both include equality.", header]
    for c in curves:
        p, counts = c["population"], c["population"]["counts"]
        row = [c["name"], c["pol"], c["group"], c["unit"], counts["valid"], counts["selected"], counts["zero"], f"{100*p['coverage']:.9g}"]
        if config["threshold_enabled"]:
            row += [f"{config['threshold']:.9g}", *(f"{v:.9g}" for v in c["lookup"])]
        rows.append("\t".join(map(str, row)))
    return "\n".join(rows)
