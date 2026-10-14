"""Explicit empirical populations shared by GUI and headless CDF plots.

Quadrature represents uniform measure over the selected coordinate hull,
including gaps. It is not dwell-time or spherical solid-angle weighting.
"""
from __future__ import annotations

import numpy as np
from .modes import common


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


def population(dataset, selection, *, scale="dbsm", weighting="samples"):
    ai, ei, fi, pi = selection
    pi = int(np.asarray(pi).ravel()[0])
    power = np.asarray(dataset.power[np.ix_(ai, ei, fi, [pi])][..., 0], dtype=float)
    valid = np.isfinite(power) & (power >= 0.)
    zero = valid & (power == 0.)
    if scale == "linear":
        display = power
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            display = np.asarray(dataset.linear_to_default_db(
                np.where(valid, power, np.nan),
                frequency_value=np.asarray(dataset.frequencies)[fi][None, None, :], eps=0.), dtype=float)
        display[zero] = -np.inf
    included = valid & (np.isfinite(display) | zero)
    selected_axes = {"samples": (), "azimuth": (0,), "frequency": (2,),
                     "azimuth_frequency": (0, 2), "all_axes": (0, 1, 2)}[weighting]
    coords = (np.asarray(dataset.azimuths)[ai], np.asarray(dataset.elevations)[ei], np.asarray(dataset.frequencies)[fi])
    # Each axis is normalized to prevent unit-dependent numerical overflow.
    ws = [quadrature_weights(v) if i in selected_axes else np.full(len(v), 1. / len(v))
          for i, v in enumerate(coords)]
    grid_weights = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    values, weights = display[included], grid_weights[included]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    counts = dict(selected=power.size, valid=values.size, zero=int(np.count_nonzero(zero)), excluded=int(power.size-values.size))
    return {"values": values, "weights": weights, "counts": counts,
            "coverage": float(np.sum(weights)), "zero_weight": float(np.sum(grid_weights[zero & included]))}


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


def prepare_curve(name, dataset, selection, group, config, *, scale):
    pop = population(dataset, selection, scale=scale, weighting=config["weighting"])
    values, weights = pop["values"], pop["weights"]
    if not len(values) or pop["coverage"] <= 0:
        return None
    levels, starts = np.unique(values, return_index=True)
    masses = np.add.reduceat(weights, starts)
    cdf = np.cumsum(masses) / pop["coverage"] * 100.
    exceed = np.cumsum(masses[::-1])[::-1] / pop["coverage"] * 100.
    pi = int(np.asarray(selection[3]).ravel()[0])
    pol = dataset.polarizations[pi]
    unit = ({"sigma_3d": "m²", "sigma_2d": "m", "power_ratio": "dimensionless", "ratio": "dimensionless"}.get(str(dataset.linear_quantity()).lower(), "linear") if scale == "linear" else dataset.default_log_unit())
    weighted = config["weighting"] != "samples"
    if weighted:
        median = levels[min(int(np.searchsorted(cdf, 50.)), len(levels)-1)]
    else:
        median = float(np.median(values))
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
    threshold = config["threshold"]
    lookup = (100. * np.sum(weights[values <= threshold]) / pop["coverage"],
              100. * np.sum(weights[values >= threshold]) / pop["coverage"])
    key = ("cdf", bool(config["exceedance"]), config["grouping"], group, config["weighting"],
           tuple(float(v) for v in dataset.azimuths[selection[0]]),
           tuple(float(v) for v in dataset.elevations[selection[1]]),
           tuple(float(v) for v in dataset.frequencies[selection[2]]), str(pol))
    # Sorted population is transient: retain only the distinct levels and mass.
    pop.pop("values"); pop.pop("weights")
    return dict(dataset=dataset, name=name, pol=str(pol), group=group, unit=unit,
                levels=levels, cdf=cdf, exceed=exceed, population=pop, label=label, key=key, lookup=lookup)


def bounds(curves, threshold=None):
    finite = [c["levels"][np.isfinite(c["levels"])] for c in curves]
    finite = [v for v in finite if v.size]
    low, high = (min(v[0] for v in finite), max(v[-1] for v in finite)) if finite else (-1., 1.)
    if threshold is not None:
        low, high = min(low, threshold), max(high, threshold)
    pad = max(1e-9, .02 * (high-low))
    return float(low-pad), float(high+pad)


def draw_values(curve, limits, *, exceedance=False):
    levels, pop = curve["levels"], curve["population"]
    finite = np.isfinite(levels)
    zero_percent = 100. * pop["zero_weight"] / pop["coverage"]
    left = (100.-zero_percent if exceedance else zero_percent) if np.any(np.isneginf(levels)) else (100. if exceedance else 0.)
    return (np.r_[limits[0], levels[finite], limits[1]],
            np.r_[left, curve["exceed" if exceedance else "cdf"][finite], 0. if exceedance else 100.])


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
