"""Sample-supported image cells and cuts on physical coordinate axes.

Local cadence separates disconnected selections without treating regular
selection strides or changes in acquisition density as missing samples.
Gap cells are display placeholders only, never additional native observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GRIM_Backend.isar.geometry import angular_bands


def native_sample_width(reference, dataset, axis):
    """Native spacing in displayed units, with a visible singleton fallback."""
    from GRIM_Backend.plotting.modes import common
    attribute = {"azimuth": "azimuths", "elevation": "elevations", "frequency": "frequencies"}[axis]
    values = common.values_for_display(reference, dataset, axis, getattr(dataset, attribute))
    steps = np.diff(np.unique(values))
    if steps.size:
        return float(np.min(steps))
    unit = "MHz" if axis == "frequency" else "deg"
    return float(common.convert_axis_values([1.], axis, unit, common.axis_unit(reference, axis))[0])


@dataclass
class AxisSupport:
    edges: np.ndarray
    indices: np.ndarray
    bands: list

    def index(self, coordinate):
        if coordinate is None or not np.isfinite(coordinate):
            return None
        if coordinate < self.edges[0] or coordinate > self.edges[-1]:
            return None
        cell = min(len(self.indices)-1, int(np.searchsorted(self.edges, coordinate, side="right"))-1)
        index = int(self.indices[cell])
        return index if index >= 0 else None


def axis_support(values, *, width=1.):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError("Image coordinates must be finite and strictly increasing")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Image singleton display width must be positive")
    bands = angular_bands(range(len(values)), values)
    bounds = []
    for band in bands:
        centers = values[band]
        if len(band) == 1:
            bounds.append(np.array([centers[0]-width/2, centers[0]+width/2]))
        else:
            steps = np.diff(centers)
            bounds.append(np.r_[centers[0]-steps[0]/2, centers[:-1]+steps/2, centers[-1]+steps[-1]/2])
    for i in range(len(bands)-1):
        left, right = values[bands[i][-1]], values[bands[i+1][0]]
        gap = right-left
        bounds[i][-1] = min(bounds[i][-1], left+gap/4)
        bounds[i+1][0] = max(bounds[i+1][0], right-gap/4)
    edges, indices = [bounds[0][0]], []
    for i, (band, cell_edges) in enumerate(zip(bands, bounds)):
        if i:
            indices.append(-1)
            edges.append(cell_edges[0])
        indices.extend(band)
        edges.extend(cell_edges[1:])
    return AxisSupport(np.asarray(edges), np.asarray(indices, dtype=int), bands)


def _display_axis(support, target):
    """Pool within support only; retain a masked cell between every band."""
    count = sum(map(len, support.bands))
    target = max(1, min(count, int(target)))
    starts = set((np.arange(target, dtype=np.int64)*count)//target)
    # Split global blocks at every support boundary. No block spans a gap.
    starts.update(band[0] for band in support.bands)
    starts = np.asarray(sorted(starts), dtype=int)
    lookup = {int(native): cell for cell, native in enumerate(support.indices) if native >= 0}
    edges, mapping, centers = [support.edges[0]], [], []
    for i, band in enumerate(support.bands):
        if i:
            mapping.append(-1)
            centers.append(np.nan)
            edges.append(support.edges[lookup[band[0]]])
        band_starts = starts[(starts >= band[0]) & (starts <= band[-1])]
        stops = np.r_[band_starts[1:], band[-1]+1]
        for first, stop in zip(band_starts, stops):
            mapping.append(int(np.searchsorted(starts, first)))
            centers.append((first+stop-1)//2)
            edges.append(support.edges[lookup[int(stop-1)]+1])
    return np.asarray(edges), starts, np.asarray(mapping), np.asarray(centers)


def image_grid(x, y, values, *, x_width=1., y_width=1., target_shape=None, return_samples=False):
    """Return explicit X/Y edges and a row-by-column, gap-masked image.

    ``target_shape`` is (rows, columns). Band boundaries may add a few blocks
    beyond the requested count so no measured sector is dropped or joined.
    """
    values = np.ma.filled(values, np.nan)
    if values.shape != (len(y), len(x)):
        raise ValueError("Image dimensions must match coordinate axes")
    xs, ys = axis_support(x, width=x_width), axis_support(y, width=y_width)
    rows, columns = values.shape if target_shape is None else target_shape
    xe, xi, xm, xc = _display_axis(xs, columns)
    ye, yi, ym, yc = _display_axis(ys, rows)
    reduced = np.fmax.reduceat(np.fmax.reduceat(values, yi, axis=0), xi, axis=1)
    output = reduced[np.ix_(np.maximum(ym, 0), np.maximum(xm, 0))]
    output[(ym < 0), :] = np.nan
    output[:, (xm < 0)] = np.nan
    if not return_samples:
        return xe, ye, output
    # If native snapshot storage is unavailable, cuts must refer to these
    # same pooled blocks. Keep only measured-block representatives; synthetic
    # gap cells are not sample coordinates in snap controls or image records.
    samples = {
        "x": np.asarray(x)[xc[np.isfinite(xc)].astype(int)],
        "y": np.asarray(y)[yc[np.isfinite(yc)].astype(int)],
        "values": reduced[np.ix_(ym[ym >= 0], xm[xm >= 0])],
        "x_cut_breaks": np.cumsum(xm >= 0)[xm < 0].tolist(),
        "y_cut_breaks": np.cumsum(ym >= 0)[ym < 0].tolist(),
    }
    return xe, ye, output, samples


def line_with_gaps(coordinates, values, *, breaks=None):
    """Break a native cut between disconnected coordinate bands."""
    coordinates, values = np.asarray(coordinates), np.asarray(values)
    bands = (angular_bands(range(len(coordinates)), coordinates) if breaks is None
             else np.split(np.arange(len(coordinates)), breaks))
    if len(bands) == 1:
        return coordinates, values
    x, y = [], []
    for i, band in enumerate(bands):
        if i:
            x.append(np.array([(coordinates[band[0]-1]+coordinates[band[0]])/2]))
            y.append(np.array([np.nan]))
        x.append(coordinates[band]); y.append(values[band])
    return np.concatenate(x), np.concatenate(y)
