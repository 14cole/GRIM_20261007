"""Bounded Cartesian interpolation with consistent measured support.

Windowed-sinc interpolation is used only on fully observed regular stencils.
Nonuniform axes use local polynomial interpolation on well-conditioned stencils.
At a hole, both the zero-weighted field and coverage use the same positive
linear stencil. An origin point therefore has identical numerator/weight.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import numpy as np

# Tabulate fractional delays at 1/4096-sample spacing. The half-MiB table
# avoids millions of repeated special functions with sub-microlevel error.
_SINC_RADIUS = 16
_SINC_STEPS = 4096
_delay = np.linspace(0., 1., _SINC_STEPS + 1)[:, None] - np.arange(1-_SINC_RADIUS, _SINC_RADIUS+1)
_SINC_TABLE = (np.sinc(_delay) * np.i0(6. * np.sqrt(np.maximum(0., 1.-(_delay/_SINC_RADIUS)**2))) / np.i0(6.)).astype(np.float32)
_SINC_TABLE.flags.writeable = False
del _delay
_PLAN_BYTES = 8 * 1024**2 - _SINC_TABLE.nbytes
_PLANS = OrderedDict()
_LOCK = threading.RLock()
_BYTES = 0


def plan_cache_bytes():
    with _LOCK:
        return _BYTES + _SINC_TABLE.nbytes


def _plan(coordinates, n):
    global _BYTES
    # Cache small blocks only. Coordinate hashes bind reuse to actual geometry.
    raw = np.ascontiguousarray(coordinates, dtype=np.float64)
    cacheable = raw.size * 10 <= _PLAN_BYTES // 4
    key = (n, raw.shape, hashlib.blake2b(memoryview(raw).cast('B'), digest_size=16).digest()) if cacheable else None
    with _LOCK:
        cached = _PLANS.get(key)
        if cached is not None:
            _PLANS.move_to_end(key)
            return cached
    base = np.clip(np.floor(raw), 0, n - 2).astype(np.int32)
    t = (raw - base).astype(np.float32)
    valid = (raw >= -1e-8) & (raw <= n - 1 + 1e-8)
    np.clip(t, 0, 1, out=t)
    interior = (base >= 1) & (base <= n - 3)
    value = (base, t, valid, interior)
    nbytes = sum(a.nbytes for a in value)
    if nbytes <= _PLAN_BYTES // 4:
        for a in value:
            a.flags.writeable = False
        with _LOCK:
            old = _PLANS.pop(key, None)
            if old is not None:
                _BYTES -= sum(a.nbytes for a in old)
            _PLANS[key] = value
            _BYTES += nbytes
            while _BYTES > _PLAN_BYTES:
                _, old = _PLANS.popitem(last=False)
                _BYTES -= sum(a.nbytes for a in old)
    return value


def interpolate_pair(field, coverage, coordinates):
    """Apply one geometry plan to a complex numerator and positive coverage."""
    n = field.shape[0]
    base, t, valid, interior = _plan(coordinates, n)
    take = lambda a, j: np.take_along_axis(a, j, axis=0)
    f0, f1 = take(field, base), take(field, base + 1)
    w0, w1 = take(coverage, base), take(coverage, base + 1)
    out = f0 + (f1 - f0) * t
    weight = w0 + (w1 - w0) * t
    if np.any(interior):
        im1, i2 = np.maximum(base - 1, 0), np.minimum(base + 2, n - 1)
        wm1, w2 = take(coverage, im1), take(coverage, i2)
        complete = interior & (w0 == 1) & (w1 == 1) & (wm1 == 1) & (w2 == 1)
        if np.any(complete):
            cm1 = -t * (t - 1) * (t - 2) / 6
            c1 = -(t + 1) * t * (t - 2) / 2
            c2 = (t + 1) * t * (t - 1) / 6
            # Difference form preserves constant fields exactly despite roundoff.
            cubic = f0 + (take(field, im1) - f0) * cm1 + (f1 - f0) * c1 + (take(field, i2) - f0) * c2
            out[complete] = cubic[complete]
            weight[complete] = 1
    # Cubic interpolation appreciably attenuates a complex sinusoid well below
    # Nyquist. A 32-tap Kaiser-windowed sinc preserves those off-origin returns.
    # Accumulate one tap at a time: scratch is bounded by the caller's block,
    # rather than constructing a (samples, channels, taps) temporary.
    radius = _SINC_RADIUS
    complete = valid & (base >= radius - 1) & (base + radius < n)
    if np.any(complete):
        numerator = np.zeros_like(out)
        denominator = np.zeros_like(t)
        table_position = t * _SINC_STEPS
        table_index = np.minimum(table_position.astype(np.int32), _SINC_STEPS-1)
        mix = table_position - table_index
        for offset in range(1 - radius, radius + 1):
            index = np.clip(base + offset, 0, n - 1)
            complete &= take(coverage, index) == 1
            tap = offset + radius - 1
            k0 = _SINC_TABLE[table_index, tap]
            kernel = (k0 + (_SINC_TABLE[table_index+1, tap] - k0) * mix).astype(np.float32)
            numerator += (take(field, index) - f0) * kernel
            denominator += kernel
        out[complete] = (f0 + numerator / denominator)[complete]
        weight[complete] = 1
    out[~valid] = 0
    weight[~valid] = 0
    return out, weight


def resample_pair(source, field, coverage, target, *, axis, support, cancel_check=None):
    """Resample phase history without bridging holes or missing phase.

    Up to 16 local samples are used, symmetrically about the interpolation
    interval. Ill-conditioned, boundary, and incomplete stencils fall back to
    the same positive linear stencil for numerator and coverage. Geometry is
    shared by all channels; no interpolation of wrapped phase is performed.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if np.array_equal(source, target):
        return field, coverage
    values, weights = np.moveaxis(field, axis, 0), np.moveaxis(coverage, axis, 0)
    if values.ndim != 2 or weights.shape != values.shape:
        raise ValueError('Phase resampling requires matching two-dimensional field and coverage')
    n = len(source)
    right = np.clip(np.searchsorted(source, target), 1, n - 1)
    left = right - 1
    fraction = np.clip((target - source[left]) / (source[right] - source[left]), 0, 1).astype(np.float32)
    out = values[left] + (values[right] - values[left]) * fraction[:, None]
    weight = weights[left] + (weights[right] - weights[left]) * fraction[:, None]
    steps = np.diff(source)
    cadence = np.median(np.sort(steps)[:max(1, (len(steps) + 1)//2)])
    # A stencil stays inside one contiguous acquisition sector.
    sectors = np.r_[0, np.cumsum(steps > 2.5 * cadence)]
    tap_count = np.minimum(16, 2 * np.minimum(right, n - right))
    for taps in (4, 6, 8, 10, 12, 14, 16):
        selected = np.flatnonzero((tap_count == taps) & np.asarray(support))
        for start in range(0, len(selected), 2048):
            if cancel_check is not None and cancel_check():
                raise InterruptedError('ISAR computation cancelled')
            ids = selected[start:start + 2048]
            indices = right[ids, None] + np.arange(-taps//2, taps//2)[None, :]
            contiguous = sectors[indices[:, 0]] == sectors[indices[:, -1]]
            # Local coordinates avoid cancellation at GHz-scale frequencies.
            nodes = (source[indices] - source[left[ids], None]) / (source[right[ids]] - source[left[ids]])[:, None]
            t = fraction[ids].astype(float)
            coefficients = np.ones(nodes.shape)
            for j in range(taps):
                for k in range(taps):
                    if j != k:
                        coefficients[:, j] *= (t - nodes[:, k]) / (nodes[:, j] - nodes[:, k])
            stable = contiguous & np.all(np.isfinite(coefficients), axis=1) & (np.sum(abs(coefficients), axis=1) <= 4)
            coefficients[~stable] = 0
            coefficients = coefficients.astype(np.float32)
            for column in range(0, values.shape[1], max(1, 65536//len(ids))):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError('ISAR computation cancelled')
                end = min(values.shape[1], column + max(1, 65536//len(ids)))
                reference = values[left[ids], column:end]
                interpolated = reference.copy()
                complete = np.broadcast_to(stable[:, None], reference.shape).copy()
                for j in range(taps):
                    complete &= weights[indices[:, j], column:end] == 1
                    interpolated += (values[indices[:, j], column:end] - reference) * coefficients[:, j, None]
                out[ids, column:end] = np.where(complete, interpolated, out[ids, column:end])
                weight[ids, column:end] = np.where(complete, 1, weight[ids, column:end])
    out[~np.asarray(support)] = 0
    weight[~np.asarray(support)] = 0
    return np.moveaxis(out, 0, axis), np.moveaxis(weight, 0, axis)


def cartesian_pair(field, coverage, theta, frequency, axes, *, block_size=128, cancel_check=None):
    """Two Cartesian passes; share coordinate construction across both arrays."""
    q, axis_q, v = axes
    theta, frequency = np.asarray(theta), np.asarray(frequency)
    psi = theta - theta.mean()
    fc = frequency.mean()
    az_field = np.empty_like(field)
    az_weight = np.empty_like(coverage)
    az_block = min(block_size, max(1, 65536 // len(theta)))
    for start in range(0, len(frequency), az_block):
        if cancel_check is not None and cancel_check():
            raise InterruptedError('ISAR computation cancelled')
        end = min(start + az_block, len(frequency))
        arg = q[:, None] / (frequency[None, start:end] / fc)
        coordinates = (np.arcsin(np.clip(arg, -1, 1)) - psi[0]) / np.mean(np.diff(psi))
        f, w = interpolate_pair(field[:, start:end], coverage[:, start:end], coordinates)
        f[np.abs(arg) > 1] = 0
        w[np.abs(arg) > 1] = 0
        az_field[:, start:end], az_weight[:, start:end] = f, w
    out, weight = np.empty_like(field), np.empty_like(coverage)
    u = fc * q
    range_block = min(block_size, max(1, 65536 // len(frequency)))
    for start in range(0, len(q), range_block):
        if cancel_check is not None and cancel_check():
            raise InterruptedError('ISAR computation cancelled')
        end = min(start + range_block, len(q))
        required_f = np.sqrt(v[None, :]**2 + u[start:end, None]**2)
        coordinates = (required_f - frequency[0]) / np.mean(np.diff(frequency))
        f, w = interpolate_pair(az_field[start:end].T, az_weight[start:end].T, coordinates.T)
        out[start:end], weight[start:end] = f.T, w.T
    return out, weight, axis_q, v
