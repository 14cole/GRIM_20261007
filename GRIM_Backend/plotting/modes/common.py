"""Qt-free helpers shared by the Plotting renderers.

The GUI parameter lists contain values in the active dataset's native units.
These helpers make that reference frame explicit, convert selections for each
other dataset, and keep display-only bounding logic out of the data model.
"""

from __future__ import annotations

import warnings
import tempfile
from dataclasses import dataclass, field

import numpy as np


MAX_LINE_SERIES = 128
MAX_LINE_POINTS = 20_000
TOTAL_LINE_POINT_TARGET = 256_000
# Scratch storage bounds for reductions; independent of input array size.
REDUCTION_BLOCK_CELLS = 262_144
MAX_WATERFALL_PANELS = 24
MAX_IMAGE_SIDE = 2_048
MAX_IMAGE_CELLS = 2_000_000
MAX_TOTAL_IMAGE_CELLS = 8_000_000
MAX_SYNC_SLICE_CELLS = 4_000_000
MAX_SYNC_TOTAL_CELLS = 16_000_000
MAX_EXPLICIT_TICKS = 200


_FREQUENCY_UNITS = {
    "hz": ("Hz", 1.0),
    "khz": ("kHz", 1.0e3),
    "mhz": ("MHz", 1.0e6),
    "ghz": ("GHz", 1.0e9),
}
_ANGLE_UNITS = {
    "deg": ("deg", np.pi / 180.0),
    "degree": ("deg", np.pi / 180.0),
    "degrees": ("deg", np.pi / 180.0),
    "rad": ("rad", 1.0),
    "radian": ("rad", 1.0),
    "radians": ("rad", 1.0),
}


def axis_unit(dataset, axis: str) -> str:
    """Return a supported canonical unit for an RcsGrid axis."""

    if axis == "frequency":
        raw = str((dataset.units or {}).get(axis, "GHz")).strip().lower()
        entry = _FREQUENCY_UNITS.get(raw)
    elif axis in {"azimuth", "elevation"}:
        raw = str((dataset.units or {}).get(axis, "deg")).strip().lower()
        entry = _ANGLE_UNITS.get(raw)
    else:
        raise ValueError(f"unsupported plot axis {axis!r}")
    if entry is None:
        raise ValueError(f"unsupported {axis} unit {raw!r}")
    return entry[0]


def _unit_scale(axis: str, unit: str) -> float:
    table = _FREQUENCY_UNITS if axis == "frequency" else _ANGLE_UNITS
    entry = table.get(str(unit).strip().lower())
    if entry is None:
        raise ValueError(f"unsupported {axis} unit {unit!r}")
    return float(entry[1])


def convert_axis_values(values, axis: str, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert frequency or angle values without changing the stored dataset."""

    values = np.asarray(values, dtype=float)
    return values * (_unit_scale(axis, from_unit) / _unit_scale(axis, to_unit))


def selection_for_dataset(reference, dataset, axis: str, values) -> tuple[np.ndarray, float]:
    """Convert reference-list values to a dataset's native unit and tolerance."""

    reference_unit = axis_unit(reference, axis)
    dataset_unit = axis_unit(dataset, axis)
    converted = convert_axis_values(values, axis, reference_unit, dataset_unit)
    tolerance = axis_matching_tolerance(dataset, axis)
    return converted, tolerance


def unique_axis_selection(axis_values, requested, tolerance):
    """Nearest, one-to-one numeric matches, giving exact matches precedence.

    Unlike the general dataset selector this must not deduplicate indices:
    plotting a shared X grid requires a measurement for each requested X.
    """
    native = np.asarray(axis_values, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if native.ndim != 1 or not np.all(np.isfinite(native)):
        raise ValueError("source coordinates must be finite and one-dimensional")
    order = np.argsort(native, kind="stable")
    ordered = native[order]
    if not len(ordered) or not np.all(np.isfinite(requested)):
        return None
    positions = np.searchsorted(ordered, requested)
    lo = np.clip(positions - 1, 0, len(ordered) - 1)
    hi = np.clip(positions, 0, len(ordered) - 1)
    dl, dh = np.abs(ordered[lo] - requested), np.abs(ordered[hi] - requested)
    nearest = np.where(dl < dh, lo, hi)
    distance = np.minimum(dl, dh)
    if np.any(distance > tolerance):
        return None
    roundoff = np.spacing(np.maximum(np.abs(requested), 1.0)) * 4
    tied = (lo != hi) & (distance > 0) & (np.abs(dl - dh) <= roundoff)
    if np.any(tied):
        raise ValueError("ambiguous nearest coordinates")
    indices = order[nearest]
    if np.unique(indices).size != indices.size:
        raise ValueError("multiple selected coordinates match the same source sample")
    return indices


def native_axis_selection(reference, dataset, axis: str, requested):
    """Select native samples inside each contiguous run selected on the reference.

    Disconnected list selections remain disconnected. A single selected value
    is still an exact cut, while a selected run defines an inclusive interval.
    This is a display selection, with no interpolation or extrapolation.
    """
    attribute = {"azimuth": "azimuths", "elevation": "elevations", "frequency": "frequencies"}[axis]
    requested, tolerance = selection_for_dataset(reference, dataset, axis, requested)
    reference_values = convert_axis_values(
        getattr(reference, attribute), axis, axis_unit(reference, axis), axis_unit(dataset, axis)
    )
    reference_values = np.sort(np.asarray(reference_values, dtype=float))
    selected = unique_axis_selection(reference_values, requested, tolerance)
    if selected is None or not len(selected):
        return None
    native = np.asarray(getattr(dataset, attribute), dtype=float)
    if native.ndim != 1 or not np.all(np.isfinite(native)):
        raise ValueError("source coordinates must be finite and one-dimensional")
    order = np.argsort(native, kind="stable")
    ordered = native[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("duplicate source coordinates")
    if not ordered.size:
        return None
    selected = np.sort(selected)
    run_starts = np.r_[0, np.flatnonzero(np.diff(selected) > 1) + 1]
    run_ends = np.r_[run_starts[1:] - 1, selected.size - 1]
    spans = run_ends > run_starts
    # Difference marks cover all intervals in one pass, even for a large
    # selection with many disjoint runs; never scan the full axis per run.
    marks = np.zeros(ordered.size + 1, dtype=np.int64)
    lower = reference_values[selected[run_starts[spans]]] - tolerance
    upper = reference_values[selected[run_ends[spans]]] + tolerance
    np.add.at(marks, np.searchsorted(ordered, lower, side="left"), 1)
    np.add.at(marks, np.searchsorted(ordered, upper, side="right"), -1)
    keep = np.cumsum(marks[:-1]) > 0
    singletons = reference_values[selected[run_starts[~spans]]]
    if singletons.size:
        positions = np.searchsorted(ordered, singletons)
        lo = np.clip(positions - 1, 0, ordered.size - 1)
        hi = np.clip(positions, 0, ordered.size - 1)
        distance = np.minimum(np.abs(ordered[lo] - singletons), np.abs(ordered[hi] - singletons))
        present = singletons[distance <= tolerance]
        if present.size:
            matched = unique_axis_selection(ordered, present, tolerance)
            keep[matched] = True
    indices = order[keep]
    return indices if indices.size else None


def finite_axis_limits(low, high):
    """Return explicit nonsingular limits; preserve intentional inversion."""
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("plot limits must be finite")
    if low == high:
        pad = max(abs(low) * 0.05, 1.0) if low == 0 else abs(low) * 0.05
        center = low
        low, high = low - pad, high + pad
        if low == high:
            low, high = np.nextafter(center, -np.inf), np.nextafter(center, np.inf)
    return low, high


def set_spin_value(spin, value):
    """Set an actual axis bound without Qt silently clipping its magnitude."""
    value = float(value)
    if not np.isfinite(value):
        return
    blocked = spin.blockSignals(True)
    try:
        if all(callable(getattr(spin, method, None)) for method in ("setRange", "minimum", "maximum")):
            if value and hasattr(spin, "setDecimals"):
                # QDoubleSpinBox also rounds by decimal places: preserve small
                # linear powers and tightly spaced frequency bounds.
                decimal = np.format_float_positional(value, unique=True, trim="-")
                places = len(decimal.partition(".")[2])
                if places > spin.decimals():
                    spin.setDecimals(min(323, places))
            spin.setRange(min(spin.minimum(), value), max(spin.maximum(), value))
        spin.setValue(value)
    finally:
        spin.blockSignals(bool(blocked))


def axis_matching_tolerance(dataset, axis: str) -> float:
    """Return GRIM's physical matching tolerance in a dataset's native unit.

    Frequency matching historically uses 1e-6 GHz (1 kHz), while angle
    matching uses 1e-6 degrees. Anchoring the tolerance to those physical
    units makes matching symmetric when the reference grid happens to store
    Hz/radians instead of GHz/degrees.
    """

    target_unit = axis_unit(dataset, axis)
    base_unit = "GHz" if axis == "frequency" else "deg"
    tolerance = 1.0e-6 * abs(
        _unit_scale(axis, base_unit) / _unit_scale(axis, target_unit)
    )
    return max(tolerance, np.finfo(float).eps * 16.0)


def values_for_display(reference, dataset, axis: str, values) -> np.ndarray:
    """Convert a dataset's native axis values to the reference display unit."""

    return convert_axis_values(
        values,
        axis,
        axis_unit(dataset, axis),
        axis_unit(reference, axis),
    )


def reference_dataset(named_datasets, active_dataset=None):
    """Prefer the active dataset when it is among the selected datasets."""

    for _name, dataset in named_datasets:
        if dataset is active_dataset:
            return dataset
    return named_datasets[0][1]


def angular_axis_name(dataset, axis: str) -> str:
    return "Azimuth" if axis == "azimuth" else "Elevation"


def axis_label(reference, axis: str) -> str:
    if axis == "frequency":
        return f"Frequency ({axis_unit(reference, axis)})"
    return f"{angular_axis_name(reference, axis)} ({axis_unit(reference, axis)})"


def _declared_coherent_metadata(dataset, key: str) -> str:
    """Return one producer declaration without inventing a default."""

    inspector = getattr(dataset, "inspect_scalar_metadata", None)
    if callable(inspector):
        try:
            evidence = inspector(key)
        except (TypeError, ValueError):
            return f"<unusable {key} declaration>"
        if evidence.status in {"conflicting", "malformed"}:
            return f"<unusable {key} declaration>"
    declared_getter = getattr(dataset, "_declared_scalar_metadata", None)
    if callable(declared_getter):
        try:
            raw = declared_getter(key)
        except (TypeError, ValueError):
            # Conflicting or malformed provenance must not suppress a
            # read-only plot whose numeric arrays are otherwise usable.
            return f"<unusable {key} declaration>"
    else:
        raw = (dataset.units or {}).get(key, (dataset.extra or {}).get(key, ""))
    return str(raw or "").strip()


def _canonical_coherent_metadata(dataset, key: str, value: str) -> str:
    if key == "time_convention":
        canonicalizer = getattr(dataset, "_canonical_time_convention", None)
        if callable(canonicalizer):
            return str(canonicalizer(value))
    return " ".join(str(value).split()).casefold()


def coherent_metadata_plot_warnings(named_datasets) -> tuple[str, ...]:
    """Describe assumptions made by a read-only multi-dataset phase plot.

    Producer provenance is useful context, but it is not numeric plot data.
    A phase overlay therefore remains available when a declaration is absent
    or differs.  The returned notes make that assumption visible without
    claiming that unlike phase references have somehow been reconciled.
    """

    if len(named_datasets) < 2:
        return ()
    notes = []
    for key, label in (
        ("phase_reference", "phase reference"),
        ("time_convention", "time convention"),
        ("polarization_basis", "polarization basis"),
    ):
        declared = []
        missing = []
        for name, dataset in named_datasets:
            raw = _declared_coherent_metadata(dataset, key)
            if not raw:
                missing.append(str(name))
                continue
            declared.append(
                (
                    str(name),
                    raw,
                    _canonical_coherent_metadata(dataset, key, raw),
                )
            )
        unusable = [
            name for name, raw, _canonical in declared if raw.startswith("<unusable ")
        ]
        usable_declared = [
            item for item in declared if not item[1].startswith("<unusable ")
        ]
        distinct = {canonical for _name, _raw, canonical in usable_declared}
        if unusable:
            notes.append(
                f"{label} metadata is conflicting or malformed for "
                f"{', '.join(unusable)}; values are plotted as stored"
            )
        if len(distinct) > 1:
            details = ", ".join(
                f"{name}={raw!r}" for name, raw, _ in usable_declared
            )
            notes.append(
                f"{label} declarations differ ({details}); values are plotted "
                "as stored without phase-reference conversion"
            )
        if missing:
            notes.append(
                f"{label} is unspecified for {', '.join(missing)}; a common "
                "convention is assumed only for display"
            )
    return tuple(notes)


def validate_plot_datasets(
    named_datasets, *, phase: bool, linear: bool, allow_mixed_db: bool = False
) -> None:
    """Fail before rendering incompatible physical quantities.

    Coordinate units may differ because the modes convert them. All angle axes
    are treated as azimuth/elevation; overlaying data from different angular
    coordinate systems is the user's responsibility. Unlike linear quantities
    cannot share one linear ordinate. Visual dB overlays may opt into mixed
    native units, with explicit labels; calculations keep the strict default.
    """

    if not named_datasets:
        return
    for name, dataset in named_datasets:
        for axis in ("azimuth", "elevation", "frequency"):
            try:
                axis_unit(dataset, axis)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc

    # Linear ordinates and physical comparisons require one quantity. Native
    # dB overlays explicitly label unlike quantities. Phase is dimensionless;
    # its provenance checks below apply regardless of magnitude normalization.
    if not phase and (linear or not allow_mixed_db):
        quantities = {
            str(dataset.linear_quantity()).strip().lower()
            for _, dataset in named_datasets
        }
        if len(quantities) != 1:
            details = ", ".join(
                f"{name}={dataset.linear_quantity()}" for name, dataset in named_datasets
            )
            raise ValueError(
                f"mixed physical quantities cannot share a plot ({details})"
            )

    # Difference/comparison calculations still require a common log convention.
    if not phase and not linear and not allow_mixed_db:
        log_units = {dataset.default_log_unit().lower() for _, dataset in named_datasets}
        if len(log_units) != 1:
            details = ", ".join(
                f"{name}={dataset.default_log_unit()}" for name, dataset in named_datasets
            )
            raise ValueError(f"mixed logarithmic quantity units cannot share a plot ({details})")

    if phase:
        for note in coherent_metadata_plot_warnings(named_datasets):
            warnings.warn(
                "Phase overlay metadata note: " + note,
                UserWarning,
                stacklevel=2,
            )


def missing_coherent_metadata(named_datasets) -> tuple[str, ...]:
    """Return coherent metadata fields missing from at least one dataset."""

    missing = []
    for key in ("phase_reference", "time_convention", "polarization_basis"):
        for _name, dataset in named_datasets:
            if not _declared_coherent_metadata(dataset, key):
                missing.append(key)
                break
    return tuple(missing)


def circular_median_degrees(values, axis=0) -> np.ndarray:
    """Minimize total absolute geodesic distance, ignoring missing phases.

    For an even sample count, prefer the midpoint of the shortest flat
    minimizing interval. Remaining ties choose the smallest angle in
    [-180, 180). This is deterministic, including antipodal distributions.
    Prefix sums evaluate all candidate costs in O(n log n), without an n²
    distance matrix. Only one output column needs scratch storage at a time.
    """
    angles = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    result = np.full(angles.shape[1:], np.nan)
    for column in np.ndindex(result.shape):
        source = angles[(slice(None),) + column]
        samples = np.sort(np.mod(source[np.isfinite(source)], 360.0))
        n = samples.size
        if not n:
            continue
        prefix = np.r_[0.0, np.cumsum(samples)]

        def costs(centers):
            centers = np.mod(centers, 360.0)
            split = np.searchsorted(samples, centers)
            value = (centers * (2 * split - n) + prefix[-1] - 2 * prefix[split])
            below = np.searchsorted(samples, centers - 180.0)
            above = np.searchsorted(samples, centers + 180.0)
            value += below * (360.0 - 2 * centers) + 2 * prefix[below]
            value += (n - above) * (360.0 + 2 * centers) - 2 * (prefix[-1] - prefix[above])
            return value

        candidates = np.unique(samples)
        objective = costs(candidates)
        best = np.min(objective)
        tolerance = np.finfo(float).eps * max(n, 1) * 360.0 * 16
        winners = candidates[objective <= best + tolerance]
        if n % 2 == 0 and candidates.size > 1:
            gaps = np.diff(np.r_[candidates, candidates[0] + 360.0])
            midpoints = np.mod(candidates + gaps * 0.5, 360.0)
            flat = costs(midpoints) <= best + tolerance
            if np.any(flat):
                shortest = np.min(gaps[flat])
                winners = midpoints[flat & (gaps <= shortest + tolerance)]
        result[column] = np.min((winners + 180.0) % 360.0 - 180.0)
    return result


def wrap_phase_degrees(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    wrapped = (values + 180.0) % 360.0 - 180.0
    # Preserve +180 for positive inputs. This makes labels/statistics stable
    # without changing the shortest signed residual convention at -180.
    return np.where((wrapped == -180.0) & (values > 0.0), 180.0, wrapped)


@dataclass
class StreamingEnvelope:
    """Per-X min/max/count accumulator without a duplicate stacked array.

    Phase bands use the shortest containing circular arc at each X. Exact
    circular coverage needs the observations: spool them to a temporary file
    and sort bounded column blocks, rather than keep every curve in RAM.
    Equal largest gaps choose the smallest start angle in [-180,180).

    ``percentiles=(low, high)`` replaces min/max with those percentiles of
    the finite samples at each X (linear interpolation between samples),
    using the same spooled, column-blocked reduction.
    """

    phase_degrees: bool = False
    percentiles: tuple[float, float] | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    count: np.ndarray | None = None
    _phase_file: object = field(default=None, init=False, repr=False)
    _phase_rows: int = field(default=0, init=False, repr=False)
    _phase_dirty: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.percentiles is None:
            return
        if self.phase_degrees:
            raise ValueError("phase bands do not support percentiles")
        low, high = (float(value) for value in self.percentiles)
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        self.percentiles = (low, high)

    def update(self, values) -> None:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if self.lower is None:
            self.lower = np.full(values.shape, np.nan, dtype=float)
            self.upper = np.full(values.shape, np.nan, dtype=float)
            self.count = np.zeros(values.shape, dtype=np.int64)
        elif values.shape != self.lower.shape:
            raise ValueError("all envelope series must have the same shape")

        assert self.upper is not None and self.count is not None
        if self.phase_degrees or self.percentiles is not None:
            if self._phase_file is None:
                self._phase_file = tempfile.TemporaryFile()
            self._phase_file.seek(0, 2)
            self._phase_file.write(np.asarray(values, dtype=np.float64).tobytes())
            self._phase_rows += 1
            self._phase_dirty = True
            self.count[finite] += 1
            return
        first = finite & (self.count == 0)
        self.lower[first] = values[first]
        self.upper[first] = values[first]

        existing = finite & (self.count > 0)
        if np.any(existing):
            incoming = values[existing]
            self.lower[existing] = np.minimum(self.lower[existing], incoming)
            self.upper[existing] = np.maximum(self.upper[existing], incoming)
        self.count[finite] += 1

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.lower is None or self.upper is None or self.count is None:
            raise ValueError("cannot read an empty envelope")
        if self.percentiles is not None and self._phase_dirty and self.lower.size:
            self._phase_file.flush()
            samples = np.memmap(self._phase_file, dtype=np.float64, mode="r",
                                shape=(self._phase_rows, self.lower.size))
            block_columns = max(1, REDUCTION_BLOCK_CELLS // self._phase_rows)
            try:
                for start in range(0, self.lower.size, block_columns):
                    stop = min(self.lower.size, start + block_columns)
                    block = np.array(samples[:, start:stop])
                    with warnings.catch_warnings():
                        # All-missing columns stay NaN.
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        low, high = np.nanpercentile(block, self.percentiles, axis=0)
                    self.lower.flat[start:stop] = low
                    self.upper.flat[start:stop] = high
            finally:
                del samples
            self._phase_dirty = False
        if self.phase_degrees and self._phase_dirty and self.lower.size:
            self._phase_file.flush()
            samples = np.memmap(self._phase_file, dtype=np.float64, mode="r",
                                shape=(self._phase_rows, self.lower.size))
            block_columns = max(1, REDUCTION_BLOCK_CELLS // self._phase_rows)
            try:
                for start in range(0, self.lower.size, block_columns):
                    stop = min(self.lower.size, start + block_columns)
                    block = np.array(samples[:, start:stop])
                    finite = np.isfinite(block)
                    block = np.where(finite, (block + 180.) % 360. - 180., np.inf)
                    block.sort(axis=0)
                    count = self.count.flat[start:stop]
                    valid = count > 0
                    columns = np.arange(stop - start)
                    first = block[0]
                    last = block[np.maximum(count - 1, 0), columns]
                    with np.errstate(invalid="ignore"):
                        wrap_gap = first + 360. - last
                        gaps = np.diff(block, axis=0)
                    gaps = np.where(np.arange(self._phase_rows - 1)[:, None]
                                    < count[None, :] - 1, gaps, -np.inf)
                    if self._phase_rows > 1:
                        largest = np.argmax(gaps, axis=0)
                        interior_gap = gaps[largest, columns]
                        use_interior = interior_gap > wrap_gap
                        arc_start = np.where(use_interior, block[largest + 1, columns], first)
                        gap = np.where(use_interior, interior_gap, wrap_gap)
                    else:
                        arc_start, gap = first, wrap_gap
                    self.lower.flat[start:stop] = np.where(valid, arc_start, np.nan)
                    with np.errstate(invalid="ignore"):
                        self.upper.flat[start:stop] = np.where(valid, arc_start + 360. - gap, np.nan)
            finally:
                # Release the mapping before the backing temporary file closes
                # (also required by Windows).
                del samples
            self._phase_dirty = False
        return self.lower, self.upper, self.count

    def close(self):
        if self._phase_file is not None:
            self._phase_file.close()
            self._phase_file = None

    def __del__(self):
        self.close()


def _bucket_extrema_indices(values, edges):
    """First min/max/gap per bucket, using bounded vectorized scratch arrays."""
    n = len(values)
    minima, maxima, gaps = [], [], []
    first = 0
    while first < len(edges) - 1:
        last = max(first + 1, int(np.searchsorted(
            edges, edges[first] + REDUCTION_BLOCK_CELLS, side="right"
        )) - 1)
        last = min(last, len(edges) - 1)
        start, stop = int(edges[first]), int(edges[last])
        if stop - start > REDUCTION_BLOCK_CELLS:
            # An individual bucket may itself exceed the scratch budget.
            lo = hi = gap = n
            vmin, vmax = np.inf, -np.inf
            for offset in range(start, stop, REDUCTION_BLOCK_CELLS):
                segment = values[offset:min(stop, offset + REDUCTION_BLOCK_CELLS)]
                valid = np.isfinite(segment)
                if np.any(valid):
                    all_finite = np.all(valid)
                    low = int(np.argmin(segment if all_finite else np.where(valid, segment, np.inf)))
                    high = int(np.argmax(segment if all_finite else np.where(valid, segment, -np.inf)))
                    if segment[low] < vmin:
                        vmin, lo = segment[low], offset + low
                    if segment[high] > vmax:
                        vmax, hi = segment[high], offset + high
                if gap == n and np.any(~valid):
                    gap = offset + int(np.argmax(~valid))
            minima.append(np.array([lo])); maxima.append(np.array([hi])); gaps.append(np.array([gap]))
        else:
            segment = values[start:stop]
            valid = np.isfinite(segment)
            starts = edges[first:last] - start
            lengths = np.diff(edges[first:last + 1])
            all_finite = np.all(valid)
            # Keep exact integer comparisons too; an infinity sentinel would
            # round int64 values above 2**53 through a float conversion.
            low = np.minimum.reduceat(segment if all_finite else np.where(valid, segment, np.inf), starts)
            high = np.maximum.reduceat(segment if all_finite else np.where(valid, segment, -np.inf), starts)
            ids = np.arange(start, stop)
            minima.append(np.minimum.reduceat(np.where(
                valid & (segment == np.repeat(low, lengths)), ids, n), starts))
            maxima.append(np.minimum.reduceat(np.where(
                valid & (segment == np.repeat(high, lengths)), ids, n), starts))
            gaps.append(np.minimum.reduceat(np.where(~valid, ids, n), starts))
        first = last
    return tuple(np.concatenate(parts) for parts in (minima, maxima, gaps))


def decimate_line(x_values, y_values, max_points: int | None = None):
    """Bound a display line while preserving extrema, endpoints, and gaps.

    A NaN is a semantic break in a measured cut, not merely a value to ignore.
    Keeping one missing sample from every mixed bucket prevents Matplotlib from
    drawing a continuous line across unavailable data after display decimation.
    """

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    max_points = MAX_LINE_POINTS if max_points is None else int(max_points)
    if max_points < 4:
        raise ValueError("max_points must be at least 4")
    if x_values.shape != y_values.shape or x_values.ndim != 1:
        raise ValueError("line x/y arrays must be matching one-dimensional arrays")
    n = x_values.size
    if n <= max_points:
        return x_values, y_values, False

    # Min/max envelope decimation retains narrow peaks that simple striding can
    # erase. Allocate room for min, max, and one missing-data marker per
    # interior bucket, plus exact endpoints.
    bucket_count = max(1, (max_points - 2) // 3)
    edges = np.linspace(1, n - 1, bucket_count + 1, dtype=int)
    lo, hi, gaps = _bucket_extrema_indices(y_values, edges)
    selected = np.unique(np.r_[0, lo[lo < n], hi[hi < n], gaps[gaps < n], n - 1])
    if len(selected) > max_points:
        selected = _bounded_decimation_indices(selected, gaps[gaps < n], n, max_points)
    return x_values[selected], y_values[selected], True


def decimate_envelope(
    x_values,
    lower,
    upper,
    count=None,
    max_points: int | None = None,
):
    """Bound a filled envelope using extrema from both of its boundaries."""

    max_points = MAX_LINE_POINTS if max_points is None else int(max_points)
    if max_points < 4:
        raise ValueError("max_points must be at least 4")

    x_values = np.asarray(x_values)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    count_values = None if count is None else np.asarray(count)
    if x_values.shape != lower.shape or x_values.shape != upper.shape or x_values.ndim != 1:
        raise ValueError("envelope arrays must be matching one-dimensional arrays")
    if count_values is not None and count_values.shape != x_values.shape:
        raise ValueError("envelope count must match its coordinate array")
    n = x_values.size
    if n <= max_points:
        return x_values, lower, upper, count_values, False

    # Four extrema plus one missing-data marker per bucket, then exact ends.
    bucket_count = max(1, (max_points - 2) // 5)
    edges = np.linspace(1, n - 1, bucket_count + 1, dtype=int)
    selected = [0]
    gap_markers: list[int] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        valid_pair = np.isfinite(lower[start:stop]) & np.isfinite(upper[start:stop])
        if np.any(~valid_pair):
            gap_index = start + int(np.flatnonzero(~valid_pair)[0])
            selected.append(gap_index)
            gap_markers.append(gap_index)
        for values in (lower, upper):
            segment = values[start:stop]
            finite = np.isfinite(segment)
            if not np.any(finite):
                continue
            finite_indices = np.flatnonzero(finite)
            selected.append(start + finite_indices[int(np.argmin(segment[finite]))])
            selected.append(start + finite_indices[int(np.argmax(segment[finite]))])
    selected.append(n - 1)
    selected = _bounded_decimation_indices(selected, gap_markers, n, max_points)
    count_display = None if count_values is None else count_values[selected]
    return (
        x_values[selected],
        lower[selected],
        upper[selected],
        count_display,
        True,
    )


def _bounded_decimation_indices(selected, gap_markers, size: int, max_points: int) -> np.ndarray:
    """Keep decimator candidates within budget, prioritizing semantic gaps."""

    unique = sorted({int(index) for index in selected})
    if len(unique) <= int(max_points):
        return np.asarray(unique, dtype=int)

    endpoints = [0] if size == 1 else [0, size - 1]
    budget = max(0, int(max_points) - len(endpoints))
    gaps = [index for index in sorted(set(gap_markers)) if index not in endpoints]
    if len(gaps) > budget:
        positions = np.linspace(0, len(gaps) - 1, budget, dtype=int) if budget else []
        kept = endpoints + [gaps[int(position)] for position in positions]
        return np.asarray(sorted(set(kept)), dtype=int)

    kept = endpoints + gaps
    remaining = [index for index in unique if index not in set(kept)]
    slots = int(max_points) - len(kept)
    if slots > 0 and remaining:
        if len(remaining) <= slots:
            kept.extend(remaining)
        else:
            positions = np.linspace(0, len(remaining) - 1, slots, dtype=int)
            kept.extend(remaining[int(position)] for position in positions)
    return np.asarray(sorted(set(kept)), dtype=int)


def _bounded_image_shape(nx: int, ny: int, *, max_side: int, max_cells: int) -> tuple[int, int]:
    """Return a positive display shape that respects both image limits."""

    nx = int(nx)
    ny = int(ny)
    max_side = int(max_side)
    max_cells = int(max_cells)
    if nx < 0 or ny < 0:
        raise ValueError("image dimensions cannot be negative")
    if max_side < 1 or max_cells < 1:
        raise ValueError("image display limits must be positive")
    if nx == 0 or ny == 0:
        return nx, ny

    target_x = min(nx, max_side)
    target_y = min(ny, max_side)
    if target_x * target_y > max_cells:
        scale = np.sqrt(float(max_cells) / float(target_x * target_y))
        target_x = max(1, min(target_x, int(np.floor(target_x * scale))))
        target_y = max(1, min(target_y, int(np.floor(target_y * scale))))
        # Rounding can leave a slightly over-budget product for very small
        # limits. Trim the longer display dimension until the hard cap holds.
        while target_x * target_y > max_cells:
            if target_x >= target_y and target_x > 1:
                target_x -= 1
            elif target_y > 1:
                target_y -= 1
            else:  # pragma: no cover - max_cells >= 1 makes this unreachable
                break
    return target_x, target_y


def image_requires_decimation(
    x_count: int,
    y_count: int,
    *,
    max_side: int | None = None,
    max_cells: int | None = None,
) -> bool:
    """Return whether an image would exceed the interactive display budget."""

    nx = int(x_count)
    ny = int(y_count)
    target_x, target_y = _bounded_image_shape(
        nx,
        ny,
        max_side=MAX_IMAGE_SIDE if max_side is None else max_side,
        max_cells=MAX_IMAGE_CELLS if max_cells is None else max_cells,
    )
    return target_x < nx or target_y < ny


def bounded_image_cell_count(
    x_count: int,
    y_count: int,
    *,
    max_side: int | None = None,
    max_cells: int | None = None,
) -> int:
    """Return the number of cells retained by the interactive image bound."""

    target_x, target_y = _bounded_image_shape(
        int(x_count),
        int(y_count),
        max_side=MAX_IMAGE_SIDE if max_side is None else max_side,
        max_cells=MAX_IMAGE_CELLS if max_cells is None else max_cells,
    )
    return int(target_x) * int(target_y)


def validate_aggregate_image_cells(
    total_cells: int,
    *,
    panel_count: int,
    operation: str,
    max_cells: int = MAX_TOTAL_IMAGE_CELLS,
) -> None:
    """Reject a multi-panel image whose aggregate display storage is unsafe."""

    total_cells = int(total_cells)
    if total_cells <= int(max_cells):
        return
    raise ValueError(
        f"{operation} would retain {total_cells:,} display cells across "
        f"{int(panel_count):,} panels (limit {int(max_cells):,}); select fewer "
        "datasets/panels or narrow the plotted axes"
    )


def validate_synchronous_plot_workload(
    *,
    operation: str,
    peak_slice_cells: int,
    total_cells: int,
    max_slice_cells: int = MAX_SYNC_SLICE_CELLS,
    max_total_cells: int = MAX_SYNC_TOTAL_CELLS,
) -> None:
    """Bound NumPy work that still executes on the Qt GUI thread.

    These renderers use advanced indexing and exact medians, both of which can
    allocate several temporaries per source cell.  The preflight is deliberately
    expressed in cells rather than guessed bytes so it remains conservative for
    either float or complex datasets.
    """

    peak_slice_cells = int(peak_slice_cells)
    total_cells = int(total_cells)
    if peak_slice_cells <= int(max_slice_cells) and total_cells <= int(max_total_cells):
        return
    raise ValueError(
        f"{operation} selection requires a {peak_slice_cells:,}-cell working slice "
        f"and {total_cells:,} total source cells (limits {int(max_slice_cells):,} "
        f"and {int(max_total_cells):,}); crop/slice the dataset or select fewer "
        "axis values before plotting"
    )


def finite_data_limits(arrays) -> tuple[float, float] | None:
    """Return global finite limits without concatenating display arrays."""

    lower = float("inf")
    upper = float("-inf")
    for values in arrays:
        array = np.asarray(values)
        finite = array[np.isfinite(array)]
        if finite.size:
            lower = min(lower, float(np.min(finite)))
            upper = max(upper, float(np.max(finite)))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    return lower, upper


def decimate_image(x_values, y_values, image, *, max_side=MAX_IMAGE_SIDE, max_cells=MAX_IMAGE_CELLS):
    """Bound a display image using peak-preserving block aggregation.

    Uniform striding can completely erase a narrow scattering peak when its
    source cell happens to fall between retained indices.  Each output cell is
    therefore the finite maximum of its source block, which preserves the
    physically important high-return cell in linear and logarithmic intensity
    displays.  Output coordinates are the representative center samples of the
    corresponding source blocks.
    """

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    image = np.asarray(image)
    if image.shape != (x_values.size, y_values.size):
        raise ValueError("image shape must be (len(x_values), len(y_values))")
    nx, ny = image.shape
    target_x, target_y = _bounded_image_shape(
        nx, ny, max_side=max_side, max_cells=max_cells
    )
    if target_x >= nx and target_y >= ny:
        return x_values, y_values, image, False
    if np.iscomplexobj(image):
        raise ValueError("display image decimation requires real-valued intensity")

    # ``fmax.reduceat`` performs variable-width block reduction without a
    # Python loop over up to two million display cells.  It ignores a NaN when
    # the same block contains a finite value and retains NaN for an all-NaN
    # block, matching Matplotlib's missing-data behavior.
    x_starts = (np.arange(target_x, dtype=np.int64) * nx) // target_x
    y_starts = (np.arange(target_y, dtype=np.int64) * ny) // target_y
    image_display = np.fmax.reduceat(image, x_starts, axis=0)
    image_display = np.fmax.reduceat(image_display, y_starts, axis=1)

    x_stops = np.concatenate((x_starts[1:], np.asarray([nx], dtype=np.int64)))
    y_stops = np.concatenate((y_starts[1:], np.asarray([ny], dtype=np.int64)))
    x_centers = (x_starts + x_stops - 1) // 2
    y_centers = (y_starts + y_stops - 1) // 2
    return x_values[x_centers], y_values[y_centers], image_display, True


def bounded_ticks(start: float, stop: float, step: float, *, max_ticks=MAX_EXPLICIT_TICKS):
    """Return explicit ticks or ``None`` when the requested step is excessive."""

    if not np.isfinite(step) or step <= 0.0:
        return None
    if not np.isfinite(start) or not np.isfinite(stop):
        return None
    span = abs(stop - start)
    # Ticks must stay inside an explicit user range. Rounding the quotient to
    # the nearest integer made (0, 1, 0.6) emit 1.2, and Matplotlib then
    # silently expanded the axis beyond the requested maximum.
    tolerance = np.finfo(float).eps * max(span, step, 1.0) * 16.0
    count = int(np.floor((span + tolerance) / step)) + 1
    if count > int(max_ticks):
        return None
    direction = 1.0 if stop >= start else -1.0
    signed_step = direction * step
    ticks = start + signed_step * np.arange(count, dtype=float)
    if direction > 0.0:
        return ticks[ticks <= stop + tolerance]
    return ticks[ticks >= stop - tolerance]


def common_axis_indices(left, right, *, tolerance=1.0e-6):
    """Return one-to-one nearest indices for sorted/unsorted numeric axes."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("comparison axes must be one-dimensional")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("comparison axes must be finite")
    left_order = np.argsort(left, kind="stable")
    right_order = np.argsort(right, kind="stable")
    left_matches: list[int] = []
    right_matches: list[int] = []
    left_pos = 0
    right_pos = 0
    while left_pos < left_order.size and right_pos < right_order.size:
        left_index = int(left_order[left_pos])
        right_index = int(right_order[right_pos])
        delta = float(left[left_index] - right[right_index])
        if abs(delta) <= tolerance:
            left_matches.append(left_index)
            right_matches.append(right_index)
            left_pos += 1
            right_pos += 1
        elif delta < 0.0:
            left_pos += 1
        else:
            right_pos += 1
    return np.asarray(left_matches, dtype=int), np.asarray(right_matches, dtype=int)


MAX_SECTORS = 720


@dataclass(frozen=True)
class Sector:
    """One azimuth sector: ``start`` plus a positive ``width``.

    Explicit sectors are ``modular`` so they can wrap through ±180 (or 0/360);
    tiled sectors are plain intervals so the seam sample is counted once.
    """

    start: float
    width: float
    period: float
    closed: bool = True
    modular: bool = True
    stop_value: float | None = None

    def label(self) -> str:
        stop = self.start + self.width if self.stop_value is None else self.stop_value
        return f"{self.start:g} to {stop:g}"

    def contains(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        eps = 1.0e-9 * self.period
        if self.modular:
            if self.width >= self.period - eps:
                return np.isfinite(values)
            offset = np.mod(values - self.start + eps, self.period) - eps
        else:
            offset = values - self.start
            offset = np.where(offset >= -eps, offset, np.inf)
        if self.closed:
            return offset <= self.width + eps
        return offset < self.width - eps

    def display_pieces(self, low: float, high: float) -> list[tuple[float, float]]:
        """Plot-x extents of this sector clipped to the displayed ``[low, high]``."""
        start = self.start
        if self.modular:
            start = low + float(np.mod(self.start - low, self.period))
        pieces = [(start, start + self.width)]
        if start + self.width > low + self.period:
            pieces = [(start, low + self.period), (low, start + self.width - self.period)]
        clipped = []
        for first, last in pieces:
            first, last = max(first, low), min(last, high)
            if last > first:
                clipped.append((first, last))
        return clipped


def parse_sectors(text: str, selected, *, period: float = 360.0) -> list[Sector]:
    """Parse the Sector Stats range text.

    ``"30"`` tiles the selected azimuth span with 30-wide sectors starting at
    its first value; ``"-180:30:180"`` tiles from -180 to 180 in steps of 30;
    ``"-45:45, 45:135, 170:-170"`` lists explicit start:stop sectors, which
    wrap through ±180 (or 0/360) when stop is below start. Explicit sectors
    include both edges; tiled sectors are half-open except the last one, so
    each sample is counted once.
    """

    text = str(text or "").strip()
    selected = np.asarray(selected, dtype=float)
    selected = selected[np.isfinite(selected)]
    if not text:
        raise ValueError("enter a sector width such as 30, or ranges such as -45:45, 45:135")
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]

    def number(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a number") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{value!r} is not finite")
        return parsed

    def tiles(start: float, step: float, stop: float) -> list[Sector]:
        if step <= 0.0:
            raise ValueError("sector width must be positive")
        if stop <= start:
            raise ValueError("tiled sectors need a stop above the start")
        count = int(np.ceil((stop - start) / step - 1.0e-9))
        if count > MAX_SECTORS:
            raise ValueError(f"{count} sectors requested (limit {MAX_SECTORS}); use a wider step")
        sectors = []
        for index in range(count):
            first = start + index * step
            width = min(step, stop - first)
            sectors.append(Sector(
                first, width, period, closed=index == count - 1, modular=False,
            ))
        return sectors

    if len(parts) == 1 and ":" not in parts[0]:
        if selected.size == 0:
            raise ValueError("select azimuths to tile with sectors")
        low, high = float(selected.min()), float(selected.max())
        if high <= low:
            return [Sector(low, number(parts[0]), period, modular=False)]
        return tiles(low, number(parts[0]), high)
    if len(parts) == 1 and parts[0].count(":") == 2:
        start, step, stop = (number(value) for value in parts[0].split(":"))
        return tiles(start, step, stop)

    sectors = []
    for part in parts:
        bounds = part.split(":")
        if len(bounds) != 2:
            raise ValueError(f"{part!r} is not a start:stop sector")
        start, stop = (number(value) for value in bounds)
        if start == stop:
            raise ValueError(f"sector {part!r} is empty")
        width = float(np.mod(stop - start, period))
        # Unit conversion can leave a full revolution just above or below an
        # exact multiple of the period. Keep that a full sector, not a sliver.
        eps = 1.0e-9 * period
        full_turn = abs(stop - start) >= period - eps and (
            width <= eps or period - width <= eps
        )
        sectors.append(Sector(
            start, period if full_turn else width, period, stop_value=stop,
        ))
    if len(sectors) > MAX_SECTORS:
        raise ValueError(f"{len(sectors)} sectors requested (limit {MAX_SECTORS})")
    return sectors
