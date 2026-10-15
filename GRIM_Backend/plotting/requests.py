"""Qt-free, captured inputs and prepared data shared by plot front ends.

Requests refer to source datasets but own their selector arrays/options. A
caller must publish a result only while its source/view revision is current.
No function here reads a widget or creates a Matplotlib artist.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class PlotCancelled(Exception):
    """A superseded preparation stopped at a safe numerical boundary."""


def check_cancel(cancel: Callable[[], bool] | None):
    if cancel is not None and cancel():
        raise PlotCancelled()


@dataclass(frozen=True)
class PlotPlan:
    name: str
    dataset: Any
    selection: tuple

    def __post_init__(self):
        values = tuple(np.array(value, dtype=np.intp, copy=True) for value in self.selection)
        for value in values:
            value.flags.writeable = False
        object.__setattr__(self, "selection", values)


@dataclass(frozen=True)
class PlotRequest:
    mode: str
    reference: Any
    plans: tuple[PlotPlan, ...]
    phase: bool = False
    scale: str = "dbsm"
    options: dict = field(default_factory=dict)
    selectors: dict = field(default_factory=dict)
    max_series: int | None = None
    preserve_zeros: bool = False

    def __post_init__(self):
        object.__setattr__(self, "plans", tuple(self.plans))
        object.__setattr__(self, "options", deepcopy(self.options))
        object.__setattr__(self, "selectors", deepcopy(self.selectors))


@dataclass
class PlotResult:
    request: PlotRequest
    items: list
    omitted: int = 0
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    envelopes: list = field(default_factory=list)
    series_count: int = 0

    def close(self):
        """Release unpublished, temporary native image snapshots, if present."""
        for _plan, envelope in self.envelopes:
            envelope.close()
        for item in self.items:
            if isinstance(item, dict):
                snapshot = item.get("snapshot")
                if isinstance(snapshot, tuple) and len(snapshot) == 3:
                    snapshot[2].close()
                    item["snapshot"] = None


def prepare_sweep(request: PlotRequest, cancel=None) -> PlotResult:
    from . import sweep_analysis
    from .modes import common
    axis = "frequency" if request.mode == "frequency" else "elevation"
    result = PlotResult(request, [])
    if request.options.get("pbp"):
        envelopes = {}
        try:
            for plan in request.plans:
                check_cancel(cancel)
                az_count = 1 if request.options.get("mode") == "individual" else len(plan.selection[0])
                check_working_memory(max(common.REDUCTION_BLOCK_CELLS, az_count) * 64, "PBP preparation")
                identity = (id(plan.dataset), str(plan.dataset.polarizations[plan.selection[3][0]]))
                if identity not in envelopes:
                    envelope = common.StreamingEnvelope(
                        phase_degrees=request.phase,
                        include_negative_infinity=not request.phase and request.scale != "linear",
                        percentiles=None if request.phase else request.options.get("pbp_percentiles"))
                    envelopes[identity] = envelope
                    result.envelopes.append((plan, envelope))
                for _x, values, _label, _key in sweep_analysis.iter_series(
                        plan.name, plan.dataset, request.reference, plan.selection,
                        sweep_axis=axis, config=request.options, phase=request.phase,
                        scale=request.scale, preserve_zeros=True, cancel=cancel):
                    check_cancel(cancel)
                    if not np.any(np.isfinite(values) | ((not request.phase) & np.isneginf(values))):
                        continue
                    envelopes[identity].update(values)
                    result.series_count += 1
            for _plan, envelope in result.envelopes:
                if envelope.lower is not None:
                    envelope.result(cancel=cancel)
                envelope.close()
            return result
        except BaseException:
            result.close()
            raise
    retained_bytes = 0
    for plan in request.plans:
        check_cancel(cancel)
        az_count = 1 if request.options.get("mode") == "individual" else len(plan.selection[0])
        check_working_memory(retained_bytes + max(common.REDUCTION_BLOCK_CELLS, az_count) * 64, "Sweep preparation")
        candidates = len(plan.selection[1 if axis == "frequency" else 2])
        if request.options.get("mode", "median") == "individual":
            candidates *= len(plan.selection[0])
        if request.max_series is not None and len(result.items) >= request.max_series:
            result.omitted += candidates
            continue
        for index, series in enumerate(sweep_analysis.iter_series(
                plan.name, plan.dataset, request.reference, plan.selection,
                sweep_axis=axis, config=request.options, phase=request.phase,
                scale=request.scale, preserve_zeros=request.preserve_zeros,
                cancel=cancel)):
            check_cancel(cancel)
            if not np.any(np.isfinite(series[1]) | (request.preserve_zeros & np.isneginf(series[1]))):
                continue
            retained_bytes += int(series[0].nbytes + series[1].nbytes)
            check_working_memory(retained_bytes + common.REDUCTION_BLOCK_CELLS * 64, "Sweep preparation")
            result.items.append((plan, series))
            if request.max_series is not None and len(result.items) >= request.max_series:
                result.omitted += candidates - index - 1
                break
    return result


def prepare_cdf(request: PlotRequest, cancel=None) -> PlotResult:
    from . import distribution_analysis as distribution
    result = PlotResult(request, [])
    retained_bytes = 0
    for plan in request.plans:
        check_cancel(cancel)
        for selection, label in distribution.groups(
                plan.dataset, request.reference, plan.selection, request.options,
                azimuths=request.selectors["azimuth"], frequencies=request.selectors["frequency"]):
            check_cancel(cancel)
            if request.max_series is not None and len(result.items) >= request.max_series:
                result.omitted += 1
                continue
            check_working_memory(retained_bytes + int(np.prod([len(v) for v in selection])) * 96,
                                 "CDF population")
            curve = distribution.prepare_curve(
                plan.name, plan.dataset, selection, label, request.options, scale=request.scale)
            check_cancel(cancel)
            if curve is None:
                result.skipped.append(f"{plan.name} | {label}")
            else:
                result.items.append(curve)
                retained_bytes += sum(int(value.nbytes) for value in curve.values() if isinstance(value, np.ndarray))
    return result


def prepare_image_plane(request: PlotRequest, plan: PlotPlan, remaining_index=None, cancel=None, *, out=None):
    """Shared native waterfall values, with no GUI or display reduction."""
    from .image_analysis import heatmap_display_plane
    from .modes import common
    axes = ("azimuth", "elevation", "frequency")
    x_axis, y_axis = request.options["axes"]
    xi, yi = axes.index(x_axis), axes.index(y_axis)
    other = next(i for i in range(3) if i not in (xi, yi))
    shape = (len(plan.selection[xi]), len(plan.selection[yi]))
    reduction_width = len(plan.selection[other]) if remaining_index is None else 1
    pixels = int(np.prod(shape))
    retained = pixels * 8 if out is None else 0
    scratch = min(pixels * max(1, reduction_width), max(common.REDUCTION_BLOCK_CELLS, reduction_width)) * 64
    check_working_memory(retained + scratch, "Waterfall image")
    result = np.empty(shape, dtype=float) if out is None else out
    if result.shape != shape:
        raise ValueError("Waterfall destination shape does not match selected axes")
    # Tiling physical output axes keeps means/circular means over every
    # contributing sample intact. Cancellation is checked between tiles.
    max_pixels = max(1, common.REDUCTION_BLOCK_CELLS // max(1, reduction_width))
    y_width = min(shape[1], max_pixels)
    x_width = max(1, max_pixels // max(1, y_width))
    for x_start in range(0, shape[0], x_width):
        for y_start in range(0, shape[1], y_width):
            check_cancel(cancel)
            x_stop, y_stop = min(shape[0], x_start+x_width), min(shape[1], y_start+y_width)
            selection = list(plan.selection)
            selection[xi] = selection[xi][x_start:x_stop]
            selection[yi] = selection[yi][y_start:y_stop]
            tile = heatmap_display_plane(
                plan.dataset, selection, x_axis, y_axis, remaining_index=remaining_index,
                phase=request.phase, linear=request.scale == "linear")
            # Match Matplotlib's missing-cell behavior before block maxima;
            # avoid a second, full native image solely for finite masking.
            tile[~np.isfinite(tile)] = np.nan
            result[x_start:x_stop, y_start:y_stop] = tile
    return result


def prepare_waterfall_image(request, plan, x, y, *, remaining_index=None,
                            x_width=1., y_width=1., cancel=None):
    """Prepare bounded display data and one exact native snapshot.

    Large native planes are written directly to temporary mapped storage.
    This storage is transferred to the published artist, or closed by the
    caller's PlotResult when a request is canceled or superseded.
    """
    from .modes import common
    from .image_grid import image_grid
    native = None
    try:
        reduced = common.image_requires_decimation(len(x), len(y))
        if request.phase and reduced:
            raise ValueError("Phase Waterfall needs smaller selected axes; block maxima do not represent phase")
        if reduced:
            check_cancel(cancel)
            native = common.NativeSamples((len(x), len(y)))
            display = prepare_image_plane(request, plan, remaining_index, cancel, out=native.values)
            native.values.flush()
            bx, by, bd, reduced = common.decimate_image(x, y, display,
                cancel=lambda: check_cancel(cancel))
        else:
            display = prepare_image_plane(request, plan, remaining_index, cancel)
            bx, by, bd, reduced = common.decimate_image(x, y, display)
        check_cancel(cancel)
        if not np.any(np.isfinite(bd)):
            if native is not None:
                native.close()
            return None
        snapshot = None
        if reduced:
            if native is None:
                # Keep the existing optional-snapshot fallback for small
                # custom display budgets. Large planes never fall back to a
                # full RAM allocation if temporary storage is unavailable.
                try:
                    native = common.NativeSamples.image(display)
                except (OSError, MemoryError, ValueError):
                    snapshot = False
            if native is not None:
                snapshot = (common.coordinate_edges(x), common.coordinate_edges(y), native)
        supported = image_grid(x, y, display.T, x_width=x_width, y_width=y_width,
            target_shape=bd.T.shape, return_samples=snapshot is False,
            cancel=lambda: check_cancel(cancel))
        check_cancel(cancel)
        return dict(x=bx, y=by, display=bd, snapshot=snapshot,
            native_x=x, native_y=y, native_display=display if native is None else None,
            drawn=supported[2], x_edges=supported[0], y_edges=supported[1],
            x_width=x_width, y_width=y_width,
            display_samples=supported[3] if snapshot is False else None)
    except BaseException:
        if native is not None:
            native.close()
        raise


def check_working_memory(estimated_bytes, operation):
    """Budget actual temporary bytes rather than a GUI-thread cell ceiling."""
    from GRIM_Backend.datasets.memory import _default_dense_import_limit_bytes
    limit = _default_dense_import_limit_bytes()
    if int(estimated_bytes) > limit:
        raise MemoryError(f"{operation} needs approximately {int(estimated_bytes) / 1024**2:.0f} MiB "
                          f"of working storage; available budget is {limit / 1024**2:.0f} MiB. "
                          "Select a smaller region or fewer groups.")
