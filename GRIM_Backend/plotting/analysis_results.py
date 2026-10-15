"""Engineering analysis results independent of widgets and plotting backends."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from GRIM_Backend.datasets.metadata import canonical_time_convention
from .engineering_tools import (phase_trace, phase_residual, phase_slope_fit,
                                channel_comparison, finite_runs)
from .modes.common import convert_axis_values


def number(value):
    return "not defined" if value is None else f"{value:.9g}" if isinstance(value, (float, np.floating)) else str(value)


def run_samples(label, x, y, unit, runs):
    if len(runs) <= 1:
        return [(label, x, y, unit)]
    boundaries = [0, *(int(run[0]) for run in runs[1:]), len(x)]
    return [(f"{label} · Run {index}", x[start:stop], y[start:stop], unit)
            for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1)]


@dataclass(frozen=True)
class AnalysisOptions:
    floor_db: float = -60.
    maximum_gap: float | None = None
    region: tuple | None = None
    time_sign: str = "auto"


@dataclass
class AnalysisTrace:
    panel: int
    x: np.ndarray
    y: np.ndarray
    label: str | None = None
    runs: object = None
    split_gaps: bool = True
    style: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    axes: list[dict] = field(default_factory=list)
    traces: list[AnalysisTrace] = field(default_factory=list)
    rows: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    samples: list = field(default_factory=list)
    equal_aspect: bool = False
    share_x: bool = True

    def trace(self, panel, x, y, label=None, *, runs=None, split_gaps=True, **style):
        self.traces.append(AnalysisTrace(panel, np.asarray(x), np.asarray(y), label, runs, split_gaps, style))


def phase_analysis(cut, view, options: AnalysisOptions, *, reference=None):
    """Build native phase curves, fitted values, and export rows from a cut."""
    result = AnalysisResult()
    rows, notes, samples = result.rows, result.notes, result.samples
    rows.extend((("Source A", cut.label, f"{cut.axis} ({cut.x_unit})"),
                 ("Region", str(options.region) if options.region else "Full cut", cut.x_unit)))
    phases = phase_trace(cut.x, cut.field, floor_db=options.floor_db,
                         maximum_gap=options.maximum_gap, region=options.region,
                         native_indices=cut.native_indices)
    rows.append(("Usable phase samples", f"{phases['finite_count']} / {phases['total_count']}",
                 f"mask: {options.floor_db:g} dB relative to cut peak"))
    notes.extend(cut.notes)
    if view == "iq":
        from .iq_tools import field_unit
        unit = field_unit(cut)
        result.axes = [dict(xlabel=f"I: real field ({unit})", ylabel=f"Q: imaginary field ({unit})", title=cut.label, grid=True)]
        result.equal_aspect = True
        for run in phases["runs"]:
            result.trace(0, cut.field[run].real, cut.field[run].imag, split_gaps=False)
            result.trace(0, [cut.field[run[0]].real], [cut.field[run[0]].imag], split_gaps=False,
                         marker="o", linestyle="None", color="#2a9d8f")
        notes.append("Green points begin each valid run. I/Q uses the normalized complex field, not a dB magnitude.")
        selected = cut.field[phases["valid"]]
        samples.append(("I/Q", selected.real, selected.imag, unit))
    elif view == "residual":
        if reference is None:
            raise ValueError("Phase residual requires a reference cut.")
        rows.append(("Reference B", reference.label, f"{reference.axis} ({reference.x_unit})"))
        report = phase_residual(cut, reference, floor_db=options.floor_db,
                                maximum_gap=options.maximum_gap, region=options.region)
        values = np.rad2deg(report["phase_difference_rad"])
        result.axes = [dict(xlabel=f"{cut.axis} ({cut.x_unit})", ylabel="Wrapped phase difference (deg)",
                            title=f"{cut.label}\nminus {reference.label}", grid=True)]
        result.trace(0, report["x"], values, "A − B")
        rows.extend((("Common / usable samples", f"{report['common_count']} / {report['valid_count']}", "physical coordinates; no interpolation"),
                     ("Circular mean phase difference", number(report["circular_mean_deg"]), "deg"),
                     ("RMS wrapped phase difference", number(report["rms_wrapped_deg"]), "deg")))
        notes.extend("Reference B: " + item for item in reference.notes)
        notes.append("Phase conventions are not converted. A shared physical reference must be established before interpreting the difference.")
        samples.append(("Phase A-B", report["x"], values, "deg"))
    elif view == "slope":
        if cut.axis != "frequency":
            raise ValueError("Select Frequency as the cut axis for a phase-slope fit.")
        sign = options.time_sign
        if sign == "auto":
            declaration = canonical_time_convention(cut.time_convention)
            sign = "-jwt" if declaration == "-jwt" else "+jwt"
            recognized = declaration in {"+jwt", "-jwt"}
            notes.append(f"Delay sign: exp({sign}), " + ("dataset declaration." if recognized else "explicit default assumption; no recognized declaration supplied."))
        else:
            notes.append(f"Delay sign: user-selected exp({sign}).")
        rows.append(("Time convention for delay", f"exp({sign})", notes[-1]))
        scale = float(convert_axis_values([1.], "frequency", cut.x_unit, "Hz")[0])
        report = phase_slope_fit(cut.x * scale, cut.field, floor_db=options.floor_db,
            region_hz=tuple(value * scale for value in options.region) if options.region else None,
            maximum_gap_hz=options.maximum_gap * scale if options.maximum_gap else None,
            time_sign=sign, native_indices=cut.native_indices)
        x = report["x_hz"] / scale
        result.axes = [dict(ylabel="Phase (deg)", title=cut.label, legend=True),
                       dict(xlabel=f"Frequency ({cut.x_unit})", ylabel="Residual (deg)")]
        result.trace(0, x, np.rad2deg(report["unwrapped"]), "Local unwrapped phase", runs=report["runs"])
        result.trace(0, x, np.rad2deg(report["fitted"]), "Fit", runs=report["runs"], linestyle="--")
        result.trace(1, x, np.rad2deg(report["residual"]), "Fit residual", runs=report["runs"])
        for fit in report["fits"]:
            prefix = f"Run {fit['run']}"
            rows.append((prefix + " support", str(fit["samples"]), f"{fit['start_hz']:.9g}…{fit['stop_hz']:.9g} Hz"))
            if "note" in fit:
                rows.append((prefix, "no fit", fit["note"]))
                continue
            for metric, unit in (("apparent_delay_s", "s"), ("slope_rad_per_hz", "rad/Hz"), ("r_squared", "R²"),
                                 ("residual_rms_deg", "deg"), ("residual_max_deg", "deg"),
                                 ("alias_period_s", "s; uniform-grid phase alias period")):
                rows.append((prefix + " " + metric.replace("_", " "), number(fit[metric]), unit))
        notes.append(report["note"])
        for label, values in (("Fit residual", report["residual"]), ("Unwrapped phase", report["unwrapped"]), ("Fitted phase", report["fitted"])):
            samples.extend(run_samples(label, x, np.rad2deg(values), "deg", report["runs"]))
    else:
        result.axes = [dict(ylabel=cut.level_unit, title=cut.label),
                       dict(xlabel=f"{cut.axis} ({cut.x_unit})", ylabel="Phase (deg)", legend=True)]
        runs = list(finite_runs(cut.x, np.isfinite(cut.db), maximum_gap=options.maximum_gap,
                                native_indices=cut.native_indices))
        result.trace(0, cut.x, cut.db, cut.level_unit, runs=runs)
        result.trace(1, cut.x, np.rad2deg(phases["wrapped"]), "Wrapped", runs=phases["runs"])
        result.trace(1, cut.x, np.rad2deg(phases["unwrapped"]), "Locally unwrapped", runs=phases["runs"], linestyle="--")
        notes.append("Each finite, magnitude-qualified run is unwrapped independently; no phase continuity is assumed across gaps.")
        samples.extend(run_samples("Magnitude", cut.x, cut.db, cut.level_unit, runs))
        for label, values in (("Wrapped phase", phases["wrapped"]), ("Unwrapped phase", phases["unwrapped"])):
            samples.extend(run_samples(label, cut.x, np.rad2deg(values), "deg", phases["runs"]))
    rows.append(("Conventions and interpretation", " ".join(notes), "advisory"))
    return result


def polarization_analysis(cuts, pairs, view, options: AnalysisOptions, *, fallback):
    """Compare explicitly named native channel cuts without UI dependencies."""
    result = AnalysisResult()
    rows, samples = result.rows, result.samples
    for channel, cut in cuts.items():
        result.trace(0, cut.x, cut.db, channel)
    for left_name, right_name in pairs:
        if left_name not in cuts or right_name not in cuts:
            rows.append((left_name + "/" + right_name, "unavailable", "both labeled channels are required"))
            continue
        left, right = cuts[left_name], cuts[right_name]
        report = channel_comparison(left, right, floor_db=options.floor_db, region=options.region)
        label = left_name + (" − " if view == "phase" else " / ") + right_name
        values = np.rad2deg(report["phase_difference_rad"]) if view == "phase" else report["ratio_db"]
        result.trace(1, report["x"], values, label)
        finite = values[np.isfinite(values)]
        rows.append((label + " finite samples", str(len(finite)), "common physical coordinates"))
        if view != "phase":
            rows.append((label + " median", number(float(np.median(finite)) if len(finite) else None), "dB power ratio"))
        rows.extend(((label + " phase difference", number(report["circular_mean_deg"]), "deg; circular mean"),
                     (label + " normalized complex mismatch", number(report["normalized_complex_mismatch"]), "relative L2; requires shared phase conventions")))
        samples.append((label, report["x"], values, "deg" if view == "phase" else "dB"))
    example = next(iter(cuts.values()), fallback)
    rows[:0] = [("Source", example.label, f"{example.axis} ({example.x_unit})"),
                ("Region", str(options.region) if options.region else "Full cut", example.x_unit),
                ("Magnitude mask", number(options.floor_db), "dB below each selected channel peak")]
    rows.append(("Declared conventions", " ".join(example.notes), "advisory; no convention conversion"))
    result.axes = [dict(ylabel=example.level_unit, title=example.label.split(" | ")[0] + " — polarization cuts", legend=True, grid=True),
                   dict(xlabel=f"{example.axis} ({example.x_unit})", ylabel="Phase difference (deg)" if view == "phase" else "Power ratio (dB)", legend=True, grid=True)]
    result.notes.append("Ratios use linear power; channel phases use authoritative complex samples. Channel names are used as supplied. "
        "HV/VH equality is a diagnostic only under a reciprocal monostatic target and compatible polarization/phase conventions; "
        "it is not a pass/fail calibration verdict. " + " ".join(example.notes))
    return result
