"""Small shared adapters from engineering controls to pure analysis results."""
from GRIM_Backend.plotting.analysis_results import (
    AnalysisOptions, phase_analysis, polarization_analysis,
)


def render_analysis_result(result, figure, display_line):
    figure.clear()
    axes = ([figure.add_subplot(111)] if len(result.axes) == 1 else
            list(figure.subplots(len(result.axes), 1, sharex=result.share_x)))
    for trace in result.traces:
        display_line(axes[trace.panel], trace.x, trace.y, label=trace.label,
                     runs=trace.runs, split_gaps=trace.split_gaps, **trace.style)
    for axis, specification in zip(axes, result.axes):
        options = dict(specification)
        legend, grid = options.pop("legend", False), options.pop("grid", False)
        axis.set(**options)
        if legend and axis.lines:
            axis.legend()
        if grid:
            axis.grid(True, alpha=.25)
    if result.equal_aspect:
        axes[0].set_aspect("equal", adjustable="datalim")
    return axes


class EngineeringAnalysisPresenter:
    """Both GUI presentations capture the same explicit numerical inputs."""
    def _analysis_options(self):
        return AnalysisOptions(self.floor.value(), self._maximum_gap(),
                               self._region(), self.phase_sign.currentData())

    def _publish_analysis(self, key, result):
        figure, canvas, table, note = self._panels[key]
        render_analysis_result(result, figure, self._display_line)
        table.populate(result.rows)
        note.setText(" ".join(result.notes))
        self._samples[key] = result.samples
        canvas.draw_idle()

    def _analyze_phase(self):
        view = self.phase_view.currentData()
        result = phase_analysis(self._cut(), view, self._analysis_options(),
                                reference=self._cut(reference=True) if view == "residual" else None)
        self._publish_analysis("phase", result)

    def _analyze_polarization(self):
        available = [self.fixed["polarization"].itemText(i) for i in range(self.fixed["polarization"].count())]
        if not available:
            raise ValueError("Select a dataset with polarization channels.")
        view = self.pol_view.currentData()
        pairs = [("HH", "VV"), ("HH", "HV"), ("VV", "VH")] if view == "ratios" else [
            ("HV", "VH") if view == "reciprocity" else (self.pol_a.currentText(), self.pol_b.currentText())]
        channels = {channel for pair in pairs for channel in pair}
        cuts = {channel: self._cut(polarization=channel) for channel in available if channel in channels}
        fallback = next(iter(cuts.values()), None)
        if fallback is None:
            fallback = self._cut()
        result = polarization_analysis(cuts, pairs, view, self._analysis_options(), fallback=fallback)
        self._publish_analysis("polarization", result)
