"""Offscreen workflows exercise native data, controls, and dialog lifetime."""
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication, QWidget

from GRIM_Backend.plotting.markers import PlotMarkersMixin
from GRIM_Backend.ui.engineering_analysis import show_engineering_analysis
from test_engineering_tools import make_grid


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


class Owner(QWidget):
    _marker_arrays = staticmethod(PlotMarkersMixin._marker_arrays)

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.active_dataset = datasets[0][1] if datasets else None
        self._active_plot_tab = "plotting"
        self.plot_figure = Figure()
        self.lines = []
        self.markers = []

    def _selected_datasets(self): return self.datasets
    def _marker_lines(self): return self.lines
    def _add_plot_marker(self, line, index): self.markers.append((line, index))


@pytest.fixture
def owner(application):
    frequencies = np.linspace(9., 10., 21)
    channels = np.array([2 * np.exp(.3j), np.exp(.1j), .2 * np.exp(.5j), .2 * np.exp(.5j)])
    field = np.exp(-2j * np.pi * frequencies * .2)[None, None, :, None] * channels[None, None, None, :]
    widget = Owner([("Target", make_grid(field, polarizations=("HH", "VV", "HV", "VH")))])
    yield widget
    dialog = getattr(widget, "_engineering_analysis_dialog", None)
    if dialog is not None: dialog.close()
    widget.close(); application.processEvents()


def test_all_phase_and_polarization_views_produce_real_plots(owner, application):
    dialog = show_engineering_analysis(owner, tab="phase")
    for index in range(dialog.phase_view.count()):
        dialog.phase_view.setCurrentIndex(index); dialog._update_current()
        assert dialog._panels["phase"][0].axes
        assert dialog._panels["phase"][2].rows
        assert dialog._samples["phase"]
    dialog.tabs.setCurrentIndex(2)
    for index in range(dialog.pol_view.count()):
        dialog.pol_view.setCurrentIndex(index); dialog._update_current()
        assert len(dialog._panels["polarization"][0].axes) == 2
        assert dialog._panels["polarization"][2].rows
    dialog._panels["polarization"][2].copy()
    assert "Metric\tValue" in QApplication.clipboard().text()
    dialog._copy_samples("polarization")
    assert "Series\tX\tValue" in QApplication.clipboard().text()
    assert len(QApplication.clipboard().text().splitlines()) > 10


def test_hz_conversion_and_exp_jomega_sign_are_applied_in_ui(owner):
    owner.active_dataset.units["time_convention"] = "exp(+j omega t)"
    dialog = show_engineering_analysis(owner, tab="phase")
    dialog.phase_view.setCurrentIndex(dialog.phase_view.findData("slope")); dialog._update_current()
    rows = dialog._panels["phase"][2].rows
    delay = next(row for row in rows if "apparent delay s" in row[0])
    assert float(delay[1]) == pytest.approx(.2e-9)
    assert "dataset declaration" in dialog._panels["phase"][3].text()


def test_native_peak_marker_and_cursor_region_ignore_overview_reduction(owner):
    x = np.arange(101.)
    y = np.zeros(101); y[51] = 100.; y[71] = 30.
    ax = owner.plot_figure.add_subplot(111)
    line, = ax.plot(x[::10], y[::10], label="Native sharp peak")
    line._grim_native_samples = SimpleNamespace(values=np.vstack((x, y)))
    owner.lines = [line]
    dialog = show_engineering_analysis(owner)
    dialog.trace_combo.setCurrentIndex(1)
    dialog._mark_measurement("peak")
    assert owner.markers[-1] == (line, 51)
    assert dialog._measurement[2]["peak"]["y"] == 100.
    dialog._select_region(60, 80)
    assert dialog._measurement[2]["finite_count"] == 21
    assert dialog._measurement[2]["peak"]["x"] == 71.
    assert dialog.use_region.isChecked()
    assert "trace x" in dialog.region_start.suffix()


def test_next_peak_cycles_actual_local_peaks(owner):
    ax = owner.plot_figure.add_subplot(111)
    line, = ax.plot(np.arange(7.), [0, 3, 0, 4, 0, 5, 0])
    owner.lines = [line]
    dialog = show_engineering_analysis(owner)
    dialog.trace_combo.setCurrentIndex(1)
    for _ in range(4): dialog._mark_measurement("next")
    assert [index for _line, index in owner.markers] == [1, 3, 5, 1]


def test_dataset_phase_region_is_separate_from_plot_trace_coordinate_region(owner):
    ax = owner.plot_figure.add_subplot(111)
    line, = ax.plot(np.arange(101.), np.zeros(101))
    owner.lines = [line]
    dialog = show_engineering_analysis(owner)
    dialog.trace_combo.setCurrentIndex(1); dialog._select_region(30, 50)
    dialog.tabs.setCurrentIndex(1)
    assert dialog.region_start.value() == 9.
    assert dialog.region_stop.value() == 10.
    assert not dialog.use_region.isChecked()
    assert "GHz" in dialog.region_start.suffix()
    dialog.tabs.setCurrentIndex(0)
    assert dialog.region_start.value() == 30.
    assert dialog.region_stop.value() == 50.
    assert dialog.use_region.isChecked()


def test_reference_cut_matches_physical_fixed_coordinate_not_integer_index(owner):
    a = make_grid(azimuths=(0., 90.))
    b = make_grid(azimuths=np.deg2rad([-90., 0., 90.]), units={"azimuth": "rad"})
    # Give the wrong index a different phase so equal-index alignment cannot pass.
    b.rcs_phase[1, :, :, :] = 1.
    owner.datasets = [("A", a), ("B", b)]; owner.active_dataset = a
    dialog = show_engineering_analysis(owner, "phase")
    dialog.fixed["azimuth"].setCurrentIndex(1)
    reference = dialog._cut(reference=True)
    np.testing.assert_allclose(reference.field, 1)
    assert "1.57079633" in reference.label
    b.azimuths[0] = np.nan
    np.testing.assert_allclose(dialog._cut(reference=True).field, 1)
    b.azimuths = np.array([-.5, 0., .5])
    with pytest.raises(ValueError, match="no matching azimuth"):
        dialog._cut(reference=True)


def test_refresh_with_no_sources_clears_all_old_results(owner):
    dialog = show_engineering_analysis(owner, "phase")
    assert dialog._panels["phase"][2].rows
    owner.datasets = []; owner.active_dataset = None
    dialog.refresh_sources()
    for figure, _canvas, table, _note in dialog._panels.values():
        assert table.rows == []
        assert figure.axes == []
    assert not any(dialog._samples.values())


def test_released_native_trace_cannot_silently_fall_back_to_overview(owner):
    ax = owner.plot_figure.add_subplot(111)
    line, = ax.plot([0, 1], [0, 1])
    line._grim_native_samples = SimpleNamespace(values=np.array([[0., 1.], [0., 1.]]))
    owner.lines = [line]
    dialog = show_engineering_analysis(owner)
    dialog.trace_combo.setCurrentIndex(1)
    line._grim_native_samples.values = None
    dialog._update_current()
    assert "released" in dialog._panels["measure"][3].text()
    assert not dialog._panels["measure"][2].rows


def test_reject_cleanup_allows_singleton_to_reopen(owner, application):
    first = show_engineering_analysis(owner)
    first.reject(); application.processEvents()
    assert owner._engineering_analysis_dialog is None
    second = show_engineering_analysis(owner, "polarization")
    assert second is not first
    assert second.tabs.currentIndex() == 2
    second.close(); application.processEvents()
    assert owner._engineering_analysis_dialog is None


def test_optional_gap_limit_breaks_rendered_phase_lines(owner):
    owner.datasets = [("Gap", make_grid(frequencies=[9, 9.1, 10, 10.1]))]
    owner.active_dataset = owner.datasets[0][1]
    dialog = show_engineering_analysis(owner, "phase")
    dialog.gap_enabled.setChecked(True); dialog.gap.setValue(.2); dialog._update_current()
    for ax in dialog._panels["phase"][0].axes:
        for line in ax.lines:
            x = np.asarray(line.get_xdata())
            assert np.isnan(x).any()
            assert not np.any(np.diff(x)[np.isfinite(np.diff(x))] > .2)


def test_scroll_container_allows_laptop_height_without_squashing_chart(owner, application):
    dialog = show_engineering_analysis(owner, "phase")
    dialog.resize(1100, 650); application.processEvents()
    assert dialog.height() == 650
    assert dialog._panels["phase"][1].height() >= 240
    assert dialog._analysis_scroll.widget().height() < 800
    canvas = dialog._panels["phase"][1]
    assert canvas.parentWidget().rect().contains(canvas.geometry())
    canvas.draw()
    assert canvas.figure.bbox.height == pytest.approx(canvas.height() * canvas.device_pixel_ratio)
    bottom = canvas.figure.axes[-1].xaxis.label.get_window_extent(canvas.get_renderer()).y0
    assert bottom >= 0
