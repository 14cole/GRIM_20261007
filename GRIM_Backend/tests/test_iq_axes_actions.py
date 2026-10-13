"""Mixed I/Q and physical-sweep axes keep independent toolbar coordinates."""
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from GRIM_Backend.plotting.actions import PlotOpsMixin


class _Spin:
    def __init__(self, value=0.):
        self._value = value
        self._blocked = False

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def blockSignals(self, blocked):
        previous, self._blocked = self._blocked, bool(blocked)
        return previous


class _Harness(PlotOpsMixin):
    def __init__(self):
        self.plot_figure = Figure()
        self.plot_canvas = FigureCanvasAgg(self.plot_figure)
        grid = self.plot_figure.add_gridspec(2, 3)
        self.iq = self.plot_figure.add_subplot(grid[:, 0])
        self.magnitude = self.plot_figure.add_subplot(grid[0, 1:])
        self.phase = self.plot_figure.add_subplot(grid[1, 1:], sharex=self.magnitude)
        self.colorbar = self.plot_figure.add_axes([.92, .1, .02, .8])
        self.plot_ax = self.iq
        self.plot_axes = [self.iq, self.magnitude, self.phase]
        self.plot_figure._grim_independent_x = True
        self.plot_figure._grim_independent_y = True
        self.last_plot_mode = "engineering_phase"
        self.plot_colorbars = []
        self.status = SimpleNamespace(showMessage=lambda _message: None)
        for name, value in (("xmin", -1.), ("xmax", 1.), ("ymin", -1.),
                            ("ymax", 1.), ("xstep", 0.), ("ystep", 0.)):
            setattr(self, "spin_plot_" + name, _Spin(value))
        self.iq.set(xlim=(-2., 2.), ylim=(-3., 3.))
        self.magnitude.set(xlim=(9., 10.), ylim=(-20., 5.))
        self.phase.set(ylim=(-180., 180.))
        self.colorbar.set(xlim=(0., 1.), ylim=(9., 10.))

    def _style_plot_axes(self):
        pass

    def _set_compare_sector_controls_visible(self, _visible):
        pass


def test_iq_limits_do_not_overwrite_frequency_axes_or_colorbar():
    h = _Harness()
    before = [(ax.get_xlim(), ax.get_ylim()) for ax in (h.magnitude, h.phase, h.colorbar)]
    h.iq.set(xlim=(-.5, .5), ylim=(-.75, .75))
    h._sync_plot_limit_spins_from_axes(h.iq)
    for ax, limits in zip((h.magnitude, h.phase, h.colorbar), before):
        np.testing.assert_allclose(ax.get_xlim(), limits[0])
        np.testing.assert_allclose(ax.get_ylim(), limits[1])
    np.testing.assert_allclose(h.iq.get_xlim(), [-.5, .5])


def test_frequency_zoom_links_magnitude_and_phase_without_moving_iq():
    h = _Harness()
    original_iq = h.iq.get_xlim(), h.iq.get_ylim()
    h.magnitude.set(xlim=(9.2, 9.6), ylim=(-12., 2.))
    h._sync_plot_limit_spins_from_axes(h.magnitude)
    np.testing.assert_allclose(h.phase.get_xlim(), [9.2, 9.6])
    np.testing.assert_allclose(h.phase.get_ylim(), [-180., 180.])
    np.testing.assert_allclose(h.iq.get_xlim(), original_iq[0])
    np.testing.assert_allclose(h.iq.get_ylim(), original_iq[1])
    h.spin_plot_xmin.setValue(9.3)
    h.spin_plot_xmax.setValue(9.5)
    h._apply_plot_limits()
    np.testing.assert_allclose(h.phase.get_xlim(), [9.3, 9.5])
    np.testing.assert_allclose(h.iq.get_xlim(), original_iq[0])


def test_fit_uses_scatter_and_line_collections_but_excludes_colorbar():
    h = _Harness()
    h.iq.scatter([2., 8.], [4., 6.])
    h.magnitude.add_collection(LineCollection([[[9., -20.], [10., 5.]]]))
    h.phase.plot([9., 10.], [-90., 120.])
    original_colorbar = h.colorbar.get_xlim(), h.colorbar.get_ylim()
    h._fit_both()
    for ax, x, y in ((h.iq, [2., 8.], [4., 6.]),
                     (h.magnitude, [9., 10.], [-20., 5.]),
                     (h.phase, [9., 10.], [-90., 120.])):
        assert ax.get_xlim()[0] < min(x) < max(x) < ax.get_xlim()[1]
        assert ax.get_ylim()[0] < min(y) < max(y) < ax.get_ylim()[1]
    assert h.iq.get_xlim()[1] < h.magnitude.get_xlim()[0]
    np.testing.assert_allclose(h.colorbar.get_xlim(), original_colorbar[0])
    np.testing.assert_allclose(h.colorbar.get_ylim(), original_colorbar[1])


def test_native_bounds_preserve_unsampled_extrema_and_ignore_cursor_artists():
    h = _Harness()
    h.iq.scatter([2., 3.], [4., 5.])  # Reduced overview omits the native extrema.
    h.iq.plot([1000.], [1000.])       # Cursor/annotation must not set fit bounds.
    h.iq._grim_data_limits = (1., 8., 2., 9.)
    h.magnitude.plot([9., 10.], [-20., 5.])
    h.phase.plot([9., 10.], [-90., 120.])
    original_y = [ax.get_ylim() for ax in h.plot_axes]
    h._fit_x()
    assert h.iq.get_xlim()[0] < 1. and 8. < h.iq.get_xlim()[1] < 10.
    for ax, limits in zip(h.plot_axes, original_y):
        np.testing.assert_allclose(ax.get_ylim(), limits)
    original_x = [ax.get_xlim() for ax in h.plot_axes]
    h._fit_y()
    assert h.iq.get_ylim()[0] < 2. and 9. < h.iq.get_ylim()[1] < 10.
    for ax, limits in zip(h.plot_axes, original_x):
        np.testing.assert_allclose(ax.get_xlim(), limits)


def test_clear_and_standard_plot_reset_mixed_coordinate_behavior():
    h = _Harness()
    h.plot_figure._grim_iq_held = ("signature", [object()])
    h._clear_plot()
    assert not h.plot_figure._grim_independent_x
    assert not h.plot_figure._grim_independent_y
    assert h.plot_figure._grim_iq_held is None
    assert h.plot_axes is None
    h.plot_figure._grim_independent_x = True
    h.plot_figure._grim_independent_y = True
    h.plot_figure._grim_iq_held = ("signature", [object()])
    h.last_plot_mode = "frequency"
    h._start_plot_render()
    assert not h.plot_figure._grim_independent_x
    assert not h.plot_figure._grim_independent_y
    assert h.plot_figure._grim_iq_held is None


def test_ordinary_shared_x_diagnostics_retain_existing_limit_behavior():
    h = _Harness()
    h.plot_figure._grim_independent_x = False
    h.plot_axes = [h.magnitude, h.phase]
    h.plot_ax = h.magnitude
    h.magnitude.set(xlim=(9.2, 9.7), ylim=(-10., 1.))
    h._sync_plot_limit_spins_from_axes(h.magnitude)
    np.testing.assert_allclose(h.magnitude.get_xlim(), [9.2, 9.7])
    np.testing.assert_allclose(h.phase.get_xlim(), [9.2, 9.7])
    np.testing.assert_allclose(h.phase.get_ylim(), [-180., 180.])


def test_single_iq_panel_fit_uses_native_extrema_without_fitting_guides():
    h = _Harness()
    h.plot_axes = [h.iq]
    h.plot_figure._grim_independent_x = False
    h.plot_figure._grim_independent_y = False
    h.iq.scatter([1e-12, 2e-12], [3e-12, 4e-12])
    h.iq.plot([0., 1.], [0., 1.])  # Radius/phase guides are not measured data.
    h.iq._grim_data_limits = (1e-12, 2e-12, 3e-12, 4e-12)
    h._fit_both()
    assert 0. < h.iq.get_xlim()[0] < 1e-12
    assert 2e-12 < h.iq.get_xlim()[1] < 3e-12
    assert 2e-12 < h.iq.get_ylim()[0] < 3e-12
    assert 4e-12 < h.iq.get_ylim()[1] < 5e-12


def test_iq_phase_guide_text_follows_theme_without_recoloring_user_annotations():
    h = _Harness()
    h.application_palette = {"panel_bg": "white", "grid": "gray", "text": "black", "border": "gray"}
    h.plot_bg_color = h.plot_grid_color = h.plot_text_color = None
    guide = h.iq.annotate("+90°", (0., 1.), color="white")
    guide.set_label("_iq_guide")
    note = h.iq.annotate("Selected target", (1., 1.), color="orange")
    h._style_axes(h.iq)
    assert guide.get_color() == "black"
    assert note.get_color() == "orange"
    h.plot_text_color = "cyan"
    h._style_axes(h.iq)
    assert guide.get_color() == "cyan"
    assert note.get_color() == "orange"
