"""Complete solver datasets remain compact through table, summary and plots."""
import os
from pathlib import Path
import sys
from unittest import mock
import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableView, QTableWidget
from ghost_backend.twod.samples import FIELDS, SampleTable
from ghost_backend.ui.result_table import ResultRows, RowSelection, SolverResultsTableModel
from ghost_backend.ui import solver


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('kind', ['2d_monostatic', '2d_bistatic', 'bor'])
def test_table_keeps_original_order_headers_and_display_formats(app, kind):
    source = [dict(frequency_ghz=f, polarization=p, theta_inc_deg=i, theta_scat_deg=o, rcs_linear=v)
              for f, p, i, o, v in [(3., 'VV', 10., 25., .456789), (1., 'HH', 50., 20., 0.),
                                     (1., 'HH', 10., 30., 1.23456789), (1., 'VV', 5., 15., 2.)]]
    result = dict(samples=source, solver='bor_mom_rcs' if kind == 'bor' else 'mom_2d',
                  scattering_mode='bistatic' if kind == '2d_bistatic' else 'monostatic')
    model = SolverResultsTableModel()
    model.set_result(result, kind, solver._display_db_value)
    assert model.rowCount() == len(source)
    assert model.columnCount() == (6 if kind == '2d_bistatic' else 5)
    assert model.headerData(model.columnCount()-1, Qt.Horizontal) == ('RCS (dBsm)' if kind == 'bor' else 'Width (dBke)')
    expected = sorted(source, key=lambda row: (row['frequency_ghz'], row['polarization'], row['theta_inc_deg'], row['theta_scat_deg']))
    for index, row in enumerate(expected):
        cells = [f"{row['frequency_ghz']:.6g}", row['polarization']]
        if kind == '2d_bistatic':
            cells.append(f"{row['theta_inc_deg']:.6g}")
        cells += [f"{row['theta_scat_deg']:.6g}", f"{row['rcs_linear']:.6e}", f"{solver._display_db_value(result, row):.3f}"]
        assert [model.data(model.index(index, column)) for column in range(model.columnCount())] == cells
    assert not (model.flags(model.index(0, 0)) & Qt.ItemIsEditable)


def test_label_overlay_is_lazy_and_does_not_change_original_samples():
    original = [{'frequency_ghz': 1., 'polarization': 'internal'}]
    rows = ResultRows(dict(co_solved_samples={'VV': original}))
    assert rows[0]['polarization'] == 'VV'
    assert original[0]['polarization'] == 'internal'
    row = rows[0]
    row['frequency_ghz'] = 9.
    assert rows[0]['frequency_ghz'] == 1.
    assert isinstance(rows[:1], RowSelection)
    fallback = ResultRows(dict(samples=[{'frequency_ghz': 1.}], polarization_export='HH'))
    assert fallback[0]['polarization'] == 'HH'


class GuardedSamples(SampleTable):
    allow_rows = False
    row_reads = 0
    def __iter__(self):
        raise AssertionError('GUI expanded the compact dataset into rows')
    def __getitem__(self, index):
        if not self.allow_rows:
            raise AssertionError('Table setup/plotting read a row instead of compact columns')
        self.row_reads += 1
        return super().__getitem__(index)


def test_100k_rows_stay_lazy_and_complete_through_gui_and_plot(app):
    count = 50_000
    data = np.zeros((count, len(FIELDS)), dtype=float)
    data[:, FIELDS.index('frequency_ghz')] = 3.
    data[:, FIELDS.index('theta_inc_deg')] = np.arange(count, dtype=float)
    data[:, FIELDS.index('theta_scat_deg')] = np.arange(count, dtype=float)[::-1]
    data[:, FIELDS.index('rcs_linear')] = 1.
    vv, hh = GuardedSamples(data), GuardedSamples(data)
    result = dict(co_solved_samples={'VV': vv, 'HH': hh}, metadata={})
    ui = solver.SolverTab()
    try:
        assert isinstance(ui.table_results, QTableView)
        assert not isinstance(ui.table_results, QTableWidget)
        ui._populate_results_table(result)
        model = ui.table_results.model()
        assert model.rowCount() == 100_000
        assert isinstance(model.rows, RowSelection)
        assert model.rows.indices.nbytes == 100_000*np.dtype(np.intp).itemsize
        assert model.rows.rows.blocks[0][0] is vv
        assert model.rows.rows.blocks[1][0] is hh
        assert solver._result_sample_counts(result)['observation_count'] == count
        with mock.patch.object(ui.canvas, 'draw_idle'):
            ui._plot_results(result)
        assert len(ui.canvas.ax.lines) == 2
        assert sum(len(line.get_xdata()) for line in ui.canvas.ax.lines) == 100_000
        for line in ui.canvas.ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), np.arange(count))
        assert vv.row_reads == hh.row_reads == 0
        vv.allow_rows = hh.allow_rows = True
        assert model.data(model.index(99_999, 0)) == '3'
        assert model.data(model.index(99_999, 1)) == 'VV'
        assert model.data(model.index(99_999, 2)) == '0'
        assert vv.row_reads == 3
    finally:
        ui.deleteLater()
        app.processEvents()


def test_plot_groups_keep_bistatic_incidence_channels_separate(app):
    rows = [dict(frequency_ghz=2., theta_inc_deg=inc, theta_scat_deg=obs,
                 polarization=pol, rcs_linear=obs+1.)
            for inc in (0., 30.) for pol in ('VV', 'HH') for obs in (90., 0., 45.)]
    result = dict(samples=rows, scattering_mode='bistatic')
    groups = solver._result_plot_groups(result)
    assert len(groups) == 4
    assert all(isinstance(group, RowSelection) for group in groups.values())
    assert sum(map(len, groups.values())) == len(rows)
    ui = solver.SolverTab()
    try:
        with mock.patch.object(ui.canvas, 'draw_idle'):
            ui._plot_results(result)
        assert len(ui.canvas.ax.lines) == 4
        for line in ui.canvas.ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), [0., 45., 90.])
            expected = [solver.compute_dbke_from_linear(value+1., 2.) for value in (0., 45., 90.)]
            np.testing.assert_array_equal(line.get_ydata(), expected)
    finally:
        ui.deleteLater()
        app.processEvents()
