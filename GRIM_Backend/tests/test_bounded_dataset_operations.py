"""Large-array paths must derive only the samples currently being processed."""
import csv

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.storage import SampleComponent
from GRIM_Backend.datasets import arithmetic, combine
from GRIM_Backend.io import csv as csv_io


def grid(dtype=np.complex128, *, azimuths=None, factor=1):
    axes = np.arange(5.) if azimuths is None else np.asarray(azimuths)
    n = len(axes) * 2 * 7 * 2
    fields = (np.arange(1, n + 1) + 1j * np.arange(n, 0, -1)).astype(dtype)
    fields *= factor
    return RcsGrid(axes, [0., 1.], np.arange(1., 8.), ['HH', 'VV'],
                   rcs=fields.reshape(len(axes), 2, 7, 2))


@pytest.fixture
def bounded(monkeypatch):
    monkeypatch.setattr(arithmetic, '_COHERENT_OPERATION_BLOCK_CELLS', 9)
    monkeypatch.setattr(combine, '_JOIN_MERGE_BLOCK_CELLS', 9)
    monkeypatch.setattr(csv_io, '_CSV_BLOCK_CELLS', 9)
    original = SampleComponent.__getitem__
    def read(component, selection):
        result = original(component, selection)
        assert np.size(result) <= 9, 'whole sample component was derived'
        return result
    def materialize(*args, **kwargs):
        raise AssertionError('whole sample component was materialized')
    monkeypatch.setattr(SampleComponent, '__getitem__', read)
    monkeypatch.setattr(SampleComponent, '__array__', materialize)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('operation', ['incoherent_add', 'incoherent_subtract', 'arithmetic_db_subtract'])
def test_arithmetic_blocks_match_independent_power(dtype, operation, bounded):
    a, b = grid(dtype, factor=2), grid(dtype)
    p, q = a.rcs.real ** 2 + a.rcs.imag ** 2, b.rcs.real ** 2 + b.rcs.imag ** 2
    expected = {'incoherent_add': lambda: p + q,
                'incoherent_subtract': lambda: p.astype(float) - q.astype(float),
                'arithmetic_db_subtract': lambda: p.astype(float) / q.astype(float)}[operation]()
    result = getattr(a, operation)(b)
    np.testing.assert_array_equal(result.native_sample_payload()['rcs_power'], expected)
    if operation == 'arithmetic_db_subtract':
        assert result.linear_quantity() == 'power_ratio'


def test_mixed_precision_addition_preserves_sequential_rounding(bounded):
    grids = [grid(np.complex64, factor=value) for value in (1, 1.e-4)] + [grid(np.complex128)]
    expected_field = (grids[0].rcs + grids[1].rcs) + grids[2].rcs
    np.testing.assert_array_equal(grids[0].coherent_add_many(*grids[1:]).rcs, expected_field)
    expected_power = sum((item.rcs.real ** 2 + item.rcs.imag ** 2 for item in grids))
    np.testing.assert_array_equal(grids[0].incoherent_add_many(*grids[1:]).native_sample_payload()['rcs_power'], expected_power)


@pytest.mark.parametrize('operation', ['join_many', 'stitch_many'])
def test_combine_derives_blocks_only(operation, bounded):
    a, b = grid(azimuths=[0., 1.]), grid(azimuths=[2., 3.], factor=2)
    result = getattr(RcsGrid, operation)(a, b)
    expected = np.concatenate([a.rcs, b.rcs])
    np.testing.assert_allclose(result.rcs, expected, rtol=5.e-15)
    np.testing.assert_array_equal(result.azimuths, [0., 1., 2., 3.])


@pytest.mark.parametrize('include_phase', [False, True])
def test_csv_blocks_preserve_row_order_and_values(tmp_path, include_phase, bounded, monkeypatch):
    source = grid()
    values = source.rcs.copy()
    if not include_phase:
        def unwanted_phase(*args, **kwargs):
            raise AssertionError('magnitude-only export derived phase')
        monkeypatch.setattr(source, 'phase_slice', unwanted_phase)
    path = tmp_path / 'rows.csv'
    csv_io.write_flat_csv(source, path, include_phase=include_phase)
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    assert len(rows) == source.size
    for row, index in zip(rows, np.ndindex(source.shape)):
        assert [float(row[key]) for key in ('azimuth', 'elevation', 'frequency')] == [
            axis[i] for axis, i in zip((source.azimuths, source.elevations, source.frequencies), index)
        ]
        assert row['polarization'] == source.polarizations[index[-1]]
        assert float(row['magnitude_power_linear']) == values[index].real ** 2 + values[index].imag ** 2
        if include_phase:
            assert float(row['phase_deg']) == pytest.approx(np.degrees(np.angle(values[index])))


def test_subtraction_reports_all_negative_cells_across_blocks(bounded):
    a, b = grid(), grid(factor=2)
    minimum = float(np.min(a.rcs.real ** 2 + a.rcs.imag ** 2 - b.rcs.real ** 2 - b.rcs.imag ** 2))
    with pytest.raises(ValueError, match=rf'{a.size} cell\(s\); minimum difference is {minimum:.17g}'):
        a.incoherent_subtract(b)


@pytest.mark.parametrize('operation', ['incoherent_add', 'incoherent_add_many'])
def test_adopted_power_sum_keeps_historical_nonfinite_and_negative_cleanup(operation, request):
    a, b = grid(), grid()
    a.rcs_power.flat[:3] = [-100., np.inf, np.nan]
    b.rcs_power.flat[:3] = [1., 2., 3.]
    a_before, b_before = a.rcs_power.copy(), b.rcs_power.copy()
    request.getfixturevalue("bounded")
    result = getattr(a, operation)(b)
    np.testing.assert_array_equal(result.native_sample_payload()['rcs_power'].flat[:3], [0., np.nan, np.nan])
    np.testing.assert_array_equal(a.rcs_power, a_before)
    np.testing.assert_array_equal(b.rcs_power, b_before)


def test_adopted_complex_sum_masks_overflow_as_one_missing_sample(bounded):
    a, b = grid(), grid()
    values = a.rcs.copy()
    values.flat[0] = 1.e308 + 1j
    a.set_complex(Ellipsis, values)
    b.set_complex(Ellipsis, values)
    with np.errstate(over='ignore', invalid='ignore'):
        result = a.coherent_add(b)
    assert np.isnan(result.rcs.flat[0].real)
    assert np.isnan(result.rcs.flat[0].imag)
    assert np.isfinite(a.rcs.flat[0].real)


@pytest.mark.parametrize('dtype', [np.int8, np.int64, np.float16])
@pytest.mark.parametrize('operation', ['incoherent_add', 'incoherent_add_many'])
def test_legacy_numeric_setters_sum_before_conversion_and_cleanup(dtype, operation, request):
    a, b = grid(), grid()
    a.rcs_power = np.full(a.shape, 120, dtype=dtype)
    b.rcs_power = np.full(b.shape, 120, dtype=dtype)
    expected = (a.rcs_power + b.rcs_power).astype(np.float32)
    expected = np.maximum(expected, 0.)
    request.getfixturevalue('bounded')
    result = getattr(a, operation)(b)
    actual = result.native_sample_payload()['rcs_power']
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
