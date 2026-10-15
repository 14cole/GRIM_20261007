"""Mutable raw metadata and native array-layout regression checks."""
import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.memory import _preflight_native_archive_allocation


def grid(field):
    return RcsGrid(np.arange(field.shape[0]), np.arange(field.shape[1]),
                   8. + np.arange(field.shape[2]), ['VV'], rcs=field,
                   units={'frequency': 'GHz', 'rcs_linear_quantity': 'sigma_3d',
                          'rcs_log_unit': 'dBsm'})


@pytest.mark.parametrize('readonly_view', [False, True])
def test_raw_metadata_nan_edit_cannot_erase_canonical_samples(tmp_path, readonly_view):
    field = np.array([1+2j, 3+4j]).reshape(2, 1, 1, 1)
    dataset = grid(field)
    real, imag = (field / np.sqrt(4*np.pi)).real.copy(), (field / np.sqrt(4*np.pi)).imag.copy()
    supplied = (real.view(), imag.view()) if readonly_view else (real, imag)
    for value in supplied:
        value.flags.writeable = not readonly_view
    dataset.extra.update(rcs_amp_real=supplied[0], rcs_amp_imag=supplied[1])
    np.testing.assert_allclose(dataset.rcs, field, rtol=1e-15)
    real[0, 0, 0, 0] = imag[0, 0, 0, 0] = np.nan
    np.testing.assert_array_equal(dataset.rcs, field)
    np.testing.assert_array_equal(dataset.power, field.real**2 + field.imag**2)
    # Conflicting supplied metadata is rejected at export rather than silently
    # making a valid physical sample disappear in the saved data.
    with pytest.raises(ValueError):
        dataset.save(tmp_path / 'inconsistent.grim')
    dataset.extra.pop('rcs_amp_real')
    dataset.extra.pop('rcs_amp_imag')
    restored = RcsGrid.load(dataset.save(tmp_path / 'physical.grim'))
    np.testing.assert_array_equal(restored.rcs, field)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_native_preflight_accounts_for_fortran_copy(tmp_path, dtype):
    field = np.full((32, 2, 16, 1), 1+2j, dtype=dtype)
    paths = [grid(np.array(field, order=order)).save(tmp_path / order, compressed=False)
             for order in ('C', 'F')]
    reports = [_preflight_native_archive_allocation(path) for path in paths]
    assert reports[1]['estimated_peak_bytes'] >= reports[0]['estimated_peak_bytes'] + field.nbytes
    with pytest.raises(MemoryError):
        RcsGrid.load(paths[1], max_output_bytes=reports[0]['estimated_peak_bytes'])
    for path, report in zip(paths, reports):
        np.testing.assert_array_equal(RcsGrid.load(path, max_output_bytes=report['estimated_peak_bytes']).rcs, field)


def test_native_preflight_does_not_adopt_non_native_complex_dtype(tmp_path):
    dataset = grid(np.ones((32, 2, 16, 1), dtype=np.complex128))
    payload = dataset._native_archive_payload()
    native = tmp_path / 'native.grim'
    foreign = tmp_path / 'foreign.grim'
    with native.open('wb') as stream:
        np.savez(stream, **payload)
    payload['rcs_complex'] = payload['rcs_complex'].astype(np.dtype(np.complex128).newbyteorder('S'))
    with foreign.open('wb') as stream:
        np.savez(stream, **payload)
    expected = _preflight_native_archive_allocation(native)['estimated_peak_bytes']
    actual = _preflight_native_archive_allocation(foreign)['estimated_peak_bytes']
    assert actual >= expected + payload['rcs_complex'].nbytes
    np.testing.assert_array_equal(RcsGrid.load(foreign).rcs, dataset.rcs)
