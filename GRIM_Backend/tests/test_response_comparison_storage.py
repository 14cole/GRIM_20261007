"""Assembly comparisons stream legacy and compact native response cuts."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from GRIM_Backend.assembly.response_comparison import read_response_cut


UNITS = json.dumps(dict(azimuth='deg', elevation='deg', frequency='GHz',
                       rcs_linear_quantity='sigma_3d', rcs_log_unit='dBsm'))


def save(path, **samples):
    with path.open('wb') as stream:
        np.savez_compressed(stream, azimuths=[0., 10., 20.], elevations=[-5., 0.],
                            frequencies=[8., 9.], polarizations=['VV', 'HH'],
                            units=UNITS, **samples)


@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('domain,factor', [('physical', 1.), ('raw_field', 4. * np.pi)])
def test_compact_cut_matches_direct_power_without_loading_full_samples(tmp_path, order, dtype, domain, factor):
    field = np.array(np.arange(24).reshape(3, 2, 2, 2) + .25j, dtype=dtype, order=order)
    field[0, 1, 1, 1] = 0.
    field[2, 1, 1, 1] = complex(np.nan, np.nan)
    path = tmp_path / 'complex.grim'
    save(path, grim_format_version=2, storage_kind='complex', storage_domain=domain, rcs_complex=field)
    original = np.lib.npyio.NpzFile.__getitem__

    def metadata_only(archive, key):
        assert key not in {'rcs_complex', 'rcs_power', 'rcs_phase'}, 'full sample member was loaded'
        return original(archive, key)

    with patch.object(np.lib.npyio.NpzFile, '__getitem__', metadata_only):
        az, db = read_response_cut(path, frequency=9., elevation=0., polarization='HH')
    selected = field[:, 1, 1, 1].astype(np.complex128)
    with np.errstate(divide='ignore', invalid='ignore'):
        expected = 10. * np.log10((selected.real**2 + selected.imag**2) * factor)
    np.testing.assert_array_equal(az, [0., 10., 20.])
    np.testing.assert_allclose(db, expected, equal_nan=True)


@pytest.mark.parametrize('kind', ['legacy', 'power', 'power_phase'])
def test_power_only_and_mixed_phase_cut_remain_supported(tmp_path, kind):
    power = np.arange(24, dtype=float).reshape(3, 2, 2, 2)
    data = dict(rcs_power=power)
    if kind != 'legacy':
        data.update(grim_format_version=2, storage_kind=kind, storage_domain='physical')
    if kind == 'power_phase':
        data['rcs_phase'] = np.full(power.shape, np.nan)
    path = tmp_path / 'power.grim'
    save(path, **data)
    np.testing.assert_allclose(read_response_cut(path, frequency=9., elevation=0., polarization='HH')[1],
                               10. * np.log10(power[:, 1, 1, 1]))


def test_legacy_delta_normalization_is_preserved(tmp_path):
    path = tmp_path / 'delta.grim'
    save(path, grim_format_version=2, storage_kind='complex', storage_domain='raw_field',
         rcs_complex=np.full((3, 2, 2, 2), 3. + 4.j), rcs_domain='delta', power_domain='delta_amp_sq')
    np.testing.assert_allclose(read_response_cut(path, frequency=9., elevation=0., polarization='HH')[1],
                               10. * np.log10(25.))


def test_conflicting_authorities_are_not_silently_selected(tmp_path):
    path = tmp_path / 'conflict.grim'
    save(path, grim_format_version=2, storage_kind='complex', storage_domain='physical',
         rcs_complex=np.ones((3, 2, 2, 2), complex), rcs_power=np.zeros((3, 2, 2, 2)))
    with pytest.raises(ValueError, match='conflicting'):
        read_response_cut(path, frequency=9., elevation=0., polarization='HH')


def test_duplicate_raw_members_match_native_reader_rejection(tmp_path):
    path = tmp_path / 'duplicate.grim'
    save(path, grim_format_version=2, storage_kind='complex', storage_domain='raw_field',
         rcs_complex=np.ones((3, 2, 2, 2), complex), rcs_amp_real=np.zeros((3, 2, 2, 2)))
    with pytest.raises(ValueError, match='duplicate raw'):
        read_response_cut(path, frequency=9., elevation=0., polarization='HH')
