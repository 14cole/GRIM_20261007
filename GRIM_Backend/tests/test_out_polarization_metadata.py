"""Optional OUT channel comments keep old tables readable."""
import numpy as np
import pytest

from GRIM_Backend.datasets.api import RcsGrid


@pytest.mark.parametrize('comment, expected', [
    ('VV', 'VV'), ('  HV  ', 'HV'), ('"custom channel"', 'CUSTOM CHANNEL'),
    ('', 'HH'), ('"unfinished', 'HH'), ('{}', 'HH'), ('""', 'HH'),
])
def test_optional_channel_comments(tmp_path, comment, expected):
    path = tmp_path / 'sample_HH.out'
    path.write_text('# polarization: ' + comment + '\n10 90 0 45\n', encoding='utf-8')
    restored = RcsGrid.load_out(path)
    assert restored.polarizations.tolist() == [expected]
    np.testing.assert_allclose(restored.phase, np.pi / 4.)


def test_explicit_channel_overrides_legacy_filename_tokens(tmp_path):
    path = tmp_path / 'sample_HH_VV.out'
    path.write_text('# polarization: VH\n10 90 0 0\n', encoding='utf-8')
    assert RcsGrid.load_out(path).polarizations.tolist() == ['VH']
