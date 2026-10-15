"""Known physical answers from reusable analysis results, without a dialog."""
from dataclasses import replace
import subprocess
import sys

import numpy as np

from GRIM_Backend.plotting.analysis_results import AnalysisOptions, phase_analysis, polarization_analysis
from GRIM_Backend.plotting.engineering_tools import EngineeringCut


def cut(phase=0., power=4., label='A'):
    x = np.arange(1., 12.) * .01 + 8.
    field = np.sqrt(power) * np.exp(1j * np.broadcast_to(phase, x.shape))
    return EngineeringCut(x, abs(field)**2, field, np.full(len(x), 10*np.log10(power)),
        'frequency', 'GHz', 'dBsm', label, '+jwt', [], 'sigma_3d', 'VV', {}, np.arange(len(x)))


def test_phase_result_retains_branch_boundaries_and_known_delay():
    source = cut()
    delay = 5.e-9
    source = replace(source, field=2*np.exp(-2j*np.pi*source.x*1e9*delay))
    selected = np.array([0, 1, 2, 8, 9, 10])
    source = replace(source, x=source.x[selected], field=source.field[selected],
                     power=source.power[selected], db=source.db[selected], native_indices=selected)
    result = phase_analysis(source, 'slope', AnalysisOptions())
    delays = [float(value) for name, value, _unit in result.rows if name.endswith('apparent delay s')]
    np.testing.assert_allclose(delays, [delay, delay], rtol=1.e-10)
    assert len(result.traces[0].runs) == 2
    assert len(result.samples) == 6
    assert all('Run' in sample[0] for sample in result.samples)


def test_residual_rows_traces_and_export_samples_describe_same_phase():
    result = phase_analysis(cut(np.deg2rad(50.)), 'residual', AnalysisOptions(),
                            reference=cut(np.deg2rad(20.), label='B'))
    np.testing.assert_allclose(result.traces[0].y, 30.)
    np.testing.assert_array_equal(result.samples[0][2], result.traces[0].y)
    row = next(row for row in result.rows if row[0] == 'Circular mean phase difference')
    assert float(row[1]) == 30.


def test_polarization_result_uses_linear_power_ratio_and_circular_phase():
    left, right = cut(np.deg2rad(179.), power=4.), cut(np.deg2rad(-179.), power=1., label='B')
    ratio = polarization_analysis({'HH': left, 'VV': right}, [('HH', 'VV')],
                                  'ratios', AnalysisOptions(), fallback=left)
    np.testing.assert_allclose(ratio.traces[-1].y, 10*np.log10(4.))
    phase = polarization_analysis({'HH': left, 'VV': right}, [('HH', 'VV')],
                                  'phase', AnalysisOptions(), fallback=left)
    np.testing.assert_allclose(phase.traces[-1].y, -2.)
    assert not any('median' in row[0] for row in phase.rows)


def test_native_iq_result_retains_physical_units_and_equal_scale_contract():
    source = cut(np.deg2rad(50.), power=5.)
    result = phase_analysis(source, 'iq', AnalysisOptions())
    assert result.equal_aspect
    assert result.axes[0]['xlabel'] == 'I: real field (m)'
    iq = result.traces[0]
    np.testing.assert_allclose(iq.x**2 + iq.y**2, 5.)


def test_analysis_services_import_without_qt_or_pyplot():
    script = '''
import importlib.abc, sys
class RejectUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('PySide', 'PyQt', 'matplotlib.pyplot')):
            raise AssertionError('Numerical service imported ' + fullname)
sys.meta_path.insert(0, RejectUI())
from GRIM_Backend.plotting.analysis_results import phase_analysis, polarization_analysis
from GRIM_Backend.plotting.requests import prepare_sweep, prepare_cdf, prepare_image_plane
'''
    completed = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
