"""Exercise installed 2-D and BoR solvers against analytic reference cases.

Run with the target environment's Python from an unrelated working directory.
This script intentionally does not add the source checkout to sys.path.
"""
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def main():
    import numpy as np
    import scipy
    from ghost_backend.twod import solver as td
    from ghost_backend.bor import solver as bor
    from ghost_backend.validation.cylinder import sigma_pec_cylinder
    from ghost_backend.validation.sphere import sigma_pec_sphere, sigma_dielectric_sphere

    installation = Path(td.__file__).resolve().parents[2]
    if 'site-packages' not in installation.parts:
        raise RuntimeError('This check must import an installed wheel, not the source checkout.')
    required = {
        'ghost_backend.bor.native': ['bor_stream_kernel.c', 'build_kernel.py'],
        'ghost_backend.twod.assembly.native': ['table.c', 'far.c', 'build.py'],
        'ghost_backend.geometry': ['templates/point_features_template.csv',
                                   'templates/line_features_template.csv', 'geometries/body.geo'],
    }
    for package, names in required.items():
        for name in names:
            if not importlib.resources.files(package).joinpath(name).is_file():
                raise RuntimeError('Missing installed asset: '+package+'/'+name)
    radius, frequency = .1, .3e9
    angle = np.linspace(0., -2*np.pi, 65)
    points = radius*np.column_stack((np.cos(angle), np.sin(angle)))
    body = dict(segments=[dict(name='circle', seg_type=2, properties=['2', '1', '0', '0', '0'],
        point_pairs=[dict(x1=float(a[0]), y1=float(a[1]), x2=float(b[0]), y2=float(b[1]))
                     for a, b in zip(points[:-1], points[1:])])], ibcs=[], dielectrics=[])
    errors = {}
    for polarization in ('TE', 'TM'):
        result = td.solve_monostatic_rcs_2d_single_polarization(body, [frequency/1e9], [0., 90.], polarization,
            geometry_units='meters', strict_quality_gate=False, execution_options=dict(factorization='dense'))
        truth = sigma_pec_cylinder(radius, frequency, polarization)
        errors['2d_'+polarization] = max(abs(10*np.log10(row['rcs_linear']/truth)) for row in result['samples'])
    points = bor.sphere_generatrix(radius, 24)
    pec = bor.solve_bor(points, frequency, [0., 90.], workers=1,
                        assembly='streaming', bor_options=dict(factorization='dense'))
    truth = sigma_pec_sphere(radius, frequency)
    errors['bor_pec'] = float(np.max(np.abs(10*np.log10(np.r_[pec['sigma_vv'], pec['sigma_hh']]/truth))))
    dielectric = bor.solve_bor_dielectric(points, frequency, [0., 90.], 3., workers=1,
                                          bor_options=dict(factorization='dense'))
    truth = sigma_dielectric_sphere(radius, 3., 1., frequency)
    errors['bor_dielectric'] = float(np.max(np.abs(10*np.log10(np.r_[dielectric['sigma_vv'], dielectric['sigma_hh']]/truth))))
    if not all(np.isfinite(value) and value < .15 for value in errors.values()):
        raise AssertionError('Analytic smoke-check tolerance exceeded: '+repr(errors))
    if any(name.split('.')[0] in ('PySide6', 'PyQt6', 'PySide2', 'PyQt5') for name in sys.modules):
        raise AssertionError('A headless solver imported Qt.')
    print(json.dumps(dict(version=importlib.metadata.version('ghost-em2d'), python=sys.version.split()[0],
        numpy=np.__version__, scipy=scipy.__version__, installed_path=str(installation),
        maximum_rcs_error_db=errors), indent=2))


if __name__ == '__main__':
    main()
