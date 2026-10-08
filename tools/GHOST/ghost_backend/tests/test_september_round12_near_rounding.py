"""Round 12: the BoR near angular rule hit a rounding floor on very close point pairs (tiny or graded elements)."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import kernels, solver as bor
import legacy_near_rules as legacy

# The pair on which a graded 240-element sphere stopped: 2.3e-7 m apart at rho = 0.1 m, same element.
POINT = (0.0999995129968009, 0.00030750568258239226, 0.002454366796437894, -0.9999969880372783,
         0.09999951356208991, 0.00030727536358078936, 0.002454366796437894, -0.9999969880372783)
K, M_MAX = 33., 16


def _separated(ratio, rho=.1):
    gap = ratio*rho
    return tuple(np.array([value]) for value in (rho, .5*gap, 0., -1., rho, -.5*gap, 0., -1.))


def _scale(values):
    return max(float(np.max(np.abs(value))) for value in values)


def _distance(first, second):
    return max(float(np.max(np.abs(a-b))) for a, b in zip(first, second))/_scale(second)


class StableBracketTests(unittest.TestCase):
    def test_closed_forms_are_the_sampled_brackets(self):
        # where the sampled forms are accurate (separations of 1e-4 to 1e-2 rho, skew tangents, lossy medium)
        rng = np.random.default_rng(5)
        count = 40
        rho_p, z_p = .1+.01*rng.random(count), .02*rng.random(count)
        angle = 2*np.pi*rng.random(count)
        gap, direction = 10**rng.uniform(-5, -3, count), 2*np.pi*rng.random(count)
        skew = angle+rng.normal(0, .3, count)
        points = (rho_p, z_p, np.cos(angle), np.sin(angle), rho_p+gap*np.cos(direction), z_p+gap*np.sin(direction),
                  np.cos(skew), np.sin(skew))
        for rule in (legacy._mfie_kernels_near_rule, legacy._ibc_kernels_near_rule):
            for wavenumber in (33., 33.-4j):
                with self.subTest(rule=rule.__name__, k=wavenumber):
                    sampled = rule(*points, wavenumber, 6, order=64, tail_order=96)
                    stable = rule(*points, wavenumber, 6, order=64, tail_order=96, stable=True)
                    self.assertLess(_distance(stable, sampled), 1e-10)
                    with patch.object(kernels, '_native_brackets', lambda *args, **kwargs: None):   # the NumPy forms
                        self.assertLess(_distance(stable, rule(*points, wavenumber, 6, order=64, tail_order=96)), 1e-10)

    def test_sampled_brackets_have_a_rounding_floor_the_closed_forms_do_not(self):
        point = _separated(1e-6)
        for rule in (legacy._mfie_kernels_near_rule, legacy._ibc_kernels_near_rule):
            with self.subTest(rule=rule.__name__):
                sampled = rule(*point, K, M_MAX, order=3000, tail_order=3000)
                stable = rule(*point, K, M_MAX, order=3000, tail_order=3000, stable=True)
                lower = rule(*point, K, M_MAX, order=2000, tail_order=2000, stable=True)
                self.assertLess(_distance(lower, stable), 2e-9)            # converged: 4e-10
                self.assertGreater(_distance(sampled, stable), kernels.NEAR_ANGULAR_RTOL)   # 3e-7: above the tolerance

    def test_checked_rules_recover_instead_of_raising(self):
        # "BoR near angular quadrature did not converge at the maximum order ... relative change=4.83e-08"
        for checked, rule in ((kernels.mfie_kernels_near, legacy._mfie_kernels_near_rule),
                              (kernels.ibc_kernels_near, legacy._ibc_kernels_near_rule)):
            with self.subTest(rule=rule.__name__):
                values = checked(*POINT, 3.3/.1, M_MAX)
                reference = rule(*[np.array([value]) for value in POINT], 3.3/.1, M_MAX, order=3000, tail_order=3000,
                                 stable=True)
                self.assertLess(_distance(values, reference), 2*kernels.NEAR_ANGULAR_RTOL)

    def test_ordinary_pairs_never_reach_the_closed_forms(self):
        # results that converged before must not change: the rescue starts only at the maximum order
        calls = []
        original = kernels._stable_brackets
        def watched(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        with patch.object(kernels, '_stable_brackets', watched):
            out = bor.solve_bor(bor.sphere_generatrix(.1, 24), 3.3*299792458./(2*np.pi*.1), [0., 90.], zs=100+50j,
                                workers=1, bor_options=dict(factorization='dense'))
        self.assertEqual(calls, [])
        self.assertTrue(np.all(np.isfinite(out['sigma_vv'])))

    def test_a_generatrix_with_tiny_elements_solves(self):
        # 30 elements, the two at the equator split six times (h/64): raised before
        theta = list(np.linspace(0., np.pi, 31))
        step = theta[1]-theta[0]
        cuts = [step*.5**level for level in range(1, 7)]
        theta = sorted(set(theta+[np.pi/2-cut for cut in cuts]+[np.pi/2+cut for cut in cuts]))
        points = np.column_stack([.1*np.sin(theta), .1*np.cos(theta)])
        points[0, 0] = points[-1, 0] = 0.
        middle = .5*(np.array(theta[:-1])+np.array(theta[1:]))
        out = bor.solve_bor(points, 3.3*299792458./(2*np.pi*.1), [0., 90., 180.], formulation='cfie',
                            zs=np.where(middle < np.pi/2, 0., 100+50j), workers=1, bor_options=dict(factorization='dense'))
        self.assertEqual(out['n_unknowns'], 2*len(points))
        self.assertTrue(np.all(np.isfinite(out['sigma_vv'])) and np.all(np.asarray(out['sigma_vv']) > 0))


if __name__ == '__main__':
    unittest.main()
