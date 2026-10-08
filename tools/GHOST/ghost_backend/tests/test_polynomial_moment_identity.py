"""Persistent moments distinguish welded quadrature rules across generations."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import polynomial_quadrature as pq


def element(start, stop, node_ids, panel_index):
    start, stop = np.array(start), np.array(stop)
    return SimpleNamespace(p0=start, p1=stop, node_ids=node_ids, panel_index=panel_index,
                           normal=np.array([0., 1.]), length=np.linalg.norm(stop-start))


def classified(obs, src):
    task = pq._Task(0, 0, 0, ((0., 1.), (0., 1.)))
    pq._classify(task, obs, src)
    return task


class MomentIdentityTests(unittest.TestCase):
    def test_changed_endpoint_welding_recomputes_the_current_quadrature_rule(self):
        obs = element([0., 0.], [.01, 0.], (0, 1, 2), 0)
        welded = element([.0100000015, 0.], [.02, 0.], (1, 3, 4), 1)
        separate = copy.deepcopy(welded)
        separate.node_ids = (5, 3, 4)
        old_task, new_task = classified(obs, welded), classified(obs, separate)
        self.assertEqual((old_task.kind, old_task.shared), ('shared', (False, True)))
        self.assertEqual((new_task.kind, new_task.shared), ('regular', None))
        cache = pq.MomentCache(4096)
        def compute(source, task, retained):
            return pq._moments([task], [(obs, source)], 30., True, 20, retained, 1, None)[0]
        old = compute(welded, old_task, cache)
        current = compute(separate, new_task, cache)
        self.assertEqual((cache.hits, cache.stores), (0, 2))
        np.testing.assert_array_equal(current, compute(separate, new_task, None))
        self.assertFalse(np.array_equal(old, current))

    def test_changed_welded_corner_orientation_has_a_distinct_key(self):
        obs = element([0., 0.], [1e-9, 0.], (0, 1, 2), 0)
        first = element([0., 1.5e-9], [1e-9, 1.5e-9], (1, 3, 4), 1)
        second = copy.deepcopy(first)
        second.node_ids = (3, 0, 4)
        a, b = classified(obs, first), classified(obs, second)
        self.assertEqual(a.kind, b.kind)
        self.assertEqual(a.shared, (False, True))
        self.assertEqual(b.shared, (True, False))
        self.assertNotEqual(pq._moment_key(30., True, obs, first, a, 20),
                            pq._moment_key(30., True, obs, second, b, 20))

    def test_degree_change_retains_the_same_kernel_moment_identity(self):
        quadratic = element([0., 0.], [.01, 0.], (0, 1, 2), 0)
        cubic = copy.deepcopy(quadratic)
        cubic.node_ids = (0, 1, 2, 3)
        a, b = classified(quadratic, quadratic), classified(cubic, cubic)
        self.assertEqual(pq._moment_key(30., True, quadratic, quadratic, a, 20),
                         pq._moment_key(30., True, cubic, cubic, b, 20))


if __name__ == '__main__':
    unittest.main()
