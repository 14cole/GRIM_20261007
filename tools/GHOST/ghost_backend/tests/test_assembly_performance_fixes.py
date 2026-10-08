"""Feature-assembly hot-path fixes: body lookup and the oriented pattern cache.

These pin behaviour that the optimizations must not change: the body is still
resolved only at explicitly solved aspects, and the oriented-pattern cache is a
pure speed device whose presence or size never moves an amplitude.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
    sys.path.insert(0, str(BACKEND.parent))

import ghost_backend.assembly.fields as feature_sum  # noqa: E402


class BodyAspectLookupTests(unittest.TestCase):
    """`_bor_amp_interp` resolves stored nodes only, and never interpolates."""

    @staticmethod
    def _body(count=181, seed=2):
        rng = np.random.default_rng(seed)
        theta = np.linspace(0.0, 180.0, count)
        return {
            "theta_deg": theta,
            "amp_vv": rng.standard_normal(count) + 1j * rng.standard_normal(count),
        }

    def test_returns_the_stored_sample_for_every_node(self):
        body = self._body()
        got = feature_sum._bor_amp_interp(body, "amp_vv", body["theta_deg"])
        np.testing.assert_array_equal(got, body["amp_vv"])

    def test_repeated_and_shuffled_queries_keep_their_order(self):
        body = self._body()
        rng = np.random.default_rng(7)
        order = rng.integers(0, len(body["theta_deg"]), 500)
        got = feature_sum._bor_amp_interp(
            body, "amp_vv", body["theta_deg"][order]
        )
        np.testing.assert_array_equal(got, body["amp_vv"][order])

    def test_unsolved_aspect_is_refused_rather_than_interpolated(self):
        body = self._body(count=19)
        midpoint = 0.5 * (body["theta_deg"][3] + body["theta_deg"][4])
        with self.assertRaisesRegex(ValueError, "no explicitly solved aspect"):
            feature_sum._bor_amp_interp(body, "amp_vv", np.asarray([midpoint]))

    def test_near_miss_beyond_tolerance_is_refused(self):
        body = self._body(count=19)
        query = np.asarray([body["theta_deg"][5] + 1e-7])
        with self.assertRaisesRegex(ValueError, "no explicitly solved aspect"):
            feature_sum._bor_amp_interp(body, "amp_vv", query)

    def test_query_below_and_above_the_axis_is_refused(self):
        body = self._body(count=19)
        for outside in (-5.0, 200.0):
            with self.subTest(aspect=outside):
                with self.assertRaisesRegex(ValueError, "no explicitly solved aspect"):
                    feature_sum._bor_amp_interp(
                        body, "amp_vv", np.asarray([outside])
                    )

    def test_scalar_and_shaped_queries_are_preserved(self):
        body = self._body(count=19)
        nodes = body["theta_deg"]
        scalar = feature_sum._bor_amp_interp(body, "amp_vv", np.asarray(nodes[2]))
        self.assertEqual(np.shape(scalar), ())
        self.assertEqual(scalar, body["amp_vv"][2])
        shaped = feature_sum._bor_amp_interp(
            body, "amp_vv", nodes[:6].reshape(3, 2)
        )
        self.assertEqual(shaped.shape, (3, 2))
        np.testing.assert_array_equal(shaped.ravel(), body["amp_vv"][:6])


def _pattern(seed=3):
    rng = np.random.default_rng(seed)
    azimuths = np.linspace(-180.0, 180.0, 25)
    elevations = np.linspace(0.0, 90.0, 10)
    amplitude = (
        rng.standard_normal((25, 10, 1, 3))
        + 1j * rng.standard_normal((25, 10, 1, 3))
    )
    return feature_sum.PreparedPointPattern(
        azimuths, elevations, np.asarray([10.0]), amplitude,
        {"VV": 0, "HH": 1, "VH": 2},
    )


class OrientedPatternCacheTests(unittest.TestCase):
    """The cache is bounded by memory only, and never changes an amplitude."""

    def setUp(self):
        self.pattern = _pattern()
        directions, _, _ = feature_sum.directions_from_aspect_roll(
            np.linspace(10.0, 170.0, 9), np.linspace(0.0, 315.0, 8)
        )
        self.directions = directions
        rng = np.random.default_rng(12)
        self.placements = []
        for index in range(24):
            # Twelve recurring orientations, well past the old four-entry cap.
            angle = (index % 12) * (2.0 * np.pi / 12.0)
            normal = np.asarray([np.cos(angle), np.sin(angle), 0.0])
            self.placements.append((
                np.asarray([
                    0.5 * np.cos(angle), 0.5 * np.sin(angle),
                    rng.uniform(-1.0, 1.0),
                ]),
                normal,
            ))

    def _evaluate(self, cache):
        return [
            feature_sum.point_scatterer_amplitude(
                self.pattern, location, normal, self.directions, 10.0,
                _interpolator_cache={}, _oriented_pattern_cache=cache,
            )
            for location, normal in self.placements
        ]

    def test_cached_and_uncached_amplitudes_are_identical(self):
        cached = self._evaluate({})
        uncached = self._evaluate(None)
        self.assertEqual(len(cached), len(uncached))
        for left, right in zip(cached, uncached):
            for channel in ("F_vv", "F_hh", "F_vh"):
                np.testing.assert_allclose(
                    left[channel], right[channel], rtol=1e-12, atol=0.0
                )

    def test_recurring_orientations_are_retained(self):
        cache = {}
        self._evaluate(cache)
        # Twelve distinct orientations all fit the memory bound at this size.
        self.assertEqual(len(cache), 12)

    def test_cache_is_still_bounded_by_memory(self):
        wide, _, _ = feature_sum.directions_from_aspect_roll(
            np.linspace(5.0, 175.0, 120), np.linspace(0.0, 350.0, 120)
        )
        entry_bytes = len(wide) * 48
        cache = {}
        rng = np.random.default_rng(21)
        for _ in range(6):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            feature_sum.point_scatterer_amplitude(
                self.pattern, np.zeros(3),
                np.asarray([np.cos(angle), np.sin(angle), 0.0]), wide, 10.0,
                _interpolator_cache={}, _oriented_pattern_cache=cache,
            )
        self.assertLessEqual(len(cache) * entry_bytes, 32 * 1024 ** 2)


if __name__ == "__main__":
    unittest.main()
