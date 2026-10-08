#!/usr/bin/env python3
"""Focused regressions for backend trust, frame, and unit invariants."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))

import ghost_backend.assembly.fields as feature_sum
import ghost_backend.bor.streaming as bor_streaming
import ghost_backend.bor.native.build_kernel as build_bor_stream_kernel  # noqa: E402
import ghost_backend.twod.solver as rcs_solver


class NativeLoaderTrustTests(unittest.TestCase):
    def test_windows_build_exposes_compiler_runtime_to_helper_processes(self) -> None:
        compiler = REPO / "fake-msys2" / "ucrt64" / "bin" / "gcc.exe"
        original_path = str(REPO / "unrelated-tools")
        with mock.patch.dict(os.environ, {"PATH": original_path}):
            environment = build_bor_stream_kernel._compiler_environment(
                str(compiler), "windows"
            )

        path_entries = environment["PATH"].split(os.pathsep)
        self.assertEqual(path_entries[0], str(compiler.resolve().parent))
        self.assertEqual(path_entries[1:], [original_path])

    def test_bor_native_extensions_are_host_specific(self) -> None:
        self.assertEqual(bor_streaming._native_extensions("Windows"), (".dll",))
        self.assertEqual(bor_streaming._native_extensions("Linux"), (".so",))

    def test_windows_loader_ignores_checked_in_linux_shared_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bor_stream_kernel.so").write_bytes(b"foreign")
            with (
                mock.patch.object(bor_streaming, "native_kernel_root", return_value=root),
                mock.patch.object(platform, "system", return_value="Windows"),
                mock.patch.object(platform, "machine", return_value="AMD64"),
                mock.patch.object(
                    ctypes,
                    "CDLL",
                    side_effect=AssertionError("foreign .so must not reach ctypes"),
                ) as loader,
            ):
                self.assertIsNone(bor_streaming._load_native())
        loader.assert_not_called()

    def test_native_candidates_never_come_from_working_directory(self) -> None:
        loaded = []
        fake_library = SimpleNamespace(
            sample_g=mock.Mock(), sample_mfie=mock.Mock(), sample_ibc=mock.Mock()
        )
        with tempfile.TemporaryDirectory() as directory:
            trusted = Path(directory) / "backend"
            untrusted = Path(directory) / "cwd"
            trusted.mkdir()
            untrusted.mkdir()
            (trusted / "bor_stream_kernel.so").write_bytes(b"trusted")
            for name in (
                "bor_stream_kernel.so",
                "bor_stream_kernel.dll",
            ):
                (untrusted / name).write_bytes(b"untrusted")

            previous = Path.cwd()
            os.chdir(untrusted)
            try:
                with (
                    mock.patch.object(bor_streaming, "native_kernel_root", return_value=trusted),
                    mock.patch.object(platform, "system", return_value="Linux"),
                    mock.patch.object(platform, "machine", return_value="x86_64"),
                    mock.patch.object(
                        ctypes,
                        "CDLL",
                        side_effect=lambda path: (
                            loaded.append(Path(path).resolve()) or fake_library
                        ),
                    ),
                ):
                    library = bor_streaming._load_native()
            finally:
                os.chdir(previous)

        self.assertIs(library, fake_library)
        self.assertTrue(loaded)
        self.assertTrue(
            all(candidate.parent == trusted.resolve() for candidate in loaded),
            loaded,
        )


class BodyFrameExportTests(unittest.TestCase):
    class _ReachedPostAxisValidation(RuntimeError):
        pass

    def _validate_axis_without_export(self, axis) -> None:
        with mock.patch.object(
            feature_sum,
            "surface_of_revolution_normal",
            side_effect=self._ReachedPostAxisValidation,
        ):
            with self.assertRaises(self._ReachedPostAxisValidation):
                feature_sum.export_signature_grim(
                    "unused.grim",
                    bor_result=None,
                    placements=[],
                    generatrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                    frequencies_ghz=[1.0],
                    aspects_deg=[0.0],
                    axis=axis,
                )

    def test_legacy_export_does_not_mutate_float_axis_array(self) -> None:
        axis = np.asarray([0.0, 0.0, 2.0], dtype=float)
        original = axis.copy()
        self._validate_axis_without_export(axis)
        np.testing.assert_array_equal(axis, original)

    def test_legacy_export_accepts_read_only_canonical_axis(self) -> None:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        axis.setflags(write=False)
        self._validate_axis_without_export(axis)
        self.assertFalse(axis.flags.writeable)

    def test_legacy_body_frame_export_rejects_non_z_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "signature.grim"
            with self.assertRaisesRegex(ValueError, r"canonical \+z BoR axis"):
                feature_sum.export_signature_grim(
                    str(destination),
                    bor_result=None,
                    placements=[],
                    generatrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                    frequencies_ghz=[1.0],
                    aspects_deg=[0.0],
                    axis=(1.0, 0.0, 0.0),
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


def _open_chain_snapshot(points_m, meters_scale):
    pairs = []
    for start, end in points_m:
        pairs.append(
            {
                "x1": start[0] / meters_scale,
                "y1": start[1] / meters_scale,
                "x2": end[0] / meters_scale,
                "y2": end[1] / meters_scale,
            }
        )
    return {
        "segments": [
            {
                "name": "sheet",
                "seg_type": 1,
                "properties": ["1", "0", "0", "0", "0"],
                "point_pairs": pairs,
            }
        ],
        "ibcs": [],
        "dielectrics": [],
    }


def _sheet_pair_snapshot(chain_a_m, chain_b_m, meters_scale):
    """Two independent TYPE 1 sheets, for clearance/intersection checks."""

    def sheet(name, points_m):
        return {
            "name": name,
            "seg_type": 1,
            "properties": ["1", "0", "0", "0", "0"],
            "point_pairs": [
                {
                    "x1": start[0] / meters_scale,
                    "y1": start[1] / meters_scale,
                    "x2": end[0] / meters_scale,
                    "y2": end[1] / meters_scale,
                }
                for start, end in points_m
            ],
        }

    return {
        "segments": [sheet("upper", chain_a_m), sheet("lower", chain_b_m)],
        "ibcs": [],
        "dielectrics": [],
    }


class GeometryPreflightUnitTests(unittest.TestCase):
    def test_equivalent_meter_and_inch_geometry_returns_same_report(self) -> None:
        physical = [
            ((0.0, 0.0), (0.01, 0.0)),
            ((0.01, 0.0), (0.02, 0.0)),
        ]
        reports = []
        for scale in (1.0, 0.0254):
            reports.append(
                rcs_solver.validate_geometry_snapshot_for_solver(
                    _open_chain_snapshot(physical, scale),
                    base_dir=".",
                    meters_scale=scale,
                )
            )
        self.assertEqual(reports[0], reports[1])

    def test_equivalent_submicron_crack_has_same_meter_space_diagnosis(self) -> None:
        physical = [
            ((0.0, 0.0), (0.01, 0.0)),
            ((0.0100001, 0.0), (0.02, 0.0)),
        ]
        messages = []
        for scale in (1.0, 0.0254):
            with self.assertRaisesRegex(ValueError, "Geometry crack") as raised:
                rcs_solver.validate_geometry_snapshot_for_solver(
                    _open_chain_snapshot(physical, scale),
                    base_dir=".",
                    meters_scale=scale,
                )
            messages.append(str(raised.exception).split(" -- ", 1)[1])
        self.assertEqual(messages[0], messages[1])


class IntersectionToleranceTests(unittest.TestCase):
    """`_segment_intersects_strict` must treat `tol` as a perpendicular distance.

    The predicate compares cross products, which carry units of length squared.
    Testing them against `tol` directly made the effective clearance tolerance
    ``tol / primitive_length``: coarse for long primitives and unbounded for
    short ones, so thin real features were rejected as intersections.
    """

    TOL = 1.0e-6

    def _intersects(self, a1, a2, b1, b2):
        return rcs_solver._segment_intersects_strict(a1, a2, b1, b2, self.TOL)

    def test_clearance_threshold_does_not_depend_on_primitive_length(self) -> None:
        for length in (1.0e-5, 1.0e-3, 1.0e-1):
            with self.subTest(length=length):
                # 5 um of real clearance -- five times the tolerance -- between
                # two boundaries leaving a shared region at slightly different
                # angles.  This is the layout that rejected 0.002 in features.
                gap = 5.0e-6
                self.assertIs(
                    self._intersects(
                        (0.0, gap), (length, 4.0 * gap),
                        (0.0, 0.0), (length, 2.0 * gap),
                    ),
                    False,
                )
                # 0.5 um of clearance is inside the tolerance and must still be
                # reported, whatever the primitive length.
                self.assertIs(
                    self._intersects(
                        (0.0, 0.0), (length, 0.0),
                        (-0.37 * length, 0.5e-6), (0.61 * length, 0.5e-6),
                    ),
                    True,
                )

    def test_thin_parallel_boundaries_clear_the_preflight(self) -> None:
        # A 6.35 mm dielectric primitive running 50.8 um (0.002 in) above a
        # parallel PEC primitive: a real thin feature, not an intersection.
        upper = [((0.0, 50.8e-6), (6.35e-3, 563.0e-6))]
        lower = [((0.0, 0.0), (6.35e-3, 406.0e-6))]
        for scale in (1.0, 0.0254):
            with self.subTest(meters_scale=scale):
                report = rcs_solver.validate_geometry_snapshot_for_solver(
                    _sheet_pair_snapshot(upper, lower, scale),
                    base_dir=".",
                    meters_scale=scale,
                )
                self.assertEqual(report["primitive_count"], 2)

    def test_short_primitive_does_not_inflate_the_tolerance(self) -> None:
        # A 50.9 um primitive whose neighbour ends 14 um away, inside its
        # bounding box.  Dividing `tol` by that length turned 14 um of real
        # clearance into an apparent overlap.
        short = [((0.0, 0.0), (36.0e-6, 36.0e-6))]
        neighbour = [((100.0e-6, -30.0e-6), (30.0e-6, 10.0e-6))]
        report = rcs_solver.validate_geometry_snapshot_for_solver(
            _sheet_pair_snapshot(short, neighbour, 1.0),
            base_dir=".",
            meters_scale=1.0,
        )
        self.assertEqual(report["primitive_count"], 2)

    def test_genuine_crossings_are_still_rejected(self) -> None:
        crossing = [((0.0, -5.0e-3), (0.0, 5.0e-3))]
        run = [((-5.0e-3, 0.0), (5.0e-3, 0.0))]
        with self.assertRaisesRegex(ValueError, "unsupported segment intersection"):
            rcs_solver.validate_geometry_snapshot_for_solver(
                _sheet_pair_snapshot(crossing, run, 1.0),
                base_dir=".",
                meters_scale=1.0,
            )

    def test_collinear_overlap_and_touching_tips_are_still_rejected(self) -> None:
        self.assertIs(
            self._intersects((0.0, 0.0), (1.0e-3, 0.0), (0.5e-3, 0.0), (1.5e-3, 0.0)),
            True,
        )
        # T-junction: the tip of one primitive lands on the interior of another.
        self.assertIs(
            self._intersects((0.0, 0.0), (1.0e-3, 0.0), (0.5e-3, 0.0), (0.5e-3, 1.0e-3)),
            True,
        )

    def test_gui_audit_uses_the_same_clearance_rule(self) -> None:
        # The editor audit carries its own copy of the predicate, in file units
        # rather than meters.  It has to agree with the solver preflight, or a
        # geometry passes in the editor and fails at solve time.
        from ghost_backend.geometry.validation import GeometryAudit

        audit = GeometryAudit([], [], [], ".")
        gap = 5.0e-6
        for length in (1.0e-5, 1.0e-3, 1.0e-1):
            with self.subTest(length=length):
                self.assertIs(
                    audit._segments_intersect(
                        (0.0, gap), (length, 4.0 * gap),
                        (0.0, 0.0), (length, 2.0 * gap),
                        self.TOL,
                    ),
                    False,
                )
                self.assertIs(
                    audit._segments_intersect(
                        (0.0, 0.0), (length, 0.0),
                        (-0.37 * length, 0.5e-6), (0.61 * length, 0.5e-6),
                        self.TOL,
                    ),
                    True,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
