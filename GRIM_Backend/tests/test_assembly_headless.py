"""The Assembly recipe path must run, and gate, without Qt.

These check two separate things: that the modules a headless build needs import
with PySide6 unavailable, and that the runner keeps the operator review gate the
Assembly tab enforces rather than publishing past it.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
GHOST_BACKEND = REPO / "tools" / "GHOST"
for entry in (str(REPO), str(GHOST_BACKEND)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


class _BlockPySide6:
    """Import hook that hides PySide6 from a subsequent import."""

    def find_module(self, name, path=None):
        if name == "PySide6" or name.startswith("PySide6."):
            return self

    def load_module(self, name):
        raise ImportError(f"PySide6 blocked for test ({name})")


HEADLESS_REQUIRED = (
    "GRIM_Backend.assembly.values",
    "GRIM_Backend.assembly.recipe",
    "GRIM_Backend.assembly.model",
    "GRIM_Backend.assembly.headless",
    "GRIM_Backend.integrations.ghost",
)


class HeadlessImportTests(unittest.TestCase):
    """Every module on the recipe-to-platform path imports without Qt."""

    def test_required_modules_import_without_pyside6(self):
        for name in HEADLESS_REQUIRED:
            with self.subTest(module=name):
                script = (
                    "import sys\n"
                    "class B:\n"
                    " def find_module(s,n,p=None):\n"
                    "  return s if n=='PySide6' or n.startswith('PySide6.') else None\n"
                    " def load_module(s,n):\n"
                    "  raise ImportError('blocked')\n"
                    "sys.meta_path.insert(0,B())\n"
                    f"import {name}\n"
                    "assert 'PySide6' not in sys.modules, 'PySide6 was imported'\n"
                    "print('ok')\n"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=str(REPO), capture_output=True, text=True,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"{name} failed to import headless:\n{result.stderr}",
                )

    def test_widget_stub_is_actionable_without_qt(self):
        script = (
            "import sys\n"
            "class B:\n"
            " def find_module(s,n,p=None):\n"
            "  return s if n=='PySide6' or n.startswith('PySide6.') else None\n"
            " def load_module(s,n):\n"
            "  raise ImportError('blocked')\n"
            "sys.meta_path.insert(0,B())\n"
            "from GRIM_Backend.integrations.ghost import GUI_AVAILABLE, GhostIntegrationWidget\n"
            "assert GUI_AVAILABLE is False\n"
            "try:\n"
            "    GhostIntegrationWidget()\n"
            "except RuntimeError as exc:\n"
            "    assert 'requires PySide6' in str(exc), exc\n"
            "else:\n"
            "    raise AssertionError('stub did not raise')\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(REPO),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_discovery_works_without_qt(self):
        from GRIM_Backend.integrations.ghost import load_ghost_module

        service = load_ghost_module("feature_workflow", None)
        self.assertTrue(hasattr(service, "__name__"))


def _recipe_with_base(directory: Path):
    """A minimal but genuine recipe: a real base .grim and an output target."""
    from ghost_backend.assembly.fields import export_radar_grim
    from GRIM_Backend.assembly.recipe import write_feature_assembly_recipe
    from GRIM_Backend.assembly.values import FeatureAssemblyValues

    base = directory / "body.grim"
    export_radar_grim(
        str(base), bor_result=None, placements=[], frequencies_ghz=[1.0],
        azimuths_deg=[0.0, 45.0], elevations_deg=[0.0],
        axis_az_deg=0.0, axis_el_deg=0.0, roll_deg=0.0,
    )
    values = FeatureAssemblyValues(
        base_grim=str(base), output_grim=str(directory / "platform.grim")
    )
    return write_feature_assembly_recipe(
        values, directory / "platform.json", name="Platform A", variant="baseline"
    )


class HeadlessRunTests(unittest.TestCase):
    """A recipe round-trips to a published platform, with the gate intact."""

    def test_prepare_then_publish_produces_the_declared_output(self):
        from GRIM_Backend.assembly import headless

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = _recipe_with_base(root)
            prepared = headless.prepare_recipe(recipe)
            self.assertEqual(prepared.name, "Platform A")
            self.assertEqual(prepared.variant, "baseline")
            self.assertTrue(prepared.plan_sha256)
            self.assertFalse(prepared.review_required)
            # prepare_recipe must not publish anything on its own.
            self.assertFalse((root / "platform.grim").exists())

            dispatch = headless.run_recipe(recipe)
            self.assertTrue(Path(dispatch.output_path).exists())
            self.assertEqual(
                Path(dispatch.output_path).resolve(),
                (root / "platform.grim").resolve(),
            )

    def test_warned_plan_is_refused_until_acknowledged(self):
        from GRIM_Backend.assembly import headless

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = _recipe_with_base(root)
            real = headless.prepare_recipe(recipe)
            warned = SimpleNamespace(
                recipe_path=real.recipe_path, name=real.name, variant=real.variant,
                model=real.model, service=real.service, plan=real.plan,
                source_warnings=(), validation_warnings=("workload is large",),
                review_required=True,
                plan_sha256=real.plan_sha256, workload_text="",
            )
            with mock.patch.object(headless, "prepare_recipe", return_value=warned):
                with self.assertRaisesRegex(RuntimeError, "acknowledge_warnings"):
                    headless.run_recipe(recipe)
                self.assertFalse((root / "platform.grim").exists())

    def test_acknowledgement_is_bound_to_the_reviewed_plan(self):
        from GRIM_Backend.assembly import headless

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = _recipe_with_base(root)
            real = headless.prepare_recipe(recipe)
            seen = {}

            def capture(service, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(output_path="x", features_only_output_path="y",
                                       features_only_output_published=False,
                                       reused_validated_plan=True)

            warned = SimpleNamespace(
                recipe_path=real.recipe_path, name=real.name, variant=real.variant,
                model=SimpleNamespace(assemble_validated=capture, values=real.model.values),
                service=real.service, plan=real.plan, source_warnings=(),
                validation_warnings=("workload is large",), review_required=True,
                plan_sha256=real.plan_sha256, workload_text="",
            )
            with mock.patch.object(headless, "prepare_recipe", return_value=warned):
                headless.run_recipe(recipe, acknowledge_warnings=True)
            self.assertEqual(seen.get("acknowledged_plan_sha256"), real.plan_sha256)

    def test_unwarned_plan_sends_no_acknowledgement(self):
        from GRIM_Backend.assembly import headless

        with tempfile.TemporaryDirectory() as directory:
            recipe = _recipe_with_base(Path(directory))
            real = headless.prepare_recipe(recipe)
            seen = {}

            def capture(service, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(output_path="x", features_only_output_path="y",
                                       features_only_output_published=False,
                                       reused_validated_plan=True)

            real.model.assemble_validated = capture
            with mock.patch.object(headless, "prepare_recipe", return_value=real):
                headless.run_recipe(recipe, acknowledge_warnings=True)
            self.assertIsNone(seen.get("acknowledged_plan_sha256"))


class HeadlessCommandLineTests(unittest.TestCase):
    """The CLI reports, publishes, and returns distinguishable exit codes."""

    def _run(self, *argv, directory):
        return subprocess.run(
            [sys.executable, "-m", "GRIM_Backend.assembly.headless", *argv],
            cwd=str(REPO), capture_output=True, text=True,
            env={**_environment(), "PYTHONPATH": f"{REPO}{';' if sys.platform=='win32' else ':'}{GHOST_BACKEND}"},
        )

    def test_validate_reports_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = _recipe_with_base(root)
            result = self._run("--quiet", "--json", "validate", str(recipe),
                               directory=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["name"], "Platform A")
            self.assertFalse(payload["review_required"])
            self.assertFalse((root / "platform.grim").exists())

    def test_run_publishes_and_reports_the_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = _recipe_with_base(root)
            result = self._run("--quiet", "--json", "run", str(recipe), directory=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(Path(payload["output_path"]).exists())

    def test_missing_recipe_exits_with_an_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run("--quiet", "validate", str(root / "absent.json"),
                               directory=root)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            self.assertTrue(result.stderr.strip())


def _environment():
    import os
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


if __name__ == "__main__":
    unittest.main()
