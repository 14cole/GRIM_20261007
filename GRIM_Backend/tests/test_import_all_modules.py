"""Every GRIM_Backend module imports from this checkout.

The suite otherwise exercises only the modules its tests happen to import, so a
broken import in an untested module would go unnoticed until a user hit it.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import pkgutil
import unittest

import GRIM_Backend


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ImportEveryModuleTests(unittest.TestCase):
    def test_every_module_imports_from_the_checkout(self) -> None:
        # Test modules are imported by test collection already, and the examples
        # package holds runnable scripts that do their work at module level.
        # python_env and "offline packages" are not packages, so the walk never
        # enters them.
        skipped_prefixes = ("GRIM_Backend.tests", "GRIM_Backend.examples")
        walk_failures: list[str] = []
        names = [
            info.name
            for info in pkgutil.walk_packages(GRIM_Backend.__path__, "GRIM_Backend.", onerror=walk_failures.append)
            if not info.name.startswith(skipped_prefixes)
        ]
        self.assertGreater(len(names), 100, names)
        for name in names:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(PACKAGE_ROOT), module.__file__)
        # A package that fails to import is skipped by the walk, so its
        # submodules would otherwise be missing from the sweep silently.
        self.assertEqual(walk_failures, [])


if __name__ == "__main__":
    unittest.main()
