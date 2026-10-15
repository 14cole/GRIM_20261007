from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class PortableLauncherTests(unittest.TestCase):
    def _text(self, relative_path: str) -> str:
        return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")

    def _assert_windows_priority(self, relative_path: str) -> None:
        text = self._text(relative_path)
        repository_environment = text.find(".venv\\Scripts\\python.exe")
        active_environment = text.find("VIRTUAL_ENV")
        system_launcher = text.find("where py.exe")
        self.assertGreaterEqual(repository_environment, 0, relative_path)
        self.assertGreater(active_environment, repository_environment, relative_path)
        self.assertGreater(system_launcher, active_environment, relative_path)

    def test_windows_launchers_share_repository_environment_priority(self) -> None:
        # GRIM's own launchers (Setup_GRIM.bat, Launch_GRIM.bat and the offline
        # pair) run the embedded python_env interpreter instead and are covered
        # by test_online_setup.py and test_offline_setup.py.
        for relative_path in (
            "tools/FREDDY/Launch_FREDDY_GUI.bat",
            "tools/GHOST/Launch_GHOST_GUI.bat",
        ):
            with self.subTest(launcher=relative_path):
                self._assert_windows_priority(relative_path)

    def test_standalone_windows_launchers_walk_to_repository_root(self) -> None:
        for relative_path in (
            "tools/FREDDY/Launch_FREDDY_GUI.bat",
            "tools/GHOST/Launch_GHOST_GUI.bat",
        ):
            text = self._text(relative_path)
            self.assertIn('%~dp0..\\..', text)
            self.assertIn('%GRIM_REPO_ROOT%\\.venv', text)

if __name__ == "__main__":
    unittest.main()
