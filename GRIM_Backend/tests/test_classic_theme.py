"""Windows Classic palette: opt-in overrides and glyph files.

Delete this file together with ``GRIM_Backend/ui/classic_theme.py`` when the
Windows Classic palette is removed.
"""

from __future__ import annotations

import os
import re
import unittest

from GRIM_Backend.ui.classic_theme import classic_qss_overrides, is_classic_palette
from GRIM_Backend.ui.palette import APPLICATION_PALETTES
from GRIM_Backend.ui.theme import build_qss


class ClassicThemeTests(unittest.TestCase):
    def test_only_windows_classic_opts_in(self) -> None:
        for name, palette in APPLICATION_PALETTES.items():
            with self.subTest(palette=name):
                self.assertEqual(is_classic_palette(palette), name == "Windows Classic")
                self.assertEqual(
                    "Windows Classic overrides" in build_qss(palette),
                    name == "Windows Classic",
                )

    def test_classic_qss_squares_corners_and_bevels(self) -> None:
        qss = classic_qss_overrides(APPLICATION_PALETTES["Windows Classic"])
        self.assertIn("border-radius: 0px", qss)
        self.assertIn("#ffffff #404040 #404040 #ffffff", qss)  # raised bevel
        self.assertIn("#808080 #ffffff #ffffff #808080", qss)  # sunken bevel
        self.assertNotIn("data:image", qss)  # Qt ignores data URIs in url()

    def test_glyph_files_exist(self) -> None:
        qss = classic_qss_overrides(APPLICATION_PALETTES["Windows Classic"])
        paths = sorted(set(re.findall(r'url\("([^"]+)"\)', qss)))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(path))
                self.assertTrue(path.endswith(".svg"))


if __name__ == "__main__":
    unittest.main()
