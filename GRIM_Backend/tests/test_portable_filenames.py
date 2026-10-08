"""Saved and exported dataset filenames stay readable by external programs."""

from __future__ import annotations

import unittest

from GRIM_Backend.ui.dataset_actions import _sanitize_filename


class PortableFilenameTest(unittest.TestCase):
    def test_operation_symbols_are_spelled_out_in_ascii(self) -> None:
        # Names produced by earlier GRIM versions (or typed by users) still
        # save under the same ASCII spelling the operations now use.
        cases = {
            "DUT [Wedge→Conic normal conic]": "DUT [Wedge-to-Conic normal conic]",
            "Scan [SENTRi El→GRIM]": "Scan [SENTRi El-to-GRIM]",
            "Scan [El->Az360]": "Scan [El-to-Az360]",
            "Body [→ dBke L=10 in]": "Body [to dBke L=10 in]",
            "Scan [Mirror 90°]": "Scan [Mirror 90deg]",
            "Scan [Wrap azimuth -180–180°]": "Scan [Wrap azimuth -180-180deg]",
            "A ⊕ B [Merge priority-first]": "A + B [Merge priority-first]",
            "A ÷ B": "A div B",
            "A Δ B": "A Delta B",
            "DUT [Range Cal: Exact; ΔR +0 m]": "DUT [Range Cal_ Exact; DeltaR +0 m]",
            "Join[A | B]": "Join[A _ B]",
            "Scan [Swap El/Az]": "Scan [Swap El_Az]",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(_sanitize_filename(name), expected)

    def test_accents_and_compatibility_forms_fold_to_ascii(self) -> None:
        self.assertEqual(_sanitize_filename("Café Größe µm²"), "Cafe Grosse um2")
        self.assertEqual(_sanitize_filename("ｆｕｌｌ ｗｉｄｔｈ"), "full width")

    def test_result_is_always_printable_ascii(self) -> None:
        for name in ("日本語", "tab\there", "line\nbreak", "…", "∞ ≠ ∅"):
            with self.subTest(name=name):
                cleaned = _sanitize_filename(name)
                self.assertTrue(cleaned)
                self.assertTrue(cleaned.isascii() and cleaned.isprintable(), cleaned)

    def test_plain_ascii_names_are_unchanged(self) -> None:
        for name in ("DUT [Crop]", "Scan [Offset +3] (v2)", "a_b-c.d"):
            with self.subTest(name=name):
                self.assertEqual(_sanitize_filename(name), name)

    def test_windows_reserved_and_empty_names_are_made_usable(self) -> None:
        self.assertEqual(_sanitize_filename("aux"), "aux_")
        self.assertEqual(_sanitize_filename("COM1"), "COM1_")
        self.assertEqual(_sanitize_filename("nul.backup"), "nul_.backup")
        self.assertEqual(_sanitize_filename("Console"), "Console")
        self.assertEqual(_sanitize_filename(" name. "), "name")
        self.assertEqual(_sanitize_filename(" . "), "dataset")
        self.assertEqual(_sanitize_filename(None), "dataset")


if __name__ == "__main__":
    unittest.main()
