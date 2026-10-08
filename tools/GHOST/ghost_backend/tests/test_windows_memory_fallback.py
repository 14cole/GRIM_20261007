"""RAM admission still sees resident memory without optional psutil."""
import os
import unittest
from unittest import mock
from ghost_backend.execution.memory import windows_process_memory
from ghost_backend.twod.solver import _process_rss_bytes


@unittest.skipUnless(os.name == 'nt', 'Windows native memory fallback')
class WindowsMemoryFallbackTests(unittest.TestCase):
    def test_native_counters_are_available_without_psutil(self):
        with mock.patch.dict('sys.modules', {'psutil':None}):
            self.assertGreater(_process_rss_bytes(), 0)
            info=windows_process_memory()
            self.assertGreater(info.rss, 0)
            self.assertGreaterEqual(info.peak_wset, info.rss)
            self.assertGreater(info.private, 0)
