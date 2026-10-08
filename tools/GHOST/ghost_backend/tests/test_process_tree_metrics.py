"""Worker memory accounting remains explicit, bounded and optional."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.execution.metrics import SolveMetrics, progress_listener


def process(pid, rss, private=None):
    value = mock.Mock()
    value.pid = pid
    info = dict(rss=rss)
    if private is not None:
        info['private'] = private
    value.memory_info.return_value = SimpleNamespace(**info)
    value.children.return_value = []
    return value


class ProcessTreeMetricsTests(unittest.TestCase):
    def test_windows_rss_and_private_commit_are_separate_and_children_count_once(self):
        metric = SolveMetrics()
        parent, child = process(1, 100, 200), process(2, 50, 80)
        parent.children.return_value = [child, child]
        metric._process = parent
        metric.sample_memory(force_tree=True)
        self.assertEqual(metric.current_rss, 100)
        self.assertEqual(metric.current_tree_rss, 150)
        self.assertEqual(metric.current_tree_private, 280)
        self.assertEqual(metric.tree_process_count, 2)
        parent.children.return_value = []
        metric.sample_memory(force_tree=True)
        self.assertEqual(metric.current_tree_rss, 100)
        self.assertEqual(metric.peak_tree_rss, 150)
        metric.elapsed = 1.
        report = metric.report()
        self.assertIn('shared pages', report['process_tree_memory_semantics'])
        self.assertEqual(report['sampled_peak_process_tree_private_bytes'], 280)

    def test_linux_rss_is_available_without_inventing_private_commit(self):
        metric = SolveMetrics()
        metric._process = process(1, 100)
        metric._process.children.return_value = [process(2, 30)]
        metric.sample_memory(force_tree=True)
        self.assertEqual(metric.current_tree_rss, 130)
        self.assertIsNone(metric.current_tree_private)

    def test_access_denied_snapshot_is_not_reported_as_complete_or_zero(self):
        metric = SolveMetrics()
        metric._process = process(1, 100)
        metric.sample_memory(force_tree=True)
        child = process(2, 30)
        child.memory_info.side_effect = OSError('process exited or denied')
        metric._process.children.return_value = [child]
        metric.sample_memory(force_tree=True)
        self.assertIsNone(metric.current_tree_rss)
        self.assertEqual(metric.peak_tree_rss, 100)
        self.assertEqual(metric.current_rss, 100)
        self.assertEqual(metric.tree_incomplete_samples, 1)

    def test_tree_enumeration_is_throttled_but_parent_samples_continue(self):
        metric = SolveMetrics()
        metric._process = process(1, 100)
        with mock.patch('ghost_backend.execution.metrics.time.monotonic', side_effect=[1., 1.05, 1.1, 1.3]):
            for _ in range(4):
                metric.sample_memory()
        self.assertEqual(metric._process.memory_info.call_count, 4)
        self.assertEqual(metric._process.children.call_count, 2)

    def test_absent_psutil_and_listener_errors_do_not_break_solve(self):
        events = []
        with progress_listener(events.append):
            metric = SolveMetrics()
        metric._process = None
        metric.start()
        metric.finish()
        self.assertIsNone(metric.report()['sampled_peak_process_tree_rss_bytes'])
        self.assertIsNone(events[0]['process_tree_rss_bytes'])


if __name__ == '__main__':
    unittest.main()
