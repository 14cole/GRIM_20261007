"""Driver/scheduler audit fixes (September 2026).

- Host memory is detected on Windows (psutil, then GlobalMemoryStatusEx)
  instead of the 8 GB fallback, and the scheduler's unit is GiB throughout.
- A unit whose worker process dies no longer kills the sweep: the pool is
  rebuilt, innocent siblings are retried alone, and the crash is recorded.
- Concurrent BoR units reserve the scratch disk their far-block spill needs.
- BoR drivers sweep stale spill directories at startup, when the backend
  offers the sweep.
- Each BoR unit runs inside its own CPU allocation (and nothing else).
"""
import ctypes
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND.parent), str(BACKEND)]

import ghost_backend.hpc.scheduler as hpc_scheduler  # noqa: E402

GIB = 1024 ** 3
_SCHEDULER_ENV = ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
                  "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")


class _FakeVirtualMemory:
    def __init__(self, total, available):
        self.total = total
        self.available = available


class _FakePsutil:
    """The two psutil calls the scheduler makes."""

    def __init__(self, total=None, available=None, physical=None, logical=None):
        self._memory = _FakeVirtualMemory(total, available)
        self._physical, self._logical = physical, logical

    def virtual_memory(self):
        if self._memory.total is None:
            raise RuntimeError("no memory figure")
        return self._memory

    def cpu_count(self, logical=True):
        return self._logical if logical else self._physical


class _NoLinuxProbe:
    """Path stand-in: cgroup and /proc/meminfo are unreadable (Windows/macOS)."""

    def __init__(self, *parts):
        self._path = Path(*parts)

    def read_text(self, *args, **kwargs):
        raise OSError("not on this host")


def _without_scheduler_env():
    return mock.patch.dict(os.environ, {name: "" for name in _SCHEDULER_ENV})


class MemoryDetectionTests(unittest.TestCase):
    """detect_memory_gb on a host without SLURM, cgroup or /proc/meminfo."""

    def test_psutil_total_replaces_the_fixed_fallback(self):
        fake = _FakePsutil(total=int(33.4e9), available=int(20e9))
        with _without_scheduler_env(), \
                mock.patch.object(hpc_scheduler, "Path", _NoLinuxProbe), \
                mock.patch.dict(sys.modules, {"psutil": fake}):
            detected = hpc_scheduler.detect_memory_gb()
        self.assertAlmostEqual(detected, 33.4e9 / GIB, places=9)
        self.assertNotEqual(detected, hpc_scheduler.FALLBACK_MEMORY_GIB)

    def test_windows_probe_without_psutil(self):
        class _Kernel32:
            @staticmethod
            def GlobalMemoryStatusEx(reference):
                status = reference._obj
                status.ullTotalPhys = 32 * GIB
                status.ullAvailPhys = 12 * GIB
                return 1

        windll = type("WinDLL", (), {"kernel32": _Kernel32()})()
        with _without_scheduler_env(), \
                mock.patch.object(hpc_scheduler, "Path", _NoLinuxProbe), \
                mock.patch.dict(sys.modules, {"psutil": None}), \
                mock.patch.object(hpc_scheduler.sys, "platform", "win32"), \
                mock.patch.object(ctypes, "windll", windll, create=True):
            self.assertEqual(hpc_scheduler.detect_memory_gb(), 32.0)
            self.assertEqual(hpc_scheduler.detect_available_memory_gb(), 12.0)

    @unittest.skipUnless(sys.platform.startswith("win"), "native Windows probe")
    def test_native_windows_probe_matches_installed_memory(self):
        status = hpc_scheduler._windows_memory_status()
        self.assertIsNotNone(status)
        with _without_scheduler_env(), \
                mock.patch.dict(sys.modules, {"psutil": None}):
            detected = hpc_scheduler.detect_memory_gb()
        self.assertAlmostEqual(detected, status[0] / GIB, places=6)
        self.assertGreater(detected, 1.0)

    def test_fallback_only_when_every_probe_fails(self):
        with _without_scheduler_env(), \
                mock.patch.object(hpc_scheduler, "Path", _NoLinuxProbe), \
                mock.patch.object(hpc_scheduler, "_psutil_total_bytes", lambda: None), \
                mock.patch.object(hpc_scheduler, "_windows_total_bytes", lambda: None), \
                mock.patch.object(hpc_scheduler, "_sysconf_total_bytes", lambda: None):
            self.assertEqual(hpc_scheduler.detect_memory_gb(),
                             hpc_scheduler.FALLBACK_MEMORY_GIB)

    def test_slurm_allocation_stays_authoritative(self):
        fake = _FakePsutil(total=1024 * GIB, available=900 * GIB)
        with mock.patch.dict(os.environ, {"SLURM_MEM_PER_NODE": str(64 * 1024)}), \
                mock.patch.dict(sys.modules, {"psutil": fake}):
            self.assertEqual(hpc_scheduler.detect_memory_gb(), 64.0)

    def test_workstation_budget_is_capped_by_available_memory(self):
        fake = _FakePsutil(total=32 * GIB, available=10 * GIB)
        with _without_scheduler_env(), \
                mock.patch.object(hpc_scheduler, "Path", _NoLinuxProbe), \
                mock.patch.dict(sys.modules, {"psutil": fake}):
            budget, installed, available = hpc_scheduler.local_memory_budget_gb(0.75)
        self.assertEqual((installed, available), (32.0, 10.0))
        self.assertAlmostEqual(budget, 9.0)
        roomy = _FakePsutil(total=32 * GIB, available=31 * GIB)
        with _without_scheduler_env(), \
                mock.patch.object(hpc_scheduler, "Path", _NoLinuxProbe), \
                mock.patch.dict(sys.modules, {"psutil": roomy}):
            budget, _installed, _available = hpc_scheduler.local_memory_budget_gb(0.75)
        self.assertAlmostEqual(budget, 24.0)

    def test_scheduler_unit_is_gib(self):
        self.assertEqual(hpc_scheduler.BYTES_PER_GIB, GIB)
        self.assertAlmostEqual(hpc_scheduler.decimal_gb_to_gib(1.0), 1.0e9 / GIB)
        # A BoR estimate of exactly the installed bytes is exactly the budget.
        self.assertAlmostEqual(hpc_scheduler.decimal_gb_to_gib(32 * GIB / 1e9), 32.0)

    def test_local_bor_driver_sizes_units_on_logical_cpus(self):
        # A/B (Sept 2026): 4 x 4 threads on 16 logical CPUs beat 2 x 4 on the
        # 8 physical cores by ~1.6x, so the local driver keeps logical sizing.
        from ghost_backend import run_local_bor
        source = inspect.getsource(run_local_bor.main)
        self.assertIn("cores = hpc_scheduler.detect_cores()", source)


# -- dispatcher fakes ---------------------------------------------------------

class _Handle:
    """Completes after a few polls; ``error`` is raised by get()."""

    def __init__(self, pool, payload, error=None, polls=2):
        self.pool, self.payload, self.error = pool, payload, error
        self._polls, self._done = polls, False

    def ready(self):
        self._polls -= 1
        if self._polls > 0:
            return False
        self._finish()
        return True

    def _finish(self):
        if not self._done:
            self._done = True
            self.pool.finish(self)

    def get(self):
        if self.error is not None:
            raise self.error
        return self.payload["name"]


class _TrackingPool:
    """Inline pool recording concurrent memory/disk and supporting rebuild().

    Like a ``ProcessPoolExecutor``, a worker that dies fails every unit then
    in flight with ``BrokenProcessPool``.
    """

    def __init__(self, broken_submits=0, crashing=()):
        self.live_disk = self.peak_disk = 0.0
        self.live = self.peak = 0
        self.order = []
        self.running = []
        self.rebuilt = 0
        self.broken_submits = int(broken_submits)
        self.crashing = set(crashing)

    def finish(self, handle):
        self.running.remove(handle)
        self.live -= 1
        self.live_disk -= handle.payload.get("disk", 0.0)
        if handle.payload["name"] in self.crashing:
            for other in list(self.running):
                other.error = BrokenProcessPool("pool broken by a sibling")
                other._polls = 0
                other._finish()

    def apply_async(self, function, args):
        if self.broken_submits:
            self.broken_submits -= 1
            raise BrokenProcessPool("pool broke before the submission")
        payload = args[0]
        self.order.append(payload["name"])
        self.live += 1
        self.live_disk += payload.get("disk", 0.0)
        self.peak = max(self.peak, self.live)
        self.peak_disk = max(self.peak_disk, self.live_disk)
        error = (BrokenProcessPool("worker died")
                 if payload["name"] in self.crashing else None)
        handle = _Handle(self, payload, error)
        self.running.append(handle)
        return handle

    def rebuild(self):
        self.rebuilt += 1
        return "worker exit code 3"


def _prepare(unit):
    return unit["name"], unit["gb"], (lambda payload: payload, (unit,))


def _resources(unit):
    return unit["gb"], unit.get("cpus", 1), unit.get("disk", 0.0)


# Stand-ins for solver units in real spawned workers.  They are stdlib
# callables, so a spawned child need not import this module.
_SLEEP = time.sleep
_CRASH = os._exit


class DispatcherCrashTests(unittest.TestCase):
    def test_crashed_worker_is_recorded_and_the_sweep_continues(self):
        """A real ExecutorPool whose worker calls os._exit mid-sweep."""
        from ghost_backend.hpc.common import ExecutorPool

        units = [
            {"name": "sibling", "call": (_SLEEP, (2.0,))},
            {"name": "crasher", "call": (_CRASH, (3,))},
            {"name": "after0", "call": (_SLEEP, (0.1,))},
            {"name": "after1", "call": (_SLEEP, (0.1,))},
        ]
        done, failed = [], {}
        with ExecutorPool(processes=2, max_tasks_per_child=2) as pool:
            dispatcher = hpc_scheduler.MemoryAwareDispatcher(
                pool, budget_gb=8.0, max_concurrent=2, cpu_budget=2,
                poll_seconds=0.02,
            )
            dispatcher.run(
                units,
                lambda unit: (unit["name"], 1.0, unit["call"]),
                lambda key, _value: done.append(key),
                lambda key, exc: failed.__setitem__(key, exc),
                resource_request=lambda _unit: (1.0, 1),
            )
        self.assertEqual(sorted(done), ["after0", "after1", "sibling"])
        self.assertEqual(list(failed), ["crasher"])
        self.assertIsInstance(failed["crasher"], hpc_scheduler.WorkerCrashError)
        self.assertIn("exit code 3", str(failed["crasher"]))
        self.assertGreaterEqual(dispatcher.pool_rebuilds, 1)
        self.assertEqual(dispatcher.pool_rebuilds, pool.rebuilds)

    def test_rebuild_does_not_hang_on_a_payload_stuck_in_the_call_queue(self):
        """A crash while a large unit payload is still being fed to the pool.

        The payload exceeds the pipe buffer, so the dead pool's feeder thread
        blocks; teardown used to join it forever (in rebuild() and at exit).
        """
        script = textwrap.dedent(f"""
            import os, sys
            sys.path[:0] = [{str(BACKEND.parent)!r}]
            if __name__ == "__main__":
                from ghost_backend.hpc.common import ExecutorPool
                from ghost_backend.hpc import scheduler
                payload = b"x" * 1_000_000
                units = [("crash", (os._exit, (3,))), ("big0", (len, (payload,))),
                         ("big1", (len, (payload,)))]
                done, failed = {{}}, {{}}
                with ExecutorPool(processes=1, max_tasks_per_child=2) as pool:
                    scheduler.MemoryAwareDispatcher(
                        pool, budget_gb=10.0, max_concurrent=3, poll_seconds=0.02,
                    ).run(units, lambda unit: (unit[0], 1.0, unit[1]),
                          lambda key, value: done.__setitem__(key, value),
                          lambda key, exc: failed.__setitem__(key, type(exc).__name__),
                          resource_request=lambda unit: (1.0, 1))
                print(sorted(done.items()), sorted(failed.items()))
        """)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feeder_case.py"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(path)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, universal_newlines=True, timeout=180,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[('big0', 1000000), ('big1', 1000000)] "
                      "[('crash', 'WorkerCrashError')]", result.stdout)

    def test_innocent_casualties_retry_alone_and_crasher_fails(self):
        pool = _TrackingPool(crashing={"bad"})
        units = [{"name": name, "gb": 1.0} for name in ("a", "bad", "b", "c")]
        done, failed = [], {}
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=10.0, max_concurrent=4, poll_seconds=0.0)
        dispatcher.run(units, _prepare, lambda key, _v: done.append(key),
                       lambda key, exc: failed.__setitem__(key, exc), _resources)
        self.assertEqual(sorted(done), ["a", "b", "c"])
        self.assertEqual(list(failed), ["bad"])
        self.assertIn("retried alone", str(failed["bad"]))
        # "a" was still in flight when "bad" broke the pool: it is retried,
        # alone, and completes; "bad" crashes again alone and is recorded.
        self.assertEqual(pool.order.count("a"), 2)
        self.assertEqual(pool.order.count("bad"), 2)
        self.assertEqual(pool.order[-2:], ["bad", "a"])
        self.assertGreaterEqual(pool.rebuilt, 2)

    def test_fixed_order_admission_recovers_too(self):
        pool = _TrackingPool(crashing={"bad"})
        units = [{"name": name, "gb": 1.0} for name in ("a", "bad", "b")]
        done, failed = [], {}
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=10.0, max_concurrent=3, poll_seconds=0.0)
        dispatcher.run(units, _prepare, lambda key, _v: done.append(key),
                       lambda key, exc: failed.__setitem__(key, exc))
        self.assertEqual(sorted(done), ["a", "b"])
        self.assertEqual(list(failed), ["bad"])

    def test_refused_submission_is_requeued_without_reclaiming(self):
        pool = _TrackingPool(broken_submits=1)
        units = [{"name": name, "gb": 1.0} for name in ("a", "b")]
        prepared, done = [], []

        def prepare(unit):
            prepared.append(unit["name"])
            return _prepare(unit)

        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=10.0, max_concurrent=2, poll_seconds=0.0)
        dispatcher.run(units, prepare, lambda key, _v: done.append(key),
                       lambda key, exc: self.fail(f"{key}: {exc!r}"), _resources)
        self.assertEqual(sorted(done), ["a", "b"])
        self.assertEqual(prepared, ["a", "b"])  # the refused unit kept its claim
        self.assertEqual(pool.rebuilt, 1)

    def test_a_pool_that_never_completes_stops_with_every_unit_reported(self):
        names = [f"u{i}" for i in range(6)]
        pool = _TrackingPool(crashing=set(names))
        failed = {}
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=10.0, max_concurrent=2, poll_seconds=0.0,
            max_pool_failures=3)
        with self.assertRaises(hpc_scheduler.WorkerPoolFailure):
            dispatcher.run([{"name": n, "gb": 1.0} for n in names], _prepare,
                           lambda key, _v: self.fail(key),
                           lambda key, exc: failed.__setitem__(key, exc),
                           _resources)
        prepared = set(pool.order)
        self.assertTrue(prepared)
        self.assertEqual(set(failed), prepared)  # no prepared unit went unreported
        self.assertLessEqual(pool.rebuilt, 3)

    def test_pool_without_rebuild_reports_and_raises(self):
        class _Plain(_TrackingPool):
            rebuild = None

        pool = _Plain(crashing={"a"})
        failed = {}
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=10.0, max_concurrent=2, poll_seconds=0.0)
        with self.assertRaises(hpc_scheduler.WorkerPoolFailure):
            dispatcher.run([{"name": "a", "gb": 1.0}], _prepare,
                           lambda key, _v: None,
                           lambda key, exc: failed.__setitem__(key, exc), _resources)
        self.assertEqual(list(failed), ["a"])


class LocalBorDriverCrashTests(unittest.TestCase):
    """End to end: a unit whose worker process dies does not stop the sweep."""

    def test_crashed_unit_is_failed_and_the_others_are_written(self):
        with tempfile.TemporaryDirectory(prefix="ghost_crash_") as folder:
            work = Path(folder)
            geometry = work / "geo"
            geometry.mkdir()
            shutil.copy2(str(BACKEND / "geometry" / "geometries" / "body.geo"),
                         str(geometry / "body.geo"))
            # Replaces the pool entry point by name, so spawned workers import
            # it from here: the 1.5 GHz unit's worker dies like a native crash.
            (work / "crash_hook.py").write_text(textwrap.dedent("""
                import os
                import run_local_bor

                def star(args):
                    if abs(float(args[0]["frequency_ghz"]) - 1.5) < 1e-9:
                        os._exit(7)
                    return run_local_bor._solve_and_export_star(args)
            """), encoding="utf-8")
            script = textwrap.dedent(f"""
                import sys
                sys.path[:0] = [{str(work)!r}, {str(BACKEND)!r}]
                import run_local_bor as driver
                import crash_hook
                driver._solve_and_export_star = crash_hook.star
                driver.GEOMETRY_DIRS = [{str(geometry)!r}]
                driver.FREQUENCIES_GHZ = [1.0, 1.5, 1.25]
                driver.AZIMUTHS_DEG = [0.0, 90.0, 180.0]
                driver.ELEVATIONS_DEG = [0.0]
                driver.OUTPUT_DIR = {str(work / "runs")!r}
                driver.GEOMETRY_UNITS = "meters"
                driver.MESH_CERTIFICATION = False
                driver.WORKERS_PER_UNIT = 1
                driver.main()
            """)
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=str(work), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=1200,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            runs = sorted((work / "runs").glob("run_*"))
            self.assertEqual(len(runs), 1, output)
            manifest = json.loads((runs[0] / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed", output)
            written = sorted(path.name for path in
                             (runs[0] / "results" / "by_frequency").glob("*.grim"))
            self.assertEqual(written, [
                "HH_1.000GHz_body.grim", "HH_1.250GHz_body.grim",
                "VV_1.000GHz_body.grim", "VV_1.250GHz_body.grim",
            ], output)
            self.assertIn("WorkerCrashError", output)
            self.assertIn("exit code 7", output)
            self.assertIn("failed=1", output)


class DiskAdmissionTests(unittest.TestCase):
    def test_concurrent_spills_fit_the_scratch_disk(self):
        units = ([{"name": f"spill{i}", "gb": 1.0, "disk": 30.0} for i in range(3)]
                 + [{"name": f"plain{i}", "gb": 1.0} for i in range(3)])
        pool = _TrackingPool()
        done = []
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=100.0, max_concurrent=8, cpu_budget=8,
            poll_seconds=0.0, disk_budget_gb=50.0)
        dispatcher.run(units, _prepare, lambda key, _v: done.append(key),
                       lambda key, exc: self.fail(f"{key}: {exc!r}"), _resources)
        self.assertEqual(sorted(done), sorted(unit["name"] for unit in units))
        self.assertLessEqual(pool.peak_disk, 50.0)
        self.assertGreater(pool.peak, 1, "units without spill still run alongside")
        # The two spills that fit together run together.
        roomy = _TrackingPool()
        hpc_scheduler.MemoryAwareDispatcher(
            roomy, budget_gb=100.0, max_concurrent=8, poll_seconds=0.0,
            disk_budget_gb=lambda: 65.0,
        ).run(units[:3], _prepare, lambda *_: None,
              lambda key, exc: self.fail(repr(exc)), _resources)
        self.assertEqual(roomy.peak_disk, 60.0)

    def test_one_oversized_spill_still_runs_alone(self):
        pool = _TrackingPool()
        done = []
        hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=100.0, max_concurrent=4, poll_seconds=0.0,
            disk_budget_gb=10.0,
        ).run([{"name": "huge", "gb": 1.0, "disk": 40.0},
               {"name": "other", "gb": 1.0, "disk": 5.0}],
              _prepare, lambda key, _v: done.append(key),
              lambda key, exc: self.fail(repr(exc)), _resources)
        self.assertEqual(sorted(done), ["huge", "other"])
        self.assertEqual(pool.peak, 1)

    def test_driver_spill_reservation_carries_the_solver_margin(self):
        from ghost_backend import run_local_bor
        from ghost_backend import run_hpc_bor_monostatic
        from ghost_backend.bor.streaming import STREAM_SPILL_SAFETY_FACTOR as margin
        expected = 10.0e9 * margin / GIB
        for reservation in (
            lambda estimate, free: run_local_bor._spill_reservation_gib(
                {"stream_spill_candidate_gb": estimate}, free),
            run_hpc_bor_monostatic._spill_reservation_gib,
        ):
            self.assertAlmostEqual(reservation(10.0, 100.0), expected)
            self.assertAlmostEqual(reservation(10.0, None), expected)
            # A spill that can never fit leaves the solve in memory.
            self.assertEqual(reservation(10.0, 5.0), 0.0)
            self.assertEqual(reservation(0.0, 100.0), 0.0)

    def test_paired_hpc_units_carry_the_spill_estimate(self):
        from ghost_backend import run_hpc_bor_monostatic as driver
        base = dict(geometry="/g/body.geo", geometry_stem="body",
                    geometry_input_sha256="0" * 64, frequency_ghz=4.0,
                    estimated_peak_gb=2.0, estimated_spill_gb=7.5)
        pairs = driver._paired_solve_units(
            [dict(base, polarization="VV"), dict(base, polarization="HH")])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["estimated_spill_gb"], 7.5)


@mock.patch('ghost_backend.twod.solver._solve_memory_limit_gb', lambda *a: 64.0)
@mock.patch('ghost_backend.bor.solver._solve_memory_limit_gb', lambda *a: 64.0)
class PreviewSpillRecordTests(unittest.TestCase):
    """estimate_bor_resources records the far-block spill it priced."""

    @staticmethod
    def _preview(**options):
        from ghost_backend.bor import dispatch
        from ghost_backend.bor.solver import sphere_generatrix
        points = sphere_generatrix(0.04, 12)
        snapshot = {
            "title": "PEC sphere", "ibcs": [], "dielectrics": [],
            "segments": [{
                "name": "sphere", "seg_type": 2,
                "properties": ["2", "40", "0", "0", "0"],
                "point_pairs": [
                    {"x1": float(a[0]), "y1": float(a[1]),
                     "x2": float(b[0]), "y2": float(b[1])}
                    for a, b in zip(points[:-1], points[1:])
                ],
            }],
        }
        return dispatch.estimate_bor_resources(
            snapshot, 1.0, [0.0, 90.0], geometry_units="meters", n_modes=20,
            workers=2, table_precision="double", assembly="streaming",
            stream_budget_gb=0.1, mesh_certification=False,
            bor_options=dict(factorization="dense", **options))

    def test_spilling_plan_records_its_size_and_directory(self):
        from ghost_backend.bor import streaming
        with tempfile.TemporaryDirectory() as scratch, \
                mock.patch.object(streaming, "spill_directory", lambda _gb: scratch):
            preview = self._preview()
            self.assertEqual(preview["stream_spill_directory"], scratch)
        per_mode = streaming.estimate_streaming_block_gb(
            preview["mesh_elements"], preview["mode_cap_estimate"], 1, "cfie",
            False, False)
        expected = (preview["mode_cap_estimate"] + 1) * per_mode
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(preview["stream_spill_gb_estimate"], expected)
        self.assertAlmostEqual(preview["stream_spill_candidate_gb"], expected)

    def test_no_room_keeps_the_candidate_but_plans_no_spill(self):
        from ghost_backend.bor import streaming
        with mock.patch.object(streaming, "spill_directory", lambda _gb: None):
            preview = self._preview()
        self.assertEqual(preview["stream_spill_gb_estimate"], 0.0)
        self.assertGreater(preview["stream_spill_candidate_gb"], 0.0)
        self.assertIsNone(preview["stream_spill_directory"])

    def test_spill_off_records_nothing(self):
        preview = self._preview(stream_spill="off")
        self.assertEqual(preview["stream_spill_gb_estimate"], 0.0)
        self.assertEqual(preview["stream_spill_candidate_gb"], 0.0)


class StaleSpillSweepTests(unittest.TestCase):
    def test_sweep_runs_when_the_backend_offers_it(self):
        import ghost_backend.bor.streaming as streaming
        calls, log = [], []
        with mock.patch.object(streaming, "remove_stale_spill_directories",
                               lambda: calls.append(1) or 3, create=True):
            self.assertEqual(hpc_scheduler.sweep_stale_bor_spill(log.append), 3)
        self.assertEqual(calls, [1])
        self.assertTrue(any("3" in line for line in log))

    def test_sweep_summary_reports_only_actual_removals(self):
        import ghost_backend.bor.streaming as streaming
        for result, expected in (
            (dict(removed=0, bytes=0, live=2, failed=0), []),
            (dict(removed=2, bytes=3 * 1024 ** 3, live=0, failed=0),
             ["  Stale BoR spill removed: 2 directories, 3.00 GiB"]),
        ):
            log = []
            with mock.patch.object(streaming, "remove_stale_spill_directories",
                                   lambda value=result: value, create=True):
                self.assertEqual(hpc_scheduler.sweep_stale_bor_spill(log.append), result)
            self.assertEqual(log, expected)

    def test_missing_or_failing_sweep_never_stops_a_driver(self):
        import ghost_backend.bor.streaming as streaming
        with mock.patch.dict(streaming.__dict__):
            streaming.__dict__.pop("remove_stale_spill_directories", None)
            self.assertIsNone(hpc_scheduler.sweep_stale_bor_spill(None))

        def broken():
            raise PermissionError("locked by another process")

        log = []
        with mock.patch.object(streaming, "remove_stale_spill_directories",
                               broken, create=True):
            self.assertIsNone(hpc_scheduler.sweep_stale_bor_spill(log.append))
        self.assertTrue(log and "failed" in log[0])

    def test_drivers_sweep_before_planning(self):
        from ghost_backend import run_local_bor, run_hpc_bor_monostatic
        local = inspect.getsource(run_local_bor.main)
        self.assertLess(local.index("sweep_stale_bor_spill()"),
                        local.index("_plan(solve_units"))
        worker = inspect.getsource(run_hpc_bor_monostatic.worker)
        self.assertLess(worker.index("sweep_stale_bor_spill()"),
                        worker.index("_compute_resource_plan("))


class CpuAllocationTests(unittest.TestCase):
    def test_scope_sets_the_allocation_and_no_2d_profile(self):
        from ghost_backend.execution import options
        from ghost_backend.execution.provenance import runtime_environment_fingerprint
        outside = options.allocated_cpu_budget()
        fingerprint = runtime_environment_fingerprint()
        with hpc_scheduler.cpu_allocation_scope(3):
            self.assertEqual(options.allocated_cpu_budget(), 3)
            self.assertIsNone(options.current_options())
            # The drivers verify this fingerprint inside every unit.
            self.assertEqual(runtime_environment_fingerprint(), fingerprint)
        self.assertEqual(options.allocated_cpu_budget(), outside)
        with hpc_scheduler.cpu_allocation_scope(None):
            self.assertEqual(options.allocated_cpu_budget(), outside)

    def test_bor_units_solve_inside_their_allocation(self):
        from ghost_backend.execution import options
        from ghost_backend import run_local_bor, run_hpc_bor_monostatic
        seen = []

        def solve(*_args, **_kwargs):
            seen.append(options.allocated_cpu_budget())
            return "written", "path"

        with mock.patch.object(run_local_bor, "_solve_and_export", solve):
            self.assertEqual(
                run_local_bor._solve_and_export_star(({}, {}, "dir", 3))[0], "ok")
            run_local_bor._solve_and_export_star(({}, {}, "dir"))
        with mock.patch.object(run_hpc_bor_monostatic, "_solve_and_export", solve):
            self.assertEqual(run_hpc_bor_monostatic._solve_and_export_star(
                ({}, {}, "base", "run", 5))[0], "ok")
        self.assertEqual(seen[0], 3)
        self.assertEqual(seen[1], options.allocated_cpu_budget())
        self.assertEqual(seen[2], 5)


if __name__ == "__main__":
    unittest.main()
