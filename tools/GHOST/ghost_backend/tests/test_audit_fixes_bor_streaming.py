"""Audit fixes: far-tile threads and pricing, spill lifecycle, DLL search,
EFIE far-block symmetry and adaptive RHS compression."""
import errno
import gc
import glob
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ghost_backend.bor import kernels as bor_kernels
from ghost_backend.bor import solver as bor_solver
from ghost_backend.bor import streaming as bor_streaming
from ghost_backend.bor.options import BorAdmissionError
from ghost_backend.execution.paths import native_kernel_root
from ghost_backend.linalg import sweep
from ghost_backend.linalg.dense import DenseFactor

C0 = bor_solver.C0
FREQUENCY_HZ = 1.0e9
OWNER = bor_streaming.STREAM_SPILL_OWNER_FILE


def _solver(elements=24, ka=2.0, **kwargs):
    radius = ka * C0 / (2.0 * math.pi * FREQUENCY_HZ)
    return bor_solver.BorPecSolver(bor_solver.sphere_generatrix(radius, elements), FREQUENCY_HZ, **kwargs)


def _cores(count):
    return mock.patch.object(bor_kernels, "physical_cpu_count", return_value=count)


def _assert_blocks_close(testcase, actual, expected, rtol=1e-13):
    scale = float(np.max(np.abs(expected)))
    testcase.assertLessEqual(float(np.max(np.abs(np.asarray(actual) - expected))), rtol * scale)


class FarTileThreadTests(unittest.TestCase):
    """Far tiles run on the physical cores, within the one priced tile budget."""

    def test_tile_threads_follow_cores_and_mode_workers_only_align_ranges(self):
        solver = _solver(16)
        with _cores(6):
            single = bor_streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, workers=1)
            paired = bor_streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, workers=2,
                                                      mode_block=3)
        self.assertEqual(single._workers, 6)
        self.assertEqual(single._native_threads, 1)
        self.assertEqual(single.mode_workers, 1)
        self.assertEqual(bor_streaming.streaming_tile_threads(), bor_kernels.physical_cpu_count())
        # The range is aligned to the two mode workers (3 -> 4), never to the
        # six tile threads (which would round it past the planned budget).
        self.assertEqual(paired.mode_block, 4)
        self.assertEqual(paired._workers, 6)

    def test_self_and_cross_tile_threads_are_capped_at_the_cores(self):
        solver = _solver(12)
        sp = bor_solver.BorPecSolver(bor_solver.sphere_generatrix(.035, 10), FREQUENCY_HZ,
                                     gauss_order=3, medium=(2.5 - .05j, 1.))
        sq = bor_solver.BorPecSolver(bor_solver.sphere_generatrix(.02, 8), FREQUENCY_HZ,
                                     gauss_order=3, medium=(2.5 - .05j, 1.))
        cross = bor_solver.BorCrossOperators(sp, sq)
        with _cores(4):
            stream = bor_streaming.StreamingFarBlocks(solver, 3, workers=15, tile_threads=15)
            crossed = bor_streaming.StreamingCrossFarBlocks(cross, 3, workers=15)
        self.assertEqual(stream._workers, 4)
        self.assertEqual(crossed._workers, 4)
        self.assertEqual(crossed.mode_workers, 15)

    def test_every_concurrent_tile_fits_its_share_of_the_budget(self):
        # The 10 GHz ogive of the audit: 2036 elements, m_max 26, 2048 samples.
        for threads in (1, 8, 32, 64):
            for budget in (1.0, 0.1):
                count, rows, sources, work = bor_streaming._plan_banded_tiles(
                    2036, 4, 2036, 4, 29, 27, True, True, budget, threads, 2048)
                per_tile = bor_streaming.BOR_STREAM_TILE_SLACK * bor_streaming._banded_tile_bytes(
                    rows, sources, 4, 4, 29, 27, True, True, work, 2048)
                self.assertLessEqual(count * per_tile, budget * 1.0e9 * (1.0 + 1e-12))
                # At the production budget every core gets a tile; a tiny
                # budget drops threads only below the one-element-pair floor.
                if budget == 1.0:
                    self.assertEqual(count, threads)
                else:
                    self.assertLessEqual(count, threads)
                    self.assertGreaterEqual(count, min(threads, 16))
        # A small share splits the source elements before it drops threads.
        count, rows, sources, _ = bor_streaming._plan_banded_tiles(
            2036, 4, 2036, 4, 29, 27, True, True, 0.1, 8, 2048)
        self.assertEqual((count, rows), (8, 1))
        self.assertLess(sources, 2036)

    def test_measured_tile_workspace_stays_within_the_budget(self):
        solver = _solver(60, ka=3.0)
        budget_gb = 0.004
        with _cores(4):
            gc.collect()
            tracemalloc.start()
            try:
                stream = bor_streaming.StreamingFarBlocks(solver, 10, efie=True, mfie=True,
                                                          tile_budget_gb=budget_gb)
                resident, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(stream._workers, 4)
        self.assertLess(stream._tile_sources, solver.gen.n_elems)   # source chunks engaged
        # Transient peak above what the build keeps (blocks, weights, caches):
        # the live tiles of all four threads.
        self.assertLessEqual(peak - resident, budget_gb * 1.0e9)

    def test_blocks_do_not_depend_on_threads_chunks_or_symmetry(self):
        solver = _solver(30, ka=2.5)
        reference = bor_streaming.StreamingFarBlocks(solver, 8, efie=True, mfie=True, tile_threads=1)
        with mock.patch.object(bor_streaming, "STREAM_EFIE_SYMMETRY", False):
            full = bor_streaming.StreamingFarBlocks(solver, 8, efie=True, mfie=True, tile_threads=1)
        with _cores(4):
            chunked = bor_streaming.StreamingFarBlocks(solver, 8, efie=True, mfie=True,
                                                       tile_budget_gb=0.001)
        self.assertTrue(reference._symmetric_efie)
        self.assertFalse(full._symmetric_efie)
        self.assertLess(chunked._tile_sources, solver.gen.n_elems)
        # The symmetric build keeps only the packed strict-upper triangles.
        self.assertTrue(reference._packed_efie)
        self.assertFalse(full._packed_efie)
        self.assertEqual(reference.Z.shape, (4, 9, bor_streaming.packed_upper_size(solver.Nn)))
        for other in (full, chunked):
            _assert_blocks_close(self, other.full_blocks("efie"), reference.full_blocks("efie"))
            _assert_blocks_close(self, other.K, reference.K)
        # The symmetric build is symmetric exactly; the fully sampled one to
        # rounding, which is the property the symmetric build relies on.
        Z = reference.full_blocks("efie")
        np.testing.assert_array_equal(Z[0], np.swapaxes(Z[0], -1, -2))
        np.testing.assert_array_equal(Z[3], np.swapaxes(Z[3], -1, -2))
        np.testing.assert_array_equal(Z[1], -np.swapaxes(Z[2], -1, -2))
        scale = float(np.max(np.abs(full.Z)))
        self.assertLess(float(np.max(np.abs(full.Z[0] - np.swapaxes(full.Z[0], -1, -2)))), 1e-13 * scale)
        self.assertLess(float(np.max(np.abs(full.Z[1] + np.swapaxes(full.Z[2], -1, -2)))), 1e-13 * scale)
        # What packing drops -- the diagonal and lower triangle of each
        # accumulated block -- is exactly zero: a full-storage symmetric build
        # with its completion suppressed holds only the strict upper U.
        with mock.patch.object(bor_streaming, "_adjacent_pairs_near", return_value=False), \
                mock.patch.object(bor_streaming, "_symmetrize_efie_blocks"):
            upper = bor_streaming.StreamingFarBlocks(solver, 8, efie=True, tile_threads=1)
        self.assertTrue(upper._symmetric_efie)
        self.assertFalse(upper._packed_efie)
        self.assertGreater(float(np.max(np.abs(upper.Z))), 0.0)
        on_or_below = np.tril(np.ones((solver.Nn, solver.Nn), dtype=bool))
        self.assertEqual(float(np.max(np.abs(upper.Z[..., on_or_below]))), 0.0)

    def test_packed_blocks_write_every_signed_mode_like_the_full_symmetric_build(self):
        solver = _solver(26, ka=2.5)
        packed = bor_streaming.StreamingFarBlocks(solver, 4, efie=True, tile_threads=1)
        with mock.patch.object(bor_streaming, "STREAM_EFIE_SYMMETRY", False):
            full = bor_streaming.StreamingFarBlocks(solver, 4, efie=True, tile_threads=1)
        Nn = solver.Nn
        scale = 0.37 - 1.9j
        for m in (-4, -1, 0, 1, 3):
            Z = np.full((2 * Nn, 2 * Nn), np.nan, dtype=np.complex128)
            quads = (Z[:Nn, :Nn], Z[:Nn, Nn:], Z[Nn:, :Nn], Z[Nn:, Nn:])
            packed.write_efie_blocks(m, quads, scale)
            expected = np.empty_like(Z)
            reference = (expected[:Nn, :Nn], expected[:Nn, Nn:], expected[Nn:, :Nn], expected[Nn:, Nn:])
            full.write_efie_blocks(m, reference, scale)
            self.assertFalse(np.isnan(Z).any())
            _assert_blocks_close(self, Z, expected)
            for block, other in zip(packed.efie_blocks(m), full.efie_blocks(m)):
                _assert_blocks_close(self, block, other)
            views, signs = packed.stored_blocks("efie", m)
            for view, sign, other in zip(views, signs, full.efie_blocks(m)):
                _assert_blocks_close(self, view * sign, other)

    def test_packed_efie_pricing_matches_the_built_stream(self):
        solver = _solver(18, ka=2.0)
        for symmetric in (True, False):
            with mock.patch.object(bor_streaming, "STREAM_EFIE_SYMMETRY", symmetric):
                stream = bor_streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True)
                expected = bor_streaming.estimate_streaming_block_gb(
                    solver.gen.n_elems, 5, 6, "cfie", False, False)
                total = bor_streaming.estimate_streaming_gb(solver.gen.n_elems, 5, "cfie")
            self.assertEqual(stream._packed_efie, symmetric)
            self.assertAlmostEqual((stream.Z.nbytes + stream.K.nbytes) / 1.0e9, expected, places=15)
            self.assertAlmostEqual(total, expected, places=15)

    def test_fft_sampler_and_rotated_pv_families_are_unchanged(self):
        solver = _solver(20, ka=2.0)
        zs = np.full(solver.P, 80.0 + 30.0j)
        with mock.patch.object(bor_kernels, "BANDED_FFT", False):
            symmetric = bor_streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, ibc_zs_pt=zs)
            with mock.patch.object(bor_streaming, "STREAM_EFIE_SYMMETRY", False), _cores(1):
                serial = bor_streaming.StreamingFarBlocks(solver, 5, efie=True, mfie=True, ibc_zs_pt=zs)
        for family in ("efie", "mfie", "ibc"):
            _assert_blocks_close(self, symmetric.full_blocks(family), serial.full_blocks(family))

    def test_cross_blocks_do_not_depend_on_threads_or_chunks(self):
        sp = bor_solver.BorPecSolver(bor_solver.sphere_generatrix(.035, 14), FREQUENCY_HZ,
                                     medium=(2.5 - .05j, 1.))
        sq = bor_solver.BorPecSolver(bor_solver.sphere_generatrix(.02, 10), FREQUENCY_HZ,
                                     medium=(2.5 - .05j, 1.))
        cross = bor_solver.BorCrossOperators(sp, sq)
        reference = bor_streaming.StreamingCrossFarBlocks(cross, 5, tile_threads=1)
        with _cores(3):
            chunked = bor_streaming.StreamingCrossFarBlocks(cross, 5, tile_budget_gb=2e-4)
        self.assertLess(chunked._tile_sources, sq.gen.n_elems)
        _assert_blocks_close(self, chunked.Z, reference.Z)
        _assert_blocks_close(self, chunked.B, reference.B)

    def test_retained_block_planning_ignores_the_tile_threads(self):
        plans = []
        for cores in (1, 64):
            with _cores(cores):
                plans.append((
                    bor_streaming.plan_streaming_mode_block(1000, 100, "cfie", False, False, 8.0, 64),
                    bor_streaming.plan_combined_streaming_mode_block(
                        60, ((80, 80, True, False), (50, 80, True, False)), 0.25, 32)))
        self.assertEqual(plans[0], plans[1])


class SpillLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="ghost-spilltest-")
        self.addCleanup(shutil.rmtree, self.base, True)
        self.solver = _solver(20)

    def spill_dirs(self):
        gc.collect()
        return glob.glob(os.path.join(self.base, "ghost-bor-*"))

    def test_spilled_blocks_match_ram_and_go_with_close_and_the_last_view(self):
        ram = bor_streaming.StreamingFarBlocks(self.solver, 6, efie=True, mfie=True)
        spilled = bor_streaming.StreamingFarBlocks(self.solver, 6, efie=True, mfie=True, spill=self.base)
        directory = spilled._spill.path
        self.assertTrue(os.path.isfile(os.path.join(directory, OWNER)))
        self.assertIsInstance(spilled.Z, np.memmap)
        self.assertEqual(spilled.memory_gb(), 0.0)
        _assert_blocks_close(self, spilled.Z, ram.Z)
        _assert_blocks_close(self, spilled.full_blocks("efie"), ram.full_blocks("efie"))
        _assert_blocks_close(self, spilled.K, ram.K)
        spilled_gb = spilled.spilled_gb()
        self.assertGreater(spilled_gb, 0.0)
        # (Packed EFIE blocks are reconstructed on read; MFIE views alias the file.)
        views, _signs = spilled.stored_blocks("mfie", 2)
        self.assertIsInstance(views[0], np.memmap)
        expected = np.array(views[0])
        spilled.close()
        self.assertEqual(spilled.spilled_gb(), spilled_gb)
        # A view held across close() keeps its own mapping: reading it is safe.
        np.testing.assert_array_equal(np.array(views[0]), expected)
        with self.assertRaises(RuntimeError):
            spilled.efie_blocks(1)          # a closed stream is never rebuilt in RAM
        views = _signs = None
        self.assertEqual(self.spill_dirs(), [])

    def test_failed_build_leaves_no_spill(self):
        with mock.patch.object(bor_streaming, "_efie_band", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                bor_streaming.StreamingFarBlocks(self.solver, 4, efie=True, mfie=True, spill=self.base)
        self.assertEqual(self.spill_dirs(), [])

    def test_files_vanish_with_a_killed_process_and_the_sweep_removes_its_directory(self):
        code = textwrap.dedent(f"""
            import math, os, sys
            sys.path.insert(0, {str(ROOT)!r})
            from ghost_backend.bor import solver as s, streaming as st
            radius = 2.0 * s.C0 / (2.0 * math.pi * 1e9)
            solver = s.BorPecSolver(s.sphere_generatrix(radius, 20), 1e9)
            stream = st.StreamingFarBlocks(solver, 4, efie=True, mfie=True, spill={self.base!r})
            print(stream._spill.path, sorted(os.listdir(stream._spill.path)), flush=True)
            os._exit(3)
        """)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                timeout=900)
        self.assertEqual(result.returncode, 3, result.stderr[-2000:])
        directory = result.stdout.strip().splitlines()[-1].split(" [")[0]
        # The data files went with the process: only the owner marker is left.
        self.assertEqual(os.listdir(directory), [OWNER])
        live = tempfile.mkdtemp(prefix="ghost-bor-far-", dir=self.base)
        bor_streaming._write_owner_marker(live)
        report = bor_streaming.remove_stale_spill_directories(self.base)
        self.assertTrue(report)
        self.assertEqual((report["removed"], report["live"]), (1, 1))
        self.assertFalse(os.path.exists(directory))
        self.assertTrue(os.path.isdir(live))
        self.assertFalse(bor_streaming.remove_stale_spill_directories(self.base))

    def test_sweep_removes_only_dead_or_abandoned_owners(self):
        host = socket.gethostname()
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        dead_pid = child.pid
        del child               # release the process handle (Windows keeps it until then)
        gc.collect()

        def make(name, record=None, age_s=0.0, payload=1000):
            path = os.path.join(self.base, name)
            os.mkdir(path)
            if record is not None:
                with open(os.path.join(path, OWNER), "w", encoding="utf-8") as handle:
                    json.dump(record, handle)
            with open(os.path.join(path, "efie.npy"), "wb") as handle:
                handle.write(b"x" * payload)
            if age_s:
                stamp = time.time() - age_s
                for item in [path] + [os.path.join(path, n) for n in os.listdir(path)]:
                    os.utime(item, (stamp, stamp))
            return path

        dead = make("ghost-bor-far-dead", dict(pid=dead_pid, host=host, process_start=None))
        live = make("ghost-bor-far-live", dict(pid=os.getpid(), host=host,
                                               process_start=bor_streaming._process_start_time(os.getpid())))
        foreign = make("ghost-bor-cross-foreign", dict(pid=dead_pid, host=host + "-elsewhere"))
        young = make("ghost-bor-far-young")
        legacy = make("ghost-bor-far-legacy", age_s=2 * 3600.0, payload=5000)
        other = make("unrelated-dir", dict(pid=dead_pid, host=host))
        report = bor_streaming.remove_stale_spill_directories(self.base)
        self.assertEqual(report["removed"], 2)
        self.assertEqual((report["live"], report["foreign"], report["recent"], report["failed"]),
                         (1, 1, 1, 0))
        self.assertGreaterEqual(report["bytes"], 6000)
        self.assertFalse(os.path.exists(dead) or os.path.exists(legacy))
        for kept in (live, foreign, young, other):
            self.assertTrue(os.path.isdir(kept))
        try:
            import psutil  # noqa: F401
        except ImportError:
            return
        # A running process whose start time differs is a reused id, not the owner.
        reused = make("ghost-bor-far-reused", dict(pid=os.getppid(), host=host, process_start=1.0))
        bor_streaming.remove_stale_spill_directories(self.base)
        self.assertFalse(os.path.exists(reused))

    def test_ram_backed_directories_are_refused(self):
        with mock.patch.object(bor_streaming, "memory_backed_filesystem", return_value="tmpfs"):
            self.assertIsNone(bor_streaming.spill_directory(1e-6))
            with self.assertRaises(bor_streaming.StreamingSpillError) as caught:
                bor_streaming.StreamingFarBlocks(self.solver, 4, spill=self.base)
        self.assertIsInstance(caught.exception, BorAdmissionError)
        self.assertTrue(caught.exception.streaming)
        self.assertEqual(self.spill_dirs(), [])

    def test_linux_ram_filesystem_detection(self):
        with mock.patch.object(bor_streaming.sys, "platform", "linux"):
            with mock.patch.object(bor_streaming, "_statfs_filesystem", return_value="ramfs"):
                self.assertEqual(bor_streaming.memory_backed_filesystem("/x"), "ramfs")
            with mock.patch.object(bor_streaming, "_statfs_filesystem", return_value=None), \
                    mock.patch.object(bor_streaming, "_mounted_filesystem", return_value="tmpfs"):
                self.assertEqual(bor_streaming.memory_backed_filesystem("/tmp"), "tmpfs")
            with mock.patch.object(bor_streaming, "_statfs_filesystem", return_value=None), \
                    mock.patch.object(bor_streaming, "_mounted_filesystem", return_value="ext4"):
                self.assertIsNone(bor_streaming.memory_backed_filesystem("/var/tmp"))
        mounts = ("sysfs /sys sysfs rw 0 0\n/dev/sda1 / ext4 rw 0 0\n"
                  "tmpfs /tmp tmpfs rw 0 0\n/dev/sdb1 /tmp/disk xfs rw 0 0\n")
        with mock.patch.object(bor_streaming, "open", mock.mock_open(read_data=mounts), create=True), \
                mock.patch.object(bor_streaming.os.path, "realpath", side_effect=lambda p: p):
            self.assertEqual(bor_streaming._mounted_filesystem("/tmp/x"), "tmpfs")
            self.assertEqual(bor_streaming._mounted_filesystem("/tmp/disk/y"), "xfs")
            self.assertEqual(bor_streaming._mounted_filesystem("/home/u"), "ext4")
        self.assertEqual(bor_streaming._MEMORY_FILESYSTEM_MAGIC[0x01021994], "tmpfs")

    def test_short_disk_and_failed_allocation_are_admission_errors(self):
        real = shutil.disk_usage

        def tight(path):
            return real(path)._replace(free=1024)

        with mock.patch.object(bor_streaming.shutil, "disk_usage", side_effect=tight):
            with self.assertRaisesRegex(bor_streaming.StreamingSpillError, "only .* GB is free"):
                bor_streaming.StreamingFarBlocks(self.solver, 4, efie=True, mfie=True, spill=self.base)
        self.assertEqual(self.spill_dirs(), [])
        failure = OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(bor_streaming, "_mapped_spill_array", side_effect=failure):
            with self.assertRaisesRegex(bor_streaming.StreamingSpillError, "disk is full") as caught:
                bor_streaming.StreamingFarBlocks(self.solver, 4, efie=True, mfie=True, spill=self.base)
        self.assertTrue(caught.exception.streaming)
        self.assertEqual(self.spill_dirs(), [])

    def test_posix_preallocation(self):
        calls = []
        with mock.patch.object(bor_streaming.os, "posix_fallocate", create=True,
                               side_effect=lambda fd, offset, size: calls.append((fd, offset, size))):
            bor_streaming._preallocate(7, 4096)
        self.assertEqual(calls, [(7, 0, 4096)])
        with mock.patch.object(bor_streaming.os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(OSError):
                bor_streaming._preallocate(7, 4096)
        with mock.patch.object(bor_streaming.os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.EOPNOTSUPP, "unsupported")), \
                mock.patch.object(bor_streaming.os, "ftruncate") as truncate:
            bor_streaming._preallocate(7, 4096)
        truncate.assert_called_once_with(7, 4096)

    def test_spilled_solve_matches_tables_and_cleans_up(self):
        from ghost_backend.execution.options import execution_scope
        points = bor_solver.sphere_generatrix(0.1, 40)
        common = dict(formulation="cfie", bor_options=dict(factorization="dense"))
        angles = [0.0, 45.0, 90.0, 135.0, 180.0]
        tables = bor_solver.solve_bor(points, FREQUENCY_HZ, angles, assembly="tables", **common)
        with execution_scope(dict(temporary_directory=self.base)):
            spilled = bor_solver.solve_bor(points, FREQUENCY_HZ, angles, assembly="streaming",
                                           stream_budget_gb=0.002, workers=2, **common)
        self.assertGreater(spilled["stream_spill_gb"], 0.0)
        self.assertEqual(spilled["stream_sweeps"], 1)
        for key in ("amp_vv", "amp_hh"):
            np.testing.assert_allclose(spilled[key], tables[key], rtol=2e-10, atol=2e-12)
        self.assertEqual(self.spill_dirs(), [])


def _pe_imports(path):
    data = Path(path).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    sections_count, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    import_rva = struct.unpack_from("<I", data, optional + (112 if magic == 0x20B else 96) + 8)[0]
    sections = [struct.unpack_from("<8sIIII", data, optional + optional_size + 40 * i)
                for i in range(sections_count)]

    def offset(rva):
        for _name, vsize, vaddr, rsize, rptr in sections:
            if vaddr <= rva < vaddr + max(vsize, rsize):
                return rva - vaddr + rptr
        raise ValueError(rva)

    names, cursor = [], offset(import_rva)
    while True:
        name_rva = struct.unpack_from("<5I", data, cursor)[3]
        if not name_rva:
            return names
        start = offset(name_rva)
        names.append(data[start:data.index(b"\0", start)].decode())
        cursor += 20


class NativeDllSearchTests(unittest.TestCase):
    def dll_patches(self, added):
        return [mock.patch.object(bor_streaming, "_windows_dll_search_supported", return_value=True),
                mock.patch.object(bor_streaming, "_DLL_DIRECTORIES_ADDED", set()),
                mock.patch.object(bor_streaming, "_DLL_DIRECTORY_HANDLES", []),
                mock.patch.object(bor_streaming.os, "add_dll_directory", create=True,
                                  side_effect=lambda d: added.append(d) or object()),
                mock.patch.object(bor_streaming.shutil, "which",
                                  return_value=os.path.join(os.sep, "toolchain", "bin", "gcc.exe"))]

    def test_only_the_kernel_directory_joins_the_search(self):
        added = []
        patches = self.dll_patches(added)
        for patch in patches:
            patch.start()
        try:
            bor_streaming._prepare_windows_dll_search()
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual(added, [os.path.normcase(os.path.abspath(str(native_kernel_root())))])

    def test_compiler_directories_only_after_a_failed_load(self):
        added, attempts = [], []

        def cdll(path):
            attempts.append(path)
            if len(attempts) == 1:
                raise OSError("missing compiler runtime")
            return "library"

        patches = self.dll_patches(added) + [
            mock.patch.object(bor_streaming.os.path, "isdir", return_value=True),
            mock.patch.object(bor_streaming.ctypes, "CDLL", side_effect=cdll),
            mock.patch.dict(os.environ, {"GHOST_NATIVE_DLL_DIR": os.path.join(os.sep, "custom")})]
        for patch in patches:
            patch.start()
        try:
            self.assertEqual(bor_streaming._load_library("kernel.dll"), "library")
            self.assertEqual(len(attempts), 2)
            self.assertIn(os.path.normcase(os.path.abspath(os.path.join(os.sep, "custom"))), added)
            self.assertIn(os.path.normcase(os.path.abspath(os.path.join(os.sep, "toolchain", "bin"))), added)
            # Nothing new to add: the second failure is reported, not retried.
            attempts.clear()
            with mock.patch.object(bor_streaming.ctypes, "CDLL", side_effect=OSError("still missing")):
                with self.assertRaises(OSError):
                    bor_streaming._load_library("kernel.dll")
        finally:
            for patch in reversed(patches):
                patch.stop()

    @unittest.skipUnless(sys.platform == "win32", "Windows kernel build")
    def test_released_kernel_imports_only_system_runtime(self):
        paths = glob.glob(os.path.join(str(native_kernel_root()), "bor_stream_kernel*.dll"))
        if not paths:
            self.skipTest("native kernel not built")
        for path in paths:
            for name in _pe_imports(path):
                self.assertTrue(name.upper() == "KERNEL32.DLL" or name.lower().startswith("api-ms-win-crt-"),
                                f"{os.path.basename(path)} imports {name}")


class AdaptiveRhsCompressionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(20260922)
        self.n = 800
        self.a = np.eye(self.n) * 4 + .01j * self.rng.randn(self.n, self.n)

    def rhs(self, count, rank=None):
        if rank is None:
            return self.rng.randn(self.n, count) + 1j * self.rng.randn(self.n, count)
        return self.rng.randn(self.n, rank) @ (self.rng.randn(rank, count) + 1j * self.rng.randn(rank, count))

    def solve_batches(self, rhs, setting, hint=None, width=96):
        factor, basis = DenseFactor(self.a), sweep.SweepBasis(256)
        parts = [sweep.solve(factor, rhs[:, i:i + width], basis, setting=setting, hint=hint)
                 for i in range(0, rhs.shape[1], width)]
        result = np.column_stack(parts)
        np.testing.assert_allclose(result, np.linalg.solve(self.a, rhs), rtol=1e-11, atol=1e-12)
        return factor.event["sweep_compression"]

    def test_factor_is_suspended_after_a_batch_that_did_not_pay(self):
        original = sweep._qr_basis
        with mock.patch.object(sweep, "_qr_basis", side_effect=original) as qr:
            event = self.solve_batches(self.rhs(288), "auto")
        self.assertEqual(qr.call_count, 1)
        self.assertEqual((event["fallback_batches"], event["suspended_batches"]), (1, 2))
        self.assertEqual(event["auto_suspended"], "fallback")
        self.assertEqual(event["solved_columns"], 288)

    def test_compression_continues_while_it_pays(self):
        event = self.solve_batches(self.rhs(288, rank=5), "auto")
        self.assertEqual(event["accepted_batches"], 3)
        self.assertNotIn("auto_suspended", event)
        self.assertLessEqual(event["solved_columns"], 10)

    def test_auto_reconstructs_only_ranks_below_half_the_batch(self):
        rhs = self.rhs(100, rank=60)
        auto = self.solve_batches(rhs, "auto", width=100)
        forced = self.solve_batches(rhs, "on", width=100)
        self.assertEqual((auto["accepted_batches"], auto["solved_columns"]), (0, 100))
        self.assertEqual((forced["accepted_batches"], forced["solved_columns"]), (1, 60))

    def test_qr_is_skipped_when_it_cannot_pay(self):
        self.assertTrue(sweep._auto_qr_cannot_pay(600, 256))
        self.assertFalse(sweep._auto_qr_cannot_pay(2644, 128))
        a = np.eye(600) * 4 + .01j * self.rng.randn(600, 600)
        rhs = self.rng.randn(600, 256) + 0j
        factor = DenseFactor(a)
        with mock.patch.object(sweep, "_qr_basis", side_effect=AssertionError("QR")):
            result = sweep.solve(factor, rhs, sweep.SweepBasis(256), setting="auto")
        np.testing.assert_allclose(result, np.linalg.solve(a, rhs), rtol=1e-11, atol=1e-12)
        event = factor.event["sweep_compression"]
        self.assertEqual(event["small_batches"], 1)
        self.assertNotIn("auto_suspended", event)

    def test_hint_backs_off_across_factors_and_resets_when_compression_pays(self):
        hint = sweep.CompressionHint()
        outcomes = []
        for _ in range(8):
            event = self.solve_batches(self.rhs(192), "auto", hint=hint)
            outcomes.append(event.get("auto_suspended"))
        self.assertEqual(outcomes, ["fallback", "hint", "fallback", "hint", "hint",
                                    "fallback", "hint", "hint"])
        self.assertEqual(hint.evidence(), dict(factors=8, probed=3, skipped=5, productive=0,
                                               unproductive=3))
        # Two more skips remain; the next probe finds a compressible sweep and
        # every later factor attempts again.
        for _ in range(2):
            self.assertEqual(self.solve_batches(self.rhs(192), "auto", hint=hint)["auto_suspended"], "hint")
        for _ in range(2):
            event = self.solve_batches(self.rhs(192, rank=4), "auto", hint=hint)
            self.assertEqual(event["accepted_batches"], 2)
        self.assertEqual(hint.evidence()["productive"], 2)

    def test_on_and_off_ignore_the_hint(self):
        hint = sweep.CompressionHint()
        hint.record(False)
        forced = self.solve_batches(self.rhs(192, rank=4), "on", hint=hint)
        direct = self.solve_batches(self.rhs(192, rank=4), "off", hint=hint)
        self.assertEqual(forced["accepted_batches"], 2)
        self.assertEqual(direct["accepted_batches"], 0)
        self.assertEqual(hint.evidence()["factors"], 0)

    def test_hint_is_thread_safe(self):
        hint = sweep.CompressionHint()

        def work():
            for index in range(200):
                if hint.begin_factor():
                    hint.record(index % 3 == 0)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        evidence = hint.evidence()
        self.assertEqual(evidence["factors"], 1600)
        self.assertEqual(evidence["probed"] + evidence["skipped"], 1600)
        self.assertEqual(evidence["productive"] + evidence["unproductive"], evidence["probed"])


if __name__ == "__main__":
    unittest.main()
