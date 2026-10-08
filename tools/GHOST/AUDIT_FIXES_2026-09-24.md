# GHOST audit fixes, 24 September 2026

This release implements the ranked fixes of the 24 September architecture
audit of the BoR and 2D solvers (time, memory and robustness). Every finding
was measured before it was changed, and every change was checked against the
previous code (`tools/GHOST_backup_before_audit3_fixes_20260924.zip` holds that
version).

## What changes when you rerun

- **BoR results** move by at most 5e-7 relative (8e-6 dB), because far element
  pairs, the excitation and the far-field projection now use three Gauss points
  per element instead of four (see [numerical methods](NUMERICAL_METHODS.md)).
  Against exact series the errors are unchanged to 1e-4 dB on PEC (CFIE and
  EFIE), impedance, lossy and lossless dielectric and coated spheres; the change
  is five to six orders of magnitude below the discretization error. The near
  blocks are now contracted by matrix products (rounding-level changes, 3e-14
  relative in the amplitudes). Everything else is an exact reorganization.
- **2D results** change at rounding level only (at most 6e-13 relative),
  because dense linear algebra now runs on the physical cores instead of all
  SMT threads (fix 10); every other 2D change is bitwise identical.
- **Memory**: spawned worker processes no longer commit a gigabyte of BLAS
  buffers each; spilled far blocks no longer stay in the working set; a large
  dense factor keeps its original coefficients in memory whenever they fit.
  The 10 GHz ogive runs, which used to leave the system with 8-10 MB of
  available memory, now leave 14-15 GB.
- **Planning**: far tables are 44 % smaller and symmetric self-surface EFIE
  blocks are stored as one triangle, so automatic plans near a memory boundary
  can resolve differently (tables where streaming was chosen, for instance).
  The mode window starts fewer modes past the converged tail. The material
  and junction solvers now spill their far blocks when the stream budget
  cannot hold every mode (fix 11), and the previews price it.
- **Crash fixed** (fix 12): an in-memory streamed solve whose far blocks were
  rebuilt during the mode sweep could end the process with an access
  violation inside OpenBLAS; the code before these fixes had the same defect
  under small CPU allocations.
- **Follow-ups 11-13** change results at rounding level only (the far tiles
  are summed per node).

## Verification

**Test suite.** `python ghost_backend/tests/run_suite.py` passes on Windows:
1,192 tests (1,073 headless and 119 in the 15 Qt modules), one skipped. New
tests: `tests/test_audit_fixes_2026_09_24.py` (26, of which 10 for follow-ups
11-13) and the physical-core cap in
`tests/test_bor_performance_fixes.py`; tests whose numbers were
calibrated to four far Gauss points or to full EFIE storage were updated (the
round-5 planning windows use bodies with the same number of Gauss points, a
fore-aft symmetry tolerance follows the three-point rule, streaming-block
counts follow the packed storage).

**Against the previous code.** 25 regression cases (BoR tables and streamed
CFIE, EFIE, uniform and tapered impedance, lossy dielectric, coated, coated
two-layer, partial coating, layered patch, banded coating, cylinder, ogive at
1.5 and 4 GHz; 2D airfoil 1 and 3 GHz certified, circle, mixed, dielectric,
re-entrant and sheet fixtures): BoR amplitudes within 2.5e-9 to 4.7e-7
relative (8e-6 dB), 2D within 6e-13. Each fix was checked separately; all but
the Gauss order were bitwise identical or within 3e-14 (the BLAS thread cap
6e-13 on the 3 GHz airfoil). After follow-ups 11-13 the same cases are
bitwise identical to fixes 1-10 on the tables path and in 2D, and within
5e-15 on the streamed path (the per-node tile sums).

**End to end**, on the 8-core workstation (Ryzen 9800X3D, 31 GB, 2.1 GB
pagefile), baseline and new code alternated; seconds, and the lowest system
available memory during the run:

| Case | Before | After |
| --- | --- | --- |
| BoR 120 x 5 in ogive survey, 4 GHz, 181 aspects | 22.2 / 22.7 (19 GB available) | 15.5 / 15.7 |
| same, peak private memory including workers | 19.7 GB | 6.3 GB |
| BoR ogive survey, 10 GHz | 130.3 (8 MB available) | 108-121 (14 GB available) |
| BoR ogive certified, 10 GHz (3,360-element fine mesh) | 714.5 (10 MB available, 39 GB spill, 25.7 GB peak working set) | 401.1 (15 GB available, 29 GB spill, 11.2 GB) |
| 2D airfoil certified, 10 GHz, dense | 207 (audit) | 145-163 |
| same, original spooled to disk (as under memory pressure) | over 23 minutes, stopped | 209 |
| 2D airfoil certified, 1 / 3 GHz | 5.0-5.5 / 16.2-16.6 | 3.0 / 11.2 |

Run-to-run variation on this desktop is about 10-15 % (the far build of the
10 GHz survey took 68 s and 83 s in two runs of identical code), so ranges are
given where runs were repeated.

# Changes by fix

## 1. Spawned workers start with one BLAS thread (`execution/runtime.py`, `bor/near_parallel.py`, `compressed/tile_processes.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| OpenBLAS reserves its per-thread buffers when NumPy and SciPy are imported, in a spawned worker's bootstrap, before any initializer can limit it: each near-preparation worker committed about 1.05 GB while touching 50-70 MB (15 workers, 16 GB of the 35.5 GB commit limit), and the plan priced 448 MB. | `pin_worker_environment` sets `OPENBLAS/OMP/MKL_NUM_THREADS=1` (reference counted, restored by the last release) while a near-preparation or compressed-tile pool can spawn workers. | 40 MB of commit per worker (measured); the 4 GHz survey's peak private memory 19.7 GB to 6.3 GB. |

## 2. Packed symmetric EFIE storage (`bor/streaming.py`, `bor/solver.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| The symmetric self-surface EFIE build sampled one triangle, then completed the whole `[4, modes, Nn, Nn]` store in one pass after the build: on a spill larger than the free RAM that pass paged the file at about 340 MB/s (213 s of the certified 10 GHz solve, 5 s on the base mesh). | Only the strictly upper triangle is stored, packed row by row (the near stencil excludes adjacent elements, so nothing else is ever written); `write_efie_blocks` unpacks a mode into the system quadrants and completes it there. Pricing (`estimate_streaming_gb`, `estimate_streaming_block_gb`) counts the triangle. | Bitwise identical (the completion adds exact zeros); a test builds with the completion suppressed and checks the dropped parts are exactly zero. EFIE spill 50 % smaller (39 GB to 29 GB with the MFIE blocks). |
| Reading a packed mode unpacked it row by row, two Python-level calls per row, serialized on the GIL across mode workers. | One masked assignment per row chunk (the upper positions of a chunk in row-major order are its packed segment); completion blocks of 128 instead of 512. | Bitwise identical; modal assembly of the 4 GHz survey 12.4 to 6 stage-seconds. |

## 3. Residual storage by need (`linalg/residual_spool.py`, `linalg/dense.py`, `bor/factor.py`, `linalg/workspace.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| 'auto' spooled every eligible matrix of 512 MiB or more to disk, although both admission plans price the copy (the 2D estimate two matrices, BoR three per mode worker): the certified 10 GHz airfoil re-read its 7.8 GB system about 20 times (64 s of 207 s). | 'auto' keeps the original in memory unless the memory available as the matrix is factored cannot hold the copy (plus an eighth, at least 256 MiB); 'disk' and 'memory' are unchanged. The same rule serves the 2D and BoR factors. | The airfoil ran in 145-161 s kept in memory, 16.40 GB above its starting point against a 16.73 GB (15.58 GiB) estimate: the audit's claim that the estimate was too low compared GiB with GB and counted the process's own 1 GB. |
| Four full passes over the matrix before the LU (finite check, infinity norm, two equilibration passes). | One pass (`checked_row_norms`) gives the finite check, the infinity norm (bitwise as before) and the row maxima; the equilibration's column pass is computed once, before any in-place LU. | Condition estimates and residual evidence bitwise identical; new tests. |

## 4. Three far Gauss points (`bor/kernels.py`, `bor/solver.py`, `bor/dispatch.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Far pairs are at least two element lengths apart at 20 elements per wavelength, yet were integrated with four points per element; three changed the 4 GHz ogive by 2e-7 against a 9e-3 discretization error and saved 19 %. | `kernels.FAR_GAUSS_ORDER = 3` is the default of every solver, estimate and preview; two estimates in dispatch still hard-coded 4 (a preview/gate mismatch the change exposed) and now use the constant. | Mie: see [numerical methods](NUMERICAL_METHODS.md) (2e-9 to 2e-8 relative, errors unchanged to 1e-4 dB); far tables 44 % smaller. |

## 5. Excitation records shared by all modes (`bor/solver.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Every mode re-evaluated two Bessel orders (`jv`, 280-400 ns per value) and the mode-independent axial phase at every point and aspect: about 70 CPU-s of a certified 10 GHz sweep. | One `_AngularChunk` per aspect chunk holds `u`, the phase and `J` at a top order above the sweep's modes; each mode follows by the stable downward recurrence (0.3 ns per value and order). Values whose top orders underflow (axis aspects, tiny `u`) are evaluated directly. The records share a 256 MiB process-wide budget, priced once in the mode-phase estimate. | Recurrence within 1e-13 absolute of `jv` from `u = 0` to `k rho = 300`, no floating-point exceptions; excitation 92 to 42 CPU-s at 10 GHz. |

## 6. Mode window (`bor/solver.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| A mode's increment is judged only when it is consumed in order, so a full window kept `workers - 1` modes beyond the converged tail running (16 computed to use 10 at 2 GHz, 27 for 24 at 10 GHz), and the executor waited for them. | The window stops at a predicted end: before the tail, the tail start plus the automatic cap's transition allowance; inside it, the geometric extrapolation of the last two relative increments (which errs late). A tail that stops decaying lifts the limit. Modes running past convergence (or behind a failure) stop at their next checkpoint. | Identical fields; a synthetic geometric tail starts 13 modes instead of 21; results record `mode_tasks_started`. |

## 7. Near preparation in batches (`bor/solver.py`, `bor/near_parallel.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Each near pair evaluated its points in its own graded-rule calls (82,746 native calls for 2,574 pairs), with about 40 % of the time in Python between them; `einsum` contractions took 65 us per 256-point chunk. | A batch of pairs (32 per process task or local batch) is evaluated by one graded-rule call per kind and bounded point block, then contracted chunk by chunk in each pair's own order; disjoint pairs refine together level by level. Each contraction is one matrix product. | The rule treats every point independently, so the kernel values are bitwise those of separate calls; the matrix-product contraction moves blocks by rounding (amplitudes within 3e-14). 2,574 pairs: 3,268 native calls, 20.5 s to 16.3 s serially; small solves on the serial/thread path 1.8-2x faster. |

## 8. Bounded spill working set (`bor/streaming.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Pages a process touches in a mapped file stay in its working set: the spilled 10 GHz runs held the whole spill resident (23.7-26 GB working set against a 14.9-20.4 GB plan that assumed two resident modes). | Completed node rows of every mode are flushed and released (`VirtualUnlock` on Windows, `MADV_DONTNEED` elsewhere; whole pages inside a finished range only) as the far build finishes them, and a mode's blocks are released chunk by chunk as they are copied into a system matrix (`write_efie_blocks`, `add_blocks`, cross-surface reads). | Build and read of the 10 GHz base-mesh stream: working set 16.3 GB to 2.0 GB (tile workspace included), 0.4 GB after the read; spilled and in-memory streams assemble bitwise the same systems. |

## 9. Far build arithmetic (`bor/native/bor_stream_kernel.c`, `bor/streaming.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| The native samplers evaluated `exp(ki R)` for a real wavenumber. | `GHOST_DECAY`: exactly 1 for a real `k`, so the call is skipped (far and near samplers). | Bitwise identical (rebuilt DLL, all 13 entry points); Green's-function sampler 2.9 s to 2.4 s, MFIE 9.3 s to 8.7 s at 4 GHz. |
| NumPy promoted the real basis weights of the tile contraction to complex, so half of its multiplications were by zeros. | The contraction multiplies the real weights by the real view of the complex samples. | Bitwise identical; EFIE tile contraction 13.6 s to 11.3 s (CPU, 4 GHz). |
| (Tried) planar native samples for real-GEMM transforms. | Not kept: the transform is memory-bound at about 20 orders, and the planar stores made the sampler slower (26.1 s against 31.0 s). | Measured. |

## 10. BLAS threads on the physical cores (`execution/options.py`, `bor/solver.py`, `linalg/residual_spool.py`)

Found while measuring fix 3: a 2D solve whose original was spooled to disk
ran for over 23 minutes instead of about 3.

| Finding | Fix | Verification |
| --- | --- | --- |
| Dense linear algebra used the scheduler's CPU budget, which counts SMT threads (16 on this 8-core host). OpenBLAS products of fewer than about 256 rows then took 0.5 to 2.1 s instead of 1.4 to 3.7 ms on 8 threads (complex, 8,000 columns, 256 right-hand sides), and LU was slower too (N = 8,000: 3.06 s against 2.19 s). The spool's residual products use 47-row blocks at 22k unknowns; the RHS-compression projections are products with few rows. | `blas_core_budget()`: the CPU allocation, at most the physical cores, for the 'auto' 2D BLAS threads and BoR's per-worker share (explicit settings unchanged). Spool blocks have at least 256 rows. | Spooled 10 GHz airfoil: over 23 minutes to 209 s. Certified airfoil 1 GHz 4.8 to 3.0 s, 3 GHz 14.7 to 11.2 s; results within 6e-13. |

## 11. One far build for the material and junction solvers (`bor/solver.py`, `bor/streaming.py`, `bor/dispatch.py`)

A follow-up after the ten fixes above.

| Finding | Fix | Verification |
| --- | --- | --- |
| Only `solve_bor` spilled its far blocks. The dielectric, coated, partial-coating and multi-region solvers rebuilt every stream once per mode range when the stream budget could not hold every mode, and aligned the ranges to fewer mode workers. The 120 x 5 in dielectric ogive needs two ranges at 5 GHz and three at 6 GHz (8 mode workers), at the default 8 GB budget. | The same decision as `solve_bor` (`plan_stream_spill`): every stream of the solve is built once into memory-mapped files (`combined_stream_mode_gb` prices one mode of all of them), two modes are priced resident, and every requested mode worker runs. `estimate_bor_resources` and the dense/compressed chooser price the same plan: exactly for the dielectric and coated solves; for junction layouts, whose preview streams only approximate the solve's, a spill is priced only when the streams the solve always builds cannot hold every mode, and otherwise the in-memory block with every requested worker. Results record `stream_spill_gb`. | Spilled, in-memory-range and one-range solves agree within 2e-10 (dielectric, coated, partial coating, two-layer coating; new tests, spill directories removed). Dielectric ogive, final code: at 5 GHz 2 far builds instead of 4 and 9.9-11.2 GB peak private instead of 14.2 GB, in the same time (153-166 s spilled, 159-166 s in memory; the memory-planned mode workers vary between runs); at 6 GHz 2 builds instead of 6, 258 s instead of 284 s, 10.0 GB instead of 14.4 GB (18.8 GB spilled). |

## 12. Far tiles and near threads make their BLAS calls on one thread (`execution/options.py`, `bor/streaming.py`, `bor/solver.py`)

Found while measuring fix 11: the in-memory 5 GHz dielectric solve died
without a message in two of three runs.

| Finding | Fix | Verification |
| --- | --- | --- |
| A streamed range rebuilt inside a mode worker runs its eight far-tile threads under that worker's BLAS share. When the share is more than one thread, OpenBLAS 0.3.31 (pthreads layer, Windows) faults with an access violation in its own worker threads under the tiles' many concurrent multithreaded products. The code before these fixes has the same defect: it gave each of two mode workers 8 of 16 logical CPUs and never crashed in four solves, but crashed in its first solve with a 4-CPU allocation (2 threads per call), as a batch unit with a small allocation would run. Fix 10's physical-core share (8 cores for 3 workers, 2 threads each) made it likely on this workstation. | `single_thread_blas()`: while a pool of Python threads issues BLAS calls (far tiles, local near-preparation threads), the process-wide BLAS limit is one thread; sections nest across threads, the first sets the limit and the last restores it. The tiles already occupy the cores. | Repeated 3 GHz in-memory multi-range dielectric solves: without the guard, 1 of 3 and (with the complex contraction of the old code) 3 of 3 crashed; with BLAS on one thread 4 of 4 ran, and with the guard 3 of 3 and 3 of 3, about 5 % faster (the tiles no longer oversubscribe the cores). Tests check that tiles and near threads see one thread and that the limits come back. |

## 13. Far tiles assembled per node (`bor/streaming.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| A far tile contracted each of its test elements' two basis functions with the sources separately, and combined every EFIE term into modes once per basis-function pair (36 combinations per tile, with their temporaries). | The two test functions of a node are summed before the source-side product (one product per left kind instead of two); each EFIE term sums its source node's two functions in order space and is combined into modes once, in place, in reused buffers; the bracket families likewise. The tile workspace model counts the node sums. | The same sums to rounding (3.5e-16 of the largest entry on captured tiles; a new test holds the former assembly within 1e-13, real and lossy k); 37.5 to 18.3 ms per EFIE tile and 11.5 to 10 ms per bracket tile (4 GHz ogive, five-row tiles). 4 GHz survey 15.9-16.0 s to 15.0-15.3 s; 2 GHz dielectric 2.5 % faster; no change at 10 GHz, whose tiles are one element tall (a node then has nothing to share). |

The largest remaining far-build cost is the azimuthal sampling and its
transform (58 % of the 10 GHz build's thread time, the transform GEMMs
already at about 80 GFLOPS: a real-arithmetic transform with transposed
copies was slower). Its size is set by the per-pair sample count, whose
decay rule targets 4e-14 of each pair's largest coefficient
(`FAR_DECAY_LENGTH`): the closest far pairs need up to 1,024 samples. A
looser target would shorten the build roughly in proportion, but it is an
accuracy decision and was not changed.

## Other

- The batch scheduler's BoR cost model (`hpc/scheduler.bor_unit_cost`) is
  elements^2 x modes instead of elements^3 x modes: the ogive took 5.7x longer
  at 10 GHz than at 4 GHz, which the new model puts at 7.2x and the old one at
  16x (it orders bin packing only).

## Not done

- **Removing the tables path** and **taking compressed BoR out of the
  automatic fallback**: many plans and tests rely on the tables path, and with
  the spill now bounded, compression is only chosen when even a spilled
  streamed plan does not fit, where it is the remaining way to finish.
- **A single near check level** (the coarse level is 40 % of the native
  near-rule work): the coarse/fine comparison is what certifies each point, so
  it was kept.
- **Concurrent multithreaded BLAS elsewhere** (fix 12): mode workers still run
  their factorizations and products on their share of the cores at the same
  time, and the compressed operator's product threads widen BLAS by design.
  Those are few, large calls and never faulted here, but they are the same
  OpenBLAS pattern; a per-thread BLAS limit (this OpenBLAS build exports
  none) or a fixed OpenBLAS would close it.
- **A fused native far kernel** (sample, project and contract per tile) and a
  **hierarchical hp basis for 2D** (the P2 system as the leading block of P3):
  large redesigns, left for separate work; the measured breakdowns above say
  where they would pay.
