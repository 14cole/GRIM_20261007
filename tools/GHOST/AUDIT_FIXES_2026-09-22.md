# GHOST audit fixes, 22 September 2026

This release implements the findings of the 22 September speed, memory and
accuracy audit of the BoR and 2D solvers. Every finding below was measured
before it was changed, and every change was verified against the previous code
(`tools/GHOST_backup_before_audit_fixes_20260922.zip` holds that version).

## What changes when you rerun

- **BoR**: results agree with the previous version to 1e-13 relative (7e-15
  single-threaded), with one intended exception: partial-coating, layered-patch
  and banded-coating solves move by 1-2e-3 relative because their junction
  quadrature was corrected (its error was 1.2e-3; it is now within 2e-5 of a
  refined reference).
- **2D**: results move where a defect or an under-resolved rule was fixed:
  - touching panel pairs use an accurate corner rule (changes up to 5e-4 of the
    peak on a mixed-material fixture, 7.8e-6 on the airfoil);
  - automatic panel counts no longer jump at integer boundaries (the airfoil
    at 1 GHz: 1938 to 1926 panels, amplitudes change 7.9e-8);
  - reactive-impedance conductors get enough panels for their bound surface
    waves (more panels, much smaller errors: 12.7% to 0.24% on one case);
  - certification fine meshes use 1.5x the base panels per chain (was 2x per
    primitive); hp meshes are considered from 512 reference panels (was 1024);
  - certification compares dB and phase only above -30 dB of the peak;
  - samples are returned in request order; condition estimates are
    deterministic (a few differ from earlier, seed-dependent values);
  - geometries that used to be silently wrong are now rejected or corrected
    (split snapped vertices, wrong-winding voids, reversed cores in two-layer
    coatings, near-coincident vertex copies).
- **Drivers**: memory is detected correctly on Windows (31 GiB instead of 8 on
  this workstation), so local batch runs schedule more units at once; a
  crashed worker no longer ends the sweep.

## Verification

**Test suite.** `python ghost_backend/tests/run_suite.py` passes on Windows: 1,163 tests (1,044 headless and 119 in the 15 Qt modules, each in its own interpreter), one skipped, 530 subtests.

**Against the previous code.** Both versions were run alternately on a quiet
8-core workstation (Ryzen 9800X3D, 31 GB), two repetitions each; seconds:

| Case | Before | After | Largest result change |
| --- | --- | --- | --- |
| BoR PEC sphere `ka = 10`, 100 elements, CFIE, one worker | 32.0 | 8.9 | 2e-14 |
| BoR 120 x 5 in ogive survey, 1.5 GHz, 91 aspects, four workers | 23.8 | 6.4 | 2e-11 (deep nulls) |
| BoR PEC sphere `ka = 3`: CFIE tables / streamed / EFIE | 2.9 / 2.6 / 0.9 | 1.0 / 1.0 / 0.5 | 1.2e-14 |
| BoR impedance sphere, uniform / tapered | 4.2 / 4.2 | 1.5 / 1.5 | 1.9e-14 |
| BoR PEC cylinder `ka = 3` | 4.4 | 1.7 | 1.6e-14 |
| BoR lossy dielectric sphere | 6.7 | 2.6 | 2.6e-14 |
| BoR coated PEC sphere / coated two-layer | 9.6 / 4.6 | 3.7 / 1.8 | 5.8e-14 |
| BoR coated / partial / layered-patch / banded fixtures | 3.2 / 2.5 / 4.2 / 2.8 | 1.3 / 0.9 / 1.6 / 1.2 | 6e-15 / 1.2e-3 / 1.0e-3 / 2e-5 (junction fix) |
| 2D airfoil, 1 GHz, automatic run | 4.5 | 3.7 | 8e-6 |
| 2D circles and fixtures (11 cases, 0.2-0.5 s each) | | 7-34% faster | 3e-7 to 5e-5; 2.7e-3 mixed fixture (touching-pair rule) |

Result changes are the largest per-aspect relative difference (a null inflates
it). Repeated runs of the new code are bitwise identical, including the
four-worker survey. The large 2D gains are on specific paths: explicit
`direct` solves 39.5 s to 5.5 s, domains beyond about 163 wavelengths
45.7-49.3 s to 6.1 s, the separate W operator 12.5-14.1 s to 3.2 s.

**Against exact series** (final code; errors in dB, VV/HH or TE/TM):

| Case | 10 per wavelength | 20 | 40 |
| --- | --- | --- | --- |
| BoR PEC sphere `ka = 3`, CFIE | 0.090 / 0.086 | 0.020 / 0.020 | 0.0039 / 0.0037 |
| BoR PEC sphere `ka = 10`, CFIE | 0.011 / 0.011 | 0.0021 / 0.0021 | 0.0005 / 0.0005 |
| BoR impedance sphere `100+50j`, `ka = 3` | 0.24 / 0.23 | 0.056 / 0.053 | 0.013 / 0.012 |
| BoR dielectric sphere `4-0.1j`, `ka = 3` | 0.0033 / 0.0039 | 0.0006 / 0.0008 | 0.0003 / 0.0001 |
| 2D PEC cylinder `ka = 5` | 0.0015 / 0.0052 | 0.0008 / 0.0013 | 0.0003 / 0.0003 |
| 2D PEC cylinder `ka = 60` | 0.0033 / 0.0002 | 0.0004 / 0.0000 | 0.0001 / 0.0000 |
| 2D dielectric cylinder `4-0.2j`, `ka = 5` | 0.061 / 0.319 | 0.016 / 0.083 | 0.004 / 0.021 |

The BoR errors are identical to the previous code's; the 2D errors agree
with it within 0.0006 dB, so accuracy is unchanged while the defects listed
below are gone.

# Changes by area

## BoR solver core (`bor/solver.py`, `bor/factor.py`, `bor/near_parallel.py`)

**Speed**

| Finding | Fix | Verification |
| --- | --- | --- |
| Near preparation kept one task per worker and consumed results in order, so a slow pair idled every other worker. | `_iter_near_pairs` keeps `NEAR_SUBMIT_DEPTH` (4) tasks per worker in flight, still consumed in order; process tasks carry up to `NEAR_PROCESS_BATCH` (8) pairs per submission. | 480-element ogive at 2 GHz, 8 process workers: 11.7 s to 6.6 s (1.78x). 960 elements at 4 GHz, 15 workers: 20.8 s to 12.3 s (1.68x). Bitwise identical. |
| Table-path assembly contracted far tables through dense `Nn x P` basis matrices (two per surface, rebuilt lazily and retained). | Element-local contraction (`_efie_tables_into`, `_bracket_tables_into`, `_local_basis`), used by the EFIE, MFIE, IBC and cross-surface T/P blocks. The dense basis matrices are never formed on this path. | 1.6-1.8x faster table assembly at 300 elements, identical checksums; baseline cases match the original to 6e-15. |
| The mode sweep ran fixed waves (`executor.map`), so one slow mode idled the others and modes queued past convergence still ran. | A sliding window of `workers` modes, consumed in mode order, never crossing a streamed mode block while the previous block is pending; queued modes are cancelled after convergence; progress is reported per mode. | New scheduling test (no block crossing, bounded concurrency); results unchanged. |
| Each cross-surface pair was integrated twice (forward and reverse), and the partial-coating solver assembled some cross blocks two or three times per mode. | Reverse cross operators are derived exactly from the forward blocks by reciprocity (`_ReverseCrossOperators`); within a mode the forward block is shared with its reverse (`_cross_block`). Partial-coating impedance maps are applied as column scaling instead of a dense `(2N)^2` product. | Coated and coated-2 results match the original to 2e-15; multi-surface solves 1.3-2x faster on the test fixtures. |
| Material solves (partial, multi-region) solved one aspect at a time. | Batched excitation and far-field hooks; the sweep now applies the junction constraints itself (`assemble` returns the sparse transform). | New tests: batched equals per-aspect to 1e-12 (multi-region hooks and full partial-coating solve). |
| Weighted-PMCHWT jump terms built a dense `2N x 2N` rotation matrix every mode (and cached an `N x N` Gram). | The tridiagonal Gram is added through its bands (`_add_rotation_mass_into`). | Bitwise identical on all eight multi-surface cases. |
| 'Auto' RHS compression never paid on ogive-size systems but was retried on every mode. | The sweep shares one `CompressionHint` across its modes (backoff re-probing); evidence in `modal_execution.rhs_compression_hint`. | See the streaming section for the solve-stage numbers. |

**Memory**

| Finding | Fix | Verification |
| --- | --- | --- |
| The IBC/sheet CFIE closure held about eight full matrices while the gate priced three (out-of-memory risk). | The dual-IBC mixing, the IBC extra term and the MFIE part are accumulated in place (`_mix_dual_ibc_in_place`, `assemble_ibc_extra(out=, scale=)`). | IBC and tapered-IBC baselines match the original to 5e-15. |
| Junction constraint transforms were dense (`n_full x n_reduced` per category, 6.4 GB at 10k unknowns) and priced that way. | Sparse transforms in every backend with a gather-based reduction (`_ConstraintEntries`, `_reduce_constrained_operator`); priced as sparse. Signed-mode symmetry never builds `m = -1`. | Storage-model tests updated; junction solvers unchanged apart from the quadrature fix below. |
| A mode worker held its system matrix and the LU copy (two dense matrices). | `ModalFactor(owned=True)` factors large systems in place and spools the original coefficients to disk for the residual checks (policy `dense_residual_storage`, 'auto' at 512 MiB and up, the same policy as the 2-D factor; resolved in the calling thread because executor threads do not inherit it). | New tests: in-place factor (C and F order), residuals 1e-15, condition estimate, spooled vs in-memory BoR solve identical to 1e-12. |
| Single-precision far tables were built in double and converted, peaking at three times the retained table; nothing priced that peak. | Reduced-precision tables are built in 256 MiB row blocks (`_tables_by_rows`), for the self EFIE/MFIE/IBC and cross tables. | Blocked equals whole-table build exactly (3e-15); single vs double 4e-8. |
| Material solvers left streamed far blocks (GBs, possibly spill files) to garbage collection. | Streams are closed in `finally` blocks in the dielectric, coated, partial and multi-region solvers. | Streaming equivalence tests pass. |
| Reverse cross operators were stored and priced. | Derived reverses are excluded from the storage model and the streaming plans. | Storage tests updated. |

**Accuracy and errors**

| Finding | Fix | Verification |
| --- | --- | --- |
| Cross-surface element pairs that touch at a junction used the order-4, depth-4 corner cells: 1.2e-3 relative far-field error at partial, layered and banded junctions (against an order-12, depth-8 reference). | `JUNCTION_CELL_ORDER = 10`, `JUNCTION_CELL_DEPTH = 7` for those pairs, in the serial and process paths. | Error vs reference 4e-6 to 1.6e-5. A vacuum coating (which must be invisible) vs the exact PEC sphere: neutral at ka = 1, better at ka = 3 (0.0095 dB, equal to the plain PEC mesh). Junction-solver results move by 1-2e-3; everything else is unchanged. |
| Negative-mode Bessel triplets used an upward recurrence in the order (unstable). | `_bessel_triplet` evaluates `|m|` and maps the signs. | Baselines unchanged. |
| A single resistive element made a mostly lossless closed IBC body pass the EFIE resonance guard. | The reactive test is per value and weighted by area (`EFFECTIVELY_REACTIVE_AREA_FRACTION = 0.5`). | New behaviour covered by existing guards. |
| Stale MFIE/IBC tables could be reused for a larger mode cap. | The caches record their cap (`_K_tables_m_max`, `_KI_tables_m_max`). | Regression tests. |
| Near/far routing and the far sampling gap used one global scale for pairs at very different radii. | Per-pair routing threshold and a radius-scaled far gap (`_configure_near_pair_routing`, `_far_gap`). | Baselines unchanged (6e-15). |
| No run-time evidence of near-quadrature error. | `near_quadrature.efie_near_block_asymmetry_max` (reciprocity of the retained EFIE blocks, a free lower bound) with a warning above 1e-3. | Reported in results. |
| Banded coatings called `_solve_multiregion` directly, bypassing automatic mode-cap extension and admission fallbacks. | New configured entry `solve_bor_banded_multiregion`, used by dispatch. | Dispatch tests. |

## BoR kernels and native C (`bor/kernels.py`, `bor/native/`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Near angular projection: two trigonometric-moment calls per pair, poor loop order, projection outside C. | `trig_moments` restructured; new `parity_moments`; fused native `near_green_rule` and `near_brackets_rule` build nodes, sample and project with the GIL released. | 5.3-6.1x faster projection; bitwise equal to `trig_moments`, independent of the thread count. |
| `_checked_near_kernels` accepted a comparison when the tail order saturated at 4096 (spurious convergence): errors up to 1.1e-3 at d/a = 1e-8, and on a tiny-element self cell 526 of 920 points above tolerance (max 2.3e-2) with no error raised. | A comparison is accepted only when every order grew; brackets restart with the stable forms, otherwise the call raises. | Audit case: 1e-12. Tiny-element self cell: max error 2.2e-15. |
| The near rule's sample count grew without bound near the singularity (up to 25,903 samples per point). | Graded rule (sinh core, ratio-2 geometric panels to s = 0.25, one tail panel sized from its phase) for G, MFIE and IBC, with the coarse/fine check kept. | G <= 3.8e-14 against independent references for d/a 1e-9..0.5; brackets <= 4e-14; 110-640 samples per point; production self cells 1.5-1.8x fewer samples. |
| Bracket kernels lost accuracy by cancellation at small separations. | Native `near_brackets_stable` (C port of the stable forms), used by every native bracket point. | Agrees with the NumPy forms to 6.7e-16. |
| The far azimuthal grid was about twice oversampled. | Per-pair bandwidth rule in `banded_modal_kernels` (half grids including both ends, grouped by size); `n_xi_for_pairs` keeps its signature and never exceeds its old value. | <= 6.4e-14 against 16,384-sample references and the former production results; 0.23-0.50x the samples; streamed EFIE+MFIE far build 36.5 s to 13.6 s, table build 33.4 s to 9.2 s (one thread). |
| `physical_cpu_count` ignored affinity masks, SLURM allocations and the CPU budget. | Minimum of physical cores, affinity, SLURM allocation and `allocated_cpu_budget()`. | Tests. |
| Transform tables rebuilt per call; Gauss-Legendre cache too small; near/far chunk sizes not budgeted per sample. | 64 MB LRU of transform tables (no sine table on the G path); `cached_leggauss` holds 1024 orders; chunks budgeted per sample. | Included in the timings below. |
| A partially built DLL silently fell back to NumPy. | `build_kernel.py` requires all 13 entry points; a one-time notice reports a NumPy fallback. | Tests. |
| `mfie_for_mode` accepted tables of the wrong width. | Validates the width and `|m| <= m_max`. | Tests. |
| No reduced-precision output from the far builders. | `out_dtype` on `banded_modal_kernels`, `modal_kernels_fft`, `nonnegative_bracket_tables`. | Tests. |

End to end (alternating processes against the original code): near-heavy sphere ka = 10, 100 elements, one worker 25.6 s to 10.3 s (2.49x), four workers 12.8 s to 5.3 s; a tiny-element sphere 43.8 s to 11.6 s (3.79x); the reference solves 1.5-2.5x. Every amplitude matches the original to 6.7e-14; errors against Mie are unchanged to 1e-12 dB. New tests: `tests/test_audit_fixes_bor_kernels.py` (26).

## BoR streaming, spill and RHS compression (`bor/streaming.py`, `linalg/sweep.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Far-block tile threads were tied to the mode-worker count (one worker built on one thread), and each concurrent tile cost about 0.1 GB outside the priced tile budget. | Tiles run on `streaming_tile_threads()` (the allocation-aware physical cores); each tile is sized to budget / threads by a live-memory model of the sampler, splitting source columns when needed. | 4 GHz ogive CFIE far build: 43.6 s (one mode worker) to 11.9 s; measured tile memory 0.53-0.62 GB against a 1 GB budget; blocks within 1.7e-15 of the original. |
| One global lock serialized every accumulation into the streamed blocks. | Each tile sums locally and adds once per block family. | 17.3 s to 13.5 s at 8 threads (symmetry off), identical results. |
| The EFIE self blocks are exactly symmetric but both triangles were sampled. | Self streams sample only source >= test element pairs and complete the blocks per mode (`tt, ff` symmetric, `tf = U_tf - U_ft^T`), when the near relation is symmetric. | EFIE sampling halved (13.5 s to 11.9 s); results within 1.3e-14. |
| Spill files outlived crashes; a forced mmap close could invalidate live views. | Delete-on-close files (Windows `O_TEMPORARY`, POSIX preallocate-map-unlink), owner markers and `remove_stale_spill_directories`, tmpfs refused, low disk raises `StreamingSpillError` (an admission error, so automatic plans fall back). | Files vanish after normal close, failed builds and hard kills (verified on Windows). |
| The DLL search path added compiler/MSYS2 directories up front. | Only the kernel's own directory; toolchain directories only after a failed load. | Import-table test (KERNEL32 and the CRT only). |
| 'Auto' RHS compression paid a QR on every batch although it never helped on ogive-size systems (+25-55% solve stage). | A factor whose first attempt falls back solves directly; the QR is skipped when no batch can gain; ranks above half a batch are not reconstructed; a thread-safe `CompressionHint` carries the outcome across the modes of a sweep (backoff re-probing). | Ogive-size geometry, ten modes: 'off' 2.62 s, former 'auto' 3.42 s, new 'auto' with the hint 2.81 s; 19-mode sweep: 15 of 19 modes skip the futile QR, identical fields. A 2-D case where compression helps still gains (3.49 s to 3.29 s). |

Not done: triangle storage of the symmetric blocks (the stored-block views are full blocks). Linux-only spill paths are covered by mocked tests. New tests: `tests/test_audit_fixes_bor_streaming.py` (27).

## BoR drivers, HPC and dispatch (`hpc/`, `run_local_*.py`, `run_hpc_*.py`, `bor/dispatch.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| `detect_memory_gb()` returned the 8 GiB fallback on Windows (the local drivers scheduled against about 6 GB of a 31 GB machine). | psutil, then `GlobalMemoryStatusEx`, then sysconf before the fallback; GiB throughout (`decimal_gb_to_gib` for BoR estimates); local budget `min(0.75 x installed, 0.9 x available at start)`. | 31.1 GiB detected; 18.8 GiB schedulable in a test run. |
| A crashed worker (`BrokenProcessPool`) ended the whole sweep; a pool that broke while arguments were being sent hung forever on teardown. | `ExecutorPool.rebuild()`; units running at the break are retried once, alone; a second crash is recorded (`WorkerCrashError`); `WorkerPoolFailure` after 4 breaks without progress; manifests record "failed"/"interrupted"; exit 1. | Real-pool tests with `os._exit`; the teardown finishes in 0.3 s. |
| Stale spill directories from killed runs were never removed. | `sweep_stale_bor_spill()` at driver and HPC-worker start. | Integration check removed a stale directory. |
| Concurrent units could over-commit the spill disk. | Per-unit disk reservation (spill x 1.25, the solver's margin), re-read when idle; manifests carry `estimated_spill_gb`. | Tests. |
| BoR units had no per-unit CPU allocation. | `cpu_allocation_scope` per unit (cores / units that fit), CPU budget in the dispatcher; mode workers now inherit it (the solver runs each mode in a copy of the caller's context). | Tests. |
| Dispatch counted both directions of every cross operator. | One per surface pair for dense solves (both when compressed). | Coated 130/91 table estimate 3.641 to 3.192 GB; coated streaming preview equals the solver's plan exactly. |
| Banded coatings bypassed the configured entry. | `solve_bor_banded_multiregion`. | Bitwise identical results. |

Measured and not adopted: sizing units on physical cores (four 1.5 GHz ogive units 38-44 s logical vs 67-69 s physical). Round-5 planning tests recalibrated to the new storage model with every assertion kept. New tests: `tests/test_audit_fixes_drivers.py` (29).

## 2D operators and assembly (`twod/operators.py`, `twod/assembly/`, `twod/polynomial_quadrature.py`, `twod/fields.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Kernel tables were refused beyond about 163 wavelengths (a check of 2e-13 against a conditioning-limited error), sending large domains to SciPy Hankels (6x slower). | Tolerance `max(2e-13, 8 eps |k| r)`; tables capped at their 4096-interval budget (about 377 wavelengths), the far expansion beyond; no futile degree-16 retry. | Circle N = 2000 at R = 80 lambda: 45.7-49.3 s to 6.1-6.2 s (S 3.2e-15, K' 1.5e-13 vs SciPy); 100 and 150 lambda likewise. |
| The separate W operator used a slow independent path. | W-only pass of the fused engine behind the same signature. | N = 2000, one thread: 12.5-14.1 s to 3.2 s; bitwise equal to fused W. |
| Touching tolerances differed between code paths (a 3e-10 m gap raised `FloatingPointError`). | One 1e-9 m tolerance (the node-snap width) plus a same-node rule. | The case assembles; the change is the geometric perturbation (S 3.6e-8). |
| Touching pairs used Duffy batches with errors up to 6.3e-4 at reflex corners (O(h) quadrature error). | Graded `t^4` corner rule, 20 points per direction, corner-stable distances. | Per block 2.6e-13 collinear, 1.3e-10 (K') at a 170 degree corner; circle eigenvalues now clean O(h^2); mixed fixture 5.0e-4 to 3.0e-7 of peak from a converged reference; airfoil amplitudes change <= 7.8e-6 of peak. |
| The W far floor (10 points) was higher than needed. | 8 points for degree 1, 9 for degrees 2-3. | Worst W block 1.5e-14 to 4.6e-14; fused S/K'/D/W N = 2000 one thread 6.05-6.50 s to 4.26-4.59 s. |
| The near pass scattered pair by pair (164,860 calls on the airfoil). | Vectorized bookkeeping, one ordered scatter per output and route, batched Maue W blocks. | Bitwise identical outputs; airfoil 3 GHz, 4 threads: operators 9.5-10.6 s to 7.3 s. |
| The tile scatter was column-innermost and NumPy-bound. | Native `ghost_scatter_tile` (16-row blocks, strided views, NumPy-identical complex rounding), identity route. | Bitwise equal to NumPy on 300 layouts; a 2 x 256 x 256 tile 0.52 ms to 0.13-0.16 ms. |
| Graded far orders were not degree-aware; strongly lossy media had no separate table. | Degree-dependent orders, a separate table for `arg k < -45 deg`. | Every bin <= 1e-12 over arg k 0 to -89 deg; far block +3% on the airfoil. |
| A panel end hovering over another panel's interior used the far rule. | Adaptive near rule within 0.5 L. | Worst K' 1.3e-8 to 7.4e-12, S 4.7e-10 to 2.2e-13. |
| Self blocks with `|k l| > 8` fell back to a generic Duffy rule (4.4e-2 at 8.1, O(1) from 40). | Sub-interval construction. | <= 1.5e-14 for `|k l|` 8.1-160. |
| Lossy near-batch tables lived in an unbounded module cache. | Held by the CPU state's table store (32 MiB) or the solve's scope. | Released with the scope. |
| Far-tile scatter order depended on thread timing (runs differed by 1.4-1.7e-12). | Computed on threads, scattered in fixed tile order. | 1 vs 4 threads bitwise identical. |
| Monostatic batches rebuilt degree-2/3 plane-wave loads for the far field. | Reused within the batch. | Amplitudes <= 2.2e-15; far-field stage 0.010-0.068 s to 0.001-0.003 s. |

Checked, no bug: the medium of a formulation is never picked from the normal direction in these files (only a diagnostic orientation-conflict refusal). Speed changes were verified bitwise (or to 3.4e-16 where two outputs share entries) with the former quadrature restored, over 34 cases. New tests: `tests/test_audit_fixes_2d_operators.py` (22).

## 2D solver, quality and execution (`twod/solver.py`, `runs/quality.py`, `execution/`)

| Finding | Fix | Verification |
| --- | --- | --- |
| Warnings accumulated across frequencies in the shared material library and tripped the strict warning gate (a certified 8-frequency run failed with 11 warnings). | Per-solve notices (`_SolveNotices`); the library keeps the run-level union; the large-memory note is information. | Both failing runs complete with 0 warnings per frequency. |
| Certification compared dB and phase at deep nulls, so converged patterns failed (1.43 dB / 12.3 deg on a PEC plate whose complex change was 2.2e-4). | dB compared on 20 log10(|A| + 3e-2 peak); phase only where both meshes exceed 3e-2 of the peak; policy field `db_floor_relative`; unfloored values still reported. | Plate passes (0.053 dB, 0.2 deg); unconverged synthetic patterns still fail. |
| Explicit `direct`, mixed-precision automatic runs, bistatic mixed precision and boundary densities bypassed the kernel tables. | Kernel-only CPU state for those paths (mixed precision still factors in mixed precision). | 1440-panel circle: `direct` 39.5 s to 5.5 s; mixed 40.0 s to 6.7 s; densities 3.15 s to 0.43 s; results <= 3.2e-13 from the original. |
| Condition estimates depended on the global random state. | Fixed private seed under a lock; the caller's RNG state is restored. | Lossless dielectric case: always 296.5 (was 151.5 or 296.5). |
| Merged metadata dropped NaN/inf. | NaN and inf propagate; complex metrics included in the merged mesh summary. | Tests. |
| cgroup limit readable but usage unreadable returned 0 available (every solve refused); SLURM subtracted only this process. | Limit minus own resident memory; SLURM subtracts every local process in the allocation; 2D memory values are GiB throughout. | Tests. |
| Samples were returned sorted by polarization. | Request order, VV before HH per sample. | Consumers look samples up by coordinates. |
| Desktop 2D runs used four assembly threads. | Automatic profile `assembly_threads='auto'` (physical cores, at most 16, bounded by a scheduler allocation). | Airfoil 3 GHz: operators 11.5-11.9 s to 7.9-8.6 s; results 1.3e-12. |
| The operator cache key missed far-quadrature and tile settings. | Included. | Tests. |

New tests: `tests/test_audit_fixes_2d_solver.py` (24).

## 2D geometry and validation (`twod/geometry.py`, `twod/adaptive_geometry.py`, `geometry/io.py`)

| Finding | Fix | Verification |
| --- | --- | --- |
| F1: a primitive's last point was recomputed as `p0 + (p1 - p0) * 1.0`, splitting snapped vertices (13 nodes for 12 panels, open contours, TM silently unprotected). | Exact end points; tolerant end checks in junction grading. | 0 split vertices in 120,000 trials; closed loops. |
| F2: `_point_in_polygon` skipped the closing edge (wrong-winding voids accepted, 2.7-6.3 dB errors). | Closing edge included. | Wrong void rejected, valid one accepted. |
| F3: nesting depth counted sheet contours. | Only TYPE 2/3 contours count (also for stitched open chains). | CW/CCW cases inside sheet shells. |
| F4: near-coincident copies of a vertex (2.5e-10 m) produced separate nodes. | `_NodeWelder` (1e-9 m) in both mesh builders, loops, sheets, junction lookup, validation and `io.py` chain matching. | 16 nodes / 16 panels, fields within 8.9e-8 of the exact geometry. |
| F5: primitives or explicit panels of 1e-9 m and below were accepted. | Rejected with a clear message. | Tests. |
| F6: the crack scan refused 2e-9..1e-6 m stubs and notches. | Works on welded vertices with a local tolerance cap. | Valid without warnings; a real 1e-7 m crack still raises. |
| F7: bound surface waves on reactive impedance conductors were under-meshed (12.7% field error on a 1-lambda square, TM -60j). | Per-segment bound-wave mesh wavelength, capped at 4x density; strongly damped or very tightly bound waves left alone (measured). | 12.7% to 0.24% (TM -60j), 10.8% to 0.93% (TE +2000j); circle TE +600j vs exact series +0.44 dB to +0.007 dB. |
| F8: panel counts jumped at integer boundaries (128 to 256 panels between 3.05 and 3.06 GHz). | `ceil(L/h - 0.05)`. | 128/128/128 panels; airfoil 1938 to 1926 panels, amplitudes 7.9e-8. |
| Certification fine meshes doubled every primitive. | Each chain gets `max(B+1, ceil(1.5 B))` panels, longest first, never below the base; mirror-image primitives are refined together, so symmetric bodies keep symmetric fine meshes. | 256-gon certified 2.4 s to 1.1 s, published error 0.0381 to 0.0382 dB; 200 random symmetric chains all stay symmetric (a one-at-a-time tie-break broke 85 of them). |
| F9: hp-mesh eligibility used the P1 panel count. | Predicted hp size; `MIN_AUTOMATIC_REFERENCE_PANELS` 1024 to 512 (measured). | 64-gon: hp 1.1 s vs P1 3.1 s; 1024-gon refused. |
| F10: O(n^2) orientation checks. | Bounding-box prefilter. | 1,600 8-gons: validation 1.87 s to 0.23 s. |
| F11: the explicit-N floor used the global minimum wavelength. | The segment's own material. | PEC square with a distant eps = 100 rod solves. |
| F12: material table ends rejected values one ulp outside. | Clamped within 1e-12. | `sample(1.2000000000000002)` works. |
| F13: no warning for faceting error. | Information advisory when the sagitta exceeds lambda/200. | 32-gon at ka 30 flagged, 128-gon not. |
| S2: a reversed core inside a two-layer coating was accepted (17-31 dB wrong). | Checked against the innermost enclosing contour. | Rejected. |
| S3: a stray unused `neg_mat` flag split a contour's nodes (13-58% field change). | Node signature ignores flags a TYPE does not use. | Identical to the clean geometry. |

26 regression solves are bitwise identical to the original (one differs by 7.3e-13, the F1 end-point move). New tests: `tests/test_audit_fixes_2d_geometry.py` (33).

## Investigated and deliberately not changed

- **BoR self-element quadrature.** The graded cell rule misses the model
  integral of `log|s - s'|` by 8e-3 (about 6e-4 relative far field at 27
  elements per wavelength). A rule graded in `|s - s'|` removes that error
  (2-3e-6) with a third of the points, but against exact series for PEC,
  impedance and dielectric spheres (ka 1-10, 10-80 elements per wavelength) it
  was not more accurate end to end at practical densities and cost about 10%
  more: the cell rule's error partly offsets the flat-segment geometry error.
  The cell rule stays; the evidence is in `NUMERICAL_METHODS.md`.
- **BoR memory units.** BoR estimates are decimal GB while the solve limit is
  GiB, which makes BoR admission 7% conservative. Correcting it would shift
  every calibrated admission threshold, so it was left as a safe margin.
- **Physical-core sizing of local BoR units.** Measured slower (four 1.5 GHz
  ogive units: 38-44 s with logical-CPU sizing, 67-69 s with physical cores).
- **Triangle storage of the symmetric streamed EFIE blocks.** The sampling is
  halved; storage is not, because streamed blocks are handed out as full views.
- **2D batch drivers on `multiprocessing.Pool`.** A hard worker crash there
  still loses its unit and the sweep waits. Moving them to the BoR drivers'
  crash-tolerant `ExecutorPool` would change 2D process semantics (non-daemonic
  workers, spawn on Linux), so it was left for a separate decision.
- **2D junction panel matching (S1)** and **2D medium selection from normals**
  (item 14): checked, no defect (effects at most 0.002 percentage points; no
  formulation picks a medium from a normal).

## Known limitations

- The BoR junction formulation needs a moderate mesh: a vacuum coating (which
  must be invisible) is 1.5 dB off at about 8 elements per wavelength
  (ka = 3, 6 elements per hemisphere) but converges normally from about 16
  (0.034 dB) and matches the plain PEC mesh at 24.
- 2D bound surface waves tighter than about TM `|X| < 31` ohm or TE
  `X > 4.5` kohm are not resolved at the default density; mesh certification
  must catch them.
- With `m_max` between about 1030 and 1360 and a far pair right at the
  direct-routing gap, the new BoR far rule can exceed its 8192-sample cap
  (a clear error, not a wrong result); not seen in a realistic case.
- An HPC submission decides spilling from the login node's temporary
  directory; an automatic mode-cap extension can spill more than reserved.
- The Linux spill paths (tmpfs refusal, preallocation) are covered by mocked
  tests only.

## Tests

New modules (161 tests): `test_audit_fixes_bor_kernels.py` (26),
`test_audit_fixes_bor_streaming.py` (27), `test_audit_fixes_drivers.py` (29),
`test_audit_fixes_2d_operators.py` (22), `test_audit_fixes_2d_solver.py` (24),
`test_audit_fixes_2d_geometry.py` (33); most fail on the previous code. New
tests in existing modules: in-place factorization with spooled residuals and a
spooled BoR solve (`test_bor_memory_planning`), mode workers inheriting the
caller's execution options, the sliding mode window never crossing a streamed
block, batched multi-region and partial-coating hooks against the per-aspect
path, the sparse constraint transforms, and symmetric certification fine meshes.

Existing tests updated because the intended behaviour changed (each keeps its
intent): streamed-cross and storage-model expectations (reverse cross operators
are derived, one per surface pair), sparse junction-projection pricing, the
round-5 planning thresholds recalibrated to the new storage model, 2D
certification fine-mesh counts (178 to 176 nodes, 28,000 to 21,000 panels), the
hp crossover case (-1400 to -1000 panels per wavelength), the automatic
thread profile (`assembly_threads='auto'`), and the cgroup memory fallback.
