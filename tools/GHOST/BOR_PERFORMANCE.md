# BOR computation and controls

The default `factorization='auto'` compares a resource estimate with the solve
memory limit. Geometry-snapshot planning checks every requested frequency,
including material dispersion and the certification refinement factor. Direct
generatrix APIs price the plan the receiving solver will run (described below);
their detailed resource admission remains the authority. Invalid geometry and
settings propagate as errors.

Requested mode workers are an upper bound. The preview and runtime select the
largest concurrency that fits the memory limit, including operator storage,
factorization workspaces, near-integration scratch and safety margins. If one
worker still cannot fit, the solve rejects before preparing operators. The
desktop preview resolves Auto across the complete frequency sweep and reports
the selected backend and worker range. Each frequency's `modal_execution`
records its admitted `worker_plan`; the precise runtime plan uses the actual
near-pair lists and can admit more workers than the conservative preview.

`compressed_storage_mib=0` resolves to the automatic storage budget, divided
among active workers. The estimator and factor construction use the same cap.
The cap covers retained compressed payload; FFT, factor construction, and RHS
workspaces also need memory. Estimates are reservations, not measured RSS.

`table_precision='auto'` retains double precision. Single precision requires
an explicit `single` setting and cannot be used with compressed assembly.
Residual and condition checks do not certify quadrature/coefficient accuracy.

Far-kernel FFT samples are grouped by point-pair separation and radius. Each
group retains the existing modal bandwidth, oscillation, and peak-resolution
criteria. Singular and near pairs use separate integration. Table and streamed
paths share the grouped sampler. Near blocks are consumed from a bounded queue
instead of retaining a duplicate list of all integrated values.

Retained self and cross near-field blocks contain only modes `0..m_max`.
Negative modes are reconstructed with exact even/odd tangential-block parity,
without conjugation, before the PMCHWT row rotation. This nearly halves the
near-coefficient cache in dense, streaming and compressed solves. Meridian
quadrature, double precision, material properties and mesh density are unchanged.

Near preparation chooses concurrency from the requested workers, CPU allocation
and remaining solve memory. Preview and runtime reserve 192 MiB of scratch per
concurrent task, plus 256 MiB per process worker when process workers will
actually be selected, in addition to retained operators and mode-solve
workspaces. Planner and executor share one policy: an explicit `processes`
request, or an automatic job whose largest preparation call reaches 8,000
(near pairs) x (prepared modes). Preview knows those geometric pair counts; a
48-element sphere that stays on threads is again priced at 2.7 GB instead of
5.3 GB. When the workload is unknown while planning, the thread reservation is
kept and `process_workers` records the pool size verified to fit including the
overhead, which bounds the executor. Threads and processes are priced as
alternatives, so a started pool serves every later call of that preparation,
small ones included: its idle workers stay resident until the scope ends and a
thread team sized without them would exceed the admitted plan.

Direct already-meshed calls with `factorization='auto'` first price the dense
plan with the same table/streamed decision as the solver that receives the call
(`solve_bor`, `solve_bor_dielectric` and `solve_bor_coated_pec` decide on their
far tables alone; the junction planner adds its retained auxiliaries). Those
solvers choose tables below a fixed 2 GB whatever the memory limit is, so when
the caller left `assembly='auto'` and that plan cannot fit, dense streaming is
priced next and compression is selected only when neither fits. Plans are
ordered by speed (tables, streaming, compression), not by size: fixed
workspaces make a streamed or compressed plan the larger one on a small body.
For a coated sphere (0.1/0.07 m, 130/91 elements, `ka=12`, one worker) under a
3.3 GB limit the dense tables are rejected (3.51 GB) and the streamed plan
(2.23 GB) now runs in 138 s; compression, the former fallback, took 1,020 s, and
dense tables under a 4 GB limit take 142 s, all within 0.0084 dB of the exact
series. The solver's own admission, which runs before any operator
preparation, remains the authority. It raises `BorAdmissionError` (a
`MemoryError`); an automatic plan it rejects is replaced by the next one, and
the result records `automatic_factorization_fallback` (every rejected plan with
its requirement) and `automatic_assembly` when streaming was imposed. Only that
rejection changes the plan: an allocation failure while a solve runs
propagates, and an explicit `factorization` is never replaced. An explicit
`assembly` is never exchanged for the other dense assembly (streaming is imposed
only on `assembly='auto'`), but, as it always has, `factorization='auto'` still
sends a rejected explicit-assembly plan to compression, which has its own
assembly; request `factorization='dense'` to keep the rejection instead. A
single-precision request, in any spelling the solvers accept, never reaches
compression (which needs double precision) and keeps its memory diagnostic, and
a plan that follows an automatic mode-cap extension starts from the extended
cap. When no plan fits, the error reports the smallest requirement among the
plans priced at the final mode cap and names that plan (a requirement priced
before a later extension is obsolete: that cap is known not to converge); the
other rejections are chained with their caps.

Snapshot entries (`solve_monostatic_rcs_bor`, its certified and survey forms, and
`estimate_bor_resources`) choose once for the whole sweep, from the preview of
every frequency: the caller's assembly; then, for `assembly='auto'`, dense
streaming imposed on every frequency; then compression. A single-precision sweep
is priced the same way but has no compressed plan: when neither dense plan fits
the preview, the smaller one is left to the solve's admission, whose rejection
is the memory diagnostic. The same sphere through
the snapshot entry under a 3.3 GB limit went from 1,017 s (compression) to 136 s
with identical RCS. Preview and solve resolve the same plan, so an automatic
preview prices what will run. The run-setup summary resolves it itself, under
the caller's execution options (they size the aspect batches that the plan
prices), and shows `dense (streamed far blocks)` when streaming is imposed.
Results record `automatic_assembly` and keep the
caller's own `assembly_requested`. These entries have no run-time fallback: a
rejection by the solver's admission still propagates. The preview prices coated
tables far more cautiously than the run-time gate (8.59 GB against 3.51 GB for
this sphere at cap 20; the measured process peak was 2.04 GB, and 0.33 GB
streamed), so streaming is imposed over a wider range of limits here than on
direct calls. Since round 10 the preview, the direct chooser and every run-time
gate price a dense table plan with one model (`_table_plan_peak_gb` mirrors
`estimate_bor_operator_storage_gb` from mesh counts): 1.10 x the retained
tables and basis matrices plus the 256 MB that bounds the banded FFT build. The
blanket 3.5 x tables it replaces dated from an unbounded table build: measured
process peaks at cap 20 are 1.34 GB for 1.09 GB of CFIE tables (the gate asked
5.32 GB, now 2.52), 2.21 GB for 1.95 GB with an impedance (8.93, now 3.67),
2.45 GB for a dielectric sphere's 2.17 GB and 2.04 GB for the coated sphere's
1.78 GB. The preview is now 4 to 8 % above the gate for table plans (40 % for
the small partial-coating fixture, whose fixed auxiliaries dominate) instead of
up to three times, and 0.3 to 13 % above it for streamed plans on those
spheres. Every gate still exceeds the measured peak by at least 1.5 times
(safety factor 1.2 and 0.5 GB margin included). The coated sphere above is now
previewed at 3.76 GB for tables, so compression or streaming is chosen below
that limit instead of below 8.59 GB.

Snapshot entries have no fallback, so the preview must not fall below the gate
it predicts. Round 10 claimed it never did; Codex's review disproved that with
a closed corrugated impedance conductor (201 panels, 28,233 geometrically close
pairs, tables just under the solvers' 2 GB rule). The previews assumed two
near-operator families per surface, and a CFIE with a nonzero impedance retains
three (EFIE, MFIE and the impedance operator): the table preview was 0.038 GB
below its gate, the streamed one 0.19 GB and the compressed one 0.20 GB, and an
automatic call between 4.239 and 4.277 GB imposed tables and was rejected
although its streamed plan needs 2.92 GB. The allowance removed in round 10
had hidden this; the streamed and compressed previews were wrong before it.
`solve_bor` now states its families once (`conductor_operator_kinds`) for its
preparation, its gate and both previews, and gate and preview price a family
with the same functions (`bor_near_cache_bytes`, `bor_cross_near_cache_bytes`).
For a conductor the preview then exceeds the gate by construction: by 1.2 x
(the 128 MB near workspace it counts on top of the FFT workspace, plus the
junction projection it assumes) for tables, 0.15 to 0.17 GB on the bodies
compared, and by that projection alone for streamed and compressed plans (0.1
to 20 MB): spheres, finned and corrugated bodies, PEC and impedance, caps 20 to
80. Material kinds already counted two families per medium side, which is what
their solvers prepare, and stay 0.003 to 0.93 GB above their gates on crowded
bodies (finned dielectric, 1 and 2 mm coating gaps, a coated finned body, the
partial-coating fixture).

Since round 12 nothing is mirrored by hand. `bor_operator_storage_bytes` is the
one storage model: tables, basis matrices, near contractions, junction
projections and the two build workspaces, from plain records
(`BorSurfaceStorage`, `BorCrossStorage`). The run-time gate fills the records
from its constructed solvers; the previews fill them from mesh counts
(`dispatch._layout_storage`): two medium sides per interface, every directed
surface pair a cross operator, the larger of the geometric near-pair count and
a stencil floor, every unknown a junction constraint, the FFT workspace at its
bound. The model is monotone in every field, so records that bound the solvers'
own bound the gate; the refactor reproduces every earlier preview and gate
number. Two inputs the previews had guessed are now supplied: the dense
impedance maps of the bare pieces of a partial coating (the gate's
`extra_retained_gb`), and, for direct calls, the geometric near-pair counts of
the supplied generatrices (the chooser used the stencil floor alone and priced
corrugated conductors 0.11 to 0.57 GB below their gates). Layered, N-layer and
banded snapshots, never compared before, are 0.09 to 0.63 GB above their gates,
and 0.16 to 1.38 GB with 1 to 2 mm layers, where cross-surface near pairs
dominate.

The separate BoR execution option
`near_backend` accepts `auto`, `threads`, or `processes`. Process preparation
uses bounded ordered results and reuses one pool across the preparation phase.
Auto keeps small jobs on threads to avoid spawn/import costs. Entry scripts
need the standard `if __name__ == '__main__':` guard. Unguarded scripts,
frozen executables and interactive hosts use threads.
`near_preparation` records that plan. Direct private preparation without a solve
plan retains the conservative 1 GiB scratch cap. Paired native sampling runs
within these workers without creating an additional OpenMP team per task.

MFIE and IBC angular projection uses exact parity: tt/ff components are even
and tf/ft components are odd, including complex wavenumbers. Only the positive
quadrature grid is sampled. The native paired sampler supports both families
and complex media; missing/newer entry points fall back to NumPy. Rebuild with
`python ghost_backend/bor/native/build_kernel.py` on platforms without the
updated binary. Meridian rules, angular convergence checks and streamed tiling
are unchanged.

Automatic mode selection starts with a margin
`max(12, ceil(4.05*x^(1/3)+2))` above the incident
bandwidth `x`, retaining the existing axial-look floor. This is a starting-cap
heuristic, not a tail-convergence certificate. If it still reaches the cap it
can expand the cap twice, repeating resource admission and preparation. Because
preparation cost grows with the cap, the first extension is sized from the
measured tail (`log2(tail/tolerance) + 8` further modes, at least 12, never
more than the former 1.5x rule, which remains the second attempt). The
completed complex field sums, tail history and linear diagnostics are retained;
only new modes are solved. Operator preparation uses the enlarged cap so new
angular coefficients have the proper quadrature. The chosen extensions are
recorded in `automatic_mode_cap_extensions`. Explicit
`n_modes` values are honored. Every path still rejects an unconverged tail.

Streamed EFIE storage contracts the nine quadrature terms into four final
nodal blocks per nonnegative mode. MFIE/IBC far tables and streamed blocks
also retain only nonnegative modes, with exact parity for negative modes.
Compressed tiles pass the near mask to grouped FFT sampling and index nearby
contractions by observer row. Backend metadata distinguishes NumPy Green
sampling with native brackets from a fully native sampler.

RAM admission includes full accumulated/contribution field arrays and the
frequency/aspect output grid, even when RHS solves use bounded batches.
BLAS limits are selected after mode-worker admission and share the allocated
CPU budget among the workers actually used.

## September 22 speed changes (conductor solves)

A 120 in x 5 in PEC ogive from 1 to 10 GHz motivated a speed audit of the
BoR path. Every change below preserves the quadrature rules, meshes and
precision: eight regression cases (far tables, streamed far blocks, CFIE,
EFIE and IBC spheres, a lossy dielectric sphere, a coated PEC sphere) agree
with the former code to 1e-14 relative in the complex amplitudes, and the
BoR test modules pass. Measured on an 8-core workstation with 31 GB (181
aspects): 4 GHz survey 130 s to 71 s, 6 GHz survey 359 s to 159 s, 4 GHz
certified 517 s to 204 s, and the certified 10 GHz solve of the ogive
(2,040 and 3,120 elements, 23 of 26 modes, 33.7 GB of far blocks spilled)
14 minutes, where the former streamed path was extrapolated at 75 to 90
minutes and the compressed backend it resolved to at about 13 hours.

Backend selection was the largest cost. The dense memory model charged eight
matrix equivalents per mode worker (5.4 GB for one worker at 6,242 unknowns)
on top of the 8 GB stream budget, so the automatic plan of the certified
1-10 GHz sweep resolved to the compressed backend on that workstation.
Compressed assembly re-samples the far field for every mode through
four-node tiles; the identical 2 GHz solve took 409 s compressed against
31 s streamed. The conductor assembly now forms the CFIE combination in one
buffer (`assemble_mode(out=, scale=)`, `assemble_mfie_mode(out=, scale=,
accumulate=)`, and `reduce_pole_operator`, an O(n) form of the `|m| = 1`
pole reduction), so a mode holds its matrix and the LAPACK copy and the
model is `BOR_DENSE_MATRIX_EQUIVALENTS = 3`. Near preparation is priced per
phase: its process pool is shut down before any factorization, so the gate
requires the larger of the preparation phase (retained operators plus near
scratch) and the mode phase (retained operators plus linear workspaces), not
their sum (`plan_near_preparation`, `_guard_bor_dense_memory` with
`preparation_peak_gb`). The same sweep now resolves to dense streaming with
2 mode workers and 15 near-preparation workers admitted.

The grouped far sampler (`banded_modal_kernels`) samples only the half grid
`xi` in `[-pi, 0]`: the Green's function is even in the azimuth offset and
the bracket components have exact parity, so the periodic trapezoid sums
are real cosine and sine transforms of those samples, formed as one GEMM per
bracket for the requested orders (the full-length FFT computed every bin to
keep about thirty). Sampling runs in the native paired kernels
`sample_g_pairs` and `sample_brackets_pairs` (real or complex wavenumber,
OpenMP team sized to the physical cores left by the tile threads); the
NumPy forms remain the reference and the fallback, and the backend name is
`banded_native_pairs`. Per point pair at 10 GHz, single-threaded: 7.7 to
1.8 microseconds for the Green's function and 13.8 to 3.7 for the MFIE
brackets; 0.56 and 1.55 on eight OpenMP threads. Table builds use every
physical core; tile threads beyond the physical cores measured no gain and
are capped.

A stream budget short of every mode used to rebuild the far blocks once per
mode range, and every range re-samples every far pair (two ranges on the
ogive's base mesh at 10 GHz, three to five on the certified mesh). With the
BoR option `stream_spill='auto'` (the default; `'off'` restores the
per-range rebuild) a conductor solve accumulates every mode in one sweep
into memory-mapped files under the temporary directory when it can hold
them with a 1.25 x margin (`plan_stream_spill`); the resident cost is the
two modes being read, previews price the same decision, results record
`stream_spill_gb`, and the files are removed when the solve ends. The
material solvers keep their per-range rebuild.

Near preparation projects each pair's angular samples onto the modes with
the native recurrence `trig_moments` (one multiply-add per sample and order
instead of a trigonometric table; bit-level agreement to 2e-15), halving
the Green's-function rule. The local driver runs its units in
`concurrent.futures` process workers (`_ExecutorPool`), which are not
daemonic, so a unit's near preparation keeps its process pool
(`process_capable` requires a non-daemonic process and a guarded entry
module); `multiprocessing.Pool` workers could not start children and fell
back to GIL-bound threads (2.3 x on eight threads). The excitation and the
far-field projection of one aspect batch share their Bessel functions and
axial phases (`_angular_batch`), halving that stage. The streamed-block
contraction is a test-side GEMM over the tile's Gauss rows followed by a
batched source-side product (`_contract_test_side`,
`_contract_source_group`): 209 ms to 46 ms per element on the 3,055-element,
27-mode tile of the certified 10 GHz ogive, with no per-term copies of the
sampled tile.

## September 22 audit fixes

A second audit on 22 September measured the remaining costs and errors. Apart
from the junction quadrature (see [numerical methods](NUMERICAL_METHODS.md)),
which moves partial, layered-patch and banded results by 1-2e-3, every change
below keeps the discretization: the BoR regression cases (far tables, streamed
blocks, CFIE/EFIE/IBC/tapered-IBC spheres, a cylinder, a lossy dielectric
sphere, a coated sphere, coated-2 layers and the ogive survey) agree with the
former code to 6e-14 relative (the four-worker ogive survey to 2e-11 at its
deepest nulls), and repeated runs are now bitwise identical (the survey used
to vary by 1.7e-12 between runs).

End to end, alternating the former and the new code on a quiet 8-core
workstation (two repetitions each, seconds):

| Case | Before | After |
| --- | --- | --- |
| PEC sphere `ka = 10`, 100 elements, CFIE, one worker | 32.0 | 8.9 |
| 120 x 5 in ogive survey, 1.5 GHz, 91 aspects, four workers | 23.8 | 6.4 |
| PEC sphere `ka = 3`: CFIE tables / streamed / EFIE | 2.9 / 2.6 / 0.9 | 1.0 / 1.0 / 0.5 |
| Impedance sphere, uniform / tapered | 4.2 / 4.2 | 1.5 / 1.5 |
| PEC cylinder `ka = 3` | 4.4 | 1.7 |
| Lossy dielectric sphere | 6.7 | 2.6 |
| Coated PEC sphere | 9.6 | 3.7 |
| Coated / partial / layered-patch / banded / two-layer fixtures | 3.2 / 2.5 / 4.2 / 2.8 / 4.6 | 1.3 / 0.9 / 1.6 / 1.2 / 1.8 |

Near preparation keeps four tasks per worker in flight (it kept one, and a
slow pair idled every other worker): 1.68-1.78x on 8 and 15 process workers,
bitwise identical. The near angular rule is graded and runs natively
(`near_green_rule`, `near_brackets_rule`, `parity_moments`): 110 to 640 samples
per point instead of up to 25,903, and a near-heavy sphere (`ka = 10`, 100
elements, one worker) takes 10.3 s instead of 25.6 s. Far pairs are sampled by
a per-pair bandwidth rule (0.23 to 0.50 of the former sample counts, agreement
6.4e-14): one-thread table builds 33.4 s to 9.2 s, streamed EFIE+MFIE builds
36.5 s to 13.6 s. Streamed tiles use every allocated physical core whatever
the number of mode workers, each tile is sized to its share of the tile budget
by a live-memory model, accumulation takes one lock per tile and block family,
and the exactly symmetric EFIE self blocks sample one triangle: the 4 GHz ogive
CFIE far build went from 43.6 s (one mode worker) to 11.9 s.

Table-path assembly contracts the far tables element by element instead of
through dense `Nn x P` basis matrices (1.6-1.8x, identical results). Reverse
cross-surface operators are derived from the forward ones by reciprocity, and a
forward block is shared with its reverse within a mode, so each surface pair is
integrated and assembled once. The mode sweep keeps a sliding window of mode
workers instead of fixed waves and cancels queued modes after convergence.
'Auto' RHS compression stops attempting the QR on a factor whose first attempt
fell back, and one `CompressionHint` per sweep carries that outcome to later
modes (`modal_execution.rhs_compression_hint`): on an ogive-size body the
solve stage of ten modes took 2.81 s against 3.42 s before ('off': 2.62 s),
with identical fields. The partial-coating and
multi-region solvers use batched excitations and far fields.

Memory: the impedance and sheet CFIE assembly accumulates its dual, extra and
MFIE parts in one buffer (the former closure held about eight full matrices
while the gate priced three); junction constraint transforms are sparse in
every backend and priced that way; a mode worker keeps the original next to its
LU (the plan prices both) and factors a system of 512 MiB or more in place,
spooling the original coefficients for its residual checks, only when the
memory available at that moment cannot hold the copy (`dense_residual_storage`
'auto'; 'disk' always spools);
single-precision tables are built in 256 MiB row blocks (the whole-table
double build peaked at three times the retained table, unpriced); tridiagonal
Gram, jump and sheet-mass terms are added as bands; derived reverse operators
are neither stored nor priced; every material solver closes its streamed
blocks when it finishes or fails; spill files are deleted on close (and by the
drivers' stale-spill sweep after a hard kill). Mode workers run in a copy of
the caller's context, so the unit's CPU allocation and execution options reach
them.

## September 24 audit fixes

A third audit measured where the remaining time and memory went; the fixes
and their verification are listed in [the audit record](AUDIT_FIXES_2026-09-24.md).
On the 120 x 5 in ogive (181 aspects, same workstation, baseline and new code
alternated): the 4 GHz survey takes 15.5 s instead of 22.5 s, the 10 GHz
survey 108-121 s instead of 130 s, and the certified 10 GHz solve 401 s
instead of 715 s. The larger change is memory: those runs used to drive the
system's available memory to 8-10 MB; they now leave 14-15 GB free.

- Near-preparation (and compressed-tile) workers start with one BLAS thread:
  each used to commit about 1.05 GB of OpenBLAS buffers at import (15 workers,
  16 GB of commit on a 2 GB pagefile); now about 40 MB.
- Symmetric self-surface EFIE far blocks are stored as one packed triangle and
  completed per mode as they are copied into the system matrix; the former
  completion pass over the whole store paged a spilled store from disk (213 s
  of the certified 10 GHz solve). Spilled rows are written back and released
  as the build completes them, and a mode's blocks as they are read: the
  10 GHz spill peaked at 2 GB of working set instead of 16 GB.
- Far pairs, excitation and far-field projection use three Gauss points per
  element (`FAR_GAUSS_ORDER`), 2e-9 to 5e-7 relative from four points.
- The excitation keeps one record per aspect chunk for the whole sweep (the
  axial phase and two top Bessel orders) and derives each mode by the downward
  recurrence: two `jv` evaluations per value and mode were about 70 CPU-s of a
  certified 10 GHz sweep.
- The mode window stops at a predicted end of the tail, and modes still
  running past convergence stop at their next checkpoint.
- Near pairs are integrated in batches (one graded-rule call for a batch of
  pairs instead of several per pair) and contracted by matrix products.
- The native far and near samplers skip `exp(ki R)` for a real wavenumber;
  the contraction multiplies real basis weights by the real view of the
  complex samples (half the operations of the promoted complex product).
- A dense factor keeps its original in memory whenever the copy fits (the plan
  prices it); the disk spool is only a guard against memory the plan did not
  foresee.
- BLAS threads never exceed the physical cores (`blas_core_budget`): a single
  mode worker used to run OpenBLAS on every SMT thread, where LU is slower and
  products of fewer than about 256 rows take seconds instead of milliseconds.
- The dielectric, coated, partial-coating and multi-region solvers spill their
  far blocks as `solve_bor` does: when the stream budget cannot hold every
  mode, every stream is built once into memory-mapped files instead of once
  per mode range, and every mode worker runs (dielectric ogive: at 5 GHz 2 far
  builds instead of 4 in the same time with 3-4 GB less peak memory; at 6 GHz
  2 instead of 6, 258 s instead of 284 s, 10.0 GB instead of 14.4 GB). The
  resource preview prices the same plan.
- Far tiles and local near-preparation threads make their BLAS calls on one
  thread (`single_thread_blas`). A range rebuilt inside a mode worker used to
  inherit that worker's multithreaded share, and OpenBLAS crashed the process
  (access violation) under the tiles' concurrent multithreaded products; the
  old code crashed the same way under a small CPU allocation.
- Far tiles are assembled per node: a node's two test functions are summed
  before the source-side product and each EFIE term is combined into modes
  once (half the products, a quarter of the combinations; 1.9x per EFIE tile
  of five rows, nothing to share in the one-row tiles of very large meshes).

## September 25 architecture changes

The dense far-block store was the largest stage of every high-frequency solve
(work and memory growing as N^2 M, 12 GB written to disk for the 10 GHz ogive
survey, 24 GB on its certified mesh). Far interactions between well-separated
parts of a generatrix have low numerical rank, the same across modes and
families, so large surfaces now keep their far blocks compressed. On the
120 x 5 in ogive at 10 GHz (181 aspects, eight cores) the survey takes 53 s
instead of 92-95 s and the certified solve 159 s instead of 334 s, with no
spill (12 and 29 GB before); RCS agrees with the dense store to 4e-9 of the
largest amplitude (1.5e-7 dB within 40 dB of the peak).

- `CompressedFarBlocks` (`bor/compressed_far.py`) builds the streamed far
  blocks as an H-matrix over the generatrix nodes with GHOST's own sampler and
  contractions: near-diagonal leaves as tiles, admissible blocks by cross
  approximation of the stack of all families and modes (one pivot row or
  column tile yields every mode), then each family and mode slice truncated
  to its own rank. Every mode is built at once and kept in RAM; mode assembly
  writes the blocks into the system quadrants, so the LU, solves and
  certification are unchanged. The 10 GHz ogive's far blocks: 17 s on eight
  process workers (25 s on threads) against 43 s for the streamed build in
  memory and 62 s spilled, 1.08 GB against 12.1 GB, agreement 2e-11 per mode.
  Compressed double-precision modal slices now retain a shared orthonormal
  left basis when it costs less than separate expanded left factors. The
  existing factors, truncation tolerance, and reconstruction order remain
  unchanged. Explicit single precision still expands in double before casting.
- The BoR option `far_compression` selects it: `auto` (default) for surfaces
  of at least `FAR_COMPRESSION_MIN_NODES` (1,000) nodes, `on`, `off`. The
  streaming estimates price the compressed store (2.0-2.4 times above the
  measured stores) for every mode at once; the dense per-range stream budget
  and the spill no longer apply to it. Conductor and dielectric solves price
  it so; the coated, partial-coating and multi-region planners still price
  their self streams as dense (an upper bound). Their rectangular cross
  streams now have a separately verified bounded representation, described below.
- Tiles are sampled in spawn processes when the near-preparation scope admits
  a process pool (the same capability test and size); otherwise on threads.
- Mode factors: a mode system of at least 10,000 unknowns on a single surface
  is factored as the checked HODLR inverse of the 2-D dense factor (see
  [numerical methods](NUMERICAL_METHODS.md)), priced at two matrix copies per
  worker instead of three, and falls back to LU if rejected.
  Dielectric, coated, partial-coating, and multi-region systems also supply
  their reduced-coordinate ordering to the same checked factor. These newer
  paths retain the conservative LU memory estimate; the original modal
  matrix, residual and conditioning gates still decide acceptance and LU fallback.
- Mirror symmetry: a surface symmetric about a plane normal to its axis
  (nodes, impedances and sheets) factors each mode as its even and odd halves
  (`bor.factor.MirrorSplit`, a quarter of the LU), refined against the exact
  system, whose 1e-8 asymmetry (near/far routing) costs one refinement step
  per batch; LU replaces the halves if refinement does not reach the gate. It
  is used while the unknowns exceed four times the right-hand sides (54.0 s
  against 56.5 s for the 10 GHz survey; RCS identical to 1.5e-15).

Which paths ran. `scripts/check_speed_paths.py` reports whether a machine can
use the native samplers, the 2-D native libraries and an optimized BLAS (exit
status 1 when any would fall back). Each BoR frequency's
`metadata["per_frequency"]` entry records what a solve used:
`stream_sampling_backend` (`banded_native_pairs` natively), `stream_far_compression`
(null for the dense store; its `backend` is `processes`, or `threads` when no
process pool was admitted, for example because the entry script has no
`if __name__ == "__main__":` guard), `stream_spill_gb`, `near_preparation.backend`,
and in `modal_execution.systems` each mode's factor `backend` with any
`mirror_fallback` or `hierarchical_fallback`. A top-level
`automatic_factorization_fallback` marks a switch to the compressed
factorization. 2-D results record `linear_backend` (`cpu_hierarchical` for the
hierarchical factor, `cpu` for LU), `dense_fallback_reasons` and, for automatic
runs, `backend_selection`.

## Bounded near and rectangular storage

Self near blocks accumulate directly into preindexed nodal destinations. EFIE
and MFIE combine repeated node pairs; IBC also keeps the source element in the
key so arbitrary element impedances are weighted exactly as before. No full
uncoalesced coefficient cache is constructed first. Independently integrated
reciprocal EFIE pairs are compared before accumulation, preserving the existing
near-quadrature diagnostic. Planning keeps the older conservative upper bound.
For the 2,000-element sphere topology, 63,088 raw corner entries reduce to
19,773 nodal entries or 35,544 source-aware entries: at a mode cap of 128,
520.9 MB per raw four-component family becomes 163.2 MB or 293.5 MB respectively.
These are retained coefficient byte counts, not measured process peak memory.

Double-precision cross streams selected by `far_compression` use complete
original quadrature samples in rectangular nodal tiles. Only geometrically
separated tiles whose supporting elements contain no near pair may be reduced.
An SVD proposes factors; comparison against **every original coefficient** must
pass the existing 1e-10 relative Frobenius tolerance and save payload bytes.
Otherwise the exact tile remains dense. Near integration, signed-mode parity,
mode caps, mesh, and material properties are unchanged. This avoids a new
coefficient-accuracy assumption based on sampled rows or solve residuals.

The rectangular store preserves admitted mode ranges and disk spilling. Its
numerical payload cannot exceed the corresponding dense range. Resident values
occupy an arena with at most 1 MiB of unused final capacity; spilled values use
one delete-on-close file. Each component uses three int64 index values rather
than a retained Python record. Index/container overhead and arena slack are
reserved within the existing tile-work budget; tile dimensions and modal
batches shrink to fit the existing sampler/contraction memory model. If the
index or minimum tile allowance cannot fit, the original dense stream is used.
The existing dense rectangular resource forecast remains in force. This change
primarily reduces retained RAM and spill I/O; it still samples every coefficient
and adds small SVDs, so it is not an unconditional preparation-time speedup.

Cross EFIE and rotated-PV assembly can also write directly into a caller's
strided system-matrix quadrants. This removes intermediate component matrices;
their scaling and near additions occur in the original numerical order.
Material results report `stream_far_compression` by operator name, including
coefficient-check evidence, retained/spilled payload, compact index allowance,
and any budget fallback. A small complete lossy coated-sphere solve compares
complex fields against the original streamed store; separate tests compare
real/complex-medium coefficients, signed modes, near preservation, spill
cleanup, and forced hierarchical acceptance/rejection on all material paths.

## Geometry and angle conventions

Generatrices use `(rho, z)` with nonnegative radius. A closed body runs from
the positive-z axis endpoint toward the negative-z endpoint, so the left normal
`(-z', rho')` points outward. Aspect angles are measured from positive z and
span 0 to 180 degrees. The solver uses outgoing `exp(-i*k*R)` kernels and
passive material inputs with the corresponding signed loss convention.

BoR scattering uses `sigma = 4*pi*abs(amplitude)**2`. Two-dimensional scattering
width uses a different normalization; do not combine raw amplitudes without
the conversion implemented by the feature-assembly tools. See the
[geometry guide](GEOMETRY_INPUT_CHEATSHEET.md) and
[material format](MATERIAL_CSV_FORMAT.md).

## Verification limits

Analytic sphere tests, table/streamed equivalence, complex-field comparisons,
and mesh convergence cover different error sources. The fixed graded rule for
self and adjacent meridian integrals is unchanged by default; its measured
error and the reason it was kept are in [numerical methods](NUMERICAL_METHODS.md).
The optional `quadrature_check='refine'` setting performs a second same-mesh
solve with deeper self/adjacent/junction integration and checks complex-field
agreement. The desktop calls this "Compare refined integration". It returns
the refined result only after agreement, with separate comparison evidence;
the additional solve costs time and does not certify geometric faceting error.
Junction pairs of two surfaces use a refined rule checked against a deeper one.
Results report `near_quadrature.efie_near_block_asymmetry_max`, the
reciprocity defect of the retained EFIE near blocks (a free lower bound on
their quadrature error), and warn above 1e-3.
