# GHOST solver workflow audit: 2-D and BoR, start to finish (5-6 October 2026)

This audit traces every step a production 2-D solve and a production BoR solve execute, from the batch driver or GUI
through geometry preparation, planning, meshing, assembly, factorization, angle solves, certification and export. For
each step it states what the step computes, why it exists, how often it runs and what it costs, then lists the steps
that are redundant, unused or dead, and the computational-efficiency defects, ranked by expected whole-solve impact.
The three earlier audits (22 and 24 September, 2 October) are the baseline; nothing they fixed is re-reported.

Implementation status (6 October, later the same day): the recommendations were implemented under a 1e-8
accuracy rule; `AUDIT_FIXES_2026-10-06.md` lists each change, its measured equivalence and effect, and the
items deliberately left out of that first pass. Those items (dead code, the check-only and coarse-level
sampling, finer far-rule grading rows, the compressed TM partner in RAM, admission sampling on threads, the
far-tile 2 pi glue, grouped dense-table contractions, a far-ratio split, the HODLR threshold) were then
measured as overlays in `experiments/solver_upgrades_20261006/` at the repository root; the dead code was
removed, the BoR sampled coarse-level checks (no field change, 8-21% of BoR wall) and the drivers' overhead
items were ported, and every rejected candidate is recorded there with its measurement
(`AUDIT_FIXES_2026-10-06.md` sections 2.5-2.7 and 4).

Method: full read of the production call chains; five subsystem reviews with their own measurements (2-D assembly,
2-D dense/compressed stack, BoR far-block pipeline, BoR solver core, drivers/planning/provenance/IO); a profiling
campaign of production-configured solves in fresh processes (certified 2-D airfoil at 1, 3, 6 and 10 GHz and a 3 GHz
survey, 181 angles; BoR PEC cylinder survey and certified at 2 GHz and survey at 6 GHz, 181 aspects; direct sphere
ka=10); instrumented call counting inside one certified 2-D solve; an A/B of the BoR mode cap; a package-wide static
reachability scan. Scripts and raw outputs are in the session scratchpad (`scratchpad/agent_*`), not in the repository.

Host: Windows 11, Python 3.11.0, NumPy 2.4.2, SciPy 1.16.3 (OpenBLAS 0.3.31), 8 physical cores / 16 threads, 31 GB.
Every native speed path loads (`scripts/check_speed_paths.py` all OK), so the numbers reflect the intended fast
configuration. Timings are +-5-15% run to run; other measurements ran concurrently on the host for some runs, which is
noted where it matters.

## 0. Summary

Where the time goes (production profile, 181 angles, 8 threads):

| Solve | Wall | Dominant stage | Second | Third |
|---|---:|---|---|---|
| 2-D certified airfoil 1 GHz (414 panels, P2->P3, 1,294->1,936 unknowns) | 2.9 s | operators 64% | remainder (planning, hashing, imports, mesh) 23% | rhs+factor+condition 13% |
| 2-D certified 3 GHz (864 panels, 2,646->3,964) | 5.5-5.8 s | operators 53% | remainder 19% | factorization 13% |
| 2-D certified 6 GHz (1,612 panels, 4,902->7,348) | 14.1 s | operators 42% | factorization 30% | remainder 13% |
| 2-D certified 10 GHz (2,472 panels, 7,526->11,284; HODLR on the P3 system) | 32.7 s | factorization 44% | operators 30% | remainder 9% |
| 2-D survey 3 GHz (global P1 mesh, 5,666 panels, 8,608 unknowns) | 14.2 s | operators 41% | factorization 35% | remainder 11% |
| BoR cylinder survey 2 GHz (126 unknowns, 13 modes) | 1.04 s | far tables + near preparation 81% | mode sweep (4 threads) ~20% | |
| BoR cylinder certified 2 GHz (base + fine 1.5x) | 2.7 s | preparation 84% | | |
| BoR cylinder survey 6 GHz (370 unknowns, 24 modes) | 4.4 s | preparation 80% | mode sweep ~30% (concurrent) | |
| BoR sphere ka=10 direct (202 unknowns, 20 modes) | 2.3 s | preparation 86% | | |

The ten findings with the largest expected effect, in the order I would take them:

1. **BoR: everything is prepared for every mode up to the cap, but the sweep converges earlier** (18 prepared / 13 used
   on the cylinder, 23 / 19 on the sphere, 26 / 24 at 6 GHz). Preparation is 80-86% of wall. Pinning the cap to the
   modes used gave identical fields (1e-14) and 7-34% less wall. The far-block audit shows the far build is only
   ~30% mode-independent, so the saving is in near preparation, which already has a band-extension mechanism.
2. **2-D: the polynomial near-moment accumulation is a 70-pass NumPy loop** (`polynomial_quadrature._evaluate_chunk`)
   that is one GEMM plus a 4x4 transform per task: measured 1.5x per chunk in situ, 10-18% of a certified solve.
3. **2-D: no attenuation cut in the dense far pass for lossy regions**: 70% of the coating's far pairs have kernels
   below 1e-14 of the near-diagonal scale yet are fully evaluated; ~11% of a 3 GHz solve, growing like N^2 (the
   compressed backend already applies this cut).
4. **BoR: tile planning produces one-row far tiles on large meshes** (1 GB tile budget / 8 threads): the same build is
   1.9x slower than with 5-20-row tiles; 10-20% of large conductor solves. Per-node test-side contraction halves the live
   set and removes five copies of the tested kernel.
5. **BoR: far-tile threads are GIL-bound** (parallel efficiency 0.47 at 8 threads; ~20% of each tile is Python glue):
   10-15% of large solves; one native call per tile group fixes it.
6. **BoR: the dense-tables path contracts tables per mode at 38x the cost of the streamed read**; 15-25% of
   small-body solves (N below ~370 elements, which is exactly where the 2 GB rule selects tables). Stream always.
7. **2-D: the certified hp path re-forecasts the backend three times and rebuilds meshes six times**, and the
   driver's batch plan does not suppress the two in-solve forecasts; 4-10% of small units. The cost model rates dense
   and compressed within 0.3% in every run and once picked the slower backend by tie-break (14.1 s dense vs 11.8 s
   compressed at the 3 GHz survey).
8. **2-D: near kernel tables are rebuilt on assembly worker threads** (context variables do not propagate into the
   pool, so the per-thread cache misses) and the table-reach loop is per-pair Python: 5-7% of a certified solve, two
   one-line fixes. Fixed-order near pairs are also re-integrated for the P3 candidate (6-7.5%).
9. **Condition estimates are optional diagnostics that cost 5.8% (dense) to 7.3% (compressed) of a whole 2-D solve**,
   and the HODLR factor that engages at 10,000 unknowns saved memory but not time on this host (7.4 s build vs ~7.4 s
   extrapolated LU, at half the cores, with slower downstream phases). The learned LU/HODLR crossover can never collect
   its samples by construction.
10. **The bare Python API runs single-threaded** (`solver_method='auto'` without `execution_options` resolves to one
    assembly and one BLAS thread: 60 s instead of 14 s for the 3 GHz survey). GUI and drivers are unaffected.

About 50 top-level functions have no production callers (Appendix A), including roughly 530 lines of superseded
manifest/attestation code in `execution/provenance.py`, the old NumPy near rules in `bor/kernels.py`, the non-banded
sampler branches in `bor/streaming.py`, four research prototypes, and about 900 lines of the compressed/linalg stack
that only explicit non-default options reach.

## 1. The 2-D production workflow, step by step

Verdicts: **N** = necessary for correctness or the result contract; **P** = required by the certification or admission
policy (cost noted); **D** = optional diagnostic; **R** = redundant (recomputes something already available);
**X** = dead or never reached in a default production run.

### 1.1 Driver (`run_local_monostatic.py`), per run and per unit

| # | Step | Where | Purpose | Frequency | Measured cost | Verdict |
|---|---|---|---|---|---|---|
| D1 | Validate CONFIG, discover `.geo`, require unique stems | `_validate_config`, `_discover_geometries` | fail early | once | ms | N |
| D2 | Per-geometry input fingerprint (SHA-256 of the file and its material CSVs) | `assembly.fields.geometry_input_fingerprint` | bind outputs to exact inputs | once per geometry, then 3x per unit (`_verify_unit_input`) | 0.7 ms per call | N once; the repeats are R |
| D3 | Manifest: source fingerprint + per-file inventory (178 files, 4.44 MB) + runtime fingerprint | `provenance.backend_source_fingerprint/inventory`, `runtime_environment_fingerprint` | attest the solver version | once per run (55 + 51 + 220 ms first call), then verified 3x per unit through a stat-keyed cache (7.5 ms cached, 51 ms uncached, 65 ms on a worker's first unit) | 25 ms per unit cached | N once; third check (after publication) is R |
| D4 | Plan every unit from the exact candidate meshes, pricing both backends | `hpc.scheduler.predict_2d_resources_many` -> `_resource_records_for_frequency` (panels, infos, mesh, layout, `_estimate_memory_gb`) | dearest-first order, RAM admission | once per geometry, serial in the parent before the pool starts | ~100 ms per frequency per geometry (1.1 s per 3 frequencies if no execution scope is active) | P; the meshes are discarded and rebuilt in the worker (R, see F-2D-7) |
| D5 | Batch backend choice over simulated schedules | `runs.batch.select_batch_backends` | minimize predicted batch completion | once | 0.3 ms (5 units), 72 ms (12 units, 4,096 combinations), 1.7 ms (20) | P (cheap) |
| D6 | Spawn worker pool; initializer pins BLAS, installs the fingerprint cache, imports the solver | `ExecutorPool`, `_pool_initializer` | isolation, crash tolerance | per worker life (recycled every 4 units) | ~470 ms spawn + imports + 58 ms cold hash, i.e. ~130 ms per unit amortized | P; recycle interval is a tunable (F-D-2) |
| D7 | Per unit: verify provenance and input before the solve, after the solve, after the export | `_verify_run_provenance`, `_verify_unit_input` | refuse mixed-state outputs | 3x per unit | 25 ms cached; 150-210 ms uncached (GUI frequency workers, after the 300 s cache flush, network filesystems) | first N; second D; third R |
| D8 | Parse the snapshot (per-process cache; spawn means no inheritance) | `runs.inputs.load_geometry_snapshot` | solver input | once per unit per child | 0.4 ms | N |
| D9 | Solve under `compact_samples` and `linear_precision('double')` | `solve_monostatic_rcs_2d_certified` or `_survey` | the work | once | dominant | N |
| D10 | Embed attestation, export `.grim` | `embed_output_attestation`, `io.grim.export_result_to_grim` | deliverable | once | 22 ms (362 samples, 70 KB metadata JSON) | N |

Measured non-numerical overhead of one 3 GHz certified unit: ~0.45 s of 4.96 s (9%), ~3% of a 15 s unit: in-process
~200 ms (three verifications 25 ms, timing-history key 20 ms, two hidden backend re-forecasts 129 ms, export 22 ms),
worker respawn ~130 ms amortized, parent planning ~100-160 ms per unit.

### 1.2 Public entry decorators (every public 2-D solve)

| # | Step | Where | Purpose | Cost | Verdict |
|---|---|---|---|---|---|
| E1 | `prepared_execution`: run-owned preparation scope (material library, validated geometry, 16 MiB forecast cache, run resources) and the sweep's sizing frequencies | `twod.preparation` | share immutable preparation across TE/TM and base/fine | negligible | N |
| E2 | `configured_execution`: resolve the automatic profile (`from_environment()` + adaptive factorization/mesh), normalize angle arrays, build a timing-history request key (hashes the whole source bundle), read the host timing cache, install learned factor choices, run `select_backend` unless a batch selection exists, wrap the solve in `execution_scope(limit_blas)`, record timing | `execution.options.configured_execution`, `execution.selection`, `execution.timing_history` | automatic dense/compressed choice and evidence-based re-ranking | request key 20 ms cached / 53 uncached; `select_backend` 0.21-0.45 s per public call (1 ms on a cache hit); the re-ranking machinery never changes a decision in default runs (section 5, F-2D-9) | P; the timing/crossover machinery is inert by construction (R) |
| E3 | `profiled_solve`: 50 ms RSS sampler thread and stage timers | `execution.metrics` | runtime evidence | ~1% of one core (process-tree walk 3 ms every 250 ms) | D |
| E4 | `experimental_monostatic`: kernel-table CPU state (validated Chebyshev tables, native far evaluation, 64 MiB operator cache) | `execution.cpu` | fast kernel evaluation | tables built once per wavenumber per request (3 tables, 0.05-0.09 s); the 64 MiB operator cache never stores in production (`reuse_operators=False`) | N (tables); X (cache) |
| E5 | Multi-frequency requests re-enter the public function per frequency and merge | `_frequency_local_co_solve`, `_merge_frequency_results` | one matrix lifetime per frequency | 1.5 ms merge | N |

### 1.3 Certification controller (`_run_certified_2d_pair` -> `adaptivity.run_certified`)

| # | Step | Where | Purpose | Cost | Verdict |
|---|---|---|---|---|---|
| C1 | hp eligibility: reference P1 panels >= 512, cubic DOFs <= 1.5x reference panels, no thin layer, taper or explicit N | `adaptive_geometry.eligible_snapshot` | choose hp pair or P1 pair | ms | P |
| C2 | P2 candidate solve on the coarsened primitives (factor 8 for well-sampled input, else 4); `select_backend` again for this mesh | `adaptivity.solve` -> `solve_monostatic_rcs_2d` | base of the convergence pair | 55% (1 GHz), 49% (3 GHz), 41% (6 GHz), 35% (10 GHz) of wall; its field is used only for the comparison | P (certificate); re-forecast is R |
| C3 | P3 accuracy solve on the same panels (near moments shared through `moment_cache_scope`, 100% hits on the P3 pass); `select_backend` again | same | the published result | 45-65% of wall | N; re-forecast is R |
| C4 | Compare complex fields (peak-normalized), dB above a -30 dB floor, phase; publish P3 | `_finish_certified_2d_pair`, `evaluate_mesh_convergence` | certificate | 2-7 ms | N |
| C5 | On failure: mark primitives from the cubic Legendre tails, local h-refinement (up to 3), conservative retry at coarsening 4, then the P1 base/fine pair | `adaptivity._run_certified` | robustness | not triggered in any measured run | P |

### 1.4 One polarization of one frequency (`solve_monostatic_rcs_2d_single_polarization`)

| # | Step | Where | Purpose | Frequency | Measured cost | Verdict |
|---|---|---|---|---|---|---|
| S1 | `prepare_geometry` (cached): material library from inline rows/CSV (fingerprinted twice), strict preflight (cracks, duplicates, intersections, winding, nesting) | `twod.preparation`, `geometry.validate_geometry_snapshot_for_solver` | reject bad input once | once per run | 3 ms cold, 0.4 ms cached | N |
| S2 | Mesh: shortest material wavelength, panels (per-primitive counts with a 0.05 ceiling tolerance, impedance-junction grading, bound-surface-wave sizing, faceting advisory), coupled panel infos, interface-aware linear mesh (node welding by signature), enrichment to degree p | `_mesh_wavelength_for_snapshot`, `_build_panels`, `_build_coupled_panel_info`, `_build_linear_mesh_interface_aware`, `basis.enrich` | discretization | TE builds, TM reuses through the shared cache (no sheets) | ~20 ms at 864 panels; but built 6 (panels) / 10 (meshes) / 22 (infos) times per certified solve across forecasts and steps | N once; repeats R |
| S3 | Coupled infos on the mesh; formulation checks; `_dense_formulation_resources` (regional layout, near-pair count by KD-tree); `_estimate_memory_gb` gate; paired-assembly admission | `_build_linear_coupled_infos`, `_dense_formulation_resources`, `_estimate_memory_gb`, `plan_paired_assembly` | admission and routing | per polarization, plus twice more inside each forecast (12 per request) | 0.18 s per 12 calls at 864 panels | N once; repeats R |
| S4 | Junction-constraint diagnostics and node report (never applied by any active formulation) | `_build_linear_junction_constraints(materialize=False)`, `_linear_coupled_node_report` | notices and metadata, orientation-conflict refusal | per polarization | 5 ms at 864, 31 ms at 5,666 panels | D (the refusal is N; the matrix is not) |
| S5 | Assemble the system: TE step runs `assemble_multi` (or `assemble_pair` building TE and TM in one traversal when conductor rows exceed 1/8 of the unknowns and the second matrix fits); TM step takes the retained matrix or converts the TE one (`_reuse_te_system` for open junctions, `_reuse_combined_system` row strips for closed ones) | `twod.formulations.regions.assemble_system`, `twod.assembly.scatter`, `twod.operators._assemble_multi` | the operator | per polarization (TM 6-11% of TE) | 1 GHz: 2.58 s of 3.3 s (near pairs 2.19 s, far tiles 0.29 s); 3 GHz: 5.45 s of 8.0 s (near 3.63 s, far 1.60 s) at 2 threads | N |
| S5a | inside S5: far tiles: per-tile pair classification, graded far order from the tile's longest panel and closest pair, native Chebyshev-table evaluation (46-57 ns per kernel), ordered native scatter on the calling thread | `_far_tile`, `native/far.c`, `scatter_tile` | O(N^2 q^2 / T) | per traversal (one per wavenumber per TE step) | 0.49 thread-s (1 GHz) / 2.96 thread-s (3 GHz); serial commit 19% of the far wall at T=2 | N; order pinning and missing attenuation cut are R work (F-2D-2, F-2D-5) |
| S5b | inside S5: near pairs: vectorized classification (self, touching, adaptive, fixed order), fixed-order pairs by the 16x16 box rule (kernel tables for lossy k, SciPy Bessel for real k), polynomial self/touching pairs by `near_blocks` (orders 20 and 36 compared, bisect on failure; moment cache), D by reversed K', one ordered scatter per output | `_near_pair_blocks`, `polynomial_quadrature.near_blocks`, `_integrate_linear_pairs_box_sk_batched` | O(near pairs x samples) | per traversal | 66% (1 GHz) / 45% (3 GHz) of wall; `_evaluate_chunk` is 2.0 s tottime of which kernels 40-45% | N; the 70-pass accumulation, table rebuilds, per-pair Python reach, and P3 re-integration are R work (F-2D-1, F-2D-4) |
| S6 | Factor: `DenseFactor`: one blocked pass for finite check, inf-norm and row maxima; equilibration column pass; LU on an F-ordered copy (`copy_for_lu`; in place with the original spooled only when the copy does not fit); from 10,000 unknowns a randomized HODLR refined to 3e-15 with LU fallback; condition estimate by `onenormest` on the equilibrated inverse (4 two-column LU solves) because certified/survey always request it | `twod.fields._solve_fields`, `linalg.dense`, `linalg.hierarchical` | solution and quality gate | per polarization | LU 2.48 s at 8,608; row norms 0.076 s (N), equilibration 0.13-0.16 s (D), copy 0.10 s (N), condition 0.25 s (D) = 16% of LU; HODLR at 11,284: 7.0-7.4 s | N for LU; D for condition (F-2D-8); HODLR crossover mis-set on this host (F-2D-8) |
| S7 | Angle batches of 256: polynomial plane-wave loads once per batch, `sweep.solve` (incremental pivoted-QR incident basis, 'auto'), original-matrix residual and backward-error gate (1e-12), refinement, far-field projection reusing the batch's loads | `twod.fields`, `linalg.sweep`, `assembly.kernels.incident_loads` | fields | per batch (one batch at 181 angles) | QR reduces 181 angles to 28-66 solved columns; net gain for compressed/HODLR, net loss (+3%) for LU (F-2D-8) | N; the LU case is R work |
| S8 | Record samples into the compact table, ~80-key metadata, quality gate (residual, condition, warnings) | `record_samples`, `evaluate_quality_gate` | result contract | per polarization | ms; 181 progress callbacks with identical timestamps | N (progress spam is cosmetic) |
| S9 | Merge TE and TM (`_merge_co_polarized_2d_results`), then certification merge | | one co-polarized result | per frequency | 1.5 ms | N |

## 2. The BoR production workflow, step by step

### 2.1 Driver (`run_local_bor.py`)

| # | Step | Where | Purpose | Cost | Verdict |
|---|---|---|---|---|---|
| B-D1 | Validate config and radar grid; derive the unique body aspects the azimuth/elevation grid needs | `_validate_config`, `assembly.fields.radar_grid_aspects` | solve only the required aspects | 12-18 ms (72x25 looks -> 300 aspects) | N |
| B-D2 | Fingerprints, manifest, stale spill sweep | as 2-D | attestation, disk hygiene | ms | N |
| B-D3 | Plan per unit: extent/cost heuristic plus a full `estimate_bor_resources` (mesh counts, storage model, both assemblies) | `_plan` | dearest-first order, RAM and spill-disk reservations | 4-10 ms per pair (20 ms first) | P |
| B-D4 | Pool of non-daemonic spawned workers (so near preparation can start its own pool); per-unit CPU reservation | `ExecutorPool`, `cpu_allocation_scope` | concurrency | ~0.5 s per worker life, recycled every 2 pairs (~265 ms per pair) | P; recycle interval tunable |
| B-D5 | Per pair: verify x3, snapshot, certified or survey solve under `compact_samples`, split channels, export two restart `.grim` files | `_solve_and_export` | deliverable + restart records | 20 ms per export | N; third verify R |
| B-D6 | After the sweep: read every restart file back, merge into one monostatic body `.grim` per geometry with the radar axes | `read_unit_grims`, `save_monostatic_grim` | feature-ready product | seconds per geometry; the writer saves a temp file, re-loads it, writes again | N; the re-load is R |
| B-D7 | The parent pre-parses snapshots and imports the solver "before the pool forks" | `run_local_bor.py:806-811` | intended copy-on-write inheritance | 0.1 s | X (the pool uses spawn; nothing is inherited) |

### 2.2 Public entry (`solve_monostatic_rcs_bor_certified` = base solve + fine solve + comparison)

| # | Step | Where | Purpose | Cost | Verdict |
|---|---|---|---|---|---|
| B-E1 | `configured`: options; `resolve_automatic_plan` prices every frequency for the caller's assembly, then streaming; admission retry loop over plans on `BorAdmissionError`; `numerical_preparation` scope (surface contexts reused across cap retries); mode-cap extension loop (up to two) | `bor.options.configured`, `bor.dispatch.resolve_automatic_plan` | a plan that fits RAM | 4-10 ms per frequency | P |
| B-E2 | `reserve_output`, `profiled_solve` | | output RAM reservation, timers | negligible | D |
| B-E3 | Certified: whole base solve, whole fine solve on 1.5x elements per segment, VV/HH complex-field comparison | `solve_monostatic_rcs_bor_certified` | mesh certificate | fine solve ~1.6x the base (certified 2.73 s against a 1.04 s survey at 2 GHz); no work is shared across the two meshes | P |

### 2.3 One frequency (`solve_monostatic_rcs_bor` -> `solve_bor` for conductors)

| # | Step | Where | Purpose | Frequency | Measured cost | Verdict |
|---|---|---|---|---|---|---|
| B1 | `prepare_geometry` (the 2-D validator), material-library copy, mesh wavelength per frequency, chains from the snapshot, body classification, group preparation (stitched generatrices) | `bor.dispatch` | geometry model | once per call | ms | N |
| B2 | Per frequency: impedance-junction grading marks, surface layout, element limit, `_mesh_generatrix` (Python loop per primitive and element), per-element impedance | `_mesh_generatrix` | discretization | per frequency | ms (O(N) Python) | N |
| B3 | `solve_bor`: validate the generatrix; `BorPecSolver` (Generatrix arrays, 3 Gauss points per element, per-pair near routing, far-gap preflight); `_bor_mode_limits` (cap = bandwidth + max(12, ceil(4.05 bw^(1/3) + 2)), tail start = bandwidth) | `bor.solver` | operator geometry; modal truncation | once per frequency, reused across cap retries | 0.04-0.13 s | N |
| B4 | Estimates: table GB, streaming GB, tables-vs-streaming (tables below 2 GB), mode block, spill plan, operator storage model, assembly peak | `estimate_bor_table_gb`, `estimate_streaming_gb`, `plan_streaming_mode_block`, `plan_stream_spill`, `estimate_bor_operator_storage_gb` | far representation and RAM pricing | once | ms | P; the 2 GB tables rule selects the slow per-mode path on small bodies (F-BoR-3) |
| B5 | `_mode_sweep`: `plan_bor_mode_workers`, `_guard_bor_dense_memory` | | admission | once | ms | P |
| B6 | `prepare(m_max)`: far blocks or tables for every mode 0..m_max (streamed: native half-grid sampling per pair, cosine/sine GEMM transforms, test/source contractions, packed symmetric EFIE, spill or compressed H-blocks; tables: full P x P tables); then near pairs in batches (graded native rules, coarse/fine check, parity moments, compact nodal accumulation, reciprocity diagnostic) on threads or a spawned 4-process pool | `BorPecSolver.enable_streaming`, `prepare_operators`, `near_parallel` | every mode's coefficients | once per frequency per cap attempt | 80-86% of wall on all measured bodies; near preparation 74-88% of that on 300-800-element spheres; the spawn pool ramps 0.3-0.5 s and is started for a 1.4 s job on the certified fine mesh | N for the modes used; the modes beyond the converged tail are R (F-BoR-1) |
| B7 | Mode window (W workers, strict-order consumption, predicted horizon): per \|m\|: EFIE + MFIE into one buffer (CFIE), pole reduction at \|m\|=1 or `Z[np.ix_(mask, mask)]` gather; `ModalFactor` (LU on an F copy, or mirror halves, or HODLR from 10,000; `gecon` condition per mode); per 64-aspect batch: `rhs_vv_hh_batch` (shared Bessel and axial-phase records), `sweep.solve`, refinement to 1e-12, `farfield_vv_hh_batch`; accumulate; tail test (two quiet modes after the physical bandwidth) | `_mode_sweep_impl`, `bor.factor`, `linalg.sweep` | the solution | per mode used | per mode at 370 unknowns: assembly 52 ms (NumPy gathers), LU 2 ms, excitation 13 ms per 128 columns, far field 5 ms, gecon < 0.4 ms; packed-EFIE unpack + completion 20-35% of the LU time per mode at 800-2,200 nodes | N; unpack and gathers are R work (F-BoR-5) |
| B8 | Convergence requirement, near-asymmetry summary, result dict | | gate | once | ms | N |
| B9 | Dispatcher: per-aspect power/amplitude consistency check and sample rows (Python loop over 2 channels x aspects), per-frequency metadata, quality gate | `solve_monostatic_rcs_bor` | result contract | per frequency | ms | N (`mode_tasks_started` is not copied into `per_frequency`; cosmetic) |

## 3. Measured time budgets

### 3.1 2-D certified airfoil (hp strategy, VV+HH, 181 angles, 8 assembly threads, fresh process per solve)

| Run | Wall | operators | factorization | rhs (sweep + solve) | excitation | condition est. + scaling | remainder | busy cores mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GHz, 414 panels, 1,294 -> 1,936 unknowns | 2.93 s | 1.87 (64%) | 0.10 (3.5%) | 0.17 (5.7%) | 0.05 | 0.06 (2.0%) | 0.67 (23%) | 2.1 / 2.8 |
| 3 GHz, 864 panels, 2,646 -> 3,964 | 5.48-5.81 s | 2.93-3.08 (53%) | 0.72-0.77 (13%) | 0.40 (7%) | 0.12 | 0.25-0.29 (5%) | 1.05-1.10 (19%) | 3.1 / 6.4 |
| 6 GHz, 1,612 panels, 4,902 -> 7,348 | 14.14 s | 5.90 (42%) | 4.22 (30%) | 1.04 (7%) | 0.23 | 0.84 (6%) | 1.89 (13%) | 4.6 / 8.2 |
| 10 GHz, 2,472 panels, 7,526 -> 11,284 (HODLR on P3) | 32.66 s | 9.89 (30%) | 14.49 (44%) | 2.63 (8%) | 0.36 | 2.39 (7%) | 2.85 (9%) | 3.7 / 8.8 |
| survey 3 GHz, global P1 mesh, 5,666 panels, 8,608 | 14.16 s | 5.76 (41%) | 4.95 (35%) | 0.84 (6%) | 0.09 | 0.80 (6%) | 1.50 (11%) | 5.0 / 8.7 |

- The P2 base solve (used only as the certification reference) is 55 / 49 / 41 / 35% of wall at 1 / 3 / 6 / 10 GHz.
- The survey is 2.5x slower than the certified solve at 3 GHz: it solves the global P1 mesh (8,608 unknowns) while
  the certified path solves the hp-coarsened mesh (at most 3,964). The "cheap" mode is the expensive one for this
  geometry.
- The pre-first-stage gap is 0.40-0.85 s at one busy core: `select_backend` 0.21-0.45 s, source hashing for the
  timing key 0.09 s, lazy imports 0.19 s on the first solve, `threadpoolctl` DLL enumeration 0.05 s per solve. The
  planning repeats inside the adaptive controller (2 x 0.14 s). The cost model rates dense and compressed within 0.3%
  in every run (0.9876 vs 0.9904 at 1 GHz ... 30.18 vs 30.28 at 10 GHz), so it does not discriminate the candidates it
  spends this time on.
- The operators stage reaches only 2.1 (1 GHz) to ~5 (10 GHz) busy cores of the 8 configured threads: the polynomial
  near-block quadrature is GIL-holding Python (`_evaluate_chunk` 2.0 s tottime, 0.9 s ufunc reductions, 0.5 s array
  copies, 181,000 `np.linalg.norm` calls at 3 GHz) while the native far tiles release the GIL.
- Inside the operators stage at 3 GHz (2 threads, 7.99 s wall in the instrumented run): `_near_pair_blocks` 3.63 s
  (polynomial `near_blocks` 2.80 s, fixed-order box rule 1.03 s), far tiles 1.60 s wall (2.96 thread-s; serial
  scatter commit 0.30 s), TM derived from TE 0.61 s, kernel tables 21 builds 0.31 s, `_table_for` 60 calls 0.47 s,
  `_dense_formulation_resources` 12 calls 0.18 s.
- Condition estimate plus the single-threaded equilibration pass: 2% of wall at 1 GHz, 7% at 10 GHz.
- HODLR at 11,284 unknowns: 7.0 + 7.4 s builds at 3.2 busy cores against a dense LU extrapolated to ~7.4 s at 6.6
  cores. It saves memory (137 MB factor vs ~2 GB LU) but not time on this host, and it slows the downstream phases
  (rhs 2.6 s vs 1.0 s at 6 GHz, condition 1.7 s vs 0.5 s, linear solve 1.2 s vs 0.4 s). Measured directly on the 8,608
  system: HODLR build 7.37 s vs LU 2.48 s, batch solve 2.15 s vs 0.53 s; 47% of the HODLR build is `Block.dense`
  gathers of an interleaved DOF layout (3.1e8 entries = 4.2 N^2).
- Sweep QR compression pays in 2-D: 181 angles become 28 (1 GHz) to 64-66 (10 GHz) solved columns.
- Backend choice at the 3 GHz survey: the prior tied (12.196 dense vs 12.234 compressed), the tie-break took dense
  (14.1 s); the compressed backend measured 11.8 s for the same request (assembly 6.07 s, factor 2.43 s, condition
  0.87 s, admission sampling 0.86 s serial in the parent, TM spool 0.38 s).

### 3.2 BoR (PEC cylinder r=4 in h=10 in, 181 aspects, workers=4; direct sphere a=0.1 m, 100 elements, 37 aspects)

| Run | Wall | operators (far + near preparation) | modal_assembly | factorization | rhs | excitation | far_field | modes used / cap (prepared) | unknowns | near backend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| survey 2 GHz | 1.04 s | 0.84 (81%) | 0.11 | 0.005 | 0.04 | 0.26 | 0.07 | 12 / 17 (18) | 126 | threads |
| certified 2 GHz (base + fine 1.5x) | 2.73 s | 2.29 (84%) | 0.27 | 0.011 | 0.10 | 0.56 | 0.15 | 12 / 17 | 188 (fine) | threads; fine: 4 processes |
| survey 6 GHz | 4.44 s | 3.55 (80%) | 1.30 | 0.046 | 0.20 | 0.92 | 0.34 | 23 / 25 (26) | 370 | 4 processes |
| direct sphere ka=10 | 2.28 s | 1.96 (86%) | 0.31 | 0.012 | 0.02 | 0.10 | 0.04 | 19 / 22 (23; 21 started) | 202 | 4 processes |

(Mode-worker stages run concurrently on four threads, so their sum exceeds wall; the union of stage intervals covers
92-97% of wall.)

- Preparation serves modes that never run: 28% of prepared modes unused at 2 GHz, 17% on the sphere, 8% at 6 GHz.
  A/B with the cap pinned to the modes used (fields identical to 1e-14): wall -34% (n=40 sphere, 16 -> 8 modes),
  -7% (n=100, 23 -> 19), -8% (n=200, 23 -> 19). Even with the pinned cap, preparation is 86% of wall on the
  402-unknown sphere: the preparation cost per mode, not the modal solves, governs small and medium bodies.
- Far build internals (sphere of 800 elements at 5 GHz, 24 modes, 8 tile threads, 7-8 s wall, 56-59 thread-s): native
  sampling 26% of thread time, transform GEMMs 19%, contractions 20%, scatter/scaling ~8%, locks and GIL waits ~20%
  (the `_half_grid_tables` lock alone 8-12%), band adds 2%. Parallel efficiency 0.47 at 8 threads (serial 12.7 s,
  8 threads 3.37 s on the 600-element case).
- Tile planning: `_plan_banded_tiles(3,055 elements, 28 orders, 1 GB, 8 threads)` gives one-row tiles (live set
  113 MB per full-width row); forcing 1 / 2 / 5 / 10 / 20 rows on the 800-element sphere gave 13.35 / 8.62 / 7.03 /
  6.63 / 6.47 s for bitwise-equivalent blocks (5e-16).
- Dense-tables path (selected below ~370 elements): per mode 83 ms EFIE + 34 ms MFIE of contraction versus 2.0 + 1.1
  ms for the streamed read, identical to 7e-16; the table build samples both triangles (+33% pair-samples) and needs
  ~7x the memory. A 300-element solve took 10.4 s with tables and 8.8 s streamed.
- `workers=4` (the driver's per-unit default) caps every BoR phase at ~4.2 of 8 cores; the spawn pool for near
  preparation ramps over 0.3-0.5 s and leaves a 0.25 s idle dip at teardown.

### 3.3 BoR near preparation cost split

Measured on PEC spheres A (100 elements, 4.77 GHz, 688 near pairs, cap 22) and B (300 elements, 3 GHz, 2,088 pairs,
cap 19), 37 aspects, CFIE, serial near preparation (`scratchpad/agent_bor_core/`):

| Component | A (4.88 s) | B (13.75 s) |
|---|---:|---:|
| Native graded angular rules (`_near_rule_moments`) | 72% | 72% |
| Grouping and refinement overhead (`np.unique(axis=0)` + stable argsort 8%, `_parity_outputs` identity gather 3%, chunking) | 12% | 13% |
| Coarse/fine acceptance check (`_near_check`, hypot per re/im pair) | 6% | 6% |
| Python Galerkin contraction (`_contract_near_chunk`) | 6% | 6% |
| Layout, point chunks, reciprocity diagnostic, pickling | 2% | 3% |
| Process-pool spawn ramp (per solve, not per pair) | 0.31 s | 0.33 s |

- 198 quadrature points per pair on average (self cells 920, disjoint pairs 36 + 144); 95-99 angular samples per
  point evaluation; the coarse level is 40% of all angular samples, and 100% of points pass the first coarse/fine
  check on these bodies (and 100% of disjoint pairs pass the first meridian refinement, 7e-8 against rtol 2e-5).
- Scaling with the cap: ~2.2-2.9% of near preparation per mode (sub-linear: the angular orders grow with the top
  order, the contraction width linearly); the far tables are linear in the modes (18% of B's wall).
- Worker scaling (far tables excluded): A serial 4.85 s, threads x4 1.65, processes x4 1.79, x8 1.32 / 1.32;
  B 13.47, 4.48, 4.13, 3.66, 2.59 s. Processes and serial results are bitwise identical. The near phase is capped by
  the requested mode-worker count (`preparation_workers=workers`), so the driver's `workers=4` leaves half the host idle
  during the phase that is 64-78% of wall.
- Band extension (preparing near modes lazily to the sweep's horizon) is supported by `_prepare_near_families` and
  `ModalBands`, but the extension path is defective on closed bodies: with `mode_start > 0` the NumPy sampler path is
  taken and `_near_check` scales the error by the band's own orders, which are rounding noise for near-axis points, so
  the check never converges and `NEAR_ANGULAR_MAX_ORDER` raises. The production automatic cap-expansion retry hits the
  same path (reproduced end to end). The band path is also slower per point than a full native rebuild (1.4-2.5x).
  Preparing exactly the modes used would save 7-11% of near preparation (5-7% of wall) plus the far-table share; the
  7-34% measured by pinning `n_modes` comes mostly from the far tables, the sweep overshoot and the memory plan.

## 4. Redundant, unused and dead steps

IDs: R = redundant (recomputed or unconsumed), X = dead or unreachable in default production. Impact is the share of
one solve or unit unless stated.

| ID | Finding | Evidence | Impact |
|---|---|---|---|
| R-2D-1 | Backend forecasts run three times per certified solve (public boundary, P2 step, P3 step); each rebuilds the candidate meshes and prices both backends; the driver's batch plan suppresses only the first | `execution/options.py:587-592`, `twod/adaptivity.py:151-153`; call counts: `select_backend` 3, `_build_panels` 6, `_build_linear_mesh_interface_aware` 10, `_build_coupled_panel_info` 22, `regions.build_layout` 18, `_dense_formulation_resources` 12, `_estimate_memory_gb` 20 per certified solve | 0.25-0.33 s (4-10%) per small solve; 129 ms per 3 GHz driver unit |
| R-2D-2 | The first forecast prices the P1 reference pair that is never solved when the hp candidate succeeds | `adaptivity._run_certified` + `candidate_meshes` | part of R-2D-1 |
| R-2D-3 | Fixed-order near pairs are re-integrated for the P3 candidate (and for TE->TM conductor rows) although their 16x16 samples depend only on geometry and k; only the basis projection differs | `operators.py:223-320`, no moment cache (the polynomial `near_blocks` cache hit 100% on P3) | 0.51 s / 22 calls at 1 GHz, 1.03 s / 36 at 3 GHz; the P3 half is 6-7.5% of wall |
| R-2D-4 | Near kernel tables rebuilt on assembly worker threads: `_kernel_table` resolves the CPU state and moment cache through context variables that do not propagate into `map_checked` workers, so it falls to a single module slot that thrashes between two domains | `polynomial_quadrature.py:366-412`, `check_ctx.py`: 21 table builds at 3 GHz (3 far + 18 near, 17 bypassing `cached_table`), 9-22 ms each | ~0.2 s (3%) at 3 GHz, 9 of 12 builds at 1 GHz |
| R-2D-5 | `_table_for` reaches every pair with four `np.linalg.norm` calls in a Python loop, per batch, for a result that is constant per chunk | `polynomial_quadrature.py:402-412`; 5 ms per 976 pairs vs 0.11 ms vectorized | ~0.25 s (3%) at 3 GHz |
| R-2D-6 | `_classify`/`_shared_ends` recompute per task what `_near_fixed_order_positions` already computed vectorized | `polynomial_quadrature.py:37, 333`; 13.4 us per task | 0.16 s per `near_blocks` call on the 5,666-element mesh |
| R-2D-7 | Driver planning builds exact meshes and resource records that are discarded after costing and rebuilt in the worker; the GUI forecast cache is defeated because its key contains the CPU/RAM allocation and `frequency_sweep._plan` runs under a different allocation | `hpc/scheduler.py:521-655`, `execution/selection.py:52-61`, `execution/frequency_sweep.py:84-85` | ~100 ms per frequency per geometry (parent, serial); ~100 ms per GUI frequency |
| R-2D-8 | Junction-constraint matrices and node reports are built per polarization and never applied; the assembly-components profile is recorded per assembly and dropped by the certified merge | `twod/solver.py:2498-2506`, `operators.py:2964`, `assembly/profiling.py` | 5 ms at 864 panels, 31 ms at 5,666 |
| R-2D-9 | `dof_coordinates` built for every field solve but consumed only by HODLR (n >= 10,000); `near.write('S', ...)` copies a RAM chunk onto itself; `KernelTable.__init__` recomputes the degree-only Chebyshev-to-power matrix per build and keeps three representations | `regions.py:581`, `operators.py:2413`, `polynomial_quadrature.py` (table class) | 10 ms at 5,666; 2 ms per table build |
| R-2D-10 | The timing-history and LU/HODLR crossover machinery is executed on every public call but inert by construction: re-ranking needs clean samples of both backends for the same key, and normal runs never execute the non-selected backend; `observation` records only the chosen factor variant, so `choices` can never collect paired samples | `execution/timing_history.py:150-232`, `execution/factor_timing.py:25-108`, `linalg/crossover.py`; every profiled run shows `timing_model: null` | 24 ms per public call (57 uncached) for no decision |
| R-2D-11 | `recycling` (inverse cache) and `block_products.ProductTeam` are called on every forecast/factor under a lock although their capacity is zero and the team is disabled; the 64 MiB operator cache in `CPUState` never stores (`reuse_operators=False`) | `compressed/recycling.py:27-149`, `compressed/block_products.py:19-152`, `execution/cpu.py:110-118, 270-311` | negligible time; dead paths |
| R-2D-12 | Condition estimates are requested by every certified/survey solve: equilibration pass + `onenormest` (four two-column LU solves, each streaming the whole 1.19 GB factor) on the dense path; `equilibrate()` reconstructs every compressed tile a second time and probes with fully refined solves at 1e-9 on the compressed path | `twod/solver.py:3767, 4090`, `linalg/dense.py:46, 180`, `compressed/factor.py:267-284`, `operator.py:365-387` | 16% of LU time (5.8% of a 3 GHz survey solve); 35% of the ACA build (7.3% of a compressed solve) |
| R-2D-13 | `sweep.solve` 'auto' attempts QR reconstruction for LU factors where it cannot pay: a 38-column LU solve costs 113 ms against 272 ms for 256 columns (60 ms memory-bound floor), so the QR (0.10 s) plus the second residual (0.23 s) exceed the saving | `linalg/sweep.py:27-43, 222, 241-269`; measured 0.563 s vs 0.533 s direct per 256-column batch | ~3% of dense solves (keeps paying for compressed/HODLR) |
| R-2D-14 | Compressed path: admission sampling of 20 tiles runs serially in the parent while an 8-process pool is idle; the TM partner operator is spooled to disk with per-array SHA-256 and read back although it fits in RAM | `compressed/memory.py:43-113`, `polarization_cache.py:9-80` | 0.86 s (7.3%) and 0.38 s (3.2%) of an 11.8 s compressed solve |
| R-D-1 | Inputs verified three times per unit, the third after the artifact is published; HPC BoR hashes the geometry per channel (six per pair) | all four drivers | 25 ms per unit cached, 150-210 ms uncached |
| R-D-2 | The whole source tree is hashed by three independent paths (provenance 55 / 6.7 ms, timing-history key 53 / 10.9 ms, checkpoint identity 10.8 ms uncached, twice per GUI run); digests are not shared | `execution/provenance.py:149-157`, `timing_history.py:115`, `twod/checkpoints.py:35-44` | 40-80 ms per unit or GUI frequency |
| R-D-3 | Worker respawn every 4 units (2-D) / 2 pairs (BoR): ~0.5 s spawn + imports + cold hash | `run_local_monostatic.py:109-111`, `run_local_bor.py` TASKS_PER_CHILD | ~130 ms per 2-D unit, ~265 ms per BoR pair |
| R-D-4 | BoR final merge writes a temp `.grim`, re-loads it, writes again; every restart file is re-opened; parent pre-parse "before the pool forks" under a spawn context; stale copy-on-write comments | `assembly/fields.py:4436-4439`, `hpc/common.py:758-841`, `run_local_bor.py:167-169, 806-811`, `runs/inputs.py:49-53` | seconds per geometry; 0.1 s dead work |
| R-D-5 | HPC drivers: every BoR array task re-prices all units (2-4 `estimate_bor_resources` each), every 2-D task re-runs the batch choice, every finishing task re-opens every `.grim` for the run status | `run_hpc_bor_monostatic.py:1150-1286`, `run_hpc_monostatic.py:1017-1026, 1207` | plausible seconds per task on large sweeps |
| R-BoR-1 | Far tables or blocks and near bands are prepared for every mode up to the automatic cap; the sweep's horizon bounds the modes started, not the modes prepared | `bor/solver.py` `prepare(m_max)` before `_mode_sweep`; `_bor_mode_limits`; measured 18/13, 23/19, 26/24 prepared/used | 7-34% of wall (A/B with the pinned cap); see section 3.3 for the near-preparation split |
| R-BoR-2 | Dense-tables path: both triangles of the symmetric G table sampled (+33% pair-samples); 12 `_local_basis` arrays and 3 concatenations rebuilt per mode; per-mode contraction through strided order gathers of the `[P, P, M]` layout at 38x the streamed read | `bor/solver.py:1706, 943-1016` | 15-25% of small-body solves |
| R-BoR-3 | Near pairs zeroed twice (sampler masks them, `_zero_near` zeroes again); per-family duplication of pair bookkeeping (near mask, coordinate broadcast, sample-count rule with sqrt/arcsinh/cbrt per pair, argsort/unique); `_streaming_tile_shape` computed every build only for dead branches | `bor/streaming.py:1826, 1838, 2346, 2356, 445-456, 925`, `kernels.py:580-600` | ~5% of the far build together |
| R-BoR-4 | Locks in hot loops: `_half_grid_tables` global lock once per group per tile (17,000 acquisitions per build, 30 us serial vs 270-430 us at 8 threads); the per-family band-add lock held for the whole strided add | `bor/kernels.py:527`, `bor/streaming.py:1764` | 8-12% + 2-3% of far-build thread time |
| R-BoR-5 | ACA pivot rows and columns of the compressed far store discard 2/3 of each contracted band (one node row kept of three); evidence dictionaries updated per (tile, family, mode, uv) component in the compressed cross store | `bor/compressed_far.py:264, 313, 317`, `compressed_cross.py:244-246` | ~10% of the compressed block phase; negligible |
| R-BoR-6 | Compressed rectangular cross store is 3.7x slower than the dense one (serial tile loop, SVD + full residual check per component) for a 25% memory saving | `bor/compressed_cross.py:83, 225-246`; 2.33 s vs 0.63 s on 300x200 elements | material solves with >= 1,000-node surfaces |
| X-1 | Zero production callers (static scan + per-subsystem grep): see Appendix A | | maintenance surface; all of it is hashed into every fingerprint |

## 5. Computational-efficiency findings, ranked

Confidence: CONFIRMED = measured here or unambiguous from code; PLAUSIBLE = inferred, not measured end to end.

### 5.1 2-D

| Rank | ID | Finding | Evidence | Fix sketch | Expected gain | Confidence |
|---|---|---|---|---|---|---|
| 1 | F-2D-1 | Polynomial near-moment accumulation is a 70-pass loop: `_evaluate_chunk` materializes (tasks x samples) x and y, then per channel 4 copies + 12 multiplies + 16 row reductions + 3 products (x2 channels); kernels are only 40-45% of its time. Every task in a chunk shares its quadrature nodes up to an affine map, so the moments are one GEMM `kernel(T x q) @ monomials(q x 16)` plus a 4x4 binomial transform per task | `polynomial_quadrature.py:449-513`; 5,666-element P3 mesh at 3 GHz, one thread: `near_blocks` 7.7-8.4 s per wavenumber, `_evaluate_chunk` 6.9-7.8 s of which kernels 3.1-3.2 s; in-situ GEMM replica 19.67 -> 13.46 ms per chunk (order 36), 10.97 -> 7.12 ms, max relative difference <= 3.7e-16 | GEMM formulation of the moments; keep per-thread bitwise order (tests require 1-vs-N-thread equality) | 15-18% of a 1 GHz certified solve, 10-12% at 3 GHz | CONFIRMED |
| 2 | F-2D-2 | No attenuation cut in the dense far pass for lossy regions: every pair at ratio >= 3 is evaluated at full order; the coating region (k = 1262-614j) owns 470,000 of 581,000 far pairs per traversal, and 70% of its far pairs have -Im(k)(d - Li - Lj) > 32 (kernel below 1e-14 of the near-diagonal scale); the compressed path already drops such routes with the Hankel envelope bound at cut 32 | `operators.py:2270-2277` vs `compressed/regional_coefficients.py:12, 91, 180-203`; census of 329,090 coating pairs | AND the far mask with `attenuation <= cut` from the centre distance and lengths already in hand; `far.c:116` skips masked pairs for free; record the bound in the evidence | ~55% fewer far kernel evaluations, ~11% of a 3 GHz solve; the far share grows like N^2 | census CONFIRMED, implementation PLAUSIBLE |
| 3 | F-2D-3 | Fixed-order near pairs lack a moment cache across degrees (R-2D-3) | see R-2D-3 | store scaled monomial moments to degree 3 keyed like `_moment_key`, project per degree | 6-7.5% | CONFIRMED |
| 4 | F-2D-4 | Worker-thread table rebuilds and the per-pair table-reach loop (R-2D-4, R-2D-5) | see above | resolve the table on the calling thread and pass it in (as `far_table` already is), or run `map_checked` jobs under `contextvars.copy_context().run`; vectorize the reach from the existing `p0`/segment arrays | 5-7% | CONFIRMED |
| 5 | F-2D-5 | Far-order grading pinned per tile: `_graded_far_order` returns the cap (8) for \|k\|Lmax > 3.0 (the calibration table ends at 3.00), `kl_max` uses the tile's longest panel and `ratio_min` its closest pair: at 3 GHz every coating tile ran at order 8 and 54 of 66 air tiles took the "ratio < 5" column although most pairs are at ratio >= 10; cost ~q^2 (8 -> 6 saves 35%) | `operators.py:1360-1393, 2297-2300`; `run_orders_3ghz.txt` | extend the calibration beyond \|k\|L = 3 for attenuating media (or rely on F-2D-2); split a tile's far pairs into two native calls by ratio bin | 20-45% of the far kernel work | PLAUSIBLE |
| 6 | F-2D-6 | Closed PEC/IBC/coated contours: when W is wanted both tile axes are clamped to max(graded order, W floor 8/9, obs_order 8, src_order 8), so far grading never goes below 8; the W floor is calibrated only at separation 3 | `operators.py:2308, 1398` | calibrate W per ratio bin like S/K'/D and drop the obs/src clamp; 5-7 points on ratio >= 10 tiles | 1.3-2.6x on far kernel cost of the combined formulation (not exercised by the airfoil) | PLAUSIBLE |
| 7 | F-2D-7 | Planning repeated inside the certified solve and in the driver (R-2D-1, R-2D-2, R-2D-7); the cost model does not discriminate (dense and compressed within 0.3%) and once picked the slower backend by tie-break (14.1 s dense vs 11.8 s compressed at the 3 GHz survey) | `adaptivity.py:151-161`, `policy.py:51-57, 90-103`; profiling campaign | skip the in-solve forecasts when a batch selection exists (use its candidates and retry order) or ship the planner's resource records so `_refresh_memory_forecast` reprices in ~1 ms; forecast only the hp pair when eligible; prefer the lower forecast or allow one calibration run per family when candidates tie | 4-10% of small units; up to 16% where the tie-break errs | CONFIRMED |
| 8 | F-2D-8 | Diagnostics and factor policy: condition estimates 16% of LU (dense) and 35% of the ACA build (compressed) (R-2D-12); `sweep.solve` QR for LU (R-2D-13); HODLR engages at 10,000 unknowns but on this host builds 3x slower than LU at 8,608 and no faster at 11,284, with 47% of its build in `Block.dense` gathers of an interleaved layout and slower downstream phases; the learned crossover cannot calibrate (R-2D-10) | `linalg/dense.py:46, 180`, `compressed/factor.py:267-284`, `linalg/sweep.py:27-43`, `linalg/hierarchical.py:88-134, 222-270`, `linalg/crossover.py` | accumulate column maxima in `store_tile` and probe with one relaxed application; widen the one-norm probes (4-8 columns cost the same memory-bound pass as 2); per-factor-variant cost model in `sweep.solve` (direct for LU below ~20k unknowns); permute A into spatial order once so HODLR blocks are contiguous slices and gather each panel once per block; re-measure the HODLR threshold on the target hosts or let the threshold learn from a one-off paired run | condition 3x cheaper (4-5% of a solve); sweep 3%; HODLR build ~45%; threshold choice up to 2-3x on 10-14k systems | CONFIRMED (costs), PLAUSIBLE (HODLR at 14k) |
| 9 | F-2D-9 | Check-only 20-point rule: `near_blocks` always evaluates orders 20 and 36, and on the 5,666-element mesh every one of 17,010 singular pairs converged at 36 (no refinement), so 800 of 3,392 samples per pair (24%) serve only the error estimate | `polynomial_quadrature.py:600-601` | a cheaper estimator (e.g. 24 vs 36, or the 36-rule's own split); this is the same certification trade-off the 24 September audit kept for BoR | ~20% of polynomial near work | PLAUSIBLE (accuracy trade-off) |
| 10 | F-2D-10 | Per-task Python in `near_blocks` (~18 us per task: `_classify` 13.4, `_project` 5.0, `_moment_key` 1.7) = 0.6 s per 17,010-pair call (all of the fully cached P3 pass); serial far commit 19% of the far wall at 2 threads; P1 reference path per-pair loops (touching rule 217 ns per sample vs 110 for the box rule); Python per-element builders (`sparse_mass` 12.6 ms x2-6 per assembly at 5,666, `geometric_near_pair_count` 96 ms at 10,000 x12) | `polynomial_quadrature.py:37, 333, 557`, `operators.py:1545-1593, 617-636`, `assembly/mass.py:11-13`, `regions.py:14` | vectorize `_classify`, batch `_project`; dependency-aware concurrent commit (bitwise order kept); vectorize the touching rule and builders | 2-7% of near work; grows with thread count | CONFIRMED |
| 11 | F-2D-11 | Overhead items: three source hashes and three verifications per unit, worker respawn, serial parent planning, GUI cache key, BoR merge double write (R-D-1 to R-D-5) | drivers audit | one memoized source identity shared by provenance, timing key and checkpoint identity; one hash + two stat checks per unit, drop the post-export check; raise `max_tasks_per_child` to 16-32 or recycle on RSS; start the pool before planning; remove the allocation from the forecast key; return the export payload instead of re-loading | ~0.3 s per unit (6% of a 5 s unit, 2% of 15 s) | CONFIRMED |
| 12 | F-2D-12 | API: bare public calls run single-threaded (DEFAULTS assembly_threads=1, blas_threads=1 unless `execution_options=automatic_options()` is passed) | `execution/options.py:16-33, 530-536`; 60 s vs 14 s | make `solver_method='auto'` resolve threads to 'auto' when no profile is supplied | up to 4x for API users; none for GUI/driver users | CONFIRMED |

### 5.2 BoR

| Rank | ID | Finding | Evidence | Fix sketch | Expected gain | Confidence |
|---|---|---|---|---|---|---|
| 1 | F-BoR-1 | Preparation to the cap for modes never run (R-BoR-1). The far build is only ~30% mode-independent (native sampling 26% + bookkeeping) and ~50% per-mode (transform GEMMs, contractions, scatter), and its initial horizon already equals the cap for bandwidth >= ~12, so a mode-incremental far build saves little (<= 6-8% of the far build) and loses 17-35% whenever a second range is needed; the near preparation, which dominates small and medium bodies and already has a band mechanism (`_prepare_near_family_band`, `mode_start`, `ModalBands`), is where the unused modes cost | section 3.2; far audit m2/m3/m6; A/B pinned cap -7 to -34% | prepare near bands to the sweep's predicted horizon and extend them on demand; keep the far build whole; tighten the automatic starting margin (max(12, ...) above the bandwidth) where the measured tails justify it | 5-7% of wall from the near kernels, more from far tables and the sweep overshoot (section 3.3); the band-extension defect must be fixed first | CONFIRMED (cost), CONFIRMED (mechanism limits; section 3.3) |
| 2 | F-BoR-2 | One-row far tiles on large meshes: the 1 GB tile budget / 8 threads live-set model yields one test row per tile from ~2,000 elements, doubling sampler calls and contraction work for the same samples (13.35 s vs 7.03 s at 5 rows, 6.47 s at 20 rows); the `tested` buffer is 10 kinds x P x orders x 16 B and is copied five times into nodal sums and nine times into term sums | `bor/streaming.py:807-934, 176-246`; far audit m3/m12 | contract the test side per node (strided windows, weights [re+1, 5, 2g], K=2g): halves the live set and removes the nodal copies; contract per left-kind group; or price a 2-4 GB tile budget | 10-20% of large conductor solves (far build 1.9x) | CONFIRMED |
| 3 | F-BoR-3 | Dense-tables path per-mode contraction 38x the streamed read, both triangles sampled, 7x memory; selected by the 2 GB tables rule exactly on the small bodies where the sweep is a large share | `bor/solver.py:4180, 943-1016, 1706`; 117 ms per mode vs 3.1 ms; 10.4 s vs 8.8 s whole solve | let `assembly='auto'` stream always (the "removing the tables path" item deferred on 24 September), or store tables order-major and reuse the streamed `_efie_band` | 15-25% of small-body solves | CONFIRMED |
| 4 | F-BoR-4 | GIL-bound tile threads: ~20% of each tile is Python/small-NumPy glue (sample-count rule 7.6%, argsort 2.4%, scatter 8%, 2*pi scaling 4%), so 8 threads serialize on it (efficiency 0.47; 4x2 OpenMP within 12% of 8x1) | far audit m7/m8; `bor/kernels.py:575-660`, `bor/streaming.py:428-475` | one native entry per tile group that samples, transforms (trig recurrence) and writes `[pairs, orders]` directly; or process workers into shared-memory stores | 20-30% of the far build (10-15% of large solves) | CONFIRMED |
| 5 | F-BoR-5 | Per-mode assembly overheads: packed-EFIE unpack by boolean-mask assignment plus symmetrization (13 ms at 801 nodes vs LU 38 ms; 100 ms at 2,161 vs LU 466 ms); `Z[np.ix_(mask, mask)]` gathers; NumPy gathers in `_efie_tables_into`/`_contract_left_into`/`bmm_einsum` (52 ms per mode at 370 unknowns) | `bor/streaming.py:370-394, 291-323`, `bor/solver.py` mode assembly | one blocked pass writing both triangles from the packed rows; mask by slicing when the mask is contiguous | 20-35% of the LU time per mode; ~5% of the mode phase | CONFIRMED |
| 6 | F-BoR-6 | Fresh buffers per chunk (sampler `np.empty` 22-30 MB per chunk, 125 MB `np.zeros` outputs per tile first-touched by the scatter); the real cosine table stored as complex doubles the transform flops; duplicate zeroing and bookkeeping (R-BoR-3); hot-loop locks (R-BoR-4) | `bor/kernels.py:304-318, 545-548`, `bor/streaming.py:1764`, `bor/kernels.py:527` | per-thread reusable buffers with GEMM `out=`; planar or in-kernel transform; cache the half-grid tables per thread; add bands outside the lock | 3-5% + <= 6% + ~5% + 8-12% of far-build thread time | CONFIRMED (costs), PLAUSIBLE (transform) |
| 7 | F-BoR-7 | Process-pool near preparation spawns for a 1.4 s job on the certified fine mesh (threshold pairs x modes >= 8,000), ramps 0.3-0.5 s and dips 0.25 s at teardown; `workers=4` caps every phase at ~4 of 8 cores | profiling campaign G/H/I; `bor/near_parallel.py:80-101`; `run_local_bor.py` WORKERS_PER_UNIT | raise the process threshold or reuse one pool per unit; size `workers` from the CPU allocation (8 for a lone solve) | up to 2x on preparation-bound solves run alone | CONFIRMED (utilization), PLAUSIBLE (gain) |
| 8 | F-BoR-8 | Compressed rectangular cross store 3.7x slower than dense (R-BoR-6); compressed self store issues 9x more sampler calls and wastes 2/3 of each ACA pivot band | `bor/compressed_cross.py:83, 225-246`, `compressed_far.py:264-317` | run cross tiles on the tile threads and batch the SVDs, or keep dense rectangular streams unless memory forces compression; contract only the kept node row in pivots | material solves with large surfaces; ~10% of the compressed block phase | CONFIRMED / PLAUSIBLE |

## 6. Justification summary: what is necessary and why

- **Geometry preflight, material fingerprints, attestation, manifest**: necessary once per run; they are what makes a
  published field traceable to exact inputs and solver bytes. Their repeats (three verifications per unit, three
  independent source hashes) are not.
- **Backend forecast and RAM admission**: necessary once per request; the gate prevents an oversized solve from
  starting. Repeating it per adaptive step rebuilds meshes the planner already built; the cost prior cannot tell the
  backends apart, so the forecast's value today is the RAM admission, not the choice.
- **hp certification pair (P2 then P3)**: required by the certification policy; the P2 solve exists only to produce the
  comparison field. The P3 result is what ships. The survey mode solves the global P1 mesh and is slower than the
  certified pair on this geometry; survey users would be better served by the hp candidate with the certificate marked off.
- **Kernel tables, far tiles, near quadrature, TM derivation, paired assembly**: necessary; this is the operator. The
  redundancies are inside them (tables rebuilt per worker thread, moments recomputed across degrees, attenuated pairs
  evaluated, orders pinned by the worst pair of a tile).
- **Factorization with residual gates**: the residual products are the correctness certificate (backward error 1e-12
  against the original coefficients) and must stay. The condition estimate is a diagnostic for a 1e6 gate; it can be
  made several times cheaper without changing the gate. HODLR is a memory tool on this host, not a speed tool.
- **BoR far and near preparation**: necessary for every mode the sweep uses; the automatic cap over-prepares by the
  starting margin, and the far build's live-set model under-fills tiles on large meshes.
- **BoR mode sweep**: necessary; its per-mode cost is already small except for the packed-block unpack and NumPy gathers.
- **Certified BoR fine solve**: policy; nothing is shared across the two meshes, so it costs ~1.6x the base solve.
- **Telemetry (stage timers, RSS sampler, timing history, crossover learning)**: diagnostics; the timing history and
  crossover learning never influence a default run and should either be given the paired samples they need (one
  calibration run per family) or removed from the hot path.

## 7. Recommended order of work

Quick, low-risk (days): F-2D-4 (two one-line fixes, 5-7%), F-2D-12 (API default), F-2D-7 part (skip in-solve forecasts
under a batch plan; forecast only the hp pair), F-BoR-3 (stream always), F-BoR-6 (buffers, lock, duplicate zeroing),
F-BoR-5 (fused unpack), R-2D-12/R-2D-13 (cheaper condition estimate, direct solves for LU), F-2D-11 (shared source
identity, verification count, respawn interval), X-1 (delete dead code).

Structural (weeks, with bitwise or 1e-15 equivalence tests): F-2D-1 (GEMM moments), F-2D-3 (moment cache across
degrees), F-2D-2 (attenuation cut) and F-2D-5/6 (order grading), F-BoR-1 (near bands to the horizon with extension),
F-BoR-2 (per-node far contraction, tile budget), F-BoR-4 (native per-tile-group sampling + transform), HODLR layout
permutation and threshold re-measurement, backend tie policy.

## 8. Limits of this audit

Timings are from one 8-core Windows host; relative shares shift with core count and memory bandwidth (the HODLR and
tie-break findings in particular). The 2-D measurements use the airfoil geometry, which is a layered coating with open
material junctions, so the combined D/W potentials, the W far floor and `assemble_pair` were traced from code but not
measured end to end (F-2D-6 is marked plausible for that reason). BoR measurements use PEC spheres and a cylinder
(conductor CFIE); material, coated and junction solvers were traced and spot-measured (cross store) but not profiled
end to end. No numerical change was
proposed that alters a quadrature rule or a tolerance; the two findings that would (the check-only 20-point rule and the
automatic mode margin) are explicitly marked as accuracy trade-offs.

## Appendix A. Functions with no production callers

Static scan of every top-level function and class in `ghost_backend` (excluding `tests/`, `validation/`,
`data_tools/`), confirmed by the per-subsystem greps. "tests" = references from tests only.

- `bor/kernels.py`: `_modal_kernels_near_rule` (787), `_mfie_kernels_near_rule` (1113), `_ibc_kernels_near_rule`
  (1314), `_project_pm_brackets` (998): the pre-graded NumPy near rules (tests only); non-banded branches of
  `modal_kernels_fft` (726-758), `mfie_kernels_fft` (968-995), `ibc_kernels_fft` (1280-1311) while `BANDED_FFT` is a
  constant True.
- `bor/solver.py`: `_map_near_pairs` (614), `bor_basis_bytes` (3488), `_efie_near_asymmetry` (5041),
  `cylinder_generatrix` (7117).
- `bor/streaming.py`: `_efie_terms` (28), `_contract_source_side` (78); the non-banded bodies of `_sample_G`
  (1941-1968), `_sample_brackets` (1870-1929) and the cross equivalents (2404-2445, 2451-2511) with their
  `_native`/`_q`/`_te`/`_cols` setup (1710-1714, 925, 1799-1803); the unreachable `_symmetrize_efie_blocks` branch
  (1855); test-only readers `full_blocks`, `efie_blocks`, `bracket_blocks`, `stored_blocks`, `_full_efie_mode`,
  `_read_mode`.
- `bor/dispatch.py`: `resolve_automatic_factorization` (481), `_resolve_direct_factorization` (640).
- `bor/polynomial.py` (research prototype, `solve_bor_polynomial` 212).
- `twod/solver.py`: `_residual_norm_many` (457), `_solve_te_robin_mfie` (1571; test contract only), `_make_elem_mask`
  (1871), `solve_monostatic_rcs_2d_certified_single_polarization` (3918), `solve_bistatic_rcs_2d_certified_single_polarization` (3965).
- `twod/operators.py`: `_run_tiled_obs_blocks` (1513), `_assemble_linear_mass_matrix` (3113),
  `_assemble_linear_weighted_mass_matrix` (3124), `_hypersingular_block_from_s_block` (1199),
  `_linear_element_incident_load_many` (2733), `_linear_element_incident_dn_load_many` (2759),
  `_integrate_linear_pair_generic` (774) with `_integrate_linear_pair_recursive` (664), `_integrate_linear_pair_box`
  (98), `_integrate_linear_self_duffy` (408), `_integrate_linear_touching_duffy` (506), `_green_2d` (3181) (reachable
  only when the exact self series returns None), `_far_tile_numpy` (2480, fallback only).
- `twod/assembly`: `compact.scatter_basis_columns` (65), `compact.scatter_operator_add` (74), `separation.close_pairs`
  (105) (imported, never called); `polynomial_pair.py` and `dense_pair_storage.py` (gated by `GHOST_POLYNOMIAL_PAIR`,
  default off; `dense_system` is still called on every assembly); `kernels.incident` (43), `kernels.incident_dn` (54).
- `twod/polynomial_quadrature.py`: `block` (60), `near_block` (124), `log_moments` (10), `hypersingular` (23); the
  width > 2 branches of `_sk_blocks_near_linear` (984-992) and `_single_layer_block_linear` (954-956).
- `twod/formulations/regions.py`: `_assemble_system_fresh` (208) and `combined_regions.add_corrections` (86).
- `twod/basis.py`: `stiffness_block` (70). `twod/nystrom.py`: research prototype (`solve_smooth_pec`, `ellipse`).
- `execution/cpu.py`: `cached_operator` never stores in production (`reuse_operators=False`).
- `compressed/`: `analytic_far.py` (whole), `fast_far.py` (only `compressed_far_method='verified_cur'`),
  `projection.py` and `retained_storage.py` (only the polynomial-pair projection path), `recycling.py` internals
  (`InverseCache`, `retained_size`, eviction; capacity 0), `inverse.py` `Oracle` (7-13), `Block.matmul/project`,
  randomized builder, `Node.matmul`, `inspect_block`, `solve_checked`; `block_products.ProductTeam.apply` and
  `product_scope`; `operator.py` `add_tile` (299), `get` (389), svd branch (147-153), projection plan (456-489);
  `regional_coefficients.hankel_envelopes` (12-26); `linalg/refined_lu.RefinedLU` (mixed precision never requested)
  and the mixed/numpy fallbacks in `linalg/dense.py`; `linalg/hierarchical.Block.row/col/error` and `compress` with
  the hierarchical `Block` (production ACA uses the inverse module's block).
- `execution/provenance.py`: `write_output_attestation` (160), `verify_output_attestation` (250),
  `write_artifact_manifest` (325), `write_artifact_in_progress` (407), `verify_artifact_manifest` (474),
  `verify_component_output_manifest` (592): no callers anywhere, about 530 lines.
- `hpc/scheduler.py`: `predict_2d_resources` (689), `unit_peak_gb` (771); `hpc/common.py`: `stage_geometry` (248),
  `latest_run_dir` (262) (tests only).
- `io/naming.py`: `format_base` (113), `group_solver_files` (137), `join_grims` (154), `parse_solver_name` (35),
  `pair_variants` (262); `io/grim.py`: `compute_linear_from_dbke` (204); `io/viewer_bridge.py`: `to_grid` (75);
  `geometry/io.py`: `save_snapshot_geo` (570); `geometry/measurements.py`: `closest_primitive_points` (87).
- Drivers: unused imports `_workflow_provenance` (all four), `Pool` and `copy_configuration` (`run_local_bor.py`).
- `assembly/` (feature workflow, outside the solver scope): `tag_component`, `verify_body_artifact_bundle`,
  `interference_metrics`, `ContributionInspector`, `is_closed_loop`, `_skin_limit` and the `place_features.py`
  helpers.

## Appendix B. Measurement artefacts

All under the session scratchpad `scratchpad/`: `agent_profile/` (campaign scripts, per-run metadata, timelines,
cProfile tables, `campaign_report.md`), `agent_2d_ops/` (assembly benchmarks and census scripts), `agent_2d_compressed/`
(dense/compressed/HODLR overhead scripts and logs), `agent_bor_far/` (m0-m12 far-build measurements), `agent_drivers/`
(fingerprint, planning, export and pool-churn measurements), `count_2d_steps.py` (call counter), `bor_cap_ab.py`
(mode-cap A/B), `reachability.py` (static scan).
