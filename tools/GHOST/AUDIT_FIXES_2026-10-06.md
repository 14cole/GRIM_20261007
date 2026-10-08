# GHOST audit fixes, 6 October 2026

This release implements the recommendations of the 6 October workflow audit
(`SOLVER_WORKFLOW_AUDIT_2026-10-06.md`) under one rule: no published field may
move by more than 1e-8 of its peak against the previous code. Every change was
measured before and after on the same inputs in fresh processes; the worst
change over the reference set is 3.5e-11 of the peak (section 3).

`scripts/audit_2026-10-06/` holds the accuracy and timing harness
(`bench.py run OUTDIR --set full`, `bench.py compare REF NEW`), the recorded
fields and timings before (`baseline/`) and after (`final/`) the changes, the
comparison (`compare_final_vs_baseline.txt`), the equivalence and A/B scripts
quoted below, and `changes.diff`, the unified diff of every source change
against the previous code (test changes are listed in section 2.4; the
previous tree itself is in the session scratchpad, `backup_before_fixes/`).

## 1. What changes when you rerun

- **2-D**: results agree with the previous version to 1.3e-12 of the peak on
  the airfoil (coated, 1-10 GHz, certified and survey), 3.5e-11 on the TYPE 4
  coating example and bitwise on the TYPE 1/2/5 material examples. The
  certified airfoil solves are 13% faster at 1-3 GHz and 34% at 10 GHz (the
  hierarchical factor builds from a spatially ordered copy), the 3 GHz survey
  9% faster; the forced compressed backend is unchanged. A bare Python API
  call (`solve_monostatic_rcs_2d(...)` without `execution_options`) now runs on
  the host's physical cores like the GUI and batch profiles, instead of one
  thread.
- **BoR**: results agree to 3e-15 of the peak. Conductor solves stream their
  far blocks by default (`assembly='tables'` remains an explicit choice), the
  near-pair preparation uses the CPU allocation's physical cores when the
  caller asked for a parallel solve, and the automatic cap-expansion retry no
  longer fails on closed bodies. Whole solves are 7-10% faster on the small
  reference bodies (126-342 unknowns) and 26-37% on the larger ones (sphere
  ka = 10, cylinder at 6 GHz); the far build of a 2,000-element sphere is
  1.7x faster (65 -> 38 s) through the tile planner.
- **Drivers**: pool workers are recycled every 8 units (2-D) and 4 pairs
  (BoR) instead of 4 and 2; the source bundle is hashed once per process per
  file state for the provenance manifest and the timing-history key instead
  of on every call. Checkpoint identities are unchanged.

## 2. Changes

IDs refer to the audit report. "Equivalence" is the measured change of the
coefficients or fields the change touches; "bitwise" means byte-identical.

### 2.1 2-D

| ID | Change | Files | Equivalence | Measured effect |
|---|---|---|---|---|
| F-2D-4 (R-2D-4) | Near-pair batches run in a copy of the caller's context, so worker threads find the solve's kernel-table store and moment cache instead of rebuilding tables per thread and domain; one thread builds a missing table and the others wait for it | `twod/polynomial_quadrature.py` (`map_checked`), `twod/assembly/kernels.py` (`cached_table`) | bitwise (same tables) | kernel-table builds per certified 3 GHz solve 21 -> 8 (near tables 18 -> 5, one per wavenumber and domain) |
| F-2D-4 (R-2D-5) | The near table's reach is taken over endpoint arrays instead of four norms per pair in Python | `polynomial_quadrature._table_for_ends` | bitwise (power-of-two domain) | removes a 5 ms Python loop per batch |
| F-2D-1 | Polynomial near moments are one matrix product against the base-node monomials plus a per-task binomial change of variable (identity on unit intervals) instead of a 70-pass accumulation loop | `polynomial_quadrature._monomial_moments` | <= 7e-15 relative per block on unit intervals, <= 2.4e-13 on bisected close pairs (`scripts/audit_2026-10-06/equiv_near_blocks.py`); 1 vs 4 threads bitwise | `near_blocks` chunk 1.5-1.6x faster |
| F-2D-3 (R-2D-3) | Polynomial fixed-order (tensor-Gauss) near pairs keep their kernel samples as scaled monomial moments in the certified request's moment cache; the cubic candidate projects the quadratic candidate's samples (50% of the 22,376 pairs of a 3 GHz certified solve are served from the cache) | `twod/operators._box_blocks_polynomial`, `_box_monomial_moments`, `polynomial_quadrature.box_moment_keys` | the nodal projection is the same linear map (`phi = Q c`), to rounding; P1 pairs unchanged (bitwise) | half the box-rule integration of a certified solve |
| F-2D-2 | Far pairs of an attenuating medium whose kernels are negligible are neither evaluated nor scattered: `-Im(k) * (centre distance - half lengths) >= 40` (kernel below 1e-18 of the near-diagonal scale by the Hankel envelope bound the compressed backend already uses at 32). The near/far classification and the graded orders are unchanged. `GHOST_FAR_ATTENUATION_CUT=0` disables the cut | `twod/operators.py` (`FAR_ATTENUATION_CUT`, `_far_tile`) | 3 GHz survey fields change 1.3e-12 of the peak, certified 3 GHz 3e-13 | 3 GHz survey operators 5.5 -> 4.2 s (with the table fixes) |
| R-2D-13 | `sweep.solve` 'auto' solves batches directly for LU-backed factors (dense LU, mirror halves) of 4,096 to 60 x (batch columns) unknowns: a LAPACK solve streams the whole factor once whatever the column count and the pivoted QR costs 2.4x its flop model, so reconstruction lost 0.03 s per 256-column batch at 8,608 unknowns. Compressed and hierarchical factors, and factors below 4,096 unknowns, keep the QR | `linalg/sweep.py` (`LU_DIRECT_MIN_UNKNOWNS`, `LU_DIRECT_RATIO`, `_lu_backed`) | both paths pass the same 1e-12 backward-error gate; fields change <= 1e-12 | survey rhs stage 0.91 -> 0.83 s |
| F-2D-7 (R-2D-1) | The adaptive controller reuses the request's (or the batch planner's) admitted forecast for its two initial candidate meshes instead of rebuilding and re-pricing them (two hidden forecasts per certified solve); local-refinement retries and the conservative HP4 retry still forecast | `execution/options.py` (`request_selection_scope`), `execution/selection.py`, `twod/adaptivity.py` (`_request_forecast_covering`) | planning only; same backend choice | 0.25-0.33 s per certified solve |
| F-2D-12 | A bare automatic request (`solver_method='auto'`, no `execution_options`, no `GHOST_ASSEMBLY_THREADS`) resolves `assembly_threads` and `blas_threads` to 'auto' | `execution/options.configured_execution` | assembly is bitwise across thread counts; BLAS products differ at rounding | API users: 60 s -> 14 s at the 3 GHz survey |
| R-2D-9 | `NearStore.write` skips the copy of a slice the batch integrator already wrote in place | `twod/assembly/near_store.py` | bitwise | 10 ms at 5,666 elements |
| F-2D-8 (HODLR) | The hierarchical factor builds from a spatially ordered copy of the matrix when the host has twice the matrix size free (`ORDERED_COPY_MAX_BYTES` 8 GiB): blocks become contiguous slices and products act on views; the copy is released after the build. The gather path remains for tight hosts | `linalg/hierarchical.py` (`OrderedBlock`, `ordered_copy`) | same sampled entries; products in a different blocking, refined to the same 3e-15 gate | see section 3 (10 GHz airfoil) |

### 2.2 BoR

| ID | Change | Files | Equivalence | Measured effect |
|---|---|---|---|---|
| F-BoR-3 | `solve_bor` streams its far blocks whenever `assembly='auto'` (the dense-tables path contracted `[P, P, M]` tables per mode at 38x the streamed read and sampled both triangles); the direct and snapshot planners price conductors and sheets the same way. Material solvers keep their 2 GB table rule | `bor/solver.py`, `bor/dispatch.py` | tables and streamed blocks agree to 3e-15 | PEC spheres: 0.59 -> 0.61 s (40 elements), 4.15 -> 3.63 s (200), 6.07 -> 5.04 s (300) |
| F-BoR-7 | Near-pair preparation of a parallel request (`workers > 1`) uses `min(CPU allocation, physical cores)` workers independently of the mode-worker count (a serial request stays serial); the memory plan prices the same count | `bor/solver.plan_bor_mode_workers` | process and thread results are bitwise identical to serial | near preparation 4.13 -> 2.59 s on a 300-element sphere (8 instead of 4 workers) |
| near-band defect (section 3.3 of the audit) | A modal band extension (`mode_start > 0`, the automatic cap-expansion retry) is evaluated over every order from 0 by the native rule and sliced when stored, so the acceptance check keeps the full-range scale; the band-only check never converged for near-axis points on closed bodies and the retry raised | `bor/kernels.py` (`_run_near_group`, `_refine_near_chunk`) | band values bitwise equal to a full build of the same cap | retry path works on spheres; `test_extended_near_bands_preserve_old_storage_and_match_full_cap` passes |
| grouping (agent finding 3) | Near-point layout groups are formed by sorting packed int64 keys instead of `np.unique(axis=0)` plus a second stable argsort | `bor/kernels._layout_groups` | identical groups and member order (checked against `np.unique`) | 8% of a serial near preparation |
| parity (agent finding 3) | For nonnegative orders the bracket parity moments are returned directly instead of through an identity gather | `bor/kernels._graded_near_kernels` | bitwise (the m = 0 sine column is zero either way) | 3% of a serial near preparation |
| F-BoR-6 (R-BoR-4) | Each tile thread keeps references to the half-grid transform tables it uses; the shared store's lock is reached only on a miss | `bor/kernels._half_grid_tables` | bitwise (same arrays) | 8-12% of far-build thread time at eight threads |
| R-BoR-3 | The banded sampler already returns masked (near, lower-triangle) pairs as zeros; the second zeroing pass per tile is skipped | `bor/streaming.py` (`_build_range`, both streams) | bitwise | per-tile Python loop removed |
| F-BoR-2 | The banded tile planner prefers 16 test rows with narrower source ranges over one or a few full-width rows at the same live set (`STREAM_TILE_MIN_ROWS`, `STREAM_TILE_MIN_SOURCES`): the contraction's inner dimension is rows x Gauss points | `bor/streaming._plan_banded_tiles` | blocks agree to 5e-16 (different summation order at tile boundaries) | far build of a PEC sphere at 5 GHz, 8 threads: 800 elements 6.63 -> 5.44 s (5 full-width rows -> 16 x 263), 2,000 elements 65.3 -> 37.7 s (2 full-width rows -> 16 x 251; `scripts/audit_2026-10-06/tile_old_plan_2000.py`) |
| F-BoR-5 | Packed EFIE rows are unpacked through cached (row, column) index pairs instead of a boolean mask scanned per mode (blocks up to 3,072 nodes) | `bor/streaming.py` (`_unpack_upper_into`, `_unpack_masks`) | bitwise | part of the 20-35% per-mode unpack cost |
| R-2D-13 (BoR) | The LU rule above also stops the 'auto' QR probing of small mode systems (74 columns of a 598-unknown system cost 0.1 s per solve) | `linalg/sweep.py` | as above | 1.5% of a 300-element solve |

### 2.3 Drivers and overhead

| ID | Change | Files | Measured effect |
|---|---|---|---|
| R-D-2 | Backend source files are hashed once per process per (path, size, mtime) for the provenance manifest and the timing-history key (`source_file_digest`); output artifacts and the checkpoint identity (whose test contract is content-based) are never memoized | `execution/provenance.py` | 53 ms -> 3 ms per public call after the first |
| R-D-3 | Pool workers are recycled every 8 units (2-D) and 4 pairs (BoR) | `run_local_monostatic.py`, `run_local_bor.py` | 65 ms per 2-D unit, 130 ms per BoR pair |

### 2.4 Test updates

- `test_audit_fixes_2d_solver.py`: the plain-API thread test now asserts the host thread count and that
  `GHOST_ASSEMBLY_THREADS=1` still pins one thread.
- `test_performance_updates.py`: the near-preparation preview expects the CPU allocation's physical cores
  for a parallel request.
- `test_september_round5_fixes.py`: the snapshot test requests `assembly='tables'` explicitly; the
  single-precision diagnostic expects one streamed dense plan for a conductor.
- `test_solver_compression.py`, `test_twod_remaining_performance.py` and `test_automatic_hpc.py` pass unchanged:
  the hierarchical build keeps its `_build(ids)` signature, every moment GEMM has the same shape (so a forced
  7-pair near batch reproduces the default assembly bitwise), and a reused batch forecast reports the live
  admission budget.
- The complete suite (`ghost_backend/tests/run_suite.py`) was run before and after. Seven failures pre-exist on
  this host with the previous code and are unchanged: four in `test_september_round5_fixes.py`
  (`test_automatic_plans_go_from_tables_to_streaming_before_compression` at the 2.9 GB limit,
  `test_run_setup_summary_resolves_its_plan_under_the_callers_options`,
  `test_single_precision_snapshot_streams_instead_of_failing_admission`,
  `test_snapshot_chooser_prices_streaming_before_compression`; host-calibrated memory thresholds),
  `test_september_round11_bor_families.py::test_automatic_snapshot_call_is_admitted_where_the_old_preview_chose_tables`
  (the same calibration) and the two `test_process_tile_window_bounds_results_and_recovers_in_original_order`
  cases of `test_runtime_allocation_updates.py` (a fake executor whose futures return a tuple of two where the
  current window reader unpacks three).

### 2.5 Dead-code sweep (second pass)

Every top-level definition of the production package with no production reference (AST scan of
`ghost_backend`, references in tests, documentation, `GRIM_Backend` and `data_tools` checked by hand;
`scripts/audit_2026-10-06/dead_code_references.py` lists the candidates and their referrers,
`dead_code_sweep.py` is the applied edit) was removed or moved; the bench fields are bitwise identical
before and after.

| Removed | Kept where tests still need it |
|---|---|
| modules `bor/polynomial.py`, `compressed/analytic_far.py`, `twod/nystrom.py` (with their tests `test_bor_polynomial_meridian.py`, `test_analytic_far_prototype.py`, `test_nystrom.py`; `scripts/check_headless.py` no longer lists `nystrom`) | the legacy BoR near rules `_modal_kernels_near_rule`, `_project_pm_brackets`, `_mfie_kernels_near_rule`, `_ibc_kernels_near_rule` now live in `tests/legacy_near_rules.py` (four test modules import it) |
| `bor/solver.py`: `_map_near_pairs`, `bor_basis_bytes`, `_efie_near_asymmetry` (now a helper of `test_bor_compute_reuse.py`) | |
| `bor/streaming.py`: `_contract_source_side` | |
| `execution/provenance.py`: `write_output_attestation`, `verify_output_attestation`, `_artifact_path`, `write_artifact_manifest`, `write_artifact_in_progress`, `verify_artifact_manifest`, `verify_component_output_manifest` (the four drivers lose their unused `_workflow_provenance` import) | |
| `twod/solver.py`: `_residual_norm_many`, `_make_elem_mask`, `_solve_te_robin_mfie`; `twod/basis.py`: `stiffness_block` (`_reference_blocks` becomes `_reference_mass`) | |
| `io/naming.py`: `format_base`, `group_solver_files`; `io/grim.py`: `compute_linear_from_dbke`; `run_local_bor.py`: the unused `copy_configuration` import | |

Kept deliberately: public entry points without an internal caller (`cylinder_generatrix`, the
`*_certified_single_polarization` solvers, `save_snapshot_geo`, `closest_primitive_points`,
`latest_run_dir`, `stage_geometry`, `predict_2d_resources`, `unit_peak_gb`, the factorization
resolvers), the test oracle `_assemble_system_fresh`, option-gated features (polynomial pairs,
projection/retained storage, `fast_far` verified CUR, recycling, refined LU), the NumPy fallbacks of
the native tiles and pair integrators, `io/viewer_bridge` (`to_grid` is used by `GRIM_Backend`),
`io/naming.pair_variants` (`data_tools`) and the assembly feature workflow.

### 2.6 Experiments (`experiments/solver_upgrades_20261006/`, repository root)

The items section 4 of the first pass left open were implemented as overlays over the package and
measured against the project in fresh processes (that folder's `README.md` has every number).  One
passed both tests, no field change and a measured saving, and was ported:

| ID | Change | Files | Measured effect |
|---|---|---|---|
| BoR sampled coarse-level checks | The graded near rules evaluate their fine level for every point of a layout chunk first and the coarse level for every `NEAR_CHECK_STRIDE`-th point (the first included); the chunk is accepted at the fine level when every probed point passes, otherwise the complete check of every point runs as before.  The disjoint meridian pairs do the same per batch (`NEAR_MERIDIAN_CHECK_STRIDE`, coarse order 6 on every 4th pair).  The fine level is the published value either way, so a chunk whose probes pass yields the complete check's values bitwise; on every reference body no probe failed (0 of 46 chunks and 2-3 batches per case) and the coarse level was 25% of the points instead of 100%.  `GHOST_BOR_NEAR_CHECK_STRIDE=0` restores the complete check. | `bor/kernels.py` (`NEAR_CHECK_STRIDE`, `_near_check_stride`, `_refine_near_chunk`), `bor/solver.py` (`NEAR_MERIDIAN_CHECK_STRIDE`, `_converged_disjoint_batch`) | BoR walls 0.79-0.92 of the reference in the experiment, section 3b after the port; fields bitwise identical |

Rejected after measurement (kept as overlays with their results): the 2-D check-only 20-point rule on
every 4th task (7.5e-15 but slower, 1.00-1.09: a failed probe re-runs the complete check and 7-70
batches per case fail), four-column condition probes (no gain), grouped dense-table contractions and
the far-tile 2 pi glue (both bitwise, both within noise on the minimum of three runs), finer far-rule
grading rows at |k| L = 1.0, 2.0 and 2.5 (bitwise identical because every production tile has
|k| L <= 0.5 or is capped by its own order; the calibration and a denser verification script remain in
`calibration/`), a far-ratio split of the far tiles (7.4e-13 but 1.06-1.13 slower), the TM partner kept
in RAM and the admission samples on threads (compressed backend, no gain).  The HODLR threshold was
confirmed (LU forced on the 10 GHz P3 system: 27.7 s against 21.1 s), and lazy near bands were not built
(the initial horizon already equals the cap for bandwidths >= 12).

### 2.7 Drivers and overhead (second round)

| ID | Change | Files | Measured effect |
|---|---|---|---|
| R-D-1 | A unit is verified before its solve and again before its export; the third verification after the artifact was published is gone (it could only report a change it can no longer prevent) | the four drivers (`_solve_and_export`) | 25 ms per unit cached, 150-210 ms uncached |
| R-D-4 | `save_monostatic_grim` builds the radar-frame payload in memory (`export_radar_grim(..., _return_payload=True, _save=False)`), embeds the body model and the metadata and writes once through `_save_grim_npz` (itself atomic), instead of writing, reading back, completing and writing again; the local BoR driver no longer re-parses every geometry before the pool (the planner already did; the spawned workers parse their own) and the fork-era comments are corrected | `assembly/fields.py`, `run_local_bor.py`, `runs/inputs.py` | one write and no read-back per geometry at the merge |
| R-2D-7 | The forecast cache key no longer contains the CPU/RAM allocation; a reused forecast is repriced under the current allocation (cost from the saved resource records with the forecast's own formula, peaks as before), so the sweep planner's per-worker selection reuses the preview's records instead of rebuilding every candidate mesh | `execution/selection.py` (`select_backend`, `_refresh_memory_forecast`) | ~100 ms per GUI frequency; a reused forecast equals a fresh one (test) |

Not changed: the HPC BoR array task still prices every candidate unit on its node (R-D-5; a
submission-time cache would need the HPC path, which this host cannot run), and the local pools are
`ProcessPoolExecutor`s whose workers spawn on the first submission, so starting them before planning
would overlap nothing.

Tests: `tests/test_audit_experiments_2026_10.py` (stride semantics and fallback with fake levels, the
sampled near rules and meridian batch against the complete check bitwise, two verifications per unit
in every driver, the single write of the BoR deliverable, the repriced reuse of a forecast);
`test_solver_pipeline_optimization.py`'s allocation test (formerly "allocation changes invalidate
forecasts") now asserts the new contract: one forecast per run, reused and repriced under a changed
allocation, equal to a fresh forecast.

### 2.8 Submit-time planning and the runtime-environment check (second round)

| ID | Change | Files | Measured effect |
|---|---|---|---|
| R-2D-7 (planning) | The submit-time 2-D resource planner builds the panels, material coefficients and interface-aware linear mesh of a frequency once and enriches a copy per basis degree: the hp certification pair (degrees 2 and 3 on one coarsened snapshot) formerly rebuilt everything for its second degree although only `basis.enrich` reads the degree.  The geometric near-pair count of a mesh is memoized on the mesh (one count per mesh instead of one per polarization and per degree; `copy_linear_mesh` shares the memo).  Geometries are planned on worker processes (`PLANNING_WORKERS` in both 2-D drivers, `GHOST_PLANNING_WORKERS`, default min(8, CPUs); serial below four geometries; a broken pool falls back to in-process planning).  Records are identical: on seven geometries, certified and not, every structural field (panels, nodes, unknowns, dense peaks, costs) matches the previous planner exactly and only the compressed storage fields that follow free memory at the time of the call differ, by the same amount between two runs of the same code | `hpc/scheduler.py` (`_resource_records_for_degrees`, `predict_2d_resources_for_geometries`, `planning_worker_count`), `twod/formulations/regions.py` (`mesh_near_pair_count`), `twod/geometry.py` (`copy_linear_mesh`), `run_hpc_monostatic.py`, `run_local_monostatic.py` | airfoil, five frequencies: certified 1.18 -> 0.41 s, uncertified 2.24 -> 1.75 s; twelve geometries certified: 4.9 s serial, 1.5 s on eight workers |
| Runtime check | The runtime fingerprint no longer includes the sections of NumPy's and SciPy's build configuration that describe hardware rather than the build: the CPU features detected at import (`SIMD Extensions`, which differ between a login node and a compute node of another CPU generation and refused every unit of a run submitted from the login node) and the build host (`Machine Information`).  Interpreter, OS family, architecture, library versions, BLAS/LAPACK build dependencies and the execution options stay strict.  A real mismatch names its fields (`describe_runtime_mismatch`, from the submission environment the manifests record; the local drivers now record it too) and `GHOST_RUNTIME_ENVIRONMENT_CHECK=warn` turns the refusal into one printed warning; `tests/diagnose_provenance.py` reports the runtime difference after the source check | `execution/provenance.py` (`runtime_compatibility_payload`, `describe_runtime_mismatch`, `verify_runtime_environment`), the four drivers, `tests/diagnose_provenance.py` | no refusal for a CPU difference; a version difference is named |

Tests: `tests/test_submit_planning_2026_10.py` (hp pair records equal per-degree planning from one panel, one mesh and one near-pair count; the memo shared by mesh copies; worker-process planning identical to serial and the broken-pool fallback; worker-count rules; CPU features informational; a real mismatch named and downgraded by the switch; every driver verifies through the shared check).

### 2.9 Node utilization on the HPC and local 2-D drivers (third round)

| ID | Change | Files | Measured effect |
|---|---|---|---|
| CPU reservation | Each unit's CPU reservation is the larger of the former fill rule (the cores divided among as many copies of the unit as memory and the pool admit) and its cost-proportional share of the task's work, `ceil(1.5 x cost / total x cores)`, capped at the node's physical cores (`cpu_reservations`).  The reservation now bounds the unit's assembly threads, its CPU allocation (`cpu_allocation_scope`) and its BLAS team (`threadpool_limits` in the pool worker, whose BLAS pool is started at the node's physical core count instead of the former fixed two threads).  The fill rule alone put every unit of a 71-frequency airfoil sweep on two threads, including the 15 GHz unit that is twice the balanced per-core work, and lost the remainder of `cores // pool` to idle cores | `hpc/scheduler.py` (`cpu_reservations`, `blas_thread_cap`), `run_hpc_monostatic.py`, `run_local_monostatic.py` | airfoil 1-15 GHz, 71 units, 96-CPU / 700 GB node model, first admission wave: 1 node 48 units on 96 CPUs with the heaviest unit on 2 threads -> 20 units on 96 CPUs, heaviest on 6; 2 nodes 36 units on 72 CPUs, heaviest on 2 -> 11 units on 96 CPUs, heaviest on 12; 4 nodes 85 -> 95 CPUs, heaviest 5 -> 23 threads; 8 nodes 90 -> 91 CPUs, heaviest 10 -> 45 threads |
| Memory evidence | Every unit records the planner's forecast and the worker's measured peak resident size (`execution_memory` in the artifact metadata, one line in the task log; the Linux peak counter is reset per unit, elsewhere the process lifetime is reported).  The 1.35x safety and 0.85 headroom factors are unchanged until such evidence from a cluster shows the forecast conservative | both 2-D drivers, `hpc/scheduler.py` (`reset_peak_rss`, `peak_rss_gib`) | evidence only |
| Mesh path report | The submit summary and `schedule.json` say which mesh path each unit takes: the hp pair (P2/P3 on one coarsened mesh), the linear pair, or a single uncertified linear mesh | `run_hpc_monostatic.py` | none (information) |
| Worker recycling | HPC 2-D pool workers are recycled every 8 units, as the local driver already does | `run_hpc_monostatic.py` | one spawn and backend import fewer per 8 units |

Measured and not changed:

- **Native per-tile-group far build (F-BoR-4).** On a 480-element cylinder at 10 GHz the far build is 2.5 s of a 27.4 s solve (9%) and on an 800-element sphere at ka = 63 it is 16.3 s of 126.3 s (13%), single worker; the near contraction is 78-82% and already runs inside the native near rule.  A fused native far build would recover 20-30% of the far build, 2-4% of the solve, which does not pay for native code.
- **Vectorized node welding in the 2-D mesh builder.** 0.2-0.3 s per mesh at 5,666 panels; the welder's tolerance semantics fix the node numbering, and an exact vectorized replica was not worth that saving.
- The batch backend chooser's internal schedule model keeps the fill rule for its relative comparison of dense and compressed schedules.

Tests: `tests/test_hpc_allocation_2026_10.py` (uniform shares fill the cores, heavy units get their cost share and light ones one CPU, memory-bound units keep the fill rule, explicit settings win, the BLAS cap and memory probes, the drivers apply reservation and evidence, the schedule records carry the mesh path).

## 3. Verification

Reference set (`scripts/audit_2026-10-06/bench.py`, one fresh process per case, production profile, 181 angles
unless stated): airfoil certified 1, 3 and 10 GHz, survey 3 GHz (dense and forced compressed); TYPE 2 IBC
(1 GHz), TYPE 5 two dielectrics (3 GHz), TYPE 1 thin dielectric (1 GHz), TYPE 4 coating (10 GHz); BoR PEC
cylinder survey 2 and 6 GHz and certified 2 GHz (181 aspects, workers 4); direct PEC sphere ka = 10
(37 aspects), dielectric and coated spheres (19 aspects).

| Case | Before (s) | After (s) | Change | Peak-relative change |
|---|---:|---:|---:|---:|
| Airfoil certified 1 GHz (414 panels, P2 -> P3) | 2.70 | 2.34 | -13% | 5.6e-15 |
| Airfoil certified 3 GHz (864 panels) | 5.12 | 4.47 | -13% | 2.2e-13 |
| Airfoil certified 10 GHz (2,472 panels, HODLR on the P3 system) | 31.41 | 20.72 | -34% | 2.8e-14 |
| Airfoil survey 3 GHz (5,666 panels, 8,608 unknowns, dense) | 14.01 | 12.71 | -9% | 1.3e-12 |
| Airfoil survey 3 GHz, compressed backend forced | 11.40 | 11.70 | +3% | 1.7e-15 |
| TYPE 2 CSV IBC square, certified 1 GHz (P1 pair) | 0.35 | 0.33 | -4% | 0.0e+00 |
| TYPE 5 two dielectrics, certified 3 GHz (P1 pair) | 0.95 | 0.93 | -3% | 0.0e+00 |
| TYPE 1 thin dielectric strip, certified 1 GHz | 0.54 | 0.53 | -2% | 0.0e+00 |
| TYPE 4 PEC-backed coating, certified 10 GHz (hp, refined once) | 1.46 | 1.34 | -8% | 3.5e-11 |
| BoR PEC cylinder survey 2 GHz (126 unknowns) | 1.04 | 0.96 | -7% | 5.9e-16 |
| BoR PEC cylinder certified 2 GHz (base + fine) | 2.69 | 2.43 | -10% | 8.2e-16 |
| BoR PEC cylinder survey 6 GHz (370 unknowns) | 4.36 | 2.77 | -37% | 5.3e-16 |
| BoR PEC sphere ka = 10, direct API (202 unknowns) | 2.25 | 1.67 | -26% | 1.6e-15 |
| BoR dielectric sphere, direct API (244 unknowns) | 2.01 | 1.82 | -10% | 2.4e-15 |
| BoR coated PEC sphere, direct API (342 unknowns) | 2.87 | 2.63 | -8% | 1.2e-15 |

Worst peak-relative change over the set: 3.5e-11 (rule: 1e-8).

Peak-relative change = max |new - old| / max |old| over the complex co-polarized amplitudes of both
channels. Timings are wall seconds of the solve call (imports excluded), single runs, +-5-10%.

### 3b. After the dead-code sweep, the ported experiment and the second driver round

Reference = the project after section 2 (the fields of the table above); after = sections 2.5-2.7.
Minimum wall over the recorded runs of each state (`experiments/solver_upgrades_20261006/results/`);
the peak-relative change is against the reference fields over the three post-port runs.

| Case | Before (s, min of runs) | After (s, min of 3) | Change | Peak-relative change |
|---|---:|---:|---:|---:|
| Airfoil certified 1 GHz | 2.32 (1) | 2.25 | -3% | 0.0e+00 |
| Airfoil certified 3 GHz | 4.44 (1) | 4.58 | +3% | 0.0e+00 |
| Airfoil certified 10 GHz | 21.11 (1) | 23.24 | +10% | 0.0e+00 |
| Airfoil survey 3 GHz (dense) | 12.64 (1) | 12.40 | -2% | 2.0e-12 |
| Airfoil survey 3 GHz (compressed) | 11.56 (1) | 11.92 | +3% | 0.0e+00 |
| TYPE 2 IBC certified 1 GHz | 0.33 (1) | 0.32 | -2% | 0.0e+00 |
| TYPE 5 two dielectrics certified 3 GHz | 0.92 (1) | 0.89 | -3% | 0.0e+00 |
| TYPE 1 thin dielectric certified 1 GHz | 0.53 (1) | 0.50 | -5% | 0.0e+00 |
| TYPE 4 coating certified 10 GHz | 1.35 (1) | 1.29 | -4% | 0.0e+00 |
| BoR cylinder survey 2 GHz | 0.96 (3) | 0.79 | -18% | 0.0e+00 |
| BoR cylinder certified 2 GHz | 2.30 (3) | 1.94 | -16% | 0.0e+00 |
| BoR cylinder survey 6 GHz | 2.68 (3) | 2.20 | -18% | 0.0e+00 |
| BoR PEC sphere, direct API | 1.57 (3) | 1.32 | -16% | 0.0e+00 |
| BoR dielectric sphere, direct API | 1.80 (3) | 1.40 | -22% | 0.0e+00 |
| BoR coated sphere, direct API | 2.63 (3) | 2.08 | -21% | 0.0e+00 |

No 2-D numerical path changed: two of the three post-port runs are bitwise identical to the reference
in every 2-D case and the third differs by 2.0e-12 in the dense 3 GHz survey, the run-to-run variation
of the threaded LU (the 2-D timing differences are run noise; the 10 GHz case ran 21.1 s in the
reference and 23.2-23.8 s in every later run).  The BoR cases are bitwise identical in all three runs
(the fine level is the published value in both checks, and no probe failed).  The full suite was run
again after these changes: 1,610 passed, 11 pre-existing failures on this host unchanged (the seven of
section 2.4 plus the three `BorMemoryGateTests` of `test_memory_safety.py`, whose expected message
predates the gate's GiB wording, and `test_review_followups.py::NearPairCleanupTests::
test_same_surface_consumer_closes_iterator_when_contraction_fails`; all four fail identically on the
unmodified pre-fix tree).

## 4. Not implemented, and why

- **F-2D-9 (check-only 20-point rule)**: measured as an experiment (section 2.6): 7.5e-15 but slower,
  because a failed probe re-runs the complete check and 7-70 batches per case fail.  The **BoR
  coarse-level sampling** passed the same measurement with no field change and was ported.
- **F-2D-5/6 (far-order grading beyond |k| L = 3, W floor per ratio bin)**: finer rows up to |k| L = 3
  were calibrated and measured (section 2.6) and proved unreachable on production meshes (every tile at
  |k| L <= 0.5 or capped by its own order); rows beyond 3 would be reachable only by elements longer
  than half a wavelength, which the mesh rules do not produce.
- **F-BoR-1 (near bands prepared lazily to the sweep horizon)**: the BoR core audit measured the saving
  at 5-7% of wall after the band defect fix, with the far tables (now streamed) the larger linear-in-modes
  cost; the concurrency between band extension and running mode workers was not worth that saving.
- **Process-pool threshold** (`AUTO_PROCESS_WORK_THRESHOLD`): raising it changed the memory pricing of
  calibrated planning tests; the measured difference between threads and processes at 4 workers is
  within 8% below 40,000 pair-modes.
- **Backend tie policy** (dense chosen at a 0.3% forecast margin where compressed measured 16% faster):
  one contended measurement; the prior was not changed.
- **F-BoR-4 (native per-tile-group sampling and transform)**: native code work; the tile planner and
  the table cache above recover part of the GIL loss.
- **Dead code (X-1)**: swept in the second pass (section 2.5).

## 5. Switches

- `GHOST_FAR_ATTENUATION_CUT=0` restores full evaluation of attenuated far pairs (any value > 1 sets the cut).
- `GHOST_ASSEMBLY_THREADS=<n>` pins the bare API's assembly threads as before.
- `assembly='tables'` keeps the dense-tables BoR path; `bor_options=dict(near_backend='threads')` keeps near
  preparation on threads.
- `GHOST_CPU_RHS_COMPRESSION=on` forces the QR sweep on LU factors.
- `GHOST_BOR_NEAR_CHECK_STRIDE=0` restores the complete coarse-level check of every BoR near point and
  meridian pair (default 4: every 4th is probed).
- `GHOST_PLANNING_WORKERS=<n>` sets the worker processes of the submit-time 2-D planning (`PLANNING_WORKERS`
  in the driver wins; 1 = serial).
- `GHOST_RUNTIME_ENVIRONMENT_CHECK=warn` lets a worker solve under a numerical runtime that differs from the
  run's recorded one, with one printed warning naming the difference (default: refuse).
