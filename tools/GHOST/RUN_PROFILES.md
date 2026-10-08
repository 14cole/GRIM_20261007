# Running 2D solves

Every 2D run, in the GHOST solver tab or a batch driver, uses one automatic
setup. You choose the physical problem; the solver chooses how to compute it.

## What you choose

| Setting | Values |
| --- | --- |
| Geometry and units | A `.geo` file or the Geometry tab; inches or meters. |
| Frequencies | A list or a start/stop/step sweep in GHz. |
| Azimuths | A list or a start/stop/step sweep in degrees. |
| Scattering mode | Monostatic, or bistatic with observation angles. |
| Mesh certification | On: solve base and fine meshes and publish the fine result only if both channels converge. Off: solve one mesh, marked `mesh_convergence_certified=false`. |
| Accuracy target | Standard, or tight (1% maximum complex change) for the base/fine comparison. The dB and phase changes are compared where the pattern is above -30 dB of its peak (deep nulls are judged by the complex change). |

## What is automatic

Monostatic runs:

- **Backend.** Dense or compressed Galerkin, ranked by predicted work for the
  geometry, materials and angle count, and admitted against available RAM
  (90% of currently available memory). Compressed solves may keep up to 60%
  of that limit (at least 2 GiB) for both polarizations' operators and the
  preconditioner. The work model separates assembly, compression,
  factorization and angle solves. Clean, repeated measurements can calibrate
  nearby frequencies with compatible meshes and execution settings; ordinary
  runs do not launch extra solves for calibration.
- **Mesh.** An adaptive polynomial mesh: a quadratic candidate with a cubic
  accuracy check, local refinement when the comparison fails, and a global
  linear mesh as the final fallback.
- **Numerics.** Double-precision LU, automatic incident-field basis reuse, and
  up to 256 angles per batch.
- **Threads.** Assembly threads default to the physical cores of the solving
  machine (at most 16), or the per-unit allocation a batch driver computed for
  a scheduled solve. Large compressed operators are
  assembled in worker processes instead: up to eight (half the host cores) for
  a desktop solve, the CPU reservation for a scheduled solve, fewer when
  memory is short, and none inside batch worker processes. Factorization and
  angle solves choose a BLAS pool by system size within the CPU reservation:
  one thread through 256 unknowns, up to four through 1024, up to eight through
  2048, then the available reservation. Explicit integer `blas_threads` values
  are caps and are never widened. `linear_algebra_execution` records the actual
  library pools used; these thresholds are a policy, not a speed guarantee.
- **Frequency scheduling.** Desktop sweeps can run two
  frequencies at once when the predicted work justifies startup and their
  combined CPU and RAM reservations fit. Larger units run first; results
  retain the requested frequency order. These workers share the total CPU
  budget and do not start nested assembly workers. Small runs and runs that
  cannot fit two frequencies remain sequential. A frequency that outgrows
  its worker reservation is retried in the parent after the workers drain.
- **Compressed assembly reuse.** Admission samples retain up to 32 MiB of
  verified tiles for assembly of the identical operator. An already assembled
  polarization partner supplies its actual storage size. Assembly workers
  persist for the run and refresh their geometry and coefficients before each
  operator; their retained memory is included in later phase forecasts.
- **Run recovery.** Every Start Run computes fresh results. Each completed
  frequency is saved in this run's unique recovery folder with both
  polarizations and its accuracy information. Parallel workers save directly
  to disk and release their completed fields instead of sending a growing
  collection to the GUI. Earlier runs are never reused by Start Run.

Bistatic runs use dense LU on the global linear mesh, with the same validated
CPU kernel tables and native far assembly as monostatic runs. Their bounded
table state spans polarization and certification passes and is released when
the request completes or is cancelled. `cpu_kernel_execution` records table
validation and use. Explicit mixed-precision API calls use the same kernel
tables and factor with mixed-precision LU; explicit GPU calls keep their
existing execution path.

## Batch drivers

Edit the CONFIG block of `ghost_backend/run_local_monostatic.py` or
`ghost_backend/run_hpc_monostatic.py`:

| Setting | Meaning |
| --- | --- |
| `FRD_DIR`, `OPN_DIR` | Input geometry folders, searched recursively. |
| `FREQUENCIES_GHZ`, `AZIMUTHS_DEG` | The sweep. |
| `OUTPUT_DIR` | Output root; each run gets a new timestamped folder. |
| `GEOMETRY_UNITS`, `MESH_CERTIFICATION`, `ACCURACY_TARGET` | As in the GUI. |
| `WORKERS` (local) | Optional ceiling on concurrent solves. |
| `MAX_SOLVE_GB` | Optional per-solve RAM ceiling in GiB. |
| SLURM settings (HPC) | `N_NODES`, `N_JOBS`, `ARRAY_THROTTLE`, partition, account, QoS, walltime, cores and memory per node, mail, extra `#SBATCH` lines, job prologue, `PYTHON_EXE`, `SUBMIT`. |

Both drivers cost every geometry/frequency unit from the mesh the solver will
build, run the most expensive units first, and admit concurrent solves against
the machine's or node's memory. 2D drivers do not read JSON configuration
files, and portable HPC bundles are for BoR requests.

## Recovering an interrupted desktop run

2D and BoR desktop runs create `Run_Recovery/run_<timestamp>_<unique ID>`
beside the input geometry. Unsaved Geometry-tab inputs use the application's
Documents/GRIM Outputs directory. The folder contains:

- `inputs/`: the captured geometry, material CSV contents and solve settings.
  Workers read these captured copies, so later edits to the original files do
  not change an active run.
- `frequencies/`: completed frequency outputs plus checksum records. Files
  are flushed and published atomically; incomplete or damaged outputs are
  excluded from recovery.
- `run.json`: the run's identity, requested frequencies, input checksums and
  status. Completion is checked against the actual frequency files, so an
  abrupt process exit does not require a final status update.

In the Solver tab's Tools section, choose **Recover Completed Run** and open
that run's `run.json`. Completed frequencies become available for plotting
and **Export Last Result**. Missing frequencies stay missing, and a partial
export is marked as partial in its metadata. Recovery does not start or
resume a solve; Start Run always starts a new run.

Final `.grim` export joins frequency planes incrementally and includes the
captured input files for provenance. It does not assemble the entire field
grid in memory. Successful complete exports are verified before publication
is recorded. Their recovery folders are removed when the result is no longer
being viewed, provided the final exported files still match their checksums.
Interrupted, unexported, partially exported and failed-export runs are kept.

Solve-time output memory is limited to active frequencies; result viewing
loads one frequency at a time. Solver matrices, run metadata and arrays used
for a particular plot still require memory. Recovery preserves completed
frequencies, not an unfinished frequency's matrix or factorization.

## Status reporting

The status text reports assembly, factorization, angle solving and mesh
certification work, with elapsed solve time and sampled process RAM. Base and
fine mesh phases are identified separately. The percentage can remain fixed
while a long stage runs. RAM is sampled every 50 ms and includes other work in
the same process; it is not an exact allocation peak. Stage timings in exported
metadata are inclusive and may overlap.
For parallel frequency runs, the largest individual worker peak is reported
separately from reserved RAM. It is not the simultaneous process-tree peak;
aggregate sampled RAM is left unavailable when it was not measured.

## Python API

Public 2D solve functions still accept `execution_options` for tests,
benchmarks and diagnostics. Production runs do not need it; they use
`ghost_backend.execution.options.automatic_run(scattering)`.

Every desktop Start Run computes fresh results for both 2D and BoR. Cross-run
solve caching and automatic resume are disabled. Existing application-cache
checkpoints are ignored. Run recovery is used only for explicit inspection
and export of completed outputs, never to skip a requested calculation.

`ghost_backend.execution.fresh_sweep.run_fresh(..., frequency_workers='auto')`
retains the desktop's bounded parallel frequency scheduler. The desktop passes
a newly created `RecoveryRun` as `recovery=...`; completed fields are saved
and the returned sample sequences load a frequency only when needed. Without
that argument, API results remain in memory and count against the run's RAM
allowance. Direct solve calls retain their normal sequential frequency loop.
Legacy explicit checkpoint utilities remain separate from desktop execution.

Two explicit experimental execution options remain off in automatic runs:
`compressed_far_method='verified_cur'` proposes low-rank far tiles but checks
every coefficient before acceptance, and `frequency_preconditioner='reuse'`
tries a bounded inverse cache for nearby frequencies with identical mesh and
DOF ordering. Reuse always tests the current operator's residual and rebuilds
the inverse if its short iteration allowance is exhausted. Changing the
default adaptive mesh generally prevents inverse reuse. Neither option is a
general speed guarantee.
