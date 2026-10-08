# GHOST solver and feature workflows

For wheel installation, clean source archives and release checks, see
[distribution and testing](DISTRIBUTION.md).

Open `Launch_GHOST_GUI.bat` to start GHOST. The top level contains the launcher,
Markdown guides, and one `ghost_backend` folder.

BOR controls include a bounded coefficient cache for compressed solves. See
[BOR controls and angle conventions](BOR_PERFORMANCE.md) and the shared
[driver workflow](#local-and-hpc-drivers) for local/HPC transfer.

| Folder | Contents |
| --- | --- |
| `ghost_backend/twod/`, `bor/` | 2-D and body-of-revolution solvers. |
| `ghost_backend/linalg/`, `compressed/` | Factorization, sweep reuse, matrix compression, and RAM forecasts. |
| `ghost_backend/assembly/` | Feature placement, field combination, and assembly validation. |
| `ghost_backend/io/` | Dataset loading, export, and filename operations. |
| `ghost_backend/runs/`, `execution/`, `hpc/` | Run configuration, CPU resources, provenance, and batch scheduling. |
| `ghost_backend/ui/` | Desktop geometry and solver controls. |
| `ghost_backend/data_tools/` | Dataset subtraction, joining, renaming, conversion, and their GUI/CLI. |
| `ghost_backend/geometry/` | Geometry loading, materials, mesh checks, rotation, sample geometries, and placement CSV templates. |
| `ghost_backend/validation/` | Material and geometry validation studies with their fixtures. |
| `ghost_backend/tests/` | Solver tests and performance benchmarks. |

Only `run_gui.py` and the four local/HPC run scripts are directly inside
`ghost_backend`. Native source, libraries, and build tools are under
`ghost_backend/bor/native/` and `ghost_backend/twod/assembly/native/`; runtime support is under `ghost_backend/execution/`.

Batch drivers save new runs under `ghost_backend/results/rcs_runs` or
`ghost_backend/results/rcs_runs_bor` by default. Explicit output paths and saved
configurations keep their selected destinations.

Use the [backend folder map](#ghost-solver-and-feature-workflows) to find solvers, data I/O,
geometry operations, feature assembly, and run management. 

2D runs are automatic. The solver tab and the 2D batch drivers ask only for
geometry, units, frequencies, azimuths, scattering mode, mesh certification and
accuracy target. The backend (dense or compressed), adaptive mesh, threads and
memory admission are chosen for each solve, and completed monostatic
frequencies are checkpointed for resume. See [running 2D solves](RUN_PROFILES.md).
The collapsed **Tools** section holds the accuracy/performance report and the
boundary-density plot.

For 2D batch sweeps, edit the CONFIG block of `run_local_monostatic.py` or
`run_hpc_monostatic.py`. BoR drivers also accept
`--config path/to/settings.config.json` (with `"driver": "bor"`) and
automatically load an adjacent file with the same stem and the `.config.json`
suffix. Each BoR driver declares its accepted setting names in `_CONFIG_KEYS`;
omitted settings keep its defaults. Portable HPC bundles package BoR requests.

The recommended desktop workflow is the top-level GRIM application. Its
**GHOST** tab embeds the same `ghost_backend/run_gui.py` workspace and the same
2-D/BoR numerical implementation found here; no solver is duplicated.

The 2-D diagnostic API defaults to `auto`; `direct` and `experimental_cpu`
remain available to Python callers for tests and diagnostics. NumPy and SciPy are required for
numerical methods and condition-number checks.
BoR supports its separate optional native streaming kernel.
BoR also has [bounded aspect batches, incident-basis reuse, and experimental
compressed modal assembly](BOR_PERFORMANCE.md), with separate controls in
the BoR solver options and local/HPC driver configurations.

In the desktop **Solve Options**, select **BoR** and enter radar **Azimuths**
and **Elevations**, each as a **Discrete List** or **Start / Stop / Step** sweep.
Elevations range from -90 to +90 degrees; azimuths range from 0 to 360 degrees
without including both 0 and 360. The body axis is horizontal with its nose
toward azimuth 0. Elevation defaults to 0 for a single horizontal cut.
The setup check reports the output grid size. The solve computes all unique
body aspects needed by that grid, then exports one monostatic `.grim` with
the requested azimuth/elevation axes and radar-frame VV, HH, and VH channels.
The embedded body model and geometry remain available for feature placement.
Assembly consumes this completed grid; it does not solve additional elevations.

The [phase and quadrature guidance](NUMERICAL_METHODS.md) describes corrected
2-D complex phase, bounded BoR near storage, and quadrature convergence
checks. Read the phase-compatibility notes before combining legacy complex
exports with newly generated results.

2-D assembly supports two optional native libraries in
`ghost_backend/twod/assembly/native/`: the kernel-table evaluator and the
far-field block quadrature and scatter. Both match the NumPy path bit for bit
wherever the kernel table covers the distance, and are skipped when they cannot
load. Portable distributions include their C sources and build script;
development checkouts may also contain binaries built for their host. A release
maintainer can use `scripts/build_distribution.py --native --output <folder>`
to bundle checked kernels in a Windows or Linux wheel. Its recipients need no
compiler; see [native releases](DISTRIBUTION.md#optional-native-releases).
After editing `table.c` or `far.c`, run
`py ghost_backend/twod/assembly/native/build.py`. Both native builders discover
MSYS2 UCRT64 in its standard Windows location and accept `--compiler` and
`--output-dir`. They compile to a temporary file, check it in a fresh process,
and replace the previous library only after validation succeeds.

Build the native BoR sampler on the worker machine with:

```powershell
py ghost_backend/bor/native/build_kernel.py
```

On Windows, install MSYS2 in its default `C:\msys64` location, open the
**MSYS2 UCRT64** terminal, and install the compiler with:

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

If the first update asks you to close the terminal, reopen **MSYS2 UCRT64**
and run both commands again. The build script discovers the default UCRT64
compiler automatically; no global PATH change is required. Verify from this
folder with `py ghost_backend/bor/native/build_kernel.py` and restart Python workers.

The build enables OpenMP outer-loop parallelism when the compiler supports it
and automatically retries a portable serial build otherwise. Use
`--no-openmp` to request the serial build explicitly. Result metadata reports
`stream_sampling_backend=native_c` or `numpy`, so production runs do not hide
which path was active.

Bounded far-block streaming is available for PEC/IBC, homogeneous dielectric
PMCHWT, simple coated-PEC bodies, and partial, layered, or banded junction
systems. The budget is enforced across every simultaneously retained self and
rectangular cross-surface block. Peak planning separately includes cached
junction projections and direct near/junction operators, which remain resident
when the far field is streamed. Result metadata records the sampling backend
for each medium side/mapping (a lossy material side uses the complex-wavenumber
NumPy sampler).

## CPU performance and memory

The Windows and Linux CPU paths use the existing NumPy/SciPy stack; these
optimizations require no additional Python package or higher minimum version.
The mesh, precision selection, integration tolerances, retained modes, and
physical acceptance checks keep their existing meanings.

* 2-D bistatic batches share bounded observation weights. Compressed dielectric
  coefficient queries fuse common kernel work and reuse the existing angular
  compression hint across compatible frequencies. Near coefficient storage is
  capped at 16 MiB in RAM, with larger stores using automatically removed
  temporary files; polynomial integration works in bounded batches. Integer
  near-pair plans still grow with the number of interactions.
* BoR near coefficients combine duplicate nodal destinations while preserving
  source impedance weights and independent reciprocal-pair diagnostics.
  Compressed far blocks retain a shared basis when that reduces storage.
  Angular projection selects complex products, packed real products, or
  DCT/DST evaluation of the same folded quadrature, including their scratch
  memory in the chunk allowance.
* Large material BoR systems can use the existing checked hierarchical
  factorization with dense fallback. Rectangular cross-surface storage can
  compress well-separated tiles only after checking every original tile's
  coefficients; nearby tiles remain dense. Mode ranges and temporary disk
  spilling remain available. Closely spaced layers may offer little compression.
* Hierarchical factorization uses bounded matrix panels and shared solve
  destinations. GMRES retains both orthogonalization passes while avoiding
  repeated copies of the growing conjugated basis.
* Batch planning reads CPU allocation once per plan and reuses each candidate's
  thread reservation across its proposed schedules. A new plan reads current
  allocation again; exhaustive searches avoid repeated OS and optional-import
  work without changing their scheduling choices.
* GUI and local/HPC BoR results use shared float64 sample arrays through
  certification and export. Public Python calls still return ordinary lists
  unless wrapped in `ghost_backend.twod.samples.compact_samples()`. GRIM
  formats, channel ordering, amplitudes, phase, and diagnostics are preserved.

Runtime reports retain the original main-process RAM measurement and add
parent-plus-worker samples. Shared pages can be counted more than once in
those sums. Windows private committed memory, when available, is reported
separately from resident RAM. These samples include other concurrent work and
are not an exclusive allocation peak.

Small Windows stage benchmarks found about 22% less dielectric assembly time
and 40% less bistatic projection time. A 100,000-sample BoR storage probe used
about 8.8 MB retained with compact rows versus 92 MB with dictionary rows.
These are component measurements, not whole-solve speedup or peak-RAM
guarantees. Linux timing and native-build qualification require a Linux host.

## GPU scope

Configured 2-D solves, including GUI, driver, survey, and certified entry points,
currently select the CPU backend. Setting `GHOST_DENSE_BACKEND=gpu` does not
change that execution profile. GPU support exists only in the low-level dense
linear-algebra diagnostic functions, using optional CuPy and a cuSOLVER health
check; it is not a production solver acceleration option. Release condition
estimates and the configured CPU assembly/factorization path use SciPy.

## Standalone GHOST

Run commands from this folder:

```powershell
py ghost_backend/run_gui.py
```

On Windows, `Launch_GHOST_GUI.bat` first changes to this folder and then opens
the same workspace.

## Local and HPC drivers

Edit the configuration block in the relevant driver, then run:

```powershell
py ghost_backend/run_local_monostatic.py
py ghost_backend/run_local_bor.py
py ghost_backend/run_hpc_monostatic.py
py ghost_backend/run_hpc_bor_monostatic.py
```

The 2-D production path co-solves VV/TE and HH/TM and writes them into one
GRIM artifact per geometry/frequency. The BoR path produces a combined
body artifact after its per-frequency restart units complete.

Each newly prepared HPC run contains its own `runtime/ghost_backend` source
and native-kernel snapshot. The saved worker and generated Slurm scripts use
that snapshot, so editing the original CONFIG block or updating the checkout
cannot alter an already prepared run. Keep the `runtime` folder with the run.
The snapshot is still integrity checked; changing it invalidates that run.
Python, NumPy, SciPy and their numerical build configuration must remain
compatible between submission and workers. Kernel-release and processor
labels are recorded as host information rather than compatibility requirements.

BoR workers refresh resource estimates inside their compute allocation and
bind each solve to its selected backend, CPU count and memory reservation.
Submit-host estimates are planning hints, not worker memory reservations.

BoR TYPE 1 sheets and connected sheet/PEC profiles export their complete
solved profile and radar response. Sheet-containing artifacts identify their
profile as a sheet and cannot be used automatically as an opaque body surface
for feature placement or shadowing. Geometry/export compatibility is checked
before solving; ordinary conductor and dielectric body exports retain their
feature-surface workflow.

See:

- [AUDIT_FIXES_2026-09-22.md](AUDIT_FIXES_2026-09-22.md) for the 22 September
  speed, memory and accuracy fixes, what changes in results, and their verification.
- [driver configuration](#local-and-hpc-drivers) for local/cluster operation and resource controls.
- [GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md) for `.geo`
  boundaries, regions, materials, winding, and units.
- [BoR conventions](BOR_PERFORMANCE.md#geometry-and-angle-conventions) for BoR geometry, polarization,
  phasor, loss, and RCS conventions.
- [ghost_backend/validation/non_bor_feature_validation/README.md](ghost_backend/validation/non_bor_feature_validation/README.md) for point and
  line-feature dataset and placement requirements.
- [ghost_backend/validation/non_bor_feature_validation/README.md](ghost_backend/validation/non_bor_feature_validation/README.md)
  for the independent four-artifact clean/featured validation ladder and
  manifest-driven complex-field gates.
- [ghost_backend/validation/non_bor_line_reconstruction/README.md](ghost_backend/validation/non_bor_line_reconstruction/README.md)
  for the checked-in finite-plate, door-outline, and folded-panel line tests.
- [ghost_backend/validation/non_bor_curved_feature_placement/README.md](ghost_backend/validation/non_bor_curved_feature_placement/README.md)
  for the triaxial-ellipsoid point/line regression and shared-facet normal-tie
  controls.

## Feature assembly service

The GRIM Assembly form and automation wrapper both call
`ghost_backend/assembly/workflow.py`. `ghost_backend/assembly/place_features.py` remains a thin
settings-based wrapper for unattended work. Source certificates, solver-version
tags, and feature/surface manifests are normally advisory. Explicit phase,
amplitude, time-sign, and polarization conventions must be compatible; missing
supported declarations are recorded as assumptions. Numerical units, fields,
axes, and placement geometry are checked. Strict library metadata and certified-body profiles
are optional. Only an explicitly selected strict profile or a large workload
review requires a warning acknowledgement. The GUI uses the same placement,
phase, expansion, and shadowing implementation.

The integrated tab supports editable placements, point rows/circles/polyline
patterns, line paths, explicit surface projection and normal derivation,
body-only baselines, exact stored-grid subsets, and body/feature/total response
comparison. Newly selected BoR bodies can generate their own bounded-error
shadow mesh. See the [Assembly workflow guide](ghost_backend/validation/non_bor_feature_validation/README.md).
The 2-D `amplitude_version` is advisory during subtraction and line loading.
Operations record convention assumptions without inferring a field conversion.
An explicitly declared coherent line subtraction may retain the physical 2D
input convention; the loader validates it before assigning the delta role.
That declaration does not permit a different phase origin or normalization.

FREDDY nominal IBC and dielectric CSVs store frequencies in Hz and are readable
with or without headers. GHOST converts Hz to its internal GHz scale and
preserves signed complex material values. Analysis/uncertainty CSVs are separate
from nominal material tables.

Create and check a reviewed feature-response sidecar with:

```powershell
py ghost_backend/assembly/create_feature_manifest.py create --help
py ghost_backend/assembly/create_feature_manifest.py check --help
py ghost_backend/assembly/create_feature_manifest.py create-surface-binding --help
py ghost_backend/assembly/create_feature_manifest.py check-surface-binding --help
```

For `validated` libraries this is now an evidence-binding and integrity tool:
it consumes the full-wave validator report, re-hashes all four case artifacts,
and proves that the assembled prediction used the exact response. Team review
is still required because software cannot establish external-solver
independence or mesh convergence. See
[ghost_backend/validation/non_bor_feature_validation/README.md](ghost_backend/validation/non_bor_feature_validation/README.md) for the exact
manifest fields, headless settings, reduced-order limitations, and required
independent full-wave evidence.

Use `python ghost_backend/data_tools/run_cli.py subtract OPN FRD Deltas` for
canonical OPN-FRD 2-D deltas. General joins and dataset conversion use the same
CLI. See [the CLI command definitions](ghost_backend/data_tools/run_cli.py).

## Tests

From this folder:

```powershell
py -m pip install -e ".[test,gui]"
py ghost_backend/tests/run_suite.py
```

## Material and IBC files

Use headered, comma-separated `.csv` files with frequency in Hz, following the
[shared GHOST/FREDDY file format](MATERIAL_CSV_FORMAT.md). FREDDY material
and nominal IBC exports can be used directly. The geometry editor validates
CSV selections before adding them. Space/tab-separated tables, headerless
CSVs, and implicit `mat.<flag>` references are not accepted.
