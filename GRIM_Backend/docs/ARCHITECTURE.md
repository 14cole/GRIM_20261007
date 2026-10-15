# Architecture and extension points

GRIM's numerical data model is independent of Qt. The desktop application,
Python examples, and embedded tools share the same physical quantity and file
contracts. Keep new numerical behavior in these shared layers so a button and
a Python script produce the same result.

## Dataset samples and operations

`datasets/grid.py` exposes `RcsGrid`. Its storage layer keeps one authoritative
sample representation, with slice access to physical complex fields, power,
phase, or raw solver fields. See [complex storage](complex_storage.md) for units,
missing phase, writable legacy arrays, and native file compatibility.

Combination, common arithmetic, and CSV export process bounded sample blocks.
The final output still occupies memory; blocking reduces temporary full-grid
arrays and does not make the application disk backed. Use `power[selection]`,
`phase[selection]`, or `rcs_slice(selection)` inside new algorithms. Do not call
`np.asarray(grid.power)` before selecting the needed cells.

`datasets/metadata.py` owns the shared metadata-retention rules. Exact selection
and rearrangement retain sample metadata; derived responses remove metadata
whose physical meaning no longer applies. The format-specific readers remain
behind the existing `RcsGrid` API and `io/loaders.py` dispatch.

`datasets/operations.py` adds common history at the public operation boundary.
Its decorator records effective parameters, input lineage, and output shape in
`operation_provenance_json`; GUI actions use the same record instead of appending
a duplicate entry. Nested implementation calls produce one public event.
Execution UUIDs distinguish independent branches and survive native saves.
They identify lineage, not numerical content or a cache key. Large coordinate
arrays are summarized with shape, dtype, and a hash. The Python recorder remains
the source for a complete replay script.

## Plot preparation and display

`plotting/registry.py` declares plot names and headless support. Add a supported
mode here instead of maintaining separate GUI-recorder and Python mode lists.

`plotting/requests.py` defines captured requests, plans, and prepared results for
frequency, elevation, CDF, and waterfall views. Selection arrays and options are
copied; the dataset itself is referenced. These functions do not read widgets or
create Matplotlib artists. GUI modes and `scripting/plotting.py` share this
preparation, including physical units and aggregation. A dataset must not be
mutated concurrently while a preparation is reading it.

`plotting/analysis_results.py` prepares phase and polarization traces, metrics,
notes, and export samples from explicit cuts and options. The inline analysis
panel and dialog share `ui/analysis_presenter.py` to display these results.
Interactive I/Q controllers still own their cursor and artist behavior.

Each Plotting/ISAR canvas permanently owns a `PlotContext` and `PlotSession`.
The compatibility facade forwards existing action properties to the active
context; switching tabs does not copy plot state between contexts. Sessions
own cancellation and temporary resources. Clear, replacement, and window close
release the resources of the affected view.

Large ordinary preparations run through `execution/plot_jobs.py`. Each session
has at most one running job and one pending replacement. New requests cancel
superseded work; worker results reach a Qt slot for drawing only if their tab,
canvas, and generation still match. Numerical code checks cancellation between
blocks. A single NumPy calculation may finish before cancellation takes effect.
Small requests run directly to avoid worker overhead. ISAR retains its existing
formation/job service.

## GHOST and distribution

GHOST remains usable without importing GRIM or Qt. Its standalone native codec
in `ghost_backend/data_tools/cem_tools/native_samples.py` reads legacy files and
the compact version 2 representation. Component views derive selected power or
phase only when requested. Solver output, HPC assembly, and CEM readers use raw
complex fields directly where possible, preserving their normalization.

The supported distribution is application source plus embedded Python and an
offline wheel folder. [Release validation](RELEASE_VALIDATION.md) documents the
acceptance check and export of a folder with exact dependency versions and
source/runtime/wheel hashes. A developer smoke test is distinct from acceptance
under the actual embedded release runtime.

## Validation and future work

Regression tests cover known numerical results, mixed precision and missing
phase, format round trips, shared GUI/Python results, cancellation, stale-result
discard, and tab ownership. Performance comparisons should measure both peak
memory and elapsed time; less allocation does not imply faster execution.

The current architecture remains incremental. A broad file-adapter extraction
and a chunked, disk-backed storage format can be considered separately if real
workloads justify them. Neither is needed to use the bounded access and shared
services described here.

## Large dataset loading

The arrow beside **Load** opens **Load options** and **Load a region from .grim**.
Automatic loading budgets half of currently available RAM. A custom GiB budget
applies to the current batch and its readers, without halving it again after
other files load. If the estimate exceeds the budget, the load options dialog
shows available RAM, the estimate, and the budget. **Continue once** bypasses
that policy cap for the current batch and loads serially; it does not change
saved preferences or bypass format and numerical validation. Actual allocation
failures remain possible. Ordinary loads below the budget do not prompt.

The native scheduler and reader share a header-based estimate that accounts for
sample dtype, layout, selected region, and storage mode. Recognized flat CSV,
SENTRi, and CST tables use their actual discovered axes and typed allocation
plan. Other readers retain conservative estimates. Estimates are planning
values, not guarantees about memory used by the operating system or other apps.

**Disk** storage applies to native `.grim`, PIO (`.pio`/`.cmplx_di`), and SS files. Numeric archive members are
extracted in bounded blocks to a private temporary cache and mapped there;
the input file is never modified. A selected region is copied in bounded blocks
to RAM or mapped output, as requested. Each fully extracted numeric member is
released and removed after its independent selected/converted output is complete.
ZIP members must still be decompressed sequentially; selection does not provide
random reads within compressed files. PIO and SS inputs are mapped read-only and
decoded in bounded blocks into independent destinations, preserving byte order,
axis units, exceptional samples, and precision. Cache files remain
in the system temporary folder unless a different working folder is selected
in Load options. Choose a drive with enough space for expanded data. They remain
valid while a dataset, held plot, undo entry, or array view still references them,
and are released when their last owner is gone. Operation results have a
separate storage policy, described below; loading a file on disk does not
automatically change the policy for subsequent operations.

Region ranges are inclusive and use the file's declared axis units. A selected
region appears as an unsaved dataset so it is not mistaken for the complete
source. The Python recorder retains selection and storage options. Scripts can
use the same interface:

```python
from GRIM_Backend.scripting.api import load_dataset

data = load_dataset(
    "run.grim",
    storage="disk",
    selection={"frequency": (8.0, 12.0), "elevation": (0.0, 0.0),
               "polarization": ["HH", "VV"]},  # Frequency values assume this file uses GHz.
    max_output_bytes=4 * 1024**3,
)
```

SENTRi and CST imports discover axes and then fill typed arrays with repeatable
row iterators. They retain duplicate and unit diagnostics without keeping all
text records in memory. Large Waterfalls fill a temporary native image in tiles
and derive display tiles from it. Native probes, cuts, and numerical exports
still use original-resolution values; display reduction remains explicit.

## Operation result storage

**Dataset Operations → Result storage** chooses **Auto**, **RAM**, or **Disk**
for new operation results. The GUI defaults to Auto with a 256 MiB RAM
budget for live managed allocations per operation batch. Auto admits fitting
destination arrays to RAM and writes larger ones to private temporary files.
Released arrays return their allowance; plain NumPy views retain it until their
last reference disappears. An individual result can contain both RAM and mapped
arrays. The dataset name tooltip reports its actual sample storage. Changing
the setting affects future jobs, not existing datasets or a running worker.

Operations allocate their destinations through the same storage service and
fill them in bounded blocks. This includes exact copies and crops, axis
transforms, interpolation, joins/stitches, arithmetic, calibration, statistics,
gating, and phase-center changes. The result is fully computed before it appears
in the table; this is disk-backed storage, not deferred numerical evaluation.
Small coordinate arrays, metadata objects, and numerical scratch still use RAM.
Numerical scratch is bounded independently. Exact median/percentile populations
use disk scratch and in-place selection, while means and stable variance stream
through the population. Individual FFT traces still require RAM. The Auto budget controls managed
allocations, not the total memory of Python, NumPy, or the operating system.

The working folder can be selected in the same dialog. Each mapped array owns
its temporary files through its NumPy view chain. Held views, plots, and Undo
Delete keep referenced data valid; files are removed after their final owner
is released. Unpublished arrays from failures or cancellation follow the same
lifetime rules. The input file is never used as writable result storage.
Disk capacity is checked before allocation, and I/O failures are reported.

RAM-mode preflight offers a choice when an estimate exceeds the working
budget: switch this operation to Auto or Disk, cancel, or Continue once with
the chosen settings. This one-time choice does not overwrite saved preferences.
Format and numerical validation remain in effect.

Save writes an ordinary compatible `.grim` file and does not unload or relocate
the live result. Temporary disk files are not a durable saved dataset. Python
replay records the operation's storage policy, and scripts can use it directly:

```python
from GRIM_Backend.scripting.api import load_dataset, operation_result_storage
from GRIM_Backend.scripting.workspace import offset_db, crop_dataset

source = load_dataset("run.grim", storage="disk")
with operation_result_storage("auto", ram_budget_bytes=256 * 1024**2):
    shifted = offset_db(source, 3.0)
    selected = crop_dataset(shifted, frequency_range=(8.0, 12.0))
selected.save("selected.grim")
```

The context is local to its thread, nests safely, and defaults to RAM when not
explicitly used by headless callers. GUI workers capture the options at launch
and enter their own context, so concurrent jobs and later settings changes do
not change each other's allocation policy.


## Bounded analysis and storage management

**Dataset Operations → Storage usage** reports loaded sample buffers, additional
buffers retained by Undo, tracked temporary files, and abandoned caches. Clear
Undo releases only restore history; active plots and other views retain their
references. Cleanup requires a marked GRIM cache directory with a verifiably
ended owning process. Unmarked folders, live processes, inaccessible owners,
and redirected directories are preserved. The cache manifest includes process
creation time to distinguish reused process IDs. Disk results remain temporary;
Save is required for durable datasets.

ISAR Advanced controls now expose an automatic or custom memory budget and an
explicit option to continue above it. Automatic uses half the available RAM;
these captured options also travel through Python replay and saved recipes.
The Fourier path and Range–Frequency transform sweeps in bounded batches and
write managed intermediate arrays. Range–Frequency no longer rejects an image
based on total transform work. Batching preserves coherent FFT normalization,
windowing, axis conventions, and exact incoherent averaging. Large requests use
background preparation with cancellation; Matplotlib publication stays on Qt.

Statistics stream reduction populations; exact rank statistics use disk scratch
instead of approximating percentiles. CDF retains exact sorted levels and
cumulative weights, using bounded scratch for preparation. Threshold-only edits
reuse that distribution without rereading samples or sorting. Changing inputs,
clearing the canvas, or leaving CDF invalidates the retained distribution.

Large phase/IQ preparation captures control values and reads selected native
indices before deriving fields. A bounded native-cut cache avoids rereading the
same immutable cut during option changes. Legacy mutable sample exposure bypasses
this cache. Large preparation runs in a latest-request-wins worker; canceled or
stale requests cannot publish over a newer plot. Gate previews likewise cache the
unchanged Before profile and process FFT sweeps in byte-bounded batches; the
azimuth median remains exact.

GUI native saves choose compression using bounded samples from C-contiguous,
Fortran-contiguous, and strided arrays. Storage layout alone no longer causes a
compressible dataset to be saved uncompressed. The native file schema and legacy
readers remain unchanged.
