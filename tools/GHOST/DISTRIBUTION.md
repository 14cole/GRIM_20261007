# GHOST 0.1.1 distribution and testing

This release contains the 2-D and body-of-revolution solvers, the round-11/12
accuracy and storage changes, and mesh-allocation guards added during independent
review. Both fixed and adaptive 2-D meshing and the BoR resource preview reject
oversized meshes before allocating their panel coordinates. BoR counting and
meshing share an allocation-free grading plan.

## Install

Use Python 3.10 or later in a virtual environment. From the distribution folder:

```console
python -m pip install ghost_em2d-0.1.1-py3-none-any.whl
```

NumPy, SciPy and psutil are required. Their supported version ranges are declared
in the wheel. Qt and matplotlib are optional desktop dependencies; install the
`gui` extra when using the desktop interface. Configured solves use the CPU.

The default wheel is platform-independent and contains no prebuilt native
binaries. Both solvers have NumPy fallbacks. The optional BoR and 2-D C sources
and build scripts are included, together with example geometries and placement
templates. A separate native wheel can include these accelerators for Windows
or Linux; installing that wheel requires no compiler on the recipient machine
and adds no Python dependencies. Existing binaries in a development checkout
are never redistributed by the release builder.

## Source and tests

Extract `ghost-em2d-0.1.1-source.zip` for the complete clean `GHOST` folder,
including tests, fixtures, documentation and release tools. It excludes audit
archives, bytecode, caches, generated solver outputs and native binaries.
The standard `.tar.gz` source distribution is also provided for package builders.

From the extracted `GHOST` folder:

```console
python -m pip install ".[test,gui]"
python ghost_backend/tests/run_suite.py
```

The test runner isolates Qt module lifetimes from numerical workers. For the
qualified headless solver subset, install `.[test]` and run
`python scripts/check_headless.py`. Run `scripts/check_installation.py` with the
installed environment's Python from an unrelated working directory to check
wheel resources and 2-D/BoR analytic reference cases. It deliberately refuses
imports from a source checkout. `scripts/check_speed_paths.py` checks the
checkout it belongs to instead: whether the native BoR samplers, the 2-D native
libraries and an optimized BLAS work on this machine, and whether a setting
switches a fast path off; it exits with status 1 when anything would fall back.

Tests that exercise the separate GRIM or FREDDY projects explicitly skip when
those companion checkouts are absent. GHOST's own solver and file-format tests
remain available. Companion integration was also tested in the development
workspace where those projects are present.

Rebuild the artifacts with an environment containing `setuptools>=68` and `wheel`:

```console
python scripts/build_distribution.py --output ../distributions
```

The default build emits a portable wheel, standard source distribution,
source/test ZIP, and a SHA-256 manifest. It checks required wheel resources and
excludes platform binaries and generated data. The ZIP has stable member
metadata for reproducible source checksums.

## Optional native releases

On a release machine with a C99 compiler, use:

```console
python scripts/build_distribution.py --native --output ../native-distributions
```

This compiles the 2-D table and far-field kernels and the BoR streaming kernel
from the staged sources. Each library must load and export every required
entry point in a fresh Python process before it is packaged. The manifest
records library checksums and exports. Library rebuilds use temporary files
and only replace an existing library after this check succeeds.

Use `--compiler /path/to/gcc` to select a compiler, or let the builders use
`CC`, PATH, or the standard Windows MSYS2 UCRT64 installation. Compiler and
OpenMP runtimes are linked statically on Windows. `--no-openmp` builds the BoR
kernel without OpenMP; this reduces its internal parallelism but can simplify
Linux deployment where an OpenMP runtime is unavailable. The 2-D native kernels
keep their existing Python-managed threading.

Build separately on Windows and Linux, using the target Python architecture.
For example, a 64-bit Windows release emits
`ghost_em2d-0.1.1-py3-none-win_amd64.whl`, while an x86-64 Linux release emits
`ghost_em2d-0.1.1-py3-none-linux_x86_64.whl`. These ctypes libraries have no
CPython extension ABI dependency. The native wheel is marked as platform
specific and retains the package's Python version requirement. Its source ZIP
and standard source distribution remain free of binaries.

Linux wheels use a conservative platform tag, not a `manylinux` compatibility
claim. Build on the oldest supported target system and validate its system C
and, when enabled, OpenMP runtime requirements on the recipient baseline.
Use the installed wheel's environment to run `scripts/check_installation.py`
from an unrelated directory before distributing it. Separate platform builds
and recipient smoke checks are necessary; a successful Windows build does not
validate a Linux binary. Recipients can install the native wheel offline with
the same approved NumPy, SciPy and psutil wheels used by the portable release.

## Numerical scope

Mesh certification and resource admission remain necessary for production inputs.
The quadrature rescue retains its convergence check; unresolved cases still raise
an error. BoR impedance grading currently covers closed conductor snapshots,
with four fixed refinement levels. Partial/banded material junctions and sheets
retain their existing discretization. The 2-D contrast threshold and added levels
improve the tested junctions but do not establish accuracy for arbitrary corners,
open ends or triple junctions. See `NUMERICAL_METHODS.md` and
`GEOMETRY_INPUT_CHEATSHEET.md` for the supported inputs and limitations.

GPU support is limited to low-level diagnostics. No real CuPy hardware execution
is claimed by this release. Validation for this release was performed on Windows;
Linux native compilation and numerical smoke checks require a Linux release host.
