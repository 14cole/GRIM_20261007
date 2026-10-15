# Validating an offline GRIM release

Normal setup stays flexible: `Setup_GRIM.bat` downloads compatible wheels, and
`GRIM_Backend/Setup_GRIM_Offline.bat` installs supplied wheels without internet.
For a release you intend to give to other engineers, validate and freeze one
working environment rather than combining downloads from different releases.

1. Unzip the intended Windows x64 embedded Python into `GRIM_Backend/python_env`.
2. Put its compatible wheels and all dependencies in `GRIM_Backend/offline packages`.
3. Double-click **Validate_GRIM.bat** in the root. This command uses only local
   files, creates an isolated environment, and checks it before activation.

The acceptance check imports the application from this source tree and dependencies
from that environment. It loads a small complex dataset, applies a known 6 dB
offset, saves and reloads native, CSV, PIO, and PTM files, renders a waterfall with cuts, and opens and
closes the actual GRIM window with FREDDY and GHOST in Qt's offscreen mode. It uses
temporary data and settings. It does not start a solver run or Microsoft Office,
and does not replace comprehensive numerical regression tests or a manual check
of display/Office integration on the destination computer.

Release acceptance also checks the installed dependency relationships and requires
the supported `pip` and `virtualenv` packages in the environment. Keep both in
`offline packages/requirements.txt` with their dependency wheels, as supplied in
the skeleton. This ensures that the exact exported wheel set can create its own
environment on an air-gapped machine. An existing environment customized to omit
these setup packages can still run GRIM normally; rerun `Validate_GRIM.bat` after
restoring them to the requirements when preparing a distributable release.

If installation or acceptance fails, the last active environment remains selected.
A successful check writes `GRIM_Backend/release-validation.json` with the source
commit when available, source-file hashes, exact Python version and architecture,
exact installed dependencies, selected compatible wheel SHA256 hashes, embedded
runtime hashes, and the checks performed. For a downloaded source ZIP without Git,
use `--source-commit HASH` to record its known commit; otherwise that field is
explicitly unknown and the actual source-file hashes still identify the content.

To validate an existing active environment without reinstalling, run from the root:

```powershell
.\GRIM_Backend\python_env\python.exe -I -B .\GRIM_Backend\release_validation.py
```

To export a previously validated release into a **new folder outside this checkout**:

```powershell
.\GRIM_Backend\python_env\python.exe -I -B .\GRIM_Backend\release_validation.py --export-manifest .\GRIM_Backend\release-validation.json --bundle C:\Releases\GRIM-reviewed
```

Export verifies all recorded inputs before publication, stages the folder, and
includes application code/assets, embedded Python, and only the exact selected
wheels. The exported `requirements.txt` pins every dependency; old cached wheel
versions and installed environments are excluded. User datasets are not bundled.
Documented GHOST validation templates and material input tables are included;
generated validation responses and reports are excluded.
The output includes `release-manifest.json`; its source hashes describe the
validated source, and its bundle section records the generated pinned requirements.
The destination must not already exist. Copy the complete folder to the air-gapped
computer, run `GRIM_Backend/Setup_GRIM_Offline.bat`, then `Launch_GRIM.bat`.
Use `Validate_GRIM.bat` there when you want the same acceptance check before activation.

When the release runtime is unavailable on a development machine, the following
performs the application smoke check with an existing Python installation:

```powershell
py -3.11 .\GRIM_Backend\release_validation.py --developer-python C:\Path\To\python.exe --manifest C:\Temp\grim-smoke.json
```

That report is explicitly **not** a validated embedded distribution and cannot be
used to export a release. A Python 3.11 smoke check cannot certify Python 3.14.6;
run final acceptance with the actual embedded runtime and wheels on the release machine.
