GRIM offline distribution skeleton (Windows x64)
==============================================

This repository contains the setup scripts and requirements. Supply the
embedded Python and wheels on the computer where you prepare the release.
You do not need to download or install them on this development computer.

Prepare the distribution
------------------------
Keep Setup_GRIM.bat, Launch_GRIM.bat, GRIM_Backend, and the sibling tools
folder together. FREDDY and GHOST use their existing files in tools; Python
wheels do not replace those files.

Extract your Windows x64 embeddable Python into GRIM_Backend\python_env.
python.exe must be directly inside python_env, not a second nested folder.
The launchers use that exact interpreter. The computer's system Python does
not determine which Python runs GRIM. Python 3.14.6 is the intended release
runtime; setup also permits compatible CPython versions starting with 3.11
so the skeleton can be tested with the available development interpreter.

One-click online setup on the other computer
-------------------------------------------
After extracting embedded Python, double-click Setup_GRIM.bat in the GRIM
root folder. An internet connection is required. You do not need a separate
Python installation, pip installation, or an administrator terminal.

The script downloads a verified pip bootstrap, uses python_env\python.exe
to resolve compatible wheels and all dependencies from PyPI, and saves the
wheels in GRIM_Backend\offline packages. It then creates, checks, and
activates the application environment. Older cached wheels do not affect
the online installation: it uses the freshly resolved wheel set.

When setup finishes, double-click Launch_GRIM.bat in the root folder.
The setup window stays open so any errors can be read. If a download fails,
correct the connection problem and run Setup_GRIM.bat again. The embedded
Python files are left unchanged, and failed downloads or installation do
not replace a previously working environment.

Manual wheel download (optional)
--------------------------------
If you prefer to prepare wheels manually using a normal Python 3.14
installation with pip, open a terminal in GRIM_Backend and run:

py -3.14 -m pip download --only-binary=:all: --dest "offline packages" -r "offline packages\requirements.txt"

This collects the direct requirements and their dependencies for that
Python version and Windows architecture. Your normal Python and embedded
Python should match the release's version and architecture. The root setup
button handles pip bootstrapping automatically when using embedded Python.

If you already have the matching wheels, keep them in this folder. Include
pip, virtualenv, and their dependencies as well as the application packages.
requirements.txt is a download manifest with compatible version ranges; it
does not force the versions installed on the development computer. You may
replace ranges with exact == pins for a release you have validated.

Offline setup and launch
------------------------
1. Place the complete distribution in a writable folder on the destination.
2. Double-click GRIM_Backend\Setup_GRIM_Offline.bat.
3. When setup succeeds, double-click Launch_GRIM.bat in the root folder
   (GRIM_Backend\Launch_GRIM_Offline.bat also works).

Offline setup uses only python_env\python.exe and the local wheels. It detects the
supplied interpreter and lets pip select packages compatible with that
interpreter, including Python 3.14 wheels when using Python 3.14.6. Offline setup
does not contact a package index or download anything. Missing dependencies
are reported so you can add the corresponding wheels and rerun setup.

Each environment is created in its final location under .offline_envs.
Setup checks package dependencies, Python isolation, and native imports
before activating it. Failed setup leaves the previous active environment
available, and the original embedded Python files remain unchanged.

Use the same embedded Python and wheel set for each distributed copy.
Resolved package versions are recorded inside each completed environment.
If you move or copy GRIM to a different location, rerun setup there because
virtual environments contain absolute paths. Distribute the source, embedded
Python, and wheels; you do not need to distribute .offline_envs.

Microsoft PowerPoint must also be installed for PowerPoint automation.
The required Python COM support (pywin32) is included in requirements.txt.
Do not run pywin32_postinstall inside this virtual environment.

Development tests check the installer logic and an embedded-style layout
using the interpreter available locally. Validate the complete Python 3.14.6
installation on the destination with its actual embedded runtime and wheels.
