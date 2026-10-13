GRIM offline distribution skeleton (Windows x64)
==============================================

This repository contains the setup scripts and requirements. Supply the
embedded Python and wheels on the computer where you prepare the release.
You do not need to download or install them on this development computer.

Prepare the distribution
------------------------
Keep GRIM_Backend and the sibling tools folder together. FREDDY and GHOST
use their existing files in tools; Python wheels do not replace those files.

Extract your Windows x64 embeddable Python into GRIM_Backend\python_env.
python.exe must be directly inside python_env, not a second nested folder.
The launchers use that exact interpreter. The computer's system Python does
not determine which Python runs GRIM. Python 3.14.6 is the intended release
runtime; setup also permits compatible CPython versions starting with 3.11
so the skeleton can be tested with the available development interpreter.

Download wheels on the other computer
-------------------------------------
If that computer has a normal Python 3.14 installation with pip, open a
terminal in GRIM_Backend and run:

py -3.14 -m pip download --only-binary=:all: --dest "offline packages" -r "offline packages\requirements.txt"

This collects the direct requirements and their dependencies for that
Python version and Windows architecture. Your normal Python and embedded
Python should match the release's version and architecture. The embedded
package itself does not ship pip, so use the normal installation to download.

If you already have the matching wheels, keep them in this folder. Include
pip, virtualenv, and their dependencies as well as the application packages.
requirements.txt is a download manifest with compatible version ranges; it
does not force the versions installed on the development computer. You may
replace ranges with exact == pins for a release you have validated.

Set up and launch
-----------------
1. Place the complete distribution in a writable folder on the destination.
2. Double-click GRIM_Backend\Setup_GRIM_Offline.bat.
3. When setup succeeds, double-click GRIM_Backend\Launch_GRIM_Offline.bat.

Setup uses only python_env\python.exe and the local wheels. It detects the
supplied interpreter and lets pip select packages compatible with that
interpreter, including Python 3.14 wheels when using Python 3.14.6. Setup
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
