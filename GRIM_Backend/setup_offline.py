"""Build GRIM's Windows environment using only the supplied interpreter and wheels.

The embeddable Python distribution has neither pip nor ensurepip. Bootstrap
virtualenv from local wheels without changing python_env or its ._pth file.
Each environment is built in its final directory; only the active pointer is
replaced, so failed updates do not damage the previously working environment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import uuid
import sysconfig
import zipfile


ENVIRONMENTS = ".offline_envs"
ACTIVE_FILE = "active_environment.json"
MINIMUM_PIP = (25, 3)
VIRTUALENV_REQUIREMENT = "virtualenv>=20.36.1"
PIP_SCRIPT = ("import sys, runpy; sys.path.insert(0, sys.argv.pop(1)); "
              "runpy.run_module('pip', run_name='__main__')")


class SetupError(RuntimeError):
    """A setup problem with an actionable message."""


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def clean_environment(source=None) -> dict[str, str]:
    """Ignore another Python/pip/virtualenv installation's process settings."""
    result = dict(os.environ if source is None else source)
    for key in list(result):
        upper = key.upper()
        if upper.startswith(("PYTHON", "PIP_", "VIRTUALENV_")) or upper == "VIRTUAL_ENV":
            result.pop(key)
    result.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1",
                  PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
                  PYTHONDONTWRITEBYTECODE="1", VIRTUALENV_NO_PERIODIC_UPDATE="1",
                  VIRTUALENV_CONFIG_FILE=os.devnull)
    return result


def read_requirements(path: Path, pip_wheel: Path) -> list[str]:
    """Accept package specifiers/markers, but never remote or nested inputs."""
    if not path.is_file():
        raise SetupError(f"Missing requirements file: {path}")
    sys.path.insert(0, str(pip_wheel))
    try:
        from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
        requirements = []
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                requirement = Requirement(line)
                if requirement.url:
                    raise InvalidRequirement("Direct URLs are not local wheel requirements")
            except InvalidRequirement as exc:
                raise SetupError(f"{path.name}:{number}: use a package name with an optional version "
                                 f"specifier, not a URL, path, or pip option: {line!r}.") from exc
            requirements.append(str(requirement))
        if not requirements:
            raise SetupError(f"No package requirements were found in {path}.")
        return requirements
    finally:
        sys.path.remove(str(pip_wheel))


def validate_base(backend: Path) -> Path:
    base = (backend / "python_env" / "python.exe").resolve()
    if not base.is_file():
        raise SetupError("Unzip the supplied CPython embedded runtime for Windows x64 into GRIM_Backend\\python_env first "
                         "(python_env\\python.exe must exist).")
    if Path(sys.executable).resolve() != base:
        raise SetupError("Run the root Setup_GRIM.bat, or Setup_GRIM_Offline.bat for local wheels, "
                         "so setup uses GRIM_Backend\\python_env\\python.exe.")
    if (sys.platform != "win32" or sys.implementation.name != "cpython" or sys.version_info[:2] < (3, 11)
            or struct.calcsize("P") != 8 or sysconfig.get_platform() != "win-amd64"):
        raise SetupError("Use CPython 3.11 or newer for Windows x64 in python_env, with wheels matching that runtime.")
    return base


def pip_bootstrap_wheel(wheelhouse: Path, requirements: Path | None = None) -> Path:
    """Use the newest compatible stable universal pip wheel supplied locally."""
    candidates = []
    for path in wheelhouse.glob("pip-*-py3-none-any.whl"):
        match = re.fullmatch(r"pip-(\d+(?:\.\d+)*)-py3-none-any\.whl", path.name)
        if match:
            version = tuple(int(part) for part in match.group(1).split("."))
            if version >= MINIMUM_PIP:
                candidates.append((version, path))
    for _version, path in sorted(candidates, reverse=True):
        sys.path.insert(0, str(path))
        try:
            with zipfile.ZipFile(path) as archive:
                metadata = Parser().parsestr(archive.read(
                    f"pip-{'.'.join(map(str, _version))}.dist-info/METADATA").decode("utf-8"))
            try:
                from pip._vendor.packaging.requirements import Requirement
                from pip._vendor.packaging.specifiers import SpecifierSet
            except (ImportError, SyntaxError):
                # A future pip may itself need a newer Python. Try another
                # supplied candidate instead of preventing local setup.
                continue
            python_requirement = metadata.get("Requires-Python", "")
            if not SpecifierSet(python_requirement).contains(".".join(map(str, sys.version_info[:3])), prereleases=True):
                continue
            if requirements is not None:
                pip_requirements = [Requirement(line) for line in read_requirements(requirements, path)]
                version = ".".join(map(str, _version))
                if any(normalized_name(item.name) == "pip"
                       and (not item.marker or item.marker.evaluate())
                       and not item.specifier.contains(version, prereleases=True) for item in pip_requirements):
                    continue
            return path
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise SetupError(f"Cannot read the supplied pip wheel {path.name}: {exc}") from exc
        finally:
            sys.path.remove(str(path))
    raise SetupError(f"Missing a compatible pip >=25.3 universal wheel (pip-VERSION-py3-none-any.whl) "
                     f"that satisfies requirements.txt. Place it with the application wheels in {wheelhouse}.")


def pip_command(python: Path, pip_wheel: Path) -> list[str]:
    return [str(python), "-I", "-B", "-c", PIP_SCRIPT, str(pip_wheel), "--isolated",
            "--disable-pip-version-check", "--no-cache-dir"]


def local_install_options(wheelhouse: Path) -> list[str]:
    return ["install", "--no-index", "--find-links", str(wheelhouse), "--only-binary=:all:",
            "--no-warn-script-location"]


def run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    result = subprocess.run(command, env=env, cwd=cwd, check=False)
    if result.returncode:
        raise SetupError(f"Command failed with exit code {result.returncode}. See the details above.")


def create_environment(base: Path, target: Path, wheelhouse: Path, requirements: list[str],
                       pip_wheel: Path, env: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="grim-offline-bootstrap-") as temporary:
        bootstrap = Path(temporary)
        # Let pip select the dependencies appropriate to this supplied Python.
        # Constraints honor optional user pins without installing application
        # packages into the temporary bootstrap directory.
        sys.path.insert(0, str(pip_wheel))
        try:
            from pip._vendor.packaging.requirements import Requirement
        finally:
            sys.path.remove(str(pip_wheel))
        constraints = bootstrap / "constraints.txt"
        constraint_lines = []
        for line in requirements:
            requirement = Requirement(line)
            constraint_lines.append(requirement.name + str(requirement.specifier)
                                    + (f"; {requirement.marker}" if requirement.marker else ""))
        constraints.write_text("\n".join(constraint_lines) + "\n", encoding="utf-8")
        run(pip_command(base, pip_wheel) + local_install_options(wheelhouse)
            + ["--ignore-installed", "--target", str(bootstrap), "--constraint", str(constraints),
               VIRTUALENV_REQUIREMENT], env=env, cwd=target.parent)
        script = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "from virtualenv import cli_run; cli_run(sys.argv[1:])"
        )
        run([str(base), "-I", "-B", "-c", script, str(bootstrap),
             "--python", str(base), "--creator", "builtin", "--copies", "--no-seed",
             "--no-download", "--no-periodic-update", "--app-data", str(bootstrap / "cache"),
             str(target)], env=env, cwd=target.parent)


def install_requirements(target: Path, wheelhouse: Path, requirements: Path,
                         pip_wheel: Path, env: dict[str, str]) -> None:
    python = target / "Scripts" / "python.exe"
    minimum_pip = "pip>=" + ".".join(map(str, MINIMUM_PIP))
    run(pip_command(python, pip_wheel) + local_install_options(wheelhouse)
        + [minimum_pip, "-r", str(requirements)], env=env, cwd=target)
    run([str(python), "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "check"],
        env=env, cwd=target)


def verify_environment(target: Path, base: Path, requirements: list[str], env: dict[str, str]) -> None:
    # Keep verification independent of Qt's display availability. Import its
    # native modules and the numerical stack without constructing a GUI.
    script = r"""
import importlib, importlib.metadata, json, pathlib, site, sys, sysconfig
from pip._vendor.packaging.requirements import Requirement
target, base, requirements = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve(), json.loads(sys.argv[3])
assert pathlib.Path(sys.prefix).resolve() == target, ('Unexpected environment', sys.prefix)
# With the embeddable runtime, virtualenv copies the DLL and stdlib ZIP into
# Scripts, and CPython reports Scripts as base_prefix. Verify its source through
# pyvenv.cfg instead of assuming the full-installer base_prefix convention.
config = dict(line.split('=', 1) for line in (target / 'pyvenv.cfg').read_text(encoding='utf-8').splitlines() if '=' in line)
config = {key.strip(): value.strip() for key, value in config.items()}
assert pathlib.Path(config['home']).resolve() == base.parent, ('Unexpected source Python', config['home'])
assert pathlib.Path(sys.executable).resolve() == target / 'Scripts' / 'python.exe', ('Unexpected executable', sys.executable)
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE, 'Python environment is not isolated'
for line in requirements:
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    version = importlib.metadata.version(requirement.name)
    assert requirement.specifier.contains(version, prereleases=True), ('Version mismatch', line, version)
for name in ('numpy', 'scipy.linalg', 'matplotlib._path', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'psutil', 'win32com.client', 'pythoncom'):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(target), ('External package path', name, module.__file__)
resolved = sorted({f"{distribution.metadata['Name']}=={distribution.version}"
                   for distribution in importlib.metadata.distributions()}, key=str.lower)
(target / 'requirements-resolved.txt').write_text('\n'.join(resolved) + '\n', encoding='utf-8')
(target / 'runtime.json').write_text(json.dumps({'python': sys.version, 'platform': sysconfig.get_platform(),
    'base_python': str(base), 'executable': sys.executable}, indent=2) + '\n', encoding='utf-8')
print('Verified isolated Python, resolved package versions, and application imports.')
"""
    run([str(target / "Scripts" / "python.exe"), "-I", "-B", "-c", script,
         str(target), str(base), json.dumps(requirements)], env=env, cwd=target)


@contextmanager
def setup_lock(root: Path):
    """Prevent concurrent setup runs from publishing conflicting environments."""
    lock = root / "setup.lock"
    try:
        handle = lock.open("x", encoding="ascii")
    except FileExistsError as exc:
        raise SetupError(f"Another setup is running. If a previous setup was interrupted, close it and delete {lock}.") from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish_environment(backend: Path, target: Path, requirements: Path, base: Path) -> None:
    root = backend / ENVIRONMENTS
    payload = {"environment": target.name, "base_python": str(base),
               "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest()}
    temporary = root / f"active-{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root / ACTIVE_FILE)
    finally:
        temporary.unlink(missing_ok=True)


def active_environment(backend: Path) -> Path:
    root = (backend / ENVIRONMENTS).resolve()
    try:
        data = json.loads((root / ACTIVE_FILE).read_text(encoding="utf-8"))
        name = data["environment"]
        if not isinstance(name, str) or not re.fullmatch(r"env-[0-9a-f]{32}", name):
            raise ValueError("Invalid environment name")
        target = (root / name).resolve()
        if target.parent != root or not (target / "Scripts" / "python.exe").is_file():
            raise ValueError("The environment is missing or was moved")
        return target
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SetupError("No usable offline environment was found. Run Setup_GRIM.bat in the GRIM root folder, "
                         "or Setup_GRIM_Offline.bat if wheels are already available.") from exc


def setup(backend: Path, *, wheelhouse: Path | None = None) -> Path:
    """Install from local wheels; online setup can supply its fresh download set."""
    backend = backend.resolve()
    base = validate_base(backend)
    requirements = backend / "offline packages" / "requirements.txt"
    wheelhouse = (wheelhouse or backend / "offline packages").resolve()
    pip_wheel = pip_bootstrap_wheel(wheelhouse, requirements)
    requested = read_requirements(requirements, pip_wheel)
    env = clean_environment()
    root = backend / ENVIRONMENTS
    root.mkdir(exist_ok=True)
    with setup_lock(root):
        target = root / f"env-{uuid.uuid4().hex}"
        try:
            print(f"Creating an isolated environment from the supplied Python {sys.version.split()[0]}...", flush=True)
            create_environment(base, target, wheelhouse, requested, pip_wheel, env)
            print("Installing compatible packages from local wheels...", flush=True)
            install_requirements(target, wheelhouse, requirements, pip_wheel, env)
            verify_environment(target, base, requested, env)
            publish_environment(backend, target, requirements, base)
        except BaseException:
            # Only this run's unique unpublished directory can be removed.
            if target.resolve().parent == root.resolve() and target.name.startswith("env-") and target.exists():
                try:
                    shutil.rmtree(target)
                except OSError:
                    print(f"The unfinished environment could not be removed: {target}", file=sys.stderr)
            raise
    print(f"GRIM is ready. Start the root Launch_GRIM.bat or GRIM_Backend\\Launch_GRIM_Offline.bat.\nEnvironment: {target}")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Run GRIM with the last successfully installed environment")
    arguments, app_arguments = parser.parse_known_args(argv)
    backend = Path(__file__).resolve().parent
    try:
        if arguments.launch:
            target = active_environment(backend)
            result = subprocess.run([str(target / "Scripts" / "python.exe"), "-I", "-B",
                                     str(backend / "run_gui.py"), *app_arguments],
                                    cwd=backend.parent, env=clean_environment(), check=False)
            return result.returncode
        if app_arguments:
            parser.error("unrecognized arguments: " + " ".join(app_arguments))
        setup(backend)
        return 0
    except (SetupError, OSError) as exc:
        print(f"\nGRIM setup: {exc}\nNo previously active environment was replaced.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
