"""Download matching wheels with the supplied embedded Python, then set up GRIM."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from urllib.request import urlopen
import uuid

# -I and the embedded ._pth omit this script's directory from sys.path.
if __package__:
    from . import setup_offline as offline
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import setup_offline as offline


# Bootstrap only: the installed pip version is resolved from requirements.txt.
# Universal wheel and SHA256 published at https://pypi.org/pypi/pip/26.2.1/json.
BOOTSTRAP_FILENAME = "pip-26.2.1-py3-none-any.whl"
BOOTSTRAP_URL = (
    "https://files.pythonhosted.org/packages/f3/6e/"
    "1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/"
    + BOOTSTRAP_FILENAME
)
BOOTSTRAP_SHA256 = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
BOOTSTRAP_SIZE = 1816632


def download_bootstrap(directory: Path) -> Path:
    """Fetch a verified pip wheel without installing anything into python_env."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / BOOTSTRAP_FILENAME
    partial = target.with_suffix(".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(BOOTSTRAP_URL, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > BOOTSTRAP_SIZE:
                    raise offline.SetupError("The pip bootstrap download has an unexpected size. Rerun setup.")
                digest.update(chunk)
                output.write(chunk)
        if size != BOOTSTRAP_SIZE or digest.hexdigest() != BOOTSTRAP_SHA256:
            raise offline.SetupError("The pip bootstrap download failed its integrity check. Rerun setup.")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def online_environment() -> dict[str, str]:
    env = offline.clean_environment()
    env.pop("PIP_NO_INDEX", None)
    return env


def cache_wheels(source: Path, destination: Path) -> None:
    """Keep complete downloads for later offline installations."""
    wheels = sorted(source.glob("*.whl"))
    if not wheels:
        raise offline.SetupError("No compatible wheels were downloaded. Review the download output and rerun setup.")
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        target = destination / wheel.name
        temporary = destination / f"{wheel.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(wheel, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def setup(backend: Path) -> Path:
    backend = backend.resolve()
    base = offline.validate_base(backend)
    wheelhouse = backend / "offline packages"
    requirements = wheelhouse / "requirements.txt"
    if not requirements.is_file():
        raise offline.SetupError(f"Missing download requirements: {requirements}")
    print(f"Using embedded Python {sys.version.split()[0]}: {base}", flush=True)
    with tempfile.TemporaryDirectory(prefix="grim-online-") as temporary:
        scratch = Path(temporary)
        print("Downloading the pip bootstrap from PyPI...", flush=True)
        pip_wheel = download_bootstrap(scratch / "bootstrap")
        requested = offline.read_requirements(requirements, pip_wheel)
        manifest = scratch / "requirements.txt"
        manifest.write_text("\n".join(requested) + "\n", encoding="utf-8")
        staged = scratch / "wheels"
        staged.mkdir()
        print("Downloading compatible wheels and their dependencies. This may take several minutes...", flush=True)
        offline.run(offline.pip_command(base, pip_wheel) + [
            "download", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--dest", str(staged), "-r", str(manifest),
        ], env=online_environment(), cwd=backend)
        cache_wheels(staged, wheelhouse)
        print(f"Wheels saved in {wheelhouse}. Creating and checking the environment...", flush=True)
        # Use this run's resolved wheel set; older cached wheels cannot change it.
        target = offline.setup(backend, wheelhouse=staged)
    print("\nSetup complete. Double-click Launch_GRIM.bat in the GRIM root folder.", flush=True)
    return target


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        setup(Path(__file__).resolve().parent)
        return 0
    except KeyboardInterrupt:
        print("\nGRIM setup canceled. Any previous active environment is still available.", file=sys.stderr)
        return 130
    except (offline.SetupError, OSError) as exc:
        print(f"\nGRIM setup: {exc}\nCheck the message above and rerun Setup_GRIM.bat. "
              "Downloads require an internet connection.\n"
              "Any previous active environment is still available.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
