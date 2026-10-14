"""Online setup orchestration without network downloads or runtime installation."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error

import pytest

from GRIM_Backend import setup_online as online


def bootstrap_response(monkeypatch, content=b"verified bootstrap wheel"):
    monkeypatch.setattr(online, "BOOTSTRAP_FILENAME", "pip-test-py3-none-any.whl")
    monkeypatch.setattr(online, "BOOTSTRAP_SIZE", len(content))
    monkeypatch.setattr(online, "BOOTSTRAP_SHA256", hashlib.sha256(content).hexdigest())
    calls = []

    def urlopen(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(content)

    monkeypatch.setattr(online, "urlopen", urlopen)
    return content, calls


def test_bootstrap_download_verifies_bytes_before_publication(tmp_path, monkeypatch):
    content, calls = bootstrap_response(monkeypatch)
    result = online.download_bootstrap(tmp_path)
    assert result == tmp_path / online.BOOTSTRAP_FILENAME
    assert result.read_bytes() == content
    assert list(tmp_path.iterdir()) == [result]
    assert len(calls) == 1 and calls[0][1] == 60
    requested = calls[0][0]
    assert (requested if isinstance(requested, str) else requested.full_url) == online.BOOTSTRAP_URL


@pytest.mark.parametrize("bad_response", [b"", b"short", b"same sized wrong data!!!", b"x" * 100])
def test_truncated_corrupt_or_oversized_bootstrap_is_not_published(tmp_path, monkeypatch, bad_response):
    bootstrap_response(monkeypatch)
    monkeypatch.setattr(online, "urlopen", lambda *_args, **_kwargs: io.BytesIO(bad_response))
    with pytest.raises((online.offline.SetupError, OSError, ValueError)):
        online.download_bootstrap(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_same_size_wrong_hash_is_rejected(tmp_path, monkeypatch):
    content, _calls = bootstrap_response(monkeypatch)
    wrong = b"x" * len(content)
    monkeypatch.setattr(online, "urlopen", lambda *_args, **_kwargs: io.BytesIO(wrong))
    with pytest.raises((online.offline.SetupError, OSError, ValueError)):
        online.download_bootstrap(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_bootstrap_removes_partial_file(tmp_path, monkeypatch):
    bootstrap_response(monkeypatch)

    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("Connection interrupted")
            return super().read(3)

    monkeypatch.setattr(online, "urlopen", lambda *_args, **_kwargs: Interrupted(b"partial body"))
    with pytest.raises((online.offline.SetupError, OSError)):
        online.download_bootstrap(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_online_environment_keeps_python_isolation_and_enables_index(monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "outside Python")
    monkeypatch.setenv("PYTHONPATH", "outside packages")
    monkeypatch.setenv("PIP_TARGET", "outside target")
    monkeypatch.setenv("PIP_INDEX_URL", "https://unrelated.invalid/simple")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "outside environment")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    result = online.online_environment()
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_INDEX_URL", "PIP_NO_INDEX", "VIRTUAL_ENV"):
        assert name not in result
    assert result["PIP_CONFIG_FILE"] == os.devnull
    assert result["HTTPS_PROXY"] == "http://proxy.invalid:8080"


def test_wheel_cache_preserves_existing_other_wheels_and_ignores_nonwheels(tmp_path):
    source, destination = tmp_path / "staged", tmp_path / "offline packages"
    source.mkdir()
    destination.mkdir()
    (source / "new-1-py3-none-any.whl").write_bytes(b"new wheel")
    (source / "replace-1-py3-none-any.whl").write_bytes(b"complete replacement")
    (source / "noise.txt").write_text("do not cache")
    (destination / "replace-1-py3-none-any.whl").write_bytes(b"previous")
    (destination / "previous-1-py3-none-any.whl").write_bytes(b"keep this")
    (destination / "requirements.txt").write_text("numpy>=2\n")
    online.cache_wheels(source, destination)
    assert (destination / "new-1-py3-none-any.whl").read_bytes() == b"new wheel"
    assert (destination / "replace-1-py3-none-any.whl").read_bytes() == b"complete replacement"
    assert (destination / "previous-1-py3-none-any.whl").read_bytes() == b"keep this"
    assert (destination / "requirements.txt").read_text() == "numpy>=2\n"
    assert not (destination / "noise.txt").exists()
    assert not list(destination.glob("*.part"))


def test_wheel_cache_creates_missing_destination(tmp_path):
    source = tmp_path / "staged"
    source.mkdir()
    (source / "wanted-1-py3-none-any.whl").write_bytes(b"wheel")
    destination = tmp_path / "offline packages"
    online.cache_wheels(source, destination)
    assert (destination / "wanted-1-py3-none-any.whl").read_bytes() == b"wheel"


def test_failed_cache_publication_preserves_existing_wheel(tmp_path, monkeypatch):
    source, destination = tmp_path / "staged", tmp_path / "cached"
    source.mkdir()
    destination.mkdir()
    name = "wanted-1-py3-none-any.whl"
    (source / name).write_bytes(b"replacement")
    (destination / name).write_bytes(b"previous complete wheel")

    def fail(*_args):
        raise PermissionError("injected publication failure")

    monkeypatch.setattr(online.os, "replace", fail)
    with pytest.raises(PermissionError, match="injected"):
        online.cache_wheels(source, destination)
    assert (destination / name).read_bytes() == b"previous complete wheel"
    assert list(destination.iterdir()) == [destination / name]


def test_offline_setup_uses_requested_fresh_wheelhouse_through_install(tmp_path, monkeypatch):
    offline = online.offline
    backend = tmp_path / "GRIM_Backend"
    canonical = backend / "offline packages"
    canonical.mkdir(parents=True)
    requirements = canonical / "requirements.txt"
    requirements.write_text("pip>=25.3\n")
    (canonical / "pip-99-py3-none-any.whl").write_bytes(b"stale cached wheel")
    staged = tmp_path / "fresh wheels"
    staged.mkdir()
    pip_wheel = staged / "pip-26.2.1-py3-none-any.whl"
    pip_wheel.write_bytes(b"fresh resolved pip")
    base = backend / "python_env" / "python.exe"
    seen = []
    monkeypatch.setattr(offline, "validate_base", lambda _backend: base)

    def select(wheelhouse, manifest):
        assert wheelhouse == staged and manifest == requirements
        seen.append("select")
        return pip_wheel

    def read(manifest, pip):
        assert manifest == requirements and pip == pip_wheel
        return ["pip>=25.3"]

    def create(actual_base, target, wheelhouse, requested, pip, environment):
        assert actual_base == base and wheelhouse == staged and pip == pip_wheel
        assert requested == ["pip>=25.3"] and environment["PIP_NO_INDEX"] == "1"
        (target / "Scripts").mkdir(parents=True)
        (target / "Scripts" / "python.exe").write_bytes(b"test environment")
        seen.append("create")

    def install(target, wheelhouse, manifest, pip, environment):
        assert wheelhouse == staged and manifest == requirements and pip == pip_wheel
        assert environment["PIP_NO_INDEX"] == "1"
        seen.append("install")

    monkeypatch.setattr(offline, "pip_bootstrap_wheel", select)
    monkeypatch.setattr(offline, "read_requirements", read)
    monkeypatch.setattr(offline, "create_environment", create)
    monkeypatch.setattr(offline, "install_requirements", install)
    monkeypatch.setattr(offline, "verify_environment", lambda *args: seen.append("verify"))
    target = offline.setup(backend, wheelhouse=staged)
    assert offline.active_environment(backend) == target
    assert seen == ["select", "create", "install", "verify"]
    assert (canonical / "pip-99-py3-none-any.whl").read_bytes() == b"stale cached wheel"


@pytest.fixture
def setup_fixture(tmp_path, monkeypatch):
    backend = tmp_path / "GRIM_Backend"
    requirements = backend / "offline packages" / "requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("pip>=25.3\nnumpy>=2\n")
    base = backend / "python_env" / "python.exe"
    base.parent.mkdir()
    base.write_bytes(b"untouched runtime")
    pth = base.parent / "python314._pth"
    pth.write_text("python314.zip\n.\n# import site\n")
    root = backend / online.offline.ENVIRONMENTS
    root.mkdir()
    pointer = root / online.offline.ACTIVE_FILE
    pointer.write_text('{"environment": "previous working environment"}')
    seen = {"download": [], "install": [], "parse": []}
    monkeypatch.setattr(online.offline, "validate_base", lambda _backend: base)

    def bootstrap(directory):
        directory.mkdir(parents=True, exist_ok=True)
        wheel = directory / "pip-test-py3-none-any.whl"
        wheel.write_bytes(b"bootstrap")
        return wheel

    def read_requirements(path, pip_wheel):
        seen["parse"].append((path, pip_wheel))
        return ["pip>=25.3", "numpy>=2"]

    def run(command, *, env, cwd):
        seen["download"].append((command, env, cwd))
        destination = Path(command[command.index("--dest") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        manifest = Path(command[command.index("-r") + 1])
        seen["manifest"] = manifest.read_text()
        for name in ("pip-26.2.1-py3-none-any.whl", "numpy-2-cp314-cp314-win_amd64.whl"):
            (destination / name).write_bytes(name.encode("ascii"))

    def install(path, *, wheelhouse):
        seen["install"].append((path, wheelhouse))
        assert wheelhouse.is_dir()
        for item in wheelhouse.glob("*.whl"):
            assert (requirements.parent / item.name).read_bytes() == item.read_bytes()
        return root / "env-new"

    monkeypatch.setattr(online, "download_bootstrap", bootstrap)
    monkeypatch.setattr(online.offline, "read_requirements", read_requirements)
    monkeypatch.setattr(online.offline, "run", run)
    monkeypatch.setattr(online.offline, "setup", install)
    return backend, base, requirements, pointer, pth, seen


def test_setup_downloads_for_exact_embedded_runtime_then_installs_verified_stage(setup_fixture):
    backend, base, requirements, pointer, pth, seen = setup_fixture
    original_runtime, original_pth = base.read_bytes(), pth.read_bytes()
    result = online.setup(backend)
    assert result == backend / online.offline.ENVIRONMENTS / "env-new"
    assert len(seen["download"]) == len(seen["install"]) == 1
    command, environment, cwd = seen["download"][0]
    assert command[0] == str(base)
    assert {"-I", "-B", "--isolated", "download", "--only-binary=:all:"} <= set(command)
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert "--no-index" not in command and "PIP_NO_INDEX" not in environment
    assert cwd == backend
    staged = Path(command[command.index("--dest") + 1])
    assert seen["install"] == [(backend, staged)]
    assert staged != requirements.parent
    assert seen["parse"][0][0] == requirements
    assert seen["manifest"].splitlines() == ["pip>=25.3", "numpy>=2"]
    assert not staged.exists(), "Temporary downloads must be cleaned after installation"
    assert base.read_bytes() == original_runtime and pth.read_bytes() == original_pth
    assert "previous working environment" in pointer.read_text()


@pytest.mark.parametrize("stage", ["bootstrap", "requirements", "download", "cache"])
def test_failed_download_preparation_never_activates_an_environment(setup_fixture, monkeypatch, stage):
    backend, _base, requirements, pointer, _pth, seen = setup_fixture
    original = pointer.read_bytes()

    def fail(*_args, **_kwargs):
        raise online.offline.SetupError(f"injected {stage} failure")

    owner, name = {
        "bootstrap": (online, "download_bootstrap"),
        "requirements": (online.offline, "read_requirements"),
        "download": (online.offline, "run"),
        "cache": (online, "cache_wheels"),
    }[stage]
    monkeypatch.setattr(owner, name, fail)
    with pytest.raises(online.offline.SetupError, match="injected"):
        online.setup(backend)
    assert not seen["install"]
    assert pointer.read_bytes() == original
    assert requirements.read_text() == "pip>=25.3\nnumpy>=2\n"


def test_missing_runtime_is_reported_before_downloading(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(online, "download_bootstrap", lambda *args: calls.append(args))
    with pytest.raises(online.offline.SetupError):
        online.setup(tmp_path)
    assert not calls


def test_missing_requirements_is_reported_before_downloading(setup_fixture, monkeypatch):
    backend, _base, requirements, _pointer, _pth, _seen = setup_fixture
    requirements.unlink()
    calls = []
    monkeypatch.setattr(online, "download_bootstrap", lambda *args: calls.append(args))
    with pytest.raises(online.offline.SetupError):
        online.setup(backend)
    assert not calls


@pytest.mark.parametrize("failure", [online.offline.SetupError("wheel not available"),
                                     urllib.error.URLError("unreachable server"),
                                     PermissionError("destination is read only"),
                                     KeyboardInterrupt()])
def test_main_reports_setup_failure_without_traceback(monkeypatch, capsys, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(online, "setup", fail)
    assert online.main([]) != 0
    output = capsys.readouterr()
    assert "Traceback" not in output.out + output.err
    assert output.out + output.err


def isolated_python_fixture(embedded):
    """Use local CPython files with ._pth isolation, without installing packages."""
    embedded.mkdir(parents=True, exist_ok=True)
    host = Path(sys.base_prefix)
    shutil.copy2(host / "python.exe", embedded / "python.exe")
    for dll in host.glob("python*.dll"):
        shutil.copy2(dll, embedded / dll.name)
    tag = f"{sys.version_info.major}{sys.version_info.minor}"
    (embedded / f"python{tag}._pth").write_text(str(host / "Lib") + "\n" + str(host / "DLLs") + "\n")


def run_batch(path, cwd):
    # Pass the Windows command line directly: list2cmdline would backslash-escape
    # its inner quotes, which cmd.exe does not treat as C-runtime quoting.
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    command = f'"{comspec}" /d /s /c ""{path}""'
    return subprocess.run(command, cwd=cwd, input="\n", capture_output=True, text=True, timeout=20)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows embedded runtime")
def test_actual_online_entrypoint_loads_sibling_module_under_embedded_isolation(tmp_path):
    backend = tmp_path / "GRIM (isolated)!" / "GRIM_Backend"
    embedded = backend / "python_env"
    isolated_python_fixture(embedded)
    for module in ("setup_online.py", "setup_offline.py"):
        shutil.copy2(Path(online.__file__).parent / module, backend / module)
    # No requirements file: the real entry point must import successfully and
    # report the local setup issue before reaching any download call.
    result = subprocess.run([str(embedded / "python.exe"), "-I", "-B", str(backend / "setup_online.py")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert "Missing download requirements" in result.stderr, result.stdout + result.stderr
    assert "Traceback" not in result.stderr and "Downloading" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows root launchers")
@pytest.mark.parametrize("filename", ["Setup_GRIM.bat", "Launch_GRIM.bat"])
def test_root_batch_missing_runtime_from_other_directory_and_special_path(tmp_path, filename):
    root = tmp_path / "GRIM & test (spaces)!"
    root.mkdir()
    source = Path(online.__file__).resolve().parent.parent / filename
    shutil.copy2(source, root / filename)
    other = tmp_path / "unrelated working folder"
    other.mkdir()
    result = run_batch(root / filename, other)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "python_env" in result.stdout and "python.exe" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows root launchers")
@pytest.mark.parametrize("filename,script,expected_argument", [
    ("Setup_GRIM.bat", "setup_online.py", None),
    ("Launch_GRIM.bat", "setup_offline.py", "--launch"),
])
def test_root_batch_runs_local_python_isolated_and_propagates_failure(tmp_path, filename, script, expected_argument):
    root = tmp_path / "GRIM & test (spaces)!"
    backend = root / "GRIM_Backend"
    embedded = backend / "python_env"
    isolated_python_fixture(embedded)
    source = Path(online.__file__).resolve().parent.parent / filename
    shutil.copy2(source, root / filename)
    (backend / script).write_text(
        "import sys\n"
        "print('ISOLATED=' + str(sys.flags.isolated))\n"
        "print('BYTECODE=' + str(sys.dont_write_bytecode))\n"
        "print('SCRIPT=' + sys.argv[0])\n"
        "print('ARGUMENTS=' + repr(sys.argv[1:]))\n"
        "raise SystemExit(23)\n")
    other = tmp_path / "unrelated working folder"
    other.mkdir()
    result = run_batch(root / filename, other)
    assert result.returncode == 23, result.stdout + result.stderr
    assert "ISOLATED=1" in result.stdout and "BYTECODE=True" in result.stdout
    assert f"SCRIPT={backend / script}" in result.stdout
    assert "ARGUMENTS=" + repr([expected_argument] if expected_argument else []) in result.stdout
