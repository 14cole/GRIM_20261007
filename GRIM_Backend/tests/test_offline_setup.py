"""Offline installer checks; no network or user runtime is modified."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile

import pytest

from GRIM_Backend import setup_offline as offline


def write_requirements(path, extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("pip>=25.3\nvirtualenv>=20.36.1\n" + extra)


@pytest.fixture
def bundled_pip():
    import ensurepip
    return next((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))


def wheel(path, entries):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize("invalid", ["thing @ https://example.invalid/thing.whl", "-r other.txt", "--index-url https://example.invalid", "C:/outside/package.whl"])
def test_requirements_reject_remote_paths_or_pip_options(tmp_path, bundled_pip, invalid):
    path = tmp_path / "requirements.txt"
    write_requirements(path, invalid + "\n")
    with pytest.raises(offline.SetupError, match="optional version specifier"):
        offline.read_requirements(path, bundled_pip)


def test_requirements_accept_ranges_pins_extras_and_markers(tmp_path, bundled_pip):
    path = tmp_path / "requirements.txt"
    write_requirements(path, 'numpy>=2,<3\nPySide6_Addons==6.10.2\nexample[feature]~=1.2; python_version >= "3.14"\n')
    result = offline.read_requirements(path, bundled_pip)
    assert "numpy<3,>=2" in result
    assert "PySide6_Addons==6.10.2" in result
    assert 'example[feature]~=1.2; python_version >= "3.14"' in result


def test_supplied_python_3146_is_accepted(tmp_path, monkeypatch):
    base = tmp_path / "python_env" / "python.exe"
    base.parent.mkdir()
    base.write_bytes(b"runtime placeholder")
    monkeypatch.setattr(offline.sys, "executable", str(base))
    monkeypatch.setattr(offline.sys, "platform", "win32")
    monkeypatch.setattr(offline.sys, "version_info", (3, 14, 6, "final", 0))
    monkeypatch.setattr(offline.struct, "calcsize", lambda _format: 8)
    monkeypatch.setattr(offline.sysconfig, "get_platform", lambda: "win-amd64")
    assert offline.validate_base(tmp_path) == base.resolve()


def test_clean_environment_removes_other_python_and_pip_configuration():
    result = offline.clean_environment({"Path": "kept", "PYTHONHOME": "other",
        "pythonpath": "other", "PIP_TARGET": "elsewhere", "PIP_INDEX_URL": "https://example.invalid",
        "VIRTUAL_ENV": "other", "VIRTUALENV_PYTHON": "other"})
    assert result["Path"] == "kept"
    assert "PYTHONHOME" not in result and "pythonpath" not in result
    assert "PIP_TARGET" not in result and "PIP_INDEX_URL" not in result
    assert "VIRTUAL_ENV" not in result and "VIRTUALENV_PYTHON" not in result
    assert result["PIP_NO_INDEX"] == "1" and result["PIP_CONFIG_FILE"] == os.devnull
    assert result["VIRTUALENV_CONFIG_FILE"] == os.devnull


def test_pip_selection_honors_runtime_and_requirement(tmp_path, monkeypatch, bundled_pip):
    # Use real pip's parser; these metadata-only candidates are never executed.
    sys.path.insert(0, str(bundled_pip))
    try:
        from pip._vendor.packaging.requirements import Requirement  # noqa: F401
    finally:
        sys.path.remove(str(bundled_pip))
    for version, python in [("25.3", ">=3.9"), ("26.0", ">=3.9"), ("27.0", ">=3.15"), ("9.0", ">=3.0")]:
        wheel(tmp_path / f"pip-{version}-py3-none-any.whl", {
            f"pip-{version}.dist-info/METADATA": f"Name: pip\nVersion: {version}\nRequires-Python: {python}\n"})
    monkeypatch.setattr(offline.sys, "version_info", (3, 14, 6, "final", 0))
    assert offline.pip_bootstrap_wheel(tmp_path).name == "pip-26.0-py3-none-any.whl"
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("pip==25.3\n")
    assert offline.pip_bootstrap_wheel(tmp_path, requirements).name == "pip-25.3-py3-none-any.whl"


@pytest.mark.parametrize("python_version,abi,expected", [("3.11", "cp311", "1.0"), ("3.14", "cp314", "2.0")])
def test_real_pip_resolves_matching_wheels_from_mixed_python_versions(tmp_path, bundled_pip, python_version, abi, expected):
    for version, tag in [("1.0", "cp311-cp311-win_amd64"), ("2.0", "cp314-cp314-win_amd64"),
                         ("2.5", "cp314-cp314-manylinux_2_17_x86_64")]:
        info = f"wanted-{version}.dist-info"
        wheel(tmp_path / f"wanted-{version}-{tag}.whl", {
            "wanted.py": "VALUE = 1\n",
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: wanted\nVersion: {version}\n",
            f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
            f"{info}/RECORD": ""})
    report = tmp_path / "report.json"
    command = offline.pip_command(Path(sys.executable), bundled_pip) + offline.local_install_options(tmp_path)
    command += ["--dry-run", "--ignore-installed", "--target", str(tmp_path / "target"),
                "--python-version", python_version, "--platform", "win_amd64", "--implementation", "cp",
                "--abi", abi, "--report", str(report), "wanted>=1,<3"]
    result = subprocess.run(command, env=offline.clean_environment(), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(report.read_text())["install"][0]["metadata"]["version"] == expected


def test_virtualenv_bootstrap_resolves_dependencies_with_exact_interpreter_and_no_downloads(tmp_path, monkeypatch, bundled_pip):
    calls = []
    constraints = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "--constraint" in command:
            constraints.append(Path(command[command.index("--constraint") + 1]).read_text())
    monkeypatch.setattr(offline, "run", fake_run)
    base, target = tmp_path / "python_env" / "python.exe", tmp_path / "env"
    offline.create_environment(base, target, tmp_path, ["virtualenv>=20.36.1", "filelock==3.20.1"], bundled_pip, {})
    pip = calls[0][0]
    assert pip[0] == str(base)
    assert {"--no-index", "--only-binary=:all:", "--ignore-installed", "--target", "virtualenv>=20.36.1"} <= set(pip)
    assert "filelock==3.20.1" in constraints[0]
    command = calls[1][0]
    assert command[0] == str(base)
    assert command[command.index("--python") + 1] == str(base)
    assert {"-I", "--no-download", "--no-periodic-update", "--no-seed", "--copies"} <= set(command)
    assert command[-1] == str(target)


def test_missing_wheels_preserve_existing_environment(tmp_path, monkeypatch):
    backend = tmp_path / "GRIM_Backend"
    write_requirements(backend / "offline packages" / "requirements.txt")
    root = backend / offline.ENVIRONMENTS
    root.mkdir()
    pointer = root / offline.ACTIVE_FILE
    pointer.write_text('{"environment": "existing"}')
    monkeypatch.setattr(offline, "validate_base", lambda _backend: backend / "python_env" / "python.exe")
    with pytest.raises(offline.SetupError, match="Missing a compatible pip"):
        offline.setup(backend)
    assert pointer.read_text() == '{"environment": "existing"}'
    assert list(root.iterdir()) == [pointer]


@pytest.mark.parametrize("failure_stage", ["create", "install", "verify"])
def test_failed_update_preserves_active_environment(tmp_path, monkeypatch, failure_stage):
    backend = tmp_path / "GRIM_Backend"
    requirements = backend / "offline packages" / "requirements.txt"
    write_requirements(requirements)
    root = backend / offline.ENVIRONMENTS
    root.mkdir()
    previous = root / ("env-" + uuid.uuid4().hex)
    previous.mkdir()
    (previous / "kept.txt").write_text("working")
    pointer = root / offline.ACTIVE_FILE
    pointer.write_text(json.dumps({"environment": previous.name}))
    original = pointer.read_bytes()
    monkeypatch.setattr(offline, "validate_base", lambda _backend: backend / "python_env" / "python.exe")
    monkeypatch.setattr(offline, "pip_bootstrap_wheel", lambda *args: tmp_path / "pip.whl")
    monkeypatch.setattr(offline, "read_requirements", lambda *args: ["pip>=25.3"])
    def create(_base, target, *_args):
        target.mkdir()
        (target / "partial.txt").write_text("incomplete")
        if failure_stage == "create":
            raise offline.SetupError("injected creation failure")
    def install(*_args):
        if failure_stage == "install":
            raise offline.SetupError("injected dependency failure")
    def verify(*_args):
        if failure_stage == "verify":
            raise offline.SetupError("injected import failure")
    monkeypatch.setattr(offline, "create_environment", create)
    monkeypatch.setattr(offline, "install_requirements", install)
    monkeypatch.setattr(offline, "verify_environment", verify)
    with pytest.raises(offline.SetupError, match="injected"):
        offline.setup(backend)
    assert pointer.read_bytes() == original
    assert (previous / "kept.txt").read_text() == "working"
    assert sorted(item.name for item in root.iterdir()) == sorted([previous.name, offline.ACTIVE_FILE])


def test_publication_failure_preserves_active_pointer(tmp_path, monkeypatch):
    backend = tmp_path / "GRIM_Backend"
    root = backend / offline.ENVIRONMENTS
    root.mkdir(parents=True)
    requirements = tmp_path / "requirements.txt"
    write_requirements(requirements)
    pointer = root / offline.ACTIVE_FILE
    pointer.write_text("previous pointer")
    def fail_replace(*_args):
        raise PermissionError("injected replace failure")
    monkeypatch.setattr(offline.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        offline.publish_environment(backend, root / "env-new", requirements, tmp_path / "python.exe")
    assert pointer.read_text() == "previous pointer"
    assert list(root.iterdir()) == [pointer]


def test_active_environment_cannot_select_outside_runtime_folder(tmp_path):
    root = tmp_path / offline.ENVIRONMENTS
    root.mkdir()
    for name in ["../../elsewhere", "C:\\external", "env-", None]:
        (root / offline.ACTIVE_FILE).write_text(json.dumps({"environment": name}))
        with pytest.raises(offline.SetupError, match="No usable offline environment"):
            offline.active_environment(tmp_path)


def test_successful_update_publishes_without_moving_environment(tmp_path, monkeypatch):
    backend = tmp_path / "GRIM_Backend"
    write_requirements(backend / "offline packages" / "requirements.txt")
    monkeypatch.setattr(offline, "validate_base", lambda _backend: backend / "python_env" / "python.exe")
    monkeypatch.setattr(offline, "pip_bootstrap_wheel", lambda *args: tmp_path / "pip.whl")
    monkeypatch.setattr(offline, "read_requirements", lambda *args: ["pip>=25.3"])
    seen = []
    def create(_base, target, *_args):
        (target / "Scripts").mkdir(parents=True)
        (target / "Scripts" / "python.exe").write_bytes(b"test double")
        seen.append(target)
    monkeypatch.setattr(offline, "create_environment", create)
    monkeypatch.setattr(offline, "install_requirements", lambda *args: None)
    monkeypatch.setattr(offline, "verify_environment", lambda *args: None)
    first, second = offline.setup(backend), offline.setup(backend)
    assert first != second and seen == [first, second]
    assert first.exists() and second.exists()
    assert offline.active_environment(backend) == second


def test_launcher_uses_active_python_with_isolated_flags(tmp_path, monkeypatch):
    backend = tmp_path / "GRIM_Backend"
    backend.mkdir()
    target = backend / offline.ENVIRONMENTS / ("env-" + uuid.uuid4().hex)
    (target / "Scripts").mkdir(parents=True)
    (target / "Scripts" / "python.exe").write_bytes(b"test double")
    (target.parent / offline.ACTIVE_FILE).write_text(json.dumps({"environment": target.name}))
    monkeypatch.setattr(offline, "__file__", str(backend / "setup_offline.py"))
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(offline.subprocess, "run", fake_run)
    assert offline.main(["--launch"]) == 0
    command, options = calls[0]
    assert command == [str(target / "Scripts" / "python.exe"), "-I", "-B", str(backend / "run_gui.py")]
    assert options["cwd"] == backend.parent


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher")
@pytest.mark.parametrize("filename", ["Setup_GRIM_Offline.bat", "Launch_GRIM_Offline.bat"])
def test_batch_missing_runtime_from_path_with_spaces(tmp_path, filename):
    backend = tmp_path / "GRIM with spaces" / "GRIM_Backend"
    backend.mkdir(parents=True)
    source = Path(offline.__file__).parent / filename
    shutil.copy2(source, backend / filename)
    result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", filename],
                            cwd=backend, input="\n", capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert "python_env" in result.stdout and "python.exe" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CPython embedded-layout fixture")
def test_embedded_layout_real_python_pip_and_site_processing(tmp_path, monkeypatch):
    """Real interpreter + local pip, with an embedded-layout fixture.

    No virtualenv wheel is available locally. Reproduce its documented Windows
    copy layout directly, then exercise actual Python path initialization and
    installation. This does NOT claim to execute virtualenv's creator.
    """
    import ensurepip
    version_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    version_string = ".".join(map(str, sys.version_info[:3]))
    host = Path(sys.base_prefix)
    base = tmp_path / "embedded Python"
    base.mkdir()
    shutil.copy2(host / "python.exe", base / "python.exe")
    for parent in (host, host / "DLLs"):
        for source in parent.iterdir():
            if source.suffix.lower() in (".dll", ".pyd"):
                shutil.copy2(source, base / source.name)
    # The official embedded package uses compiled stdlib in this ZIP. Python
    # source modules exercise the same ZIP/path discovery in this local fixture.
    with zipfile.ZipFile(base / f"python{version_tag}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in (host / "Lib").rglob("*.py"):
            relative = source.relative_to(host / "Lib")
            if relative.parts[0] not in {"site-packages", "test", "tests", "ensurepip", "venv", "idlelib", "tkinter"}:
                archive.write(source, relative.as_posix())
    pth = base / f"python{version_tag}._pth"
    pth.write_text(f"python{version_tag}.zip\n.\n# import site\n")
    original = pth.read_bytes()
    probe = subprocess.run([str(base / "python.exe"), "-I", "-c",
        "import sys, json; print(json.dumps([sys.flags.isolated, sys.flags.no_site, sys.path]))"],
        capture_output=True, text=True, check=True)
    isolated, no_site, paths = json.loads(probe.stdout)
    assert isolated == no_site == 1
    assert str(base / f"python{version_tag}.zip") in paths

    target = tmp_path / "created environment"
    scripts = target / "Scripts"
    packages = target / "Lib" / "site-packages"
    scripts.mkdir(parents=True)
    packages.mkdir(parents=True)
    for source in base.iterdir():
        if source.suffix.lower() in (".exe", ".dll", ".pyd", ".zip"):
            shutil.copy2(source, scripts / source.name)
    (target / "pyvenv.cfg").write_text(f"home = {base}\ninclude-system-site-packages = false\nversion = {version_string}\n")
    (packages / "probe_site.pth").write_text("import sys; sys.grim_site_processing = True\n")
    child = subprocess.run([str(scripts / "python.exe"), "-I", "-B", "-c",
        "import sys, site, ssl, socket, sqlite3, json; print(json.dumps([sys.prefix, sys.base_prefix, site.ENABLE_USER_SITE, sys.grim_site_processing]))"],
        env=offline.clean_environment(), capture_output=True, text=True, check=True)
    prefix, base_prefix, user_site, processed = json.loads(child.stdout)
    assert Path(prefix) == target and Path(base_prefix) == scripts
    assert user_site is False and processed is True
    wheelhouse = tmp_path / "offline packages"
    wheelhouse.mkdir()
    bundled = next((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
    pip_wheel = wheelhouse / bundled.name
    shutil.copy2(bundled, pip_wheel)
    version = bundled.name.split("-")[1]
    # The fixture exercises locally available pip. Production still requires
    # current pip for 3.14 tags; do not relabel or download the bundled wheel.
    monkeypatch.setattr(offline, "MINIMUM_PIP", tuple(int(part) for part in version.split(".")))
    requirements = wheelhouse / "requirements.txt"
    requirements.write_text(f"pip=={version}\n")
    offline.install_requirements(target, wheelhouse, requirements, pip_wheel, offline.clean_environment())
    installed = subprocess.run([str(scripts / "python.exe"), "-I", "-B", "-c",
        "import pip; print(pip.__file__)"], capture_output=True, text=True, check=True)
    assert Path(installed.stdout.strip()).is_relative_to(target)
    assert pth.read_bytes() == original
