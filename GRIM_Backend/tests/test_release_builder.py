"""Acceptance manifests and exact-wheel exports for the embedded distribution."""
import json
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from GRIM_Backend import release_validation as release
from GRIM_Backend import setup_offline as offline
from GRIM_Backend import release_probe


def wheel(root, version="1.0", tag="py3-none-any", build=""):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"example-{version}{build}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"example-{version}.dist-info/METADATA", f"Name: Example\nVersion: {version}\n")
        archive.writestr(f"example-{version}.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
    return path


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for name in ("GRIM_Backend", "tools/GHOST", "tools/FREDDY"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "sample.py").write_text("# source\n")
    (root / "Launch_GRIM.bat").write_text("@echo off\n")
    embedded = root / "GRIM_Backend/python_env"
    embedded.mkdir()
    (embedded / "python.exe").write_bytes(b"embedded Python fixture")
    wheelhouse = root / "GRIM_Backend/offline packages"
    wheel(wheelhouse)
    (wheelhouse / "requirements.txt").write_text("Example>=1\n")
    return root


def manifest(root):
    dependencies = [{"name": "Example", "version": "1.0"}]
    return {"schema": release.SCHEMA, "validated_distribution": True,
            "source": release.source_identity(root, "0123456789abcdef"),
            "python": {"version": "3.14.6", "platform": "win-amd64", "architecture_bits": 64},
            "dependencies": dependencies,
            "wheels": release.wheel_inventory(root / "GRIM_Backend/offline packages", dependencies, ["py3-none-any"]),
            "runtime_files": release.runtime_inventory(root),
            "acceptance": {"status": "passed", "checks": ["fixture"]}}


def test_export_uses_exact_versions_and_preserves_required_assets(source, tmp_path):
    wheelhouse = source / "GRIM_Backend/offline packages"
    wheel(wheelhouse, "2.0")
    wheel(wheelhouse, "1.0", "cp314-cp314-win_arm64")
    for name in ("GRIM_Backend/reports/templates/report.pptx", "tools/FREDDY/materials/material.csv",
                 "tools/GHOST/ghost_backend/twod/assembly/native/ghost_far.dll"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    (source / "GRIM_Backend/user.csv").write_text("private test data")
    (source / "GRIM_Backend/user.txt").write_text("private test data")
    (source / "GRIM_Backend/user.grim").write_bytes(b"private data")
    cached = source / "GRIM_Backend/.offline_envs/old/Lib/site-packages/cache.py"
    cached.parent.mkdir(parents=True)
    cached.write_text("# installed package")
    data = manifest(source)
    output = release.export_bundle(source, data, tmp_path / "release")
    wheels = sorted(path.name for path in (output / "GRIM_Backend/offline packages").glob("*.whl"))
    assert wheels == ["example-1.0-py3-none-any.whl"]
    assert (output / "GRIM_Backend/offline packages/requirements.txt").read_text() == "example==1.0\n"
    assert (source / "GRIM_Backend/offline packages/requirements.txt").read_text() == "Example>=1\n"
    assert (output / "GRIM_Backend/reports/templates/report.pptx").read_bytes() == b"asset"
    assert (output / "tools/FREDDY/materials/material.csv").read_bytes() == b"asset"
    assert (output / "tools/GHOST/ghost_backend/twod/assembly/native/ghost_far.dll").read_bytes() == b"asset"
    assert not (output / "GRIM_Backend/user.csv").exists()
    assert not (output / "GRIM_Backend/user.txt").exists()
    assert not (output / "GRIM_Backend/user.grim").exists()
    assert not (output / "GRIM_Backend/.offline_envs").exists()
    exported = json.loads((output / "release-manifest.json").read_text())
    assert exported["source"]["commit"] == "0123456789abcdef"
    assert exported["bundle"]["requirements_sha256"] == release.sha256(output / "GRIM_Backend/offline packages/requirements.txt")


@pytest.mark.parametrize("group", ["source", "runtime", "wheel"])
def test_changed_inputs_never_publish_release(source, tmp_path, group):
    data = manifest(source)
    path = {"source": source / "GRIM_Backend/sample.py", "runtime": source / "GRIM_Backend/python_env/python.exe",
            "wheel": source / "GRIM_Backend/offline packages/example-1.0-py3-none-any.whl"}[group]
    path.write_bytes(path.read_bytes() + b"changed")
    output = tmp_path / "release"
    with pytest.raises(release.ReleaseError, match="changed"):
        release.export_bundle(source, data, output)
    assert not output.exists()
    assert not list(tmp_path.glob("release.staging-*"))


def test_copy_failure_does_not_publish_partial_bundle(source, tmp_path, monkeypatch):
    data = manifest(source)
    real_copy = release.shutil.copyfile
    count = 0
    def copy(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("injected copy failure")
        return real_copy(*args, **kwargs)
    monkeypatch.setattr(release.shutil, "copyfile", copy)
    with pytest.raises(OSError, match="injected"):
        release.export_bundle(source, data, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob("release.staging-*"))


def test_existing_release_is_never_replaced(source, tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    (output / "keep").write_text("existing release")
    with pytest.raises(release.ReleaseError, match="new release folder"):
        release.export_bundle(source, manifest(source), output)
    assert (output / "keep").read_text() == "existing release"


def test_release_refuses_development_only_report_and_incomplete_lock(source, tmp_path):
    data = manifest(source)
    data["validated_distribution"] = False
    with pytest.raises(release.ReleaseError, match="successful embedded"):
        release.export_bundle(source, data, tmp_path / "release")
    data["validated_distribution"] = True
    data["wheels"] = []
    with pytest.raises(release.ReleaseError, match="every exact dependency"):
        release.export_bundle(source, data, tmp_path / "release")


def test_wheel_manifest_rejects_missing_and_ambiguous_builds(tmp_path):
    deps = [{"name": "Example", "version": "1.0"}]
    wheel(tmp_path, "2.0")
    with pytest.raises(release.ReleaseError, match="installed example==1.0"):
        release.wheel_inventory(tmp_path, deps, ["py3-none-any"])
    wheel(tmp_path)
    wheel(tmp_path, build="-1")
    with pytest.raises(release.ReleaseError, match="Ambiguous"):
        release.wheel_inventory(tmp_path, deps, ["py3-none-any"])


def test_manifest_paths_cannot_escape_source(source, tmp_path):
    data = manifest(source)
    data["source"]["files"][0]["path"] = "../outside.py"
    with pytest.raises(release.ReleaseError, match="unsafe path"):
        release.export_bundle(source, data, tmp_path / "release")


def test_changed_source_during_acceptance_is_not_certified(source, monkeypatch):
    def probe(*args, **kwargs):
        (source / "GRIM_Backend/sample.py").write_text("# changed while testing")
        return {"dependencies": [{"name": "Example", "version": "1.0"}], "wheel_tags": ["py3-none-any"]}
    monkeypatch.setattr(release, "run_acceptance", probe)
    with pytest.raises(release.ReleaseError, match="changed during acceptance"):
        release.validate_release(source, source / "environment")


def test_failed_acceptance_keeps_last_working_environment(source, monkeypatch):
    backend = source / "GRIM_Backend"
    environments = backend / offline.ENVIRONMENTS
    environments.mkdir()
    pointer = environments / offline.ACTIVE_FILE
    pointer.write_text('{"environment":"previous"}')
    previous = pointer.read_bytes()
    monkeypatch.setattr(offline, "validate_base", lambda _: backend / "python_env/python.exe")
    monkeypatch.setattr(offline, "pip_bootstrap_wheel", lambda *args: Path("pip.whl"))
    monkeypatch.setattr(offline, "read_requirements", lambda *args: ["Example==1.0"])
    monkeypatch.setattr(offline, "create_environment", lambda base, target, *args: target.mkdir())
    monkeypatch.setattr(offline, "install_requirements", lambda *args: None)
    monkeypatch.setattr(offline, "verify_environment", lambda *args: None)
    def fail(target, base):
        assert pointer.read_bytes() == previous
        raise release.ReleaseError("application probe failed")
    with pytest.raises(release.ReleaseError, match="application probe failed"):
        offline.setup(backend, acceptance_check=fail)
    assert pointer.read_bytes() == previous
    assert list(environments.iterdir()) == [pointer]


def test_successful_acceptance_precedes_activation(source, monkeypatch):
    backend = source / "GRIM_Backend"
    events = []
    monkeypatch.setattr(offline, "validate_base", lambda _: backend / "python_env/python.exe")
    monkeypatch.setattr(offline, "pip_bootstrap_wheel", lambda *args: Path("pip.whl"))
    monkeypatch.setattr(offline, "read_requirements", lambda *args: ["Example==1.0"])
    monkeypatch.setattr(offline, "create_environment", lambda base, target, *args: target.mkdir())
    monkeypatch.setattr(offline, "install_requirements", lambda *args: events.append("install"))
    monkeypatch.setattr(offline, "verify_environment", lambda *args: events.append("imports"))
    monkeypatch.setattr(offline, "publish_environment", lambda *args: events.append("activate"))
    offline.setup(backend, acceptance_check=lambda *args: events.append("acceptance"))
    assert events == ["install", "imports", "acceptance", "activate"]


def test_release_preserves_validation_tool_inputs_without_run_outputs(source, tmp_path):
    validation = source / "tools/GHOST/ghost_backend/validation"
    inputs = (
        "feature_family_studies/study.template.json",
        "non_bor_feature_validation/external_case_plan.json",
        "non_bor_feature_validation/cases/wedge_ramp/case_spec.json",
        "non_bor_feature_validation/cases/wedge_ramp/feature_case.template.json",
        "material_examples/radome_material.csv",
        "material_examples/surface_impedance.csv",
        "pec_backed_ibc/example/example_coating_30mil.csv",
        "pec_backed_ibc/example/example_dielectric.csv",
    )
    outputs = ("generated/private-response.csv", "generated/private-response.grim",
               "pec_backed_ibc/example/planar_assessment.json")
    for name in inputs + outputs:
        path = validation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    output = release.export_bundle(source, manifest(source), tmp_path / "release")
    bundled = output / "tools/GHOST/ghost_backend/validation"
    for name in inputs:
        assert (bundled / name).read_text() == name
    for name in outputs:
        assert not (bundled / name).exists()


def _bootstrap_distributions(monkeypatch, environment, *, virtualenv="20.36.1", pip="25.3", external=False):
    # The requirement parser is supplied by the same installed pip used by the
    # real acceptance subprocess. Only distribution discovery is substituted.
    from pip._vendor.packaging.requirements import Requirement  # noqa: F401
    real_distribution = importlib.metadata.distribution

    def distribution(name):
        version = {"pip": pip, "virtualenv": virtualenv}.get(name)
        if name not in {"pip", "virtualenv"}:
            return real_distribution(name)
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        location = environment.parent / "outside" if external else environment
        return SimpleNamespace(version=version, locate_file=lambda _path: location / "Lib/site-packages")

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


@pytest.mark.parametrize("options", [
    {"virtualenv": None}, {"virtualenv": "20.0"}, {"pip": "25.2"}, {"external": True},
])
def test_embedded_release_requires_bootstrap_packages_in_its_lock(tmp_path, monkeypatch, options):
    environment = tmp_path / "environment"
    _bootstrap_distributions(monkeypatch, environment, **options)
    with pytest.raises(RuntimeError, match="Offline release bootstrap requires"):
        release_probe.verify_bootstrap_environment(environment)


@pytest.mark.parametrize("returncode", [0, 1])
def test_embedded_release_checks_transitive_bootstrap_dependencies(tmp_path, monkeypatch, returncode):
    environment = tmp_path / "environment"
    _bootstrap_distributions(monkeypatch, environment)
    commands = []

    def check(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=returncode, stdout="Missing virtualenv dependency", stderr="")

    monkeypatch.setattr(release_probe.subprocess, "run", check)
    if returncode:
        with pytest.raises(RuntimeError, match="Missing virtualenv dependency"):
            release_probe.verify_bootstrap_environment(environment)
    else:
        release_probe.verify_bootstrap_environment(environment)
    assert commands[0][-1] == "check"
    assert "-I" in commands[0] and "--isolated" in commands[0]
