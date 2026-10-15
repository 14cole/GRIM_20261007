"""Validate an offline GRIM installation and export a reproducible release folder."""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from GRIM_Backend import setup_offline as offline

SCHEMA = "grim.release.v1"
_EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".offline_envs", "python_env", "offline packages",
             "tests", "Run_Recovery", "Codex_Audit", "node_modules"}
_SOURCE_SUFFIXES = {".py", ".bat", ".toml", ".md", ".svg", ".ico", ".png", ".c", ".h", ".cpp", ".m"}


def _validation_input(relative: Path):
    """Ship documented example inputs, not generated validation responses."""
    if relative.parts[:4] != ("tools", "GHOST", "ghost_backend", "validation"):
        return False
    if relative.suffix.lower() == ".json":
        return (relative.name.endswith(".template.json")
                or relative.name in {"external_case_plan.json", "case_spec.json"})
    if relative.suffix.lower() == ".csv":
        return (relative.parent.name == "material_examples"
                or (relative.parts[4:-1] == ("pec_backed_ibc", "example")
                    and relative.name in {"example_coating_30mil.csv", "example_dielectric.csv"}))
    return False


class ReleaseError(RuntimeError):
    """Release acceptance or an input integrity check failed."""


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(root: Path, path: Path):
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ReleaseError(f"Release files must stay within their source tree: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def source_inventory(root: Path):
    """Include application code/assets, excluding datasets, runtimes, caches and tests."""
    paths = list(root.glob("*.bat"))
    for name in ("GRIM_Backend", "tools/GHOST", "tools/FREDDY"):
        folder = root / name
        if not folder.is_dir():
            raise ReleaseError(f"Missing application source: {folder}")
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in _EXCLUDED or part.startswith(".") for part in relative.parts):
                continue
            suffix = path.suffix.lower()
            if (suffix in _SOURCE_SUFFIXES
                    or _validation_input(relative)
                    or (suffix == ".txt" and (path.name.lower().startswith(("readme", "requirements"))
                                               or "license" in path.name.lower()))
                    or (suffix == ".geo" and any(part in {"geometry", "validation"} for part in relative.parts))
                    or (suffix in {".csv", ".json", ".pptx"} and any(
                        part in {"materials", "templates", "classic_icons"} for part in relative.parts))
                    or (suffix in {".dll", ".so", ".dylib"} and "native" in relative.parts)):
                paths.append(path)
    for name in ("requirements.txt", "README.txt"):
        path = root / "GRIM_Backend" / "offline packages" / name
        if path.is_file():
            paths.append(path)
    return [_entry(root, path) for path in sorted(set(paths))]


def source_identity(root: Path, commit=None):
    if commit is not None and not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise ReleaseError("Source commit must be a Git commit hash")
    dirty = None
    if commit is None:
        try:
            location = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                                      text=True, timeout=15, check=False)
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=15, check=False)
            if location.returncode == result.returncode == 0 and Path(location.stdout.strip()).resolve() == root.resolve():
                commit = result.stdout.strip()
                status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                        capture_output=True, text=True, timeout=15, check=False)
                dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    files = source_inventory(root)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"commit": commit, "tracked_changes": dirty, "tree_sha256": digest, "files": files}


def runtime_inventory(root: Path):
    runtime = root / "GRIM_Backend" / "python_env"
    if not (runtime / "python.exe").is_file():
        raise ReleaseError("The embedded runtime is missing from GRIM_Backend/python_env")
    return [_entry(root, path) for path in sorted(runtime.rglob("*")) if path.is_file()]


def wheel_inventory(wheelhouse: Path, dependencies, tags):
    """Select only each exact installed version and its best compatible wheel."""
    ranks = {tag: index for index, tag in enumerate(tags)}
    wanted = {offline.normalized_name(value["name"]): str(value["version"]) for value in dependencies}
    candidates = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        try:
            with zipfile.ZipFile(wheel) as archive:
                metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata_names) != 1:
                    raise ValueError("expected one wheel metadata record")
                metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
                name, version = offline.normalized_name(metadata["Name"]), metadata["Version"]
                if wanted.get(name) != version:
                    continue
                details = Parser().parsestr(archive.read(metadata_names[0].replace("METADATA", "WHEEL")).decode("utf-8"))
                matching = [ranks[tag] for tag in details.get_all("Tag", []) if tag in ranks]
                if matching:
                    candidates.setdefault(name, []).append((min(matching), wheel))
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ReleaseError(f"Cannot inspect wheel {wheel.name}: {exc}") from exc
    result = []
    for name, version in sorted(wanted.items()):
        matches = sorted(candidates.get(name, []), key=lambda item: (item[0], item[1].name))
        if not matches:
            raise ReleaseError(f"Missing compatible wheel for installed {name}=={version}")
        best_rank, wheel = matches[0]
        if len(matches) > 1 and matches[1][0] == best_rank:
            raise ReleaseError(f"Ambiguous same-version wheels for {name}=={version}; keep one validated build")
        result.append({"name": name, "version": version, **_entry(wheelhouse, wheel)})
    return result


def run_acceptance(root: Path, python: Path, *, environment: Path | None = None):
    with tempfile.TemporaryDirectory(prefix="grim-release-check-") as directory:
        scratch = Path(directory)
        result_path = scratch / "result.json"
        command = [str(python), "-I", "-B", str(root / "GRIM_Backend" / "release_probe.py"),
                   str(root), str(scratch), str(result_path)]
        if environment is not None:
            command += ["--environment", str(environment)]
        env = offline.clean_environment()
        for name in ("GHOST_BACKEND_PATH", "FREDDY_ROOT_PATH", "GRIM_BACKEND_PATH", "GRIM_REVISED_2_PATH"):
            env.pop(name, None)
        env.update(QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(command, cwd=scratch, env=env, capture_output=True, text=True,
                                   timeout=180, check=False)
        if completed.returncode or not result_path.is_file():
            def diagnostic(output):
                return (output if len(output) <= 24000 else
                        output[:12000] + "\n... output truncated ...\n" + output[-12000:])
            raise ReleaseError("Application acceptance failed:\n" + diagnostic(completed.stdout)
                               + diagnostic(completed.stderr))
        return json.loads(result_path.read_text(encoding="utf-8"))


def validate_release(root: Path, target: Path, *, source_commit=None, wheelhouse=None):
    """Capture before checking, and refuse a manifest for changing inputs."""
    source = source_identity(root, source_commit)
    runtime = runtime_inventory(root)
    report = run_acceptance(root, target / "Scripts" / "python.exe", environment=target)
    wheels = wheel_inventory(wheelhouse or root / "GRIM_Backend" / "offline packages",
                             report["dependencies"], report["wheel_tags"])
    if source["files"] != source_inventory(root) or runtime != runtime_inventory(root):
        raise ReleaseError("Source or embedded Python changed during acceptance; run validation again")
    return {"schema": SCHEMA, "validated_distribution": True, "source": source,
            "python": report["python"], "dependencies": report["dependencies"], "wheels": wheels,
            "runtime_files": runtime, "acceptance": {"status": report["status"], "checks": report["checks"]}}


def write_manifest(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_path(root: Path, entry):
    relative = PurePosixPath(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"] or ":" in entry["path"]:
        raise ReleaseError("Manifest contains an unsafe path")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ReleaseError(f"Manifest input is missing or outside its source: {relative}")
    if sha256(path) != entry["sha256"]:
        raise ReleaseError(f"Manifest input changed: {relative}")
    return path


def export_bundle(root: Path, manifest, destination: Path):
    """Publish only recorded sources/runtime/wheels; never copy cached environments."""
    if manifest.get("schema") != SCHEMA or manifest.get("validated_distribution") is not True:
        raise ReleaseError("A successful embedded-distribution manifest is required to export a release")
    if manifest.get("acceptance", {}).get("status") != "passed":
        raise ReleaseError("The manifest has no successful acceptance result")
    required = {(offline.normalized_name(value["name"]), value["version"]) for value in manifest["dependencies"]}
    supplied = {(offline.normalized_name(value["name"]), value["version"]) for value in manifest["wheels"]}
    if not required or required != supplied or len(supplied) != len(manifest["wheels"]):
        raise ReleaseError("Manifest wheels must match every exact dependency once")
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(root.resolve()):
        raise ReleaseError("Choose a new release folder outside the source tree")
    wheelhouse = root / "GRIM_Backend" / "offline packages"
    inputs = [(root, entry, entry["path"]) for entry in manifest["source"]["files"] + manifest["runtime_files"]]
    inputs += [(wheelhouse, entry, "GRIM_Backend/offline packages/" + entry["path"]) for entry in manifest["wheels"]]
    for base, entry, _ in inputs:
        _checked_path(base, entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        for base, entry, relative in inputs:
            source = _checked_path(base, entry)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != entry["sha256"]:
                raise ReleaseError(f"Input changed while copying: {relative}")
        lock = "\n".join(sorted(f"{offline.normalized_name(item['name'])}=={item['version']}"
                                  for item in manifest["dependencies"])) + "\n"
        requirements = stage / "GRIM_Backend" / "offline packages" / "requirements.txt"
        requirements.write_text(lock, encoding="utf-8")
        payload = dict(manifest)
        payload["bundle"] = {"requirements_sha256": sha256(requirements),
                             "note": "requirements.txt pins this validated environment; original source hashes are retained above."}
        write_manifest(stage / "release-manifest.json", payload)
        if destination.exists():
            raise ReleaseError("Release destination appeared during export")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Install only local wheels and validate before activation")
    parser.add_argument("--manifest", type=Path, help="Output acceptance manifest")
    parser.add_argument("--source-commit", help="Commit hash when the source has no .git directory")
    parser.add_argument("--bundle", type=Path, help="Export a new folder with only validated source/runtime/wheels")
    parser.add_argument("--export-manifest", type=Path, help="Export a previous successful manifest without rerunning acceptance")
    parser.add_argument("--developer-python", type=Path, help="Run smoke checks only; does not certify or export an embedded release")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    backend = root / "GRIM_Backend"
    manifest_path = args.manifest or backend / "release-validation.json"
    try:
        if args.export_manifest:
            if not args.bundle or args.install or args.developer_python:
                parser.error("--export-manifest requires --bundle and cannot install or use developer Python")
            payload = json.loads(args.export_manifest.read_text(encoding="utf-8"))
        elif args.developer_python:
            if args.install or args.bundle:
                parser.error("Developer smoke checks cannot install or export a validated release")
            payload = {"schema": SCHEMA, "validated_distribution": False,
                       "acceptance": run_acceptance(root, args.developer_python),
                       "limitation": "Development Python smoke check; embedded runtime and wheel bundle were not validated."}
            write_manifest(manifest_path, payload)
        else:
            if args.install:
                checked = []
                def acceptance(target, base):
                    checked.append(validate_release(root, target, source_commit=args.source_commit))
                offline.setup(backend, acceptance_check=acceptance)
                payload = checked[0]
            else:
                payload = validate_release(root, offline.active_environment(backend), source_commit=args.source_commit)
            write_manifest(manifest_path, payload)
        if args.bundle:
            print(f"Release folder: {export_bundle(root, payload, args.bundle)}")
        print(f"Acceptance passed. {'Embedded distribution validated.' if payload['validated_distribution'] else payload['limitation']}")
        if not args.export_manifest:
            print(f"Manifest: {manifest_path}")
        return 0
    except (ReleaseError, offline.SetupError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"GRIM release validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
