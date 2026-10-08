"""Runtime compatibility and isolated HPC snapshot regression checks."""
import copy
from pathlib import Path
import sys
from unittest import mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.execution import provenance
from ghost_backend.hpc import runtime_snapshot


def source_tree(root):
    files = {"run_hpc_monostatic.py": "FREQUENCIES = [1.]\n",
        "bor/__init__.py": "", "bor/solver.py": "VALUE = 17\n",
        "bor/native/__init__.py": "", "bor/native/kernel.c": "/* native */",
        "bor/native/kernel.dll": "DLL", "bor/native/kernel.pyd": "extension",
        "bor/native/kernel.dylib": "dylib", "geometry/__init__.py": "",
        "geometry/geometries/BOR/example.geo": "geometry",
        "execution/__init__.py": "", "execution/DATACLASSES_LICENSE.txt": "notice",
        "tests/test_unused.py": "unused", "results/generated.py": "generated",
        "bor/__pycache__/solver.pyc": "cache", "bor/output.npz": "output"}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_kernel_and_processor_are_informational():
    before = provenance.runtime_environment_payload()
    fingerprint = provenance.runtime_environment_fingerprint()
    with mock.patch.object(provenance.platform, "release", return_value="compute-kernel"), \
            mock.patch.object(provenance.platform, "processor", return_value="compute-cpu"):
        after = provenance.runtime_environment_payload()
        assert after["platform_release"] == "compute-kernel"
        assert after["platform_processor"] == "compute-cpu"
        assert provenance.runtime_compatibility_payload(before) == provenance.runtime_compatibility_payload(after)
        assert provenance.runtime_environment_fingerprint() == fingerprint


@pytest.mark.parametrize("field", ["python_version", "python_cache_tag", "platform_system",
    "platform_machine", "byteorder", "numpy_version", "scipy_version", "numpy_config",
    "scipy_config", "execution_options"])
def test_numerical_runtime_changes_remain_incompatible(field):
    payload = copy.deepcopy(provenance.runtime_environment_payload())
    fingerprint = provenance.runtime_environment_fingerprint()
    payload[field] = "changed"
    with mock.patch.object(provenance, "runtime_environment_payload", return_value=payload):
        assert provenance.runtime_environment_fingerprint() != fingerprint


def test_snapshot_isolates_checkout_and_retains_native_integrity(tmp_path):
    source = source_tree(tmp_path / "source")
    before = provenance.backend_source_inventory(str(source))
    frozen = runtime_snapshot.snapshot_backend_runtime(source, tmp_path / "run") / "ghost_backend"
    inventory = provenance.backend_source_inventory(str(frozen))
    assert {k:v for k,v in inventory.items() if k != "ghost_backend/__init__.py"} == before
    assert (frozen / "__init__.py").is_file()
    assert not (source / "__init__.py").exists()
    assert (frozen / "geometry/geometries/BOR/example.geo").read_text() == "geometry"
    assert (frozen / "execution/DATACLASSES_LICENSE.txt").is_file()
    for relative in ("tests", "results", "bor/__pycache__", "bor/output.npz"):
        assert not (frozen / relative).exists()
    (source / "bor/solver.py").write_text("VALUE = 99")
    assert provenance.backend_source_inventory(str(frozen)) == inventory
    for suffix in (".dll", ".pyd", ".dylib"):
        path = frozen / "bor/native" / ("kernel" + suffix)
        path.write_bytes(path.read_bytes() + b"changed")
    changed = provenance.compare_source_inventories(inventory, provenance.backend_source_inventory(str(frozen)))
    assert len(changed["changed"]) == 3
    with pytest.raises(FileExistsError, match="already exists"):
        runtime_snapshot.snapshot_backend_runtime(source, tmp_path / "run")


@pytest.mark.parametrize("fault", ["copy_error", "corrupt_copy", "source_edit"])
def test_failed_snapshot_never_publishes_partial_runtime(tmp_path, monkeypatch, fault):
    source = source_tree(tmp_path / "source")
    run = tmp_path / "run"
    original = runtime_snapshot.shutil.copy2
    triggered = False
    def copy(src, dst, *args, **kwargs):
        nonlocal triggered
        assert not (run / "runtime").exists()
        if not triggered and fault == "copy_error":
            triggered = True
            raise OSError("injected copy failure")
        result = original(src, dst, *args, **kwargs)
        if not triggered:
            triggered = True
            if fault == "corrupt_copy":
                Path(dst).write_bytes(Path(dst).read_bytes() + b"corrupt")
            else:
                (source / "run_hpc_monostatic.py").write_text("FREQUENCIES = [9.]")
        return result
    monkeypatch.setattr(runtime_snapshot.shutil, "copy2", copy)
    with pytest.raises(OSError if fault == "copy_error" else RuntimeError):
        runtime_snapshot.snapshot_backend_runtime(source, run)
    assert triggered and not (run / "runtime").exists()
    assert not (run / ".runtime-snapshot.lock").exists()
    assert not list(run.glob(".runtime-staging-*"))


def test_existing_creator_lock_is_preserved(tmp_path):
    source = source_tree(tmp_path / "source")
    run = tmp_path / "run"
    run.mkdir()
    lock = run / ".runtime-snapshot.lock"
    lock.write_text("another creator")
    with pytest.raises(RuntimeError, match="Another backend snapshot"):
        runtime_snapshot.snapshot_backend_runtime(source, run)
    assert lock.read_text() == "another creator"
