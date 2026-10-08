"""Native releases keep portable defaults and publish only checked libraries."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from ghost_backend.twod.assembly.native import build as builder
from ghost_backend.bor.native import build_kernel as bor_builder


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ghost_distribution', ROOT/'scripts/build_distribution.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize('system,filename', [('Windows', 'ghost_far.dll'), ('Linux', 'libghost_far.so')])
def test_builder_validates_staged_library_then_replaces_old_one(tmp_path, monkeypatch, system, filename):
    output = tmp_path/filename
    output.write_bytes(b'previous library')
    monkeypatch.setattr(builder.platform, 'system', lambda: system)
    monkeypatch.setattr(builder, '_find_compiler', lambda requested: str(tmp_path/'compiler/gcc'))
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if command[0] != sys.executable:
            staged = Path(command[command.index('-o')+1])
            assert staged != output
            assert '-ffp-contract=off' in command
            assert '-ffast-math' not in command
            assert ('-fPIC' in command) == (system == 'Linux')
            assert ('-static' in command) == (system == 'Windows')
            staged.write_bytes(b'checked library')
        else:
            assert command[1:3] == ['-I', '-c']
            assert command[5:] == list(builder.REQUIRED_SYMBOLS['far'])
            assert output.read_bytes() == b'previous library'
            assert Path(command[4]).read_bytes() == b'checked library'
            # Compiler PATH is deliberately confined to the compilation child.
            assert 'env' not in options
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.build_one(tmp_path, 'far') == output
    assert output.read_bytes() == b'checked library'
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('failure', ['compile', 'load'])
def test_failed_native_build_preserves_previous_binary_and_cleans_stage(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(builder.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(builder, '_find_compiler', lambda requested: 'gcc')
    output = tmp_path/'libghost_table.so'
    output.write_bytes(b'previous library')

    def run(command, **options):
        compiling = command[0] != sys.executable
        if compiling:
            Path(command[command.index('-o')+1]).write_bytes(b'incomplete library')
        return subprocess.CompletedProcess(command, int(compiling == (failure == 'compile')),
                                           'compiler diagnostic', 'missing exports: ghost_table_eval')

    monkeypatch.setattr(builder.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='previous library was not replaced'):
        builder.build_one(tmp_path, 'table')
    assert output.read_bytes() == b'previous library'
    assert list(tmp_path.iterdir()) == [output]


def test_fresh_child_rejects_non_library(tmp_path):
    invalid = tmp_path/'invalid.dll'
    invalid.write_bytes(b'not a shared library')
    with pytest.raises(RuntimeError, match='load check failed'):
        builder.validate_library(invalid, ['ghost_table_eval'])


def test_load_timeout_preserves_previous_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(builder, '_find_compiler', lambda requested: 'gcc')
    output = tmp_path/'libghost_table.so'
    output.write_bytes(b'previous library')

    def run(command, **options):
        if command[0] == sys.executable:
            raise subprocess.TimeoutExpired(command, options['timeout'])
        Path(command[command.index('-o')+1]).write_bytes(b'new library')
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(builder.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='timed out'):
        builder.build_one(tmp_path, 'table')
    assert output.read_bytes() == b'previous library'
    assert list(tmp_path.iterdir()) == [output]


def test_bor_fresh_load_check_rejects_invalid_library_without_os_dialog(tmp_path):
    invalid = tmp_path/'invalid.dll'
    invalid.write_bytes(b'not a shared library')
    checked = subprocess.run([sys.executable, '-I', '-c', bor_builder._LOAD_CHECK,
                              str(invalid), *bor_builder.REQUIRED_SYMBOLS],
                             capture_output=True, text=True, timeout=10,
                             **builder._subprocess_options())
    assert checked.returncode != 0
    assert 'OSError' in checked.stderr


def _portable_wheel(tmp_path):
    wheel = tmp_path/'ghost_em2d-0.1.1-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('ghost_em2d-0.1.1.dist-info/WHEEL',
                         'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr('ghost_em2d-0.1.1.dist-info/METADATA',
                         'Metadata-Version: 2.1\nName: ghost-em2d\nVersion: 0.1.1\nRequires-Python: >=3.10\n')
        archive.writestr('ghost_em2d-0.1.1.dist-info/RECORD', '')
        archive.writestr('ghost_backend/example.py', 'value = 1\n')
    return wheel


@pytest.mark.parametrize('tag,filename', [('win_amd64', 'ghost_far.dll'), ('linux_x86_64', 'libghost_far.so')])
def test_native_wheel_platform_metadata_and_every_record_hash(tmp_path, tag, filename):
    portable = _portable_wheel(tmp_path)
    original = portable.read_bytes()
    binary = tmp_path/filename
    binary.write_bytes(b'tested native bytes')
    name = 'ghost_backend/twod/assembly/native/'+filename
    built = release.write_native_wheel(portable, tmp_path,
        {name: dict(path=binary, sha256=hashlib.sha256(binary.read_bytes()).hexdigest())}, tag=tag)
    assert built.name == f'ghost_em2d-0.1.1-py3-none-{tag}.whl'
    assert portable.read_bytes() == original
    with zipfile.ZipFile(built) as archive:
        wheel = archive.read('ghost_em2d-0.1.1.dist-info/WHEEL').decode()
        assert 'Root-Is-Purelib: false' in wheel
        assert f'Tag: py3-none-{tag}' in wheel
        assert 'py3-none-any' not in wheel
        assert archive.read(name) == binary.read_bytes()
        assert 'Requires-Python: >=3.10' in archive.read('ghost_em2d-0.1.1.dist-info/METADATA').decode()
        rows = list(csv.reader(io.StringIO(archive.read('ghost_em2d-0.1.1.dist-info/RECORD').decode())))
        assert len(rows) == len(archive.namelist())
        for entry, digest, length in rows:
            if entry.endswith('/RECORD'):
                assert digest == length == ''
                continue
            data = archive.read(entry)
            assert int(length) == len(data)
            assert digest == 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()


def test_native_wheel_rejects_changed_or_unexpected_library(tmp_path):
    portable = _portable_wheel(tmp_path)
    binary = tmp_path/'ghost_far.dll'
    binary.write_bytes(b'changed after validation')
    with pytest.raises(RuntimeError, match='changed after validation'):
        release.write_native_wheel(portable, tmp_path,
            {'ghost_backend/twod/assembly/native/ghost_far.dll': dict(path=binary, sha256='old')}, tag='win_amd64')
    with pytest.raises(RuntimeError, match='Unexpected native library'):
        release.write_native_wheel(portable, tmp_path,
            {'unrelated.dll': dict(path=binary, sha256='old')}, tag='win_amd64')


def test_portable_source_inventory_still_excludes_native_and_generated_files(tmp_path):
    for name in ('kernel.c', 'readme.md', 'ghost_table.dll', 'libghost_far.so', 'other.dylib',
                 'cached.pyc', 'generated/output.txt', '__pycache__/temp.py', 'build/temp.txt'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test')
    assert set(release.release_files(tmp_path)) == {Path('kernel.c'), Path('readme.md')}


@pytest.mark.parametrize('system,tag', [('Windows', 'win-amd64'), ('Linux', 'linux-x86_64')])
def test_native_platform_tag_uses_current_python_architecture(monkeypatch, system, tag):
    monkeypatch.setattr(release.platform, 'system', lambda: system)
    monkeypatch.setattr(release.sysconfig, 'get_platform', lambda: tag)
    assert release.native_platform_tag() == tag.replace('-', '_')


def test_platform_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(release.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(release.sysconfig, 'get_platform', lambda: 'win-amd64')
    with pytest.raises(RuntimeError, match='Cannot tag'):
        release.native_platform_tag()
