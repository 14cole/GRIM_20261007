"""Build clean source distributions and a portable or optional native wheel.

Run from any directory with a Python that provides setuptools>=68 and wheel:
    python scripts/build_distribution.py --output /path/to/distributions
The default wheel contains portable Python and C sources. --native builds and
load-checks Windows DLLs or Linux shared objects on the release machine, then
packages a wheel tagged for that platform; recipients do not need a compiler.
"""
import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.git', 'build', 'dist', 'results', 'generated'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.dll', '.so', '.dylib', '.grim', '.lock'}


def release_files(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink()
                or any(part in SKIP_DIRS or part.endswith('.egg-info') for part in relative.parts)
                or path.suffix.lower() in SKIP_SUFFIXES or path.name == '.coverage'):
            continue
        yield relative


def _load_builder(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_platform_tag():
    system = platform.system().lower()
    if system not in ('windows', 'linux'):
        raise RuntimeError('Native releases support Windows and Linux only.')
    tag = sysconfig.get_platform().replace('-', '_').replace('.', '_')
    if not tag.startswith('win' if system == 'windows' else 'linux_'):
        raise RuntimeError(f'Cannot tag native libraries for this Python platform: {tag}')
    return tag


def build_native(stage, *, compiler=None, no_openmp=False):
    """Build only from staged source, never redistribute checkout binaries."""
    native_platform_tag()
    table_dir = stage/'ghost_backend/twod/assembly/native'
    bor_dir = stage/'ghost_backend/bor/native'
    twod = _load_builder(table_dir/'build.py', '_ghost_release_twod_builder')
    bor = _load_builder(bor_dir/'build_kernel.py', '_ghost_release_bor_builder')
    binaries = twod.build(compiler=compiler)
    command = [sys.executable, '-B', str(bor_dir/'build_kernel.py'), '--output-dir', str(bor_dir)]
    if compiler:
        command.extend(['--compiler', compiler])
    if no_openmp:
        command.append('--no-openmp')
    subprocess.run(command, check=True, **twod._subprocess_options())
    system = platform.system().lower()
    extension = '.dll' if system == 'windows' else '.so'
    kernel = bor_dir/f'bor_stream_kernel.{system}-{platform.machine().lower()}{extension}'
    binaries.append(kernel)
    checked = {}
    for path in binaries:
        symbols = (bor.REQUIRED_SYMBOLS if path == kernel
                   else twod.REQUIRED_SYMBOLS['table' if 'table' in path.stem else 'far'])
        twod.validate_library(path, symbols)
        checked[path.relative_to(stage).as_posix()] = dict(
            path=path, symbols=list(symbols), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return checked


def write_native_wheel(portable, destination, binaries, *, tag=None):
    """Retag a validated pure wheel and record its explicitly selected kernels.

    ctypes libraries have no CPython ABI dependency. Keep the Python tag py3,
    retain Requires-Python metadata, and mark the root as platform-specific.
    This intentionally makes no manylinux compatibility claim.
    """
    tag = tag or native_platform_tag()
    prefix, python_tag, abi_tag, old_platform = portable.stem.rsplit('-', 3)
    if (python_tag, abi_tag, old_platform) != ('py3', 'none', 'any'):
        raise RuntimeError('Expected a portable py3-none-any input wheel.')
    target = destination/f'{prefix}-py3-none-{tag}.whl'
    with zipfile.ZipFile(portable) as source:
        payload = {item.filename: source.read(item) for item in source.infolist()}
    wheel_metadata = next(name for name in payload if name.endswith('.dist-info/WHEEL'))
    record = next(name for name in payload if name.endswith('.dist-info/RECORD'))
    lines = payload[wheel_metadata].decode('utf-8').splitlines()
    lines = [line for line in lines if not line.startswith(('Root-Is-Purelib:', 'Tag:'))]
    lines.extend(['Root-Is-Purelib: false', f'Tag: py3-none-{tag}'])
    payload[wheel_metadata] = ('\n'.join(lines)+'\n').encode('utf-8')
    for name, details in binaries.items():
        if name in payload:
            raise RuntimeError(f'Native library conflicts with an existing wheel member: {name}')
        if (not name.startswith(('ghost_backend/bor/native/', 'ghost_backend/twod/assembly/native/'))
                or '..' in Path(name).parts or Path(name).suffix not in ('.dll', '.so')):
            raise RuntimeError(f'Unexpected native library location: {name}')
        data = details['path'].read_bytes()
        if hashlib.sha256(data).hexdigest() != details['sha256']:
            raise RuntimeError(f'Native library changed after validation: {name}')
        payload[name] = data
    rows = io.StringIO(newline='')
    writer = csv.writer(rows, lineterminator='\n')
    for name, data in sorted(payload.items()):
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
            writer.writerow((name, 'sha256='+digest, len(data)))
    writer.writerow((record, '', ''))
    payload[record] = rows.getvalue().encode('utf-8')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', action='store_true', help='Build a platform wheel with checked native kernels.')
    parser.add_argument('--compiler', help='C compiler for --native (otherwise CC, PATH, or standard MSYS2).')
    parser.add_argument('--no-openmp', action='store_true', help='Build the native BoR kernel without OpenMP.')
    args = parser.parse_args()
    if (args.compiler or args.no_openmp) and not args.native:
        parser.error('--compiler and --no-openmp require --native')
    if args.native:
        native_platform_tag()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit('Use an output directory outside the GHOST source tree.')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ghost-release-') as temporary:
        stage = Path(temporary)/'GHOST'
        artifacts_dir = Path(temporary)/'artifacts'
        artifacts_dir.mkdir()
        files = list(release_files(ROOT))
        for relative in files:
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        command = ('import sys; from setuptools.build_meta import build_wheel, build_sdist; '
                   'destination = sys.argv[1]; build_wheel(destination); build_sdist(destination)')
        subprocess.run([sys.executable, '-B', '-c', command, str(artifacts_dir)], cwd=stage, check=True)
        # Obtain the version from the project's own built metadata, including on Python 3.10.
        import email
        wheels = sorted(artifacts_dir.glob('ghost_em2d-*.whl'), key=lambda path: path.stat().st_mtime_ns)
        wheel = wheels[-1]
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith('/METADATA'))))
            required = {'ghost_backend/bor/native/bor_stream_kernel.c',
                        'ghost_backend/twod/assembly/native/table.c',
                        'ghost_backend/twod/assembly/native/far.c',
                        'ghost_backend/geometry/templates/point_features_template.csv'}
            missing = required - set(names)
            if missing:
                raise RuntimeError('Incomplete wheel: '+', '.join(sorted(missing)))
            if any(Path(n).suffix.lower() in SKIP_SUFFIXES or '/__pycache__/' in n
                   or n.startswith('ghost_backend/tests/') for n in names):
                raise RuntimeError('Wheel contains generated outputs, tests, or native binaries.')
        version = metadata['Version']
        native = None
        if args.native:
            binaries = build_native(stage, compiler=args.compiler, no_openmp=args.no_openmp)
            wheel = write_native_wheel(wheel, artifacts_dir, binaries)
            native = dict(platform=native_platform_tag(), libraries={
                name: {key: value for key, value in detail.items() if key != 'path'}
                for name, detail in binaries.items()})
        source = artifacts_dir/f'ghost-em2d-{version}-source.zip'
        inventory = {}
        with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                data = (stage/relative).read_bytes()
                # Stable metadata makes repeated source archives byte-identical.
                info = zipfile.ZipInfo('GHOST/'+relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
                inventory[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        sdists = [*artifacts_dir.glob(f'ghost_em2d-{version}.tar.gz'), *artifacts_dir.glob(f'ghost-em2d-{version}.tar.gz')]
        if not sdists:
            raise RuntimeError('Source distribution was not written to the output directory.')
        artifacts = [wheel, source, *sdists]
        manifest = dict(version=version, files=inventory, artifacts={
            p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in artifacts})
        if native is not None:
            manifest['native'] = native
        for path in artifacts:
            shutil.copyfile(path, output/path.name)
        destination = output/f'ghost-em2d-{version}-manifest.json'
        destination.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(manifest=str(destination), artifacts=list(manifest['artifacts'])), indent=2))


if __name__ == '__main__':
    main()
