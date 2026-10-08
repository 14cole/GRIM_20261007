"""Build and load-check the optional 2-D C99 kernels before replacing them."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


REQUIRED_SYMBOLS = {
    'table': ('ghost_table_eval',),
    'far': ('ghost_far_block', 'ghost_scatter_columns', 'ghost_scatter_tile',
            'ghost_scatter_has_fused'),
}
_LOAD_CHECK = (
    'import ctypes, sys\n'
    "if sys.platform == 'win32': ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    'lib = ctypes.CDLL(sys.argv[1])\n'
    'missing = [name for name in sys.argv[2:] if not hasattr(lib, name)]\n'
    'if missing:\n'
    "    sys.stderr.write('missing exports: ' + ', '.join(missing) + '\\n')\n"
    '    sys.exit(3)\n'
)


def _find_compiler(requested=None):
    if requested:
        return shutil.which(requested) or (
            str(Path(requested).resolve()) if Path(requested).is_file() else None)
    for name in ('cc', 'gcc', 'clang'):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    if platform.system().lower() == 'windows':
        for path in ('C:/msys64/ucrt64/bin/gcc.exe', 'C:/msys64/mingw64/bin/gcc.exe'):
            if Path(path).is_file():
                return path
    return None


def _subprocess_options():
    if os.name != 'nt':
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    return {'startupinfo': startup}


def validate_library(path, symbols):
    """A fresh worker must load every export without a compiler-specific PATH.

    Loading in this process would retain a Windows mapping of the staged DLL
    and prevent its atomic rename. The isolated child also avoids NumPy and
    any loaded numerical libraries masking a missing runtime dependency.
    """
    try:
        checked = subprocess.run(
            [sys.executable, '-I', '-c', _LOAD_CHECK, str(Path(path).resolve()), *symbols],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=30, **_subprocess_options())
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Native library load check timed out for {Path(path).name}. '
                           'The previous library was not replaced.') from exc
    if checked.returncode:
        detail = checked.stderr.strip() or checked.stdout.strip() or '(no output)'
        raise RuntimeError(f'Native library load check failed for {Path(path).name} '
                           f'(exit code {checked.returncode}):\n{detail}\n'
                           'The previous library was not replaced.')


def build(output_dir=None, compiler=None):
    root = Path(__file__).resolve().parent
    return [build_one(root, name, output_dir=output_dir, compiler=compiler)
            for name in REQUIRED_SYMBOLS]


def build_one(root, name, *, output_dir=None, compiler=None):
    if name not in REQUIRED_SYMBOLS:
        raise ValueError(f'Unknown native kernel: {name}')
    system = platform.system().lower()
    if system not in ('windows', 'linux'):
        raise RuntimeError('Native 2-D builds support Windows and Linux; use the NumPy fallback.')
    resolved = _find_compiler(compiler or os.environ.get('CC'))
    if resolved is None:
        raise RuntimeError('No C compiler is available; the Python evaluator remains usable.')
    root = Path(root).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    filename = f'ghost_{name}.dll' if system == 'windows' else f'libghost_{name}.so'
    output = destination / filename
    temporary = output.with_name(f'.{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}')
    # No fast-math or contraction: preserve double-precision interpolation.
    flags = ['-O3', '-std=c99', '-ffp-contract=off', '-shared']
    flags += (['-static', '-static-libgcc', '-Wl,--no-insert-timestamp']
              if system == 'windows' else ['-fPIC'])
    environment = os.environ.copy()
    if system == 'windows':
        environment['PATH'] = str(Path(resolved).resolve().parent) + os.pathsep + environment.get('PATH', '')
    command = [resolved, *flags, str(root/(name + '.c')), '-o', str(temporary), '-lm']
    try:
        compiled = subprocess.run(command, env=environment, check=False,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, **_subprocess_options())
        if compiled.returncode:
            raise RuntimeError(f'Native {name} compilation failed (exit code {compiled.returncode}):\n'
                               f'{compiled.stdout or "(no compiler output)"}\n'
                               'The previous library was not replaced.')
        validate_library(temporary, REQUIRED_SYMBOLS[name])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--compiler', default=os.environ.get('CC'))
    args = parser.parse_args()
    for output in build(args.output_dir, args.compiler):
        print(f'Built and load-checked {output}')
    print('Restart Python workers so they load the new native kernels.')


if __name__ == '__main__':
    main()
