"""Run every test, isolating Qt module lifetimes from numerical worker threads.

Usage: python ghost_backend/tests/run_suite.py [additional pytest arguments]
Each Qt-related module gets a fresh interpreter. Failed groups do not prevent
later groups from running, and any failure/crash makes the final exit nonzero.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def groups(directory):
    headless, isolated = [], []
    for path in sorted(directory.glob('test*.py')):
        source = path.read_text(encoding='utf-8-sig')
        if any(word in source for word in ('PySide', 'PyQt', 'ghost_backend.ui', 'gui_entrypoint')):
            isolated.append([path])
        else:
            headless.append(path)
    return [headless] + isolated


def main():
    directory = Path(__file__).resolve().parent
    env = dict(os.environ)
    env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env.setdefault(name, '1')
    failures = []
    for index, paths in enumerate(groups(directory)):
        if not paths:
            continue
        label = 'headless' if index == 0 else paths[0].name
        print(f'Running {label} ({len(paths)} modules)', flush=True)
        with tempfile.TemporaryDirectory(prefix='ghost-pytest-') as temporary:
            result = subprocess.run([sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q',
                                     '--basetemp',str(Path(temporary)/'tests'),
                                     *map(str, paths), *sys.argv[1:]], env=env)
        if result.returncode:
            failures.append((label, result.returncode))
    if failures:
        print('Failed groups: ' + ', '.join(f'{name} (exit {code})' for name,code in failures))
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
