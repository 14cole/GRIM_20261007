"""Optional module lookup is cached; live execution limits are not."""
import builtins
import ctypes
import sys
from types import SimpleNamespace
from unittest import mock

from ghost_backend.execution import options


def test_absent_psutil_is_searched_once_and_later_loaded_module_is_seen():
    real_import = builtins.__import__
    attempts = []
    def unavailable(name, *args, **kwargs):
        if name == 'psutil':
            attempts.append(name)
            raise ModuleNotFoundError("No module named 'psutil'", name='psutil')
        return real_import(name, *args, **kwargs)
    with mock.patch.dict(sys.modules), mock.patch.object(options, '_PSUTIL_MISSING', False), \
            mock.patch.object(builtins, '__import__', side_effect=unavailable):
        sys.modules.pop('psutil', None)
        for _ in range(20):
            assert options._optional_psutil() is None
        assert attempts == ['psutil']
        module = SimpleNamespace()
        sys.modules['psutil'] = module
        assert options._optional_psutil() is module
        sys.modules['psutil'] = None
        assert options._optional_psutil() is None
        assert attempts == ['psutil']


def test_other_import_failures_remain_retryable():
    for failure in (ImportError('temporary loader failure'),
                    ModuleNotFoundError('missing extension', name='_psutil_windows')):
        with mock.patch.dict(sys.modules), mock.patch.object(options, '_PSUTIL_MISSING', False), \
                mock.patch.object(builtins, '__import__', side_effect=failure) as loader:
            sys.modules.pop('psutil', None)
            assert options._optional_psutil() is None
            assert options._optional_psutil() is None
            assert loader.call_count == 2
            assert options._PSUTIL_MISSING is False


def test_psutil_affinity_is_queried_again_and_failure_keeps_fallback():
    process = mock.Mock()
    process.cpu_affinity.side_effect = [[0, 1, 2, 3], [0, 1], RuntimeError('probe unavailable')]
    module = SimpleNamespace(Process=mock.Mock(return_value=process))
    host = SimpleNamespace(cpu_count=lambda: 8, environ={}, name='posix')
    with mock.patch.dict(sys.modules, {'psutil': module}), mock.patch.object(options, 'os', host):
        assert options._usable_logical_cpus() == 4
        assert options._usable_logical_cpus() == 2
        assert options._usable_logical_cpus() == 8
    assert process.cpu_affinity.call_count == 3


def test_windows_fallback_affinity_slurm_and_scoped_allocation_remain_live():
    masks = iter((0b11111111, 0b11, 0b1111, 0b11111111, 0b11111111))
    def affinity(handle, process_mask, system_mask):
        process_mask._obj.value = next(masks)
        system_mask._obj.value = 0b11111111
        return 1
    kernel = SimpleNamespace(GetCurrentProcess=mock.Mock(return_value=42),
                             GetProcessAffinityMask=mock.Mock(side_effect=affinity))
    host = SimpleNamespace(cpu_count=lambda: 8, environ={}, name='nt')
    with mock.patch.object(options, '_optional_psutil', return_value=None), \
            mock.patch.object(options, 'os', host), \
            mock.patch.object(ctypes, 'windll', SimpleNamespace(kernel32=kernel), create=True):
        assert options._usable_logical_cpus() == 8
        assert options._usable_logical_cpus() == 2
        host.environ['SLURM_CPUS_PER_TASK'] = '1'
        assert options._usable_logical_cpus() == 1
        host.environ.clear()
        with options._ASSEMBLY_ALLOCATION.override(3):
            assert options.allocated_cpu_budget() == 3
        assert options.allocated_cpu_budget() == 8
    assert kernel.GetProcessAffinityMask.call_count == 5
