import gc
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from GRIM_Backend.datasets import storage_registry as registry
from GRIM_Backend.datasets.result_storage import allocate_result, array_storage, operation_result_storage, result_storage_info
from GRIM_Backend.datasets.grid import RcsGrid


def test_auto_reuses_allowance_only_after_final_plain_view_is_released(tmp_path):
    with operation_result_storage('auto', ram_budget_bytes=1024, cache_dir=tmp_path):
        first = allocate_result(1024, np.uint8, fill=17)
        view = np.asarray(first)[::2]
        del first
        gc.collect()
        assert result_storage_info()['reserved_ram_bytes'] == 1024
        second = allocate_result(1024, np.uint8)
        assert array_storage(second) == 'disk'
        np.testing.assert_array_equal(view, np.full(512, 17))
        del view, second
        gc.collect()
        assert result_storage_info()['reserved_ram_bytes'] == 0
        third = allocate_result(1024, np.uint8)
        assert array_storage(third) == 'memory'
    del third
    gc.collect()
    assert not list(tmp_path.iterdir())


def test_storage_inventory_counts_shared_samples_once_without_deriving_arrays():
    field = np.ones((2, 1, 3, 1), complex)
    dataset = RcsGrid([0, 1], [0], [9, 10, 11], ['HH'], rcs=field)
    with mock.patch.object(dataset, 'power_slice', side_effect=AssertionError('no materialization')):
        usage = registry.dataset_storage_bytes([dataset, dataset])
    assert usage == {'memory': field.nbytes, 'disk': 0}


def test_cleanup_preserves_live_and_unknown_folders_but_reclaims_crashed_cache(tmp_path):
    source = str(Path(__file__).resolve().parents[2])
    code = '''
import os, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from GRIM_Backend.datasets.result_storage import operation_result_storage, allocate_result
with operation_result_storage('disk', cache_dir=sys.argv[2]):
    data = allocate_result(4096, np.uint8, fill=11)
    os._exit(0)
'''
    unknown = tmp_path / 'grim-result-unmarked'
    unknown.mkdir()
    (unknown / 'keep.txt').write_text('untouched')
    subprocess.run([sys.executable, '-B', '-c', code, source, str(tmp_path)], check=True)
    with mock.patch.object(registry, '_CACHES', {}), mock.patch.object(registry.tempfile, 'gettempdir', return_value=str(tmp_path)):
        with operation_result_storage('disk', cache_dir=tmp_path):
            live = allocate_result(32, np.uint8, fill=5)
            rows = registry.cache_inventory((tmp_path,))
            assert len(rows) == 2 and sum(row['abandoned'] for row in rows) == 1
            result = registry.cleanup_abandoned((tmp_path,))
            assert result['removed'] == 1 and result['bytes'] == 4096 and not result['errors']
            np.testing.assert_array_equal(live, np.full(32, 5))
            assert (unknown / 'keep.txt').read_text() == 'untouched'
        del live
        gc.collect()
        assert registry.cache_inventory((tmp_path,)) == []


def test_cleanup_rejects_mismatched_record_path(tmp_path):
    folder = tmp_path / 'grim-result-invalid'
    folder.mkdir()
    registry.register_cache(folder)
    marker = folder / registry._MARKER
    payload = json.loads(marker.read_text())
    payload.update(pid=2147483647, path=str(tmp_path))
    marker.write_text(json.dumps(payload))
    with mock.patch.object(registry, '_CACHES', {}), mock.patch.object(registry.tempfile, 'gettempdir', return_value=str(tmp_path)):
        assert registry.cleanup_abandoned((tmp_path,))['removed'] == 0
    assert folder.is_dir()
    registry.forget_cache(folder)


@pytest.mark.parametrize('payload', [None, [], 42, 'text', {'pid': 1}])
def test_cleanup_ignores_malformed_marker_objects(tmp_path, payload):
    folder = tmp_path / 'grim-result-invalid'
    folder.mkdir()
    (folder / registry._MARKER).write_text(json.dumps(payload))
    with mock.patch.object(registry, '_CACHES', {}), mock.patch.object(registry.tempfile, 'gettempdir', return_value=str(tmp_path)):
        assert registry.cache_inventory((tmp_path,)) == []
        assert registry.cleanup_abandoned((tmp_path,))['removed'] == 0
    assert folder.is_dir()
