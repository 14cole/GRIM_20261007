"""Core history is consistent across Python, GUI publication, and native files."""
import copy
import json
import time
from unittest import mock

import numpy as np
import pytest
from PySide6.QtCore import Qt

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.operations import OPERATION_PROVENANCE_KEY, operation_events
from GRIM_Backend.datasets.transforms import offset_db, shift_dataset
from test_plot_analysis_features import _WindowCase


def grid():
    field = np.array([1.+2j, 3.-4j, 2.+1j, 4.-3j]).reshape(2, 1, 2, 1)
    return RcsGrid([0., 90.], [0.], [1., 2.], ['HH'], rcs=field,
                   history='Loaded original', source_path='original.grim',
                   units={'frequency': 'GHz', 'azimuth': 'deg', 'elevation': 'deg'})


def test_composed_operation_records_effective_parameters_once_and_leaves_source_untouched(tmp_path):
    source = grid()
    original = source.rcs
    first = offset_db(source, 3.)
    second = shift_dataset(first, azimuth_degrees=5., phase_degrees=90.)
    events = operation_events(second)
    assert [event['operation'] for event in events] == ['offset_db', 'shift_dataset']
    assert events[-1]['parameters'] == {'azimuth_degrees': 5., 'elevation_degrees': None, 'phase_degrees': 90.}
    assert events[-1]['inputs'][0]['previous_operation_id'] == events[0]['id']
    assert source.history == 'Loaded original'
    assert not operation_events(source)
    np.testing.assert_array_equal(source.rcs, original)
    assert second.history.count('GRIM operation:') == 2
    restored = RcsGrid.load(second.save(tmp_path / 'result.grim'))
    assert operation_events(restored) == events
    assert restored.history == second.history
    np.testing.assert_array_equal(restored.rcs, second.rcs)
    continued = offset_db(restored, 2.)
    assert operation_events(continued)[-1]['inputs'][0]['source_id'] == events[-1]['output_id']


def test_merge_records_both_branch_lineages_and_common_ancestor_once():
    first = offset_db(grid(), 1.)
    left = first.shift_azimuth(1.)
    right = first.shift_azimuth(2.)
    result = RcsGrid.join_many(left, right)
    events = operation_events(result)
    assert len(events) == 4
    assert events[-1]['operation'] == 'join_many'
    assert [value['previous_operation_id'] for value in events[-1]['inputs']] == [
        operation_events(left)[-1]['id'], operation_events(right)[-1]['id']
    ]
    assert events[-1]['parameters']['overlap'] == 'error'


def test_independent_unnamed_sources_and_repeated_executions_keep_both_branches():
    a, b = grid(), grid()
    a.source_path = b.source_path = None
    b.set_complex(Ellipsis, b.rcs * 2)
    left, right = offset_db(a, 1.), offset_db(b, 1.)
    left_event, right_event = operation_events(left)[-1], operation_events(right)[-1]
    assert left_event['id'] != right_event['id']
    assert left_event['inputs'][0]['source_id'] != right_event['inputs'][0]['source_id']
    result = combine_datasets((item for item in (left, right)), 'join', overlap='last')
    events = operation_events(result)
    assert len(events) == 3
    assert {event['id'] for event in events[:-1]} == {left_event['id'], right_event['id']}
    # An edit in place does not require a full sample checksum to avoid
    # conflating the subsequent execution with its predecessor.
    a.set_complex(Ellipsis, a.rcs * 3)
    repeated = offset_db(a, 1.)
    assert operation_events(repeated)[-1]['id'] != left_event['id']


def test_failed_and_noop_operations_do_not_change_provenance():
    source = offset_db(grid(), 1.)
    history, extra = source.history, copy.deepcopy(source.extra)
    assert source.coherent_add_many() is source
    with pytest.raises(ValueError):
        source.round_axes(-3)
    assert source.history == history
    assert source.extra == extra
    # A failed decorated operation must also release its nesting context.
    assert len(operation_events(offset_db(source, 2.))) == 2


def test_metadata_policy_retains_ancillary_models_and_drops_response_claims():
    source = grid()
    source.extra.update(mesh_vertices=np.arange(9).reshape(3, 3),
                        solver_certification='source only',
                        solver_samples=np.ones(source.shape),
                        user_note='retain me')
    result = offset_db(source, 2.)
    assert result.extra['user_note'] == 'retain me'
    np.testing.assert_array_equal(result.extra['mesh_vertices'], source.extra['mesh_vertices'])
    assert not result.extra['mesh_vertices'].flags.writeable
    assert 'solver_certification' not in result.extra
    assert 'solver_samples' not in result.extra


@pytest.mark.parametrize('container', [np.asarray, list])
def test_large_axis_parameters_are_identified_without_inflating_history(container):
    source = grid()
    target = container(np.linspace(0., 90., 1001))
    result = source.interpolate_axis('azimuth', target)
    parameter = operation_events(result)[-1]['parameters']['new_values']
    if container is list:
        assert parameter['sequence_length'] == 1001
    else:
        assert parameter['array_shape'] == [1001]
    assert len(parameter['sha256']) == 64
    assert len(result.extra[OPERATION_PROVENANCE_KEY]) < 3000



def test_record_captures_normalized_parameters_assumptions_and_metadata_changes():
    source = grid()
    source.extra.update(solver_certification="certified source", user_note="keep")
    summary = source.statistics_dataset("p90", axes=("azimuth",), domain="magnitude")
    event = operation_events(summary)[-1]
    assert event["parameters"]["statistic"] == "p90"
    assert event["effective_parameters"]["statistic"] == "percentile"
    assert event["effective_parameters"]["percentile"] == 90.0
    assert event["details"]["reduced_axes"] == ["azimuth"]
    assert event["metadata_changes"][0]["extra"]["removed"]["solver_certification"] == "certified source"
    assert event["recorded_utc"].endswith("+00:00")
    combined = source.incoherent_add(grid())
    event = operation_events(combined)[-1]
    assert len(event["inputs"]) == 2
    assert len(event["metadata_changes"]) == 2
    assert "Linear power" in event["assumptions"]["combination_domain"]
    assert source.extra["solver_certification"] == "certified source"


def test_record_does_not_copy_sample_sized_metadata_into_lineage(monkeypatch):
    source = grid()
    source.extra["ancillary_array"] = np.arange(1_000_000, dtype=np.float64)
    result = offset_db(source, 1.)
    event = operation_events(result)[-1]
    assert len(result.extra[OPERATION_PROVENANCE_KEY]) < 4000
    assert result.extra["ancillary_array"].shape == (1_000_000,)
    assert event["metadata_changes"] == [{}]


def test_nested_metadata_arrays_are_described_without_content_scans():
    from GRIM_Backend.datasets.operations import _metadata_snapshot
    class MetadataArray(np.ndarray):
        def tolist(self):
            raise AssertionError("nested metadata should not be expanded")
        def __array__(self, *args, **kwargs):
            raise AssertionError("nested metadata should not be copied")
    source = grid()
    source.extra["nested"] = {"samples": np.arange(1000).view(MetadataArray)}
    metadata = _metadata_snapshot(source)
    assert metadata['extra']['nested']['samples'] == {
        'array_shape': [1000], 'dtype': source.extra['nested']['samples'].dtype.str,
        'array_values_compared': False,
    }


def test_inferred_periodic_median_choice_is_recorded_as_effective():
    from GRIM_Backend.datasets.transforms import medianize_azimuth
    source = RcsGrid(np.arange(0., 361., 90.), [0.], [10.], ['HH'],
                     rcs_power=np.ones((5, 1, 1, 1)))
    result = medianize_azimuth(source, window_degrees=90., slide_degrees=90.)
    event = operation_events(result)[-1]
    assert event['parameters']['periodic'] is None
    assert event['effective_parameters']['periodic'] is True
    assert 'periodic=True' in result.history
    assert source.history is None


def test_gate_records_actual_compensation_and_time_convention():
    from GRIM_Backend.datasets.transforms import time_gate
    source = RcsGrid([0.], [0.], np.linspace(8., 12., 33), ['HH'],
                     rcs=np.ones((1, 1, 33, 1), dtype=np.complex128),
                     units={'frequency': 'GHz'})
    result = time_gate(source, start_m=-0.2, stop_m=0.2, mode=" REMOVE ", compensate=True)
    event = operation_events(result)[-1]
    assert event['parameters']['compensate'] is True
    assert event['effective_parameters']['compensate'] is False
    assert event['effective_parameters']['mode'] == 'remove'
    assert event['assumptions']['time_convention_used'] == 'exp(+jwt)'
    assert 'compensation off' in result.history


class TestGuiOperationHistory(_WindowCase):
    def test_offset_button_has_identical_history_and_parameters_to_python(self):
        self.select_rows(0)
        source = self.window.table.item(0, 0).data(Qt.UserRole)
        expected = offset_db(source, 3.)
        before = self.window.table.rowCount()
        with mock.patch('GRIM_Backend.ui.dataset_actions.QInputDialog.getDouble', return_value=(3., True)):
            self.window._offset_selected()
        deadline = time.monotonic() + 10
        while self.window._background_job_active() and time.monotonic() < deadline:
            self.app.processEvents()
        self.app.processEvents()
        assert not self.window._background_job_active()
        assert self.window.table.rowCount() == before + 1
        actual = self.window.table.item(before, 0).data(Qt.UserRole)
        assert actual.history == expected.history
        actual_event, expected_event = operation_events(actual)[-1], operation_events(expected)[-1]
        for key in ('operation', 'parameters', 'inputs', 'output_shape'):
            assert actual_event[key] == expected_event[key]
        assert actual_event['id'] != expected_event['id']
        assert self.window.table.item(before, 2).text() == expected.history
        np.testing.assert_array_equal(actual.rcs, expected.rcs)
