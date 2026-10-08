"""Batch scheduling resolves live CPU limits once per independent plan."""
from unittest import mock

from ghost_backend.execution.options import validate_options
from ghost_backend.runs import batch


def _unit(name):
    return dict(unit=name, backend_candidates=dict(
        dense=dict(cost=10., peak_gb=2.), compressed=dict(cost=14., peak_gb=1.)))


def test_exhaustive_search_probes_cpu_once_and_preserves_serial_cost():
    records = [_unit(str(i)) for i in range(12)]
    records.append(dict(unit='fixed', backend_candidates=dict(dense=dict(cost=10., peak_gb=2.))))
    options = validate_options(dict(assembly_threads=1, blas_threads=1))
    with mock.patch.object(batch, 'blas_thread_reservation', return_value=1) as probe:
        choices, summary = batch.select_batch_backends(records, 1, 1, 8., options)
    assert probe.call_count == 1
    assert summary['search'] == 'all_backend_combinations'
    assert summary['selected_cost'] == 130.
    assert all(choice['selected'] == 'dense' for choice in choices.values())


def test_independent_plans_reread_changed_cpu_reservation():
    options = validate_options(dict(assembly_threads=1, blas_threads='auto'))
    with mock.patch.object(batch, 'blas_thread_reservation', side_effect=[1, 2]) as probe:
        _, first = batch.select_batch_backends([_unit('a'), _unit('b')], 2, 2, 8., options)
        _, second = batch.select_batch_backends([_unit('a'), _unit('b')], 2, 2, 8., options)
    assert probe.call_count == 2
    assert first['selected_cost'] == 10.
    assert second['selected_cost'] == 20.


def test_standalone_simulation_keeps_existing_signature_and_live_reservations():
    records = [_unit('a'), _unit('b')]
    choices = dict(a='dense', b='dense')
    options = validate_options(dict(assembly_threads=1, blas_threads='auto'))
    with mock.patch.object(batch, 'blas_thread_reservation', side_effect=[1, 2]) as probe:
        assert batch._simulate(records, choices, 2, 2, 8., options) == 10.
        assert batch._simulate(records, choices, 2, 2, 8., options) == 20.
    assert probe.call_count == 2
