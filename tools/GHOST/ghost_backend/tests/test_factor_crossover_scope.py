"""Measured choices agree with planning, preserve explicit overrides and restore."""
import os
from unittest import mock
import numpy as np
import pytest
from ghost_backend.linalg import crossover
from ghost_backend.linalg.hierarchical import automatic_hierarchical
from ghost_backend.linalg.dense import DenseFactor


def test_scope_installs_choices_and_restores_after_failure():
    with mock.patch.dict(os.environ,{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':''}):
        assert not automatic_hierarchical(6000)
        with crossover.scope({}):
            crossover.install({'6000':'hodlr',16000:'lu'})
            assert automatic_hierarchical(6000)
            assert not automatic_hierarchical(16000)
            with pytest.raises(InterruptedError):
                with crossover.scope({6000:'lu'}):
                    assert not automatic_hierarchical(6000)
                    raise InterruptedError('cancelled')
            assert automatic_hierarchical(6000)
        assert not automatic_hierarchical(6000)
    with pytest.raises(RuntimeError):crossover.install({6000:'hodlr'})


def test_explicit_threshold_remains_authoritative():
    with crossover.scope({6000:'hodlr',16000:'lu'}):
        with mock.patch.dict(os.environ,{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'0'}):
            assert not automatic_hierarchical(6000)
        with mock.patch.dict(os.environ,{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'5000'}):
            assert automatic_hierarchical(16000)


def test_factor_work_records_constructor_and_multiple_checked_batches():
    from ghost_backend.execution.options import execution_scope,validate_options
    events=[]
    with execution_scope(validate_options(dict(factorization='dense'))), \
            mock.patch.dict(os.environ,{'GHOST_HIERARCHICAL_MIN_UNKNOWNS':'0'}):
        factor=DenseFactor(np.eye(12,dtype=complex),diagnostics={},evidence=events)
        first=events[0]['factor_work_seconds']
        factor.solve(np.ones((12,3),complex));second=events[0]['factor_work_seconds']
        factor.solve(np.ones((12,2),complex))
    assert 0<first<second<events[0]['factor_work_seconds']
    assert events[0]['rhs_batches']==2 and events[0]['factor_variant']=='lu'
    assert events[0]['factor_rebuilds']==0 and not events[0]['factor_fallback']
