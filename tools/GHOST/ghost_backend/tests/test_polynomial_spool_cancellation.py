"""A one-shot real cancellation must not be mistaken for optional disk failure."""
from contextlib import ExitStack
import os
from unittest import mock

import pytest

from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.twod.assembly import polynomial_pair as pp
from ghost_backend.twod.assembly.session import AssemblySession, _SESSION
from test_polynomial_pair import inputs


class OneShotAbort:
    def __init__(self):
        self.armed = False
        self.raised = 0

    def is_set(self):
        if self.armed:
            self.armed = False
            self.raised += 1
            return True
        return False


def setup(stack, backend, directory):
    stack.enter_context(mock.patch.dict(os.environ, GHOST_POLYNOMIAL_PAIR='auto'))
    stack.enter_context(execution_scope(validate_options(dict(
        factorization=backend, assembly_threads=1, blas_threads=1,
        ram_budget_gib=12., temporary_directory=str(directory)))))
    stack.enter_context(mock.patch.object(pp, 'DENSE_SPOOL_MIN_BYTES', 0))
    mesh, infos = inputs('mixed', count=48)
    session = AssemblySession()
    session.abort_event = OneShotAbort()
    if backend == 'compressed':
        session.compressed_partner = (mesh, infos)
        from ghost_backend.compressed.runtime import regional
        from ghost_backend.compressed.retained_storage import RetainedOperatorSpool
        from ghost_backend.compressed import projection
        spool, projection_module, projection_name = RetainedOperatorSpool, projection, 'project_operator'
        call = regional
    else:
        from ghost_backend.twod.assembly import dense_pair_storage
        spool, projection_module, projection_name = dense_pair_storage.DensePairSpool, dense_pair_storage, 'project_spooled'
        call = pp.dense_system
    stack.enter_context(_SESSION.override(session))
    pair = stack.enter_context(pp.polynomial_pair_scope(3))
    return mesh, infos, session, pair, spool, projection_module, projection_name, call


@pytest.mark.parametrize('backend', ['dense', 'compressed'])
@pytest.mark.parametrize('stage', ['write', 'project'])
def test_one_shot_cancellation_during_optional_build_stops_without_fallback(tmp_path, backend, stage):
    retained = []
    with ExitStack() as stack:
        mesh, infos, session, pair, spool, module, name, call = setup(stack, backend, tmp_path)
        original_init, original_project = spool.__init__, getattr(module, name)

        def initialize(self, *args, **kwargs):
            retained.append(self)
            if stage == 'write':
                session.abort_event.armed = True
            original_init(self, *args, **kwargs)

        def project(*args, **kwargs):
            if stage == 'project':
                session.abort_event.armed = True
            return original_project(*args, **kwargs)

        stack.enter_context(mock.patch.object(spool, '__init__', initialize))
        stack.enter_context(mock.patch.object(module, name, project))
        with pytest.raises(InterruptedError, match='canceled by user'):
            call(mesh, infos, 'TE')
        assert session.abort_event.raised == 1
        assert not session.abort_event.is_set()  # a rebuild could otherwise succeed
        assert not pair.building and not pair.pending and session.pending is None
        assert not any(event.get('action') == 'independent_assembly' for event in pair.evidence)
    assert all(item.file is None or item.file.closed for item in retained)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('backend', ['dense', 'compressed'])
def test_one_shot_cancellation_during_retained_restore_stops_without_reassembly(tmp_path, backend):
    with ExitStack() as stack:
        mesh, infos, session, pair, _, _, _, call = setup(stack, backend, tmp_path)
        coarse, _ = call(mesh, infos, 'TE')
        assert coarse is not None and pair.pending
        retained = [value[0] for value in pair.pending.values()]
        session.pending = None
        session.abort_event.armed = True
        with pytest.raises(InterruptedError, match='canceled by user'):
            call(pp.cubic_mesh(mesh), infos, 'TE')
        assert session.abort_event.raised == 1
        assert not session.abort_event.is_set()
        assert not pair.pending
        assert not any(event.get('action') == 'independent_assembly' for event in pair.evidence)
    assert all(item.file is None or item.file.closed for item in retained)
    assert not list(tmp_path.iterdir())
