"""Parallel projection keeps bounded ownership and deterministic error evidence."""
from unittest import mock
import numpy as np
import pytest
from scipy import sparse

from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed.projection import project_operator
from ghost_backend.compressed.retained_storage import RetainedOperatorSpool
from ghost_backend.execution.options import execution_scope,validate_options
from test_compressed_path import Exact


def fixture():
    rng=np.random.default_rng(71)
    a=3*np.eye(320)+rng.standard_normal((320,7)) @ rng.standard_normal((7,320))/100
    a=a.astype(complex)+.02j
    op=StreamedOperator(Exact(a),np.arange(320)[:,None],tile=64)
    p=sparse.csc_matrix((np.tile([1.,-.25],160),(np.arange(320),np.repeat(np.arange(160),2))),shape=(320,160))
    op.row_error+=1e-14;op.column_error+=2e-14
    return a,op,p


def test_parallel_projection_matches_serial_payload_bounds_and_adjoint():
    a,fine,p=fixture();results=[]
    for workers in (1,4):
        with execution_scope(validate_options(dict(assembly_threads=workers,blas_threads=1))):
            results.append(project_operator(fine,p,np.arange(160)[:,None],2**27,tile=64))
    expected=np.asarray(p.T @ a @ p)
    for op in results:
        np.testing.assert_allclose(op.matmul(np.eye(160)),expected,atol=2e-13,rtol=2e-13)
        np.testing.assert_allclose(op.matmul(np.eye(160),2),expected.conj().T,atol=2e-13,rtol=2e-13)
    np.testing.assert_array_equal(results[0].row_error,results[1].row_error)
    np.testing.assert_array_equal(results[0].column_error,results[1].column_error)
    np.testing.assert_array_equal(results[0].row_norm,results[1].row_norm)
    assert results[1].evidence['projection_workers']==4
    assert results[1].evidence['projected_coefficient_queries']==len(results[1].tiles)


def test_disk_retention_releases_payload_and_restores_original_identity_and_bounds(tmp_path):
    a,operator,_=fixture();bounds=operator.row_error.copy();identity=operator.recycling_identity
    payload=operator.bytes
    spool=RetainedOperatorSpool(operator,tmp_path)
    assert spool.bytes==0 and spool.disk_bytes>0 and not operator.tiles
    restored=spool.restore()
    assert restored is operator and restored.bytes==payload and restored.recycling_identity==identity
    np.testing.assert_array_equal(restored.row_error,bounds)
    np.testing.assert_allclose(restored.matmul(np.eye(len(a))),a,atol=2e-13,rtol=2e-13)
    assert not list(tmp_path.iterdir())


def test_corrupt_retention_discards_partial_restoration_and_removes_file(tmp_path):
    _,operator,_=fixture();spool=RetainedOperatorSpool(operator,tmp_path)
    spool.file.seek(-1,2);spool.file.write(b'\xff');spool.file.flush()
    with pytest.raises(OSError,match='checksum'):
        spool.restore()
    assert not operator.tiles and not list(tmp_path.iterdir())


def test_cancelled_retention_does_not_destroy_source_operator(tmp_path):
    a,operator,_=fixture()
    with mock.patch.object(operator,'checkpoint',side_effect=RuntimeError('cancelled')):
        with pytest.raises(RuntimeError,match='cancelled'):
            RetainedOperatorSpool(operator,tmp_path)
    np.testing.assert_allclose(operator.matmul(np.eye(len(a))),a,atol=2e-13,rtol=2e-13)
    assert not list(tmp_path.iterdir())


def test_projection_cancellation_closes_destination_spool(tmp_path):
    _,operator,p=fixture()
    with execution_scope(validate_options(dict(assembly_threads=4,blas_threads=1))), \
            mock.patch.object(operator,'project_sparse',side_effect=RuntimeError('cancelled')):
        with pytest.raises(RuntimeError,match='cancelled'):
            project_operator(operator,p,np.arange(160)[:,None],2**27,tile=64,spool_directory=tmp_path)
    assert not list(tmp_path.iterdir())
