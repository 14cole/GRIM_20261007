"""Admission samples transfer verified coefficients without changing bounds."""
import unittest
from unittest import mock
import numpy as np

from ghost_backend.compressed import pilots
from ghost_backend.compressed.memory import sample_operator
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.twod.assembly.session import AssemblySession, _SESSION


class Oracle:
    def __init__(self, n=96):
        self.n=n;self.calls=0;self.entries=0;self.max_entries=0;self.dropped_routes=0
        x=np.arange(n)/n
        self.matrix=np.exp(1j*x[:,None])*np.exp(-2j*x[None,:])+np.eye(n)
    def get_with_error(self, rows, cols):
        self.calls+=1;self.entries+=len(rows)*len(cols)
        self.max_entries=max(self.max_entries,len(rows)*len(cols))
        value=self.matrix[np.ix_(rows,cols)].copy()
        return value, np.full(value.shape,1e-18)


class AdmissionTileReuseTests(unittest.TestCase):
    def test_sampled_tiles_are_not_queried_again_and_errors_match(self):
        xy=np.column_stack((np.arange(96),np.zeros(96)))
        oracle=Oracle();oracle.pilot_identity=b'physical-identity'
        with _SESSION.override(AssemblySession()):
            sample=sample_operator(oracle,xy,tile=16,pilot_identity=oracle.pilot_identity)
            before=oracle.calls
            reused=StreamedOperator(oracle,xy,tile=16)
            self.assertEqual(reused.pilot_reuses,sample['samples'])
            self.assertEqual(oracle.calls-before,len(reused.groups)**2-sample['samples'])
        reference=StreamedOperator(Oracle(),xy,tile=16)
        np.testing.assert_array_equal(reused@np.eye(96),reference@np.eye(96))
        np.testing.assert_array_equal(reused.row_error,reference.row_error)
        np.testing.assert_array_equal(reused.column_error,reference.column_error)

    def test_different_identity_or_dof_partition_cannot_reuse(self):
        xy=np.column_stack((np.arange(96),np.zeros(96)))
        oracle=Oracle();oracle.pilot_identity=b'first'
        with _SESSION.override(AssemblySession()):
            sample_operator(oracle,xy,tile=16,pilot_identity=b'first')
            oracle.pilot_identity=b'other'
            self.assertEqual(StreamedOperator(oracle,xy,tile=16).pilot_reuses,0)
            oracle.pilot_identity=b'first'
            self.assertEqual(StreamedOperator(oracle,xy,tile=32).pilot_reuses,0)

    def test_cache_eviction_is_bounded(self):
        xy=np.column_stack((np.arange(96),np.zeros(96)))
        session=AssemblySession()
        with _SESSION.override(session),mock.patch.object(pilots,'CACHE_BYTES',20000):
            sample_operator(Oracle(),xy,tile=16,pilot_identity=b'first')
            self.assertLessEqual(session.pilot_bytes,20000)

    def test_completed_partner_uses_actual_payload_only_on_exact_key(self):
        from ghost_backend.twod.assembly.session import system_key
        from test_compact_multi_region import prepared
        mesh,infos,_=prepared('mixed','TM',32)
        session=AssemblySession()
        operator=mock.Mock(n=200,bytes=123456,evidence={'finished':True})
        session.pending=(system_key(mesh,infos,'compressed_region',8,8),(operator,{}))
        with _SESSION.override(session):
            value=pilots.assembled_partner(mesh,infos,'TM','multi_region',200)
            self.assertEqual(value['operator_bytes'],123456)
            self.assertFalse(value['sampled'])
            self.assertIsNone(pilots.assembled_partner(mesh,infos,'TE','multi_region',200))
            self.assertIsNone(pilots.assembled_partner(mesh,infos,'TM','multi_region',201))


if __name__=='__main__':unittest.main()
