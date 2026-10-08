"""Independent regressions for the agreed Assembly audit corrections."""
import sys
from pathlib import Path
import unittest
import tempfile
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.assembly.fields import _assume_field_metadata
from ghost_backend.assembly.line_expansion import SeamCoefficients, expand_perimeter, C0
from ghost_backend.geometry.occlusion import Occluder
from ghost_backend.geometry.surface import TriangleSurface


def rectangle(x0, x1):
    return [[[x0,-1,.1],[x1,-1,.1],[x1,1,.1]],
            [[x0,-1,.1],[x1,1,.1],[x0,1,.1]]]


class AssemblyReviewFixTests(unittest.TestCase):
    def test_explicit_opposite_phase_is_never_relabelled(self):
        data={'time_convention':'exp(-jwt)', 'amp':np.array([1+1j])}
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            _assume_field_metadata(data,'body',{'time_convention':'exp(+jwt)'})
        self.assertEqual(data['time_convention'],'exp(-jwt)')
        np.testing.assert_array_equal(data['amp'],[1+1j])

    def test_missing_conventions_are_recorded(self):
        data={}
        _assume_field_metadata(data,'body',{'time_convention':'exp(+jwt)'})
        self.assertEqual(str(data['time_convention']),'exp(+jwt)')
        self.assertIn('unspecified',str(data['metadata_advisories_json']))

    def test_exact_partial_line_visibility_and_phase(self):
        coef=SeamCoefficients(1.,np.array([0.,90.,180.]),np.ones(3,complex),np.ones(3,complex))
        line=np.array([[[0.,0.,0.],[1.,0.,0.]]])
        normals=np.array([[[0.,0.,1.],[0.,0.,1.]]])
        for direction in ([0.,0.,1.],[.4,0.,np.sqrt(.84)]):
            boundary=.995-.1*direction[0]/direction[2]
            length=1-boundary
            k=2*np.pi*1e9/C0
            expected=length*np.exp(2j*k*direction[0]*(boundary+length/2))*np.sinc(k*direction[0]*length/np.pi)/(4*np.pi)
            occluder=Occluder(np.array(rectangle(-.1,.995)))
            for divisor in [20,40,160]:
                result=expand_perimeter(line,coef,None,[direction],segment_normals=normals,
                    max_piece_length_m=C0/1e9/divisor,occluder=occluder)
                np.testing.assert_allclose(result['F_vv'],[expected],rtol=2e-11,atol=1e-14)

    def test_narrow_interior_visibility_is_not_inferred_from_sample_points(self):
        occluder=Occluder(np.array(rectangle(-.1,.499)+rectangle(.501,1.1)))
        rows,lo,hi=occluder.visible_line_intervals(np.array([[0.,0,0]]),np.array([[1.,0,0]]),[0,0,1])
        np.testing.assert_array_equal(rows,[0])
        np.testing.assert_allclose([lo[0],hi[0]],[.499,.501],atol=1e-13)

    def test_batched_partial_shadow_matches_analytic_complex_integrals(self):
        x=np.linspace(.01,.5,24)
        directions=np.column_stack((x,np.zeros(len(x)),np.sqrt(1-x*x)))
        boundary=.995-.1*x/directions[:,2]
        length=1-boundary
        k=2*np.pi*1e9/C0
        expected=length*np.exp(2j*k*x*(boundary+length/2))*np.sinc(k*x*length/np.pi)/(4*np.pi)
        coef=SeamCoefficients(1.,np.array([0.,90.,180.]),np.ones(3,complex),np.ones(3,complex))
        for batch in (1,32):
            result=expand_perimeter(np.array([[[0.,0,0],[1.,0,0]]]),coef,None,directions,
                segment_normals=np.array([[[0.,0,1],[0.,0,1]]]),max_piece_length_m=.05,
                occluder=Occluder(np.array(rectangle(-.1,.995))),_look_batch_size=batch)
            np.testing.assert_allclose(result['F_vv'],expected,rtol=2e-11,atol=1e-14)

    def test_batched_nearest_matches_scalar_for_small_and_large_meshes(self):
        rng=np.random.default_rng(19)
        for triangle_count in [2,12,64]:
            triangles=rng.normal(size=(triangle_count,3,3))
            surface=TriangleSurface(triangles)
            points=rng.normal(size=(237,3))
            hints=rng.normal(size=points.shape)
            for normals in [None,hints]:
                actual=surface.nearest(points,normal_hints=normals)
                reference=surface._nearest_scalar(points,normal_hints=normals)
                for a,b in zip(actual,reference):
                    np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)

    def test_normal_hint_resolves_shared_edge(self):
        surface=TriangleSurface(np.array([[[0,0,0],[1,0,0],[0,1,0]],[[0,0,0],[0,0,1],[1,0,0]]]))
        for hint in ([0,0,1],[0,1,0]):
            a=surface.nearest([[.3,0,0]],normal_hints=[hint])
            b=surface._nearest_scalar([[.3,0,0]],normal_hints=[hint])
            for left,right in zip(a,b):
                np.testing.assert_array_equal(left,right)

    def test_interval_visibility_matches_independent_point_rays(self):
        rng=np.random.default_rng(735)
        triangles=rng.normal(size=(120,3,3))
        occluder=Occluder(triangles)
        starts=rng.normal(size=(12,3)); ends=starts+rng.normal(size=(12,3))
        t=np.linspace(.00013,.99917,97)
        for direction in ([0,0,1],[.2,.6,.8],[-.8,.5,.13]):
            rows,lo,hi=occluder.visible_line_intervals(starts,ends,direction)
            actual=np.zeros((len(starts),len(t)),bool)
            for row,left,right in zip(rows,lo,hi):
                actual[row] |= (t>left)&(t<right)
            positions=(starts[:,None,:]+t[None,:,None]*(ends-starts)[:,None,:]).reshape(-1,3)
            reference=occluder.visible(positions,direction).reshape(actual.shape)
            np.testing.assert_array_equal(actual,reference)

    def test_interval_cache_reuses_geometry_is_immutable_and_honors_cancel(self):
        occluder=Occluder(np.array(rectangle(-.1,.995)))
        start=np.array([[0.,0.,0.]]); end=np.array([[1.,0.,0.]])
        first=occluder.visible_line_intervals(start,end,[0,0,1])
        with mock.patch.object(occluder,'_leaf_direction_terms',side_effect=AssertionError('recomputed')):
            second=occluder.visible_line_intervals(start,end,[0,0,1])
        self.assertIs(first,second)
        with self.assertRaises(ValueError): first[1].setflags(write=True)
        with self.assertRaises(InterruptedError):
            occluder.visible_line_intervals(start,end,[0,0,1],cancel_check=lambda:True)

    def test_fully_visible_cache_stays_compact_and_preserves_every_piece(self):
        occluder=Occluder(np.array(rectangle(-.1,.995)))
        start=np.zeros((1000,3)); start[:,2]=1.
        end=start+np.array([1.,0,0])
        first=occluder.visible_line_intervals(start,end,[0,0,1])
        self.assertLess(occluder._line_visibility_bytes,1024)
        with mock.patch.object(occluder,'_leaf_direction_terms',side_effect=AssertionError('recomputed')):
            second=occluder.visible_line_intervals(start,end,[0,0,1])
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)
            with self.assertRaises(ValueError): b.setflags(write=True)

    def test_host_mismatch_warns_in_advisory_and_rejects_in_strict_mode(self):
        from ghost_backend.assembly.workflow import validate_installed_host
        manifest={'host':{'material':'PEC','stack_id':'metal','minimum_principal_radius_m':1.0}}
        # Use the validated schema's host applicability shape.
        from test_feature_production_contracts import manifest as make_manifest
        manifest=make_manifest('bolt','point',response_content_sha256='0'*64)
        advisory=validate_installed_host(manifest,material='different material',stack_id='',minimum_radius_m=None,required=False,label='bolt')
        self.assertTrue(advisory['warnings'])
        with self.assertRaises(ValueError):
            validate_installed_host(manifest,material='different material',stack_id='',minimum_radius_m=None,required=True,label='bolt')

    def test_retarget_preserves_physics_and_rejects_input_aliases_and_output_races(self):
        from ghost_backend.assembly import workflow as w
        from test_point_scatter_physics import _write_component_grim
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            base=root/'body.grim'
            _write_component_grim(base,[0.],[45.],[1.],np.full((1,1,1,3),1+1j))
            plan=w.prepare_feature_assembly(w.FeatureAssemblyRequest(base_grim=base,output_grim=root/'old.grim'))
            updated=w.retarget_feature_assembly(plan,root/'new.grim')
            self.assertIs(updated.radar_grid,plan.radar_grid)
            self.assertEqual(updated.prepared_source_sha256,plan.prepared_source_sha256)
            self.assertNotEqual(updated.prepared_plan_sha256,plan.prepared_plan_sha256)
            self.assertIs(w.retarget_feature_assembly(updated,updated.output_path),updated)
            with self.assertRaisesRegex(ValueError,'alias|overwrit'):
                w.retarget_feature_assembly(plan,base)
            updated.output_path.write_bytes(b'created after retarget')
            with self.assertRaisesRegex((ValueError,RuntimeError),'created after|destination'):
                w.execute_feature_assembly(updated)

if __name__=='__main__': unittest.main()
