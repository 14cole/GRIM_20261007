"""Physical grid, native-cut, and real widget behavior of image analysis."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from matplotlib.backends.backend_agg import FigureCanvasAgg

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.image_analysis import (AXES, ImageRecord, heatmap_plane,
    heatmap_display_plane, compare_records, summarize_roi, compact_metadata, records_from_figure,attach_isar_record)
from GRIM_Backend.plotting.modes import waterfall_mode, common
from GRIM_Backend.ui.image_analysis import ImageAnalysisControls
from GRIM_Backend.ui.image_comparison import ImageComparisonDialog, open_image_comparison
from test_plot_renderer_correctness import _RendererHarness


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def grid():
    values = np.arange(1.,49.).reshape(3,2,4,2)
    return RcsGrid([0.,1.,2.],[10.,20.],[8.,9.,10.,11.],["HH","VV"],rcs_power=values,
        rcs_phase=np.zeros(values.shape),units={"azimuth":"deg","elevation":"deg","frequency":"GHz","rcs_linear_quantity":"sigma_3d","rcs_log_unit":"dBsm"})


def owner(dataset=None, *, phase=False):
    dataset = dataset or grid()
    obj = _RendererHarness([("Measured",dataset)],selections={"azimuth":dataset.azimuths,
        "elevation":dataset.elevations,"frequency":dataset.frequencies,"polarization":dataset.polarizations},phase=phase)
    obj.plot_canvas = FigureCanvasAgg(obj.plot_figure)
    return obj


def settings(x="azimuth",y="frequency",aggregate=False,cuts=False):
    return SimpleNamespace(axes=lambda:(x,y),aggregation=lambda:"mean_power" if aggregate else "panels",linked_cuts=lambda:cuts)


@pytest.mark.parametrize("axes",list(permutations(AXES,2)))
def test_every_axis_pair_preserves_native_orientation_polarization_and_cuts(axes):
    ds = grid()
    before = ds.rcs_power.copy()
    obj = owner(ds)
    obj.combo_plot_scale.data = "linear"
    obj.image_analysis_controls = settings(*axes)
    waterfall_mode.render(obj)
    records = records_from_figure(obj)
    remaining = next(a for a in AXES if a not in axes)
    sizes = dict(zip(AXES,ds.rcs_power.shape[:3]))
    assert len(records) == 2*sizes[remaining]
    for pi in range(2):
        for ri in range(sizes[remaining]):
            record = records[pi*sizes[remaining]+ri]
            select = [slice(None)]*3+[pi]
            select[AXES.index(remaining)] = ri
            expected = ds.rcs_power[tuple(select)]
            left_axes = [a for a in AXES if a != remaining]
            if axes[0] != left_axes[0]:
                expected = expected.T
            np.testing.assert_array_equal(record.values,expected.T)
            assert record.x_axis == axes[0] and record.y_axis == axes[1]
    np.testing.assert_array_equal(ds.rcs_power,before)


def test_mean_power_and_circular_phase_preserve_physical_aggregation():
    ds = grid()
    indices = [np.arange(n) for n in ds.rcs_power.shape[:3]] + [[0]]
    actual = heatmap_display_plane(ds,indices,"azimuth","frequency",linear=True)
    np.testing.assert_allclose(actual,ds.rcs_power[:,:,:,0].mean(axis=1))
    ds.rcs_phase[:,0,:,:] = np.deg2rad(179.)
    ds.rcs_phase[:,1,:,:] = np.deg2rad(-179.)
    actual = heatmap_display_plane(ds,indices,"azimuth","frequency",phase=True)
    np.testing.assert_allclose(abs(actual),180.)
    ds.rcs_phase[:,0,:,:],ds.rcs_phase[:,1,:,:] = 0.,np.pi
    assert np.all(np.isnan(heatmap_display_plane(ds,indices,"azimuth","frequency",phase=True)))


def test_frequency_mean_uses_linear_log_reference_and_counts_true_zero():
    ds = grid()
    ds.units.update(rcs_linear_quantity="sigma_2d",rcs_log_unit="dBke")
    ds.rcs_power[:]=0.
    ds.rcs_power[:,:,0,:]=2.
    indices = [np.arange(n) for n in ds.rcs_power.shape[:3]] + [[0]]
    actual = heatmap_display_plane(ds,indices,"azimuth","elevation")
    expected = ds.linear_to_default_db(2.,frequency_value=8.)-10*np.log10(4.)
    np.testing.assert_allclose(actual,expected)


def test_zero_return_has_no_phase_and_does_not_bias_circular_mean():
    ds=grid()
    indices=[np.arange(n) for n in ds.rcs_power.shape[:3]]+[[0]]
    ds.rcs_power[:,0,:,:]=0.
    ds.rcs_phase[:,1,:,:]=np.pi/2
    np.testing.assert_allclose(heatmap_display_plane(ds,indices,"azimuth","frequency",phase=True),90.)
    ds.rcs_power[:]=0.
    assert np.all(np.isnan(heatmap_display_plane(ds,indices,"azimuth","frequency",phase=True)))


def test_invalid_negative_power_is_not_averaged_into_valid_mean():
    ds=grid()
    indices=[np.arange(n) for n in ds.rcs_power.shape[:3]]+[[0]]
    ds.rcs_power[:,0,:,:]=-100.
    ds.rcs_power[:,1,:,:]=5.
    np.testing.assert_allclose(heatmap_display_plane(ds,indices,"azimuth","frequency",linear=True),5.)


def record(values=None,**kwargs):
    values = np.asarray([[1.,2.,3.],[4.,5.,6.]] if values is None else values,float)
    return ImageRecord("Image",np.arange(values.shape[1],dtype=float),np.arange(values.shape[0],dtype=float),values,
        "azimuth","frequency","deg","GHz","sigma_3d","m²",**kwargs)


def test_matching_converted_units_and_descending_axes_preserves_signed_delta():
    a = record()
    b = replace(a,x=np.deg2rad(a.x[::-1]),y=a.y[::-1]*1e9,values=(a.values-1)[::-1,::-1],x_unit="rad",y_unit="Hz")
    result = compare_records(a,b)
    np.testing.assert_allclose(result['difference'],1.)
    stats = summarize_roi(result,(1.,2.,0.,0.))
    assert stats['pixels']==2 and stats['mean_delta']==1.


def test_grid_policy_is_explicit_linear_interpolates_power_not_db():
    b = record([[0.,20.],[0.,20.]],logarithmic=True)
    b.value_unit="dBsm"
    a = replace(b,x=np.array([0.,.5,1.]),values=np.full((2,3),10*np.log10(50.5)))
    with pytest.raises(ValueError,match="Grids differ"):
        compare_records(a,b)
    result = compare_records(a,b,policy="linear")
    np.testing.assert_allclose(result['difference'][:,1],0.,atol=1e-12)
    nearest = compare_records(a,b,policy="nearest")
    assert nearest['b'][0,1] == 0.


def test_dbke_interpolation_uses_physical_width_and_sorts_frequency_reference():
    ds=grid()
    ds.units.update(rcs_linear_quantity="sigma_2d",rcs_log_unit="dBke")
    frequencies=np.array([10.,8.])
    powers=np.array([[3.,3.],[1.,1.]])
    db=ds.linear_to_default_db(powers,frequency_value=frequencies[:,None])
    b=replace(record(db,logarithmic=True),y=frequencies,quantity="sigma_2d",value_unit="dBke")
    a=replace(b,y=np.array([8.,9.,10.]),values=ds.linear_to_default_db(np.array([[1.,1.],[2.,2.],[3.,3.]]),frequency_value=np.array([8.,9.,10.])[:,None]))
    result=compare_records(a,b,policy="linear")
    np.testing.assert_allclose(result['difference'],0.,atol=1e-12)
    # Changing only the display scale does not create a physical difference.
    linear=replace(a,values=np.array([[1.,1.],[2.,2.],[3.,3.]]),logarithmic=False,value_unit="m")
    np.testing.assert_allclose(compare_records(a,linear)['difference'],0.,atol=1e-12)


def test_explicit_supplied_coordinates_allow_unknown_unit_with_advisory():
    a=record()
    b=replace(a,x_unit="encoder_counts")
    with pytest.raises(ValueError):
        compare_records(a,b)
    result=compare_records(a,b,as_supplied=True)
    np.testing.assert_allclose(result['difference'],0.)
    assert any('could not be converted' in note for note in result['notes'])


def test_exact_interpolation_corner_not_poisoned_by_missing_neighbors():
    a = record([[2.,np.nan],[np.nan,np.nan]])
    result = compare_records(a,a,policy="linear")
    assert result['difference'][0,0] == 0.
    stats = summarize_roi(result)
    assert stats['pixels']==1 and stats['excluded_cells']==3


def test_phase_interpolation_is_circular_and_difference_wrapped():
    b = replace(record([[179.,-179.],[179.,-179.]],phase=True),quantity="phase",value_unit="deg")
    a = replace(b,x=np.array([0.,.5,1.]),values=np.full((2,3),-179.))
    result = compare_records(a,b,policy="linear")
    np.testing.assert_allclose(result['difference'][:,1],1.)
    with pytest.raises(ValueError,match="two level images"):
        compare_records(a,b,mode="relative")
    zero=replace(b,values=np.zeros((2,2)))
    stats=summarize_roi(compare_records(b,zero))
    assert abs(stats['mean_delta'])==pytest.approx(180.)
    assert stats['mean_definition']=='circular'


def test_unlike_quantities_can_compare_relative_or_explicit_numeric():
    a = record(logarithmic=True)
    a.value_unit="dBsm"
    b = replace(a,values=a.values+10.,quantity="sigma_2d",value_unit="dBke")
    with pytest.raises(ValueError,match="quantities differ"):
        compare_records(a,b)
    np.testing.assert_allclose(compare_records(a,b,mode="relative")['difference'],0.,atol=1e-12)
    np.testing.assert_allclose(compare_records(a,b,mode="displayed")['difference'],-10.)
    b=replace(b,x_axis="elevation")
    with pytest.raises(ValueError,match="Axis meanings differ"):
        compare_records(a,b,mode="displayed")
    assert compare_records(a,b,mode="displayed",as_supplied=True)['notes']


def test_compact_metadata_never_copies_solver_arrays_and_capture_is_frozen():
    class NoCopy(np.ndarray):
        def __deepcopy__(self,memo):
            raise AssertionError("raw solver array copied")
    huge = np.zeros((4,4)).view(NoCopy)
    assert compact_metadata({"raw_real":huge,"phase_reference":"origin","phase_center_xyz_m":[0,1,2]}) == {
        "phase_reference":"origin","phase_center_xyz_m":[0.,1.,2.]}
    obj=owner()
    obj.active_dataset.extra.update(raw_real=huge,phase_reference="origin")
    mesh=obj.plot_ax.imshow(np.ones((2,3)))
    band={"x_range":np.arange(3.),"y_range":np.arange(2.),"az_values":np.array([0.,1.])}
    attach_isar_record(obj,mesh,band,{"dataset":obj.active_dataset,"pol_idx":0,"elevation_deg":10.,"unit_name":"m"})
    captured=mesh._grim_image_record.frozen()
    assert 'raw_real' not in captured.metadata['manifest']['source']['metadata']
    captured.close()
    a=record()
    frozen=a.frozen()
    handle=frozen.storage._file
    a.values[:]=100.
    assert frozen.values[0,0]==1.
    frozen.close()
    assert handle.closed


def test_reduced_heatmap_cuts_probe_native_weak_samples_and_release(app):
    obj=owner()
    obj.combo_plot_scale.data="linear"
    obj._selections[obj.list_pol]=["HH"]
    obj._selections[obj.list_elev]=[10.]
    obj.image_analysis_controls=settings(cuts=True)
    original=common.decimate_image
    with mock.patch.object(common,"decimate_image",side_effect=lambda x,y,z:original(x,y,z,max_side=2,max_cells=4)):
        waterfall_mode.render(obj)
    controller=obj.plot_figure._grim_linked_cuts
    controller.update(1.,9.)
    hline,vline,*_=controller.artists[0]
    np.testing.assert_array_equal(hline.get_ydata(),obj.active_dataset.rcs_power[:,0,1,0])
    np.testing.assert_array_equal(vline.get_xdata(),obj.active_dataset.rcs_power[1,0,:,0])
    storage=obj.plot_ax.collections[0]._grim_native_samples
    handle=storage._file
    obj._release_native_plot_samples()
    assert not controller.panels and handle.closed


def test_controls_recover_from_invalid_settings_and_keep_distinct_axes(app):
    controls=ImageAnalysisControls()
    for malformed in (None,[],{"axes":None},{"axes":[[],[]]}):
        controls.restore_options(malformed)
    controls.combo_x.setCurrentIndex(2)
    assert len(set(controls.axes()))==2
    settings={"axes":["elevation","azimuth"],"aggregation":"mean_power","linked_cuts":False}
    controls.restore_options(settings)
    assert controls.options()==settings


def test_dialog_links_views_blinks_swipes_summarizes_roi_and_releases_on_reject(app):
    obj=owner()
    obj.image_analysis_controls=settings()
    waterfall_mode.render(obj)
    dialog=open_image_comparison(obj)
    assert dialog.result is not None
    np.testing.assert_allclose(dialog.mesh_a.get_clim(),dialog.image_axes[1].collections[0].get_clim())
    dialog.image_axes[0].set_xlim(.25,1.75)
    assert dialog.image_axes[1].get_xlim()==(.25,1.75)
    dialog.combo_view.setCurrentIndex(1)
    assert dialog.timer.isActive()
    dialog._blink_b=True
    dialog._update_primary()
    np.testing.assert_allclose(dialog.mesh_a.get_array(),dialog.result['b'])
    dialog.combo_view.setCurrentIndex(2)
    dialog.swipe.setValue(0)
    np.testing.assert_allclose(np.asarray(dialog.mesh_a.get_array())[:,1:],dialog.result['b'][:,1:])
    dialog._on_roi(SimpleNamespace(xdata=0.,ydata=8.),SimpleNamespace(xdata=1.,ydata=9.))
    assert "4 finite cells" in dialog.summary.text()
    handles=[dialog.a.storage._file,dialog.b.storage._file]
    dialog.reject()
    assert all(handle.closed for handle in handles)
    assert obj._image_comparison_dialog is None
    again=open_image_comparison(obj)
    assert again is not dialog and again.result is not None
    again.close()


def test_capture_after_main_clear_never_uses_disposed_native_record(app):
    obj=owner()
    obj.image_analysis_controls=settings()
    waterfall_mode.render(obj)
    dialog=ImageComparisonDialog(obj)
    expected=np.array(dialog.a.values)
    obj._clear_plot()
    dialog.capture('b')
    np.testing.assert_array_equal(dialog.a.values,expected)
    assert "Render a Waterfall" in dialog.advisory.text()
    dialog.close()


def test_comparison_allocation_failure_preserves_captures(app):
    obj=owner()
    obj.image_analysis_controls=settings()
    waterfall_mode.render(obj)
    dialog=ImageComparisonDialog(obj)
    with mock.patch('GRIM_Backend.ui.image_comparison.compare_records',side_effect=MemoryError('allocation')):
        dialog.render()
    assert "allocation failed" in dialog.advisory.text()
    assert dialog.a is not None and dialog.b is not None
    dialog.close()
