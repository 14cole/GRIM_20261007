"""Physical sweep summaries and explicitly weighted empirical populations."""
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting import sweep_analysis as sweep, distribution_analysis as dist
from GRIM_Backend.plotting.modes import frequency_mode, elevation_sweep_mode, cdf_mode
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.ui.analysis_controls import PlotAnalysisControls
from PySide6.QtWidgets import QApplication
from test_plot_renderer_correctness import _RendererHarness


def grid(power, *, az=None, frequencies=None, elevations=None, units=None):
    power = np.asarray(power, float)
    if power.ndim == 1:
        power = power[:, None, None, None]
    return RcsGrid(np.arange(power.shape[0]) if az is None else az,
                   np.arange(power.shape[1]) if elevations is None else elevations,
                   np.arange(power.shape[2])+9 if frequencies is None else frequencies,
                   ["HH"], rcs_power=power,
                   units=units or {"azimuth":"deg", "elevation":"deg", "frequency":"GHz"})


def selection(dataset):
    return (np.arange(len(dataset.azimuths)), np.arange(len(dataset.elevations)), np.arange(len(dataset.frequencies)), [0])


def owner(dataset, *, config=None, cdf=None):
    result = _RendererHarness([("Spike", dataset)], selections={
        "azimuth":dataset.azimuths, "elevation":dataset.elevations,
        "frequency":dataset.frequencies,"polarization":["HH"]})
    result.analysis_controls = SimpleNamespace(sweep_aggregation=lambda:config or {}, cdf_options=lambda:cdf or {}, cdf_exceedance=lambda:bool((cdf or {}).get("exceedance",False)))
    return result


@pytest.mark.parametrize("mode,expected", [("mean",25.75),("median",1.),("percentile",25.75),("max",100.)])
def test_narrow_spike_linear_power_summary_precedes_log_conversion(mode, expected):
    ds = grid([1.,1.,1.,100.])
    config = {"mode":mode,"percentile":75.}
    h = owner(ds,config=config)
    frequency_mode.render(h)
    assert h.plot_ax.lines[0].get_ydata()[0] == pytest.approx(10*np.log10(expected))
    assert "4/4" in h.plot_ax.lines[0].get_label()
    fig = plot_datasets([("Spike",ds)],mode="frequency",azimuths=ds.azimuths,elevations=ds.elevations,frequencies=ds.frequencies,polarization="HH",sweep_options=config)
    assert fig.axes[0].lines[0].get_ydata()[0] == pytest.approx(10*np.log10(expected))


def test_individual_cuts_and_summary_identity_remain_distinct():
    ds=grid(np.arange(1.,13.).reshape(3,2,2,1))
    for renderer,mode in ((frequency_mode,"frequency"),(elevation_sweep_mode,"elevation_sweep")):
        h=owner(ds,config={"mode":"individual"})
        renderer.render(h)
        assert len(h.plot_ax.lines)==6
        fig=plot_datasets([("Spike",ds)],mode=mode,azimuths=ds.azimuths,elevations=ds.elevations,frequencies=ds.frequencies,polarization="HH",scale="linear",sweep_options={"mode":"individual"})
        assert len(fig.axes[0].lines)==6
    a=list(sweep.iter_series("A",ds,ds,selection(ds),sweep_axis="frequency",config={"mode":"mean"}))
    b=list(sweep.iter_series("A",ds,ds,selection(ds),sweep_axis="frequency",config={"mode":"max"}))
    assert a[0][3]!=b[0][3]


def test_zero_and_missing_sweep_coverage_and_circular_phase():
    reduced,count=sweep.reduce_power(np.array([[0.,np.nan],[2.,np.nan]]),sweep.options({"mode":"mean"}))
    np.testing.assert_equal(count,[2,0]); assert reduced[0]==1.; assert np.isnan(reduced[1])
    field=np.exp(1j*np.deg2rad(np.array([[179.,0.],[-179.,180.]])))
    phase,count=sweep.reduce_phase(field,sweep.options({"mode":"mean"}))
    assert abs(phase[0])==pytest.approx(180.); assert np.isnan(phase[1])
    # A zero field has no phase and must not pull a 90-degree observation toward zero.
    for mode in ("mean", "median", "percentile", "max"):
        phase,count=sweep.reduce_phase(np.array([[0j],[1j]]),sweep.options({"mode":mode}))
        assert phase[0]==pytest.approx(90.)
        assert count[0]==1
    ranked=np.array([[1.], [2.j], [-3.]])
    phase,_=sweep.reduce_phase(ranked,sweep.options({"mode":"percentile","percentile":50.}))
    assert phase[0]==pytest.approx(90.)


def test_irregular_quadrature_zero_mass_and_missing_weight_are_separate():
    # Trapezoidal weights for coordinates 0,1,10 are .05,.5,.45.
    ds=grid([0.,1.,np.nan],az=[0.,1.,10.])
    cfg=dist.options({"weighting":"azimuth","threshold_enabled":True,"threshold":0.})
    curve=dist.prepare_curve("Run",ds,selection(ds),"All",cfg,scale="dbsm")
    assert curve["population"]["counts"]==dict(selected=3,valid=2,zero=1,excluded=1)
    assert curve["population"]["coverage"]==pytest.approx(.55)
    assert curve["lookup"]==pytest.approx((100.,100.*.5/.55))
    x,y=dist.draw_values(curve,dist.bounds([curve]))
    assert y[0]==pytest.approx(100.*.05/.55)
    text=dist.table_text([curve],cfg)
    assert "including gaps" in text and "Valid weight" in text
    np.testing.assert_allclose(dist.quadrature_weights([0,0,1,10]),[.025,.025,.5,.45])


def test_threshold_ties_are_inclusive_and_zero_remains_in_both_scales():
    ds=grid([0.,1.,1.,10.])
    for scale,threshold in (("linear",1.),("dbsm",0.)):
        cfg=dist.options({"threshold_enabled":True,"threshold":threshold})
        c=dist.prepare_curve("Run",ds,selection(ds),"All",cfg,scale=scale)
        assert c["lookup"]==pytest.approx((75.,75.))
        h=owner(ds,cdf=cfg); h.combo_plot_scale.data=scale
        cdf_mode.render(h)
        assert h.plot_ax.lines[0]._grim_cdf_threshold_lookup==pytest.approx((75.,75.))
        fig=plot_datasets([("Run",ds)],mode="cdf",azimuths=ds.azimuths,elevations=ds.elevations,frequencies=ds.frequencies,polarization="HH",scale=scale,cdf_options=cfg)
        assert fig.axes[0].lines[0]._grim_cdf_threshold_lookup==pytest.approx((75.,75.))


def test_threshold_outside_population_still_intersects_known_cdf_tail():
    ds=grid([1.,10.])
    fig=plot_datasets([("Run",ds)],mode="cdf",azimuths=ds.azimuths,elevations=ds.elevations,
        frequencies=ds.frequencies,polarization="HH",cdf_options={"threshold_enabled":True,"threshold":50.})
    line=fig.axes[0].lines[0]
    assert line.get_xdata()[-1]>50.
    assert line.get_ydata()[-1]==pytest.approx(100.)
    assert line._grim_cdf_threshold_lookup==pytest.approx((100.,0.))


def test_groups_cover_every_sample_once_and_convert_reference_units():
    ds=grid(np.ones((3,1,3,1)),az=np.deg2rad([0.,30.,60.]),frequencies=np.array([9.,10.,11.])*1e9,
            units={"azimuth":"rad","elevation":"deg","frequency":"Hz"})
    ref=grid(np.ones((3,1,3,1)),az=[0.,30.,60.],frequencies=[9.,10.,11.])
    cfg=dist.options({"grouping":"band_sector","band_count":2,"sector_width_deg":30.})
    result=list(dist.groups(ds,ref,selection(ds),cfg,azimuths=ref.azimuths,frequencies=ref.frequencies))
    assert len(result)==6
    pairs=[(a,f) for sel,_ in result for a in sel[0] for f in sel[2]]
    assert len(pairs)==len(set(pairs))==9
    assert any("Band 2: 10" in label for _,label in result)


def test_dbke_frequency_conversion_occurs_before_cdf_ranking_and_runs_stay_separate():
    from test_plot_renderer_correctness import _grid
    width=_grid(quantity="sigma_2d",log_unit="dBke")
    width.rcs_power[:]=1.
    cfg=dist.options()
    c=dist.prepare_curve("Width",width,selection(width),"All",cfg,scale="dbsm")
    expected=10*np.log10(2*np.pi*width.frequencies*1e9/299792458.)
    np.testing.assert_allclose(c["levels"],expected)
    rcs=_grid();rcs.rcs_power[:]=1.
    fig=plot_datasets([("RCS",rcs),("Width",width)],mode="cdf",azimuths=rcs.azimuths,elevations=rcs.elevations,frequencies=rcs.frequencies,polarization="HH")
    assert len(fig.axes[0].lines)==2
    assert fig.axes[0].get_xlabel()=="Mixed dB"
    assert "[dBsm]" in fig.axes[0].lines[0].get_label()
    assert "[dBke]" in fig.axes[0].lines[1].get_label()
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata()[1:-1],[100.])


def test_controls_roundtrip_signals_and_malformed_settings():
    app=QApplication.instance() or QApplication([])
    controls=PlotAnalysisControls()
    seen=[]; controls.changed.connect(seen.append)
    value={"sweep":{"mode":"percentile","percentile":92.5},"cdf":{"grouping":"band_sector","band_count":3,"sector_width_deg":15.,"weighting":"azimuth_frequency","threshold_enabled":True,"threshold":-12.}}
    controls.restore_analysis_options(value)
    assert seen==[]
    assert controls.sweep_aggregation()==value["sweep"]
    assert controls.cdf_options()["weighting"]=="azimuth_frequency"
    assert controls.spin_cdf_threshold.isEnabled()
    controls.spin_sweep_percentile.setValue(80.)
    assert seen[-1]=="sweep"
    for malformed in ([],{"sweep":None,"cdf":42},{"sweep":{"percentile":"invalid"},"cdf":{"band_count":float('nan'),"threshold":None}}):
        controls.restore_analysis_options(malformed)
    controls.cdf_dialog.close()


@pytest.mark.parametrize("axes",[("azimuth","frequency"),("elevation","frequency"),("azimuth","elevation")])
def test_headless_heatmap_physical_planes_and_linked_cuts(axes):
    ds=grid(np.arange(1.,25.).reshape(3,2,4,1))
    fig=plot_datasets([("A",ds)],mode="heatmap",azimuths=ds.azimuths,elevations=ds.elevations,frequencies=ds.frequencies,polarization="HH",scale="linear",heatmap_axes=axes,heatmap_aggregation="mean_power",heatmap_cuts=True,show_colorbar=False)
    remaining=({"azimuth","elevation","frequency"}-set(axes)).pop()
    index=("azimuth","elevation","frequency").index(remaining)
    expected=ds.rcs_power[...,0].mean(axis=index)
    order=[a for a in ("azimuth","elevation","frequency") if a!=remaining]
    if order[0]!=axes[0]: expected=expected.T
    np.testing.assert_allclose(np.asarray(fig.axes[0].collections[0].get_array()).reshape(expected.T.shape),expected.T)
    assert len(fig._grim_heatmap_cut_callbacks)==1
    assert len(fig.axes)==3


def waterfall_example(*, options=None, legacy=False, multiple=False):
    data = 100.*np.arange(3)[:, None] + 10.*np.arange(4)[None, :]
    ds = grid(data[:, None, :, None], az=[-3., .75, 9.5], frequencies=[2., 4., 11., 16.])
    datasets = [("A", ds)]
    if multiple:
        datasets.append(("B", grid((data+1000.)[:, None, :, None], az=ds.azimuths, frequencies=ds.frequencies)))
    fig = plot_datasets(datasets, mode="waterfall", azimuths=ds.azimuths,
                        elevations=ds.elevations, frequencies=ds.frequencies,
                        polarization="HH", scale="linear", heatmap_cuts=legacy,
                        heatmap_options=options, show_colorbar=False)
    return fig, ds, data


@pytest.mark.parametrize("horizontal,vertical", [(False,False), (True,False), (False,True), (True,True)])
def test_headless_waterfall_independent_cuts_use_requested_irregular_sample(horizontal, vertical):
    fig, ds, data = waterfall_example(options={"linked_cuts":True,
        "horizontal_cuts":horizontal,"vertical_cuts":vertical,"cut_position":[5.,8.]})
    assert len(fig.axes) == 1 + horizontal + vertical
    image = fig.axes[0]
    assert len(image.lines) == horizontal + vertical
    assert len(fig._grim_heatmap_cut_callbacks) == int(horizontal or vertical)
    slot = 1
    if horizontal:
        cut = fig.axes[slot]; slot += 1
        np.testing.assert_allclose(cut.lines[0].get_xdata(), ds.azimuths)
        np.testing.assert_allclose(cut.lines[0].get_ydata(), data[:,2])
        assert "11 GHz" in cut.get_title()
        np.testing.assert_allclose(image.lines[0].get_ydata(), [11.,11.])
    if vertical:
        cut = fig.axes[slot]
        np.testing.assert_allclose(cut.lines[0].get_ydata(), ds.frequencies)
        np.testing.assert_allclose(cut.lines[0].get_xdata(), data[1,:])
        assert "0.75 deg" in cut.get_title()
        np.testing.assert_allclose(image.lines[-1].get_xdata(), [.75,.75])


@pytest.mark.parametrize("options,legacy,expected", [
    (None,True,2), ({"linked_cuts":True},False,2),
    ({"linked_cuts":True,"horizontal_cuts":False},False,1),
    ({"vertical_cuts":False},True,1),
    ({"horizontal_cuts":True},False,1),
    ({"linked_cuts":False},True,0),
])
def test_headless_waterfall_legacy_flags_and_default_position(options, legacy, expected):
    fig, ds, data = waterfall_example(options=options, legacy=legacy)
    assert len(fig.axes) == 1 + expected
    for cut in fig.axes[1:]:
        if cut.get_xlabel().startswith("Azimuth"):
            np.testing.assert_allclose(cut.lines[0].get_ydata(), data[:,2])
        else:
            np.testing.assert_allclose(cut.lines[0].get_xdata(), data[1,:])


def test_headless_waterfall_outside_position_snaps_to_endpoints():
    fig, ds, data = waterfall_example(options={"linked_cuts":True,"cut_position":[99.,-100.]})
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), data[:,0])
    np.testing.assert_allclose(fig.axes[2].lines[0].get_xdata(), data[-1,:])
    assert "2 GHz" in fig.axes[1].get_title()
    assert "9.5 deg" in fig.axes[2].get_title()


def test_headless_waterfall_reversed_axes_use_display_coordinates_for_snapping():
    # X is now frequency and Y is azimuth. A tie at 7.5 GHz consistently
    # chooses the first sorted sample (4 GHz), not an interpolated column.
    fig, ds, data = waterfall_example(options={"axes":["frequency","azimuth"],
        "linked_cuts":True,"cut_position":[7.5,6.]})
    horizontal, vertical = fig.axes[1:]
    np.testing.assert_allclose(horizontal.lines[0].get_xdata(),ds.frequencies)
    np.testing.assert_allclose(horizontal.lines[0].get_ydata(),data[-1,:])
    np.testing.assert_allclose(vertical.lines[0].get_ydata(),ds.azimuths)
    np.testing.assert_allclose(vertical.lines[0].get_xdata(),data[:,1])
    assert "9.5 deg" in horizontal.get_title()
    assert "4 GHz" in vertical.get_title()


def test_headless_waterfall_click_updates_enabled_cuts_on_each_panel():
    from matplotlib.backend_bases import MouseEvent
    fig, ds, data = waterfall_example(options={"vertical_cuts":True,"cut_position":[-3.,2.]},multiple=True)
    fig.canvas.draw()
    image_a, cut_a, image_b, cut_b = fig.axes
    assert len(image_a.lines) == len(image_b.lines) == 1
    click = MouseEvent("button_press_event", fig.canvas, *image_b.transData.transform((8.,5.)), button=1)
    fig.canvas.callbacks.process("button_press_event",click)
    np.testing.assert_allclose(cut_a.lines[0].get_xdata(),data[-1,:])
    # Navigation tools own mouse input while active.
    fig.canvas.toolbar = SimpleNamespace(mode="pan/zoom")
    click = MouseEvent("button_press_event", fig.canvas, *image_a.transData.transform((-2.,3.)), button=1)
    fig.canvas.callbacks.process("button_press_event",click)
    np.testing.assert_allclose(cut_a.lines[0].get_xdata(),data[-1,:])
    fig.canvas.toolbar = None
    np.testing.assert_allclose(cut_b.lines[0].get_xdata(),data[-1,:]+1000.)
    np.testing.assert_allclose(image_a.lines[0].get_xdata(),[9.5,9.5])
    np.testing.assert_allclose(image_b.lines[0].get_xdata(),[9.5,9.5])
    # A context-menu click must not change the selected sample.
    click = MouseEvent("button_press_event", fig.canvas, *image_a.transData.transform((-2.,3.)), button=3)
    fig.canvas.callbacks.process("button_press_event",click)
    np.testing.assert_allclose(cut_a.lines[0].get_xdata(),data[-1,:])


@pytest.mark.parametrize("position", [[1.], [1.,2.,3.], [np.nan,2.], [1.,np.inf]])
def test_headless_waterfall_invalid_positions_have_clear_error(position):
    with pytest.raises(ValueError,match="two finite displayed coordinates"):
        waterfall_example(options={"horizontal_cuts":True,"cut_position":position})


def test_new_headless_modes_never_import_qt():
    code=r'''
import importlib.abc, sys
class RejectQt(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('PySide','PyQt','GRIM_Backend.ui')):
            raise AssertionError('Headless plot imported Qt/UI: '+fullname)
sys.meta_path.insert(0,RejectQt())
import numpy as np
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets
d=RcsGrid([0,1],[0],[9,10],['HH'],rcs_power=np.ones((2,1,2,1)))
args=dict(datasets=[('Run',d)],azimuths=d.azimuths,elevations=d.elevations,frequencies=d.frequencies,polarization='HH')
for mode,options in [('frequency',{'sweep_options':{'mode':'max'}}),('cdf',{'cdf_options':{'threshold_enabled':True}}),('heatmap',{'heatmap_options':{'axes':['azimuth','frequency'],'linked_cuts':True}}),('waterfall',{'heatmap_options':{'horizontal_cuts':True,'vertical_cuts':False,'cut_position':[.9,9.4]}})]:
    f=plot_datasets(mode=mode,**args,**options)
    f.canvas.draw()
assert not any(n.startswith(('PySide','PyQt')) for n in sys.modules)
'''
    result=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
