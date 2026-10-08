"""Bounded interpolation of clean solver stage measurements.

Different backends are calibrated independently. Geometry/material/source/host
identity is retained by the request family. This never launches pilot solves,
extrapolates across factor regimes, or treats inclusive nested stage timers as
independent wall time.
"""
import math
import statistics
import time

FREQUENCY_RATIO=1.15
DOF_RATIO=1.20
ANGLE_RATIO=1.20
WORK_RATIO=1.50
RATE_SPREAD=1.20
UNCERTAINTY=.15
WIN_MARGIN=.10
MIN_SAMPLES=3
STAGES=('assembly','factorization','rhs','other')


def _ratio(a,b):
    if not math.isfinite(a) or not math.isfinite(b):return math.inf
    if min(a,b)<=0:return 1. if a==b else math.inf
    return max(a,b)/min(a,b)


def measured_stages(metadata,mode,total):
    profile=metadata.get('runtime_profile',{})
    values=profile.get('stage_seconds',{})
    assembly=(values.get('compressed_assembly',0.)+values.get('compressed_memory_sampling',0.)
              if mode=='compressed' else values.get('operators',0.))
    # rhs_compression contains linear_solve, which contains rhs_solve. Only
    # the outermost available timer contributes; excitation/projection do not.
    rhs=values.get('rhs_compression',values.get('linear_solve',values.get('rhs_solve',0.)))
    rhs+=values.get('excitation',0.)+values.get('far_field',0.)
    factor=values.get('factorization',0.)
    if min(assembly,factor,rhs)<=0:return None
    accounted=assembly+factor+rhs
    if accounted>total*(1+1e-8):return None
    return dict(assembly=assembly,factorization=factor,rhs=rhs,other=max(0.,total-accounted))


def _actual_meshes(metadata):
    actual=_cpu_state(metadata).get('stage_cost_meshes')
    if actual:return actual
    if metadata.get('stage_cost_meshes'):return metadata['stage_cost_meshes']
    adaptation=metadata.get('adaptive_mesh',{})
    if adaptation.get('used'):
        return [row for step in adaptation.get('steps',[])
                for row in step.get('backend_selection',{}).get('meshes',[])]
    return metadata.get('backend_selection',{}).get('meshes',[])


def _cpu_state(metadata):
    return metadata.get('experimental_cpu') or metadata.get('cpu_kernel_execution',{})


def observation(key,mode,seconds,metadata):
    descriptor=getattr(key,'stage',None)
    if not descriptor:return None
    from ghost_backend.execution.policy import workload
    try:
        meshes=_actual_meshes(metadata)
        systems=_cpu_state(metadata).get('systems',[])
        # Planning counts alone cannot prove which adaptive meshes ran.
        if (not meshes or not systems or any(row.get('factorizations')!=1 for row in systems) or
            sorted(int(row['unknowns']) for row in systems)!=sorted(int(row['unknowns']) for row in meshes)):
            return None
        features=workload(meshes,descriptor['angle_count'],mode,descriptor['options'])
        stages=measured_stages(metadata,mode,seconds)
        if features is None or stages is None:return None
        if not all(math.isfinite(v) and v>=0 for v in stages.values()):return None
        return dict(stamp=time.time(),family=descriptor['family'],frequency_ghz=descriptor['frequency_ghz'],
                    features=features,seconds=stages)
    except (KeyError,TypeError,ValueError,OverflowError):return None


def _stage_ratios(target,source):
    # Unknown near/far/compression mixture: bound the assembly ratio by all
    # nonzero work-component ratios rather than inventing individual timings.
    assembly=[]
    for name in ('assembly_far','assembly_near','compression'):
        a,b=target[name],source[name]
        if _ratio(a,b)>WORK_RATIO:return None
        if b>0:assembly.append(a/b)
    ratios=dict(assembly=(min(assembly),max(assembly)))
    for name in ('factorization','rhs','other'):
        a,b=target[name],source[name]
        if _ratio(a,b)>WORK_RATIO:return None
        ratios[name]=(a/b,a/b)
    return ratios


def predict(key,selection,entries,max_age):
    descriptor=getattr(key,'stage',None)
    if not descriptor:return {},None
    from ghost_backend.execution.policy import workload
    predictions={};now=time.time()
    for mode in selection.get('candidates',{}):
        if mode not in ('dense','compressed'):continue
        try:target=workload(selection.get('meshes',[]),descriptor['angle_count'],mode,descriptor['options'])
        except (KeyError,TypeError,ValueError,OverflowError):continue
        if target is None:continue
        samples=[]
        for entry in entries.values():
            if not isinstance(entry,dict):continue
            stage_rows=entry.get('_stages',{})
            if not isinstance(stage_rows,dict):continue
            rows=stage_rows.get(mode,[])
            if not isinstance(rows,list):continue
            for row in rows[-5:]:
                try:
                    if (row['family']!=descriptor['family'] or not 0<=now-row['stamp']<=max_age or
                        _ratio(row['frequency_ghz'],descriptor['frequency_ghz'])>FREQUENCY_RATIO):continue
                    source=row['features']
                    if (source['signature']!=target['signature'] or
                        len(source['unknowns'])!=len(target['unknowns']) or
                        any(_ratio(a,b)>DOF_RATIO for a,b in zip(source['unknowns'],target['unknowns'])) or
                        _ratio(source['angles'],target['angles'])>ANGLE_RATIO):continue
                    ratios=_stage_ratios(target['work'],source['work'])
                    if ratios is None:continue
                    stages=row['seconds']
                    if not all(math.isfinite(stages[k]) and stages[k]>=0 for k in STAGES):continue
                    low={k:stages[k]*ratios[k][0] for k in STAGES}
                    high={k:stages[k]*ratios[k][1] for k in STAGES}
                    samples.append((row['stamp'],low,high))
                except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):continue
        # Bound stale/noisy evidence; require repeatability after work scaling.
        samples=sorted(samples,key=lambda sample:sample[0])[-5:]
        if len(samples)<MIN_SAMPLES:continue
        centers={k:[(low[k]+high[k])/2 for _,low,high in samples] for k in STAGES}
        if any(_ratio(max(values),min(values))>RATE_SPREAD for values in centers.values()):continue
        midpoint={k:statistics.median(values) for k,values in centers.items()}
        lower=sum(min(low[k] for _,low,_ in samples) for k in STAGES)*(1-UNCERTAINTY)
        upper=sum(max(high[k] for _,_,high in samples) for k in STAGES)*(1+UNCERTAINTY)
        if not math.isfinite(upper) or lower<=0:continue
        predictions[mode]=dict(seconds=sum(midpoint.values()),stage_seconds=midpoint,
                              lower_seconds=lower,upper_seconds=upper,samples=len(samples))
    if set(predictions)!=set(('dense','compressed')):return {},None
    winner=min(predictions,key=lambda mode:predictions[mode]['seconds'])
    loser='compressed' if winner=='dense' else 'dense'
    if predictions[winner]['upper_seconds']>(1-WIN_MARGIN)*predictions[loser]['lower_seconds']:
        return {},None
    return {mode:record['seconds'] for mode,record in predictions.items()},dict(
        model='matched_stage_rates_v1',predictions=predictions,winner=winner,
        frequency_ratio_limit=FREQUENCY_RATIO,dof_ratio_limit=DOF_RATIO,
        work_ratio_limit=WORK_RATIO,uncertainty_fraction=UNCERTAINTY)
