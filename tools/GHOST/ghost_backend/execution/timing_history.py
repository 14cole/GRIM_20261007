"""Host-local timing evidence for identical or tightly nearby requests.

Only repeated successful measurements of at least two backends can change a
ranking. Nearby reuse also requires stable paired samples and a large winning
margin; it never extrapolates an asymptotic speedup. Source, geometry, materials,
tolerances and CPU settings stay identical. This is never a numerical cache.
A missing/unwritable cache leaves the work prior intact.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import uuid
import time
from ghost_backend.execution.runtime import unlink_if_exists

MAX_BYTES=2*1024**2
MAX_ENTRIES=128
MAX_AGE=14*24*3600
NEARBY_FREQUENCY_RATIO=1.02
NEARBY_DOF_RATIO=1.02
NEARBY_ANGLE_RATIO=1.05
NEARBY_SPREAD_RATIO=1.10
NEARBY_UNCERTAINTY=.10
NEARBY_WIN_MARGIN=.20


class RequestKey(str):
    """Backward-compatible exact digest carrying optional interpolation inputs."""
    def __new__(cls, value, nearby=None, stage=None, factor=None):
        result=str.__new__(cls,value)
        result.nearby=nearby
        result.stage=stage
        result.factor=factor
        return result


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def with_factor_choices(key, choices):
    """Do not mix whole-run timings before and after a learned factor change."""
    if not choices:
        return key
    signature = sorted((int(n), variant) for n, variant in choices.items())
    def scoped(descriptor):
        if descriptor is None:
            return None
        return dict(descriptor, family=_digest((descriptor['family'], signature)))
    return RequestKey(_digest((str(key), signature)), scoped(key.nearby),
                      scoped(key.stage), key.factor)


def _nearby_descriptor(payload,options,arguments):
    from ghost_backend.execution.options import allocated_cpu_budget, allocated_memory_budget
    frequencies=payload.get('frequencies_ghz',[])
    angles=payload.get('elevations_deg',[])
    if (len(frequencies)!=1 or not angles or not math.isfinite(float(frequencies[0]))
            or frequencies[0]<=0):return None
    family=dict(payload)
    family.pop('frequencies_ghz',None)
    family['assembly_method']=arguments.get('solver_method','auto')
    # Uniform angle grids may vary slightly in count, but retain their range
    # and direction. Irregular grids retain every angle in the family digest.
    if len(angles)>2:
        step=(angles[-1]-angles[0])/(len(angles)-1)
        if step and all(abs((value-angles[0])-index*step)<=1e-10*max(1.,abs(step))
                        for index,value in enumerate(angles)):
            family['elevations_deg']=dict(uniform_span=[angles[0],angles[-1]])
    family['reservation']=(options.get('ram_budget_gib'),allocated_memory_budget(),allocated_cpu_budget())
    family['runtime_overrides']={name:value for name,value in os.environ.items()
                                 if name.startswith('GHOST_') and name not in ('GHOST_TIMING_CACHE_DIR','GHOST_CPU_FACTORIZATION')}
    batch=max(1,int(options.get('angle_batch_size',256)))
    family['angle_work_regime']=((len(angles)+batch-1)//batch,len(angles)>=32)
    return dict(family=_digest(family),frequency_ghz=float(frequencies[0]),angle_count=len(angles))


def cache_path():
    base=os.environ.get('GHOST_TIMING_CACHE_DIR')
    if not base:
        base=os.path.join(os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or
                          os.path.join(os.path.expanduser('~'),'.cache'),'GHOST','solver-timings')
    return Path(base)/'timings-v1.json'


def request_key(arguments,options,name):
    from ghost_backend.execution.provenance import backend_source_records, source_bundle_fingerprint
    from ghost_backend.execution.options import effective_assembly_threads
    from ghost_backend.twod.preparation import material_fingerprints, mesh_frequencies
    from ghost_backend.twod.solver import _material_base_dir_for_snapshot
    from ghost_backend.linalg.refined_lu import requested_precision
    from ghost_backend.execution.thread_control import threadpool_info
    import numpy as np
    import scipy
    snapshot=arguments.get('geometry_snapshot')
    if not isinstance(snapshot,dict):return None
    ignored={'abort_event','progress_callback','geometry_snapshot','solver_method','material_base_dir'}
    payload={k:v for k,v in arguments.items() if k not in ignored and not k.startswith('_')}
    if arguments.get('mesh_reference_ghz') is not None:
        # A checkpointed frequency may inherit sizing from its complete
        # sweep. That changes the fixed mesh even when this subrequest matches.
        payload['mesh_sizing_frequencies_ghz']=list(mesh_frequencies(arguments['frequencies_ghz']))
    payload['geometry']={k:snapshot.get(k) for k in ('segments','ibcs','dielectrics')}
    payload['materials']=material_fingerprints(snapshot,_material_base_dir_for_snapshot(snapshot,arguments.get('material_base_dir')))
    payload['precision']=requested_precision()
    payload['blas']=[{k:v for k,v in item.items() if k in ('internal_api','version','threading_layer','architecture')}
                     for item in threadpool_info()]
    payload['options']={k:v for k,v in options.items() if k not in ('factorization','ram_budget_gib','temporary_directory')}
    payload['host']=(platform.node(),platform.machine(),platform.processor(),platform.platform(),
                     platform.python_version(),np.__version__,scipy.__version__,effective_assembly_threads())
    payload['source']=source_bundle_fingerprint(backend_source_records(str(Path(__file__).resolve().parents[1])))
    payload['entrypoint']=name
    payload['runtime_overrides']={key:value for key,value in os.environ.items()
        if key.startswith('GHOST_') and key not in ('GHOST_TIMING_CACHE_DIR','GHOST_CPU_FACTORIZATION')}
    nearby=_nearby_descriptor(payload,options,arguments)
    stage=dict(nearby,options=dict(options)) if nearby else None
    from ghost_backend.execution.factor_timing import descriptor
    return RequestKey(_digest(payload),nearby,stage,descriptor(payload,options))


def read():
    try:
        path=cache_path()
        if path.stat().st_size>MAX_BYTES:return {}
        result=json.loads(path.read_text(encoding='utf-8'))
        return result if isinstance(result,dict) and len(result)<=MAX_ENTRIES else {}
    except (OSError,ValueError):return {}


def _samples(entry):
    result={};now=time.time()
    if not isinstance(entry,dict):return result
    for mode,rows in entry.items():
        if mode not in ('dense','compressed') or not isinstance(rows,list):continue
        values=[]
        for row in rows[-5:]:
            if not isinstance(row,list) or len(row)!=2:continue
            stamp,seconds=row
            if (isinstance(stamp,(int,float)) and isinstance(seconds,(int,float)) and
                math.isfinite(stamp) and math.isfinite(seconds) and
                0<=now-stamp<=MAX_AGE and 0<seconds<=MAX_AGE):values.append(seconds)
        result[mode]=values
    return result


def measured_costs(key,entries=None):
    entries=read() if entries is None else entries
    return {mode:statistics.median(values) for mode,values in _samples(entries.get(key,{})).items()
            if len(values)>=2}


def _ratio(a,b):
    if (not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or
        not math.isfinite(a) or not math.isfinite(b) or min(a,b)<=0):return math.inf
    return max(a,b)/min(a,b)


def _same_dof_regime(left,right):
    from ghost_backend.linalg.hierarchical import automatic_hierarchical
    # Avoid known BLAS, leaf/tree and automatic factorization transitions.
    return ((int(left)-1).bit_length()==(int(right)-1).bit_length() and
            automatic_hierarchical(left)==automatic_hierarchical(right))


def _nearby_costs(key,selection,entries=None):
    target=getattr(key,'nearby',None)
    meshes=selection.get('meshes',[])
    if not target or not meshes:return {},None
    try:unknowns=max(int(row['unknowns']) for row in meshes)
    except (KeyError,TypeError,ValueError):return {},None
    # One paired workload supplies both timings; never compare unrelated
    # one-backend runs or pool distant requests to manufacture a crossover.
    matches=[]
    entries=read() if entries is None else entries
    for digest,entry in entries.items():
        if not isinstance(entry,dict):continue
        source=entry.get('_nearby')
        if not isinstance(source,dict) or source.get('family')!=target['family']:continue
        frequency_ratio=_ratio(source.get('frequency_ghz'),target['frequency_ghz'])
        angle_ratio=_ratio(source.get('angle_count'),target['angle_count'])
        sizes=source.get('unknowns',{})
        if not isinstance(sizes,dict):continue
        dof_ratio=max(_ratio(sizes.get(mode),unknowns) for mode in ('dense','compressed'))
        if (frequency_ratio>NEARBY_FREQUENCY_RATIO or angle_ratio>NEARBY_ANGLE_RATIO or
            dof_ratio>NEARBY_DOF_RATIO):continue
        if any(not _same_dof_regime(sizes[mode],unknowns) for mode in ('dense','compressed')):continue
        samples=_samples(entry)
        counts=source.get('sample_counts',{})
        if not isinstance(counts,dict):continue
        # The exact digest intentionally ignores RAM reservations. Retain its
        # original history semantics, but only transfer the latest consecutive
        # samples recorded under this stricter nearby family and actual mesh.
        if any(type(counts.get(mode)) is not int or not 3<=counts[mode]<=5
               for mode in ('dense','compressed')):continue
        samples={mode:values[-counts[mode]:] for mode,values in samples.items()}
        if any(len(samples.get(mode,[]))<3 for mode in ('dense','compressed')):continue
        if any(max(values)/min(values)>NEARBY_SPREAD_RATIO for values in samples.values()):continue
        medians={mode:statistics.median(values) for mode,values in samples.items()}
        winner=min(medians,key=medians.get)
        loser='compressed' if winner=='dense' else 'dense'
        upper=max(samples[winner])*(1+NEARBY_UNCERTAINTY)
        lower=min(samples[loser])*(1-NEARBY_UNCERTAINTY)
        if upper>(1-NEARBY_WIN_MARGIN)*lower:continue
        matches.append((frequency_ratio*dof_ratio*angle_ratio,digest,medians,winner,dict(
            source_frequency_ghz=source['frequency_ghz'],source_unknowns=sizes,
            target_unknowns=unknowns,source_angle_count=source['angle_count'],
            samples_per_backend={mode:len(values) for mode,values in samples.items()},
            winner_upper_seconds=upper,other_lower_seconds=lower)))
    if not matches or len({row[3] for row in matches})!=1:return {},None
    _,digest,medians,_,evidence=min(matches,key=lambda row:row[:2])
    return medians,dict(evidence,source_request=digest)


def adjust(selection,key,batch=False,entries=None):
    candidates=selection.get('candidates',{})
    # One decision uses one bounded snapshot, even if another process writes
    # timing evidence while exact, nearby and stage matches are considered.
    entries=read() if entries is None else entries
    measured={m:t for m,t in measured_costs(key,entries).items() if m in candidates}
    nearby=None
    if len(measured)<2:
        measured,nearby=_nearby_costs(key,selection,entries)
        measured={m:t for m,t in measured.items() if m in candidates}
    stages=None
    if len(measured)<2:
        from ghost_backend.execution.stage_timing import predict
        measured,stages=predict(key,selection,entries,MAX_AGE)
    if len(measured)<2:return selection
    ratios=[t/candidates[m]['cost'] for m,t in measured.items()]
    scale=statistics.median(ratios)
    revised={m:dict(c,prior_cost=c['cost'],cost=measured.get(m,c['cost']*scale),
                    timing_evidence=('matched_stage_rates' if stages else 'nearby_stable_median' if nearby else 'measured_median') if m in measured else 'scaled_prior')
             for m,c in candidates.items()}
    from ghost_backend.execution.policy import rank_candidates
    budget=(candidates[selection['selected']]['peak_gb'] if batch else selection['admission_budget_gib'])
    ranked=rank_candidates(revised,budget,margin=0. if batch else .2)
    result=dict(selection,selected=ranked[0],retry_order=ranked[1:],candidates=revised,
                timing_model='matched_stage_rates_v1' if stages else 'nearby_request_host_paired_v1' if nearby else 'identical_request_host_median_v1',measured_backends=sorted(measured),
                prior_selected_backend=selection['selected'],batch_plan_revised=bool(batch and ranked[0]!=selection['selected']),
                reason=('Ranked using matched solver-stage measurements and bounded work scaling; uncertainty and RAM guards retained.' if stages else
                        'Ranked using stable paired timings for a nearby workload on this host; uncertainty and RAM guards retained.'
                        if nearby else 'Ranked using repeated timings for this exact request and execution host; RAM admission retained.'))
    if nearby:result['nearby_timing_evidence']=nearby
    if stages:result['stage_timing_evidence']=stages
    return result


def _clean_success(metadata,mode):
    selection=metadata.get('backend_selection',{})
    if (selection.get('failed_attempts') or selection.get('selected',mode)!=mode or
        metadata.get('dense_fallback_reasons')):return False
    adaptation=metadata.get('adaptive_mesh',{})
    if any(event.get('fallback') for event in adaptation.get('polynomial_pair',[])
           if isinstance(event,dict)):return False
    if (adaptation.get('fallback') or adaptation.get('conservative_retry')
        or any(step.get('failed_backends') or step.get('backend')!=mode
                                        for step in adaptation.get('steps',[]))):return False
    if any(event.get('coarse_rejection') or event.get('compact_preconditioner') or
           event.get('frequency_preconditioner',{}).get('reused')
           for event in metadata.get('compressed_factors',[])):return False
    hierarchical=metadata.get('hierarchical_factors',[])
    if isinstance(hierarchical,dict):hierarchical=[event for rows in hierarchical.values() for event in rows]
    if any(event.get('coarse_rejection') or event.get('builds',1)>1 or event.get('tighter_rebuilds')
           for event in hierarchical):return False
    if any(event.get('factor_work_failed') or event.get('factor_fallback') or event.get('factor_rebuilds')
           for event in metadata.get('experimental_cpu',{}).get('systems',[])
           if isinstance(event,dict)):return False
    children=list(metadata.get('channel_metadata',{}).values())
    children.extend(row.get('metadata',{}) for row in metadata.get('frequency_metadata',[]))
    return all(_clean_success(child,mode) for child in children)


def record(key,mode,seconds,metadata):
    if (key is None or mode not in ('dense','compressed') or not math.isfinite(seconds) or seconds<=0 or
        seconds>MAX_AGE or not metadata.get('quality_gate',{}).get('passed') or
        not _clean_success(metadata,mode)):return
    path=cache_path();temporary=None
    try:
        entries=read();entry=entries.setdefault(key,{})
        if not isinstance(entry,dict):entry={};entries[key]=entry
        descriptor=getattr(key,'nearby',None)
        unknowns=metadata.get('dense_largest_system',0)
        if descriptor and isinstance(unknowns,(int,float)) and math.isfinite(unknowns) and unknowns>0:
            previous=entry.get('_nearby',{})
            compatible=isinstance(previous,dict) and previous.get('family')==descriptor['family']
            sizes=dict(previous.get('unknowns',{})) if compatible else {}
            counts=dict(previous.get('sample_counts',{})) if compatible else {}
            if sizes.get(mode)!=int(unknowns):counts[mode]=0
            sizes[mode]=int(unknowns)
            counts[mode]=min(5,int(counts.get(mode,0))+1)
            entry['_nearby']=dict(descriptor,unknowns=sizes,sample_counts=counts)
        rows=entry.get(mode,[])
        entry[mode]=(rows[-4:] if isinstance(rows,list) else [])+[[time.time(),seconds]]
        if mode=='dense':
            from ghost_backend.execution.factor_timing import observation as factor_observation
            factor_sample=factor_observation(key,metadata)
            if factor_sample is not None:
                previous=entry.get('_factorizations',[])
                entry['_factorizations']=(previous[-4:] if isinstance(previous,list) else [])+[factor_sample]
        from ghost_backend.execution.stage_timing import observation
        sample=observation(key,mode,seconds,metadata)
        if sample is not None:
            stage_rows=entry.setdefault('_stages',{})
            if not isinstance(stage_rows,dict):stage_rows={};entry['_stages']=stage_rows
            previous=stage_rows.get(mode,[])
            stage_rows[mode]=(previous[-4:] if isinstance(previous,list) else [])+[sample]
        # Concurrent writers may lose a timing sample, never a solver result.
        while len(entries)>MAX_ENTRIES:entries.pop(next(iter(entries)))
        path.parent.mkdir(parents=True,exist_ok=True)
        # Windows tempfile can retry PermissionError for an effectively
        # unwritable directory billions of times. This optional cache gets one
        # exclusive creation attempt and never holds up a completed solve.
        candidate=path.with_name('.timings-'+uuid.uuid4().hex+'.tmp')
        with open(candidate,'x',encoding='utf-8') as stream:
            temporary=candidate
            json.dump(entries,stream,allow_nan=False)
        os.replace(temporary,path)
    except (OSError,ValueError):pass
    finally:
        if temporary is not None:
            try:unlink_if_exists(temporary)
            except OSError:pass
