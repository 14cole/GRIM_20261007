"""Explicit compressed CPU path: bounded geometry assembly and strict rejection."""
from ghost_backend.execution.options import temporary_directory
from ghost_backend.execution.options import environment_value
import os,tempfile,time
import numpy as np
from ghost_backend.execution.metrics import timed_stage


def enabled():
    return environment_value('GHOST_CPU_FACTORIZATION','dense').strip().lower()=='compressed'


# Automatic storage keeps both polarizations' operators and the inverse within this
# share of the solve memory limit; the rest covers assembly and solve workspaces.
AUTOMATIC_STORAGE_FRACTION=.6
# Match the minimum explicit cap; a 2-GiB floor could exceed a small job's
# entire RAM allocation before assembly workspaces were counted.
AUTOMATIC_STORAGE_FLOOR=16*1024**2


def automatic_storage():
    return environment_value('GHOST_COMPRESSED_STORAGE_MIB','2048').strip()=='0'


def automatic_storage_bytes():
    """Bytes the automatic setting grants, independent of any environment variable.

    BOR carries its own execution options rather than the 2-D profile, so it
    needs the sizing rule without the GHOST_COMPRESSED_STORAGE_MIB lookup.
    """
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    return max(AUTOMATIC_STORAGE_FLOOR,int(AUTOMATIC_STORAGE_FRACTION*_solve_memory_limit_gb()*1024**3))


def storage_budget():
    text=environment_value('GHOST_COMPRESSED_STORAGE_MIB','2048').strip()
    try:value=int(text)
    except ValueError:raise ValueError('GHOST_COMPRESSED_STORAGE_MIB must be a positive integer, or 0 for automatic.')
    if value==0:
        return automatic_storage_bytes()
    if value<16:raise ValueError('GHOST_COMPRESSED_STORAGE_MIB must be at least 16 MiB.')
    return value*1024**2


def checkpoint():
    from ghost_backend.execution.cpu import current_state
    from ghost_backend.twod.assembly.session import current_session
    owner=current_state() or current_session()
    if owner is not None:owner.checkpoint()


def coordinates(mesh,n):
    xy=np.zeros((len(mesh.nodes),2))
    for e in mesh.elements:xy[list(e.node_ids)]=[mesh.nodes[i].xy for i in e.node_ids]
    return np.tile(xy,(n//len(mesh.nodes),1))


@timed_stage('compressed_assembly')
def build(oracle,xy):
    from ghost_backend.compressed.operator import StreamedOperator
    operator=StreamedOperator(oracle,xy,tile=min(512,getattr(oracle,'maximum_tile',512)),
        budget=storage_budget(),checkpoint=checkpoint)
    if hasattr(oracle,'cached'):oracle.cached=oracle.cached_columns=None
    return operator


def native(mesh,infos,pol,k0,kind,obs_order=8,src_order=8,layer=None):
    from ghost_backend.compressed.coefficients import NativeOracle, PairedNativeOracle
    from ghost_backend.compressed.polarization_cache import build_pair
    from ghost_backend.twod.assembly.session import current_session, system_key
    oracle=NativeOracle(mesh,infos,pol,k0,kind,obs_order,src_order,layer)
    from ghost_backend.compressed.pilots import attach
    attach(oracle,mesh,infos,pol,kind,k0,layer,obs_order,src_order)
    session=current_session()
    key=(system_key(mesh,infos or [],'compressed_'+kind,obs_order,src_order),layer)
    previous=session.take(key,pol) if session is not None else None
    if previous is not None:
        previous.load()
        return previous,oracle
    partner=getattr(session,'compressed_partner',None)
    if pol=='TE' and partner is not None and (partner[0] is mesh or kind=='thin'):
        other=NativeOracle(mesh,None if kind=='thin' else partner[1],'TM',k0,kind,obs_order,src_order,layer)
        attach(other,mesh,None if kind=='thin' else partner[1],'TM',kind,k0,layer,obs_order,src_order)
        if other.n==oracle.n:
            pair=build_pair(PairedNativeOracle(oracle,other),coordinates(mesh,oracle.n),
                tile=min(512,getattr(oracle,'maximum_tile',512)),budget=storage_budget(),
                checkpoint=checkpoint,spool_directory=temporary_directory())
            pair[0].reserved_partner_bytes=pair[1].bytes
            key=(system_key(mesh,partner[1] if kind!='thin' else [],'compressed_'+kind,obs_order,src_order),layer)
            session.save(key,'TE',pair[1]);session.compressed_partner=None
            oracle.cached=oracle.cached_columns=other.cached=other.cached_columns=None
            return pair[0],oracle
    operator=build(oracle,coordinates(mesh,oracle.n))
    return operator,oracle


@timed_stage('compressed_assembly')
def regional(mesh,infos,pol,obs_order=8,src_order=8):
    shared=_projected_regional(mesh,infos,pol,obs_order,src_order)
    return shared if shared is not None else _regional(mesh,infos,pol,obs_order,src_order)


def _projected_regional(mesh,infos,pol,obs_order,src_order):
    from ghost_backend.twod.assembly import polynomial_pair as pair
    owner=pair.current_pair()
    if owner is None or owner.building:return None
    previous=pair.take(mesh,infos,pol,'compressed',obs_order,src_order)
    if previous is not None:
        from ghost_backend.compressed.retained_storage import RetainedOperatorSpool
        operator,layout=previous
        if isinstance(operator,RetainedOperatorSpool):
            try:return operator.restore(),layout
            except InterruptedError:
                pair.discard_backend('compressed')
                raise
            except OSError as exc:
                pair.discard_backend('compressed')
                owner.evidence.append(dict(backend='compressed',polarization=pol,
                    action='independent_assembly',fallback='joint_storage_rejection',
                    reason='retained cubic storage unavailable: '+str(exc)))
                return None
        return previous
    from ghost_backend.twod.assembly.session import current_session,system_key
    session=current_session()
    partner=getattr(session,'compressed_partner',None)
    # A paired traversal is already the baseline. Keep it intact: project both
    # polarizations or leave the ordinary assembly path entirely unchanged.
    if pol!='TE' or partner is None or partner[0] is not mesh:return None
    plans=[pair.prepare(mesh,values,label,obs_order,src_order)
           for values,label in ((infos,'TE'),(partner[1],'TM'))]
    if any(plan is None for plan in plans):return None
    import ghost_backend.twod.solver as solver
    from ghost_backend.compressed.memory import inverse_storage
    from ghost_backend.twod.formulations.regions import dof_coordinates
    fine=plans[0]['fine_mesh'];records=[]
    for candidate in (mesh,fine):
        phase=[]
        for values,label in ((infos,'TE'),(partner[1],'TM')):
            resources=solver._dense_formulation_resources(candidate,values,label)
            peak=solver._estimate_memory_gb(resources['nodes'],False,
                system_dofs=resources['system_dofs'],n_regions=resources['n_regions'],
                operator_matrices=resources['operator_matrices'],n_rhs=owner.n_rhs,
                solver_method=owner.solver_method,formulation='multi_region',dense_resources=resources)
            phase.append((peak*1024**3,resources['memory_estimate']['operator_allowance_bytes']))
        records.append(phase)
    coarse_allowance=sum(row[1] for row in records[0]);fine_allowance=sum(row[1] for row in records[1])
    inverse=max(inverse_storage(plan['coarse_layout']['n_dof'])[0] for plan in plans)
    from ghost_backend.compressed.projection import projection_workspace_bytes
    projection_workspace=projection_workspace_bytes()
    required=max(max(row[0] for row in records[0])+fine_allowance,
                 max(row[0] for row in records[1])+fine_allowance+coarse_allowance+projection_workspace)
    limit=solver._solve_memory_limit_gb()*1024**3;storage=storage_budget()
    if required>limit or fine_allowance+coarse_allowance+inverse>storage:
        owner.evidence.append(dict(backend='compressed',action='independent_assembly',
            reason='joint polynomial storage exceeds reservation',required_gib=required/1024**3,
            budget_gib=limit/1024**3,retained_storage_required=fine_allowance+coarse_allowance+inverse,
            retained_storage_budget=storage))
        return None
    from ghost_backend.compressed.projection import project_operator
    from ghost_backend.compressed.retained_storage import RetainedOperatorSpool
    built=[];owner.building=True;started=time.perf_counter()
    try:
        session.compressed_partner=(fine,partner[1])
        fine_te,layout_te=_regional(fine,infos,'TE',obs_order,src_order)
        built.append(fine_te)
        fine_tm,layout_tm=session.take(system_key(fine,partner[1],'compressed_region',obs_order,src_order),'TM')
        built.append(fine_tm)
        fine_te.reserved_partner_bytes=0
        retained=sum(operator.bytes for operator in built)
        if storage-retained-inverse<=0:
            raise MemoryError('Retained cubic operators leave no admitted quadratic projection storage.')
        coarse_te=project_operator(fine_te,plans[0]['prolongation'],
            dof_coordinates(mesh,plans[0]['coarse_layout']),storage-retained-inverse,checkpoint)
        built.append(coarse_te)
        retained_te=RetainedOperatorSpool(fine_te,temporary_directory());built.append(retained_te)
        if storage-retained-coarse_te.bytes-inverse<=0:
            raise MemoryError('Retained polynomial operators leave no admitted partner projection storage.')
        # Keep the existing fine TM spool on disk while projecting TE. Only
        # one fine polarization needs resident tiles at a time.
        fine_tm.load()
        coarse_tm=project_operator(fine_tm,plans[1]['prolongation'],
            dof_coordinates(mesh,plans[1]['coarse_layout']),storage-retained-coarse_te.bytes-inverse,
            checkpoint,spool_directory=temporary_directory())
        built.append(coarse_tm)
        retained_tm=RetainedOperatorSpool(fine_tm,temporary_directory());built.append(retained_tm)
        coarse_te.reserved_partner_bytes=coarse_tm.bytes
        pair.remember(fine,infos,'TE',retained_te,layout_te,'compressed',obs_order,src_order)
        pair.remember(fine,partner[1],'TM',retained_tm,layout_tm,'compressed',obs_order,src_order)
        session.save(system_key(mesh,partner[1],'compressed_region',obs_order,src_order),'TE',
                     (coarse_tm,plans[1]['coarse_layout']))
        session.compressed_partner=None
        owner.evidence.append(dict(backend='compressed',action='project_quadratic',
            fine_dofs=len(fine_te),coarse_dofs=len(coarse_te),retained_fine_bytes=0,
            retained_fine_disk_bytes=retained_te.disk_bytes+retained_tm.disk_bytes,
            projection_workspace_bytes=projection_workspace,
            required_gib=required/1024**3,budget_gib=limit/1024**3,
            propagated_fine_errors=True))
        return coarse_te,plans[0]['coarse_layout']
    except InterruptedError:
        # Cancellation is an OSError subclass, but must never start a rebuild.
        session.pending=None;session.compressed_partner=partner
        for operator in built:
            if hasattr(operator,'close'):operator.close()
        pair.discard_backend('compressed')
        raise
    except (MemoryError,OSError) as exc:
        session.pending=None;session.compressed_partner=partner
        for operator in built:
            if hasattr(operator,'close'):operator.close()
        pair.discard_backend('compressed')
        owner.evidence.append(dict(backend='compressed',action='independent_assembly',
            reason=str(exc),fallback='joint_storage_rejection',spent_seconds=time.perf_counter()-started))
        # Return before ordinary assembly so the rejected build and traceback
        # no longer retain its operators while the original path starts.
        return None
    except BaseException:
        session.pending=None;session.compressed_partner=partner
        for operator in built:
            if hasattr(operator,'close'):operator.close()
        raise
    finally:
        owner.building=False


def _regional(mesh,infos,pol,obs_order=8,src_order=8):
    import ghost_backend.twod.formulations.regions as mr
    from ghost_backend.compressed.regional_coefficients import PreparedOracle, PairedOracle
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.polarization_cache import build_pair, SpooledOperator
    from ghost_backend.twod.assembly.session import current_session, system_key
    from ghost_backend.compressed.pilots import attach
    session=current_session();key=system_key(mesh,infos,'compressed_region',obs_order,src_order)
    previous=session.take(key,pol) if session is not None else None
    if previous is not None:
        operator,layout=previous
        operator.load()
        return operator,layout
    partner=getattr(session,'compressed_partner',None)
    if pol=='TE' and partner is not None and partner[0] is mesh:
        oracle=PairedOracle(mesh,infos,partner[1],cut=32,obs_order=obs_order,src_order=src_order)
        for source,values,label in zip(oracle.oracles,(infos,partner[1]),('TE','TM')):
            attach(source,mesh,values,label,'multi_region',obs_order=obs_order,src_order=src_order)
        xy=mr.dof_coordinates(mesh,oracle.oracles[0].layout)
        pair=build_pair(oracle,xy,tile=512,budget=storage_budget(),checkpoint=checkpoint,spool_directory=temporary_directory())
        pair[0].reserved_partner_bytes=pair[1].bytes
        key=system_key(mesh,partner[1],'compressed_region',obs_order,src_order)
        session.save(key,'TE',(pair[1],oracle.oracles[1].layout))
        session.compressed_partner=None
        return pair[0],oracle.oracles[0].layout
    oracle=PreparedOracle(mesh,infos,pol,cut=32,obs_order=obs_order,src_order=src_order)
    attach(oracle,mesh,infos,pol,'multi_region',obs_order=obs_order,src_order=src_order)
    xy=mr.dof_coordinates(mesh,oracle.layout)
    operator=StreamedOperator(oracle,xy,tile=512,budget=storage_budget(),checkpoint=checkpoint)
    return operator,oracle.layout


def thin(mesh,k0,pol,eps,mu,d,order=8):
    from ghost_backend.twod.assembly.kernels import incident_loads
    operator,oracle=native(mesh,None,pol,k0,'thin',order,order,(eps,mu,d))
    coefficient,B=oracle.coefficient,oracle.B
    if B==0:
        def rhs(batch):
            bu,_=incident_loads(mesh,k0,batch,want_dn=False)
            return -coefficient*bu
    else:
        C,mass_lu,endpoints,n=oracle.C,oracle.mass_lu,oracle.endpoints,oracle.nn
        def rhs(batch):
            bu,bq=incident_loads(mesh,k0,batch)
            result=np.empty((2*n,len(batch)),complex)
            result[:n]=-C@mass_lu.solve(bu);result[n:]=-B*bq;result[n+endpoints]=0
            return result
    return operator,rhs
