"""Shared CPU backend relative runtime prior.

The prior ranks work; it is not a wall-time or optimality guarantee. It retains
the measured dense preference on small/medium systems. Both desktop and
execution-node scheduling use it.
"""
import math

MODEL = 'stage_work_v5'
BACKENDS = ('dense', 'compressed')


def work_threads(dofs, options):
    """The assembly and matrix-size BLAS teams used by the configured solver."""
    from ghost_backend.execution.options import effective_assembly_threads, allocated_cpu_budget, blas_core_budget
    setting=options.get('blas_threads',1)
    desired=(1 if dofs<=256 else 4 if dofs<=1024 else 8 if dofs<=2048 else blas_core_budget()) if setting=='auto' else int(setting)
    if setting=='auto':desired=min(desired,blas_core_budget())
    return effective_assembly_threads(),min(max(1,desired),allocated_cpu_budget())


def stage_work(resources, n_angles, mode, assembly_threads=1, blas_threads=1):
    """Nonnegative work proxies, never measured seconds or a rank prediction.

    Near corrections and far coefficients stay separate so calibrated assembly
    can bound their contributions without fitting an underdetermined mixture.
    The compressed inverse has no promised rank advantage in the prior.
    """
    from ghost_backend.linalg.hierarchical import automatic_hierarchical, HIERARCHICAL_MIN_UNKNOWNS
    import os
    if mode not in BACKENDS:raise ValueError('Unknown automatic backend.')
    n=max(1,int(resources['nodes'])); d=max(1,int(resources['system_dofs']))
    angles=max(1,int(n_angles)); kernels=max(1.,resources.get('operator_matrices',3)/3.)
    width=max(2,int(resources.get('basis_width',2)))
    far=max(1.,float(resources.get('operator_entries',n*n*kernels)))
    near=max(0.,float(resources.get('geometric_near_pairs',0)))*width*width*kernels
    assembly_threads=max(1,int(assembly_threads));blas_threads=max(1,int(blas_threads))
    threshold=int(os.environ.get('GHOST_HIERARCHICAL_MIN_UNKNOWNS','') or HIERARCHICAL_MIN_UNKNOWNS)
    hierarchical=automatic_hierarchical(d)
    # Continuous at the actual LU/HODLR switch. Unknown rank does not promise
    # a compressed speedup; local measured stages supersede the work prior.
    factor=(min(d,max(1,threshold))*d*d*max(1.,math.log2(d)/math.log2(max(2,threshold)))
            if hierarchical else d**3)
    return dict(assembly_far=far/assembly_threads,assembly_near=near/assembly_threads,
        compression=(d*d*min(d,512)/assembly_threads if mode=='compressed' else 0.),
        factorization=factor/blas_threads,rhs=d*d*angles/blas_threads,other=1.,
        factor_regime='compressed_hodlr' if mode=='compressed' else 'hodlr' if hierarchical else 'lu',
        assembly_threads=assembly_threads,blas_threads=blas_threads)


def relative_cost(resources, n_angles, mode, assembly_threads=1, blas_threads=1):
    """Conservative stage prior until both backends have usable measurements."""
    if resources.get('analytic_zero'):
        return .001
    work=stage_work(resources,n_angles,mode,assembly_threads,blas_threads)
    return (.025+7e-7*(work['assembly_far']+work['assembly_near'])+
            4e-12*(work['factorization']+work['compression'])+2e-10*work['rhs'])


def workload(meshes, n_angles, mode, options):
    """Small aggregate of the actual planned systems, including their regimes."""
    from ghost_backend.execution.options import execution_scope
    with execution_scope(options):
        return _workload(meshes,n_angles,mode,options)


def _workload(meshes, n_angles, mode, options):
    totals={key:0. for key in ('assembly_far','assembly_near','compression','factorization','rhs','other')}
    systems=[]
    for row in meshes:
        resources=row['resources'];d=int(resources['system_dofs'])
        if resources.get('analytic_zero') or d<=0:return None
        assembly,blas=work_threads(d,options)
        work=stage_work(resources,n_angles,mode,assembly,blas)
        for key in totals:totals[key]+=work[key]
        signature=(row['formulation'],int(row['polynomial_degree']),row['polarization'],
                   work['factor_regime'],assembly,blas)
        systems.append((signature,d))
    if not systems:return None
    systems.sort()
    return dict(work=totals,signature=[list(signature) for signature,d in systems],
                unknowns=[d for signature,d in systems],angles=int(n_angles))


def available_candidates(candidates):
    """Copy candidates for ranking; every backend runs on any CPU host."""
    return {mode:dict(c) for mode,c in candidates.items()}


def rank_candidates(candidates, budget_gib, margin=.2):
    available=available_candidates(candidates)
    if not available:
        raise RuntimeError('No compatible solver backend is available.')
    for mode,c in available.items():
        if mode not in BACKENDS or any(not math.isfinite(float(c[k])) or c[k] <= 0 for k in ('cost','peak_gb')):
            raise ValueError('Invalid automatic backend forecast.')
    fitting=[m for m,c in available.items() if c['peak_gb'] <= (1-margin)*budget_gib]
    if not fitting:
        fitting=[m for m,c in available.items() if c['peak_gb'] <= budget_gib]
    if not fitting:
        description=', '.join('{} {:.2f} GiB'.format(m,c['peak_gb']) for m,c in available.items())
        raise MemoryError('No compatible backend fits the {:.2f} GiB solve budget ({}).'.format(budget_gib,description))
    return sorted(fitting,key=lambda m:(available[m]['cost'],BACKENDS.index(m)))
