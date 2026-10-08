"""Checked inverse of a compressed operator, including adjoint evidence."""
import numpy as np
from scipy.sparse.linalg import LinearOperator,onenormest
from ghost_backend.compressed.inverse import CompressedSystem
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from ghost_backend.execution.metrics import timed_stage

# Solves that produce fields must meet this original-coefficient backward error.
SOLVE_BACKWARD_ERROR_LIMIT=1e-12
# Condition probes only locate ||A^-1|| for a gate at 1e6. A backward error eps
# perturbs that norm by about kappa*eps relative, so 1e-9 moves any estimate
# the gate can pass by at most ~1e-3; larger estimates are not accepted at it.
CONDITION_PROBE_BACKWARD_ERROR_LIMIT=1e-9
CONDITION_PROBE_ACCEPTED_PERTURBATION=1e-2
# Stalled refinements finish with GMRES on up to GMRES_BATCH columns at once, so
# each iteration shares one multi-column operator product and preconditioner solve.
GMRES_BATCH=16
GMRES_RESTART=20
GMRES_ITERATION_CAP=120
# Refinement and GMRES both stop at this normwise backward error against the tile operator.
REFINEMENT_BACKWARD_ERROR=3e-15
# A 1e-8 preconditioner refines in one or two steps even at 18 GHz, where 1e-6
# needs nine or more, for about 30% more storage; 1e-6 is kept for when it does not fit.
PRECONDITIONER_TOLERANCE=1e-8
COMPACT_PRECONDITIONER_TOLERANCE=1e-6


class CompressedFactor:
    def __init__(self,operator,diagnostics=None,label='compressed system',evidence=None,checkpoint=None,
                 storage_budget_bytes=None, check_precision=True, recycling_key=None,recycling_frequency=None,
                 recycling_coordinate_units='rad/m', **kwargs):
        from ghost_backend.compressed.runtime import storage_budget
        from ghost_backend.linalg.refined_lu import requested_precision
        if check_precision and requested_precision()!='double':raise ValueError('Compressed factorization requires double precision.')
        self.a=operator;self.diagnostics=diagnostics;self.label=label
        self.checkpoint=checkpoint or operator.checkpoint
        self.matrix_inf=max(float(np.max(operator.row_norm-operator.row_error)),0.)
        self.relative_residual=np.empty(0)
        self.factor=None;self.reported=0
        self.recycling_key=recycling_key;self.frequency=recycling_frequency
        self.recycling_coordinate_units=recycling_coordinate_units
        self.original_frequency=recycling_frequency;self.recycled=False
        allowance = storage_budget() if storage_budget_bytes is None else int(storage_budget_bytes)
        self.budget=allowance-operator.bytes-getattr(operator,'reserved_partner_bytes',0)
        from ghost_backend.twod.assembly.polynomial_pair import retained_bytes as polynomial_retained_bytes
        self.budget-=polynomial_retained_bytes('compressed')
        from ghost_backend.compressed.recycling import take,live_bytes,reserve_current_inverse
        borrowed=take(recycling_key,recycling_frequency,self.budget,self.checkpoint) if recycling_key is not None else None
        from ghost_backend.compressed.memory import inverse_storage
        needed=borrowed[0].bytes if borrowed is not None else inverse_storage(operator.n)[0]
        reserve_current_inverse(needed,self.budget)
        self.budget-=live_bytes()
        if self.budget<=0:raise MemoryError('No compressed inverse storage remains.')
        self.event=dict(unknowns=operator.n,factorizations=0,rhs_batches=0,max_rhs_columns=0,
            max_backward_error=0.,max_relative_residual=0.,compressed=operator.evidence,
            preconditioners=[],gmres_columns=0,max_refinements=0,refinement_steps=0)
        if evidence is not None:evidence.append(self.event)
        if borrowed is not None and borrowed[0].bytes<=self.budget and borrowed[0].n==operator.n:
            self.factor,self.original_frequency=borrowed
            self.tolerance=self.factor.tolerance;self.recycled=True
            self.event['frequency_preconditioner']=dict(self._frequency_evidence(),reused=True)
            self.event['preconditioners'].append(self.factor.evidence)
        else:
            borrowed=None
            self._fresh_build()
        borrowed=None
        if diagnostics is not None:self._condition()

    def _frequency_evidence(self):
        event=dict(original_frequency_coordinate=self.original_frequency,
                   current_frequency_coordinate=self.frequency,
                   coordinate_units=self.recycling_coordinate_units)
        if self.recycling_coordinate_units=='rad/m':
            # Preserve the existing 2D field; BoR's frequency is measured in Hz.
            event['original_wavenumber']=self.original_frequency
        elif self.recycling_coordinate_units=='Hz':
            event['original_frequency_hz']=self.original_frequency
        return event

    def _fresh_build(self):
        rejected=False
        try:
            compact=False
            try:self._build(PRECONDITIONER_TOLERANCE)
            except MemoryError as exc:
                self.event['compact_preconditioner']=str(exc)
                compact=True
            # Leave the handler before rebuilding: its traceback otherwise
            # retains the failed tree and its arrays throughout the retry.
            if compact:self._build(COMPACT_PRECONDITIONER_TOLERANCE)
        except (HierarchicalRejected,np.linalg.LinAlgError,RuntimeWarning) as exc:
            self.event['coarse_rejection']=str(exc);rejected=True
        if rejected:self._build(2e-10)

    @timed_stage('factorization')
    def _build(self,tolerance):
        self.factor=None
        self.recycled=False;self.original_frequency=self.frequency
        self.checkpoint()
        self.factor=CompressedSystem(self.a,self.a.coordinates,tolerance=tolerance,
            budget=self.budget,checkpoint=self.checkpoint,inverse_only=True)
        self.tolerance=tolerance;self.event['factorizations']+=1
        self.event['preconditioners'].append(self.factor.evidence)

    def physical_errors(self,x,b,residual=None,trans=0):
        if residual is None:residual=self.a.matmul(x,trans)-b
        rows=self.a.row_error if trans==0 else self.a.column_error
        norm=self.a.row_norm if trans==0 else self.a.column_norm
        lower=max(float(np.max(norm-rows)),0.)
        magnitude=np.max(abs(x),axis=0)
        numerator=np.max(abs(residual)+rows[:,None]*magnitude[None,:],axis=0)
        den=np.maximum(lower*magnitude+np.max(abs(b),axis=0),1e-300)
        return numerator/den

    def relative_errors(self,x,b,residual):
        error=abs(residual)+self.a.row_error[:,None]*np.max(abs(x),axis=0)[None,:]
        norm=np.linalg.norm(b,axis=0)
        from ghost_backend.twod.constants import EPS
        return np.linalg.norm(error,axis=0)/np.where(norm<=EPS,1.,norm)

    def _backward_errors(self,x,b,residual,trans):
        norm=float(np.max(self.a.row_norm if trans==0 else self.a.column_norm))
        denominator=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
        return np.max(abs(residual),axis=0)/denominator,denominator

    def _refine(self,b,trans):
        x=self.factor.apply(b,solve=True,trans=trans)
        previous=np.inf
        for step in range(10):
            self.checkpoint();residual=b-self.a.matmul(x,trans)
            errors,_=self._backward_errors(x,b,residual,trans)
            bad=~np.isfinite(errors)|(errors>REFINEMENT_BACKWARD_ERROR)
            if not np.any(bad):
                self.event['max_refinements']=max(self.event['max_refinements'],step)
                return x,-residual
            worst=float(np.max(errors))
            if step==(2 if self.recycled else 9) or not np.isfinite(worst) or step>1 and worst>previous*1.2:break
            previous=worst;x[:,bad]+=self.factor.apply(residual[:,bad],solve=True,trans=trans)
            self.event['refinement_steps']+=1
        self.event['gmres_columns']+=int(np.sum(bad))
        columns=np.flatnonzero(bad)
        for start in range(0,len(columns),GMRES_BATCH):
            chunk=columns[start:start+GMRES_BATCH]
            x[:,chunk]=self._gmres(b[:,chunk],x[:,chunk],trans)
        return x,self.a.matmul(x,trans)-b

    def _gmres(self,b,x,trans):
        """Right-preconditioned restarted GMRES, independent per column, batched products.

        Columns stop at the refinement backward error; the batch shares
        GMRES_ITERATION_CAP, and a restart cycle that does not halve the worst error
        rejects the preconditioner. Callers still check the physical backward error.
        """
        n,count=b.shape
        iteration_cap=min(8,GMRES_ITERATION_CAP) if self.recycled else GMRES_ITERATION_CAP
        x=np.array(x,complex,copy=True);iterations=0;previous=np.inf
        basis_storage=work_storage=None
        while True:
            self.checkpoint()
            residual=b-self.a.matmul(x,trans)
            errors,denominator=self._backward_errors(x,b,residual,trans)
            active=np.flatnonzero(~np.isfinite(errors)|(errors>REFINEMENT_BACKWARD_ERROR))
            if not len(active):return x
            worst=float(np.max(errors))
            if iterations>=iteration_cap or not np.isfinite(worst) or worst>previous/2:
                raise HierarchicalRejected('Compressed GMRES did not converge within its cap.')
            previous=worst
            beta=np.linalg.norm(residual[:,active],axis=0)
            c,m=len(active),min(GMRES_RESTART,iteration_cap-iterations)
            if basis_storage is None:
                # Each iteration's n-by-c RHS is contiguous. Reuse this
                # allocation across restarts, including shrinking active sets.
                basis_storage=np.empty((n,count,min(GMRES_RESTART,iteration_cap)+1),complex,order='F')
                work_storage=np.empty((n,count),complex,order='F')
            basis=basis_storage[:,:c,:m+1];work=work_storage[:,:c]
            basis[:,:,0]=residual[:,active]/beta
            hessenberg=np.zeros((c,m+1,m),complex)
            cosines=np.zeros((c,m),complex);sines=np.zeros((c,m),complex)
            rhs=np.zeros((c,m+1),complex);rhs[:,0]=beta
            steps=np.zeros(c,dtype=int)
            finished=np.zeros(c,dtype=bool)
            for j in range(m):
                self.checkpoint()
                w=self.a.matmul(self.factor.apply(basis[:,:,j],solve=True,trans=trans),trans)
                for _ in range(2):
                    # Conjugate the current vector, not the growing Krylov
                    # basis: <V,w> = conjugate(V.T @ conjugate(w)).
                    np.conjugate(w,out=work)
                    h=np.einsum('ncj,nc->cj',basis[:,:,:j+1],work).conj()
                    np.einsum('ncj,cj->nc',basis[:,:,:j+1],h,out=work)
                    w-=work
                    hessenberg[:,:j+1,j]+=h
                size=np.linalg.norm(w,axis=0)
                breakdown=size<=1e-14*np.linalg.norm(hessenberg[:,:j+1,j],axis=1)
                hessenberg[:,j+1,j]=size
                basis[:,:,j+1]=w/np.where(size>0,size,1.)
                for i in range(j):
                    top=cosines[:,i]*hessenberg[:,i,j]+sines[:,i]*hessenberg[:,i+1,j]
                    hessenberg[:,i+1,j]=-sines[:,i].conj()*hessenberg[:,i,j]+cosines[:,i].conj()*hessenberg[:,i+1,j]
                    hessenberg[:,i,j]=top
                first,second=hessenberg[:,j,j],hessenberg[:,j+1,j]
                length=np.sqrt(abs(first)**2+abs(second)**2);length=np.where(length>0,length,1.)
                cosines[:,j],sines[:,j]=first.conj()/length,second.conj()/length
                hessenberg[:,j,j],hessenberg[:,j+1,j]=length,0
                rhs[:,j+1]=-sines[:,j].conj()*rhs[:,j]
                rhs[:,j]=cosines[:,j]*rhs[:,j]
                iterations+=1
                # Each column owns its Krylov dimension. A happy breakdown
                # in an easy column must not truncate a harder column's cycle.
                stopped=breakdown|(abs(rhs[:,j+1])<=REFINEMENT_BACKWARD_ERROR*denominator[active]/4)
                newly_finished=stopped&~finished
                steps[newly_finished]=j+1
                finished|=stopped
                basis[:,finished,j+1]=0.
                if np.all(finished):break
            steps[~finished]=j+1
            y=np.zeros((c,j+1),complex)
            for count_steps in np.unique(steps):
                columns=np.flatnonzero(steps==count_steps)
                y[columns,:count_steps]=np.linalg.solve(hessenberg[columns,:count_steps,:count_steps],
                                                       rhs[columns,:count_steps,None])[...,0]
            np.einsum('ncj,cj->nc',basis[:,:,:j+1],y,out=work)
            x[:,active]+=self.factor.apply(work,solve=True,trans=trans)

    def inverse(self,rhs,trans=0,return_residual=False,limit=SOLVE_BACKWARD_ERROR_LIMIT):
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode.')
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=len(self.a) or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid compressed RHS.')
        failed=False
        reused=self.recycled
        try:x,residual=self._refine(b,trans)
        except (HierarchicalRejected,np.linalg.LinAlgError,RuntimeWarning) as exc:
            if self.tolerance<=2e-10 and not reused:raise
            if reused:self.event['frequency_preconditioner']['rejected']=str(exc)
            else:self.event['coarse_rejection']=str(exc)
            failed=True
        if failed:
            if reused:self._fresh_build()
            else:self._build(2e-10)
            x,residual=self._refine(b,trans)
        errors=self.physical_errors(x,b,residual,trans)
        if not np.all(np.isfinite(errors)) or np.max(errors)>limit:
            raise HierarchicalRejected('Compressed inverse failed the original-coefficient error bound.')
        if limit<=SOLVE_BACKWARD_ERROR_LIMIT:
            self.event['max_backward_error']=max(self.event['max_backward_error'],float(np.max(errors)))
        else:
            self.event['max_probe_backward_error']=max(self.event.get('max_probe_backward_error',0.),float(np.max(errors)))
        solution=x[:,0] if vector else x
        if return_residual:return solution,residual[:,0] if vector else residual
        return solution

    def retain_preconditioner(self):
        """Transfer the inverse only after the complete field solve succeeded."""
        if self.recycling_key is None or self.factor is None:return
        from ghost_backend.compressed.recycling import save,capacity_bytes
        capacity=capacity_bytes()
        if not capacity:return
        event=self.event.setdefault('frequency_preconditioner',self._frequency_evidence())
        event.update(cached=False,cache_capacity_bytes=capacity)
        if save(self.recycling_key,self.original_frequency,self.factor):
            event['cached']=True
            event['cached_frequency_coordinate']=self.original_frequency
            self.factor=None

    @timed_stage('condition_estimation')
    def _condition(self):
        rows,columns,norm=self.a.equilibrate()
        probe=CONDITION_PROBE_BACKWARD_ERROR_LIMIT
        # onenormest applies both probe columns at once; one checked solve serves them.
        inverse=LinearOperator(self.a.shape,
            matvec=lambda z:columns*self.inverse(rows*np.asarray(z).reshape(-1),limit=probe),
            rmatvec=lambda z:rows*self.inverse(columns*np.asarray(z).reshape(-1),trans=2,limit=probe),
            matmat=lambda z:columns[:,None]*self.inverse(rows[:,None]*np.asarray(z),limit=probe),
            rmatmat=lambda z:rows[:,None]*self.inverse(columns[:,None]*np.asarray(z),trans=2,limit=probe),dtype=complex)
        estimate=norm*float(onenormest(inverse))
        if not np.isfinite(estimate):raise HierarchicalRejected('Nonfinite compressed condition estimate.')
        if estimate*probe>CONDITION_PROBE_ACCEPTED_PERTURBATION and self.event.get('max_probe_backward_error',0.)>SOLVE_BACKWARD_ERROR_LIMIT:
            # Too ill-conditioned for the relaxed probes to bound the estimate.
            probe=SOLVE_BACKWARD_ERROR_LIMIT
            estimate=norm*float(onenormest(inverse))
            if not np.isfinite(estimate):raise HierarchicalRejected('Nonfinite compressed condition estimate.')
        self.diagnostics.update(condition_est=estimate,condition_method='equilibrated_1norm_compressed_refined_inverse',condition_label=self.label)

    @timed_stage('linear_solve')
    def solve(self,rhs):
        import ghost_backend.twod.solver as rcs
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        x,residual=self.inverse(b,return_residual=True)
        self.relative_residual=self.relative_errors(x,b,residual)
        self.event['rhs_batches']+=1
        self.event['max_rhs_columns']=max(self.event['max_rhs_columns'],b.shape[1])
        self.event['max_relative_residual']=max(self.event['max_relative_residual'],float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics.update(linear_backward_error=self.event['max_backward_error'],linear_backward_error_limit=1e-12)
        rcs._record_dense_backend_event(requested='cpu',used='cpu_compressed',n=len(self.a),label=self.label,
            factorizations=self.event['factorizations']-self.reported,rhs_columns=b.shape[1],compressed=self.event,
            sweep_compression=self.event.get('sweep_compression'),condition_method=(self.diagnostics or {}).get('condition_method'))
        self.reported=self.event['factorizations']
        return x[:,0] if vector else x
