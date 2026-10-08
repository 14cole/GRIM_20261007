"""Screened CPU kernels and exact straight-element plane-wave moments."""
import hashlib
import threading
import time
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
from scipy.fftpack import dct
from scipy.interpolate import PPoly
from scipy.special import hankel2
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.twod.basis import plane_wave_moments
from ghost_backend.execution.cpu import current_state, TABLE_BYTES


def directions(angles):
    phi = np.deg2rad(np.asarray(angles, dtype=float).reshape(-1))
    return np.stack((np.cos(phi), np.sin(phi)), axis=1)


def sinc_and_derivative(u):
    """sin(u)/u and half its derivative, i.e. d/dz sinc(z/2)."""
    u = np.asarray(u, dtype=float)
    small = np.abs(u) < .01
    safe = np.where(small, 1., u)
    sine, cosine = np.sin(safe), np.cos(safe)
    f = sine/safe
    fp = (safe*cosine-sine)/(2*safe*safe)
    v = u[small]
    v2 = v*v
    f[small] = 1+v2*(-1/6+v2*(1/120+v2*(-1/5040+v2/362880)))
    fp[small] = v*(-1/6+v2*(1/60+v2*(-1/1680+v2*(1/90720-v2/7983360))))
    return f, fp


def moments(centers, edges, lengths, k, dirs):
    z = float(k)*(edges @ dirs.T)
    f, fp = sinc_and_derivative(z/2)
    phase = lengths[:, None]*np.exp(1j*float(k)*(centers @ dirs.T))
    return phase*(.5*f+1j*fp), phase*(.5*f-1j*fp)


@timed_stage("excitation")
def incident(elem, k_air, elevations_deg, order=8):
    dirs = directions(elevations_deg)
    if len(elem.node_ids) > 2:
        return plane_wave_moments(np.array([elem.center]), np.array([elem.p1-elem.p0]),
            np.array([elem.length]), k_air, dirs, len(elem.node_ids)-1)[0]
    i0,i1 = moments(np.array([elem.center]), np.array([elem.p1-elem.p0]),
                    np.array([elem.length]), k_air, dirs)
    return np.vstack((i0[0],i1[0]))


@timed_stage("excitation")
def incident_dn(elem, k_air, elevations_deg, order=8):
    load = incident.__wrapped__(elem,k_air,elevations_deg,order)
    return load*(1j*float(k_air)*(directions(elevations_deg) @ elem.normal))[None,:]


def geometry_arrays(mesh, element_mask=None):
    elements = mesh.elements
    width = len(elements[0].node_ids) if elements else 2
    if element_mask is not None:
        mask = np.asarray(element_mask,dtype=bool).reshape(-1)
        if mask.size != len(elements):
            raise ValueError("Element mask size does not match mesh")
        elements = [e for e,keep in zip(elements,mask) if keep]
    return (np.array([e.node_ids for e in elements],dtype=np.int64).reshape(-1,width),
            np.array([e.center for e in elements],dtype=float).reshape(-1,2),
            np.array([e.p1-e.p0 for e in elements],dtype=float).reshape(-1,2),
            np.array([e.length for e in elements],dtype=float),
            np.array([e.normal for e in elements],dtype=float).reshape(-1,2))


PROJECTION_CACHE_BYTES = 8 * 1024**2


class GridProjection:
    """One solve's bounded, immutable observation weights for repeated RHS batches.

    Cache only unweighted moments. DLP weights are made in the caller's existing
    tile workspace, so SLP and DLP share the expensive phase evaluation without
    retaining two copies. Once the cap is reached, uncached tiles use the usual
    bounded calculation; there is no eviction/rebuild loop on a large mesh.
    """
    def __init__(self, mesh, k, observations, budget_bytes=PROJECTION_CACHE_BYTES):
        self.mesh, self.k = mesh, float(k)
        dirs = directions(observations)
        self.direction_key = (dirs.shape, hashlib.sha256(dirs.tobytes()).digest())
        self.budget = max(0, int(budget_bytes))
        self.bytes = 0
        self.geometry, self.weights = {}, {}
        self.hits = self.builds = 0

    def arrays(self, mesh, k, dirs, mask):
        direction_key = (dirs.shape, hashlib.sha256(dirs.tobytes()).digest())
        if mesh is not self.mesh or float(k) != self.k or direction_key != self.direction_key:
            raise ValueError('Prepared projection belongs to another mesh, frequency or observation grid.')
        key = None if mask is None else np.asarray(mask, bool).reshape(-1).tobytes()
        arrays = self.geometry.get(key)
        if arrays is None:
            arrays = geometry_arrays(mesh, mask)
            size = sum(a.nbytes for a in arrays) + (0 if key is None else len(key))
            if self.bytes + size <= self.budget:
                for array in arrays:
                    array.flags.writeable = False
                self.geometry[key] = arrays
                self.bytes += size
        return key, arrays

    def moments(self, key, part, arrays, dirs):
        token = (key, part.start, part.stop)
        value = self.weights.get(token)
        if value is not None:
            self.hits += 1
            return value
        ids, centers, edges, lengths, _ = arrays
        if ids.shape[1] > 2:
            value = plane_wave_moments(centers[part], edges[part], lengths[part], self.k,
                                      dirs, ids.shape[1]-1)
        else:
            value = moments(centers[part], edges[part], lengths[part], self.k, dirs)
        self.builds += 1
        buffers = (value,) if isinstance(value, np.ndarray) else value
        size = sum(a.nbytes for a in buffers)
        if self.bytes + size <= self.budget:
            for array in buffers:
                array.flags.writeable = False
            self.weights[token] = value
            self.bytes += size
        return value


def _load_cache():
    """The solve's plane-wave load slot while a monostatic batch is open, else None.

    ``fields._solve_fields`` marks the slot ``monostatic`` for batches whose far
    field is observed along the incidence directions; only then are loads kept.
    """
    from ghost_backend.twod.assembly.session import current_session
    session = current_session()
    if session is None or not session.plane_wave_loads.get('monostatic'):
        return None
    return session.plane_wave_loads


def _load_key(mesh, k, dirs, element_mask):
    mask = None if element_mask is None else np.asarray(element_mask, dtype=bool).reshape(-1).tobytes()
    return complex(k), np.ascontiguousarray(dirs, dtype=float).tobytes(), mask


def _cached_loads(mesh, k, dirs, element_mask):
    """Node-assembled polynomial plane-wave loads of this batch, if the RHS built them."""
    cache = _load_cache()
    if cache is None:
        return None
    entry = cache.get('latest')
    if entry is None or entry[0] is not mesh or entry[1] != _load_key(mesh, k, dirs, element_mask):
        return None
    return entry[2]


@timed_stage("far_field")
def farfield(mesh,density,k_air,observation_angles_deg,potential,order=8,
             element_mask=None,projection="matched",prepared_projection=None):
    dirs = directions(observation_angles_deg)
    rho = np.asarray(density,dtype=np.complex128)
    if rho.ndim==1: rho=rho[:,None]
    if rho.ndim!=2 or rho.shape[0]!=len(mesh.nodes):
        raise ValueError("Density height must match mesh nodes")
    if projection not in ("matched","grid") or potential not in ("SLP","DLP"):
        raise ValueError("Unsupported projection/potential")
    if projection=="matched" and rho.shape[1] not in (1,len(dirs)):
        raise ValueError("Matched projection needs one or angle-count columns")
    plan = prepared_projection if projection == 'grid' else None
    if plan is None:
        arrays = geometry_arrays(mesh,element_mask)
    else:
        key, arrays = plan.arrays(mesh, k_air, dirs, element_mask)
    ids,centers,edges,lengths,normals=arrays
    if projection == "matched":
        # A monostatic batch observes along its incidence directions, where the
        # far-field weights (plane-wave moments, times j k n.d for a DLP) are
        # the RHS loads that batch already assembled: sum_n rho[n] * load[n]
        # replaces a second (and, with a DLP, third) evaluation of the same
        # element moments.
        loads = _cached_loads(mesh, k_air, dirs, element_mask)
        weights = None if loads is None else loads.get('u' if potential == 'SLP' else 'dn')
        if weights is not None:
            if rho.shape[1] == 1:
                return rho[:, 0] @ weights
            return np.einsum('na,na->a', rho, weights)
    result = np.zeros((rho.shape[1],len(dirs)) if projection=="grid" else len(dirs),dtype=complex)

    block=max(1,min(len(ids),250000//max(1,len(dirs))))
    for start in range(0,len(ids),block):
        part=slice(start,start+block)
        if ids.shape[1] > 2:
            moments_all = (plane_wave_moments(centers[part], edges[part], lengths[part], k_air, dirs, ids.shape[1]-1)
                           if plan is None else plan.moments(key, part, arrays, dirs))
            if potential == 'DLP':
                if plan is not None:
                    moments_all = moments_all.copy()
                moments_all *= (1j*float(k_air)*(normals[part] @ dirs.T))[:, None, :]
            for local in range(ids.shape[1]):
                if projection == 'grid': result += rho[ids[part, local]].T @ moments_all[:, local]
                else: result += np.sum(rho[ids[part, local]] * moments_all[:, local], axis=0)
            continue
        i0,i1=(moments(centers[part],edges[part],lengths[part],k_air,dirs)
               if plan is None else plan.moments(key, part, arrays, dirs))
        if potential=="DLP":
            factor=1j*float(k_air)*(normals[part] @ dirs.T)
            if plan is not None:
                i0, i1 = i0.copy(), i1.copy()
            i0*=factor;i1*=factor
        if projection=="grid":
            result += rho[ids[part,0]].T @ i0 + rho[ids[part,1]].T @ i1
        else:
            result += np.sum(rho[ids[part,0]]*i0+rho[ids[part,1]]*i1,axis=0)
    return result


@timed_stage("excitation")
def incident_loads(mesh,k,angles,bu=None,bdn=None, want_u=True, want_dn=True,
                   element_mask=None, observation_coefficients=None):
    """Exact straight-element moments, optionally with an element-weighted trace.

    The weighted trace is added to the derivative load, preserving Robin's
    weak element coefficients. Unrequested arrays are not allocated.

    Inside a monostatic solve batch (see ``_load_cache``), element loads
    are also kept for that batch's far field: the trace load as the SLP
    weight and the coefficient-free derivative load as the DLP weight (formed
    separately when observation coefficients are present).
    """
    dirs=directions(angles);ids,centers,edges,lengths,normals=geometry_arrays(mesh, element_mask)
    fresh_u, fresh_dn = bu is None, bdn is None
    if want_u and bu is None:bu=np.zeros((len(mesh.nodes),len(dirs)),dtype=np.complex128)
    if want_dn and bdn is None:bdn=np.zeros((len(mesh.nodes),len(dirs)),dtype=np.complex128)
    coefficients = None
    if observation_coefficients is not None:
        coefficients = np.asarray(observation_coefficients, complex).reshape(-1)
        if len(coefficients) != len(mesh.elements) or not np.all(np.isfinite(coefficients)):
            raise ValueError('Observation coefficients must be finite and match the elements.')
        if element_mask is not None:
            coefficients = coefficients[np.asarray(element_mask, bool)]
    polynomial = ids.shape[1] > 2
    cache = _load_cache()
    keep_u = keep_dn = False
    plain_dn = None
    if cache is not None:
        # Keep a load only if it starts from zero here, so it is the pure load.
        keep_u = want_u and (fresh_u or not np.any(bu))
        weighted = coefficients is not None and bool(np.any(coefficients != 0))
        if want_dn and weighted:
            plain_dn = np.zeros((len(mesh.nodes),len(dirs)),dtype=np.complex128)
        keep_dn = want_dn and (plain_dn is not None or fresh_dn or not np.any(bdn))
    block=max(1,min(256,65536//max(1,len(dirs))))
    for start in range(0,len(ids),block):
        part=slice(start,start+block)
        def dot2(a):return a[:,0,None]*dirs[None,:,0]+a[:,1,None]*dirs[None,:,1]
        if polynomial:
            moments_all = plane_wave_moments(centers[part], edges[part], lengths[part], k, dirs, ids.shape[1]-1)
            dn = 1j*float(k)*dot2(normals[part])
            if plain_dn is not None:
                for local in range(ids.shape[1]):
                    np.add.at(plain_dn, ids[part, local], moments_all[:, local]*dn)
            if coefficients is not None: dn = dn + coefficients[part, None]
            for local in range(ids.shape[1]):
                if want_u: np.add.at(bu, ids[part, local], moments_all[:, local])
                if want_dn: np.add.at(bdn, ids[part, local], moments_all[:, local]*dn)
            continue
        f,fp=sinc_and_derivative(.5*float(k)*dot2(edges[part]))
        phase=lengths[part,None]*np.exp(1j*float(k)*dot2(centers[part]))
        i0=phase*(.5*f+1j*fp);i1=phase*(.5*f-1j*fp)
        if want_u:
            np.add.at(bu,ids[part,0],i0);np.add.at(bu,ids[part,1],i1)
        if want_dn:
            dn=1j*float(k)*dot2(normals[part])
            if plain_dn is not None:
                np.add.at(plain_dn, ids[part,0], i0*dn)
                np.add.at(plain_dn, ids[part,1], i1*dn)
            if coefficients is not None:
                dn = dn + coefficients[part,None]
            i0*=dn;i1*=dn
            np.add.at(bdn,ids[part,0],i0);np.add.at(bdn,ids[part,1],i1)
    if cache is not None:
        # Copies: callers edit their right-hand sides in place (edge pins).
        loads = {}
        if keep_u:
            loads['u'] = bu.copy()
        if keep_dn:
            loads['dn'] = plain_dn if plain_dn is not None else bdn.copy()
        cache['latest'] = (mesh, _load_key(mesh, k, dirs, element_mask), loads)
    return bu,bdn


def mesh_key(mesh):
    h=hashlib.sha256()
    for a in geometry_arrays(mesh):h.update(a.tobytes())
    h.update(np.array([e.panel_index for e in mesh.elements],dtype=np.int64).tobytes())
    h.update(str(len(mesh.nodes)).encode())
    return h.digest()


class Rejected(ValueError):pass


class DomainRejected(Rejected):
    """A table request no polynomial degree can satisfy (domain, attenuation, budget)."""


def values(k,r):
    return np.stack((.25j*hankel2(0,k*r),.25j*k*hankel2(1,k*r)),axis=-1)


# Interval budget of one table, and the reference kernels' own conditioning: an
# argument error of one rounding in k*r moves H0 and H1 by about eps*|k|*r
# relative, so no evaluator -- SciPy's, this table's or the Hankel expansion --
# agrees with another better than that far out. Measured table-vs-SciPy
# disagreement stays below 1.9 eps|k|r; the check allows 8 eps|k|r.
TABLE_INTERVALS = 4096
CONDITIONING_FACTOR = 8.0


def _next_bound(bound, k_abs, upper):
    return min(upper, bound*1.2, bound+.6/k_abs)


def table_reach(k, intervals=TABLE_INTERVALS):
    """Largest distance a table at this wavenumber covers within the interval budget."""
    k_abs=abs(complex(k));bound=5e-13
    for _ in range(int(intervals)):
        bound=_next_bound(bound,k_abs,np.inf)
    return bound


class KernelTable:
    def __init__(self,k,upper,degree=12,tolerance=2e-13):
        start=time.perf_counter();k=complex(k)
        if not (np.isfinite(k) and k.real>0 and k.imag<=0 and np.isfinite(upper) and upper>1e-12):raise DomainRejected("Unsupported kernel domain")
        if -k.imag*upper>300:raise DomainRejected("Attenuation exceeds screened range")
        bounds=[5e-13]
        while bounds[-1]<upper:
            bounds.append(_next_bound(bounds[-1],abs(k),upper))
            if len(bounds)>TABLE_INTERVALS+1:raise DomainRejected("Interval budget exceeded")
        self.bounds=np.array(bounds);lo=self.bounds[:-1,None];width=np.diff(self.bounds)[:,None]
        n=degree+1;x=np.cos(np.pi*(np.arange(n)+.5)/n)
        cheb=dct(values(k,lo+(x+1)*.5*width),type=2,axis=1)/n;cheb[:,0]*=.5

        conversion=np.zeros((n,n))
        for j in range(n):
            co=Chebyshev.basis(j,domain=[0,1]).convert(kind=Polynomial).coef
            conversion[:len(co),j]=co
        power=np.einsum("pj,bjc->bpc",conversion,cheb)
        self.normalized_power=np.ascontiguousarray(power)
        power=power/width[:,:,None]**np.arange(n)[None,:,None]
        self.polys=[PPoly(np.ascontiguousarray(power[:,:,c].T[::-1]),self.bounds,extrapolate=False) for c in (0,1)]
        rng=np.random.RandomState(557)
        test=np.r_[0.,1.,.5*(1+np.cos(np.pi*(np.arange(2*n+3)+.37)/(2*n+3))),rng.uniform(0,1,37)]
        rr=lo+test*width;ref=values(k,rr)
        got=np.stack([p(rr) for p in self.polys],axis=-1)
        # Relative agreement, each point against max(tolerance, 8 eps |k| r):
        # past |k| r ~ 110 the reference itself is no better than that.
        eps=np.finfo(float).eps
        point_tolerance=np.maximum(tolerance,CONDITIONING_FACTOR*eps*abs(k)*rr)[...,None]
        relative=np.abs(got-ref)/np.maximum(np.abs(ref),1e-280)
        error=float(np.max(relative))
        tail_relative=np.sum(np.abs(cheb[:,-3:]),axis=1)/np.maximum(np.min(np.abs(ref),axis=1),1e-280)
        interval_tolerance=np.maximum(tolerance,CONDITIONING_FACTOR*eps*abs(k)*self.bounds[1:])[:,None]
        tail=float(np.max(tail_relative))
        excess=max(float(np.max(relative/point_tolerance)),float(np.max(tail_relative/interval_tolerance)))
        if not np.all(np.isfinite(got)) or not excess<=1.:raise Rejected(f"Polynomial validation failed: error={error:g}, tail={tail:g}")
        self.joint = PPoly(np.stack([p.c for p in self.polys], axis=-1), self.bounds, extrapolate=False)
        from ghost_backend.twod.assembly.native.table import evaluate
        native=evaluate(self.bounds,self.normalized_power,rr)
        self.native_checked=False
        if native is not None:
            native_relative=abs(native-ref)/np.maximum(abs(ref),1e-280)
            native_error=float(np.max(native_relative))
            self.native_checked=bool(np.all(np.isfinite(native)) and np.max(native_relative/point_tolerance) <= 1.)
        self.evidence=dict(k_real=k.real,k_imag=k.imag,degree=degree,intervals=len(bounds)-1,
            bytes=sum(p.c.nbytes for p in self.polys)+self.joint.c.nbytes+self.bounds.nbytes+self.normalized_power.nbytes,check_points=rr.size,
            max_check_relative=error,max_tail_relative=tail,tolerance=tolerance,
            tolerance_model=f"max({tolerance:g}, {CONDITIONING_FACTOR:g} eps |k| r)",max_check_to_tolerance=excess,
            build_seconds=time.perf_counter()-start)
        self.evidence['evaluation_backend']='native_horner' if self.native_checked else 'scipy_ppoly'
        if native is not None:self.evidence['native_check_relative']=native_error

    def evaluate(self,distances,channel=-1):
        if self.native_checked:
            from ghost_backend.twod.assembly.native.table import evaluate
            result=evaluate(self.bounds,self.normalized_power,distances,channel)
            if result is not None:return result
        return self.joint(distances) if channel == -1 else self.polys[channel](distances)


_TABLE_LOCK = threading.Lock()


def _admit_table(state, key, table):
    """Insert into the state's table store within TABLE_BYTES (caller holds the lock)."""
    if table is not None:
        size = table.evidence['bytes']
        if size > TABLE_BYTES:
            return
        while state.tables and state.table_bytes + size > TABLE_BYTES:
            _, old = state.tables.popitem(last=False)
            if old is not None:
                state.table_bytes -= old.evidence['bytes']
        state.table_bytes += size
    if len(state.tables) >= 128:
        _, old = state.tables.popitem(last=False)
        if old is not None:
            state.table_bytes -= old.evidence['bytes']
    state.tables[key] = table


_TABLE_BUILDS = {}


def cached_table(state, key, build):
    """A table from the CPU state's budgeted store, built on a miss (None if rejected).

    Near-pair batches ask from assembly threads, so the store is locked; the
    build itself runs unlocked.  The first thread to miss a key builds it and
    the others wait for that build instead of each building a duplicate.
    """
    with _TABLE_LOCK:
        if key in state.tables:
            table = state.tables.pop(key)
            state.tables[key] = table
            return table
        pending = _TABLE_BUILDS.get(key)
        builder = pending is None
        if builder:
            pending = _TABLE_BUILDS[key] = threading.Event()
    if not builder:
        pending.wait()
        with _TABLE_LOCK:
            if key in state.tables:
                return state.tables[key]
        # The builder's table was rejected or belongs to another state.
        table = build()
        with _TABLE_LOCK:
            if key in state.tables:
                return state.tables[key]
            _admit_table(state, key, table)
        return table
    try:
        table = build()
    except BaseException:
        with _TABLE_LOCK:
            _TABLE_BUILDS.pop(key, None)
        pending.set()
        raise
    with _TABLE_LOCK:
        if key not in state.tables:
            _admit_table(state, key, table)
        table = state.tables.get(key, table)
        _TABLE_BUILDS.pop(key, None)
    pending.set()
    return table


def select_far_kernels(mesh, k, green, hankel, domain_upper=None):
    """Return immutable closures before launching tile threads."""
    state = current_state()
    # Real wavenumbers use the table too: its native Horner evaluation is about
    # 5x faster than the four scipy Bessel calls, validated to the same 2e-13.
    if state is None or not mesh.elements:
        return green, hankel
    if domain_upper is None:
        points = np.array([p for e in mesh.elements for p in (e.p0, e.p1)])
        upper = float(np.linalg.norm(np.ptp(points, axis=0))) * (1 + 1e-12) + 1e-12
    else:
        upper = float(domain_upper)
    key = (complex(k), upper)
    with _TABLE_LOCK:
        cached = key in state.tables
        if cached:
            table = state.tables.pop(key)
            state.tables[key] = table
    if not cached:
        state.checkpoint()
        try:
            # Past the interval budget (|k| r ~ 2300) or -Im(k) r = 128 the
            # native far block continues with the eight-term Hankel expansion,
            # accurate to a few eps once |k| r > 100; the numpy evaluator
            # falls back to exact kernels there.
            table_upper = upper if complex(k).imag == 0 else min(upper, 128./(-complex(k).imag))
            table_upper = min(table_upper, table_reach(k))
            reason = None
            try:
                table = KernelTable(k, table_upper)
            except DomainRejected:
                raise
            except Rejected as preferred:


                reason = str(preferred)
            if reason is not None:
                # Only a failed polynomial fit can improve with degree.
                table = KernelTable(k, table_upper, degree=16)
                table.evidence['preferred_degree_rejection'] = reason
            table.evidence.update(domain_upper=upper, table_upper=table_upper,
                                  partial_domain=table_upper < upper)
            if table.evidence['bytes'] > TABLE_BYTES:
                raise Rejected('Table exceeds memory budget')
            state.table_events.append(dict(used=True, **table.evidence))
        except Rejected as exc:
            table = None
            state.table_events.append(dict(used=False, reason=str(exc)))
        with _TABLE_LOCK:
            _admit_table(state, key, table)
    if table is None:
        return green, hankel
    def exact_subset(original, k0, real_k, dist, out, bad):
        dd = dist[bad]
        rr, work = np.empty_like(dd), np.empty_like(dd)
        if real_k:
            np.multiply(dd, complex(k0).real, out=rr)
        values = np.empty(dd.shape, complex)
        original(k0, real_k, dd, rr, work, values)
        out[bad] = values

    def evaluator(channel, original):
        def evaluate(k0, real_k, dist, kr, scratch, out):
            out[:] = table.evaluate(dist,channel)
            bad = ~np.isfinite(out)
            if np.any(bad):
                exact_subset(original, k0, real_k, dist, out, bad)
        return evaluate
    fg, fh = evaluator(0, green), evaluator(1, hankel)
    def pair(k0, real_k, dist, kr, scratch, g, h):
        values = table.evaluate(dist)
        g[:], h[:] = values[..., 0], values[..., 1]
        bad = ~np.all(np.isfinite(values), axis=-1)
        if np.any(bad):
            exact_subset(green, k0, real_k, dist, g, bad)
            exact_subset(hankel, k0, real_k, dist, h, bad)
    fg.pair = pair
    fg.table = table
    return fg, fh
