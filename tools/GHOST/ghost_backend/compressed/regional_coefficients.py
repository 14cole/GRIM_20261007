"""Regional coefficient queries with prepared plans and loss bounds."""
import numpy as np
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.formulations.regions as mr
import ghost_backend.twod.assembly.scatter as ss
from scipy.sparse import coo_matrix
from ghost_backend.twod.basis import integral_bounds
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.assembly.compact import CompactOperator


def hankel_envelopes(k, lower_distance):
    """Upper envelopes for |i H0/4| and |i k H1/4|, alpha*r > 1.

    DLMF 10.27.8 and 10.32.9 give |H_nu^(2)(kr)| <= 2 K_nu(alpha*r)/pi.
    cosh(t)>=1+t*t/2 and cosh(t)<=exp(t*t/2) bound K0 and K1 by Gaussians.
    Use a finite-domain bound only; no growing/real kernel is truncated.
    """
    a = -complex(k).imag*np.asarray(lower_distance)
    if complex(k).imag >= 0 or complex(k).real <= 0 or not np.all(np.isfinite(a)) or np.any(a <= 1):
        return None

    decay = np.exp(-np.minimum(a,700))
    g = decay/np.sqrt(8*np.pi*a)
    h = abs(k)*decay/np.sqrt(8*np.pi*(a-1))
    return g*(1+1e-8),h*(1+1e-8)


def hankel_envelope(k, attenuation, kind):
    """hankel_envelopes(k, lower)[kind] once attenuation = -Im(k)*lower is known to exceed 1."""
    decay = np.exp(-np.minimum(attenuation,700))
    if kind == 0:
        value = decay/np.sqrt(8*np.pi*attenuation)
    else:
        value = abs(k)*decay/np.sqrt(8*np.pi*(attenuation-1))
    return value*(1+1e-8)


def _span(ids):
    return (int(ids.min()), int(ids.max())) if len(ids) else (0, -1)


def _route_plan(row_map, column_map, weight, maps=None):
    """Frequency-owned routing only; no coefficients or queried tiles retained."""
    def prepare(mapping):
        key = mapping.tobytes() if maps is not None else None
        if maps is not None and key in maps:
            return maps[key]
        nodes = np.flatnonzero(mapping >= 0)
        dofs = mapping[nodes]
        result = nodes, dofs, _span(dofs)
        if maps is not None:
            maps[key] = result
        return result
    rn, rd, rs = prepare(row_map)
    cn, cd, cs = prepare(column_map)
    return weight, rn, cn, rd, cd, rs, cs


def _selected_route(plan, rd, cd, row_span, column_span, node_count, selected_maps=None):
    weight, rn, cn, row_dofs, column_dofs, rs, cs = plan
    if rs[1] < row_span[0] or rs[0] > row_span[1] or cs[1] < column_span[0] or cs[0] > column_span[1]:
        return None
    def select(nodes, dofs, local, cache):
        key = id(dofs)
        if key in cache:
            return cache[key]
        destination = local[dofs]
        keep = destination >= 0
        active = nodes[keep]
        mapping = None
        if len(active):
            mapping = np.full(node_count, -1, int)
            mapping[active] = destination[keep]
        cache[key] = mapping, active
        return mapping, active
    row_cache, column_cache = selected_maps if selected_maps is not None else ({}, {})
    r, ri = select(rn, row_dofs, rd, row_cache)
    if r is None:
        return None
    c, ci = select(cn, column_dofs, cd, column_cache)
    if c is None:
        return None
    return (r, c, weight), ri, ci


class PreparedOracle:
    # Plain arrays and templates: tile queries may run in worker processes.
    process_tiles=True

    def __init__(self,mesh,infos,pol,cut=None,obs_order=8,src_order=8,far_order_floors=None):
        self.obs_order,self.src_order=obs_order,src_order
        self.far_order_floors=far_order_floors
        if pol not in ('TE', 'TM'):
            raise ValueError('Polarization must be TE or TM.')
        if cut is not None and (not np.isfinite(cut) or cut <= 1):
            raise ValueError('Attenuation cut must be finite and greater than one.')
        self.mesh,self.pol,self.cut=mesh,pol,cut
        self.geometry=AssemblyGeometry(mesh)
        self.layout=mr.build_layout(mesh,infos,pol)
        self.n=self.layout['n_dof']
        self.entries=self.calls=self.max_entries=self.dropped_routes=0
        nn=len(mesh.nodes)
        self.xy=np.zeros((nn,2));self.radius=np.zeros(nn)
        self.mass=np.zeros(nn);self.normal_mass=np.zeros(nn)
        for e in mesh.elements:
            for node,point,bound in zip(e.node_ids, (mesh.nodes[i].xy for i in e.node_ids), integral_bounds(e)):
                self.xy[node]=point
                self.radius[node]=max(self.radius[node],e.length)
                self.mass[node]+=bound
                self.normal_mass[node]+=bound*np.linalg.norm(e.normal)
        self.groups=[]
        route_maps = {}
        for k,requests in mr.operator_plan(self.layout):
            templates=ss.multi_outputs(None,mesh,self.layout,k,requests)
            prepared=[]
            for request,template in zip(requests,templates):
                source=self.layout['ifaces'][request['source']]['mask']
                coefficient=(None if request['observer'] is None else
                    self.layout['ifaces'][request['observer']]['robin_alpha_elements'])
                source_mass=np.zeros(nn);weighted_mass=np.zeros(nn)
                for j,e in enumerate(mesh.elements):
                    if source[j]:np.add.at(source_mass,np.asarray(e.node_ids),integral_bounds(e))
                    weight=1 if coefficient is None else abs(coefficient[j])
                    np.add.at(weighted_mass,np.asarray(e.node_ids),weight*integral_bounds(e))
                # Nodes that each route can reach, so tile queries skip whole-mesh maps.
                reach=[]
                for output in template:
                    routes=[]
                    for old_r,old_c,weight in output.routes:
                        routes.append(_route_plan(old_r, old_c, weight, route_maps))
                    reach.append(routes)
                prepared.append((request,reach,source,coefficient,source_mass,weighted_mass))
            self.groups.append((k,prepared))

        mass=mr._sparse_mass(mesh);rr=[];cc=[];vv=[]
        for mi,iface in enumerate(self.layout['ifaces']):
            local=mass[iface['nodes'],:][:,iface['nodes']].tocoo()
            rm,rp=iface['r_m'],iface['r_p']
            if rm<0 or rp<0:
                offset,_=self.layout['dof_map'][mi,'plus' if rm<0 else 'minus']
                keep=np.ones(local.nnz,bool)
                if pol=='TM':keep=abs(iface['robin_alpha'][local.row])>rcs.EPS
                rr.extend(offset+local.row[keep]);cc.extend(offset+local.col[keep])
                vv.extend((.5 if rm<0 else -.5)*local.data[keep])
            else:
                flux,_=self.layout['dof_map'][mi,'minus'];trace,_=self.layout['dof_map'][mi,'plus']
                for column,weight in ((flux,-.5),(trace,-.5*mr._inverse_beta(self.layout,iface,pol))):
                    rr.extend(flux+local.row);cc.extend(column+local.col);vv.extend(weight*local.data)
        self.jumps=coo_matrix((vv,(rr,cc)),shape=(self.n,self.n),dtype=complex).tocsr()

    def get(self,rows,cols):
        return self.get_with_error(rows,cols)[0]

    def propose_fast_far(self, rows, cols, tolerance=1e-14):
        """Experimental interpolation; full coefficient validation is required."""
        from ghost_backend.compressed.fast_far import propose
        return propose(self, rows, cols, tolerance)

    def get_with_error(self,rows,cols,assemble=True):
        rows,cols=CompactOperator._ids(rows,self.n),CompactOperator._ids(cols,self.n)
        if len(rows)*len(cols)*24>16*1024**2:raise MemoryError('Regional coefficient query exceeds 16 MiB.')
        jumps=self.jumps[rows,:][:,cols].tocoo()
        matrix=np.zeros((len(rows),len(cols)),complex,order='F')
        np.add.at(matrix,(jumps.row,jumps.col),jumps.data)
        error=np.zeros(matrix.shape)
        self.entries+=matrix.size;self.calls+=1;self.max_entries=max(self.max_entries,matrix.size)
        rd,cd=np.full(self.n,-1,int),np.full(self.n,-1,int)
        rd[rows]=np.arange(len(rows));cd[cols]=np.arange(len(cols))
        row_span,column_span=_span(rows),_span(cols)
        nn=len(self.mesh.nodes)
        pending=[]
        selected_maps = ({}, {})
        from ghost_backend.twod.formulations.combined_regions import fused_outputs
        combined = fused_outputs(matrix, self.mesh, self.layout, rows, cols)
        empty = ss.SystemScatter(matrix, nn, [], [], [])
        for k,prepared in self.groups:
            outputs=[];masks=[];coefficients=[];additional=[]
            for request,reach,source,coefficient,source_mass,weighted_mass in prepared:
                pair=[]
                for kind,output in enumerate(reach):
                    routes=[];reached_rows=[];reached_columns=[]
                    for plan in output:
                        selected = _selected_route(plan, rd, cd, row_span, column_span, nn, selected_maps)
                        if selected is None:continue
                        (r,c,weight),ri,ci=selected
                        dropped=False
                        if self.cut is not None and complex(k).imag<0:
                            dx=self.xy[ri,0][:,None]-self.xy[ci,0][None,:]
                            dy=self.xy[ri,1][:,None]-self.xy[ci,1][None,:]
                            np.multiply(dx,dx,out=dx);np.multiply(dy,dy,out=dy)
                            lower=np.sqrt(np.add(dx,dy,out=dx),out=dx);dy=None
                            np.subtract(lower,self.radius[ri,None],out=lower)
                            np.subtract(lower,self.radius[None,ci],out=lower)
                            lower=np.maximum(0,lower,out=lower)
                            attenuation=-complex(k).imag*lower
                            # cut > 1, so passing pairs satisfy hankel_envelopes' own domain checks.
                            if np.all(attenuation>=self.cut):
                                if complex(k).real>0:
                                    obs=weighted_mass[ri] if kind==0 else self.normal_mass[ri]
                                    bound=hankel_envelope(k,attenuation,kind)*obs[:,None]*source_mass[None,ci]*abs(weight[ri,None])
                                    np.add.at(error.reshape(-1),(r[ri,None]*error.shape[1]+c[None,ci]).ravel(),bound.ravel())
                                    self.dropped_routes+=1;dropped=True
                        if not dropped:
                            routes.append((r,c,weight));reached_rows.append(ri);reached_columns.append(ci)
                    # Sorted unions of the kept routes' nodes, as a scan of their maps would give.
                    rid=np.unique(np.concatenate(reached_rows)) if routes else np.empty(0,int)
                    cid=np.unique(np.concatenate(reached_columns)) if routes else np.empty(0,int)
                    pair.append(ss.SystemScatter(matrix,nn,rid,cid,routes))
                extra = combined.get((k, request['source']), (empty, empty)) if request['kind']=='plain' else (empty, empty)
                if any(len(o.row_ids) and len(o.column_ids) for o in list(pair)+list(extra)):
                    outputs.append(pair);masks.append(source);coefficients.append(coefficient)
                    additional.append(extra)
            if outputs:
                pending.append((k,outputs,masks,coefficients,additional))
        if not assemble:return matrix,error,pending
        assemble_groups(self.mesh,self.geometry,pending,self.obs_order,self.src_order,self.far_order_floors)
        return matrix,error


def assemble_groups(mesh,geometry,groups,obs_order=8,src_order=8,far_order_floors=None):
    for k,outputs,masks,coefficients,additional in groups:
        rcs._assemble_linear_operator_matrices_multi(mesh,k,True,masks,
            compute_single_layer=any(len(o[0].row_ids) for o in outputs),
            compute_double_layer_many=[bool(len(o[1].row_ids)) for o in outputs],
            single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=[(o[0].row_ids,o[0].column_ids) for o in outputs],
            double_layer_output_node_ids_many=[(o[1].row_ids,o[1].column_ids) for o in outputs],
            operator_outputs=outputs,prepared_geometry=geometry,obs_order=obs_order,src_order=src_order,
            additional_operator_outputs=additional,
            minimum_far_order=(far_order_floors or {}).get(k,0))


def reciprocal_values(oracle, requests):
    """Assemble several opposite queries through one shared kernel traversal.

    Requests carry (rows, columns, missing source indices). Equation matrices
    are not assumed symmetric: each direction/polarization keeps its own masks,
    material routes, jump terms and attenuation bounds. The common Galerkin
    engine can then evaluate reciprocal Green kernels once and produce both
    normal-derivative directions. No completed global matrix is constructed.
    """
    if type(oracle) is PreparedOracle:
        sources = [oracle]
    elif type(oracle) is PairedOracle and all(type(source) is PreparedOracle for source in oracle.oracles):
        sources = oracle.oracles
    else:
        raise TypeError('Reciprocal assembly requires the standard regional oracle.')
    grouped, result = {}, []
    for rows, cols, missing in requests:
        values = {}
        for index in missing:
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(sources) or index in values:
                raise ValueError('Invalid reciprocal source selection.')
            matrix, error, pending = sources[index].get_with_error(rows, cols, assemble=False)
            values[index] = (matrix, error, None)
            for k, outputs, masks, coefficients, additional in pending:
                item = grouped.setdefault(k, ([], [], [], []))
                item[0].extend(outputs)
                item[1].extend(masks)
                item[2].extend(coefficients)
                item[3].extend(additional)
        result.append(values)
    source = sources[0]
    floors = {}
    for item in sources:
        for k, order in (item.far_order_floors or {}).items():
            floors[k] = max(floors.get(k, 0), order)
    assemble_groups(source.mesh, source.geometry,
                    [(k,) + tuple(values) for k, values in grouped.items()],
                    source.obs_order, source.src_order, floors)
    if type(oracle) is PairedOracle:
        oracle.calls += sum(bool(missing) for _, _, missing in requests)
        oracle.kernel_groups += len(grouped)
    return result


class PairedOracle:
    """TE/TM destinations share geometry/kernel traversal, preserving each law."""
    process_tiles=True
    def __init__(self,mesh,te_infos,tm_infos,cut=None,obs_order=8,src_order=8):
        self.obs_order,self.src_order=obs_order,src_order
        self.oracles=[PreparedOracle(mesh,te_infos,'TE',cut,obs_order,src_order),PreparedOracle(mesh,tm_infos,'TM',cut,obs_order,src_order)]
        a,b=self.oracles
        if a.n!=b.n or a.layout['dof_map']!=b.layout['dof_map']:
            raise ValueError('Paired assembly requires matching TE/TM DOF layouts.')
        b.geometry=a.geometry
        self.n=a.n;self.mesh=mesh;self.calls=0;self.kernel_groups=0

    def propose_fast_far(self, rows, cols, tolerance=1e-14):
        """Sample both polarizations through their shared coefficient traversal."""
        from ghost_backend.compressed.fast_far import propose
        return propose(self, rows, cols, tolerance)

    def get_with_error(self,rows,cols):
        data=[o.get_with_error(rows,cols,assemble=False) for o in self.oracles]
        grouped={}
        for matrix,error,requests in data:
            for k,outputs,masks,coefficients,additional in requests:
                item=grouped.setdefault(k,([],[],[],[]))
                item[0].extend(outputs);item[1].extend(masks);item[2].extend(coefficients)
                item[3].extend(additional)
        assemble_groups(self.mesh,self.oracles[0].geometry,[(k,)+tuple(v) for k,v in grouped.items()],self.obs_order,self.src_order)
        self.calls+=1;self.kernel_groups+=len(grouped)
        return [(matrix,error) for matrix,error,_ in data]
