"""Combined regional potentials on closed, continuous PEC/transmission contours.

Each regional field uses S*phi + gamma*D*phi. gamma changes sign with the
region side so the auxiliary impedance condition has consistent orientation.
Open material junctions retain the existing regional representation: their
hypersingular endpoint terms need a separate derivation.

A closed one-sided (PEC/impedance) contour may carry a spatially varying Robin
law, including PEC zones, several impedance zones and tapers. The boundary
equation is ``dn(u) + alpha(x)*u`` (or ``u`` on TM PEC nodes) applied to the
same closed-contour density, so W is never weighted by alpha and its Maue form
keeps no endpoint terms; alpha only weights S and the D trace. Uniqueness of
``S + gamma*D`` follows from the interior impedance problem with ``1/gamma`` and
does not involve alpha. Rows are therefore routed per node exactly as the base
S/K' assembly routes them.
"""
import numpy as np
from ghost_backend.twod.constants import EPS


def couplings(mesh, layout):
    if 'combined_couplings' in layout:
        return layout['combined_couplings']
    interfaces = layout['ifaces']
    closed = []
    for interface in interfaces:
        counts = {}
        for ei in interface['eids']:
            for node in mesh.elements[ei].node_ids[:2]:
                counts[node] = counts.get(node, 0) + 1
        closed.append(bool(counts) and all(count == 2 for count in counts.values()))
    result = {}
    for rid, indices in layout['region_ifaces'].items():
        if not all(closed[mi] for mi in indices):
            continue
        k = layout['region_props'][rid]['k']
        for mi in indices:
            result[mi, rid] = (1 if interfaces[mi]['r_m'] == rid else -1)/(1j*k)
    layout['combined_couplings'] = result
    return result


def _route_conductor_rows(layout, target, dest, gamma, trace_map, trace_weight, flux_map, flux_weight):
    """Route D and W per node on a one-sided interface.

    TM nodes without impedance carry the Dirichlet trace equation and receive
    ``gamma*D``. Every other node carries the flux (Neumann/Robin) equation and
    receives ``-gamma*W`` plus ``gamma*alpha*D``. The PEC test is the base
    assembly's own (nodal alpha within EPS), so both parts of one row always
    describe the same equation. Uniform contours reduce to the former
    whole-interface selection.
    """
    nodes, dest = np.asarray(target['nodes']), np.asarray(dest)
    alpha = np.asarray(target['robin_alpha'])
    impedance = np.abs(alpha) > EPS
    dirichlet = ~impedance if layout['polarization'] == 'TM' else np.zeros(len(nodes), bool)
    trace_map[nodes[dirichlet]], trace_weight[nodes[dirichlet]] = dest[dirichlet], gamma
    flux = ~dirichlet
    flux_map[nodes[flux]], flux_weight[nodes[flux]] = dest[flux], -gamma
    robin = flux & impedance
    trace_map[nodes[robin]], trace_weight[nodes[robin]] = dest[robin], gamma*alpha[robin]
    return nodes[robin], alpha[robin]


def _trace_jump_mass(mesh, interface, robin_nodes, robin_alpha):
    """Mass rows of the +-1/2 D-trace jump for one source interface.

    Dirichlet and transmission rows use the plain mass. A Robin row is later
    scaled by ``gamma*alpha_node``; its jump must instead be the element-weighted
    ``sum_e alpha_e*M_e`` that the base assembly already uses for ``alpha*S``,
    so those rows hold ``M_alpha/alpha_node``. With a uniform law both forms
    coincide; with a taper or zones the nodal form alone left a second-order
    error proportional to the steepness of the law.
    """
    from scipy.sparse import diags
    from ghost_backend.twod.assembly.mass import sparse_mass
    mass = sparse_mass(mesh, interface['mask'])
    if not len(robin_nodes):
        return mass
    weighted = sparse_mass(mesh, np.asarray(interface['robin_alpha_elements'])*interface['mask'])
    keep, scale = np.ones(len(mesh.nodes), complex), np.zeros(len(mesh.nodes), complex)
    keep[robin_nodes], scale[robin_nodes] = 0., 1./np.asarray(robin_alpha)
    return (diags(keep) @ mass + diags(scale) @ weighted).tocsr()


def add_corrections(matrix, mesh, layout, obs_order=8, src_order=8, rows=None, columns=None,
                    geometry=None):
    """Add D traces and -W fluxes in bounded native coefficient queries."""
    import ghost_backend.twod.solver as rcs
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.assembly.scatter import SystemScatter
    n, size = len(mesh.nodes), layout['n_dof']
    rows = np.arange(size) if rows is None else np.asarray(rows)
    columns = np.arange(size) if columns is None else np.asarray(columns)
    row_map, col_map = np.full(size,-1,int), np.full(size,-1,int)
    row_map[rows], col_map[columns] = np.arange(len(rows)), np.arange(len(columns))
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    for (mi, rid), gamma in couplings(mesh, layout).items():
        interface = interfaces[mi]
        side = 'minus' if interface['r_m'] == rid else 'plus'
        offset, count = dofs[mi,side]
        cc = np.full(n,-1,int)
        cc[interface['nodes']] = col_map[offset:offset+count]
        source_nodes = np.flatnonzero(cc >= 0)
        if not len(source_nodes):
            continue
        trace_map, flux_map = np.full(n,-1,int), np.full(n,-1,int)
        trace_weight, flux_weight = np.zeros(n,complex), np.zeros(n,complex)
        own_robin = (np.empty(0, int), np.empty(0, complex))
        for observer in layout['region_ifaces'][rid]:
            target = interfaces[observer]
            rm, rp, nodes = target['r_m'], target['r_p'], target['nodes']
            if rm < 0 or rp < 0:
                start, width = dofs[observer,'plus' if rm < 0 else 'minus']
                routed = _route_conductor_rows(layout, target, row_map[start:start+width], gamma,
                                               trace_map, trace_weight, flux_map, flux_weight)
                if observer == mi:
                    own_robin = routed
            else:
                flux, width = dofs[observer,'minus']
                trace, _ = dofs[observer,'plus']
                trace_map[nodes], flux_map[nodes] = row_map[trace:trace+width], row_map[flux:flux+width]
                trace_weight[nodes] = gamma * (1 if rid == rm else -1)
                flux_weight[nodes] = -gamma * (1 if rid == rm else -_inverse_beta(layout,target,layout['polarization']))
        k = layout['region_props'][rid]['k']
        mass = _trace_jump_mass(mesh, interface, *own_robin)
        for kind, mapping, weight in (('D', trace_map, trace_weight), ('W', flux_map, flux_weight)):
            active_rows = np.flatnonzero(mapping >= 0)
            output = SystemScatter(matrix,n,active_rows,source_nodes,[(mapping,cc,weight)])
            for start in range(0,len(active_rows),64):
                rr = active_rows[start:start+64]
                if kind == 'D':
                    _, operator = rcs._assemble_linear_operator_matrices_multi(mesh,k,False,
                        [interface['mask']], obs_order=obs_order,src_order=src_order,
                        compute_single_layer=False, output_node_ids_many=[([],source_nodes)],
                        double_layer_output_node_ids_many=[(rr,source_nodes)],
                        prepared_geometry=geometry)[0]
                    values = (operator.values.copy() if hasattr(operator,'values') else
                              operator[np.ix_(rr,source_nodes)].copy())
                    values += (.5 if side == 'minus' else -.5)*mass[rr,:][:,source_nodes].toarray()
                else:
                    operator = rcs._assemble_linear_hypersingular_matrix(mesh,k,
                        obs_order=obs_order,src_order=src_order,source_element_mask=interface['mask'],
                        output_node_ids=(rr,source_nodes),prepared_geometry=geometry)
                    values = operator.values
                output.scatter_add(rr[:,None],source_nodes[None,:],values)


def fused_outputs(matrix, mesh, layout, rows=None, columns=None):
    """D/W routes keyed by (wavenumber, source interface), plus trace jumps.

    The outputs write directly to the owned dense matrix or compressed tile.
    No rectangular D/W matrices or row-strip queries are retained.
    """
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.assembly.scatter import SystemScatter
    n, size = len(mesh.nodes), layout['n_dof']
    rows = np.arange(size) if rows is None else np.asarray(rows)
    columns = np.arange(size) if columns is None else np.asarray(columns)
    row_map, col_map = np.full(size, -1, int), np.full(size, -1, int)
    row_map[rows], col_map[columns] = np.arange(len(rows)), np.arange(len(columns))
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    grouped = {}
    masses = layout.setdefault('combined_sparse_mass', {})
    for (mi, rid), gamma in couplings(mesh, layout).items():
        interface = interfaces[mi]
        side = 'minus' if interface['r_m'] == rid else 'plus'
        offset, count = dofs[mi, side]
        cc = np.full(n, -1, int)
        cc[interface['nodes']] = col_map[offset:offset+count]
        source_nodes = np.flatnonzero(cc >= 0)
        if not len(source_nodes):
            continue
        trace_map, flux_map = np.full(n, -1, int), np.full(n, -1, int)
        trace_weight, flux_weight = np.zeros(n, complex), np.zeros(n, complex)
        own_robin = (np.empty(0, int), np.empty(0, complex))
        for observer in layout['region_ifaces'][rid]:
            target = interfaces[observer]
            rm, rp, nodes = target['r_m'], target['r_p'], target['nodes']
            if rm < 0 or rp < 0:
                start, width = dofs[observer, 'plus' if rm < 0 else 'minus']
                routed = _route_conductor_rows(layout, target, row_map[start:start+width], gamma,
                                               trace_map, trace_weight, flux_map, flux_weight)
                if observer == mi:
                    own_robin = routed
            else:
                flux, width = dofs[observer, 'minus']
                trace, _ = dofs[observer, 'plus']
                trace_map[nodes], flux_map[nodes] = row_map[trace:trace+width], row_map[flux:flux+width]
                trace_weight[nodes] = gamma*(1 if rid == rm else -1)
                flux_weight[nodes] = -gamma*(1 if rid == rm else -_inverse_beta(layout, target, layout['polarization']))
        k = layout['region_props'][rid]['k']
        routes = grouped.setdefault((k, mi), [[], []])
        for index, (mapping, weight) in enumerate(((trace_map, trace_weight), (flux_map, flux_weight))):
            if np.any(mapping >= 0):
                routes[index].append((mapping, cc, weight))
        if np.any(trace_map >= 0):
            # Robin rows follow the layout's own law, never the queried subset.
            if mi not in masses:
                masses[mi] = _trace_jump_mass(mesh, interface, *own_robin).tocoo()
            mass = masses[mi]
            output = SystemScatter(matrix, n, np.flatnonzero(trace_map >= 0), source_nodes,
                                   [(trace_map, cc, trace_weight)])
            output.scatter_add(mass.row, mass.col, (.5 if side == 'minus' else -.5)*mass.data)
    result = {}
    for key, pair in grouped.items():
        outputs = []
        for routes in pair:
            rr = np.unique(np.concatenate([np.flatnonzero(r >= 0) for r, c, w in routes])) if routes else []
            cc = np.unique(np.concatenate([np.flatnonzero(c >= 0) for r, c, w in routes])) if routes else []
            outputs.append(SystemScatter(matrix, n, rr, cc, routes))
        result[key] = tuple(outputs)
    return result


def exterior_double_density(mesh, layout):
    coefficients = {key: gamma for key, gamma in couplings(mesh,layout).items()
                    if layout['region_props'][key[1]]['has_incident']}
    if not coefficients:
        return None
    def density(solution):
        result = np.zeros((len(mesh.nodes),solution.shape[1]),complex)
        for (mi,rid), gamma in coefficients.items():
            interface = layout['ifaces'][mi]
            offset,count = layout['dof_map'][mi,'minus' if interface['r_m']==rid else 'plus']
            result[interface['nodes']] += gamma*solution[offset:offset+count]
        return result
    return density
