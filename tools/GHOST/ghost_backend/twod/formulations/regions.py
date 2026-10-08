"""Shared multi-region layout, compact operator plan and bounded assembly."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from ghost_backend.twod.constants import EPS
from ghost_backend.twod.geometry import _surface_robin_alpha

BLOCK_ROWS = 64
# Combined TE->TM reuse is attempted only while conductor rows are at most
# 1/8 of the unknowns; above that a fresh fused assembly is faster.
COMBINED_REUSE_CONDUCTOR_FRACTION_MAX_INVERSE = 8


def geometric_near_pair_count(centers, lengths):
    """Count ordered near pairs without letting one long panel widen every query."""
    if not len(lengths):
        return 0
    tree = cKDTree(centers)
    if np.all(lengths == lengths[0]):
        return int(tree.count_neighbors(tree, 3.0*lengths[0]))
    pairs = len(lengths)  # self pairs
    for i, (center, length) in enumerate(zip(centers, lengths)):
        # Every qualifying unordered pair appears in the larger panel's ball.
        # Equal-length pairs belong to the larger index, so none are duplicated.
        candidates = np.asarray(tree.query_ball_point(center, 3.0*length), dtype=int)
        owned = (lengths[candidates] < length) | ((lengths[candidates] == length) & (candidates < i))
        candidates = candidates[owned]
        distance = np.linalg.norm(centers[candidates]-center, axis=1)
        pairs += 2*int(np.count_nonzero(distance <= 3.0*length))
    return pairs


def mesh_near_pair_count(mesh):
    """``geometric_near_pair_count`` of a mesh's elements, memoized on the mesh.

    Element centres and lengths are fixed at construction (polynomial
    enrichment only adds nodes), and the storage forecast asks for the same
    mesh once per polarization and, in the submit-time planner, once per
    basis degree; the count was 27% of a resource plan."""
    box = near_pair_memo(mesh)
    if box is not None and box[0] is not None and box[0][0] == len(mesh.elements):
        return box[0][1]
    centers = np.asarray([e.center for e in mesh.elements])
    lengths = np.asarray([e.length for e in mesh.elements])
    count = geometric_near_pair_count(centers, lengths)
    if box is not None:
        box[0] = (len(mesh.elements), count)
    return count


def near_pair_memo(mesh):
    """The one-slot memo of ``mesh_near_pair_count`` on ``mesh`` (created on
    demand; None when the mesh object cannot hold attributes).  A copy made by
    ``copy_linear_mesh`` shares the slot, so the degrees of one planning mesh
    count their pairs once."""
    box = getattr(mesh, '_near_pair_memo', None)
    if box is None:
        box = [None]
        try:
            mesh._near_pair_memo = box
        except AttributeError:
            return None
    return box


def build_layout(mesh, infos, pol):
    elements = mesh.elements
    regions, interface_elements = {}, {}
    for ei, info in enumerate(infos):
        for rid, k, eps, mu, inc in (
            (info.minus_region, info.k_minus, info.eps_minus, info.mu_minus, info.minus_has_incident),
            (info.plus_region, info.k_plus, info.eps_plus, info.mu_plus, info.plus_has_incident)):
            if rid >= 0 and rid not in regions:
                regions[rid] = dict(k=complex(k), eps=complex(eps), mu=complex(mu), has_incident=bool(inc))
        interface_elements.setdefault((info.minus_region, info.plus_region), []).append(ei)
    ifaces = []
    region_ifaces = {}
    dof_map, n_dof = {}, 0
    for mi, ((rm, rp), eids) in enumerate(sorted(interface_elements.items())):
        nodes = sorted({n for ei in eids for n in elements[ei].node_ids})
        node_positions = {n: i for i, n in enumerate(nodes)}
        alpha = np.zeros(len(nodes), dtype=np.complex128)
        counts = np.zeros(len(nodes), dtype=int)
        alpha_elements = np.zeros(len(elements), dtype=np.complex128)
        if rm < 0 or rp < 0:
            region = regions[rp if rm < 0 else rm]
            sign = -1.0 if rm < 0 else 1.0
            for ei in eids:
                z = complex(infos[ei].robin_impedance)
                if abs(z) > EPS:
                    alpha_elements[ei] = sign * _surface_robin_alpha(
                        pol, region['eps'], region['mu'], region['k'], z)
                for n in elements[ei].node_ids:
                    j = node_positions[n]
                    alpha[j] += alpha_elements[ei]
                    counts[j] += 1
            alpha /= np.maximum(counts, 1)
            if pol == 'TM':
                # u = 0 on a PEC element, so every node it touches carries the
                # Dirichlet row, as in the standalone Robin assembler. An
                # averaged Robin row at a PEC/impedance junction would apply the
                # flux equation over the PEC half of its test function (TM
                # error 2.5% instead of 0.9% at 24 panels per wavelength).
                for ei in eids:
                    if abs(alpha_elements[ei]) <= EPS:
                        for n in elements[ei].node_ids:
                            alpha[node_positions[n]] = 0.
        mask = np.zeros(len(elements), dtype=bool)
        mask[eids] = True
        ifaces.append(dict(r_m=rm, r_p=rp, eids=eids, nodes=nodes, n=len(nodes),
            pec_minus=rm < 0, pec_plus=rp < 0, robin_alpha=alpha,
            robin_alpha_elements=alpha_elements, mask=mask))
        for side, rid in (('minus', rm), ('plus', rp)):
            if rid >= 0:
                region_ifaces.setdefault(rid, []).append(mi)
                dof_map[mi, side] = (n_dof, len(nodes))
                n_dof += len(nodes)
    return dict(polarization=pol, region_props=regions, ifaces=ifaces, region_ifaces=region_ifaces,
                dof_map=dof_map, n_dof=n_dof)


def operator_plan(layout):
    """Coalesce equal-wavenumber requests while retaining every required row."""
    ifaces, regions = layout['ifaces'], layout['region_props']
    region_ifaces = layout['region_ifaces']
    by_k = {}
    for rid, mis in region_ifaces.items():
        k = regions[rid]['k']
        slot = by_k.setdefault(k, {})
        rows = {n for mi in mis for n in ifaces[mi]['nodes']}
        for mi in mis:


            request = slot.setdefault(('plain', mi), dict(kind='plain', source=mi, observer=None,
                                                         rows=set(), s_rows=set(), k_rows=set()))
            request['rows'].update(rows)
            for observer in mis:
                target = ifaces[observer]
                if target['r_m'] >= 0 and target['r_p'] >= 0:
                    request['s_rows'].update(target['nodes'])
                    request['k_rows'].update(target['nodes'])
                else:
                    pec = np.abs(target['robin_alpha']) <= EPS if layout['polarization'] == 'TM' else np.zeros(target['n'], bool)
                    request['s_rows'].update(n for n, keep in zip(target['nodes'], pec) if keep)
                    request['k_rows'].update(n for n, keep in zip(target['nodes'], pec) if not keep)
    for mi, ifc in enumerate(ifaces):
        if not (ifc['pec_minus'] or ifc['pec_plus']):
            continue
        if not np.any(np.abs(ifc['robin_alpha_elements']) > EPS):
            continue
        rid = ifc['r_p'] if ifc['pec_minus'] else ifc['r_m']
        slot = by_k.setdefault(regions[rid]['k'], {})
        for mj in region_ifaces[rid]:
            slot['weighted', mj, mi] = dict(kind='weighted', source=mj, observer=mi,
                rows=set(ifc['nodes']), s_rows=set(ifc['nodes']), k_rows=set())
    return [(k, list(slot.values())) for k, slot in by_k.items()]


def storage_resources(mesh, layout):
    entries = matrices = map_bytes = 0
    n = len(mesh.nodes)
    for k, requests in operator_plan(layout):
        for request in requests:
            rows = len(request['rows'])
            cols = layout['ifaces'][request['source']]['n']
            count = 2 if request['kind'] == 'plain' else 1
            matrices += count
            entries += (len(request['s_rows']) + len(request['k_rows'])) * cols


            map_bytes += 2 * 8 * (2*n + rows + cols)
            route_count = sum(1 for rid, sources in layout['region_ifaces'].items()
                if request['source'] in sources and layout['region_props'][rid]['k'] == k)
            map_bytes += 56*n*route_count

    width = len(mesh.elements[0].node_ids) if mesh.elements else 2
    from ghost_backend.twod.formulations.combined_regions import couplings
    combined = couplings(mesh,layout)
    for mi, rid in combined:
        # Fused D/W routes remain live together, unlike the old row-strip
        # reference. Include their maps, weights and destination node IDs.
        map_bytes += 128*n
        trace_nodes, flux_nodes = set(), set()
        for observer in layout['region_ifaces'][rid]:
            interface = layout['ifaces'][observer]
            if interface['r_m'] >= 0 and interface['r_p'] >= 0:
                trace_nodes.update(interface['nodes'])
                flux_nodes.update(interface['nodes'])
            elif layout['polarization'] == 'TM' and not np.any(interface['robin_alpha']):
                trace_nodes.update(interface['nodes'])
            else:
                flux_nodes.update(interface['nodes'])
                if np.any(interface['robin_alpha']):
                    trace_nodes.update(interface['nodes'])
        entries += (len(trace_nodes)+len(flux_nodes))*layout['ifaces'][mi]['n']
        matrices += (len(trace_nodes)+63)//64 + (len(flux_nodes)+63)//64
    map_bytes += 16*layout['n_dof'] + 64*n
    mass_bytes = 40*width*width * len(mesh.elements) + 8 * (n + 1)
    max_interface = max((i['n'] for i in layout['ifaces']), default=0)
    block_bytes = 16 * BLOCK_ROWS * max_interface * 12


    near_pairs = mesh_near_pair_count(mesh)
    import ghost_backend.twod.operators as ops
    tile = ops._assembly_tile_size(len(mesh.elements), 312)
    largest_group = max((len(requests) for _, requests in operator_plan(layout)), default=0)
    # Fused W also retains basis-pair Green moments and derivative moments.
    pair_bytes = max(1024, 256*width*width) if combined else 1024
    tile_bytes = ops.get_assembly_threads() * tile*tile * (pair_bytes + 16*largest_group)


    near_batch_samples = min(ops._NEAR_BATCH_MAX_SAMPLES, near_pairs * 16 * 16)
    near_batch_bytes = near_batch_samples * 256
    from ghost_backend.twod.assembly.near_store import NEAR_STORAGE_BYTES
    from ghost_backend.twod.polynomial_quadrature import MOMENT_CACHE_BYTES
    # Coefficients spill above their fixed cap. Integer pair plans/sorting still
    # scale with the geometric pair count; polynomial moment reuse has its own
    # run-scoped cap and may survive into a subsequent assembly.
    near_storage = min(NEAR_STORAGE_BYTES, 3*16*width*width*near_pairs)
    near_metadata = 64*near_pairs
    moment_cache = MOMENT_CACHE_BYTES if width > 2 else 0
    assembly_workspace = near_storage + near_metadata + max(tile_bytes, near_batch_bytes)


    return dict(operator_matrices=matrices, operator_entries=entries,
                geometric_near_pairs=near_pairs,
                assembly_operator_entries=0,
                operator_map_bytes=map_bytes, mass_workspace_bytes=mass_bytes,
                block_workspace_bytes=block_bytes,
                moment_cache_bytes=moment_cache,
                assembly_workspace_bytes=assembly_workspace)


def _sparse_mass(mesh):
    from ghost_backend.twod.assembly.mass import sparse_mass
    return sparse_mass(mesh)


def _assemble_system_fresh(mesh, infos, pol, obs_order=8, src_order=8):

    import ghost_backend.twod.solver as rcs
    layout = build_layout(mesh, infos, pol)
    ifaces = layout['ifaces']
    regions = layout['region_props']
    region_ifaces = layout['region_ifaces']
    dofs = layout['dof_map']
    operators, weighted = {}, {}
    for k, requests in operator_plan(layout):
        masks = [ifaces[r['source']]['mask'] for r in requests]
        coefficients = [None if r['observer'] is None else
                        ifaces[r['observer']]['robin_alpha_elements'] for r in requests]
        outputs = rcs._assemble_linear_operator_matrices_multi(
            mesh=mesh, k0=k, obs_normal_deriv=True, source_element_masks=masks,
            obs_order=obs_order, src_order=src_order,
            compute_double_layer_many=[r['kind'] == 'plain' for r in requests],
            single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=[(sorted(r['s_rows']), ifaces[r['source']]['nodes']) for r in requests],
            double_layer_output_node_ids_many=[(sorted(r['k_rows']), ifaces[r['source']]['nodes']) for r in requests])
        for request, pair in zip(requests, outputs):
            if request['kind'] == 'plain':
                operators[k, request['source']] = pair
            else:
                weighted[k, request['source'], request['observer']] = pair[0]
    mass = _sparse_mass(mesh)
    matrix = np.zeros((layout['n_dof'], layout['n_dof']), dtype=np.complex128, order='F')

    def sub(op, rows, columns):
        return op.block(rows, columns) if hasattr(op, 'block') else op[np.ix_(rows, columns)]

    def alpha_s(k, source, observer, rows, columns):
        op = weighted.get((k, source, observer))
        return 0.0 if op is None else sub(op, rows, columns)

    for mi, ifc in enumerate(ifaces):
        rm, rp = ifc['r_m'], ifc['r_p']
        all_nodes, nm = ifc['nodes'], ifc['n']
        for start in range(0, nm, BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, nm)
            rows = all_nodes[start:stop]


            m = mass[rows, :][:, all_nodes].toarray()
            if rm < 0 or rp < 0:
                side = 'plus' if rm < 0 else 'minus'
                rid = rp if rm < 0 else rm
                k = regions[rid]['k']
                dm = dofs[mi, side][0]
                dest_rows = slice(dm + start, dm + stop)
                pec = np.abs(ifc['robin_alpha'][start:stop]) <= EPS if pol == 'TM' else np.zeros(stop-start, dtype=bool)
                s, kp = operators[k, mi]
                block = (0.5 if rm < 0 else -0.5) * m + sub(kp, rows, all_nodes) + alpha_s(k, mi, mi, rows, all_nodes)
                if np.any(pec):
                    block[pec] = sub(s, rows, all_nodes)[pec]
                matrix[dest_rows, dm:dm+nm] += block
                for mj in region_ifaces[rid]:
                    if mj == mi:
                        continue
                    src = ifaces[mj]
                    dj, nj = dofs[mj, 'minus' if src['r_m'] == rid else 'plus']
                    s, kp = operators[k, mj]
                    block = sub(kp, rows, src['nodes']) + alpha_s(k, mj, mi, rows, src['nodes'])
                    if np.any(pec):
                        block[pec] = sub(s, rows, src['nodes'])[pec]
                    matrix[dest_rows, dj:dj+nj] += block
            else:
                ds, dt = dofs[mi, 'minus'][0], dofs[mi, 'plus'][0]
                rs, rt = slice(ds+start, ds+stop), slice(dt+start, dt+stop)
                key = 'mu' if pol == 'TM' else 'eps'
                beta = regions[rp][key] / regions[rm][key] if abs(regions[rm][key]) > EPS else 1+0j
                inv = 1.0 / beta if abs(beta) > EPS else 1+0j
                sm, km = operators[regions[rm]['k'], mi]
                sp, kp = operators[regions[rp]['k'], mi]
                matrix[rs, ds:ds+nm] += -0.5*m + sub(km, rows, all_nodes)
                matrix[rs, dt:dt+nm] -= inv * (0.5*m + sub(kp, rows, all_nodes))
                matrix[rt, ds:ds+nm] += sub(sm, rows, all_nodes)
                matrix[rt, dt:dt+nm] -= sub(sp, rows, all_nodes)
                for rid, sign, flux in ((rm, 1.0, 1.0), (rp, -1.0, inv)):
                    for mj in region_ifaces[rid]:
                        if mj == mi:
                            continue
                        src = ifaces[mj]
                        dj, nj = dofs[mj, 'minus' if src['r_m'] == rid else 'plus']
                        s, kp = operators[regions[rid]['k'], mj]
                        if sign > 0:
                            matrix[rs, dj:dj+nj] += sub(kp, rows, src['nodes'])
                            matrix[rt, dj:dj+nj] += sub(s, rows, src['nodes'])
                        else:
                            matrix[rs, dj:dj+nj] -= flux * sub(kp, rows, src['nodes'])
                            matrix[rt, dj:dj+nj] -= sub(s, rows, src['nodes'])

    from ghost_backend.twod.formulations.combined_regions import add_corrections
    add_corrections(matrix, mesh, layout, obs_order, src_order)
    return matrix, layout


def _inverse_beta(layout, interface, pol):
    rm, rp = interface['r_m'], interface['r_p']
    key = 'mu' if pol == 'TM' else 'eps'
    regions = layout['region_props']
    beta = regions[rp][key] / regions[rm][key] if abs(regions[rm][key]) > EPS else 1+0j
    return 1.0 / beta if abs(beta) > EPS else 1+0j


def _reuse_te_system(matrix, old, layout, mesh, obs_order, src_order):
    """Convert owned TE coefficients to TM without retaining regional operators.

    Transmission trace rows contain S and remain unchanged. Reciprocity gives
    conductor-to-transmission S from those rows. Only conductor self/cross S
    and the change of weighted Robin S need new integration.
    """
    import ghost_backend.twod.solver as rcs
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    for mi, target in enumerate(interfaces):
        if target['r_m'] < 0 or target['r_p'] < 0:
            continue
        row, n = dofs[mi, 'minus']
        ratio = _inverse_beta(layout, target, 'TM') / _inverse_beta(old, old['ifaces'][mi], 'TE')
        rid = target['r_p']
        for mj in layout['region_ifaces'][rid]:
            source = interfaces[mj]
            column, width = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
            matrix[row:row+n, column:column+width] *= ratio
    for mi, target in enumerate(interfaces):
        if target['r_m'] >= 0 and target['r_p'] >= 0:
            continue
        rid = target['r_p'] if target['r_m'] < 0 else target['r_m']
        offset, n = dofs[mi, 'plus' if target['r_m'] < 0 else 'minus']
        pec = np.abs(target['robin_alpha']) <= EPS
        pec_positions = np.flatnonzero(pec)
        other_positions = np.flatnonzero(~pec)
        delta = target['robin_alpha_elements'] - old['ifaces'][mi]['robin_alpha_elements']
        regional = layout['region_ifaces'][rid]
        conductors = [mj for mj in regional if interfaces[mj]['r_m'] < 0 or interfaces[mj]['r_p'] < 0]
        requests = []
        if len(pec_positions):
            requests.append(('pec', pec_positions, conductors, None))
        if len(other_positions) and np.any(delta != 0):
            requests.append(('robin', other_positions, regional, delta))
        masks, output_ids, coefficients = [], [], []
        for kind, positions, sources, weights in requests:
            mask = np.zeros(len(mesh.elements), bool)
            columns = set()
            for mj in sources:
                mask[interfaces[mj]['eids']] = True
                columns.update(interfaces[mj]['nodes'])
            masks.append(mask)
            output_ids.append((np.asarray(target['nodes'])[positions], sorted(columns)))
            coefficients.append(weights)
        from ghost_backend.twod.assembly.scatter import SystemScatter
        direct_outputs = []
        for (kind, positions, sources, weights), (rows, columns) in zip(requests, output_ids):
            row_map = np.full(len(mesh.nodes), -1, np.int64)
            row_map[rows] = offset + positions
            column_map = np.full(len(mesh.nodes), -1, np.int64)
            for mj in sources:
                source = interfaces[mj]
                column, width = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
                column_map[source['nodes']] = np.arange(column, column+width)
                if kind == 'pec':
                    for start in range(0, len(positions), BLOCK_ROWS):
                        matrix[np.ix_(offset+positions[start:start+BLOCK_ROWS],
                                      np.arange(column, column+width))] = 0
            direct_outputs.append((
                SystemScatter(matrix, len(mesh.nodes), rows, columns,
                    [(row_map, column_map, np.ones(len(mesh.nodes)))]),
                SystemScatter(matrix, len(mesh.nodes), [], columns, [])))
        rcs._assemble_linear_operator_matrices_multi(mesh,
            layout['region_props'][rid]['k'], True, masks, obs_order=obs_order, src_order=src_order,
            compute_double_layer=False, single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=output_ids, operator_outputs=direct_outputs) if masks else None
        direct_outputs = None


        for mj in regional:
            source = interfaces[mj]
            if mj in conductors or not len(pec_positions):
                continue
            trace, width = dofs[mj, 'plus']
            column, _ = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
            sign = 1.0 if source['r_m'] == rid else -1.0
            for start in range(0, len(pec_positions), BLOCK_ROWS):
                local = pec_positions[start:start+BLOCK_ROWS]
                block = matrix[np.ix_(np.arange(trace, trace+width), offset+local)].T
                matrix[np.ix_(offset+local, np.arange(column, column+width))] = sign * block
        block = None
    return matrix, layout


def _reuse_combined_system(matrix, old, layout, mesh, infos, obs_order, src_order):
    """Reuse transmission traces/fluxes, independently rebuild conductor rows.

    Combined D traces are not S-reciprocal. Their conductor equations are
    queried from the actual TM operator, never transposed from a TE trace.
    """
    from ghost_backend.twod.formulations.combined_regions import couplings
    if (old['polarization'] != 'TE' or layout['polarization'] != 'TM'
            or old['dof_map'] != layout['dof_map']
            or couplings(mesh, old) != couplings(mesh, layout)):
        return None
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    if len(interfaces) != len(old['ifaces']) or any(
            (a['r_m'], a['r_p']) != (b['r_m'], b['r_p'])
            or not np.array_equal(a['nodes'], b['nodes'])
            for a, b in zip(interfaces, old['ifaces'])):
        return None
    # Conductor/Robin rows are rebuilt through bounded row-strip coefficient
    # queries, which cost several times more per row than the fused assembly.
    # That only pays when transmission rows dominate: measured break-even on
    # coated cylinders lies between 9% (reuse wins) and 20% (fresh assembly
    # wins) conductor rows. A standalone PEC/IBC body has nothing to reuse.
    conductor_count = sum(dofs[mi, 'plus' if target['r_m'] < 0 else 'minus'][1]
                          for mi, target in enumerate(interfaces)
                          if target['r_m'] < 0 or target['r_p'] < 0)
    if conductor_count * COMBINED_REUSE_CONDUCTOR_FRACTION_MAX_INVERSE > layout['n_dof']:
        return None
    for mi, target in enumerate(interfaces):
        if target['r_m'] < 0 or target['r_p'] < 0:
            continue
        row, count = dofs[mi, 'minus']
        ratio = _inverse_beta(layout, target, 'TM') / _inverse_beta(old, old['ifaces'][mi], 'TE')
        rid = target['r_p']
        for mj in layout['region_ifaces'][rid]:
            source = interfaces[mj]
            column, width = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
            matrix[row:row+count, column:column+width] *= ratio
    conductor_rows = []
    for mi, target in enumerate(interfaces):
        if target['r_m'] < 0 or target['r_p'] < 0:
            offset, count = dofs[mi, 'plus' if target['r_m'] < 0 else 'minus']
            conductor_rows.extend(range(offset, offset+count))
    if conductor_rows:
        from ghost_backend.compressed.regional_coefficients import PreparedOracle
        from ghost_backend.twod.assembly.session import current_session
        from ghost_backend.twod.operators import _graded_w_far_floor
        from ghost_backend.twod.basis import mesh_degree
        # A full group containing W raises the shared S/D far rule. A query
        # containing only TM conductor traces can prune all W routes; retain
        # that group's full-system rule even then (including custom low orders).
        lengths = np.asarray([e.length for e in mesh.elements])
        floors = {}
        for _, rid in couplings(mesh, layout):
            targets = [interfaces[i] for i in layout['region_ifaces'][rid]]
            has_flux = any((t['r_m'] >= 0 and t['r_p'] >= 0)
                           or np.any(t['robin_alpha']) for t in targets)
            if has_flux:
                k = layout['region_props'][rid]['k']
                floors[k] = max(obs_order, src_order,
                    _graded_w_far_floor(k, lengths, 3., mesh_degree(mesh)))
        oracle = PreparedOracle(mesh, infos, 'TM', obs_order=obs_order, src_order=src_order,
                                far_order_floors=floors)
        columns = np.arange(oracle.n)
        rows_per_query = max(1, 16*1024**2 // (24*oracle.n))
        session = current_session()
        for start in range(0, len(conductor_rows), rows_per_query):
            if session is not None:
                session.checkpoint()
            rows = np.asarray(conductor_rows[start:start+rows_per_query])
            matrix[rows] = oracle.get(rows, columns)
    return matrix, layout


def assemble_system(mesh, infos, pol, obs_order=8, src_order=8):
    from ghost_backend.twod.formulations.combined_regions import couplings
    from ghost_backend.compressed.runtime import enabled, regional
    if enabled():return regional(mesh,infos,pol,obs_order,src_order)
    from ghost_backend.twod.assembly.polynomial_pair import dense_system
    projected = dense_system(mesh, infos, pol, obs_order, src_order)
    if projected is not None:
        return projected
    from ghost_backend.twod.assembly.session import current_session, system_key
    session = current_session()
    key = system_key(mesh, infos, 'multi_region', obs_order, src_order) if session is not None else None
    previous = session.take(key, pol) if session is not None else None
    if previous is not None and obs_order == src_order:
        matrix, old = previous
        if old['polarization'] == pol:
            return matrix, old  # assembled in the TE step's traversal (assemble_pair)
        if couplings(mesh, old):
            layout = build_layout(mesh, infos, pol)
            reused = _reuse_combined_system(matrix, old, layout, mesh, infos, obs_order, src_order)
            if reused is not None:
                return reused
            from ghost_backend.twod.assembly.scatter import assemble_multi
            return assemble_multi(mesh,infos,pol,obs_order,src_order,destination=matrix)
        layout = build_layout(mesh, infos, pol)
        return _reuse_te_system(matrix, old, layout, mesh, obs_order, src_order)
    previous = None
    from ghost_backend.twod.assembly.scatter import assemble_multi, assemble_pair
    te_layout = (build_layout(mesh, infos, pol) if session is not None and session.pair_dense
                 and pol == 'TE' and obs_order == src_order else None)
    if te_layout is not None and _tm_would_assemble_fresh(mesh, te_layout):
        # The combined equation needs S, K', D and W where the former one needed
        # a single operator, and a conductor's TM system cannot be derived from
        # the TE one. Both share every kernel evaluation instead: measured 1.2x
        # (PEC) to 1.6x (impedance, mixed, coated core) on the co-polarized solve.
        (matrix, layout), partner = assemble_pair(mesh, infos, obs_order, src_order, te_layout)
        session.save(key, pol, partner)
        return matrix, layout
    matrix, layout = assemble_multi(mesh, infos, pol, obs_order, src_order)
    if session is not None and obs_order == src_order:
        session.save(key, pol, (matrix, layout))
    return matrix, layout


def _tm_would_assemble_fresh(mesh, layout):
    """True where ``_reuse_combined_system`` declines: combined, conductor-dominated systems."""
    from ghost_backend.twod.formulations.combined_regions import couplings
    if not couplings(mesh, layout):
        return False
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    conductor_count = sum(dofs[mi, 'plus' if target['r_m'] < 0 else 'minus'][1]
                          for mi, target in enumerate(interfaces)
                          if target['r_m'] < 0 or target['r_p'] < 0)
    return conductor_count * COMBINED_REUSE_CONDUCTOR_FRACTION_MAX_INVERSE > layout['n_dof']


def rhs_many(mesh, layout, k0, angles):
    """Integrate incident moments once, only on illuminated interfaces."""
    from ghost_backend.twod.assembly.kernels import incident_loads
    regions, interfaces, dofs = layout['region_props'], layout['ifaces'], layout['dof_map']
    mask = np.zeros(len(mesh.elements), bool)
    weights = np.zeros(len(mesh.elements), complex)
    for interface in interfaces:
        if any(regions.get(rid, {}).get('has_incident') for rid in (interface['r_m'], interface['r_p'])):
            mask[interface['eids']] = True
            weights[interface['eids']] = interface['robin_alpha_elements'][interface['eids']]
    bu, bdn = incident_loads(mesh, k0, angles, element_mask=mask,
                            observation_coefficients=weights)
    rhs = np.zeros((layout['n_dof'], len(angles)), complex)
    for mi, interface in enumerate(interfaces):
        nodes, n = interface['nodes'], interface['n']
        rm, rp = interface['r_m'], interface['r_p']
        if rm < 0 or rp < 0:
            rid, side = (rp, 'plus') if rm < 0 else (rm, 'minus')
            if regions[rid].get('has_incident'):
                offset, _ = dofs[mi, side]
                block = bdn[nodes].copy()
                if layout['polarization'] == 'TM':
                    pec = np.abs(interface['robin_alpha']) <= EPS
                    block[pec] = bu[nodes][pec]
                rhs[offset:offset+n] = -block
        else:
            flux, trace = dofs[mi, 'minus'][0], dofs[mi, 'plus'][0]
            if regions[rm].get('has_incident'):
                rhs[flux:flux+n] -= bdn[nodes]
                rhs[trace:trace+n] -= bu[nodes]
            if regions[rp].get('has_incident'):
                rhs[flux:flux+n] += _inverse_beta(layout, interface, layout['polarization']) * bdn[nodes]
                rhs[trace:trace+n] += bu[nodes]
    return rhs


def exterior_projection(mesh, layout):
    regions = layout['region_props']
    rid = next((rid for rid, region in regions.items() if region.get('has_incident')), 0)
    mask = np.zeros(len(mesh.elements), bool)
    mappings = []
    for mi, interface in enumerate(layout['ifaces']):
        side = 'minus' if interface['r_m'] == rid else 'plus' if interface['r_p'] == rid else None
        if side is not None:
            offset, count = layout['dof_map'][mi, side]
            mappings.append((interface['nodes'], offset, count))
            mask[interface['eids']] = True
    def density(solution):
        result = np.zeros((len(mesh.nodes), solution.shape[1]), complex)
        for nodes, offset, count in mappings:
            result[nodes] += solution[offset:offset+count]
        return result
    return mask, density


def dof_coordinates(mesh, layout):
    xy = np.zeros((len(mesh.nodes), 2))
    for element in mesh.elements:
        xy[list(element.node_ids)] = [mesh.nodes[i].xy for i in element.node_ids]
    result = np.empty((layout['n_dof'], 2))
    for (mi, side), (offset, count) in layout['dof_map'].items():
        result[offset:offset+count] = xy[layout['ifaces'][mi]['nodes']]
    return result
