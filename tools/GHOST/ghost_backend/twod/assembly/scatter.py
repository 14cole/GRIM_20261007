"""Route Galerkin contributions straight into the owned equation matrix."""
import numpy as np
from ghost_backend.twod.assembly.compact import CompactOperator
from ghost_backend.twod.assembly.profiling import assembly_component


class SystemScatter:
    def __init__(self, matrix, node_count, rows, columns, routes):
        self.matrix = matrix
        self.row_ids = CompactOperator._ids(rows, node_count)
        self.column_ids = CompactOperator._ids(columns, node_count)
        self.node_count = node_count
        self._maps = None
        self.routes = routes

    def _node_maps(self):
        # Built on first use: compressed tiles create many destinations whose
        # routes carry their own maps and never read these.
        if self._maps is None:
            row_map = np.full(self.node_count, -1, dtype=np.int64)
            column_map = np.full(self.node_count, -1, dtype=np.int64)
            row_map[self.row_ids] = np.arange(len(self.row_ids))
            column_map[self.column_ids] = np.arange(len(self.column_ids))
            self._maps = row_map, column_map
        return self._maps

    @property
    def row_map(self):
        return self._node_maps()[0]

    @property
    def column_map(self):
        return self._node_maps()[1]

    def scatter_add(self, rows, columns, values):
        rows, columns = np.asarray(rows), np.asarray(columns)
        if rows.ndim == 2 and columns.ndim == 2 and rows.shape[1] == 1 and columns.shape[0] == 1:
            self._scatter_outer(rows[:, 0], columns[0], values)
            return
        for row_map, column_map, weights in self.routes:
            rr, cc = np.broadcast_arrays(row_map[rows], column_map[columns])
            keep = (rr >= 0) & (cc >= 0)
            if np.any(keep):
                scaled = np.broadcast_to(values, keep.shape)
                if weights is not None:
                    scaled = scaled * np.broadcast_to(weights[rows], keep.shape)
                np.add.at(self.matrix, (rr[keep], cc[keep]), scaled[keep])

    def scatter_add_columns(self, rows, columns, values):
        """scatter_add(rows[:, None], columns[None, :, b], values[b]) for each b in order.

        Distinct routes of one destination never write the same entry (they differ
        in equation rows or in source-side columns), so one pass per route keeps
        every entry's sum order.
        """
        rows, columns = np.asarray(rows), np.asarray(columns)
        width = columns.shape[1]
        values = np.broadcast_to(values, (width, len(rows), len(columns)))
        if type(self).scatter_add is not SystemScatter.scatter_add:
            for b in range(width):
                self.scatter_add(rows[:, None], columns[None, :, b], values[b])
            return
        self.scatter_tile(rows, columns, values)

    def scatter_tile(self, rows, columns, values, scale=None):
        """scatter_add_columns(rows, columns, values * scale[None]), without the product.

        ``values`` is a (width, rows, columns) accumulator view -- strided or
        transposed planes are fine -- and ``scale`` an optional real or complex
        (rows, columns) factor. The native route forms (value*scale)*weight per
        entry exactly as numpy would and adds in the same order, so the sums equal
        the numpy route below bit for bit.
        """
        rows, columns = np.asarray(rows), np.asarray(columns)
        width = columns.shape[1]
        if type(self).scatter_add is not SystemScatter.scatter_add:
            block = values if scale is None else values * scale[None]
            for b in range(width):
                self.scatter_add(rows[:, None], columns[None, :, b], block[b])
            return
        matrix = self.matrix
        from ghost_backend.twod.assembly.native.far import scatter_tile, scatter_columns, tile_library
        for row_map, column_map, weights in self.routes:
            if scatter_tile(matrix, rows, columns, row_map, column_map, values, scale, weights):
                continue
            block = np.broadcast_to(values, (width, len(rows), len(columns)))
            if scale is not None:
                block = block * scale[None]
            if tile_library() is None:
                # An older library without the fused tile scatter: weight with
                # numpy and use its plain column scatter (same sums).
                weighted = block if weights is None else block * weights[rows][None, :, None]
                if scatter_columns(matrix, rows, columns, row_map, column_map, weighted):
                    continue
            ri = np.flatnonzero(row_map[rows] >= 0)
            if not len(ri):
                continue
            cc = column_map[columns].T
            keep = cc >= 0
            if not keep.any():
                continue
            r = row_map[rows[ri]]
            if len(ri) != len(rows):
                block = block[:, ri, :]
            if weights is not None:
                block = block * weights[rows[ri]][None, :, None]
            keep = np.broadcast_to(keep[:, None, :], block.shape)
            if matrix.flags.f_contiguous:
                index = r[None, :, None] + cc[:, None, :] * matrix.shape[0]
                np.add.at(matrix.reshape(-1, order='F'), index[keep], block[keep])
            elif matrix.flags.c_contiguous:
                index = r[None, :, None] * matrix.shape[1] + cc[:, None, :]
                np.add.at(matrix.reshape(-1), index[keep], block[keep])
            else:
                rr = np.broadcast_to(r[None, :, None], block.shape)
                np.add.at(matrix, (rr[keep], np.broadcast_to(cc[:, None, :], block.shape)[keep]), block[keep])

    @assembly_component('near_scatter')
    def scatter_pairs(self, rows, columns, values):
        """scatter_add(rows[p][:, None], columns[p][None, :], values[p]) for every p in order.

        ``rows``/``columns`` are (pairs, width) node ids and ``values`` the
        (pairs, width, width) element blocks. One ufunc.at per route visits the
        entries in the same (pair, a, b) order as the per-pair calls, so every
        destination entry receives the same sum in the same order.
        """
        rows, columns = np.asarray(rows), np.asarray(columns)
        if not len(rows):
            return
        if type(self).scatter_add is not SystemScatter.scatter_add:
            for p in range(len(rows)):
                self.scatter_add(rows[p][:, None], columns[p][None, :], values[p])
            return
        matrix = self.matrix
        for row_map, column_map, weights in self.routes:
            r, c = row_map[rows], column_map[columns]
            keep = (r >= 0)[:, :, None] & (c >= 0)[:, None, :]
            if not keep.any():
                continue
            block = values if weights is None else values * weights[rows][:, :, None]
            if matrix.flags.f_contiguous:
                index = r[:, :, None] + c[:, None, :] * matrix.shape[0]
                np.add.at(matrix.reshape(-1, order='F'), index[keep], block[keep])
            elif matrix.flags.c_contiguous:
                index = r[:, :, None] * matrix.shape[1] + c[:, None, :]
                np.add.at(matrix.reshape(-1), index[keep], block[keep])
            else:
                rr, cc = np.broadcast_arrays(r[:, :, None], c[:, None, :])
                np.add.at(matrix, (rr[keep], cc[keep]), block[keep])

    def _scatter_outer(self, rows, columns, values):
        """rows x columns tiles: same sums as scatter_add, without per-entry index arrays."""
        values = np.broadcast_to(values, (len(rows), len(columns)))
        for row_map, column_map, weights in self.routes:
            ri = np.flatnonzero(row_map[rows] >= 0)
            if not len(ri):
                continue
            ci = np.flatnonzero(column_map[columns] >= 0)
            if not len(ci):
                continue
            r, c = row_map[rows[ri]], column_map[columns[ci]]
            if len(ri) == len(rows) and len(ci) == len(columns):
                block = values if weights is None else values * weights[rows][:, None]
            else:
                block = values[np.ix_(ri, ci)]
                if weights is not None:
                    block = block * weights[rows[ri]][:, None]
            matrix = self.matrix
            # One-dimensional ufunc.at over a contiguous view is several times
            # faster than 2-D fancy indexing and adds duplicates in the same order.
            if matrix.flags.f_contiguous:
                np.add.at(matrix.reshape(-1, order='F'), (r[:, None] + c[None, :] * matrix.shape[0]).ravel(), block.ravel())
            elif matrix.flags.c_contiguous:
                np.add.at(matrix.reshape(-1), (r[:, None] * matrix.shape[1] + c[None, :]).ravel(), block.ravel())
            else:
                np.add.at(matrix, (r[:, None], c[None, :]), block)


class MatrixDestination(SystemScatter):
    """Accumulate into an owned full operator or a non-contiguous A block.

    One identity route with unit weight (``None``): tiles go through the same
    native scatter as system routes, with no weight product and the same
    per-entry order as a direct ufunc.at into the matrix.
    """
    def __init__(self, matrix, node_count):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (node_count, node_count):
            raise ValueError('Operator destination must match the mesh nodes.')
        if matrix.dtype != np.complex128 or not matrix.flags.writeable:
            raise ValueError('Operator destination must be writable complex128 storage.')
        ids = np.arange(node_count)
        super().__init__(matrix, node_count, ids, ids, [])
        identity = np.arange(node_count, dtype=np.int64)
        self._maps = identity, identity
        self.routes = [(identity, identity, None)]


def compact_destination(operator):
    """A scatter view of CompactOperator storage (identity-weighted route)."""
    view = SystemScatter(operator.values, len(operator.row_map), [], [], [(operator.row_map, operator.column_map, None)])
    view.row_ids, view.column_ids = operator.row_ids, operator.column_ids
    view._maps = operator.row_map, operator.column_map
    return view


def scatter_target(destination, node_count):
    """Scatter interface of an operator output: system routes, compact or dense storage."""
    if isinstance(destination, SystemScatter):
        return destination
    if isinstance(destination, CompactOperator):
        return compact_destination(destination)
    return MatrixDestination(destination, node_count)


def robin_outputs(matrix, mesh, requests):
    n = len(mesh.nodes)
    columns, weights = np.arange(n), np.ones(n, complex)
    outputs = []
    for srows, krows, coefficients in requests:

        if coefficients is None:
            matrix[np.asarray(srows, int)] = 0
        pair = []
        for rows in (srows, krows):
            row_map = np.full(n, -1, int)
            row_map[rows] = rows
            routes = [(row_map, columns, weights)] if len(rows) else []
            pair.append(SystemScatter(matrix, n, rows, columns, routes))
        outputs.append(tuple(pair))
    return outputs


def multi_outputs(matrix, mesh, layout, k, requests):
    """One kernel request may feed distinct media with identical wavenumber."""
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.constants import EPS
    n = len(mesh.nodes)
    ifaces, dofs, regions = layout['ifaces'], layout['dof_map'], layout['region_props']
    results = []
    for request in requests:
        source = request['source']
        iface = ifaces[source]
        s_routes, k_routes = [], []
        for rid, mis in layout['region_ifaces'].items():
            if regions[rid]['k'] != k or source not in mis:
                continue
            col = np.full(n, -1, np.int64)
            offset, count = dofs[source, 'minus' if iface['r_m'] == rid else 'plus']
            col[iface['nodes']] = np.arange(offset, offset+count)
            sr, kr = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
            sw, kw = np.zeros(n, complex), np.zeros(n, complex)
            for observer in mis:
                target = ifaces[observer]
                nodes = np.asarray(target['nodes'])
                rm, rp = target['r_m'], target['r_p']
                if request['kind'] == 'weighted' and observer != request['observer']:
                    continue
                if rm < 0 or rp < 0:
                    offset, count = dofs[observer, 'plus' if rm < 0 else 'minus']
                    dest = np.arange(offset, offset+count)
                    pec = np.abs(target['robin_alpha']) <= EPS if layout['polarization'] == 'TM' else np.zeros(count, bool)
                    if request['kind'] == 'weighted':
                        sr[nodes[~pec]], sw[nodes[~pec]] = dest[~pec], 1
                    else:
                        sr[nodes[pec]], sw[nodes[pec]] = dest[pec], 1
                        kr[nodes[~pec]], kw[nodes[~pec]] = dest[~pec], 1
                elif request['kind'] == 'plain':
                    flux, count = dofs[observer, 'minus']
                    trace, _ = dofs[observer, 'plus']
                    sr[nodes], kr[nodes] = np.arange(trace, trace+count), np.arange(flux, flux+count)
                    sw[nodes] = 1 if rid == rm else -1
                    kw[nodes] = 1 if rid == rm else -_inverse_beta(layout, target, layout['polarization'])
            if np.any(sr >= 0):
                s_routes.append((sr, col, sw))
            if np.any(kr >= 0):
                k_routes.append((kr, col, kw))
        results.append((SystemScatter(matrix, n, sorted(request['s_rows']), iface['nodes'], s_routes),
                        SystemScatter(matrix, n, sorted(request['k_rows']), iface['nodes'], k_routes)))
    return results


def assemble_multi(mesh, infos, pol, obs_order, src_order, destination=None):
    system = _prepare_system(mesh, infos, pol, destination)
    _traverse(mesh, [system], obs_order, src_order)
    return system[0], system[1]


def assemble_pair(mesh, infos, obs_order, src_order, te_layout=None):
    """TE and TM systems of one mesh from ONE kernel traversal per wavenumber.

    The regional layout takes nothing polarization-specific from ``infos``
    (``_inverse_beta`` and the Robin law are derived from the media and the
    polarization), and the fused engine applies masks and routes to finished
    tile accumulators, so both polarizations' requests can share every Green
    and normal-derivative evaluation. Each matrix is what ``assemble_multi``
    builds alone; only the second traversal is saved.
    """
    systems = [_prepare_system(mesh, infos, 'TE', layout=te_layout), _prepare_system(mesh, infos, 'TM')]
    _traverse(mesh, systems, obs_order, src_order)
    return tuple((matrix, layout) for matrix, layout, _ in systems)


def _prepare_system(mesh, infos, pol, destination=None, layout=None):
    """Owned matrix with its mass (jump) terms, layout, and combined D/W outputs."""
    import ghost_backend.twod.solver as rcs
    from ghost_backend.twod.formulations.regions import (
        build_layout,
        _sparse_mass,
        _inverse_beta,
    )
    layout = build_layout(mesh, infos, pol) if layout is None else layout
    matrix = (np.zeros((layout['n_dof'], layout['n_dof']), complex, order='F')
              if destination is None else destination)
    if destination is not None:
        if matrix.shape != (layout['n_dof'],layout['n_dof']):
            raise ValueError('Reusable regional matrix has a different DOF layout.')
        matrix.fill(0)
    mass = _sparse_mass(mesh)
    for mi, interface in enumerate(layout['ifaces']):
        nodes = interface['nodes']
        local = mass[nodes, :][:, nodes].tocoo()
        rm, rp = interface['r_m'], interface['r_p']
        if rm < 0 or rp < 0:
            offset, _ = layout['dof_map'][mi, 'plus' if rm < 0 else 'minus']
            keep = np.ones(local.nnz, bool)
            if pol == 'TM':
                keep = np.abs(interface['robin_alpha'][local.row]) > rcs.EPS
            matrix[offset+local.row[keep], offset+local.col[keep]] += (.5 if rm < 0 else -.5)*local.data[keep]
        else:
            flux, _ = layout['dof_map'][mi, 'minus']
            trace, _ = layout['dof_map'][mi, 'plus']
            matrix[flux+local.row, flux+local.col] -= .5*local.data
            matrix[flux+local.row, trace+local.col] -= .5*_inverse_beta(layout, interface, pol)*local.data
    mass = None
    from ghost_backend.twod.formulations.combined_regions import fused_outputs
    return matrix, layout, fused_outputs(matrix, mesh, layout)


def _traverse(mesh, systems, obs_order, src_order):
    """One fused engine call per wavenumber for every prepared system."""
    import ghost_backend.twod.solver as rcs
    from ghost_backend.twod.formulations.regions import operator_plan
    plans = [list(operator_plan(layout)) for _, layout, _ in systems]
    wavenumbers = []
    for plan in plans:
        wavenumbers += [k for k, _ in plan if k not in wavenumbers]
    for k in wavenumbers:
        masks, double, coefficients, s_ids, k_ids, outputs, extra = [], [], [], [], [], [], []
        for (matrix, layout, combined), plan in zip(systems, plans):
            ifaces = layout['ifaces']
            empty = SystemScatter(matrix, len(mesh.nodes), [], [], [])
            for wavenumber, requests in plan:
                if wavenumber != k:
                    continue
                masks += [ifaces[r['source']]['mask'] for r in requests]
                double += [r['kind'] == 'plain' for r in requests]
                coefficients += [None if r['observer'] is None else
                                 ifaces[r['observer']]['robin_alpha_elements'] for r in requests]
                s_ids += [(sorted(r['s_rows']), ifaces[r['source']]['nodes']) for r in requests]
                k_ids += [(sorted(r['k_rows']), ifaces[r['source']]['nodes']) for r in requests]
                outputs += multi_outputs(matrix, mesh, layout, k, requests)
                extra += [combined.get((k, r['source']), (empty, empty)) if r['kind'] == 'plain'
                          else (empty, empty) for r in requests]
        rcs._assemble_linear_operator_matrices_multi(mesh, k, True, masks,
            obs_order=obs_order, src_order=src_order, compute_double_layer_many=double,
            single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=s_ids, double_layer_output_node_ids_many=k_ids,
            operator_outputs=outputs, additional_operator_outputs=extra)
