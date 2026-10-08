"""Run-local Galerkin reuse between same-panel quadratic and cubic solves.

The spaces are nested, their interpolation nodes are not. A real sparse map
P evaluates quadratic functions at cubic nodes; matching Galerkin systems
therefore satisfy A2 = P.T A3 P. Both systems still receive independent solves,
condition checks and the complete existing field-convergence comparison.
"""
from contextlib import contextmanager
import copy
import os
import numpy as np
from scipy.sparse import coo_matrix
from ghost_backend.execution.runtime import ScopedValue
from ghost_backend.execution.metrics import timed_stage

_PAIR = ScopedValue('ghost_polynomial_pair', None)
PROJECTION_WORKSPACE_BYTES = 32 * 1024**2
DENSE_SPOOL_MIN_BYTES = 64 * 1024**2


def current_pair():
    return _PAIR.get()


class PolynomialPair:
    def __init__(self, n_rhs, solver_method):
        self.n_rhs = max(1, int(n_rhs))
        self.solver_method = solver_method
        self.pending = {}
        self.plans = {}
        self.evidence = []
        self.building = False

    def close(self):
        for operator, _ in self.pending.values():
            if hasattr(operator, 'close'):
                operator.close()
        self.pending.clear()
        self.plans.clear()


@contextmanager
def polynomial_pair_scope(n_rhs, solver_method='experimental_cpu'):
    """Enable reuse explicitly with GHOST_POLYNOMIAL_PAIR=auto.

    Independent assembly is the default: measured projection/storage traffic
    can outweigh the avoided quadratic kernel work.
    """
    requested = os.environ.get('GHOST_POLYNOMIAL_PAIR', 'off').strip().lower()
    if requested not in ('auto', 'off'):
        raise ValueError('GHOST_POLYNOMIAL_PAIR must be auto or off.')
    if requested == 'off':
        yield None
        return
    pair = PolynomialPair(n_rhs, solver_method)
    with _PAIR.override(pair):
        try:
            yield pair
        finally:
            pair.close()


def _key(mesh, infos, pol, backend, obs_order, src_order):
    from ghost_backend.twod.assembly.session import system_key
    from ghost_backend.execution.options import option
    from ghost_backend.twod import operators
    quadrature = (int(option('assembly_tile', operators._ASSEMBLY_TILE)),
                  int(option('far_quadrature_order', operators._FAR_QUAD_ORDER)),
                  bool(option('far_grading', operators._FAR_GRADED)),
                  float(operators._ASSEMBLY_COMPACT_BELOW), bool(operators._NATIVE_FAR))
    return (system_key(mesh, infos, 'polynomial_pair', obs_order, src_order),
            pol, backend, quadrature)


def remember(mesh, infos, pol, operator, layout, backend, obs_order=8, src_order=8):
    pair = current_pair()
    if pair is None:
        raise RuntimeError('Polynomial operator retention requires a pair scope.')
    key = _key(mesh, infos, pol, backend, obs_order, src_order)
    if key in pair.pending:
        raise ValueError('A polynomial operator was retained twice.')
    pair.pending[key] = (operator, layout)


def take(mesh, infos, pol, backend, obs_order=8, src_order=8):
    pair = current_pair()
    if pair is None or pair.building:
        return None
    value = pair.pending.pop(_key(mesh, infos, pol, backend, obs_order, src_order), None)
    if value is not None:
        from ghost_backend.twod.assembly.dense_pair_storage import DensePairSpool
        operator, layout = value
        if isinstance(operator, DensePairSpool):
            try:
                value = operator.restore(), layout
            except InterruptedError:
                raise
            except OSError as exc:
                pair.evidence.append(dict(backend=backend, polarization=pol,
                    action='independent_assembly', fallback='joint_storage_rejection',
                    reason='retained cubic storage unavailable: ' + str(exc)))
                return None
            finally:
                operator.close()
        pair.evidence.append(dict(backend=backend, polarization=pol, action='reuse_cubic_operator'))
    return value


def retained_bytes(backend=None):
    pair = current_pair()
    if pair is None:
        return 0
    return sum(int(getattr(operator, 'bytes', getattr(operator, 'nbytes', 0)))
               for key, (operator, _) in pair.pending.items()
               if backend is None or key[2] == backend)


def discard_backend(backend):
    """A rejected solve must not retain its optional operators during retry."""
    pair = current_pair()
    if pair is None:
        return
    backend = 'compressed' if backend == 'compressed' else 'dense'
    for key in list(pair.pending):
        if key[2] == backend:
            operator, _ = pair.pending.pop(key)
            if hasattr(operator, 'close'):
                operator.close()
    pair.evidence.append(dict(backend=backend, action='discard_after_backend_rejection'))


def cubic_mesh(mesh):
    """Preserve every endpoint identity, orientation and material junction."""
    from ghost_backend.twod.geometry import LinearMesh
    from ghost_backend.twod.basis import enrich, mesh_degree
    from ghost_backend.execution.options import current_options, execution_scope
    if mesh_degree(mesh) != 2:
        raise ValueError('Polynomial pair requires a quadratic mesh.')
    endpoints = sorted({node for e in mesh.elements for node in e.node_ids[:2]})
    if endpoints != list(range(len(endpoints))):
        raise ValueError('Polynomial pair requires unchanged endpoint numbering.')
    elements = [copy.copy(e) for e in mesh.elements]
    for e in elements:
        e.node_ids = tuple(e.node_ids[:2])
    result = LinearMesh(nodes=list(mesh.nodes[:len(endpoints)]), elements=elements)
    with execution_scope(dict(current_options() or {}, basis_order=3)):
        return enrich(result)[0]


def prolongation(coarse, fine, coarse_layout, fine_layout):
    """Sparse trial/test embedding with explicit interface-law compatibility."""
    from ghost_backend.twod.basis import abscissae, values, mesh_degree
    from ghost_backend.twod.constants import EPS
    from ghost_backend.twod.formulations.combined_regions import couplings
    if (mesh_degree(coarse) != 2 or mesh_degree(fine) != 3
            or len(coarse.elements) != len(fine.elements)
            or coarse_layout['polarization'] != fine_layout['polarization']
            or coarse_layout['region_props'] != fine_layout['region_props']
            or coarse_layout['dof_map'].keys() != fine_layout['dof_map'].keys()
            or couplings(coarse, coarse_layout) != couplings(fine, fine_layout)):
        raise ValueError('Polynomial pair has incompatible Galerkin layouts.')
    local = values(abscissae(3), 2)
    node_rows = {}
    for old, new in zip(coarse.elements, fine.elements):
        if (old.node_ids[:2] != new.node_ids[:2]
                or any(not np.array_equal(getattr(old, k), getattr(new, k))
                       for k in ('p0', 'p1', 'normal', 'tangent'))
                or any(getattr(old, k) != getattr(new, k) for k in
                       ('panel_index', 'seg_type', 'ibc_flag', 'pos_mat', 'neg_mat', 'length'))):
            raise ValueError('Polynomial pair geometry or interface identity changed.')
        for node, weights in zip(new.node_ids, local):
            row = tuple((int(i), float(w)) for i, w in zip(old.node_ids, weights) if w != 0.)
            if node in node_rows and node_rows[node] != row:
                raise ValueError('Polynomial pair has an incompatible welded endpoint.')
            node_rows[node] = row
    rows, columns, data = [], [], []
    for key, (fine_offset, fine_count) in fine_layout['dof_map'].items():
        mi, _ = key
        old, new = coarse_layout['ifaces'][mi], fine_layout['ifaces'][mi]
        if (old['r_m'], old['r_p'], old['eids']) != (new['r_m'], new['r_p'], new['eids']):
            raise ValueError('Polynomial pair interface maps differ.')
        if not np.array_equal(old['robin_alpha_elements'], new['robin_alpha_elements']):
            raise ValueError('Polynomial pair material law differs.')
        if old['r_m'] < 0 or old['r_p'] < 0:
            alpha = old['robin_alpha_elements'][old['eids']]
            if len(alpha) and np.any(alpha != alpha[0]):
                # Some conductor equations route nodal Robin factors. A varying
                # law does not in general commute with a polynomial embedding.
                raise ValueError('Polynomial pair requires a uniform conductor law per interface.')
        coarse_offset, coarse_count = coarse_layout['dof_map'][key]
        positions = {node: i for i, node in enumerate(old['nodes'])}
        for i, node in enumerate(new['nodes']):
            for coarse_node, weight in node_rows[node]:
                j = positions.get(coarse_node)
                if j is None:
                    raise ValueError('Polynomial embedding crosses a material interface.')
                # Nodal choice between Dirichlet and flux equations must commute
                # with the embedding. Mixed PEC/Robin row masks need independent
                # assembly whenever an interpolated support crosses that choice.
                if (fine_layout['polarization'] == 'TM' and (old['r_m'] < 0 or old['r_p'] < 0)
                        and (abs(old['robin_alpha'][j]) <= EPS) != (abs(new['robin_alpha'][i]) <= EPS)):
                    raise ValueError('Polynomial embedding crosses a boundary-equation switch.')
                rows.append(fine_offset+i)
                columns.append(coarse_offset+j)
                data.append(weight)
        if fine_count != len(new['nodes']) or coarse_count != len(old['nodes']):
            raise ValueError('Polynomial pair degree-of-freedom map is incomplete.')
    result = coo_matrix((data, (rows, columns)),
                       shape=(fine_layout['n_dof'], coarse_layout['n_dof'])).tocsr()
    if np.any(np.diff(result.indptr) == 0):
        raise ValueError('Polynomial pair has unrepresented fine degrees of freedom.')
    return result


def prepare(mesh, infos, pol, obs_order=8, src_order=8):
    pair = current_pair()
    from ghost_backend.twod.basis import mesh_degree
    if pair is None or pair.building or mesh_degree(mesh) != 2:
        return None
    key = _key(mesh, infos, pol, 'plan', obs_order, src_order)
    if key not in pair.plans:
        from ghost_backend.twod.formulations.regions import build_layout
        try:
            fine = cubic_mesh(mesh)
            coarse_layout, fine_layout = build_layout(mesh, infos, pol), build_layout(fine, infos, pol)
            p = prolongation(mesh, fine, coarse_layout, fine_layout)
            pair.plans[key] = dict(fine_mesh=fine, coarse_layout=coarse_layout,
                                   fine_layout=fine_layout, prolongation=p)
        except ValueError as exc:
            pair.evidence.append(dict(polarization=pol, action='independent_assembly', reason=str(exc)))
            pair.plans[key] = None
    return pair.plans[key]


@timed_stage('polynomial_projection')
def project_dense(matrix, p, checkpoint=lambda: None):
    """P.T A P with a bounded column workspace, never a dense copy of P."""
    n, m = p.shape
    if matrix.shape != (n, n) or matrix.dtype != np.complex128:
        raise ValueError('Polynomial projection needs the matching complex matrix.')
    result = np.empty((m, m), complex, order='F')
    width = max(1, min(128, PROJECTION_WORKSPACE_BYTES // max(1, 32*n+16*m)))
    pt = p.T.tocsr()
    for start in range(0, m, width):
        checkpoint()
        stop = min(start+width, m)
        # Sparse-left products avoid numpy coercion and retain O(N*width) work.
        block = (p[:, start:stop].T @ matrix.T).T
        result[:, start:stop] = pt @ block
    return result


def dense_system(mesh, infos, pol, obs_order=8, src_order=8):
    """Return a shared-degree regional system, or None for ordinary assembly."""
    pair = current_pair()
    if pair is None or pair.building:
        return None
    from ghost_backend.linalg.hierarchical import factor_mode
    mode = factor_mode()
    if mode not in ('dense', 'hierarchical', 'auto'):
        return None
    previous = take(mesh, infos, pol, 'dense', obs_order, src_order)
    from ghost_backend.twod.assembly.session import current_session, system_key
    session = current_session()
    if previous is not None:
        if session is not None:
            session.save(system_key(mesh, infos, 'multi_region', obs_order, src_order), pol, previous)
        return previous
    if pol != 'TE':
        return None
    plan = prepare(mesh, infos, pol, obs_order, src_order)
    if plan is None:
        return None
    import ghost_backend.twod.solver as s
    fine = plan['fine_mesh']
    fine_bytes = 16*plan['fine_layout']['n_dof']**2
    coarse_bytes = 16*plan['coarse_layout']['n_dof']**2
    peaks = []
    for candidate in (mesh, fine):
        resources = s._dense_formulation_resources(candidate, infos, pol, sample_compression=False)
        peaks.append(s._estimate_memory_gb(resources['nodes'], False,
            system_dofs=resources['system_dofs'], n_regions=resources['n_regions'],
            operator_matrices=resources['operator_matrices'], n_rhs=pair.n_rhs,
            solver_method=pair.solver_method, formulation='multi_region', dense_resources=resources))
    spooled = fine_bytes >= DENSE_SPOOL_MIN_BYTES
    required = (max(peaks)+(PROJECTION_WORKSPACE_BYTES/1024**3) if spooled else
                max(peaks[0]+fine_bytes/1024**3,
                    peaks[1]+(coarse_bytes+PROJECTION_WORKSPACE_BYTES)/1024**3))
    limit = s._solve_memory_limit_gb()
    if required > limit:
        pair.evidence.append(dict(backend='dense', action='independent_assembly',
            reason='joint polynomial storage exceeds reservation', required_gib=required, budget_gib=limit))
        return None
    directory = None
    if spooled:
        import shutil
        from ghost_backend.execution.options import temporary_directory
        try:
            directory = temporary_directory()
            available_disk = shutil.disk_usage(directory).free
        except InterruptedError:
            raise
        except (OSError, ValueError) as exc:
            pair.evidence.append(dict(backend='dense', action='independent_assembly',
                reason='shared cubic storage unavailable: ' + str(exc)))
            return None
        if available_disk < fine_bytes+64*1024**2:
            pair.evidence.append(dict(backend='dense', action='independent_assembly',
                reason='insufficient temporary storage for shared cubic matrix',
                required_disk_bytes=fine_bytes+64*1024**2))
            return None
    from ghost_backend.twod.assembly.scatter import assemble_multi
    checkpoint = session.checkpoint if session is not None else (lambda: None)
    pair.building = True
    retained = None
    try:
        matrix, layout = assemble_multi(fine, infos, pol, obs_order, src_order)
        if spooled:
            from ghost_backend.twod.assembly.dense_pair_storage import DensePairSpool, project_spooled
            try:
                retained = DensePairSpool(matrix, directory, checkpoint)
                matrix = None  # release fine coefficients before allocating coarse A
                coarse = project_spooled(retained, plan['prolongation'], checkpoint)
            except InterruptedError:
                raise
            except OSError as exc:
                pair.evidence.append(dict(backend='dense', action='independent_assembly',
                    fallback='joint_storage_rejection',
                    reason='shared cubic storage unavailable: ' + str(exc)))
                return None
        else:
            retained = matrix
            coarse = project_dense(matrix, plan['prolongation'], checkpoint)
        remember(fine, infos, pol, retained, layout, 'dense', obs_order, src_order)
        retained = None  # pair scope now owns the array or temporary file
    finally:
        pair.building = False
        if hasattr(retained, 'close'):
            retained.close()
    result = coarse, plan['coarse_layout']
    if session is not None:
        session.save(system_key(mesh, infos, 'multi_region', obs_order, src_order), pol, result)
    pair.evidence.append(dict(backend='dense', polarization=pol, action='project_quadratic',
        fine_dofs=plan['fine_layout']['n_dof'], coarse_dofs=coarse.shape[0],
        retained_fine_bytes=0 if spooled else fine_bytes,
        retained_fine_disk_bytes=fine_bytes if spooled else 0,
        fine_storage='disk' if spooled else 'memory', projection_workspace_bytes=PROJECTION_WORKSPACE_BYTES,
        required_gib=required, budget_gib=limit))
    return result
