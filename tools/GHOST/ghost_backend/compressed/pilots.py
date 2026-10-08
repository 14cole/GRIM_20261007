"""Bounded, run-local reuse of exactly verified admission sample tiles."""
import hashlib
import pickle
import numpy as np

CACHE_BYTES = 32 * 1024**2


def key(mesh, infos, polarization, kind, k0=0., layer=None, obs_order=8, src_order=8,
        far_order_floors=None, cut='default'):
    from ghost_backend.twod.assembly.session import system_key
    from ghost_backend.execution.options import option
    from ghost_backend.twod import operators
    kind = {'te_robin': 'robin', 'single_dielectric': 'dielectric',
            'mixed_sheet_pec': 'sheet', 'thin_dielectric_layer': 'thin'}.get(kind, kind)
    # Regional info contains all evaluated wavenumbers; its runtime entry does
    # not need a separate vacuum wavenumber argument.
    frequency = 0. if kind == 'multi_region' else complex(k0)
    # Both quadrature grading and tile layout can change completed coefficients.
    # A nested explicit execution scope must not consume another scope's pilot.
    quadrature = (int(option('assembly_tile', operators._ASSEMBLY_TILE)),
                  int(option('far_quadrature_order', operators._FAR_QUAD_ORDER)),
                  bool(option('far_grading', operators._FAR_GRADED)),
                  bool(operators._NATIVE_FAR))
    floors = tuple(sorted((complex(k).real, complex(k).imag, int(order))
                          for k, order in (far_order_floors or {}).items()))
    if cut == 'default':
        cut = 32. if kind == 'multi_region' else None
    cut = None if cut is None else float(cut)
    identity = (system_key(mesh, infos or [], 'pilot_' + kind, obs_order, src_order),
                polarization, frequency, layer, quadrature, floors, cut)
    return hashlib.sha256(pickle.dumps(identity, protocol=4)).digest()


def _state():
    from ghost_backend.twod.assembly.session import current_session
    session = current_session()
    if session is None:
        return None
    if not hasattr(session, 'pilot_tiles'):
        session.pilot_tiles = {}
        session.pilot_bytes = 0
    return session


def _size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes + 128
    if isinstance(value, (tuple, list)):
        return 64 + sum(_size(item) for item in value)
    return 32


def save(identity, rows, cols, compressed):
    if identity is None:
        return
    session = _state()
    if session is None:
        return
    record = (rows.copy(), cols.copy(), compressed)
    size = _size(record)
    if size > CACHE_BYTES:
        return
    token = (identity, compressed[0], compressed[1])
    previous = session.pilot_tiles.pop(token, None)
    if previous is not None:
        session.pilot_bytes -= previous[0]
    while session.pilot_tiles and session.pilot_bytes + size > CACHE_BYTES:
        first = next(iter(session.pilot_tiles))
        session.pilot_bytes -= session.pilot_tiles.pop(first)[0]
    session.pilot_tiles[token] = (size, record)
    session.pilot_bytes += size


def take(identity, operator):
    """Transfer ownership only when both physical identity and DOF lists match."""
    session = _state()
    if identity is None or session is None or operator.compression != 'qr' or operator.tolerance != 1e-14:
        return {}
    result = {}
    for token in list(session.pilot_tiles):
        if token[0] != identity:
            continue
        size, (rows, cols, value) = session.pilot_tiles.pop(token)
        session.pilot_bytes -= size
        i, j = token[1:]
        if (i < len(operator.groups) and j < len(operator.groups)
                and np.array_equal(rows, operator.groups[i]) and np.array_equal(cols, operator.groups[j])):
            result[i, j] = value
    return result


def attach(oracle, mesh, infos, pol, kind, k0=0., layer=None, obs_order=8, src_order=8):
    oracle.pilot_identity = key(mesh, infos, pol, kind, k0, layer, obs_order, src_order,
                               getattr(oracle, 'far_order_floors', None), getattr(oracle, 'cut', None))
    from ghost_backend.twod.assembly.kernels import mesh_key
    topology = tuple((i.minus_region,i.plus_region,i.bc_kind,i.seg_type) for i in infos or [])
    layout = getattr(oracle,'layout',{})
    identity = (mesh_key(mesh),topology,pol,kind,oracle.n,
                layout.get('dof_map') if isinstance(layout,dict) else None,
                layer,obs_order,src_order)
    oracle.recycling_identity = hashlib.sha256(pickle.dumps(identity,protocol=4)).digest()


def assembled_partner(mesh, infos, pol, kind, dofs):
    """Actual completed TM payload replaces another expensive sampling pass."""
    from ghost_backend.twod.assembly.session import current_session, system_key
    session = current_session()
    if session is None or pol != 'TM' or session.pending is None:
        return None
    names = {'multi_region': 'compressed_region', 'te_robin': 'compressed_robin',
             'robin': 'compressed_robin', 'single_dielectric': 'compressed_dielectric',
             'sheet': 'compressed_sheet', 'mixed_sheet_pec': 'compressed_sheet'}
    name = names.get(kind)
    if name is None:
        return None
    expected = system_key(mesh, infos or [], name, 8, 8)
    saved, value = session.pending
    # Native keys carry an additional layer field. Thin layers are deliberately
    # excluded because their material identity is not present in infos.
    if kind != 'multi_region':
        expected = (expected, None)
    if saved != expected:
        return None
    operator = value[0] if kind == 'multi_region' else value
    if getattr(operator, 'n', None) != dofs or not hasattr(operator, 'evidence'):
        return None
    return dict(method='assembled_partner_payload', operator_bytes=operator.bytes,
                operator_allowance_bytes=operator.bytes, sampled=False, samples=0,
                reused_assembled_partner=True, seconds=0.)
