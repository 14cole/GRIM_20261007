"""Bounded polynomial mesh candidates on the user's original primitives."""
import math
import numpy as np

# Below this many P1 reference panels the linear basis is kept. Certified dense
# solves at 3 GHz with hp forced: a 64-gon (640 reference panels, 192 elements)
# took 1.1 s against 3.1 s, a 16-gon of 624 panels 2.0 s against 5.6 s.
MIN_AUTOMATIC_REFERENCE_PANELS = 512


def point_key(point):
    return tuple(int(round(float(v) / 1e-9)) for v in point)


def primitive_key(segment, a, b):
    ends = sorted((point_key(a), point_key(b)))
    return '{}:{},{}:{},{}'.format(segment, *ends[0], *ends[1])


class ProtectedVertices(set):
    """Keys of protected vertices; ``at(point)`` also finds copies within the node weld tolerance."""

    def __init__(self, keys=(), welder=None):
        super().__init__(keys)
        self.welder = welder

    def at(self, point):
        if self.welder is None:
            return point_key(point) in self
        key = self.welder.lookup(point)
        return key is not None and key in self


def protected_vertices(snapshot, scale):
    """Protect geometric corners, open ends and junctions, including imports.

    Endpoints are welded like mesh nodes: two copies of one vertex that
    straddle a 1e-9 m rounding boundary are one vertex, not two open ends.
    """
    from ghost_backend.twod.geometry import _NodeWelder
    welder = _NodeWelder()
    incident = {}
    for segment in snapshot.get('segments', []):
        for pair in segment.get('point_pairs', []):
            a = np.array([float(pair['x1']), float(pair['y1'])]) * scale
            b = np.array([float(pair['x2']), float(pair['y2'])]) * scale
            direction = b-a
            length = np.linalg.norm(direction)
            if length <= 0: continue
            incident.setdefault(welder.key(a), []).append(direction/length)
            incident.setdefault(welder.key(b), []).append(-direction/length)
    cosine = -math.cos(math.radians(15.))
    return ProtectedVertices((key for key, directions in incident.items()
                              if len(directions) != 2 or np.dot(*directions) > cosine), welder)


def panel_parameters(snapshot, segment, a, b, base_count, fixed_count, refinement, protected,
                     max_panels=None):
    """Coarsen wavelength-based P1 reference counts, preserving explicit counts."""
    key = primitive_key(segment, a, b)
    coarsening = float(snapshot.get('_2d_hp_coarsening', 1.))
    multiplier = float(snapshot.get('_2d_hp_refinements', {}).get(key, 1.))
    if not math.isfinite(coarsening) or not 1 <= coarsening <= 8:
        raise ValueError('Invalid adaptive mesh coarsening factor.')
    if not math.isfinite(multiplier) or not 1 <= multiplier <= 64:
        raise ValueError('Invalid adaptive primitive refinement factor.')
    count = max(1, int(math.ceil(base_count * multiplier / (1 if fixed_count else coarsening))))
    if refinement > 1:
        count = max(count + 1, int(math.ceil(count * refinement)))
    if max_panels is not None and count > int(max_panels):
        raise ValueError('Discretization exceeds the configured panel limit.')
    t = np.linspace(0., 1., count + 1)
    if isinstance(protected, ProtectedVertices):
        left, right = protected.at(a), protected.at(b)
    else:
        left, right = point_key(a) in protected, point_key(b) in protected
    if left and right: t = .5 * (1 - np.cos(np.pi*t))
    elif left: t = t**1.7
    elif right: t = 1 - (1-t)**1.7
    points = a[None, :] + t[:, None] * (b-a)[None, :]
    # The ends are the primitive's own vertices: a + 1.0*(b - a) can miss b in
    # the last place and split the shared vertex from its neighbour's copy.
    points[0] = a
    points[-1] = b
    return key, points


def eligible_snapshot(snapshot, materials, frequencies=None, scale=1., mesh_reference=None,
                      coarsening=None):
    from ghost_backend.twod.formulations.thin_layer import ThinLayerDefinition
    from ghost_backend.twod.geometry import ImpedanceTaper
    if any(isinstance(value, ThinLayerDefinition) for value in materials.impedance_models.values()):
        return False, 'Thin-layer asymptotic formulation retains its qualified linear discretization.'
    if any(isinstance(value, ImpedanceTaper) and value.kind != 'constant'
           for value in materials.impedance_models.values()):
        return False, 'Spatially varying impedance retains h refinement to verify material sampling.'
    properties = [list(segment.get('properties', [])) for segment in snapshot.get('segments', [])]
    if all(len(row) > 1 and float(row[1] or 0) > 0 for row in properties):
        return False, 'Explicit fixed panel counts retain their requested discretization.'
    if frequencies is not None:
        # Count reference panels without allocating them, with the counts the
        # P1 mesh would use. Small cases cannot repay polynomial near-quadrature
        # overhead through smaller LU, and a densely drawn (faceted) input cannot
        # coarsen below one element per primitive, so its P2/P3 systems exceed
        # the P1 pair. The same rule is used in initial planning and execution.
        from ghost_backend.twod.geometry import (_reference_panel_counts,
            _mesh_wavelength_for_snapshot, _conservative_mesh_wavelength_for_frequencies)
        sizes = []
        for frequency in ([mesh_reference] if mesh_reference else frequencies):
            served = set(frequencies) | {mesh_reference} if mesh_reference else [frequency]
            wavelength = (_conservative_mesh_wavelength_for_frequencies(snapshot, materials, served)[0]
                          if mesh_reference else _mesh_wavelength_for_snapshot(snapshot, materials, frequency)[0])
            sizes.append(predicted_hp_size(
                _reference_panel_counts(snapshot, scale, wavelength, materials, served),
                initial_coarsening(snapshot) if coarsening is None else coarsening))
        reference, elements = max(sizes, default=(0, 0))
        if reference < MIN_AUTOMATIC_REFERENCE_PANELS:
            return False, 'Small reference mesh retains linear basis to avoid polynomial quadrature overhead.'
        if HP_CHECK_DEGREE * elements > HP_MAX_DOF_FRACTION * reference:
            return False, ('Drawn primitives limit coarsening: the degree-{} candidate would have {} '
                           'unknowns against {} linear reference panels.'.format(
                               HP_CHECK_DEGREE, HP_CHECK_DEGREE * elements, reference))
    return True, ''


# The hp candidate coarsens wavelength-sized counts by this factor, but never
# below one element per drawn primitive; the accuracy check runs at this degree.
HP_COARSENING = 4.
HP_RESOLVED_COARSENING = 8.
HP_RESOLVED_MIN_PANELS_PER_WAVELENGTH = 20
HP_CHECK_DEGREE = 3
# The cubic candidate may have at most this multiple of the P1 reference
# unknowns (at most one element per two reference panels; the P2/P3 LU flops
# then stay within the certified P1 pair's). Certified dense solves at 3 GHz:
# a 128-gon (one element per 2.5 panels) took 1.5 s with hp against 3.1 s, a
# 1024-gon that cannot coarsen (one element per panel) 11.3 s against 6.9 s.
HP_MAX_DOF_FRACTION = 1.5


def initial_coarsening(snapshot):
    """Spend less on the first candidate only for adequately sampled inputs.

    Total panel count alone does not establish wavelength resolution: a large
    sparse request can pass the size gate and still alias both polynomial
    candidates. Explicit counts are never coarsened. Failed aggressive
    candidates retry the existing factor-four controller before P1 fallback.
    """
    from ghost_backend.twod.constants import DEFAULT_PANELS_PER_WAVELENGTH
    from ghost_backend.twod.geometry import _segment_mesh_flags
    densities = []
    for segment in snapshot.get('segments', []):
        count = _segment_mesh_flags(segment)[1]
        if count <= 0:
            densities.append(abs(count) if count else DEFAULT_PANELS_PER_WAVELENGTH)
    if densities and min(densities) >= HP_RESOLVED_MIN_PANELS_PER_WAVELENGTH:
        return HP_RESOLVED_COARSENING
    return HP_COARSENING


def predicted_hp_size(counts, coarsening=None):
    """(P1 reference panels, hp elements) for per-primitive (count, explicit) records."""
    reference = sum(count for count, _ in counts)
    coarsening = HP_COARSENING if coarsening is None else coarsening
    elements = sum(count if explicit else max(1, int(math.ceil(count / coarsening)))
                   for count, explicit in counts)
    return reference, elements


def candidate_meshes(snapshot, materials, factor, adaptive, frequencies=None, scale=1., mesh_reference=None):
    """The initial solve/check meshes shared by desktop and HPC forecasting.

    Later error-driven candidates are admitted again on the executing node;
    the worker cannot expand its scheduler memory reservation.
    """
    import copy
    from ghost_backend.runs.quality import scale_snapshot_panel_density
    if factor > 1 and adaptive and eligible_snapshot(snapshot, materials, frequencies, scale, mesh_reference)[0]:
        candidate = copy.deepcopy(snapshot)
        candidate['_2d_hp_coarsening'] = initial_coarsening(snapshot)
        candidate['_2d_hp_refinements'] = {}
        return [('base', candidate, 2), ('fine', candidate, 3)]
    records = [('base', snapshot, 1)]
    if factor > 1:
        fine = scale_snapshot_panel_density(snapshot, factor)
        fine['_2d_certification_refinement_factor'] = factor
        fine['_2d_certification_base_segment_n'] = [
            (list(segment.get('properties', [])) + [0, 0])[1] for segment in snapshot['segments']]
        records.append(('fine', fine, 1))
    return records
