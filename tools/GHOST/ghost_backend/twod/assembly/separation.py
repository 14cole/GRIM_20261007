"""Geometric separation predicates shared by all boundary operators."""
import numpy as np

# A separated pair takes the fixed 16-point tensor rule unless one of these
# holds, in which case the adaptive rule (1e-9 target) integrates it:
#   centre distance < ADAPTIVE_CENTRE * scale, or
#   segment distance < ADAPTIVE_GAP * scale, or
#   an endpoint of one panel projects strictly inside the other panel at a
#   distance below ADAPTIVE_PROJECTION * scale.
# scale is the longer panel length. The projection clause closes the band where
# a panel end hovers over the other panel's interior (0.25 L above it: 1.3e-8
# K' error with the fixed rule, against the adaptive rule's 1e-9 target).
ADAPTIVE_CENTRE = .75
ADAPTIVE_GAP = .25
ADAPTIVE_PROJECTION = .5


def segment_distance(p0, p1, q0, q1):
    """Euclidean distance between closed 2-D segments; supports broadcasting.

    Endpoint projections also handle parallel and collinear segments. A proper
    crossing has zero distance even when no endpoint projects onto the crossing.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    def point_distance(x, a, b):
        edge = b-a
        denominator = np.sum(edge*edge, axis=-1)
        t = np.divide(np.sum((x-a)*edge, axis=-1), denominator,
                      out=np.zeros_like(denominator), where=denominator > 0)
        delta = x-a-np.clip(t, 0., 1.)[..., None]*edge
        return np.sum(delta*delta, axis=-1)
    squared = np.minimum.reduce((point_distance(p0,q0,q1), point_distance(p1,q0,q1),
                                 point_distance(q0,p0,p1), point_distance(q1,p0,p1)))
    u, v, w = p1-p0, q1-q0, q0-p0
    def cross(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    denominator = cross(u,v)
    safe = np.where(denominator != 0, denominator, 1.)
    t, s = cross(w,v)/safe, cross(w,u)/safe
    crossing = (denominator != 0) & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    return np.sqrt(np.where(crossing, 0., np.maximum(squared,0.)))


def projection_distance(p0, p1, q0, q1):
    """Smallest distance from an endpoint of one segment to the interior of the other.

    Only endpoints whose foot point falls strictly inside the other segment count;
    infinity where none does. Supports broadcasting.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    shape = np.broadcast_shapes(p0.shape, p1.shape, q0.shape, q1.shape)[:-1]
    best = np.full(shape, np.inf)
    for x, a, b in ((p0, q0, q1), (p1, q0, q1), (q0, p0, p1), (q1, p0, p1)):
        edge = b-a
        denominator = np.sum(edge*edge, axis=-1)
        t = np.divide(np.sum((x-a)*edge, axis=-1), denominator,
                      out=np.full(np.shape(denominator), -1.), where=denominator > 0)
        delta = x-a-t[..., None]*edge
        distance = np.sqrt(np.sum(delta*delta, axis=-1))
        inside = (t > 0.) & (t < 1.)
        best = np.minimum(best, np.where(inside, distance, np.inf))
    return best


def requires_adaptive(obs, src):
    scale = max(obs.length, src.length)
    distance = float(np.linalg.norm(obs.center-src.center))
    if distance < ADAPTIVE_CENTRE*scale:
        return True
    # The triangle inequality excludes distant pairs without projections: an
    # endpoint within d of the other panel keeps the centres within
    # (L_obs+L_src)/2 + d.
    if distance >= .5*(obs.length+src.length)+ADAPTIVE_PROJECTION*scale:
        return False
    if distance < .5*(obs.length+src.length)+ADAPTIVE_GAP*scale:
        if bool(segment_distance(obs.p0,obs.p1,src.p0,src.p1) < ADAPTIVE_GAP*scale):
            return True
    return bool(projection_distance(obs.p0,obs.p1,src.p0,src.p1) < ADAPTIVE_PROJECTION*scale)


def adaptive_pairs(p0, p1, q0, q1, center_distance, scale, lengths_p, lengths_q):
    """requires_adaptive for a list of pairs, term for term.

    Segment ends are (pairs, 2) arrays; distances, scales and lengths (pairs,).
    """
    center_distance, scale = np.asarray(center_distance, float), np.asarray(scale, float)
    half = .5*(np.asarray(lengths_p, float)+np.asarray(lengths_q, float))
    result = center_distance < ADAPTIVE_CENTRE*scale
    band = np.flatnonzero(~result & (center_distance < half+ADAPTIVE_PROJECTION*scale))
    if band.size:
        sc = scale[band]
        close = np.zeros(band.size, dtype=bool)
        gap_band = center_distance[band] < half[band]+ADAPTIVE_GAP*sc
        if np.any(gap_band):
            g = band[gap_band]
            close[gap_band] = segment_distance(p0[g], p1[g], q0[g], q1[g]) < ADAPTIVE_GAP*sc[gap_band]
        rest = ~close
        if np.any(rest):
            r = band[rest]
            close[rest] = projection_distance(p0[r], p1[r], q0[r], q1[r]) < ADAPTIVE_PROJECTION*sc[rest]
        result[band] = close
    return result


def close_pairs(p0, p1, q0, q1, center_distance, scale):
    """Tile mask of pairs needing the adaptive rule beyond the centre test.

    ``p0``/``p1`` hold the rows' segments and ``q0``/``q1`` the columns'.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    center_distance = np.asarray(center_distance, float)
    lengths_p = np.linalg.norm(p1-p0, axis=-1)[:, None]
    lengths_q = np.linalg.norm(q1-q0, axis=-1)[None, :]
    lp, lq = np.broadcast_arrays(lengths_p, lengths_q)
    result = np.zeros(center_distance.shape, dtype=bool)
    i, j = np.nonzero(center_distance < .5*(lp+lq)+ADAPTIVE_PROJECTION*scale)
    if len(i):
        sc = scale[i, j]
        gap = segment_distance(p0[i], p1[i], q0[j], q1[j]) < ADAPTIVE_GAP*sc
        projection = projection_distance(p0[i], p1[i], q0[j], q1[j]) < ADAPTIVE_PROJECTION*sc
        result[i, j] = gap | projection
    return result
