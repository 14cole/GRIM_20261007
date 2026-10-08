"""Triangle-exact geometric visibility along straight line pieces.

For a fixed look, ray/triangle barycentric coordinates and ray distance are
affine in position along a line. Clipping those inequalities finds shadow
intervals, including arbitrarily short intervals missed by point sampling.
"""
import hashlib
import numpy as np


def _cache_cost(value):
    return 128 + sum(a.nbytes for a in value if isinstance(a, np.ndarray))


def _decode(value):
    if isinstance(value[0], int):
        count = value[0]
        return tuple(np.frombuffer(a.tobytes(), dtype=a.dtype) for a in
                     (np.arange(count, dtype=np.intp), np.zeros(count), np.ones(count)))
    return value


def _union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return merged


def visible_intervals(occluder, starts, ends, direction, cancel_check=None):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("Line visibility cancelled.")
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    direction = np.asarray(direction, float)
    if (starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape
            or not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends))
            or direction.shape != (3,) or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= 1e-15):
        raise ValueError("Line visibility requires matching finite (n,3) endpoints and a nonzero look.")
    direction = direction / np.linalg.norm(direction)
    digest = hashlib.sha256()
    for array in (starts, ends, direction, np.asarray([occluder.bias])):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = digest.digest()
    with occluder._bvh_lock:
        cached = occluder._line_visibility_cache.get(key)
        if cached is not None:
            occluder._line_visibility_cache.move_to_end(key)
            return _decode(cached)
    occluder.prepare_acceleration(cancel_check=cancel_check)
    owners, lower, upper = [], [], []
    tolerance = 4e-9 * occluder.diag
    for offset in range(0, len(starts), 128):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Line visibility cancelled.")
        p = starts[offset:offset+128]
        delta = ends[offset:offset+128] - p
        blocked = [[] for _ in p]
        stack = [(1, np.arange(len(p)))]
        terms_cache = {}
        while stack:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Line visibility cancelled.")
            node, rows = stack.pop()
            # A conservative ray/box test against the segment-expanded box.
            lo = occluder._bvh_lo[node] - np.maximum(delta[rows], 0) - tolerance
            hi = occluder._bvh_hi[node] - np.minimum(delta[rows], 0) + tolerance
            near = np.full(len(rows), occluder.bias)
            far = np.full(len(rows), np.inf)
            keep = np.ones(len(rows), bool)
            for axis in range(3):
                if abs(direction[axis]) < 1e-15:
                    keep &= (p[rows, axis] >= lo[:, axis]) & (p[rows, axis] <= hi[:, axis])
                else:
                    a = (lo[:, axis]-p[rows, axis])/direction[axis]
                    b = (hi[:, axis]-p[rows, axis])/direction[axis]
                    near = np.maximum(near, np.minimum(a,b))
                    far = np.minimum(far, np.maximum(a,b))
            rows = rows[keep & (far >= near)]
            if not len(rows):
                continue
            rows = np.asarray([r for r in rows if blocked[r] != [[0.0, 1.0]]], dtype=int)
            if not len(rows):
                continue
            if node < occluder._bvh_base:
                stack.extend([(2*node, rows), (2*node+1, rows)])
                continue
            terms = occluder._leaf_direction_terms(node, direction, terms_cache)
            if terms is None:
                continue
            tri0, edge1, edge2, h, inverse, determinant_valid = terms
            s = p[rows, None, :] - tri0
            ds = delta[rows, None, :]
            u = np.einsum('rtj,tj->rt', s, h)*inverse
            du = np.einsum('rj,tj->rt', delta[rows], h)*inverse
            q = np.cross(s, edge1)
            dq = np.cross(ds, edge1)
            v = np.einsum('rtj,j->rt', q, direction)*inverse
            dv = np.einsum('rtj,j->rt', dq, direction)*inverse
            distance = np.einsum('rtj,tj->rt', q, edge2)*inverse
            dd = np.einsum('rtj,tj->rt', dq, edge2)*inverse
            low, high = np.zeros_like(u), np.ones_like(u)
            valid = np.broadcast_to(determinant_valid, u.shape).copy()
            for value, slope, strict in ((u,du,False),(v,dv,False),(1-u-v,-du-dv,False),
                                         (distance-occluder.bias,dd,True)):
                flat = slope == 0
                valid &= ~(flat & ((value <= 0) if strict else (value < -1e-12)))
                crossing = np.divide(-value, slope, out=np.zeros_like(value), where=~flat)
                low = np.maximum(low, np.where(slope > 0, crossing, 0))
                high = np.minimum(high, np.where(slope < 0, crossing, 1))
            valid &= high > low
            for j, row in enumerate(rows):
                hits = valid[j]
                if np.any(hits):
                    blocked[row] = _union(blocked[row] + list(zip(low[j,hits], high[j,hits])))
        for row, intervals in enumerate(blocked):
            position = 0.0
            for lo, hi in intervals:
                if lo > position:
                    owners.append(offset+row); lower.append(position); upper.append(lo)
                position = max(position, hi)
            if position < 1:
                owners.append(offset+row); lower.append(position); upper.append(1.0)
    result = tuple(np.frombuffer(array.tobytes(), dtype=array.dtype) for array in
                   (np.asarray(owners, dtype=np.intp), np.asarray(lower, float), np.asarray(upper, float)))
    # Fully visible lines are common. Store that fact compactly so a modest
    # look sweep fits the cache and reuses visibility across frequencies.
    cached_value = result
    if len(result[0]) == len(starts) and np.all(result[1] == 0) and np.all(result[2] == 1):
        cached_value = (len(starts), None, None)
    size = _cache_cost(cached_value)
    with occluder._bvh_lock:
        while occluder._line_visibility_cache and (occluder._line_visibility_bytes+size > 16*1024**2 or len(occluder._line_visibility_cache) >= 4096):
            _, old = occluder._line_visibility_cache.popitem(last=False)
            occluder._line_visibility_bytes -= _cache_cost(old)
        if size <= 16*1024**2:
            if key not in occluder._line_visibility_cache:
                occluder._line_visibility_cache[key] = cached_value
                occluder._line_visibility_bytes += size
    return result
