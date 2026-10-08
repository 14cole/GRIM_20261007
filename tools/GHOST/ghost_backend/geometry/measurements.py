"""Measurements on stored line primitives, in geometry units.

No joining edges or contact tolerance are invented: positive clearances remain
positive. Local measurements project an anchored point onto a picked primitive;
minimum row clearance includes shared endpoints.
"""
import math


def _point(value):
    result = tuple(float(component) for component in value)
    if len(result) != 2 or not all(math.isfinite(v) for v in result):
        raise ValueError("Measurement points must contain two finite coordinates.")
    return result


def _primitive(value):
    result = tuple(float(component) for component in value)
    if len(result) != 4 or not all(math.isfinite(v) for v in result):
        raise ValueError("Measurement primitives must contain four finite coordinates.")
    return result


def _project(point, primitive):
    a, b = primitive[:2], primitive[2:]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if not math.isfinite(length):
        raise ValueError("Geometry coordinate range is too large to measure.")
    if length == 0:
        return a, math.hypot(point[0] - a[0], point[1] - a[1])
    ux, uy = dx / length, dy / length
    along = math.fsum(((point[0] - a[0]) * ux, (point[1] - a[1]) * uy))
    if along <= 0:
        foot = a
    elif along >= length:
        foot = b
    else:
        foot = (a[0] + along * ux, a[1] + along * uy)
    return foot, math.hypot(point[0] - foot[0], point[1] - foot[1])


def point_to_primitive(point, primitive):
    """Return nearest point and distance, clamping to the finite primitive."""
    return _project(_point(point), _primitive(primitive))


def _cross(first, second):
    return math.fsum((first[0] * second[1], -first[1] * second[0]))


def _closest(first, second):
    a, b, c, d = first[:2], first[2:], second[:2], second[2:]
    candidates = []
    for point in (a, b):
        target, distance = _project(point, second)
        candidates.append((point, target, distance))
    for target in (c, d):
        point, distance = _project(target, first)
        candidates.append((point, target, distance))
    best = min(candidates, key=lambda result: (result[2], result[0], result[1]))
    if best[2] == 0:
        return best
    u, v = (b[0] - a[0], b[1] - a[1]), (d[0] - c[0], d[1] - c[1])
    u_scale, v_scale = max(map(abs, u)), max(map(abs, v))
    if u_scale == 0 or v_scale == 0:
        return best
    u = (u[0] / u_scale, u[1] / u_scale)
    v = (v[0] / v_scale, v[1] / v_scale)
    delta = (c[0] - a[0], c[1] - a[1])
    determinant = _cross(u, v)
    if determinant == 0:
        if _cross(delta, u) == 0:
            for point in sorted((a, b, c, d)):
                if all(min(start[k], end[k]) <= point[k] <= max(start[k], end[k])
                       for start, end in ((a, b), (c, d)) for k in (0, 1)):
                    return point, point, 0.0
        return best
    first_offset = _cross(delta, v) / determinant
    second_offset = _cross(delta, u) / determinant
    if 0 <= first_offset <= u_scale and 0 <= second_offset <= v_scale:
        point = (a[0] + first_offset * u[0], a[1] + first_offset * u[1])
        return point, point, 0.0
    return best


def closest_primitive_points(first, second):
    """Return points on two primitives and their distance (zero at contact)."""
    return _closest(_primitive(first), _primitive(second))


def iter_segment_primitives(segment):
    """Yield primitive indices and flat endpoint pairs without bridging gaps."""
    if len(segment.x) != len(segment.y) or len(segment.x) % 2:
        raise ValueError("Measurement requires matching, paired X and Y coordinates.")
    for offset in range(0, len(segment.x), 2):
        yield offset // 2, _primitive((segment.x[offset], segment.y[offset],
                                      segment.x[offset + 1], segment.y[offset + 1]))


def _bounds(primitive):
    return (min(primitive[0], primitive[2]), min(primitive[1], primitive[3]),
            max(primitive[0], primitive[2]), max(primitive[1], primitive[3]))


def _box_gap(first, second):
    return math.hypot(max(first[0] - second[2], second[0] - first[2], 0.0),
                      max(first[1] - second[3], second[1] - first[3], 0.0))


def closest_segment_points(first, second, checkpoint=None):
    """Minimum distance between two paired rows, or None if either is empty.

    Returns first_point, second_point, distance and both primitive indices.
    The optional checkpoint can raise InterruptedError to cancel large scans.
    """
    if checkpoint is not None:
        checkpoint()
    first_items = [(index, primitive, _bounds(primitive))
                   for index, primitive in iter_segment_primitives(first)]
    second_items = [(index, primitive, _bounds(primitive))
                    for index, primitive in iter_segment_primitives(second)]
    best = None
    count = 0
    for first_index, first_primitive, first_bounds in first_items:
        for second_index, second_primitive, second_bounds in second_items:
            count += 1
            if checkpoint is not None and count % 256 == 0:
                checkpoint()
            if best is not None and _box_gap(first_bounds, second_bounds) > best['distance']:
                continue
            first_point, second_point, distance = _closest(first_primitive, second_primitive)
            if best is None or distance < best['distance']:
                best = dict(first_point=first_point, second_point=second_point,
                            distance=distance, first_primitive=first_index,
                            second_primitive=second_index)
                if distance == 0:
                    if checkpoint is not None:
                        checkpoint()
                    return best
    if checkpoint is not None:
        checkpoint()
    return best
