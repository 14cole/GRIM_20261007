"""Geometry-only helpers for material fills and readable preview overlays.

The preview works from individual primitives, not inferred links between rows.
No Qt, Matplotlib, or optional polygon package is needed here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence

Point = tuple[float, float]


def _area2(points: Sequence[Point]) -> float:
    # Subtract an origin to avoid cancellation for small cells far from (0, 0).
    ox, oy = points[0]
    return math.fsum((a[0] - ox) * (b[1] - oy) - (b[0] - ox) * (a[1] - oy)
                     for a, b in zip(points, points[1:]))


def _inside(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:]):
        if (a[1] > y) != (b[1] > y):
            intercept = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < intercept:
                inside = not inside
    return inside


def _interior_point(points: Sequence[Point], tolerance: float) -> Point:
    # A near-boundary interior point avoids incorrectly treating the outer
    # boundary of a thin nested shell as contained in its inner boundary.
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= tolerance:
            continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        for distance in (tolerance * 0.25, tolerance, length * 1e-7, length * 1e-5):
            candidate = (mx - dy * distance / length, my + dx * distance / length)
            if _inside(candidate, points):
                return candidate
    return points[0]


def build_material_faces(
    segments: Iterable,
    side_materials: Callable,
    *,
    close_axis: bool = False,
) -> list[dict]:
    """Return bounded faces with material labels, colours, and nesting depth.

    ``segments`` use the editor's paired x/y coordinate arrays.
    ``side_materials(segment)`` returns left label/colour, right label/colour
    (the left side is the positive normal side). Returned dictionaries retain
    the old preview's points/label/color/consistent/rows/depth contract and add
    area and label_point. Shared material junctions are walked as separate
    faces. Discontinuous primitives never acquire invented connecting edges.

    In BoR mode, virtual axis edges close only connected components with axis
    vertices. They supply no material labels and do not modify the geometry.
    This is a visual aid, not a validity test: crossing/duplicate primitives
    must still be reported by the validator.
    """
    primitives = []
    coordinates = []
    for row, segment in enumerate(segments):
        left, left_color, right, right_color = side_materials(segment)
        for offset in range(0, min(len(segment.x), len(segment.y)) - 1, 2):
            a = (float(segment.x[offset]), float(segment.y[offset]))
            b = (float(segment.x[offset + 1]), float(segment.y[offset + 1]))
            if not all(math.isfinite(value) for value in (*a, *b)):
                continue
            coordinates.extend((a, b))
            primitives.append((a, b, row, left, left_color, right, right_color))
    if not coordinates:
        return []
    xs, ys = zip(*coordinates)
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    tolerance = max(1e-12, 1e-9 * max(span, 1.0))
    vertices = []
    buckets = defaultdict(list)

    def vertex(point):
        cell = (math.floor(point[0] / tolerance), math.floor(point[1] / tolerance))
        for ix in range(cell[0] - 1, cell[0] + 2):
            for iy in range(cell[1] - 1, cell[1] + 2):
                for index in buckets.get((ix, iy), ()):
                    other = vertices[index]
                    if math.hypot(other[0] - point[0], other[1] - point[1]) <= tolerance:
                        return index
        index = len(vertices)
        vertices.append(point)
        buckets[cell].append(index)
        return index

    # Each edge has a twin so clockwise angular successor lookup keeps the
    # face on its left. Labels belong to directed edges, not whole chains.
    edges = []
    outgoing = defaultdict(list)

    def add_edge(start, end, row, left, left_color, right, right_color):
        if start == end:
            return
        forward = len(edges)
        edges.append((start, end, row, left, left_color))
        edges.append((end, start, row, right, right_color))
        outgoing[start].append(forward)
        outgoing[end].append(forward + 1)

    for a, b, row, left, left_color, right, right_color in primitives:
        add_edge(vertex(a), vertex(b), row, left, left_color, right, right_color)

    if close_axis:
        unseen = set(outgoing)
        axis_links = {tuple(sorted((edge[0], edge[1]))) for edge in edges}
        while unseen:
            seed = min(unseen)
            stack = [seed]
            unseen.remove(seed)
            component = []
            while stack:
                index = stack.pop()
                component.append(index)
                for edge_index in outgoing[index]:
                    other = edges[edge_index][1]
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
            axis = sorted((index for index in component
                           if abs(vertices[index][0]) <= tolerance),
                          key=lambda index: vertices[index][1])
            for first, second in zip(axis, axis[1:]):
                link = tuple(sorted((first, second)))
                if link not in axis_links:
                    add_edge(first, second, None, None, None, None, None)
                    axis_links.add(link)

    positions = {}
    for vertex_index, indices in outgoing.items():
        origin = vertices[vertex_index]
        indices.sort(key=lambda edge_index: (
            math.atan2(vertices[edges[edge_index][1]][1] - origin[1],
                       vertices[edges[edge_index][1]][0] - origin[0]),
            edge_index,
        ))
        positions.update((edge_index, index) for index, edge_index in enumerate(indices))
    successors = []
    for edge_index, edge in enumerate(edges):
        around_end = outgoing[edge[1]]
        successors.append(around_end[(positions[edge_index ^ 1] - 1) % len(around_end)])

    used = set()
    faces = []
    for first in range(len(edges)):
        if first in used:
            continue
        cycle = []
        current = first
        while current not in used:
            used.add(current)
            cycle.append(current)
            current = successors[current]
        if current != first or len(cycle) < 3:
            continue
        points = [vertices[edges[index][0]] for index in cycle]
        points.append(points[0])
        area2 = _area2(points)
        if area2 <= tolerance * tolerance:
            continue
        labels = {edges[index][3] for index in cycle if edges[index][3] is not None}
        if not labels:
            continue
        consistent = len(labels) == 1
        label = next(iter(labels)) if consistent else "?"
        colors = {edges[index][3]: edges[index][4] for index in cycle}
        faces.append({
            "points": points,
            "label": label,
            "color": colors.get(label, "red"),
            "consistent": consistent,
            "rows": sorted({edges[index][2] for index in cycle if edges[index][2] is not None}),
            "outside_materials": sorted({
                (edges[index][2], edges[index ^ 1][3]) for index in cycle
                if edges[index][2] is not None
            }),
            "area": area2 / 2,
            "label_point": _interior_point(points, tolerance),
        })
    for face in faces:
        representative = face["label_point"]
        face["depth"] = sum(1 for other in faces
                            if other is not face and other["area"] > face["area"]
                            and _inside(representative, other["points"]))
    return faces



def visible_segment_midpoint(start, end, bounds):
    """Midpoint of the visible part of a screen-space segment, or None.

    Clipping keeps long boundaries inspectable when their original midpoint
    lies outside a zoomed viewport. Canonical endpoint order makes the anchor
    unchanged when the segment direction is reversed.
    """
    a, b = sorted((tuple(map(float, start)), tuple(map(float, end))))
    x0, y0, x1, y1 = map(float, bounds)
    if not all(math.isfinite(v) for v in (*a, *b, x0, y0, x1, y1)):
        return None
    if x0 > x1 or y0 > y1:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            low = max(low, t)
        else:
            high = min(high, t)
        if low > high:
            return None
    t = (low + high) / 2
    return (min(x1, max(x0, a[0] + t * dx)),
            min(y1, max(y0, a[1] + t * dy)))

def select_normal_samples(
    candidates: Iterable[dict],
    *,
    min_spacing: float = 36.0,
    bounds: Sequence[float] | None = None,
) -> list[dict]:
    """Keep deterministic, spaced normal candidates in display coordinates.

    Every candidate provides ``midpoint=(x_px, y_px)`` and may provide ``row``,
    ``priority`` (smaller first), and ``footprint=(x0, y0, x1, y1)`` covering the
    arrow and rear tick. The optional bounds rectangle clips anchors to the
    viewport. Other candidate fields are retained for the caller. Equal-row
    candidates are sorted by position so reversing a chain retains anchors.
    """
    spacing = max(float(min_spacing), 1.0)
    valid = []
    for order, candidate in enumerate(candidates):
        x, y = candidate["midpoint"]
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if bounds is not None and not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            continue
        valid.append((candidate.get("priority", 2), candidate.get("row", 0), x, y, order, candidate))
    valid.sort(key=lambda item: item[:-1])
    accepted = []
    grid = defaultdict(list)
    footprint_grid = defaultdict(list)

    def cells(rect):
        for ix in range(math.floor(rect[0] / spacing), math.floor(rect[2] / spacing) + 1):
            for iy in range(math.floor(rect[1] / spacing), math.floor(rect[3] / spacing) + 1):
                yield (ix, iy)

    for _priority, _row, x, y, _order, candidate in valid:
        cell = (math.floor(x / spacing), math.floor(y / spacing))
        crowded = any((x - px) ** 2 + (y - py) ** 2 < spacing * spacing
                      for ix in range(cell[0] - 1, cell[0] + 2)
                      for iy in range(cell[1] - 1, cell[1] + 2)
                      for px, py in grid.get((ix, iy), ()))
        if crowded:
            continue
        footprint = candidate.get("footprint", (x, y, x, y))
        if not all(math.isfinite(value) for value in footprint):
            continue
        rectangle = (min(footprint[0], footprint[2]), min(footprint[1], footprint[3]),
                     max(footprint[0], footprint[2]), max(footprint[1], footprint[3]))
        footprint_cells = list(cells(rectangle))
        nearby = {index for key in footprint_cells for index in footprint_grid.get(key, ())}
        if any(not (rectangle[2] < existing[0] or existing[2] < rectangle[0]
                    or rectangle[3] < existing[1] or existing[3] < rectangle[1])
               for index in nearby for existing in [accepted[index][1]]):
            continue
        index = len(accepted)
        accepted.append((candidate, rectangle))
        grid[cell].append((x, y))
        for key in footprint_cells:
            footprint_grid[key].append(index)
    return [candidate for candidate, _rectangle in accepted]
