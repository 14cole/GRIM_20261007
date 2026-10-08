"""Captured geometry checks, independent of Qt widgets and plotting."""
import math
from typing import Any, Dict, List, Set, Tuple
from ghost_backend.geometry.io import ChainSpec, Segment, _weld_point_keys, check_orientation_consistency
from ghost_backend.geometry.spatial import overlapping_pairs, primitive_bounds

def _parse_mesh_n_token(token: 'Any') -> 'int':
    """Parse the geometry N field without changing its solver semantics.

    Blank or zero selects automatic 20-panels-per-wavelength meshing, a
    positive integer is an explicit panel count per primitive, and a negative
    integer selects its absolute value in panels per wavelength.
    """

    text = str(token or "").strip()
    if not text:
        return 0
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("N must be an integer.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("N must be an integer.")
    return int(value)



class GeometryAudit:
    def __init__(self, segments, ibcs, dielectrics, material_dir, checkpoint=None, mode="2d"):
        self.segments = segments
        self.ibcs = ibcs
        self.dielectrics = dielectrics
        self.material_dir = material_dir
        self.checkpoint = checkpoint or (lambda: None)
        self.mode = str(mode).strip().lower()
        kinds = {self._parse_int_token(seg.properties[0], -1) if seg.properties else -1
                 for seg in self.segments}
        self._bor_sheet_layout = self.mode == "bor" and kinds in ({1}, {1, 2})
        if self.mode not in {"2d", "bor"}:
            raise ValueError("Geometry validation mode must be '2d' or 'bor'.")

    def _parse_int_token(self, token: 'str', default: 'int' = 0) -> 'int':
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("mat."):
            text = text.split("mat.", 1)[1]
        try:
            value = float(text)
            return int(value) if math.isfinite(value) and value.is_integer() else default
        except (TypeError, ValueError, OverflowError):
            return default


    def _parse_float_token(self, token: 'str', default: 'float' = 0.0) -> 'float':
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default


    def _point_key(self, x: 'float', y: 'float', tol: 'float') -> 'Tuple[int, int]':
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))


    def _segments_intersect(
        self,
        a1: 'Tuple[float, float]',
        a2: 'Tuple[float, float]',
        b1: 'Tuple[float, float]',
        b2: 'Tuple[float, float]',
        tol: 'float',
    ) -> 'bool':
        ax1, ay1 = a1
        ax2, ay2 = a2
        bx1, by1 = b1
        bx2, by2 = b2

        min_ax, max_ax = min(ax1, ax2), max(ax1, ax2)
        min_ay, max_ay = min(ay1, ay2), max(ay1, ay2)
        min_bx, max_bx = min(bx1, bx2), max(bx1, bx2)
        min_by, max_by = min(by1, by2), max(by1, by2)
        if max_ax < min_bx - tol or max_bx < min_ax - tol:
            return False
        if max_ay < min_by - tol or max_by < min_ay - tol:
            return False

        def orient(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'float':
            return (qx - px) * (ry - py) - (qy - py) * (rx - px)

        def on_seg(px: 'float', py: 'float', qx: 'float', qy: 'float', rx: 'float', ry: 'float') -> 'bool':
            return (
                min(px, qx) - tol <= rx <= max(px, qx) + tol
                and min(py, qy) - tol <= ry <= max(py, qy) + tol
            )

        o1 = orient(ax1, ay1, ax2, ay2, bx1, by1)
        o2 = orient(ax1, ay1, ax2, ay2, bx2, by2)
        o3 = orient(bx1, by1, bx2, by2, ax1, ay1)
        o4 = orient(bx1, by1, bx2, by2, ax2, ay2)

        # `orient` is a cross product (length^2); comparing it to `tol` (a
        # length) makes the effective clearance tolerance tol/length, so thin
        # features get reported as intersections.  Scale each threshold by the
        # length of the line it is measured against to recover a true
        # perpendicular distance of `tol`.
        ta = tol * max(math.hypot(ax2 - ax1, ay2 - ay1), 1e-12)
        tb = tol * max(math.hypot(bx2 - bx1, by2 - by1), 1e-12)

        if (o1 > ta and o2 < -ta or o1 < -ta and o2 > ta) and (
            o3 > tb and o4 < -tb or o3 < -tb and o4 > tb
        ):
            return True

        if abs(o1) <= ta and on_seg(ax1, ay1, ax2, ay2, bx1, by1):
            return True
        if abs(o2) <= ta and on_seg(ax1, ay1, ax2, ay2, bx2, by2):
            return True
        if abs(o3) <= tb and on_seg(bx1, by1, bx2, by2, ax1, ay1):
            return True
        if abs(o4) <= tb and on_seg(bx1, by1, bx2, by2, ax2, ay2):
            return True
        return False


    def _ensure_prop_len(self, props: 'List[str]', n: 'int') -> 'List[str]':
        props = list(props)
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props


    def _segment_primitives(self, seg: 'Segment') -> 'List[Tuple[float, float, float, float]]':
        count = min(len(seg.x), len(seg.y))
        n_pairs = count // 2
        out: 'List[Tuple[float, float, float, float]]' = []
        for i in range(n_pairs):
            idx = 2 * i
            out.append((seg.x[idx], seg.y[idx], seg.x[idx + 1], seg.y[idx + 1]))
        return out


    def _segment_plot_xy(self, seg: 'Segment') -> 'Tuple[List[float], List[float]]':
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)

        xs: 'List[float]' = []
        ys: 'List[float]' = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            if i == 0:
                xs.append(x1)
                ys.append(y1)
            xs.append(x2)
            ys.append(y2)

        if not xs or not ys:
            return list(seg.x), list(seg.y)
        return xs, ys


    def _material_sides(self, seg):
        """Left/right media as drawn, matching the solver normal convention."""
        props = self._ensure_prop_len(seg.properties, 5)
        kind = self._parse_int_token(props[0], -1)
        pos = self._parse_int_token(props[3], 0)
        neg = self._parse_int_token(props[4], 0)
        air, pec = ("air", 0), ("PEC", 0)
        if kind == 2 and self._bor_sheet_layout:
            # BoR classifies this mixed surface as a transmitting sheet with
            # zero impedance on its PEC portions, not an opaque solid.
            return air, air
        return {
            1: (air, air), 2: (air, pec),
            3: (air, ("material", pos)),
            4: (("material", pos), pec),
            5: (("material", pos), ("material", neg)),
        }.get(kind)

    def _junction_findings(self, endpoint_hits, primitives_by_row, axis_tol, tol):
        """Compare shared material sectors, including valid triple junctions."""
        findings, issue_rows = [], set()
        for hits in endpoint_hits.values():
            self.checkpoint()
            if len(hits) < 2:
                continue
            if len(hits) == 2:
                props = [self._ensure_prop_len(self.segments[hit[0]].properties, 5) for hit in hits]
                kinds = {self._parse_int_token(item[0], -1) for item in props}
                if kinds == {1, 2} and all(
                    self._parse_int_token(item[2], 0) == 0 for item in props
                    if self._parse_int_token(item[0], -1) == 2
                ):
                    # Supported open sheet / pure-PEC transition in 2D and
                    # BoR. At degree-3 nodes retain the full sector check.
                    continue
            rays = []
            for row, pidx, end in hits:
                x1, y1, x2, y2 = primitives_by_row[row][pidx]
                sides = self._material_sides(self.segments[row])
                if sides is None or math.hypot(x2-x1, y2-y1) <= tol:
                    continue
                if end:
                    x, y, dx, dy = x2, y2, x1-x2, y1-y2
                    sides = sides[::-1]
                else:
                    x, y, dx, dy = x1, y1, x2-x1, y2-y1
                rays.append((math.atan2(dy, dx), row, sides[0], sides[1], x, y))
            rays.sort(key=lambda ray: ray[0])
            if len(rays) < 2:
                continue
            # A BoR profile only occupies rho>=0; no material sector crosses
            # the axis. Its actual surface can terminate at a pole.
            on_axis = self.mode == "bor" and all(abs(ray[4]) <= axis_tol for ray in rays)
            pairs = list(zip(rays, rays[1:]))
            if not on_axis:
                pairs.append((rays[-1], rays[0]))
            reported = set()
            for first, second in pairs:
                if first[2] == second[3]:
                    continue
                rows = tuple(sorted({first[1], second[1]}))
                if rows in reported:
                    continue
                reported.add(rows)
                def medium_name(medium):
                    return (f"material {medium[1]}" if medium[0] == "material"
                            else medium[0])
                row_labels = ", ".join(str(row+1) for row in rows)
                message = (
                    f"Rows {row_labels}: material-side mismatch at "
                    f"({first[4]:.6g}, {first[5]:.6g}); the same junction sector "
                    f"is labelled {medium_name(first[2])} and "
                    f"{medium_name(second[3])}. Check TYPE, pos_mat/neg_mat, "
                    "and segment direction, or add the missing interface."
                )
                findings.append(("ERROR", rows[0], message))
                issue_rows.update(rows)
        return findings, issue_rows

    def _nested_region_findings(self):
        """Check the shared medium across separate, nested closed contours."""
        from ghost_backend.geometry.preview import build_material_faces, _inside
        def sides(segment):
            left, right = self._material_sides(segment) or (("unknown", 0), ("unknown", 0))
            return left, "", right, ""
        faces = build_material_faces(self.segments, sides)
        findings, rows = [], set()
        def name(medium):
            return f"material {medium[1]}" if medium[0] == "material" else medium[0]
        for face in faces:
            self.checkpoint()
            parents = [parent for parent in faces
                       if parent is not face and parent["area"] > face["area"]
                       and _inside(face["label_point"], parent["points"])]
            if not parents:
                continue
            parent = min(parents, key=lambda candidate: candidate["area"])
            if not parent["consistent"]:
                continue
            expected = parent["label"]
            for row, outside in face["outside_materials"]:
                if outside == expected:
                    continue
                findings.append(("ERROR", row,
                    f"Row {row+1} '{self.segments[row].name}': nested material-side mismatch; "
                    f"the surrounding region is {name(expected)}, but this boundary "
                    f"assigns {name(outside)} outside. Check TYPE, pos_mat/neg_mat, "
                    "and segment direction."))
                rows.update(parent["rows"])
                rows.add(row)
        return findings, rows

    def _bor_findings(self, primitives_by_row):
        """Use the solver topology preflight without meshing or solving."""
        self.checkpoint()
        snapshot = {"segments": [
            {"name": seg.name, "properties": list(seg.properties),
             "point_pairs": [
                 dict(zip(("x1", "y1", "x2", "y2"), primitive))
                 for primitive in primitives_by_row[row]
             ]}
            for row, seg in enumerate(self.segments)
        ]}
        try:
            from ghost_backend.bor.dispatch import (
                _chains_from_snapshot, _classify, _prepare_bor_groups, _bor_surface_runs,
                _reject_unsupported_bor_ibc_interfaces,
            )
            _reject_unsupported_bor_ibc_interfaces(snapshot)
            chains = _chains_from_snapshot(snapshot, 1.0)
            kind = _classify(chains)
            import numpy as np
            from ghost_backend.bor.solver import _validate_solve_bor_generatrix
            groups, _, axis_tol = _prepare_bor_groups(chains, kind)
            for run, _ in _bor_surface_runs(groups, kind):
                self.checkpoint()
                points = np.vstack([run[0].pts] + [chain.pts[1:] for chain in run[1:]]).copy()
                points[np.abs(points[:, 0]) <= axis_tol, 0] = 0.0
                _validate_solve_bor_generatrix(points, "efie")
        except InterruptedError:
            raise
        except Exception as exc:
            return [("ERROR", -1, f"BoR solver geometry check: {exc}")]
        self.checkpoint()
        return [("INFO", -1, f"BoR {kind} topology passes the solver geometry preflight.")]

    @staticmethod
    def _collinear_overlap(first, second, tol):
        """Shared tips are allowed, but duplicate/overlaid length is not."""
        x1, y1, x2, y2 = first
        u1, v1, u2, v2 = second
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        if length <= tol:
            return False
        if (abs(dx*(v1-y1)-dy*(u1-x1)) > tol*length
                or abs(dx*(v2-y1)-dy*(u2-x1)) > tol*length):
            return False
        lo, hi = sorted((((u1-x1)*dx+(v1-y1)*dy)/length,
                         ((u2-x1)*dx+(v2-y1)*dy)/length))
        return min(length, hi) - max(0.0, lo) > tol


    def run(self):
        ibcs_rows = self.ibcs
        dielectric_rows = self.dielectrics
        diel_flags = {
            self._parse_int_token(row[0], 0)
            for row in dielectric_rows if row
        }
        ibc_flags = {
            self._parse_int_token(row[0], 0)
            for row in ibcs_rows if row
        }

        findings: 'List[Tuple[str, int, str]]' = []
        issue_rows: 'Set[int]' = set()

        material_dir = self.material_dir
        try:


            from ghost_backend.twod.solver import MaterialLibrary
            MaterialLibrary.from_entries(
                ibcs_rows, dielectric_rows, material_dir
            )
        except Exception as exc:
            findings.append(("ERROR", -1, f"Material definition error: {exc}"))

        for ibc_idx, row in enumerate(ibcs_rows):
            if (
                len(row) == 6
                and str(row[1]).strip().lower() == "exp"
            ):
                z_parts = [
                    self._parse_float_token(row[i], float("nan"))
                    for i in (2, 3, 4, 5)
                ]
                if all(math.isfinite(value) for value in z_parts):
                    if (
                        z_parts[0] ** 2 + z_parts[1] ** 2 == 0.0
                        or z_parts[2] ** 2 + z_parts[3] ** 2 == 0.0
                    ):
                        findings.append((
                            "WARN", -1,
                            f"IBCS row {ibc_idx + 1}: exp taper endpoints "
                            "should be nonzero; prefer linear or cosine for "
                            "PEC-limit transitions.",
                        ))

        if not self.segments:
            findings.append(("ERROR", -1, "Geometry contains no segments."))
        primitives_by_row = []
        for row, seg in enumerate(self.segments):
            self.checkpoint()
            if (len(seg.x) != len(seg.y) or len(seg.x) % 2
                    or not all(math.isfinite(value) for value in list(seg.x)+list(seg.y))):
                findings.append(("ERROR", row, f"Row {row+1} '{seg.name}': coordinates "
                                 "must be finite and form complete x/y endpoint pairs."))
                issue_rows.add(row)
                primitives_by_row.append([])
            else:
                primitives_by_row.append(self._segment_primitives(seg))
        all_points = [(primitive[i], primitive[i+1])
                      for primitives in primitives_by_row for primitive in primitives
                      for i in (0, 2)]
        xs, ys = ([point[0] for point in all_points],
                  [point[1] for point in all_points])
        span = max(max(xs)-min(xs), max(ys)-min(ys), 1e-9) if all_points else 1.0
        diag = max(math.hypot(max(xs)-min(xs), max(ys)-min(ys)), 1.0) if all_points else 1.0
        # Match solver topology tolerance, not a display-sized clearance.
        # The previous 1e-6*diagonal weld merged distinct thin-layer vertices.
        tol = max(1e-12, 1e-9 * (span if self.mode == "bor" else diag))
        axis_tol = 1e-6 * span
        endpoint_records = [(row, pidx, end)
                            for row, primitives in enumerate(primitives_by_row)
                            for pidx in range(len(primitives)) for end in (0, 1)]
        endpoint_points = [primitives_by_row[row][pidx][2*end:2*end+2]
                           for row, pidx, end in endpoint_records]
        endpoint_keys = dict(zip(endpoint_records, _weld_point_keys(endpoint_points, tol)))
        endpoint_hits = {}
        for record, key in endpoint_keys.items():
            endpoint_hits.setdefault(key, []).append(record)

        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            ibc = self._parse_int_token(props[2], 0)
            pos_mat = self._parse_int_token(props[3], 0)
            neg_mat = self._parse_int_token(props[4], 0)
            primitives = primitives_by_row[row]
            label = f"Row {row + 1} '{seg.name}'"

            for index, name in ((0, "TYPE"), (2, "IBC"), (3, "pos_mat"), (4, "neg_mat")):
                token = str(props[index]).strip()
                if token and self._parse_int_token(token, -1) < 0:
                    findings.append(("ERROR", row, f"{label}: {name} must be a nonnegative integer; "
                                     f"got '{token}'."))
                    issue_rows.add(row)

            if seg_type < 1 or seg_type > 5:
                findings.append(("ERROR", row, f"{label}: invalid TYPE '{props[0]}', expected 1..5."))
                issue_rows.add(row)

            try:
                _parse_mesh_n_token(props[1])
            except ValueError:
                findings.append((
                    "ERROR",
                    row,
                    f"{label}: N must be an integer; use 0 or blank for "
                    "automatic 20-panels-per-wavelength meshing, a positive "
                    "integer for an explicit panel count per primitive, or a "
                    "negative integer for panels per wavelength. Current "
                    f"value is '{props[1]}'.",
                ))
                issue_rows.add(row)

            if not primitives:
                findings.append(("ERROR", row, f"{label}: no line primitives found."))
                issue_rows.add(row)
                continue

            for i, (x1, y1, x2, y2) in enumerate(primitives):
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length <= tol:
                    findings.append(("ERROR", row, f"{label}: primitive {i + 1} has near-zero length."))
                    issue_rows.add(row)

            for i in range(len(primitives) - 1):
                _, _, ex, ey = primitives[i]
                nx1, ny1, nx2, ny2 = primitives[i + 1]
                d_start = ((ex - nx1) ** 2 + (ey - ny1) ** 2) ** 0.5
                d_end = ((ex - nx2) ** 2 + (ey - ny2) ** 2) ** 0.5
                if d_start > tol:
                    if d_end <= tol:
                        findings.append(
                            ("ERROR", row, f"{label}: primitive {i + 2} appears reversed relative to previous one.")
                        )
                    else:
                        findings.append(("ERROR", row, f"{label}: primitive {i + 1} and {i + 2} are not connected."))
                    issue_rows.add(row)

            sx, sy, _, _ = primitives[0]
            _, _, ex, ey = primitives[-1]
            closed = (((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5) <= tol

            if closed:
                points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
                area2 = 0.0
                for i in range(len(points) - 1):
                    x0, y0 = points[i]
                    x1, y1 = points[i + 1]
                    area2 += x0 * y1 - x1 * y0
                orient = "CCW" if area2 > 0 else "CW"
                findings.append(("INFO", row, f"{label}: closed chain, orientation {orient}."))


            else:


                start_connected = len(endpoint_hits[endpoint_keys[(row, 0, 0)]]) > 1
                end_connected = len(endpoint_hits[endpoint_keys[(row, len(primitives)-1, 1)]]) > 1
                start_axis = self.mode == "bor" and abs(sx) <= axis_tol
                end_axis = self.mode == "bor" and abs(ex) <= axis_tol
                if (start_connected or start_axis) and (end_connected or end_axis):
                    detail = ("ends continue into other segments or meet the rotation axis"
                              if self.mode == "bor" else "both ends continue into other segments")
                    findings.append(("INFO", row, f"{label}: open chain; {detail}."))
                elif self._bor_sheet_layout:
                    findings.append(("INFO", row, f"{label}: open BoR sheet; free rim is allowed."))
                else:
                    detail = ("free endpoint away from the rotation axis (x = 0)"
                              if self.mode == "bor" else "start/end do not close")
                    findings.append(("WARN", row, f"{label}: open chain ({detail})."))
                    if seg_type in {3, 4, 5}:
                        issue_rows.add(row)

            if ibc > 0 and ibc not in ibc_flags:
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
                issue_rows.add(row)

            if seg_type in {3, 4, 5} and pos_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE {seg_type} requires pos_mat > 0."))
                issue_rows.add(row)
            if pos_mat > 0 and pos_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag pos_mat={pos_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if seg_type == 5 and pos_mat > 0 and pos_mat == neg_mat:
                findings.append(("ERROR", row, f"{label}: TYPE 5 requires two distinct materials; "
                                 "pos_mat and neg_mat are equal."))
                issue_rows.add(row)
            if seg_type == 5 and neg_mat <= 0:
                findings.append(("ERROR", row, f"{label}: TYPE 5 requires neg_mat > 0."))
                issue_rows.add(row)
            if neg_mat > 0 and neg_mat not in diel_flags:
                findings.append(
                    ("ERROR", row, f"{label}: dielectric flag neg_mat={neg_mat} is referenced but not defined.")
                )
                issue_rows.add(row)
            if seg_type in {1, 2, 3, 4} and neg_mat != 0:
                findings.append(("WARN", row, f"{label}: TYPE {seg_type} typically uses neg_mat=0."))
                issue_rows.add(row)


        global_primitives: 'List[Tuple[int, int, Tuple[float, float, float, float], str]]' = []
        row_type: 'Dict[int, int]' = {}
        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            seg_type = self._parse_int_token(props[0], -1)
            row_type[row] = seg_type
            for pidx, prim in enumerate(primitives_by_row[row]):
                global_primitives.append((row, pidx, prim, seg.name))

        for _key, hits in endpoint_hits.items():
            incident_rows = sorted({h[0] for h in hits})
            if len(hits) == 1:
                row, pidx, end = hits[0]
                x = primitives_by_row[row][pidx][2*end]
                on_axis = self.mode == "bor" and abs(x) <= axis_tol
                if not on_axis and not self._bor_sheet_layout and row_type.get(row, -1) in {2, 3, 4, 5}:
                    findings.append(
                        ("WARN", row, f"Row {row + 1}: dangling endpoint not connected to any other primitive.")
                    )
                    issue_rows.add(row)
            if len(hits) > 6:
                row = incident_rows[0]
                findings.append(
                    (
                        "WARN",
                        row,
                        f"Row {row + 1}: high-degree node with {len(hits)} incident primitive endpoints "
                        "(possible non-manifold junction).",
                    )
                )
                issue_rows.add(row)

        junction_findings, junction_rows = self._junction_findings(
            endpoint_hits, primitives_by_row, axis_tol, tol)
        findings.extend(junction_findings)
        issue_rows.update(junction_rows)

        max_intersections = 30
        found_intersections = 0
        bounds = primitive_bounds([entry[2] for entry in global_primitives])
        for i, j in overlapping_pairs(bounds, tol, self.checkpoint):
            row_i, pidx_i, prim_i, name_i = global_primitives[i]
            x1, y1, x2, y2 = prim_i
            k_i0 = endpoint_keys[(row_i, pidx_i, 0)]
            k_i1 = endpoint_keys[(row_i, pidx_i, 1)]
            row_j, pidx_j, prim_j, name_j = global_primitives[j]
            u1, v1, u2, v2 = prim_j
            k_j0 = endpoint_keys[(row_j, pidx_j, 0)]
            k_j1 = endpoint_keys[(row_j, pidx_j, 1)]

            shared_endpoint = k_i0 in {k_j0, k_j1} or k_i1 in {k_j0, k_j1}
            overlap = self._collinear_overlap(prim_i, prim_j, tol)
            if shared_endpoint and not overlap:
                continue

            if not self._segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                continue

            findings.append(
                (
                    "ERROR",
                    row_i,
                    (
                        f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have "
                        + ("overlapping or duplicate primitives." if overlap else
                           "a non-endpoint primitive intersection.")
                    ),
                )
            )
            issue_rows.add(row_i)
            issue_rows.add(row_j)
            found_intersections += 1
            if found_intersections >= max_intersections:
                findings.append(
                    (
                        "WARN",
                        row_i,
                        f"Intersection reporting truncated after {max_intersections} findings.",
                    )
                )
                break


        if self.mode == "bor":
            # A meridian uses axis closure and solver grouping, not 2D winding.
            if all(primitives_by_row) and primitives_by_row:
                findings.extend(self._bor_findings(primitives_by_row))
            self.checkpoint()
            return findings, issue_rows

        region_findings, region_rows = self._nested_region_findings()
        findings.extend(region_findings)
        issue_rows.update(region_rows)

        chain_specs: 'List[ChainSpec]' = []
        for row, seg in enumerate(self.segments):
            self.checkpoint()
            props = self._ensure_prop_len(seg.properties, 5)
            xs, ys = self._segment_plot_xy(seg) if primitives_by_row[row] else ([], [])
            chain_specs.append(ChainSpec(
                name=seg.name or f"segment_{row + 1}",
                seg_type=self._parse_int_token(props[0], 2),
                pos_mat=self._parse_int_token(props[3], 0),
                points=list(zip(xs, ys)),
                # Lets the orientation check resolve TYPE 5 parents, as the
                # solver-side validation (geometry.io) does.
                neg_mat=self._parse_int_token(props[4], 0),
            ))
        for severity, chain_idx, message in check_orientation_consistency(chain_specs):
            findings.append((severity, chain_idx, message))
            if severity == "ERROR" and 0 <= chain_idx < len(self.segments):
                issue_rows.add(chain_idx)

        self.checkpoint()
        return findings, issue_rows

