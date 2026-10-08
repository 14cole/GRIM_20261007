import math
import os
from pathlib import Path
import shutil
import tempfile
from ghost_backend.execution.runtime import dataclass, field, unlink_if_exists
from typing import Any, Dict, List, Optional, Set, Tuple


IBC_KINDS = ("constant", "linear", "cosine", "exp")


def is_ibc_inline_row(row: 'List[str]') -> 'bool':
    """True if the row is in the 6-token inline form (flag kind R_s X_s R_e X_e)."""
    return (
        len(row) == 6
        and str(row[1]).strip().lower() in IBC_KINDS
    )


def is_file_material_row(row: 'List[str]') -> 'bool':
    """True for the explicit ``flag filename.csv`` material-row shape."""
    return (
        len(row) == 2
        and str(row[1]).strip().lower().endswith(".csv")
    )


def is_tabulated_row(row: 'List[str]') -> 'bool':
    """True if the row references a CSV material table in Hz."""
    return is_file_material_row(row)


def _validate_material_filename(filename: 'str', context: 'str') -> 'str':
    """Validate a portable same-directory CSV material sidecar name."""
    raw_name = str(filename)
    name = raw_name.strip()
    if not name or not name.lower().endswith(".csv"):
        raise ValueError(
            f"{context} must name a .csv file; got {filename!r}."
        )


    if any(character.isspace() for character in raw_name):
        raise ValueError(
            f"{context} cannot contain whitespace because .geo material rows "
            f"use unquoted fields; got {filename!r}. Rename the CSV and try "
            "again."
        )
    if (
        name in (".", "..")
        or os.path.isabs(name)
        or os.path.basename(name) != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(
            f"{context} must be a filename in the same directory as the "
            f"geometry file (no directory components); got {filename!r}."
        )
    return name


class AtomicFileTransaction:
    """Stage and publish a small related set of files with rollback support.

    ``os.replace`` is atomic for each file but a geometry and its material
    sidecars form a set.  This helper stages every replacement first, keeps a
    same-directory backup of each existing destination, and restores already
    published destinations if a later publication fails.  Call ``commit``
    only after any associated in-memory/UI update has also succeeded; callers
    may call ``rollback`` until then.
    """

    def __init__(self) -> 'None':
        self._staged: 'List[Tuple[Path, Path]]' = []
        self._backups: 'Dict[Path, Optional[Path]]' = {}
        self._published: 'List[Path]' = []
        self._committed = False

    @staticmethod
    def _temporary_path(destination: 'Path', suffix: 'str') -> 'Path':
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=suffix,
            dir=str(destination.parent),
        )
        os.close(fd)
        return Path(temporary_name)

    def _check_destination(self, destination: 'Path') -> 'Path':
        target = Path(destination).expanduser().resolve(strict=False)
        key = os.path.normcase(str(target))
        if any(os.path.normcase(str(existing)) == key for existing, _ in self._staged):
            raise ValueError(
                f"The file transaction contains the destination more than once: {target}"
            )
        if target.exists() and not target.is_file():
            raise OSError(f"save target is not a regular file: {target}")
        return target

    def stage_copy(self, source: 'Path', destination: 'Path') -> 'None':
        """Copy ``source`` to a temporary file beside ``destination``."""

        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"source file does not exist: {source_path}")
        target = self._check_destination(destination)
        temporary = self._temporary_path(target, ".stage")
        try:


            shutil.copyfile(source_path, temporary)

            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
        except Exception:
            unlink_if_exists(temporary)
            raise
        self._staged.append((target, temporary))

    def stage_text(self, text: 'str', destination: 'Path') -> 'None':
        """Write UTF-8 ``text`` to a temporary file beside ``destination``."""

        target = self._check_destination(destination)
        temporary = self._temporary_path(target, ".stage")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            unlink_if_exists(temporary)
            raise
        self._staged.append((target, temporary))

    def publish(self) -> 'None':
        """Publish all staged files, restoring the old set on any failure."""

        if self._committed:
            raise RuntimeError("The file transaction has already been committed.")
        if self._published:
            raise RuntimeError("The file transaction has already been published.")
        try:

            for destination, _temporary in self._staged:
                if destination.exists():
                    backup = self._temporary_path(destination, ".backup")
                    try:


                        shutil.copyfile(destination, backup)
                        with backup.open("r+b") as stream:
                            os.fsync(stream.fileno())
                    except Exception:
                        unlink_if_exists(backup)
                        raise
                    self._backups[destination] = backup
                else:
                    self._backups[destination] = None

            for destination, temporary in self._staged:
                os.replace(temporary, destination)
                self._published.append(destination)
        except Exception as publish_error:
            try:
                self.rollback()
            except Exception as rollback_error:
                raise OSError(
                    f"File publication failed ({publish_error}); rollback also "
                    f"failed ({rollback_error})."
                ) from publish_error
            raise

    def rollback(self) -> 'None':
        """Restore all destinations changed by ``publish`` and remove stages."""

        errors: 'List[str]' = []
        preserved_backups: 'Set[Path]' = set()
        for destination in reversed(self._published):
            backup = self._backups.get(destination)
            try:
                if backup is None:
                    unlink_if_exists(destination)
                else:
                    os.replace(backup, destination)
            except Exception as exc:
                if backup is not None and backup.exists():
                    preserved_backups.add(backup)
                errors.append(f"{destination}: {exc}")

        self._published.clear()
        self._remove_temporary_files(preserved_backups)
        self._backups.clear()
        if errors:
            backup_note = ""
            if preserved_backups:
                backup_note = " Recovery backup(s): " + ", ".join(
                    str(path) for path in sorted(preserved_backups, key=str)
                )
            raise OSError("Could not restore " + "; ".join(errors) + backup_note)

    def commit(self) -> 'None':
        """Accept the published set and remove transaction backups."""

        if self._committed:
            return
        self._committed = True
        self._published.clear()
        self._remove_temporary_files(set())
        self._backups.clear()

    def abort(self) -> 'None':
        """Rollback a published set or discard an unpublished staged set."""

        if self._committed:
            return
        if self._published:
            self.rollback()
        else:
            self._remove_temporary_files(set())
            self._backups.clear()

    def _remove_temporary_files(self, preserve: 'Set[Path]') -> 'None':
        for _destination, temporary in self._staged:
            if temporary not in preserve:
                try:
                    unlink_if_exists(temporary)
                except OSError:
                    pass
        self._staged.clear()
        for backup in self._backups.values():
            if backup is not None and backup not in preserve:
                try:
                    unlink_if_exists(backup)
                except OSError:
                    pass


def material_filename_from_row(row: 'List[str]') -> 'Optional[str]':
    """Return the sidecar filename referenced by a material row, if any."""
    if is_file_material_row(row):
        return _validate_material_filename(
            row[1], f"Material flag {row[0]} CSV reference"
        )
    return None


def _validate_ibc_row(tokens: 'List[str]', lineno_for_err: 'str') -> 'None':
    """Raise ValueError if the row is not a supported IBC shape.

    Supported shapes (the `flag R X` and `flag taper kind ...` forms are not accepted):
      * CSV: `flag filename.csv`, for any positive integer flag.
      * Inline: `flag kind R_start X_start R_end X_end` with kind in IBC_KINDS.
        For ``kind == "constant"`` only R_start/X_start matter; the end values
        are placeholders (write 0) and are ignored on read.
    """
    if not tokens:
        return
    try:
        flag = int(tokens[0])
    except (ValueError, TypeError):
        raise ValueError(
            f"IBC row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if flag <= 0:
        raise ValueError(
            f"IBC row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if len(tokens) > 1 and tokens[1].lower() == "thin_dielectric":
        from ghost_backend.twod.formulations.thin_layer import ThinLayerDefinition
        ThinLayerDefinition.from_row(tokens)
        return
    if len(tokens) == 2:
        _validate_material_filename(
            tokens[1], f"IBC flag {flag} CSV reference"
        )
        return
    if len(tokens) == 1:
        raise ValueError(
            f"IBC flag {flag} has no definition. Use either "
            "'flag filename.csv' (comma-separated, header required, frequency in Hz) or "
            "'flag kind R_start X_start R_end X_end'."
        )
    if len(tokens) != 6:
        raise ValueError(
            f"Inline IBC row must have 6 tokens (flag kind R_start X_start R_end X_end); "
            f"got {len(tokens)}: {lineno_for_err}"
        )
    kind = tokens[1].strip().lower()
    if kind not in IBC_KINDS:
        raise ValueError(
            f"IBC kind must be one of {IBC_KINDS}; got {tokens[1]!r}: {lineno_for_err}"
        )
    try:
        values = [float(token) for token in tokens[2:]]
    except ValueError as exc:
        raise ValueError(
            f"Inline IBC resistance/reactance values must be numeric: "
            f"{lineno_for_err}"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Inline IBC resistance/reactance values must be finite: "
            f"{lineno_for_err}"
        )


def _validate_dielectric_row(tokens: 'List[str]', lineno_for_err: 'str') -> 'None':
    """Validate one strict inline, CSV, or legacy dielectric definition."""
    if not tokens:
        return
    try:
        flag = int(tokens[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Dielectric row must start with a positive integer flag: "
            f"{lineno_for_err}") from exc
    if flag <= 0:
        raise ValueError(
            f"Dielectric row must start with a positive integer flag: "
            f"{lineno_for_err}"
        )
    if len(tokens) == 2:
        _validate_material_filename(
            tokens[1], f"Dielectric flag {flag} CSV reference"
        )
        return
    if len(tokens) == 1:
        raise ValueError(
            f"Dielectric flag {flag} has no definition. Use either "
            "'flag filename.csv' (comma-separated, header required, frequency in Hz) or "
            "'flag eps_real eps_imag mu_real mu_imag'."
        )
    if len(tokens) != 5:
        raise ValueError(
            "Inline dielectric row must have 5 tokens "
            "(flag eps_real eps_imag mu_real mu_imag); "
            f"got {len(tokens)}: {lineno_for_err}")
    try:
        values = [float(token) for token in tokens[1:]]
    except ValueError as exc:
        raise ValueError(
            f"Inline dielectric values must be numeric: "
            f"{lineno_for_err}") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Inline dielectric values must be finite: {lineno_for_err}")


@dataclass
class Segment:
    name: 'str'
    seg_type: 'Optional[str]'
    properties: 'List[str]'
    x: 'List[float]'
    y: 'List[float]'


def parse_geometry(text: 'str') -> 'Tuple[str, List[Segment], List[List[str]], List[List[str]]]':
    lines = [ln.strip() for ln in text.splitlines()]
    title = "Geometry"
    segments: 'List[Segment]' = []
    ibcs_entries: 'List[List[str]]' = []
    dielectric_entries: 'List[List[str]]' = []

    state = "segments"
    current_name: 'Optional[str]' = None
    current_type: 'Optional[str]' = None
    current_props: 'List[str]' = []
    current_props_seen = False
    cur_x: 'List[float]' = []
    cur_y: 'List[float]' = []

    def flush_segment() -> 'None':
        if current_name is not None:
            if not current_props_seen:
                raise ValueError(
                    f"Segment '{current_name}' is missing its required "
                    "properties line (type n ibc pos_mat neg_mat)."
                )
            segments.append(
                Segment(
                    name=current_name,
                    seg_type=current_type,
                    properties=current_props[:],
                    x=cur_x[:],
                    y=cur_y[:],
                )
            )

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            if ":" not in ln:
                raise ValueError(f"Title line must contain ':': {ln}")
            title = ln.split(":", 1)[1].strip() or title
            continue
        if low.startswith("ibcs_resistances:") or low.startswith("ibcs:"):
            if state == "segments":
                flush_segment()
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
            state = "dielectrics"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    current_name, current_type = "Unnamed", None
                elif len(parts) == 1:
                    current_name, current_type = parts[0], None
                else:
                    current_name, current_type = parts[0], parts[1]
                current_props = []
                current_props_seen = False
                cur_x.clear()
                cur_y.clear()
                continue
            if low.startswith("properties:"):
                if current_name is None:
                    raise ValueError(
                        "properties line appears before the first "
                        f"'Segment:' header: {ln}"
                    )
                if current_props_seen:
                    raise ValueError(
                        f"Segment '{current_name}' has more than one "
                        "properties line; exactly one is required."
                    )
                current_props = ln.split(":", 1)[1].strip().split()
                if len(current_props) != 5:
                    raise ValueError(
                        f"properties line must have exactly 5 fields "
                        f"(type n ibc pos_mat neg_mat); got {len(current_props)}: {ln}"
                    )
                current_props_seen = True
                continue

            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(f"Geometry line must have 4 numbers, got {len(tokens)} {ln}")
            if current_name is None:


                raise ValueError(
                    f"Geometry data line appears before the first 'Segment:' header: {ln}"
                )
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError:
                raise ValueError(f"Geometry line must contain valid numbers: {ln}")
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])
        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                _validate_ibc_row(tokens, ln)
                ibcs_entries.append(list(tokens))
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                _validate_dielectric_row(tokens, ln)
                dielectric_entries.append(tokens)

    if state == "segments":
        flush_segment()

    return title, segments, ibcs_entries, dielectric_entries


def build_geometry_text(
    title: 'str',
    segments: 'List[Segment]',
    ibcs_entries: 'List[List[str]]',
    dielectric_entries: 'List[List[str]]',
) -> 'str':
    lines: 'List[str]' = [f"Title: {title}"]
    for seg in segments:
        segment_name = str(seg.name).strip()
        if not segment_name or any(char.isspace() for char in segment_name):
            raise ValueError(
                f"Segment name {seg.name!r} is not representable in .geo files; "
                "use a non-empty name without whitespace."
            )
        if ":" in segment_name or "#" in segment_name:
            raise ValueError(
                f"Segment name {seg.name!r} contains reserved .geo punctuation."
            )
        props = list(seg.properties)


        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        if effective_type:
            lines.append(f"Segment: {segment_name} {effective_type}")
        else:
            lines.append(f"Segment: {segment_name}")

        if len(props) < 5:
            props.extend([""] * (5 - len(props)))
        elif len(props) > 5:
            props = props[:5]


        type_token = str(props[0]).strip() if props[0] is not None and str(props[0]).strip() else str(effective_type or "2")
        out_props = [type_token]
        for p in props[1:]:
            token = str(p).strip() if p is not None else ""
            out_props.append(token if token else "0")
        lines.append("properties: " + " ".join(out_props))

        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        for i in range(0, len(seg.x), 2):
            x1, y1, x2, y2 = seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]


            lines.append(f"{float(x1)!r} {float(y1)!r} {float(x2)!r} {float(y2)!r}")

    lines.append("IBCS_Resistances:")
    for raw_row in ibcs_entries:
        raw_tokens = [str(token) for token in raw_row]
        _validate_ibc_row(raw_tokens, " ".join(raw_tokens))
        row = [token.strip() for token in raw_tokens]
        if row:
            lines.append(" ".join(row))
    lines.append("Dielectrics:")
    for raw_row in dielectric_entries:
        raw_tokens = [str(token) for token in raw_row]
        _validate_dielectric_row(raw_tokens, " ".join(raw_tokens))
        row = [token.strip() for token in raw_tokens]
        if row:
            lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


def snapshot_to_geometry_text(snapshot: 'Dict[str, Any]') -> 'str':
    """Serialize a geometry SNAPSHOT dict (the {title, segments:[{name, seg_type,
    properties, point_pairs}], ibcs, dielectrics} form used by the solvers and
    the feature pipeline) to .geo text.  Bridges the dict form to
    build_geometry_text, which wants Segment objects."""
    segs: 'List[Segment]' = []
    for s in snapshot.get("segments", []):
        x: 'List[float]' = []
        y: 'List[float]' = []
        for pp in s.get("point_pairs", []):
            x.extend([float(pp["x1"]), float(pp["x2"])])
            y.extend([float(pp["y1"]), float(pp["y2"])])
        segs.append(Segment(str(s.get("name", "seg")),
                            (str(s["seg_type"]) if s.get("seg_type") is not None else None),
                            [str(p) for p in s.get("properties", [])], x, y))
    return build_geometry_text(str(snapshot.get("title", "geometry")), segs,
                               list(snapshot.get("ibcs", [])),
                               list(snapshot.get("dielectrics", [])))


def save_snapshot_geo(snapshot: 'Dict[str, Any]', path: 'str') -> 'str':
    """Write a geometry snapshot dict to a .geo file; returns the path."""
    out = path if path.lower().endswith(".geo") else path + ".geo"
    with open(out, "w") as fh:
        fh.write(snapshot_to_geometry_text(snapshot))
    return out


def material_sidecar_paths(geometry_path: 'str') -> 'List[str]':
    """Return the exact material files referenced by a saved geometry.

    Missing sidecars raise here so cache fingerprints and HPC staging cannot
    silently omit a physical input.
    """

    geo_path = os.path.abspath(str(geometry_path))
    with open(geo_path, "r") as geo_file:
        _title, _segments, ibcs, dielectrics = parse_geometry(geo_file.read())
    folder = os.path.dirname(geo_path)
    names: 'Set[str]' = set()
    for row in list(ibcs) + list(dielectrics):
        filename = material_filename_from_row(row)
        if filename:
            names.add(filename)
    paths: 'List[str]' = []
    for name in sorted(names):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Could not locate referenced material file {name} beside "
                f"geometry file {geo_path}."
            )
        paths.append(path)
    return paths


@dataclass
class ChainSpec:
    """Neutral view of one segment's primitive chain for orientation checks."""

    name: 'str'
    seg_type: 'int'
    pos_mat: 'int'
    points: 'List[Tuple[float, float]]' = field(default_factory=list)
    neg_mat: 'int' = 0


def _chain_is_closed(points: 'List[Tuple[float, float]]', tol: 'float') -> 'bool':
    if len(points) < 4:
        return False
    return math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol


def _chain_area2(points: 'List[Tuple[float, float]]') -> 'float':
    area2 = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        area2 += x0 * y1 - x1 * y0
    return area2


def _point_in_polygon(px: 'float', py: 'float', poly: 'List[Tuple[float, float]]') -> 'bool':
    """Even-odd ray casting over every edge, the closing edge poly[-1] -> poly[0] included.

    ``poly`` is a closed vertex chain whose last vertex repeats the first one
    only to drawing precision: a circle drawn with cos/sin closes to ~5e-18 m.
    A ray through that seam crosses the closing edge, and skipping it flips
    the parity (a void inside a disk then counts as outside it).
    """

    inside = False
    for index in range(len(poly)):
        x1, y1 = poly[index - 1]
        x2, y2 = poly[index]
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _rep_point(points: 'List[Tuple[float, float]]') -> 'Tuple[float, float]':
    """A point strictly on the chain: midpoint of the first primitive."""

    return (0.5 * (points[0][0] + points[1][0]), 0.5 * (points[0][1] + points[1][1]))


# TYPEs with air on exactly one side: crossing one switches between air and a
# body, so only these set the nesting parity of the air side.
AIR_BOUNDING_TYPES = (2, 3)


def _interior_medium(chain: 'ChainSpec', drawn_ccw: 'bool') -> 'Tuple[str, int]':
    """Medium inside a closed chain as drawn: its normal (left of travel) points inward when CCW.

    Returns ('air', 0), ('pec', 0) or ('dielectric', flag); flag 0 is unknown
    (a TYPE 5 chain whose neg_mat was not supplied).
    """

    if chain.seg_type == 1:
        return ('air', 0)
    normal_side = {2: ('air', 0), 3: ('air', 0),
                   4: ('dielectric', chain.pos_mat), 5: ('dielectric', chain.pos_mat)}
    other_side = {2: ('pec', 0), 3: ('dielectric', chain.pos_mat),
                  4: ('pec', 0), 5: ('dielectric', chain.neg_mat)}
    table = normal_side if drawn_ccw else other_side
    return table.get(chain.seg_type, ('unknown', 0))


def _weld_point_keys(points: 'List[Tuple[float, float]]', tol: 'float') -> 'List[int]':
    """Cluster ids for points, joining points within ``tol`` (Euclidean) of an earlier one.

    A rounding grid alone splits two copies of one vertex that straddle a
    grid line, hiding the junction of two chains from the checks below.
    """

    cell = 2.0 * max(float(tol), 1e-300)
    grid: 'Dict[Tuple[int, int], List[Tuple[float, float, int]]]' = {}
    result: 'List[int]' = []
    for x, y in points:
        cx, cy = math.floor(x / cell), math.floor(y / cell)
        best = None
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for px, py, cluster in grid.get((i, j), ()):
                    distance = math.hypot(px - x, py - y)
                    if distance <= tol and (best is None or distance < best[0]):
                        best = (distance, cluster)
        cluster = len(result) if best is None else best[1]
        grid.setdefault((cx, cy), []).append((x, y, cluster))
        result.append(cluster)
    return result


def _geometry_tolerance(chains: 'List["ChainSpec"]') -> 'float':
    xs = [p[0] for c in chains for p in c.points]
    ys = [p[1] for c in chains for p in c.points]
    if not xs:
        return 1e-9
    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    return max(1e-12, 1e-9 * diag)


def check_orientation_consistency(
    chains: 'List[ChainSpec]',
    tol: 'Optional[float]' = None,
) -> 'List[Tuple[str, int, str]]':
    """
    Validate winding and air-side consistency of a set of segment chains.

    Returns findings as (severity, chain_index, message) with severity in
    {"ERROR", "INFO"}.  ERROR findings mean the geometry, solved as drawn,
    would produce silently wrong TE results:

    1. A closed TYPE 2/3 chain whose winding puts the air side on the wrong
       side for its nesting depth (a top-level body must be CW = normals out
       into unbounded air; a void nested inside a body must be CCW = normals
       into the enclosed air; parity alternates with depth). The depth counts
       only enclosing closed TYPE 2/3 chains: TYPE 1 sheets have air on both
       sides and TYPE 4/5 contours bound no air.
    2. A closed TYPE 4 chain enclosed by a matching-pos_mat TYPE 3/5 contour
       (standard coated-body layout) wound so its normal points away from
       the coating; without such a contour, a closed TYPE 4 chain whose
       innermost enclosing contour puts its pos_mat dielectric (or PEC) just
       outside it and whose winding disagrees.
    3. Two open TYPE 2/3 chains meeting end-to-end or start-to-start at a
       degree-2 junction: their air sides disagree there (e.g. an
       air|dielectric wall chained to an air|PEC wall the wrong way round).
    4. A closed loop stitched head-to-tail from open TYPE 2/3 chains whose
       overall winding is inside-out for its nesting depth (consistent with
       each other, but all reversed together).

    TYPE 5 chains and TYPE 1 sheets are never flagged: for TYPE 5 the
    winding IS the user's pos/neg labeling choice, and TYPE 1 is symmetric.
    """

    findings: 'List[Tuple[str, int, str]]' = []
    if not chains:
        return findings
    if tol is None:
        tol = _geometry_tolerance(chains)

    closed_polys: 'List[Tuple[int, List[Tuple[float, float]]]]' = []
    open_air_chains: 'List[int]' = []
    open_chains: 'List[int]' = []
    for idx, ch in enumerate(chains):
        if len(ch.points) < 2:
            continue
        if _chain_is_closed(ch.points, tol):
            if abs(_chain_area2(ch.points)) > 0.0:
                closed_polys.append((idx, ch.points))
            continue
        if ch.seg_type != 1:
            open_chains.append(idx)
        if ch.seg_type in (2, 3):
            open_air_chains.append(idx)

    areas = {idx: _chain_area2(poly) for idx, poly in closed_polys}
    closed_polys_by_idx = dict(closed_polys)
    # Bounding boxes filter the point-in-polygon tests: 1,600 closed 8-gons took
    # 1.9 s as an all-pairs scan over every vertex.
    import numpy as np
    boxes = np.asarray([
        (min(p[0] for p in poly), max(p[0] for p in poly),
         min(p[1] for p in poly), max(p[1] for p in poly))
        for _, poly in closed_polys
    ], dtype=float).reshape(-1, 4)

    def _enclosing(rep: 'Tuple[float, float]', exclude: 'Set[int]', types: 'Tuple[int, ...]') -> 'List[int]':
        """Chains (of the given TYPEs, not excluded) whose closed polygon contains ``rep``."""
        px, py = float(rep[0]), float(rep[1])
        candidates = np.flatnonzero((boxes[:, 0] <= px) & (px <= boxes[:, 1])
                                    & (boxes[:, 2] <= py) & (py <= boxes[:, 3]))
        found = []
        for k in candidates:
            cidx, poly = closed_polys[int(k)]
            if (cidx not in exclude and chains[cidx].seg_type in types
                    and _point_in_polygon(px, py, poly)):
                found.append(cidx)
        return found

    def _depth(rep: 'Tuple[float, float]', exclude: 'Set[int]') -> 'int':
        # Only boundaries between air and a non-air region (TYPE 2 and 3)
        # switch the air side. A TYPE 1 sheet has air on both sides; TYPE 4
        # and 5 bound no air, so a void in a layered body is still a void.
        return len(_enclosing(rep, exclude, AIR_BOUNDING_TYPES))


    for idx, poly in closed_polys:
        ch = chains[idx]
        drawn_ccw = _chain_area2(poly) > 0.0
        if ch.seg_type in (2, 3):
            depth = _depth(_rep_point(poly), {idx})
            expected_ccw = (depth % 2) == 1
            if drawn_ccw != expected_ccw:
                where = (
                    "the air side is the unbounded exterior" if depth % 2 == 0
                    else "the air side is the enclosed interior (nested void)"
                )
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE {ch.seg_type}) is a closed contour drawn "
                    f"{'CCW' if drawn_ccw else 'CW'}, but {where}, so the drawing convention "
                    f"(normal into air) requires {'CCW' if expected_ccw else 'CW'} winding. "
                    "Reverse the segment's endpoint order.",
                ))
        elif ch.seg_type == 4:
            rep = _rep_point(poly)
            has_matching_parent = any(
                chains[cidx].pos_mat == ch.pos_mat
                for cidx in _enclosing(rep, {idx}, (3, 5))
            )
            if has_matching_parent and drawn_ccw:
                findings.append((
                    "ERROR", idx,
                    f"Segment '{ch.name}' (TYPE 4) is a closed contour drawn CCW, but the "
                    f"pos_mat={ch.pos_mat} coating lies outside it, so the drawing convention "
                    "(normal into the dielectric) requires CW winding. "
                    "Reverse the segment's endpoint order.",
                ))
            elif not has_matching_parent:
                # Otherwise the region just outside the contour is the inside
                # of its innermost enclosing contour. A second coating layer
                # (a TYPE 5 parent labelled pos_mat outside) or a filled cavity
                # in a PEC body has no matching TYPE 3/5 parent, and a reversed
                # core there changed a two-layer cylinder by 17-31 dB unnoticed.
                parents = _enclosing(rep, {idx}, (1, 2, 3, 4, 5))
                parent = min(parents, key=lambda cidx: abs(areas[cidx])) if parents else None
                # An open boundary between the contour and that parent (a
                # cavity wall stitched from several segments) may change the
                # medium in between: no inference is made then.
                if parent is not None and not any(
                        _point_in_polygon(*_rep_point(chains[other].points), closed_polys_by_idx[parent])
                        and not _point_in_polygon(*_rep_point(chains[other].points), poly)
                        for other in open_chains):
                    around = _interior_medium(chains[parent], areas[parent] > 0.0)
                    expected_ccw = None
                    if around == ('dielectric', ch.pos_mat):
                        expected_ccw = False      # PEC inside, the coating outside
                    elif around == ('pec', 0):
                        expected_ccw = True       # the dielectric fills the contour
                    if expected_ccw is not None and drawn_ccw != expected_ccw:
                        side = "outside" if not expected_ccw else "inside"
                        findings.append((
                            "ERROR", idx,
                            f"Segment '{ch.name}' (TYPE 4) is a closed contour drawn "
                            f"{'CCW' if drawn_ccw else 'CW'}, but its enclosing contour "
                            f"'{chains[parent].name}' puts the pos_mat={ch.pos_mat} dielectric "
                            f"{side} it, so the drawing convention (normal into the dielectric) "
                            f"requires {'CCW' if expected_ccw else 'CW'} winding. "
                            "Reverse the segment's endpoint order.",
                        ))


    end_records = [(idx, which) for idx in open_air_chains for which in ("start", "end")]
    end_clusters = _weld_point_keys(
        [chains[idx].points[0 if which == "start" else -1] for idx, which in end_records], tol)
    ends: 'Dict[int, List[Tuple[int, str]]]' = {}
    for cluster, record in zip(end_clusters, end_records):
        ends.setdefault(cluster, []).append(record)

    adjacency: 'Dict[int, List[Tuple[str, int, str]]]' = {i: [] for i in open_air_chains}
    for key, members in ends.items():
        if len(members) != 2:
            continue
        (ia, ea), (ib, eb) = members
        if ia == ib:
            continue
        adjacency[ia].append((ea, ib, eb))
        adjacency[ib].append((eb, ia, ea))
        if ea == eb:
            ca, cb = chains[ia], chains[ib]
            findings.append((
                "ERROR", ia,
                f"Segments '{ca.name}' (TYPE {ca.seg_type}) and '{cb.name}' "
                f"(TYPE {cb.seg_type}) meet {ea}-to-{eb} at "
                f"({chains[ia].points[0 if ea == 'start' else -1][0]:.6g}, "
                f"{chains[ia].points[0 if ea == 'start' else -1][1]:.6g}): "
                "their air sides point to opposite sides of the boundary there. "
                "Reverse one segment's endpoint order so the chains run head-to-tail.",
            ))


    visited: 'set' = set()
    for start_idx in open_air_chains:
        if start_idx in visited or len(adjacency[start_idx]) != 2:
            continue

        loop = [start_idx]
        cur, arrived_via = start_idx, "end"
        ok = True
        while True:
            nxt = next(
                ((jb, eb) for (ea, jb, eb) in adjacency[cur] if ea == arrived_via),
                None,
            )
            if nxt is None:
                ok = False
                break
            jdx, joint_end = nxt
            if jdx == start_idx:
                break
            if jdx in loop or joint_end != "start":


                ok = False
                break
            loop.append(jdx)
            cur, arrived_via = jdx, "end"
        visited.update(loop)
        if not ok or len(loop) < 2:
            continue
        stitched: 'List[Tuple[float, float]]' = []
        for idx in loop:
            pts = chains[idx].points
            stitched.extend(pts if not stitched else pts[1:])
        if not _chain_is_closed(stitched, tol) or abs(_chain_area2(stitched)) <= 0.0:
            continue
        drawn_ccw = _chain_area2(stitched) > 0.0
        depth = _depth(_rep_point(stitched), set(loop))
        expected_ccw = (depth % 2) == 1
        if drawn_ccw != expected_ccw:
            names = ", ".join(f"'{chains[i].name}'" for i in loop)
            findings.append((
                "ERROR", loop[0],
                f"Segments {names} form a closed boundary drawn "
                f"{'CCW' if drawn_ccw else 'CW'}, but the drawing convention (normal "
                f"into air) requires {'CCW' if expected_ccw else 'CW'} winding for this "
                "loop. Reverse every segment in the loop.",
            ))

    return findings


def chains_from_snapshot_segments(segments: 'List[Dict[str, Any]]') -> 'List[ChainSpec]':
    """Build ChainSpecs from solver-snapshot segment dicts (point_pairs form)."""

    chains: 'List[ChainSpec]' = []
    for seg_idx, seg in enumerate(segments):
        props = list(seg.get("properties", []) or [])

        def _flag(tok: 'Any', default: 'int' = 0) -> 'int':
            try:
                text = str(tok).strip().lower()
                if text.startswith("mat."):
                    text = text[4:]
                return int(float(text))
            except (ValueError, TypeError):
                return default

        seg_type = _flag(props[0], 2) if len(props) > 0 and str(props[0]).strip() else _flag(seg.get("seg_type", 2), 2)
        pos_mat = _flag(props[3]) if len(props) > 3 else 0
        neg_mat = _flag(props[4]) if len(props) > 4 else 0
        pts: 'List[Tuple[float, float]]' = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            try:
                x1 = float(pair.get("x1", 0.0)); y1 = float(pair.get("y1", 0.0))
                x2 = float(pair.get("x2", 0.0)); y2 = float(pair.get("y2", 0.0))
            except (TypeError, ValueError):
                continue
            if i == 0:
                pts.append((x1, y1))
            pts.append((x2, y2))
        chains.append(ChainSpec(
            name=str(seg.get("name", f"segment_{seg_idx + 1}")),
            seg_type=seg_type,
            pos_mat=pos_mat,
            points=pts,
            neg_mat=neg_mat,
        ))
    return chains


def build_geometry_snapshot(
    title: 'str',
    segments: 'List[Segment]',
    ibcs_entries: 'List[List[str]]',
    dielectric_entries: 'List[List[str]]',
) -> 'Dict[str, Any]':
    segments_payload = []
    for seg in segments:
        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        point_pairs = []
        for i in range(0, len(seg.x), 2):
            point_pairs.append(
                {
                    "x1": seg.x[i],
                    "y1": seg.y[i],
                    "x2": seg.x[i + 1],
                    "y2": seg.y[i + 1],
                }
            )
        props = list(seg.properties)
        effective_type = seg.seg_type
        if not effective_type and props and str(props[0]).strip():
            effective_type = props[0]
        segments_payload.append(
            {
                "name": seg.name,
                "seg_type": effective_type,
                "properties": props,
                "point_pairs": point_pairs,
            }
        )

    return {
        "title": title,
        "segment_count": len(segments),
        "segments": segments_payload,
        "ibcs": [list(row) for row in ibcs_entries],
        "dielectrics": [list(row) for row in dielectric_entries],
    }
