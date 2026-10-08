#!/usr/bin/env python3
"""Fast wing/fin approximation: line-expand 2-D sections along their span.

Each entry of SECTIONS is one spanwise station: a 2-D cross-section ``.geo``
solved as a stand-alone object and expanded along the straight line from its
``root`` to its ``tip``. Stations laid end to end sum coherently into one wing,
which is then added to an optional base response and written as one
radar-frame monostatic ``.grim``.

Edit only the USER SETTINGS block, then run:

    python ghost_backend/assembly/expand_wing_sections.py

The GHOST Line Expansion tab and the GRIM Assembly tab call
:func:`expand_wing_sections` with the same inputs.

Frames
------
3-D points and vectors use the Assembly CAD frame (+y nose, +x right, +z up),
the same frame as the point and line placement CSVs. The origin is the body's
phase origin (the origin of its generatrix).

Each section ``.geo`` is drawn in the plane PERPENDICULAR to its span line,
with its origin ON that line (the origin is the placement phase center):

    2-D +y = ``normal``                 (unit, perpendicular to the span)
    2-D +x = span direction x normal    (span direction = root -> tip)

For a swept wing this is the cut normal to the swept span line, not the
streamwise airfoil. Both axes are reported per station; check them.

Base response
-------------
The base is any coherent radar-frame monostatic GRIM: a GHOST BoR deliverable,
an Assembly body-plus-features output, or an imported 3-D body. Its stored
field is kept sample for sample and the sections are added on its own grid.
Everything else in the file is carried over, so the output of a BoR base still
holds the embedded body model and can be chosen as the Assembly Body dataset
to add point and line features on top of the wings.

Model limits
------------
Single bounce: wing and body scatter in isolation and their fields add. No
tip, root or station-to-station diffraction; a taper is a staircase of
constant sections. SHADOW hides the parts of a span line that the embedded BoR
body blocks from the radar (line of sight from the span line itself).
OBLIQUE replaces the broadside section response by the solve at the reduced
frequency f*cos(tilt), which is the exact transverse problem for a PEC section
of infinite length; it is not valid for coated or dielectric sections.
CORNERS adds a physical-optics estimate of the wing-root double bounce whose
phase against the other terms is rough.
PSI_TM_DEG / PSI_TE_DEG are the uncertified 2-D-to-3-D phase constants used by
the rest of Assembly; compare against an independent reference before relying
on wing-body interference lobes.
"""

if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import math
import os
from pathlib import Path

import numpy as np

from ghost_backend.assembly.line_expansion import (
    C0, PSI_HH_DEG, PSI_VV_DEG, SeamCoefficients, coefficients_from_2d,
    expand_perimeter,
)
from ghost_backend.geometry.frames import UNIT_SCALE, to_axis_frame

# =============================================================================
# USER SETTINGS
# =============================================================================

# One dict per spanwise station. Stations that share a "geometry" file are
# solved once per frequency.
SECTIONS = [
    # {"geometry": "wing/station_1.geo",
    #  "root": (3.0, -10.0, 0.0), "tip": (15.0, -10.0, 0.0),
    #  "normal": (0.0, 0.0, 1.0)},
    # {"geometry": "wing/station_2.geo",
    #  "root": (15.0, -10.0, 0.0), "tip": (27.0, -10.0, 0.0),
    #  "normal": (0.0, 0.0, 1.0)},
    # Curved or twisted path: add "normal_end" per station, or build the
    # stations from a point file (x y z [nx ny nz] per line):
    #   *sections_from_path("lip.geo", *read_path_points("lip_path.txt")),
]
COORDINATE_UNITS = "inches"       # units of root/tip and of CORNERS
GEOMETRY_UNITS = "inches"         # units inside the .geo files: inches | meters

# True also places the port-side mirror image (x -> -x) of every station and
# corner.
MIRROR = False

# Optional base response .grim (see "Base response" above). None = sections only.
BODY_GRIM = None

# Radar grid, used only without BODY_GRIM; a base supplies its own grid.
FREQUENCIES_GHZ = None            # e.g. [2.0, 4.0]
AZIMUTHS_DEG = None               # e.g. list(range(0, 360, 1))
ELEVATIONS_DEG = None             # e.g. [0.0]
AXIS_AZ_DEG = None                # body attitude; None = from BODY_GRIM, else 0
AXIS_EL_DEG = None
ROLL_DEG = None

OUTPUT_GRIM = "ghost_backend/results/wing_sections/body_with_wing.grim"

# Angular step of each 2-D section solve over the full 0..360 deg cut. Must
# divide 180. The complex amplitude is interpolated linearly between samples,
# so the step has to resolve its phase; a warning is issued when it does not.
SECTION_ANGLE_STEP_DEG = 0.5

# Hide span-line pieces behind the base's embedded BoR body.
SHADOW = True

# Oblique-incidence correction for PEC sections. Looks tilted further than
# OBLIQUE_MAX_TILT_DEG out of the plane normal to the span reuse the response
# at that tilt. OBLIQUE_MAX_SOLVES caps the reduced-frequency 2-D solves per
# section geometry and look frequency.
OBLIQUE = False
OBLIQUE_MAX_TILT_DEG = 60.0
OBLIQUE_MAX_SOLVES = 24

# Passed to the certified 2-D solve, e.g. {"mesh_convergence_policy": ...}.
SECTION_SOLVER_KWARGS = {}

# 2-D-to-3-D phase mapping, as used by the Assembly line features.
PSI_TM_DEG = PSI_HH_DEG
PSI_TE_DEG = PSI_VV_DEG

# Optional wing-root double-bounce estimates (see fields.corner_amplitude), in
# the CAD frame and COORDINATE_UNITS:
#   {"fold_start": (x, y, z), "fold_end": (x, y, z),
#    "n_wing": (..), "n_body": (..), "face_width": 4.0}
CORNERS = []

# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A section is expanded as two half-space placements (+normal and -normal
# side). expand_perimeter lights a placement only for looks strictly on its
# normal side and fades it toward grazing; the fade is switched off here and
# looks lying in the split plane are tipped onto the +normal side, so every
# look sees the full section exactly once. The tip moves the two-way phase by
# 2 k r * 1e-6 rad at most.
_EDGE_TAPER_DEG = 1.0e-6
_SPLIT_PLANE_TILT = 1.0e-6

# Largest two-way phase change, in radians, allowed between neighbouring
# samples of a section response (in angle, and in reduced frequency).
_SAMPLE_PHASE_RAD = 0.5

_MIRROR_AXIS = np.array([1.0, -1.0, 1.0])  # CAD x is axis-frame y


def _unit_scale(units, label):
    key = str(units).strip().lower()
    if key not in UNIT_SCALE:
        raise ValueError(f"{label} {units!r} is not a supported length unit.")
    return UNIT_SCALE[key]


def _vector(value, label):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be three finite numbers.") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be three finite numbers.")
    return vector


def _stations(sections, coordinate_units, mirror, base_dir):
    """Validated axis-frame stations in meters: (label, geometry, segment, normals)."""

    if not sections:
        raise ValueError("Add at least one section.")
    scale = _unit_scale(coordinate_units, "Coordinate units")
    stations = []
    for index, section in enumerate(sections, 1):
        label = f"section {index}"
        geometry = Path(str(section["geometry"]).strip())
        if not geometry.is_absolute():
            geometry = Path(base_dir) / geometry
        if not geometry.is_file():
            raise ValueError(f"{label}: geometry not found: {geometry}")
        root = to_axis_frame(_vector(section["root"], f"{label} root")) * scale
        tip = to_axis_frame(_vector(section["tip"], f"{label} tip")) * scale
        span = tip - root
        length = float(np.linalg.norm(span))
        if length <= 0.0:
            raise ValueError(f"{label}: root and tip coincide.")
        span /= length
        normals = []
        end_normal = section.get("normal_end")
        for key, value in (
            ("normal", section["normal"]),
            ("normal_end", section["normal"] if end_normal is None else end_normal),
        ):
            declared = to_axis_frame(_vector(value, f"{label} {key}"))
            normal = declared - float(declared @ span) * span
            if (
                np.linalg.norm(normal) < 0.5 * np.linalg.norm(declared)
                or not np.any(normal)
            ):
                raise ValueError(
                    f"{label}: {key} is more than 60 deg from perpendicular "
                    "to the span line."
                )
            normals.append(normal / np.linalg.norm(normal))
        if float(normals[0] @ normals[1]) <= 0.0:
            raise ValueError(
                f"{label}: start and end normals differ by 90 deg or more; "
                "split the section into shorter pieces."
            )
        normals = np.array(normals)
        stations.append((label, str(geometry), np.array([root, tip]), normals))
        if mirror:
            # Reversing the mirrored span keeps 2-D +x = span x normal a true
            # mirror image, so the same section solve serves both sides.
            stations.append((f"{label} (mirror)", str(geometry),
                             np.array([tip * _MIRROR_AXIS, root * _MIRROR_AXIS]),
                             normals[::-1] * _MIRROR_AXIS))
    return stations


def _corners(corners, coordinate_units, mirror):
    """Validated axis-frame corner estimates in meters."""

    scale = _unit_scale(coordinate_units, "Coordinate units")
    prepared = []
    for index, corner in enumerate(corners, 1):
        label = f"corner {index}"
        fold = np.array([
            to_axis_frame(_vector(corner["fold_start"], f"{label} fold start")),
            to_axis_frame(_vector(corner["fold_end"], f"{label} fold end")),
        ]) * scale
        n_wing = to_axis_frame(_vector(corner["n_wing"], f"{label} wing normal"))
        n_body = to_axis_frame(_vector(corner["n_body"], f"{label} body normal"))
        try:
            face_width = float(corner["face_width"]) * scale
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: face width must be a number.") from exc
        entry = {"label": label, "fold": fold, "n_wing": n_wing,
                 "n_body": n_body, "face_width": face_width}
        for key in ("internal_phase_deg", "retro_halfwidth_deg"):
            if key in corner:
                entry[key] = float(corner[key])
        prepared.append(entry)
        if mirror:
            prepared.append(dict(
                entry, label=f"{label} (mirror)", fold=fold * _MIRROR_AXIS,
                n_wing=n_wing * _MIRROR_AXIS, n_body=n_body * _MIRROR_AXIS,
            ))
    return prepared


def read_path_points(path, default_normal=(0.0, 0.0, 1.0)):
    """Read a curved path: one ``x y z [nx ny nz]`` point per line.

    Commas or whitespace separate the numbers; blank lines and ``#`` comments
    are skipped. A point without a normal takes ``default_normal``. Returns
    ``(points, normals)`` as (n, 3) arrays in the file's own frame and units.
    """

    points, normals = [], []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        parts = raw.split("#", 1)[0].replace(",", " ").split()
        if not parts:
            continue
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{lineno}: path points must be numeric."
            ) from exc
        if len(numbers) not in (3, 6) or not all(map(math.isfinite, numbers)):
            raise ValueError(
                f"{path}:{lineno}: expected x y z or x y z nx ny nz."
            )
        points.append(numbers[:3])
        normals.append(numbers[3:] if len(numbers) == 6 else list(default_normal))
    if len(points) < 2:
        raise ValueError(f"{path}: a path needs at least two points.")
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)


def sections_from_path(geometry, points, normals):
    """One section per consecutive point pair, normals shared at the joints."""

    return [
        {
            "geometry": geometry,
            "root": tuple(points[index]),
            "tip": tuple(points[index + 1]),
            "normal": tuple(normals[index]),
            "normal_end": tuple(normals[index + 1]),
        }
        for index in range(len(points) - 1)
    ]


def _section_geometry_facts(geometry_path, geometry_scale):
    """(largest distance of a drawn point from the origin in meters, has materials)."""

    radius, has_materials, in_materials = 0.0, False, False
    for line in Path(geometry_path).read_text().splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        head = text.split()[0].lower()
        if head in ("ibcs_resistances:", "dielectrics:"):
            in_materials = True
            continue
        if head in ("title:", "segment:"):
            in_materials = False
            continue
        if in_materials:
            has_materials = True
            continue
        parts = text.split()
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(part) for part in parts)
        except ValueError:
            continue
        radius = max(radius, math.hypot(x1, y1), math.hypot(x2, y2))
    return radius * geometry_scale, has_materials


def _section_angles(step_deg):
    step = float(step_deg)
    count = 180.0 / step if math.isfinite(step) and step > 0.0 else 0.0
    if count < 2 or abs(count - round(count)) > 1e-9:
        raise ValueError("The section angle step must divide 180 (at most 90).")
    return np.arange(2 * int(round(count))) * step


def _half_space_tables(geometry_path, solve_ghz, look_ghz, angles,
                       geometry_units, solver_kwargs, cache):
    """(+normal side, -normal side) coefficient tables of one section solve.

    The section is solved at ``solve_ghz``; the tables are labelled with
    ``look_ghz``, the radar frequency they are expanded at. The two differ
    only for the oblique-incidence correction.
    """

    key = (geometry_path, round(float(solve_ghz), 9), float(look_ghz))
    if key in cache:
        return cache[key]
    from ghost_backend.runs.inputs import load_geometry_snapshot

    snapshot, material_base = load_geometry_snapshot(geometry_path, {})
    full = coefficients_from_2d(
        snapshot, float(solve_ghz), angles.tolist(),
        geometry_units=geometry_units, material_base_dir=material_base,
        label=Path(geometry_path).stem,
        solver_kwargs=dict(solver_kwargs or {}),
    )
    half = len(angles) // 2
    upper = np.arange(half + 1)
    # On the -normal side the placement frame is rotated by 180 deg about the
    # span, so its local angle phi' reads the section at phi' + 180 deg.
    lower = (upper + half) % len(angles)
    tables = tuple(
        SeamCoefficients(float(look_ghz), angles[upper], full.dA_tm[index],
                         full.dA_te[index], label=full.label)
        for index in (upper, lower)
    )
    cache[key] = tables
    return tables


def _reduced_frequencies(look_ghz, cos_min, radius_m, max_solves):
    """Solve frequencies f*cos(tilt) covering cos(tilt) from ``cos_min`` to 1.

    Returns (frequencies ascending, True when ``max_solves`` made the spacing
    coarser than the section size needs).
    """

    if cos_min >= 1.0 - 1e-12:
        return np.array([float(look_ghz)]), False
    low = float(look_ghz) * cos_min
    count = 2
    if radius_m > 0.0:
        # Two-way phase 2*k_t*r must move by less than _SAMPLE_PHASE_RAD.
        step_ghz = _SAMPLE_PHASE_RAD * C0 / (4.0 * math.pi * radius_m) / 1e9
        count = max(2, int(math.ceil((float(look_ghz) - low) / step_ghz)) + 1)
    limit = max(2, int(max_solves))
    return np.linspace(low, float(look_ghz), min(count, limit)), count > limit


def _station_field(table_sets, solve_ghz, segment, normals, frequency_ghz,
                   directions, psi_tm_deg, psi_te_deg, occluder, cancel_check):
    """Axis-frame F_vv/F_hh/F_vh of one station at one frequency.

    ``normals`` holds the start and end normal. When they differ (a curved or
    twisted path) expand_perimeter subdivides the segment and interpolates the
    section frame along it. ``table_sets[j]`` is the half-space table pair
    solved at ``solve_ghz[j]``; each look blends the two solves that bracket
    its transverse frequency f*cos(tilt).
    """

    looks = directions.copy()
    if np.allclose(normals[0], normals[1], rtol=0.0, atol=1e-12):
        # With a turning normal the split plane differs piece by piece, so
        # only isolated short pieces can sit in it; a constant frame needs
        # the looks in its one split plane tipped onto the +normal side.
        d_n = directions @ normals[0]
        in_plane = np.abs(d_n) < _SPLIT_PLANE_TILT
        looks[in_plane] += np.outer(
            _SPLIT_PLANE_TILT - d_n[in_plane], normals[0]
        )
        looks[in_plane] /= np.linalg.norm(looks[in_plane], axis=1)[:, None]

    span = segment[1] - segment[0]
    span = span / np.linalg.norm(span)
    count = len(solve_ghz)
    lower = np.zeros(len(looks), dtype=int)
    blend = np.zeros(len(looks))
    if count > 1:
        cos_tilt = np.sqrt(np.clip(1.0 - (looks @ span) ** 2, 0.0, 1.0))
        position = (
            np.clip(float(frequency_ghz) * cos_tilt, solve_ghz[0], solve_ghz[-1])
            - solve_ghz[0]
        ) / (solve_ghz[1] - solve_ghz[0])
        lower = np.clip(np.floor(position).astype(int), 0, count - 2)
        blend = position - lower

    total = {key: np.zeros(len(looks), dtype=complex)
             for key in ("F_vv", "F_hh", "F_vh")}
    for index, tables in enumerate(table_sets):
        weight = np.where(lower == index, 1.0 - blend, 0.0)
        if index > 0:
            weight = weight + np.where(lower == index - 1, blend, 0.0)
        used = weight > 0.0
        if not np.any(used):
            continue
        for table, side in zip(tables, (normals, -normals)):
            field = expand_perimeter(
                segment[None, :, :], table, None, looks[used],
                frequency_ghz=frequency_ghz,
                psi_tm_deg=psi_tm_deg, psi_te_deg=psi_te_deg,
                grazing_taper_deg=_EDGE_TAPER_DEG,
                occluder=occluder,
                segment_normals=side[None, :, :].copy(),
                cancel_check=cancel_check,
            )
            for key in total:
                total[key][used] += weight[used] * field[key]
    return total


def _load_base(path):
    """Payload, complex field and channel indices of a coherent radar-frame GRIM."""

    from ghost_backend.assembly.fields import RADAR_COMPONENT_FIELD_DOMAIN

    with np.load(path, allow_pickle=False) as stored:
        payload = {key: np.array(stored[key], copy=True) for key in stored.files}
    for key in ("azimuths", "elevations", "frequencies", "polarizations"):
        if key not in payload:
            raise ValueError(f"{path}: the base dataset has no {key} axis.")
    domain = payload.get("complex_field_domain")
    if domain is not None and str(np.asarray(domain).reshape(()).item()) != (
        RADAR_COMPONENT_FIELD_DOMAIN
    ):
        raise ValueError(
            f"{path}: the base must be a coherent radar-frame monostatic "
            "response (a BoR deliverable, an Assembly output or an imported "
            "3-D body), not an aspect-only or 2-D table."
        )
    if "rcs_amp_real" in payload and "rcs_amp_imag" in payload:
        amp = (np.asarray(payload["rcs_amp_real"], dtype=float)
               + 1j * np.asarray(payload["rcs_amp_imag"], dtype=float))
    elif "rcs_power" in payload and "rcs_phase" in payload:
        amp = (np.sqrt(np.asarray(payload["rcs_power"], dtype=float)
                       / (4.0 * math.pi))
               * np.exp(1j * np.asarray(payload["rcs_phase"], dtype=float)))
    else:
        raise ValueError(
            f"{path}: the base dataset holds no complex field; a power-only "
            "response cannot be added to coherently."
        )
    labels = [str(value).strip().upper()
              for value in np.asarray(payload["polarizations"]).ravel()]
    expected = (len(payload["azimuths"]), len(payload["elevations"]),
                len(payload["frequencies"]), len(labels))
    if amp.shape != expected:
        raise ValueError(
            f"{path}: field shape {amp.shape} does not match its "
            f"azimuth/elevation/frequency/polarization axes {expected}."
        )
    channels = {}
    for name, aliases in (("vv", ("VV",)), ("hh", ("HH",)), ("vh", ("VH", "HV"))):
        for alias in aliases:
            if alias in labels:
                channels[name] = labels.index(alias)
                break
    if "vv" not in channels or "hh" not in channels:
        raise ValueError(f"{path}: the base dataset needs VV and HH channels.")
    return payload, amp, channels


def expand_wing_sections(
    sections,
    *,
    output_grim,
    coordinate_units,
    geometry_units,
    body_grim=None,
    mirror=False,
    frequencies_ghz=None,
    azimuths_deg=None,
    elevations_deg=None,
    axis_az_deg=None,
    axis_el_deg=None,
    roll_deg=None,
    section_angle_step_deg=0.5,
    section_solver_kwargs=None,
    shadow=True,
    oblique=False,
    oblique_max_tilt_deg=60.0,
    oblique_max_solves=24,
    psi_tm_deg=PSI_HH_DEG,
    psi_te_deg=PSI_VV_DEG,
    corners=(),
    base_dir=None,
    cancel_check=None,
    progress_callback=None,
):
    """Line-expand 2-D sections, add them to the optional base, save one GRIM.

    ``sections`` is a sequence of ``{"geometry", "root", "tip", "normal"}``
    mappings in the Assembly CAD frame. An optional ``"normal_end"`` makes the
    section frame turn from ``normal`` at the root to ``normal_end`` at the
    tip, so a chain of short sections follows a curved or twisted path
    (see :func:`sections_from_path`). ``corners`` holds
    ``{"fold_start", "fold_end", "n_wing", "n_body", "face_width"}`` mappings
    in the same frame and units. ``body_grim`` is the base response described
    in the module docstring; with a base, its own radar grid is used.

    Nothing is written until every station has been expanded; a cancelled or
    failed run leaves an existing output untouched. Returns
    ``{"output", "stations", "warnings"}``.
    """

    from ghost_backend.assembly.fields import (
        corner_amplitude, export_radar_grim, load_body_requested_radar_grid,
        radar_frame_basis,
    )
    from ghost_backend.io.grim import _save_grim_npz

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Line expansion cancelled; existing output kept.")

    def resolve(value):
        path = Path(str(value).strip())
        return path if path.is_absolute() else Path(base_dir) / path

    base_dir = PROJECT_ROOT if base_dir is None else base_dir
    if not str(output_grim or "").strip():
        raise ValueError("Choose an output .grim path.")
    stations = _stations(sections, coordinate_units, bool(mirror), base_dir)
    corner_entries = _corners(corners, coordinate_units, bool(mirror))
    angles = _section_angles(section_angle_step_deg)
    geometry_scale = _unit_scale(geometry_units, "Section geometry units")
    max_tilt = float(oblique_max_tilt_deg)
    if oblique and not 0.0 < max_tilt < 90.0:
        raise ValueError("The oblique tilt limit must be between 0 and 90 deg.")

    destination = str(resolve(output_grim))
    if not destination.lower().endswith(".grim"):
        destination += ".grim"
    warnings = []
    base_payload, base_amp, channels, stored, body_path = None, None, None, {}, ""
    if str(body_grim or "").strip():
        body_path = str(resolve(body_grim))
        if os.path.normcase(os.path.abspath(destination)) == os.path.normcase(
            os.path.abspath(body_path)
        ):
            raise ValueError("The output must not replace the base dataset.")
        if any(value is not None
               for value in (frequencies_ghz, azimuths_deg, elevations_deg)):
            raise ValueError(
                "Leave frequencies, azimuths and elevations blank when a base "
                "dataset is chosen; the sections are added on its own grid."
            )
        base_payload, base_amp, channels = _load_base(body_path)
        try:
            stored = load_body_requested_radar_grid(body_path) or {}
        except (KeyError, TypeError, ValueError):
            stored = {}
        frequencies = [float(f) for f in base_payload["frequencies"]]
        azimuths = [float(a) for a in base_payload["azimuths"]]
        elevations = [float(e) for e in base_payload["elevations"]]
        if "vh" not in channels:
            warnings.append(
                "The base dataset has no VH channel; the cross-polarized part "
                "of the sections and corners is not saved."
            )
    else:
        for value, label in ((frequencies_ghz, "frequencies"),
                             (azimuths_deg, "azimuths"),
                             (elevations_deg, "elevations")):
            if value is None:
                raise ValueError(f"Enter {label} (no base dataset supplies them).")
        frequencies = [float(f) for f in frequencies_ghz]
        azimuths = [float(a) for a in azimuths_deg]
        elevations = [float(e) for e in elevations_deg]

    def attitude_value(value, key):
        if value is not None:
            return float(value)
        return float(stored.get(key, 0.0))

    attitude = dict(
        axis_az_deg=attitude_value(axis_az_deg, "axis_az_deg"),
        axis_el_deg=attitude_value(axis_el_deg, "axis_el_deg"),
        roll_deg=attitude_value(roll_deg, "roll_deg"),
    )
    directions, basis = radar_frame_basis(azimuths, elevations, **attitude)

    descriptions = []
    step_rad = math.radians(float(section_angle_step_deg))
    k_max = 2.0 * math.pi * max(frequencies) * 1e9 / C0
    facts = {}
    cos_min = 1.0
    for label, geometry, segment, normals in stations:
        span = segment[1] - segment[0]
        length = float(np.linalg.norm(span))
        x_axis = np.cross(span / length, normals[0])
        turn = math.degrees(math.acos(min(1.0, float(normals[0] @ normals[1]))))
        descriptions.append(
            f"{label}: {Path(geometry).name}, span {length:.4g} m, "
            f"2-D +x = {np.round(x_axis, 4).tolist()}, "
            f"2-D +y = {np.round(normals[0], 4).tolist()} at the start "
            f"(axis frame)"
            + (f", frame turns {turn:.3g} deg along it" if turn > 1e-6 else "")
        )
        if oblique:
            tilt_cos = np.sqrt(np.clip(
                1.0 - (directions @ (span / length)) ** 2, 0.0, 1.0
            ))
            cos_min = min(cos_min, float(np.min(tilt_cos)))
        if geometry in facts:
            continue
        radius, has_materials = facts[geometry] = _section_geometry_facts(
            geometry, geometry_scale
        )
        if radius > 0.0 and 2.0 * k_max * radius * step_rad > _SAMPLE_PHASE_RAD:
            suggested = math.degrees(_SAMPLE_PHASE_RAD / (2.0 * k_max * radius))
            warnings.append(
                f"A {float(section_angle_step_deg):g} deg section angle step "
                f"under-samples {Path(geometry).name} at "
                f"{max(frequencies):g} GHz (it reaches {radius:.3g} m from its "
                f"origin); use {suggested:.2g} deg or finer."
            )
        if oblique and has_materials:
            warnings.append(
                f"{Path(geometry).name} defines materials; the oblique "
                "correction is exact only for PEC sections."
            )
    if oblique:
        cos_min = max(cos_min, math.cos(math.radians(max_tilt)))

    occluder = None
    if shadow and base_payload is not None and (
        "body_profile_rho_m" in base_payload and "body_profile_z_m" in base_payload
    ):
        from ghost_backend.assembly.workflow import bor_shadow_triangles
        from ghost_backend.geometry.occlusion import Occluder

        if "body_profile_kind" in base_payload:
            profile_kind = str(np.asarray(base_payload["body_profile_kind"]).item())
            if profile_kind != "outer_boundary":
                raise ValueError("A transmitting sheet profile cannot supply opaque body "
                                 "shadowing. Disable body shadowing for this base response.")
        profile = np.column_stack((
            np.asarray(base_payload["body_profile_rho_m"], dtype=float).ravel(),
            np.asarray(base_payload["body_profile_z_m"], dtype=float).ravel(),
        ))
        try:
            triangles, _report = bor_shadow_triangles(
                profile,
                max_sag_m=C0 / (max(frequencies) * 1e9) / 16.0,
                normal_tolerance_deg=2.0,
            )
            occluder = Occluder(triangles)
        except (MemoryError, ValueError) as exc:
            warnings.append(f"Body shadowing was skipped: {exc}")
    elif shadow and base_payload is not None:
        warnings.append(
            "The base dataset has no embedded BoR profile, so the sections "
            "are not shadowed by the body."
        )

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    staged = os.path.join(
        os.path.dirname(destination),
        f".{os.path.basename(destination)}.tmp.{os.getpid()}.grim",
    )
    history = (
        f"line expansion: {len(stations)} 2-D section station(s), "
        f"{len(corner_entries)} corner estimate(s), "
        f"psi_tm={float(psi_tm_deg):g} psi_te={float(psi_te_deg):g}, "
        f"shadow={'on' if occluder is not None else 'off'}, "
        f"oblique={'on' if oblique else 'off'}"
    )
    try:
        check_cancel()
        if base_payload is None:
            # A zero field in the standard radar-frame layout carries every
            # convention tag the GRIM readers expect.
            _saved, payload = export_radar_grim(
                staged, bor_result=None, placements=[],
                frequencies_ghz=frequencies, azimuths_deg=azimuths,
                elevations_deg=elevations, history=history,
                cancel_check=cancel_check, _return_payload=True, **attitude,
            )
            amp = payload.pop("_amp")
            channels = {"vv": 0, "hh": 1, "vh": 2}
        else:
            payload, amp = base_payload, base_amp
            previous = str(np.asarray(payload.get("history", "")).reshape(()).item())
            payload["history"] = np.asarray(
                (previous + " | " + history).strip(" |")
            )
        shape = (len(azimuths), len(elevations))
        cache = {}
        total_steps = len(frequencies) * (len(stations) + len(corner_entries))
        done = 0
        coarse_oblique = False
        for index, frequency in enumerate(frequencies):
            scatter = np.zeros((len(directions), 2, 2), dtype=complex)

            def add(field):
                scatter[:, 0, 0] += field["F_vv"]
                scatter[:, 1, 1] += field["F_hh"]
                scatter[:, 0, 1] += field["F_vh"]
                scatter[:, 1, 0] += field["F_vh"]

            for label, geometry, segment, normals in stations:
                check_cancel()
                solve_ghz, capped = _reduced_frequencies(
                    frequency, cos_min if oblique else 1.0,
                    facts[geometry][0], oblique_max_solves,
                )
                coarse_oblique = coarse_oblique or capped
                table_sets = []
                for solve_index, value in enumerate(solve_ghz):
                    check_cancel()
                    if progress_callback is not None:
                        progress_callback(
                            done, total_steps,
                            f"{frequency:g} GHz - {label}"
                            + (f", solve {solve_index + 1}/{len(solve_ghz)}"
                               if len(solve_ghz) > 1 else ""),
                        )
                    table_sets.append(_half_space_tables(
                        geometry, value, frequency, angles, geometry_units,
                        section_solver_kwargs, cache,
                    ))
                add(_station_field(
                    table_sets, solve_ghz, segment, normals, frequency,
                    directions, float(psi_tm_deg), float(psi_te_deg),
                    occluder, cancel_check,
                ))
                done += 1
            for corner in corner_entries:
                check_cancel()
                if progress_callback is not None:
                    progress_callback(
                        done, total_steps,
                        f"{frequency:g} GHz - {corner['label']}",
                    )
                field = corner_amplitude(
                    corner["fold"], corner["n_wing"], corner["n_body"],
                    corner["face_width"], directions, frequency,
                    internal_phase_deg=corner.get("internal_phase_deg", 0.0),
                    retro_halfwidth_deg=corner.get("retro_halfwidth_deg", 45.0),
                    occluder=occluder,
                )
                note = field.pop("warning", None)
                if note and index == 0:
                    warnings.append(f"{corner['label']}: {note}")
                add(field)
                done += 1
            radar = np.einsum("nai,nab,nbj->nij", basis, scatter, basis)
            amp[:, :, index, channels["vv"]] += radar[:, 0, 0].reshape(shape)
            amp[:, :, index, channels["hh"]] += radar[:, 1, 1].reshape(shape)
            if "vh" in channels:
                amp[:, :, index, channels["vh"]] += radar[:, 0, 1].reshape(shape)
        check_cancel()
        if coarse_oblique:
            warnings.append(
                f"The oblique correction was limited to {int(oblique_max_solves)} "
                "solves per section and frequency, which is coarser than the "
                "section size needs; raise the limit or lower the tilt limit."
            )
        payload["rcs_amp_real"], payload["rcs_amp_imag"] = amp.real, amp.imag
        payload["raw_complex_amplitude_preserved"] = np.asarray(True)
        payload["rcs_power"] = (
            4.0 * math.pi * np.abs(amp) ** 2
        ).astype(np.float32)
        payload["rcs_phase"] = np.angle(amp).astype(np.float32)
        payload["line_expansion_provenance_json"] = np.asarray(json.dumps({
            "schema": "ghost.assembly.line-expansion.v1",
            "base": os.path.basename(body_path),
            "stations": descriptions,
            "corner_estimate_count": len(corner_entries),
            "phase_mapping_deg": {"TM": float(psi_tm_deg), "TE": float(psi_te_deg)},
            "body_shadowing": occluder is not None,
            "oblique_correction": bool(oblique),
            "model_scope": "single bounce; no end, junction or mutual coupling",
        }, sort_keys=True))
        _save_grim_npz(payload, staged)
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    if progress_callback is not None:
        progress_callback(1, 1, "Line expansion saved")
    return {
        "output": os.path.abspath(destination),
        "stations": descriptions,
        "warnings": warnings,
    }


def main():
    try:
        result = expand_wing_sections(
            SECTIONS,
            output_grim=OUTPUT_GRIM,
            coordinate_units=COORDINATE_UNITS,
            geometry_units=GEOMETRY_UNITS,
            body_grim=BODY_GRIM,
            mirror=MIRROR,
            frequencies_ghz=FREQUENCIES_GHZ,
            azimuths_deg=AZIMUTHS_DEG,
            elevations_deg=ELEVATIONS_DEG,
            axis_az_deg=AXIS_AZ_DEG,
            axis_el_deg=AXIS_EL_DEG,
            roll_deg=ROLL_DEG,
            section_angle_step_deg=SECTION_ANGLE_STEP_DEG,
            section_solver_kwargs=SECTION_SOLVER_KWARGS,
            shadow=SHADOW,
            oblique=OBLIQUE,
            oblique_max_tilt_deg=OBLIQUE_MAX_TILT_DEG,
            oblique_max_solves=OBLIQUE_MAX_SOLVES,
            psi_tm_deg=PSI_TM_DEG,
            psi_te_deg=PSI_TE_DEG,
            corners=CORNERS,
            base_dir=PROJECT_ROOT,
            progress_callback=lambda done, total, message: print(
                f"  [{done}/{total}] {message}", flush=True
            ),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Stations: {len(result['stations'])}")
    for line in result["stations"]:
        print(f"  {line}")
    for warning in result["warnings"]:
        print(f"  WARNING: {warning}")
    print(f"Wrote one combined monostatic dataset: {result['output']}")


if __name__ == "__main__":
    main()
