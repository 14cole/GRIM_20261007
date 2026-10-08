"""Body-of-revolution geometry dispatch, solves, and result packaging."""
from ghost_backend.bor.options import (configured, current_options, compressed_requested,
    reserve_output, output_reserved_gb, estimate_output_gb)

import cmath
import math
import os
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from ghost_backend.execution.metrics import active_metrics, profiled_solve, timed_stage

from ghost_backend.bor.kernels import C0, FAR_GAUSS_ORDER
from ghost_backend.bor.solver import (
    BOR_CONDITION_EST_MAX,
    BOR_LINEAR_BACKWARD_ERROR_MAX,
    BOR_LINEAR_RESIDUAL_MAX,
    solve_bor,
    solve_bor_dielectric,
    solve_bor_coated_pec,
    solve_bor_partial_coating,
    solve_bor_coated2_pec,
    solve_bor_coated_n_pec,
    solve_bor_coating_patch,
    solve_bor_banded_multiregion,
    _bor_mode_limits,
)
from ghost_backend.twod.solver import (
    MaterialLibrary,
    _conservative_mesh_wavelength_for_frequencies,
    _material_base_dir_for_snapshot,
    validate_geometry_snapshot_for_solver,
)
from ghost_backend.runs.quality import (
    evaluate_mesh_convergence,
    scale_snapshot_panel_density,
    validate_mesh_convergence_policy,
)
from ghost_backend.twod.preparation import prepare_geometry

DEFAULT_ELEMENTS_PER_WAVELENGTH = 20
MAX_ELEMENTS_DEFAULT = 50_000
BOR_POWER_CONSISTENCY_RTOL = 2.0e-8
BOR_STREAM_BUDGET_GB_DEFAULT = 8.0


def _unit_scale_to_meters(units: 'str') -> 'float':
    value = str(units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")


def _parse_flag(tok: 'Any', default: 'int' = 0) -> 'int':
    try:
        text = str(tok).strip().lower()
        if text.startswith("mat."):
            text = text[4:]
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _parse_int(tok: 'Any', default: 'int' = 0) -> 'int':
    try:
        text = str(tok).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _reject_unsupported_bor_ibc_interfaces(
    geometry_snapshot: 'Dict[str, Any]',
) -> 'None':
    """Reject impedance flags that the selected BoR material paths ignore.

    TYPE 2 is the only BoR segment whose IBC is currently assembled.  The
    2-D solver supports a TYPE 4 Robin backing, but the BoR coated/multiregion
    formulations currently model TYPE 4 as PEC, so accepting its flag would
    silently change the requested boundary condition.
    """

    for seg_idx, seg in enumerate(geometry_snapshot.get("segments", []) or []):
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip()
            else seg.get("seg_type", 2),
            2,
        )
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        if ibc_flag > 0 and seg_type in (3, 4, 5):
            name = str(seg.get("name", f"segment_{seg_idx + 1}"))
            raise ValueError(
                f"BoR TYPE {seg_type} segment '{name}' assigns IBC flag "
                f"{ibc_flag}, but impedance on TYPE {seg_type} is not "
                "implemented by the BoR material formulation. Remove the "
                "flag or place the IBC on a supported TYPE 2 conductor."
            )


class _SegChain:
    """One segment's polyline in scaled (rho, z) coordinates."""

    def __init__(self, name: 'str', seg_type: 'int', n_prop: 'int', ibc_flag: 'int',
                 pos_mat: 'int', neg_mat: 'int', pts: 'np.ndarray',
                 certification_base_n: 'Optional[int]' = None,
                 certification_refinement_factor: 'float' = 1.0):
        self.name = name
        self.seg_type = seg_type
        self.n_prop = n_prop
        self.ibc_flag = ibc_flag
        self.pos_mat = pos_mat
        self.neg_mat = neg_mat
        self.certification_base_n = certification_base_n
        self.certification_refinement_factor = float(
            certification_refinement_factor
        )
        self.pts = pts
        d = np.diff(pts, axis=0)
        self.prim_lengths = np.hypot(d[:, 0], d[:, 1])
        self.length = float(np.sum(self.prim_lengths))
        # Set per frequency by _mark_impedance_junctions: this end meets a
        # chain whose evaluated surface impedance jumps.
        self.grade_start = False
        self.grade_end = False


def _chains_from_snapshot(snapshot: 'Dict[str, Any]', scale: 'float') -> 'List[_SegChain]':
    chains: 'List[_SegChain]' = []
    segments = list(snapshot.get("segments", []) or [])
    refinement_factor = float(
        snapshot.get("_bor_certification_refinement_factor", 1.0) or 1.0
    )
    base_segment_n = list(
        snapshot.get("_bor_certification_base_segment_n", []) or []
    )
    if not math.isfinite(refinement_factor) or refinement_factor < 1.0:
        raise ValueError(
            "Internal BoR certification refinement factor must be finite "
            "and >= 1."
        )
    if refinement_factor > 1.0 and len(base_segment_n) != len(segments):
        raise ValueError(
            "Internal BoR certification refinement metadata does not match "
            "the geometry segment count."
        )

    for seg_idx, seg in enumerate(segments):
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2), 2)
        n_prop = _parse_int(props[1] if len(props) > 1 else 0, 0)
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        pos_mat = _parse_flag(props[3] if len(props) > 3 else 0)
        neg_mat = _parse_flag(props[4] if len(props) > 4 else 0)
        pts: 'List[Tuple[float, float]]' = []
        for i, pair in enumerate(list(seg.get("point_pairs", []) or [])):
            x1 = float(pair.get("x1", 0.0)) * scale
            y1 = float(pair.get("y1", 0.0)) * scale
            x2 = float(pair.get("x2", 0.0)) * scale
            y2 = float(pair.get("y2", 0.0)) * scale
            if i == 0:
                pts.append((x1, y1))
            elif math.hypot(x1 - pts[-1][0], y1 - pts[-1][1]) > 0:

                raise ValueError(
                    f"Segment '{seg.get('name', seg_idx)}': primitives do not chain "
                    f"head-to-tail at ({x1 / scale:.6g}, {y1 / scale:.6g}).")
            pts.append((x2, y2))
        if len(pts) < 2:
            continue
        certification_base_n = (
            _parse_int(base_segment_n[seg_idx], 0)
            if refinement_factor > 1.0 else None
        )
        chains.append(_SegChain(
            str(seg.get("name", f"segment_{seg_idx + 1}")),
            seg_type,
            n_prop,
            ibc_flag,
            pos_mat,
            neg_mat,
            np.asarray(pts, dtype=float),
            certification_base_n=certification_base_n,
            certification_refinement_factor=refinement_factor,
        ))
    if not chains:
        raise ValueError("Geometry contains no usable segments.")
    return chains


def _stitch_generatrix(chains: 'List[_SegChain]', what: 'str',
                       tol: 'float') -> 'List[_SegChain]':
    """Order chains head-to-tail into a single generatrix run.  Junctions
    must match END of one chain to START of the next AS DRAWN (reversing a
    segment silently would flip its normal and its taper direction)."""

    if len(chains) == 1:
        return chains

    def key(p) -> 'Tuple[int, int]':
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"The {what} segments do not form a single chain "
                             "(two segments start or end at the same point).")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    if len(heads) != 1:
        raise ValueError(
            f"The {what} segments do not chain head-to-tail into one generatrix. "
            "Check that consecutive segments share endpoints and that each "
            "segment is drawn in the same traversal direction (a start-to-start "
            "or end-to-end meeting means one segment's endpoint order must be "
            "reversed).")
    ordered = [heads[0]]
    while True:
        nxt = start_of.get(key(ordered[-1].pts[-1]))
        if nxt is None:
            break
        if nxt is ordered[0]:
            raise ValueError(f"The {what} segments form a closed loop in the "
                             "(rho, z) plane; a BoR generatrix must be an open "
                             "polyline with both endpoints on the axis.")
        ordered.append(nxt)
    if len(ordered) != len(chains):
        raise ValueError(f"The {what} segments split into multiple disconnected "
                         "chains; expected one generatrix.")
    return ordered


def _preflight_generatrix(ordered: 'List[_SegChain]', what: 'str', tol: 'float') -> 'None':
    pts = np.vstack([ordered[0].pts] + [c.pts[1:] for c in ordered[1:]])
    rho, z = pts[:, 0], pts[:, 1]
    if np.any(rho < -tol):
        bad = pts[np.argmin(rho)]
        raise ValueError(
            f"The {what} generatrix crosses the rotation axis (rho = x = "
            f"{bad[0]:.6g} < 0). Draw the half-profile entirely at x >= 0.")
    if rho[0] > tol or rho[-1] > tol:
        raise ValueError(
            f"The {what} generatrix endpoints must lie ON the rotation axis "
            f"(x = 0) to close the body of revolution; got start rho = "
            f"{rho[0]:.6g}, end rho = {rho[-1]:.6g}. Open BoR shells are not "
            "supported in phase 4.")
    if z[0] <= z[-1]:
        raise ValueError(
            f"The {what} generatrix must be traversed from the +z (top) axis "
            "end to the -z (bottom) axis end so the left-of-travel normal "
            f"faces the exterior; it is drawn bottom-to-top (z {z[0]:.6g} -> "
            f"{z[-1]:.6g}). Reverse the segment endpoint order.")


def _element_count(n_prop: 'int', prim_len: 'float', lam_target: 'float') -> 'int':
    if prim_len <= 0.0:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    n_wave = abs(n_prop) if n_prop < 0 else DEFAULT_ELEMENTS_PER_WAVELENGTH
    target = lam_target / max(1, n_wave)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("The controlling mesh wavelength must be positive and finite.")
    return max(1, int(math.ceil(prim_len / target)))


def _chain_element_count(
    chain: '_SegChain', prim_len: 'float', lam_target: 'float'
) -> 'int':
    """Return a primitive count, refining the realized base discretization."""

    factor = float(chain.certification_refinement_factor)
    if factor > 1.0:
        base_count = _element_count(
            int(chain.certification_base_n or 0), prim_len, lam_target
        )
        return max(base_count + 1, int(math.ceil(base_count * factor)))
    return _element_count(chain.n_prop, prim_len, lam_target)


# Where the surface impedance of a closed conductor jumps, the current has the
# singularity of the 2D junction and uniform elements converge at first order
# (PEC | 100+50j sphere, CFIE: 0.20, 0.11, 0.055 dB at 30, 60, 120 elements).
# Splitting the element on each side geometrically, four levels as in the 2D
# mesher, gives 0.058, 0.015, 0.004 dB for eight more elements (three levels:
# 0.072, 0.021, 0.007 dB; five: 0.050, 0.011, 0.002 dB). The near angular rule
# used to refuse elements that small (a rounding floor of the sampled brackets,
# see kernels._stable_brackets).
BOR_JUNCTION_GRADING_LEVELS = 4
BOR_JUNCTION_GRADING_RATIO = 0.5


def _mark_impedance_junctions(ordered: 'List[_SegChain]', materials, freq_ghz: 'float') -> 'int':
    """Mark the chain ends of one stitched conductor where the evaluated impedance jumps.

    The rule is the 2D mesher's (``impedance_jump_is_graded``): the law at the
    chain end, at this frequency, PEC being zero. Marks are per frequency, so
    they are reset here; preview and solve call this before they count or mesh.
    Returns the number of graded junctions.
    """
    from ghost_backend.twod.geometry import EPS, impedance_jump_is_graded

    def law(chain, arc_s):
        if chain.ibc_flag <= 0:
            return 0.0j
        value = complex(materials.get_impedance(chain.ibc_flag, freq_ghz, arc_s=arc_s))
        return 0.0j if abs(value) <= EPS else value

    for chain in ordered:
        chain.grade_start = chain.grade_end = False
    graded = 0
    for before, after in zip(ordered[:-1], ordered[1:]):
        if impedance_jump_is_graded(law(before, 1.0), law(after, 0.0)):
            before.grade_end = after.grade_start = True
            graded += 1
    return graded


def _primitive_mesh_plan(chain: '_SegChain', index: 'int', count: 'int'):
    """Allocation-free subdivision plan shared by counting and meshing."""
    at_start = chain.grade_start and index == 0
    at_end = chain.grade_end and index == len(chain.pts) - 2
    if count == 1 and at_start and at_end:
        count = 2
    levels = max(0, int(BOR_JUNCTION_GRADING_LEVELS))
    return count, at_start, at_end, levels


def _primitive_breaks(chain: '_SegChain', index: 'int', count: 'int') -> 'List[float]':
    """Parametric boundaries; callers admit the element count before allocating."""
    count, at_start, at_end, levels = _primitive_mesh_plan(chain, index, count)
    breaks = [i / count for i in range(count + 1)]
    if not (at_start or at_end):
        return breaks
    cuts = [BOR_JUNCTION_GRADING_RATIO ** level for level in range(levels, 0, -1)]
    if at_start:
        breaks = [0.0] + [breaks[1] * cut for cut in cuts] + breaks[1:]
    if at_end:
        breaks = breaks[:-1] + [1.0 - (1.0 - breaks[-2]) * cut for cut in reversed(cuts)] + [1.0]
    return breaks


def _mesh_generatrix(ordered: 'List[_SegChain]', lam_target: 'float',
                     max_elements: 'int', axis_tol: 'float'):
    """Subdivide the ordered chains into elements.  Returns (points [Nn,2],
    elem_seg [Ne] chain index, elem_arc_s [Ne] normalized arc position of the
    element midpoint along its own segment -- the taper coordinate)."""

    planned_elements = _run_element_count(ordered, lam_target)
    if planned_elements > max_elements:
        raise ValueError(f"BoR mesh would need {planned_elements} elements "
                         f"(> max {max_elements}). Reduce frequency or density.")
    points: 'List[Tuple[float, float]]' = []
    elem_seg: 'List[int]' = []
    elem_arc: 'List[float]' = []
    for ci, c in enumerate(ordered):
        arc0 = 0.0
        for pi in range(len(c.pts) - 1):
            p0, p1 = c.pts[pi], c.pts[pi + 1]
            plen = c.prim_lengths[pi]
            cnt = _chain_element_count(c, plen, lam_target)
            if c.grade_start or c.grade_end:
                breaks = _primitive_breaks(c, pi, cnt)
                for t0, t1 in zip(breaks[:-1], breaks[1:]):
                    if not points:
                        points.append(tuple(p0 + (p1 - p0) * t0))
                    elem_seg.append(ci)
                    elem_arc.append((arc0 + plen * 0.5 * (t0 + t1)) / max(c.length, 1e-300))
                    points.append(tuple(p0 + (p1 - p0) * t1))
                arc0 += plen
                continue
            for i in range(cnt):
                q0 = p0 + (p1 - p0) * (i / cnt)
                if not points:
                    points.append(tuple(q0))
                elem_seg.append(ci)
                elem_arc.append((arc0 + plen * (i + 0.5) / cnt) / max(c.length, 1e-300))
                q1 = p0 + (p1 - p0) * ((i + 1) / cnt)
                points.append(tuple(q1))
            arc0 += plen
    pts = np.asarray(points, dtype=float)

    pts[np.abs(pts[:, 0]) <= axis_tol, 0] = 0.0
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    if len(pts) - 1 > max_elements:
        raise ValueError(f"BoR mesh would need {len(pts) - 1} elements "
                         f"(> max {max_elements}). Reduce frequency or density.")
    return pts, np.asarray(elem_seg, dtype=int), np.asarray(elem_arc, dtype=float)


def _conductor_formulation(zs_elements):
    values = np.asarray(zs_elements, complex)
    if not np.any(abs(values) > 0):
        return "cfie"
    # Physical resistance does not remove the EFIE representation's interior
    # resonances. The closed-body CFIE covers lossy and spatially varying Zs.
    return "cfie"


def _classify(chains: 'List[_SegChain]') -> 'str':
    types = {c.seg_type for c in chains}
    if types in ({1}, {1, 2}):
        if any(c.seg_type == 2 and c.ibc_flag > 0 for c in chains):
            raise ValueError("BoR sheets can join pure PEC, but not an opaque IBC or dielectric interface.")
        return "sheet"
    if types == {2}:
        return "conductor"
    if types == {3}:
        if len({c.pos_mat for c in chains}) != 1:
            raise ValueError("All TYPE 3 segments of a homogeneous BoR body "
                             "must reference the same pos_mat material.")
        return "dielectric"
    if types in ({3, 4}, {2, 3, 4}):
        pm3 = {c.pos_mat for c in chains if c.seg_type == 3}
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if len(pm3) != 1 or len(pm4) != 1 or pm3 != pm4:
            raise ValueError("Coated BoR: the TYPE 3 interface and the TYPE 4 "
                             "covered core must reference the same pos_mat "
                             "coating material.")
        if 2 in types:
            return "partial"
        return "coated"
    if types == {2, 3, 4, 5}:


        return "banded"
    if types == {3, 4, 5}:
        if len({c.pos_mat for c in chains if c.seg_type == 4}) > 1:
            return "banded"
        pm5 = {(c.pos_mat, c.neg_mat) for c in chains if c.seg_type == 5}
        if len(pm5) > 1:
            return "layered_n"
        outer_flag, inner_flag = next(iter(pm5))
        if outer_flag <= 0 or inner_flag <= 0 or outer_flag == inner_flag:
            raise ValueError("TYPE 5 needs distinct positive pos_mat (outer "
                             "layer) and neg_mat (inner layer) flags.")
        pm4 = {c.pos_mat for c in chains if c.seg_type == 4}
        if pm4 != {inner_flag}:
            raise ValueError("Layered BoR: the TYPE 4 core's pos_mat must be "
                             "the TYPE 5 interface's neg_mat (inner layer).")
        for c in chains:
            if c.seg_type == 3 and c.pos_mat not in (outer_flag, inner_flag):
                raise ValueError("Layered BoR: every TYPE 3 pos_mat must be "
                                 "the outer-layer or inner-layer flag.")
        return "layered"
    unsupported = types - {2, 3, 4, 5}
    if unsupported:
        raise ValueError(f"Segment TYPE(s) {sorted(unsupported)} are not "
                         "supported by the BoR solver (supported: TYPE 2 "
                         "PEC/IBC, TYPE 3 dielectric, TYPE 3+4 coated, "
                         "TYPE 2+3+4 partially coated, TYPE 3+5+4 layered).")
    raise ValueError("Unsupported BoR material combination: TYPE 2 "
                     "conductors can only mix with dielectric interfaces via "
                     "the TYPE 2+3+4 partial-coating layout.")


def _validate_bor_far_controls(kind: 'str', assembly: 'str',
                               table_precision: 'str',
                               stream_budget_gb: 'float') -> 'None':
    """Apply the same far-table control policy to preview and solve paths."""

    del assembly, table_precision, stream_budget_gb
    if kind not in {
        "conductor", "sheet", "dielectric", "coated", "partial", "layered",
        "layered_n", "banded",
    }:
        raise ValueError(f"Unsupported BoR assembly-control kind {kind!r}.")


def resolve_automatic_factorization(
    arguments: 'Mapping[str, Any]', certified: 'bool' = False,
) -> 'str':
    """Pick the BOR backend from the memory estimate, as the 2-D entries do.

    Dense/streaming is kept while every frequency fits the memory limit.
    Only a resource-cap failure selects compressed as a fallback; validation
    errors propagate. Dispersive materials require pricing every frequency.
    """

    return resolve_automatic_plan(arguments, certified)[0]


def resolve_automatic_plan(
    arguments: 'Mapping[str, Any]', certified: 'bool' = False,
) -> 'Tuple[str, Optional[str]]':
    """``(factorization, assembly)`` of an automatic call.

    ``assembly`` is None unless the chooser replaces the caller's ``'auto'``:
    when the dense plan of the solvers' own assembly decision cannot fit
    (conductors stream; the material solvers choose tables below a fixed 2 GB
    whatever the limit is), dense streaming is priced next and compression is
    selected only when neither fits. Snapshot entries choose once for the whole sweep, so an imposed
    streamed assembly applies to every frequency; it costs about what tables
    cost, and compression several times more.
    """

    from ghost_backend.bor.memory import solve_memory_limit_gb as _solve_memory_limit_gb

    supplied = dict(arguments)
    nested = supplied.pop('kwargs', None)
    if isinstance(nested, dict):
        supplied.update(nested)
    if 'cfie_alpha' in supplied:
        alpha = float(supplied['cfie_alpha'])
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('BoR CFIE alpha must be finite and satisfy 0 < alpha < 1.')
    snapshot = supplied.get('geometry_snapshot')
    frequencies = supplied.get('frequencies_ghz')
    aspects = supplied.get('elevations_deg')
    # Resource previews use a scalar frequency and the explicit aspect name.
    # Resolve exactly the same backend as a solve instead of defaulting dense.
    if frequencies is None and supplied.get('frequency_ghz') is not None:
        frequencies = [supplied['frequency_ghz']]
    if aspects is None:
        aspects = supplied.get('aspects_deg')
    certified = bool(supplied.get('mesh_certification', certified))
    if snapshot is None:
        return _resolve_direct_plan(supplied)
    if frequencies is None or not len(frequencies) or aspects is None or not len(aspects):
        return 'dense', None
    # Compressed assembly requires double precision; both dense assemblies take single.
    single = str(supplied.get('table_precision', 'auto')).strip().lower() == 'single'
    checkpoint = supplied.get('check_abort')
    def price(frequency, assembly):
        if checkpoint is not None:
            checkpoint()
        return estimate_bor_resources(
            snapshot, float(frequency), aspects,
            geometry_units=supplied.get('geometry_units', 'inches'),
            material_base_dir=supplied.get('material_base_dir'),
            n_modes=supplied.get('n_modes'),
            max_elements=supplied.get('max_elements', MAX_ELEMENTS_DEFAULT),
            workers=max(1, int(supplied.get('workers') or max(1, (os.cpu_count() or 2)-1))),
            table_precision=supplied.get('table_precision', 'auto'),
            assembly=assembly,
            stream_budget_gb=supplied.get('stream_budget_gb', BOR_STREAM_BUDGET_GB_DEFAULT),
            mesh_certification=certified,
            frequency_count=int(supplied.get('frequency_count', len(frequencies))),
            expand_to_360=bool(supplied.get('expand_to_360', False)),
            fine_factor=supplied.get('fine_factor',
                validate_mesh_convergence_policy(supplied.get('mesh_convergence_policy'))['fine_factor']),
            bor_options=dict(current_options(), factorization='dense'),
        )['estimated_peak_gb']
    def peak(assembly):
        """Largest dense requirement of the sweep; None when planning itself rejects the plan."""
        try:
            value = max([price(frequency, assembly) for frequency in frequencies])
        except MemoryError:
            return None
        if not math.isfinite(value):
            raise ValueError('BOR memory planning returned a non-finite peak.')
        return value
    limit = _solve_memory_limit_gb()
    requested = supplied.get('assembly', 'auto')
    own = peak(requested)
    if own is not None and own <= limit:
        return 'dense', None
    streamed = peak('streaming') if str(requested).strip().lower() == 'auto' else None
    if streamed is not None and streamed <= limit:
        return 'dense', 'streaming'
    if not single:
        return 'compressed', None
    # A single-precision sweep has no compressed plan, and these entries have no
    # run-time fallback. Neither dense plan fits this preview, which is more
    # cautious than the solve's admission: leave the smaller one to that gate,
    # whose rejection is the caller's memory diagnostic.
    if streamed is not None and (own is None or streamed < own):
        return 'dense', 'streaming'
    return 'dense', None


# Conductor generatrices of the already-meshed public APIs. Every other
# ``points*`` argument is a two-sided material interface.
_DIRECT_CONDUCTOR_ARGUMENTS = ('points_core', 'points_covered', 'bare_pieces')


def _direct_pieces(supplied):
    """``(points, is_conductor)`` of every generatrix passed to an already-meshed public API call."""
    for name, value in supplied.items():
        if value is None:
            continue
        if name == 'points' or name.startswith('points_'):
            if not len(value):
                continue
            if isinstance(value, (list, tuple)) and np.asarray(value[0]).ndim == 2:
                pieces = [np.asarray(piece, float) for piece in value]
            else:
                array = np.asarray(value, float)
                pieces = list(array) if array.ndim == 3 else [array]
        elif name in ('interface_points', 'bare_pieces'):
            pieces = [np.asarray(v, float) for v in value]
        else:
            continue
        conductor = (name in _DIRECT_CONDUCTOR_ARGUMENTS
                     or (name == 'points' and 'eps_r' not in supplied))
        for piece in pieces:
            if (piece.ndim != 2 or piece.shape[1] != 2 or len(piece) < 2
                    or not np.all(np.isfinite(piece))):
                raise ValueError('BOR generatrices must be finite arrays with shape (N, 2).')
            yield piece, conductor


def _direct_surface_layout(supplied):
    """``[(elements, is_conductor, is_closed), ...]`` and the largest radius of a direct call."""
    layout, radius = [], 0.0
    for piece, conductor in _direct_pieces(supplied):
        # The generatrix validator snaps radii within this tolerance onto the axis.
        axis_tol = 1.0e-12 * max(1.0, float(np.max(np.abs(piece[:, 0]))))
        closed = abs(piece[0, 0]) <= axis_tol and abs(piece[-1, 0]) <= axis_tol
        layout.append((len(piece) - 1, conductor, bool(closed)))
        radius = max(radius, float(np.max(piece[:, 0])))
    return layout, radius


def _direct_near_pair_counts(supplied):
    """Geometric near-pair counts of a direct call, as the snapshot preview counts them.

    The chooser used the stencil floor alone, which prices a body with many
    geometrically close panels below its gate. A geometry that cannot be routed
    returns no counts: the solve reports why, and its admission stays the
    authority either way.
    """
    try:
        return _near_pair_counts([piece for piece, _ in _direct_pieces(supplied)])
    except Exception:
        return {}


def _resolve_direct_factorization(supplied):
    """Backend of the first plan for already-meshed public APIs without a snapshot."""
    return _resolve_direct_plan(supplied)[0]


def _resolve_direct_plan(supplied):
    """First ``(factorization, assembly)`` plan of an already-meshed public API call.

    The dense plan is priced the way the solve will run it: the same table or
    streamed assembly decision as the solver that receives the call, two
    unknown families on a conductor and four on an interface, and the shared
    mode-worker and near plan. When the caller left ``assembly='auto'`` and
    that plan cannot fit, dense streaming is priced next (the solvers choose
    tables below a fixed 2 GB whatever the limit is, and a streamed dense solve
    is several times faster than compression); compression is selected only
    when neither fits. The runtime admission of the solve remains the
    authority: a plan it rejects is replaced by the next smaller one (see
    ``configured``), so table costs that need constructed solvers are priced
    here by their retained size without the build margin. A single full-table
    bound for every assembly mode used to send small streamed solves to the
    much slower compressed backend whenever free RAM dipped.
    """
    if supplied.get('freq_hz') is None or supplied.get('thetas_deg') is None:
        return 'dense', None
    if str(supplied.get('table_precision', 'auto')).strip().lower() == 'single':
        # Compression requires double precision; admission still guards the solve.
        # (Normalized as the solvers do: they accept ' single '.)
        return 'dense', None
    layout, radius = _direct_surface_layout(supplied)
    if not layout:
        return 'dense', None
    frequency = float(supplied['freq_hz'])
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('BOR frequency must be positive and finite.')
    modes, _ = _bor_mode_limits(2*np.pi*frequency/C0, radius,
                               supplied['thetas_deg'], supplied.get('n_modes'))
    order = int(supplied.get('gauss_order', FAR_GAUSS_ORDER))
    if order < 1:
        raise ValueError('BOR Gauss order must be positive.')
    requested = str(supplied.get('assembly', 'auto')).strip().lower()
    for assembly in ((requested, 'streaming') if requested == 'auto' else (requested,)):
        try:
            plan = _direct_dense_plan(dict(supplied, assembly=assembly), layout, modes, order)
        except MemoryError:
            continue
        if plan['fits_memory']:
            return 'dense', (None if assembly == requested else assembly)
    return 'compressed', None


def _direct_dense_plan(supplied, layout, modes, order):
    """Shared mode-worker plan of the dense solve, plus the assembly it assumes."""
    from ghost_backend.bor.solver import (conductor_operator_kinds,
        estimate_bor_cross_table_gb, estimate_bor_table_gb, plan_bor_mode_workers)
    from ghost_backend.bor.streaming import (BOR_STREAM_TILE_BUDGET_GB,
        combined_stream_mode_gb, estimate_streaming_block_gb,
        plan_combined_streaming_mode_block, plan_streaming_mode_block)
    from ghost_backend.bor.memory import solve_memory_limit_gb as _solve_memory_limit_gb
    workers = max(1, int(supplied.get('workers') or 1))
    preparation_workers = workers
    assembly = str(supplied.get('assembly', 'auto')).strip().lower()
    budget = float(supplied.get('stream_budget_gb') or BOR_STREAM_BUDGET_GB_DEFAULT)
    dofs = sum((2 if conductor else 4) * (elements + 1) for elements, conductor, _ in layout)
    rhs = 2*int(np.size(supplied['thetas_deg']))
    plain = [(elements, conductor) for elements, conductor, _ in layout]
    kinds = None
    if len(layout) == 1:
        elements, conductor, closed = layout[0]
        if conductor:
            # solve_bor: 'auto' is CFIE on a closed opaque body, EFIE otherwise.
            formulation = str(supplied.get('formulation', 'auto')).strip().lower()
            if formulation == 'auto':
                formulation = 'cfie' if closed and supplied.get('sheet_zs') is None else 'efie'
            has_ibc, sides = supplied.get('zs') is not None, 1.0
            kinds = [sum(conductor_operator_kinds(formulation, has_ibc))]
        else:
            formulation, has_ibc, sides = 'efie', True, 2.0  # solve_bor_dielectric
    auxiliary = _estimate_junction_auxiliary_gb(plain, modes, _direct_near_pair_counts(supplied), kinds)['peak_gb']
    coated = (len(layout) == 2 and supplied.get('points_outer') is not None
              and supplied.get('points_core') is not None and 'eps_r' in supplied)
    if len(layout) == 1:
        decision = sides * estimate_bor_table_gb(elements, modes, formulation, has_ibc, order, False)
        # solve_bor streams every automatic conductor solve (October 2026); the
        # dielectric solver keeps its 2 GB table threshold.
        streaming = assembly == 'streaming' or (assembly == 'auto' and (conductor or decision > 2.0))
        if streaming:
            block, held, planned = plan_streaming_mode_block(elements, modes, formulation, has_ibc,
                                                             False, budget / sides, workers)
            # The solve's spill decision (solve_bor, solve_bor_dielectric).
            _, held, workers, _, _, _ = _mirror_stream_spill(
                (block, sides * held, planned), workers, modes,
                sides * estimate_streaming_block_gb(elements, modes, 1, formulation, has_ibc, False),
                surface_layout=plain)
            assembly_peak = held + BOR_STREAM_TILE_BUDGET_GB
        else:
            assembly_peak = _table_plan_peak_gb(decision, plain, modes, order)
    else:
        if coated:
            # solve_bor_coated_pec decides on its far tables alone; its
            # core-to-outer operators are derived from the outer-to-core ones.
            outer, core = layout[0][0], layout[1][0]
            decision = (2.0 * estimate_bor_table_gb(outer, modes, 'efie', True, order, False)
                        + estimate_bor_table_gb(core, modes, 'cfie', False, order, False)
                        + estimate_bor_cross_table_gb(outer, core, modes, order, order, False))
        else:
            # The junction planner adds its retained auxiliaries to the tables.
            decision = _estimate_multisurface_operator_gb(plain, modes, order) + auxiliary
        streaming = assembly == 'streaming' or (assembly == 'auto' and decision > 2.0)
        if streaming:
            specs = []
            for elements, conductor in plain:
                specs += [(elements, elements, True, False)] * (1 if conductor else 2)
            # One streamed cross per surface pair: the reverse is derived.
            specs += [(a, b, True, False) for i, (a, _) in enumerate(plain)
                      for j, (b, _) in enumerate(plain) if i < j]
            plan = plan_combined_streaming_mode_block(modes, tuple(specs), budget, workers)
            # The solve's spill decision; exact for the coated solve, the
            # junction layouts as estimate_bor_resources prices them.
            always_built = [(elements, elements, not conductor, False)
                            for elements, conductor in plain
                            for _side in range(1 if conductor else 2)]
            _, held, workers, _, _, _ = _mirror_stream_spill(
                plan, workers, modes, combined_stream_mode_gb(modes, specs), exact=coated,
                surely_short=(modes + 1) * combined_stream_mode_gb(modes, always_built) > budget,
                surface_layout=plain)
            assembly_peak = held + BOR_STREAM_TILE_BUDGET_GB
        else:
            assembly_peak = _table_plan_peak_gb(decision if coated else decision - auxiliary, plain, modes, order)
    direct_output = estimate_output_gb(1, np.size(supplied['thetas_deg']))
    # The same process limit as the snapshot chooser and the 2-D entries.
    plan = plan_bor_mode_workers(dofs, rhs, workers, modes + 1,
                                 assembly_peak + auxiliary + max(output_reserved_gb(), direct_output),
                                 memory_limit_gb=_solve_memory_limit_gb(),
                                 preparation_workers=preparation_workers)
    plan['assumed_assembly'] = 'streaming' if streaming else 'tables'
    return plan


@configured
def estimate_bor_resources(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    aspects_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    n_modes: 'Optional[int]' = None,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'int' = 1,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
    mesh_certification: 'bool' = True,
    fine_factor: 'float' = 1.5,
    frequency_count: 'int' = 1,
    expand_to_360: 'bool' = False,
) -> 'Dict[str, Any]':
    """Preview the peak allocation used for memory-aware scheduling.

    Material interpolation and panel-count arithmetic match the solve path,
    but this function performs no quadrature or matrix assembly. The peak is
    intentionally conservative and is a reservation, not an RSS promise.
    """

    frequency = float(frequency_ghz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_ghz must be positive and finite.")
    aspects = np.asarray(aspects_deg, dtype=float)
    if (
        aspects.ndim != 1 or aspects.size == 0
        or not np.all(np.isfinite(aspects))
        or np.any((aspects < 0.0) | (aspects > 180.0))
    ):
        raise ValueError("aspects_deg must be a nonempty finite [0, 180] axis.")
    worker_count = max(1, int(workers))
    if type(frequency_count) is not int or frequency_count < 1:
        raise ValueError('frequency_count must be a positive integer.')
    output_gb = max(output_reserved_gb(), estimate_output_gb(
        frequency_count, aspects.size, mesh_certification, expand_to_360))
    precision = str(table_precision).strip().lower()
    assembly_key = str(assembly).strip().lower()
    stream_budget = float(stream_budget_gb)
    if precision not in {"auto", "single", "double"}:
        raise ValueError("table_precision must be auto, single, or double.")
    if assembly_key not in {"auto", "tables", "streaming"}:
        raise ValueError("assembly must be auto, tables, or streaming.")
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")

    preview_snapshot = (
        scale_snapshot_panel_density(geometry_snapshot, float(fine_factor))
        if mesh_certification else geometry_snapshot
    )
    if mesh_certification:
        preview_snapshot["_bor_certification_refinement_factor"] = float(
            fine_factor
        )
        preview_snapshot["_bor_certification_base_segment_n"] = [
            (list(segment.get("properties", []) or []) + ["", ""])[1]
            for segment in list(geometry_snapshot.get("segments", []) or [])
        ]
    scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        preview_snapshot, material_base_dir
    )
    _reject_unsupported_bor_ibc_interfaces(preview_snapshot)
    base_dir, _, materials, scale = prepare_geometry(preview_snapshot, base_dir, geometry_units)
    wavelength, _max_index, _flags = (
        _conservative_mesh_wavelength_for_frequencies(
            preview_snapshot, materials, [frequency]
        )
    )
    chains = _chains_from_snapshot(preview_snapshot, scale)
    kind = _classify(chains)
    _validate_bor_far_controls(
        kind, assembly_key, precision, stream_budget
    )
    groups, _tol, _axis_tol = _prepare_bor_groups(chains, kind)
    # Operator families of a conductor solve, as solve_bor states them: every
    # family keeps its own near contractions (three for an impedance CFIE).
    surface_kinds = None
    if kind in {"conductor", "sheet"}:
        from ghost_backend.bor.solver import conductor_operator_kinds
        has_ibc = kind == "conductor" and any(chain.ibc_flag > 0 for chain in chains)
        sampled_zs = np.array([materials.get_impedance(c.ibc_flag, frequency, arc_s=s) for c in chains for s in (0., .5, 1.)])
        formulation = "efie" if kind == "sheet" else _conductor_formulation(sampled_zs)
        surface_kinds = [sum(conductor_operator_kinds(formulation, has_ibc))]
    if kind == "conductor":
        _mark_impedance_junctions(groups[0], materials, frequency)
    surface_layout = _bor_surface_layout(groups, kind, wavelength)
    element_count = _enforce_total_element_limit(
        surface_layout, max_elements
    )
    pair_counts = _preview_near_pair_counts(groups, kind, wavelength, _axis_tol, max_elements)
    radius = max(float(np.max(chain.pts[:, 0])) for chain in chains)
    k0 = 2.0 * math.pi * frequency * 1.0e9 / C0
    mode_cap, mode_tail_start = _bor_mode_limits(
        k0, radius, aspects, n_modes
    )
    from ghost_backend.bor.solver import plan_bor_mode_workers

    unknowns = sum(
        (2 if is_conductor else 4) * (int(elements) + 1)
        for elements, is_conductor in surface_layout
    )
    if compressed_requested():
        auxiliary = _estimate_junction_auxiliary_gb(surface_layout, mode_cap, pair_counts, surface_kinds)
        retained_far_gb, far_work_gb, stream_mode_block = 0., 0., None
        exact_spill_directory, exact_spill_candidate = None, 0.
        effective_workers = worker_count
        from ghost_backend.bor.solver import (plan_compressed_far_cache,
            plan_compressed_far_spill, COMPRESSED_FAR_WORK_GB)
        if kind in {'conductor', 'sheet'}:
            cache_plan = plan_compressed_far_cache(element_count, mode_cap, formulation, has_ibc,
                stream_budget, worker_count, 2*aspects.size, auxiliary['peak_gb']+output_gb,
                near_pairs=max(pair_counts.values(), default=0))
            if cache_plan is not None:
                stream_mode_block, retained_far_gb, effective_workers = cache_plan
                far_work_gb = COMPRESSED_FAR_WORK_GB
                exact_spill_directory, exact_spill_candidate = plan_compressed_far_spill(
                    element_count, mode_cap, formulation, has_ibc, stream_mode_block)
        worker_plan = plan_bor_mode_workers(unknowns, 2*aspects.size,
            effective_workers, mode_cap+1, auxiliary['peak_gb'] + retained_far_gb + far_work_gb + output_gb,
            near_pairs=max(pair_counts.values(), default=0), preparation_workers=worker_count)
        near_plan = worker_plan['near_preparation']
        from ghost_backend.bor.options import resolved_compression_tile
        return dict(frequency_ghz=frequency, geometry_kind=kind, mesh_elements=int(element_count),
            surface_count=len(surface_layout), n_unknowns_estimate=int(unknowns),
            mode_cap_estimate=int(mode_cap), mode_tail_start_estimate=int(mode_tail_start),
            active_mode_workers=worker_plan['workers'], assembly_estimate='compressed',
            table_precision_estimate='double', persistent_assembly_gb=auxiliary['retained_gb'] + retained_far_gb,
            held_assembly_gb=auxiliary['retained_gb'] + retained_far_gb, junction_projection_gb=auxiliary['projection_gb'],
            near_junction_operator_gb=auxiliary['near_gb'], stream_mode_block_estimate=stream_mode_block,
            estimated_peak_gb=worker_plan['estimated_peak_gb'], worker_plan=worker_plan,
            output_grid_gb=output_gb,
            near_preparation=near_plan,
            mesh_certification=bool(mesh_certification),
            memory_estimate_method='compressed_payload_cap_and_workspace',
            angle_batch_size=current_options()['angle_batch_size'],
            compression_tile_estimate=resolved_compression_tile(current_options(), stream_mode_block is not None),
            stream_spill_gb_estimate=exact_spill_candidate if exact_spill_directory else 0.,
            stream_spill_candidate_gb=exact_spill_candidate,
            stream_spill_directory=str(exact_spill_directory) if exact_spill_directory else None)
    persistent_gb = 0.0
    held_assembly_gb = 0.0
    stream_mode_block = None
    effective_workers = worker_count
    estimated_assembly = "dense-direct"
    estimated_precision = "double"
    # Far-block spill (decimal GB): what this plan writes to the temporary
    # directory, and what it would write if that directory had the room.
    spill_gb = 0.0
    spill_candidate_gb = 0.0
    spill_directory = None
    auxiliary_estimate = {
        "projection_gb": 0.0,
        "near_gb": 0.0,
        "retained_gb": 0.0,
        "peak_gb": 0.0,
    }
    if kind in {"conductor", "sheet"}:
        from ghost_backend.bor.solver import estimate_bor_table_gb
        from ghost_backend.bor.streaming import (
            BOR_STREAM_TILE_BUDGET_GB,
            estimate_streaming_gb,
            plan_streaming_mode_block,
        )

        table_double = estimate_bor_table_gb(
            element_count, mode_cap, formulation, has_ibc, FAR_GAUSS_ORDER, False
        )
        # Conductors and sheets stream whenever the caller leaves the assembly
        # automatic (solve_bor, October 2026); tables are an explicit request.
        use_streaming = assembly_key in ("streaming", "auto")
        full_double = (
            estimate_streaming_gb(
                element_count, mode_cap, formulation, has_ibc, False
            )
            if use_streaming else table_double
        )
        use_single = precision == "single"
        persistent_gb = full_double / (2.0 if use_single else 1.0)
        held_assembly_gb = persistent_gb
        if use_streaming:
            (
                stream_mode_block,
                held_assembly_gb,
                effective_workers,
            ) = plan_streaming_mode_block(
                element_count,
                mode_cap,
                formulation,
                has_ibc,
                use_single,
                stream_budget,
                worker_count,
            )
            # Mirror solve_bor: a one-sweep memory-mapped build keeps only the
            # modes being read resident and frees the mode workers from range
            # alignment, so the preview prices what the solve will run.
            from ghost_backend.bor.streaming import estimate_streaming_block_gb
            (stream_mode_block, held_assembly_gb, effective_workers, spill_gb,
             spill_candidate_gb, spill_directory) = _mirror_stream_spill(
                (stream_mode_block, held_assembly_gb, effective_workers),
                worker_count, mode_cap,
                estimate_streaming_block_gb(element_count, mode_cap, 1, formulation,
                                            has_ibc, use_single), surface_layout=surface_layout)
        elif not use_streaming:
            held_assembly_gb = _table_plan_peak_gb(persistent_gb, surface_layout, mode_cap)
        estimated_assembly = "streaming" if use_streaming else "tables"
        estimated_precision = "single" if use_single else "double"
        assembly_peak_gb = (
            held_assembly_gb + BOR_STREAM_TILE_BUDGET_GB
            if use_streaming else held_assembly_gb
        )
    elif kind == "dielectric":
        from ghost_backend.bor.solver import estimate_bor_table_gb
        from ghost_backend.bor.streaming import (
            BOR_STREAM_TILE_BUDGET_GB,
            estimate_streaming_gb,
            plan_streaming_mode_block,
        )

        dielectric_elements = int(surface_layout[0][0])
        table_double = 2.0 * estimate_bor_table_gb(
            dielectric_elements, mode_cap, "efie", True, FAR_GAUSS_ORDER, False
        )
        use_streaming = (
            assembly_key == "streaming"
            or (assembly_key == "auto" and table_double > 2.0)
        )
        if use_streaming:
            full_double = 2.0 * estimate_streaming_gb(
                dielectric_elements, mode_cap, "efie", True, False
            )
            use_single = precision == "single"
            persistent_gb = full_double / (2.0 if use_single else 1.0)
            (
                stream_mode_block,
                held_one,
                effective_workers,
            ) = plan_streaming_mode_block(
                dielectric_elements,
                mode_cap,
                "efie",
                True,
                use_single,
                0.5 * stream_budget,
                worker_count,
            )
            # Both streams spill together, as solve_bor_dielectric decides.
            from ghost_backend.bor.streaming import estimate_streaming_block_gb
            (stream_mode_block, held_assembly_gb, effective_workers, spill_gb,
             spill_candidate_gb, spill_directory) = _mirror_stream_spill(
                (stream_mode_block, 2.0 * held_one, effective_workers),
                worker_count, mode_cap,
                2.0 * estimate_streaming_block_gb(dielectric_elements, mode_cap, 1,
                                                  "efie", True, use_single), surface_layout=surface_layout)
            assembly_peak_gb = (
                held_assembly_gb + BOR_STREAM_TILE_BUDGET_GB
            )
            estimated_assembly = "streaming"
            estimated_precision = "single" if use_single else "double"
        else:


            use_single = precision == "single"
            persistent_gb = _estimate_multisurface_operator_gb(
                surface_layout, mode_cap, single_tables=use_single
            )
            held_assembly_gb = _table_plan_peak_gb(persistent_gb, surface_layout, mode_cap)
            assembly_peak_gb = held_assembly_gb
            estimated_assembly = "tables"
            estimated_precision = "single" if use_single else "double"
    elif kind == "coated":
        from ghost_backend.bor.streaming import (
            BOR_STREAM_TILE_BUDGET_GB,
            estimate_rectangular_streaming_gb,
            plan_combined_streaming_mode_block,
        )

        outer_elements = int(surface_layout[0][0])
        core_elements = int(surface_layout[1][0])
        table_double = _estimate_multisurface_operator_gb(
            surface_layout, mode_cap
        )
        use_streaming = (
            assembly_key == "streaming"
            or (assembly_key == "auto" and table_double > 2.0)
        )
        if use_streaming:
            # As solve_bor_coated_pec streams: the core-to-outer blocks are
            # derived from the outer-to-core ones, not streamed.
            stream_specs_double = (
                (outer_elements, outer_elements, True, False),
                (outer_elements, outer_elements, True, False),
                (core_elements, core_elements, True, False),
                (outer_elements, core_elements, True, False),
            )
            full_double = sum(
                estimate_rectangular_streaming_gb(
                    nt, ns, mode_cap, rotated, single
                )
                for nt, ns, rotated, single in stream_specs_double
            )
            use_single = precision == "single"
            persistent_gb = full_double / (2.0 if use_single else 1.0)
            stream_specs = tuple(
                (nt, ns, rotated, use_single)
                for nt, ns, rotated, _single in stream_specs_double
            )
            (
                stream_mode_block,
                held_assembly_gb,
                effective_workers,
            ) = plan_combined_streaming_mode_block(
                mode_cap, stream_specs, stream_budget, worker_count
            )
            # These are exactly the solve's streams: mirror its spill.
            from ghost_backend.bor.streaming import combined_stream_mode_gb
            (stream_mode_block, held_assembly_gb, effective_workers, spill_gb,
             spill_candidate_gb, spill_directory) = _mirror_stream_spill(
                (stream_mode_block, held_assembly_gb, effective_workers),
                worker_count, mode_cap, combined_stream_mode_gb(mode_cap, stream_specs),
                surface_layout=surface_layout)
            assembly_peak_gb = (
                held_assembly_gb + BOR_STREAM_TILE_BUDGET_GB
            )
            estimated_assembly = "streaming"
            estimated_precision = "single" if use_single else "double"
        else:
            use_single = precision == "single"
            persistent_gb = _estimate_multisurface_operator_gb(
                surface_layout, mode_cap, single_tables=use_single
            )
            held_assembly_gb = _table_plan_peak_gb(persistent_gb, surface_layout, mode_cap)
            assembly_peak_gb = held_assembly_gb
            estimated_assembly = "tables"
            estimated_precision = "single" if use_single else "double"
    else:
        from ghost_backend.bor.streaming import (
            BOR_STREAM_TILE_BUDGET_GB,
            estimate_rectangular_streaming_gb,
            plan_combined_streaming_mode_block,
        )


        table_double = _estimate_multisurface_operator_gb(
            surface_layout, mode_cap
        )
        auxiliary_estimate = _estimate_junction_auxiliary_gb(
            surface_layout, mode_cap, pair_counts
        )
        if kind == "partial":
            # The gate's extra_retained_gb: the sparse 2N x 2N impedance map of
            # every bare piece that carries an IBC (bor_impedance_map_bytes).
            from ghost_backend.bor.solver import BOR_RETAINED_STORAGE_FACTOR, bor_impedance_map_bytes
            maps_gb = BOR_RETAINED_STORAGE_FACTOR * sum(
                bor_impedance_map_bytes(int(elements) + 1)
                for (elements, _conductor), run in zip(surface_layout[2:], groups[2])
                if any(chain.ibc_flag > 0 for chain in run)) / 1.0e9
            auxiliary_estimate = dict(auxiliary_estimate,
                                      retained_gb=auxiliary_estimate["retained_gb"] + maps_gb,
                                      peak_gb=auxiliary_estimate["peak_gb"] + maps_gb)
        use_streaming = (
            assembly_key == "streaming"
            or (
                assembly_key == "auto"
                and table_double + auxiliary_estimate["peak_gb"] > 2.0
            )
        )
        if use_streaming:
            stream_specs_double = []
            for elements, is_conductor in surface_layout:


                for _side in range(1 if is_conductor else 2):
                    stream_specs_double.append(
                        (int(elements), int(elements), True, False)
                    )
            # One streamed cross per surface pair; the reverse is derived.
            for test_index, (test_elements, _test_cond) in enumerate(surface_layout):
                for source_index, (source_elements, _source_cond) in enumerate(surface_layout):
                    if test_index < source_index:
                        stream_specs_double.append((
                            int(test_elements), int(source_elements), True, False
                        ))
            full_far_double = sum(
                estimate_rectangular_streaming_gb(
                    nt, ns, mode_cap, rotated, False
                )
                for nt, ns, rotated, _single in stream_specs_double
            )
            use_single = precision == "single"
            stream_specs = tuple(
                (nt, ns, rotated, use_single)
                for nt, ns, rotated, _single in stream_specs_double
            )
            (
                stream_mode_block,
                held_far_gb,
                effective_workers,
            ) = plan_combined_streaming_mode_block(
                mode_cap, stream_specs, stream_budget, worker_count
            )
            # The layout prices the solve's streams approximately: it gives
            # every conductor a rotated family (the solve may stream its EFIE
            # family alone) and every surface pair one cross (the solve has
            # one per region the pair bounds).  The solve surely spills when
            # the streams it always builds -- both sides of every interface
            # and the EFIE family of every conductor -- cannot hold every
            # mode either.
            from ghost_backend.bor.streaming import combined_stream_mode_gb
            always_built = tuple(
                (int(elements), int(elements), not is_conductor, use_single)
                for elements, is_conductor in surface_layout
                for _side in range(1 if is_conductor else 2))
            (stream_mode_block, held_far_gb, effective_workers, spill_gb,
             spill_candidate_gb, spill_directory) = _mirror_stream_spill(
                (stream_mode_block, held_far_gb, effective_workers),
                worker_count, mode_cap, combined_stream_mode_gb(mode_cap, stream_specs),
                exact=False,
                surely_short=(mode_cap + 1) * combined_stream_mode_gb(mode_cap, always_built)
                > stream_budget, surface_layout=surface_layout)
            full_far_gb = full_far_double / (2.0 if use_single else 1.0)
            persistent_gb = (
                full_far_gb + auxiliary_estimate["retained_gb"]
            )
            held_assembly_gb = (
                held_far_gb + auxiliary_estimate["retained_gb"]
            )
            assembly_peak_gb = (
                held_far_gb + auxiliary_estimate["peak_gb"]
                + BOR_STREAM_TILE_BUDGET_GB
            )
            estimated_assembly = "streaming"
            estimated_precision = "single" if use_single else "double"
        else:
            use_single = precision == "single"
            far_tables_gb = _estimate_multisurface_operator_gb(
                surface_layout, mode_cap, single_tables=use_single
            )
            persistent_gb = (
                far_tables_gb + auxiliary_estimate["retained_gb"]
            )
            held_assembly_gb = (
                _table_plan_peak_gb(far_tables_gb, surface_layout, mode_cap)
                + auxiliary_estimate["peak_gb"]
            )
            assembly_peak_gb = held_assembly_gb
            estimated_assembly = "tables"
            estimated_precision = "single" if use_single else "double"

    if kind in ('conductor', 'sheet', 'dielectric', 'coated'):
        auxiliary_estimate = _estimate_junction_auxiliary_gb(surface_layout, mode_cap, pair_counts, surface_kinds)
        assembly_peak_gb += auxiliary_estimate['peak_gb']
        persistent_gb += auxiliary_estimate['retained_gb']

    # The largest single preparation call decides the near backend, exactly
    # as the executor does, so preview charges process workers only if used.
    worker_plan = plan_bor_mode_workers(unknowns, 2 * int(aspects.size),
        effective_workers, mode_cap + 1, assembly_peak_gb + output_gb,
        near_pairs=max(pair_counts.values(), default=0), preparation_workers=worker_count)
    active_modes = worker_plan['workers']
    near_plan = worker_plan['near_preparation']
    peak_gb = worker_plan['estimated_peak_gb']
    return {
        "frequency_ghz": frequency,
        "geometry_kind": kind,
        "mesh_elements": int(element_count),
        "surface_count": int(len(surface_layout)),
        "n_unknowns_estimate": int(unknowns),
        "mode_cap_estimate": int(mode_cap),
        "mode_tail_start_estimate": int(mode_tail_start),
        "active_mode_workers": int(active_modes),
        "worker_plan": worker_plan,
        "near_preparation": near_plan,
        "assembly_estimate": estimated_assembly,
        "table_precision_estimate": estimated_precision,
        "persistent_assembly_gb": float(persistent_gb),
        "held_assembly_gb": float(held_assembly_gb),
        "junction_projection_gb": float(
            auxiliary_estimate["projection_gb"]
        ),
        "near_junction_operator_gb": float(
            auxiliary_estimate["near_gb"]
        ),
        "stream_mode_block_estimate": (
            int(stream_mode_block) if stream_mode_block is not None else None
        ),
        "estimated_peak_gb": float(peak_gb),
        "output_grid_gb": output_gb,
        "mesh_certification": bool(mesh_certification),
        "stream_spill_gb_estimate": float(spill_gb),
        "stream_spill_candidate_gb": float(spill_candidate_gb),
        "stream_spill_directory": spill_directory,
    }


def _stitch_pieces(chains: 'List[_SegChain]', what: 'str',
                   tol: 'float') -> 'List[List[_SegChain]]':
    """Order chains head-to-tail into MULTIPLE maximal open runs (used for
    the bare-conductor pieces of a partial coating)."""

    def key(p) -> 'Tuple[int, int]':
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))

    start_of = {}
    end_of = {}
    for c in chains:
        ks, ke = key(c.pts[0]), key(c.pts[-1])
        if ks in start_of or ke in end_of:
            raise ValueError(f"Two {what} segments start or end at the same point.")
        start_of[ks] = c
        end_of[ke] = c
    heads = [c for c in chains if key(c.pts[0]) not in end_of]
    runs: 'List[List[_SegChain]]' = []
    used = 0
    for head in heads:
        run = [head]
        while True:
            nxt = start_of.get(key(run[-1].pts[-1]))
            if nxt is None or nxt is head:
                break
            run.append(nxt)
        runs.append(run)
        used += len(run)
    if used != len(chains):
        raise ValueError(f"The {what} segments contain a closed loop or a "
                         "branching junction; expected open head-to-tail runs.")
    return runs


def _prepare_bor_groups(chains: 'List[_SegChain]', kind: 'str'):
    """Build and preflight the material-specific generatrix layout.

    Resource preview and the actual solve must interpret a geometry in exactly
    the same way.  Keeping this topology construction in one place prevents a
    scheduler estimate from counting a different set of surfaces than the
    formulation that will eventually be assembled.
    """

    diag = max(
        float(np.ptp(np.vstack([chain.pts for chain in chains]), axis=0).max()),
        1e-9,
    )
    tol = max(1e-12, 1e-9 * diag)
    axis_tol = 1e-6 * diag

    if kind == "coated":
        outer_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 3],
            "outer-interface (TYPE 3)", tol,
        )
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        _preflight_generatrix(outer_chains, "outer-interface", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [outer_chains, core_chains]
    elif kind == "partial":
        iface_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 3],
            "coating-interface (TYPE 3)", tol,
        )
        cov_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "covered-core (TYPE 4)", tol,
        )
        bare_runs = _stitch_pieces(
            [chain for chain in chains if chain.seg_type == 2],
            "bare-conductor (TYPE 2)", tol,
        )
        merged = _stitch_generatrix(
            cov_chains + [chain for run in bare_runs for chain in run],
            "PEC core (TYPE 2 + TYPE 4)", tol,
        )
        _preflight_generatrix(merged, "PEC core", axis_tol)
        groups = [iface_chains, cov_chains, bare_runs]
    elif kind == "layered":
        outer_flag, inner_flag = next(iter({
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }))
        mid5_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 5],
            "layer-interface (TYPE 5)", tol,
        )
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        patch_chains = _stitch_generatrix(
            [
                chain for chain in chains
                if chain.seg_type == 3 and chain.pos_mat == outer_flag
            ],
            "outer-interface (TYPE 3, outer layer)", tol,
        )
        bare_mid_runs = _stitch_pieces(
            [
                chain for chain in chains
                if chain.seg_type == 3 and chain.pos_mat == inner_flag
            ],
            "exposed-inner-interface (TYPE 3, inner layer)", tol,
        )
        _preflight_generatrix(core_chains, "core", axis_tol)
        merged_mid = _stitch_generatrix(
            mid5_chains
            + [chain for run in bare_mid_runs for chain in run],
            "inner-layer interface (TYPE 5 + TYPE 3)", tol,
        )
        _preflight_generatrix(
            merged_mid, "inner-layer interface", axis_tol
        )
        if not bare_mid_runs:
            _preflight_generatrix(
                patch_chains, "outer interface", axis_tol
            )
        groups = [
            patch_chains, mid5_chains, bare_mid_runs, core_chains,
            (outer_flag, inner_flag),
        ]
    elif kind == "layered_n":
        type3 = [chain for chain in chains if chain.seg_type == 3]
        top_flags = {chain.pos_mat for chain in type3}
        if len(top_flags) != 1:
            raise ValueError(
                "N-layer stacks (multiple TYPE 5 flag pairs) support full "
                "coverage only: all TYPE 3 chains must reference the "
                "outermost layer flag (patch layouts are limited to two "
                "layers)."
            )
        top_flag = next(iter(top_flags))
        core_flags = {
            chain.pos_mat for chain in chains if chain.seg_type == 4
        }
        if len(core_flags) != 1:
            raise ValueError(
                "The TYPE 4 core segments must share one pos_mat."
            )
        bottom_flag = next(iter(core_flags))
        pair_map = {}
        for outer, inner in {
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }:
            if outer in pair_map:
                raise ValueError(
                    f"Two TYPE 5 interfaces claim outer flag {outer}."
                )
            pair_map[outer] = inner
        flag_order = [top_flag]
        while flag_order[-1] != bottom_flag:
            next_flag = pair_map.pop(flag_order[-1], None)
            if next_flag is None:
                raise ValueError(
                    "Layer-flag chain broken: no TYPE 5 interface has "
                    f"pos_mat {flag_order[-1]} (walking outer flag "
                    f"{top_flag} toward core flag {bottom_flag})."
                )
            flag_order.append(next_flag)
        if pair_map:
            raise ValueError(
                f"TYPE 5 interfaces with flags {sorted(pair_map)} are not "
                "part of the outer-to-core layer chain."
            )
        interface_groups = [
            _stitch_generatrix(
                type3, "outer-interface (TYPE 3)", tol
            )
        ]
        for outer, inner in zip(flag_order[:-1], flag_order[1:]):
            type5 = [
                chain for chain in chains
                if chain.seg_type == 5
                and (chain.pos_mat, chain.neg_mat) == (outer, inner)
            ]
            interface_groups.append(_stitch_generatrix(
                type5,
                f"layer-interface (TYPE 5, {outer}|{inner})",
                tol,
            ))
        core_chains = _stitch_generatrix(
            [chain for chain in chains if chain.seg_type == 4],
            "core (TYPE 4)", tol,
        )
        for index, group in enumerate(interface_groups):
            _preflight_generatrix(group, f"interface {index}", axis_tol)
        _preflight_generatrix(core_chains, "core", axis_tol)
        groups = [interface_groups, core_chains, flag_order]
    elif kind == "banded":
        def runs_of(predicate, what):
            subset = [chain for chain in chains if predicate(chain)]
            return _stitch_pieces(subset, what, tol) if subset else []

        covered_runs = []
        for flag in sorted({
            chain.pos_mat for chain in chains if chain.seg_type == 4
        }):
            covered_runs += [(flag, run) for run in runs_of(
                lambda chain, material=flag: (
                    chain.seg_type == 4 and chain.pos_mat == material
                ),
                f"TYPE 4 band (mat {flag})",
            )]
        outer_runs = []
        for flag in sorted({
            chain.pos_mat for chain in chains if chain.seg_type == 3
        }):
            outer_runs += [(flag, run) for run in runs_of(
                lambda chain, material=flag: (
                    chain.seg_type == 3 and chain.pos_mat == material
                ),
                f"TYPE 3 band surface (mat {flag})",
            )]
        wall_runs = []
        for outer, inner in sorted({
            (chain.pos_mat, chain.neg_mat)
            for chain in chains if chain.seg_type == 5
        }):
            if outer == inner:
                raise ValueError(
                    "A TYPE 5 band wall needs two DIFFERENT material flags "
                    "(adjacent bands of the same material are one band)."
                )
            wall_runs += [((outer, inner), run) for run in runs_of(
                lambda chain, positive=outer, negative=inner: (
                    chain.seg_type == 5
                    and (chain.pos_mat, chain.neg_mat)
                    == (positive, negative)
                ),
                f"TYPE 5 band wall ({outer}|{inner})",
            )]
        bare_runs = runs_of(
            lambda chain: chain.seg_type == 2, "bare (TYPE 2)"
        )
        for run in bare_runs:
            for chain in run:
                if chain.ibc_flag > 0:
                    raise ValueError(
                        "Banded layouts: IBC on bare TYPE 2 pieces is not "
                        "supported yet (PEC only)."
                    )
        groups = [covered_runs, outer_runs, wall_runs, bare_runs]
    else:
        ordered = _stitch_generatrix(
            chains, "sheet / PEC" if kind == "sheet" else "TYPE 2" if kind == "conductor" else "TYPE 3", tol
        )
        if kind != "sheet":
            _preflight_generatrix(ordered, "body", axis_tol)
        groups = [ordered]

    # Run the same material-side check before preview estimates or meshing.
    from ghost_backend.bor.geometry import require_containment
    def points_for(run):
        points = np.vstack([run[0].pts] + [chain.pts[1:] for chain in run[1:]]).copy()
        points[np.abs(points[:, 0]) <= axis_tol, 0] = 0.
        require_resolved_corners(points)
        return points
    from ghost_backend.bor.geometry import require_resolved_corners
    for chain in chains:
        require_resolved_corners(chain.pts)
    if kind in ('conductor', 'sheet', 'dielectric'):
        require_resolved_corners(points_for(groups[0]))
    elif kind in ('partial', 'layered', 'banded'):
        # Corners between the chains of one stitched surface, including the
        # whole PEC core of a partial coating. Angles BETWEEN surfaces at a
        # junction are not limited: no refined rule passes the angular
        # convergence check there, so no evidence-based threshold exists.
        for run, _ in _bor_surface_runs(groups, kind):
            points_for(run)
        if kind == 'partial':
            points_for(merged)
    if kind == 'coated':
        require_containment(points_for(groups[0]), points_for(groups[1]))
    elif kind == 'layered_n':
        nested = groups[0] + [groups[1]]
        for outer, inner in zip(nested[:-1], nested[1:]):
            require_containment(points_for(outer), points_for(inner), 'BoR layer stack')
    elif kind == 'layered':
        require_containment(points_for(merged_mid), points_for(core_chains), 'BoR inner layer')
        if not bare_mid_runs:
            require_containment(points_for(patch_chains), points_for(merged_mid), 'BoR outer layer')
    return groups, tol, axis_tol


def _run_element_count(run: 'List[_SegChain]', wavelength: 'float') -> 'int':
    # Count the mesher's plan without constructing a potentially enormous
    # breakpoint list before the configured element limit can reject it.
    total = 0
    for chain in run:
        for index, length in enumerate(chain.prim_lengths):
            count, start, end, levels = _primitive_mesh_plan(
                chain, index, _chain_element_count(chain, float(length), wavelength))
            total += count + levels * (int(start) + int(end))
    return total


def _bor_surface_runs(groups, kind):
    def record(run, conductor):
        return (run, bool(conductor))

    if kind in {"conductor", "sheet"}:
        return [record(groups[0], True)]
    if kind == "dielectric":
        return [record(groups[0], False)]
    if kind == "coated":
        return [record(groups[0], False), record(groups[1], True)]
    if kind == "partial":
        return [record(groups[0], False), record(groups[1], True)] + [
            record(run, True) for run in groups[2]
        ]
    if kind == "layered":
        return [record(groups[0], False), record(groups[1], False)] + [
            record(run, False) for run in groups[2]
        ] + [record(groups[3], True)]
    if kind == "layered_n":
        return [record(run, False) for run in groups[0]] + [
            record(groups[1], True)
        ]
    if kind == "banded":
        covered, outer, walls, bare = groups
        return (
            [record(run, True) for _flag, run in covered]
            + [record(run, False) for _flag, run in outer]
            + [record(run, False) for _flags, run in walls]
            + [record(run, True) for run in bare]
        )
    raise ValueError(f"Unsupported BoR resource layout kind {kind!r}.")


def _bor_surface_layout(groups, kind: 'str', wavelength: 'float'):
    """Return ``[(element_count, is_conductor), ...]`` for a prepared job."""
    return [(_run_element_count(run,wavelength),conductor)
            for run,conductor in _bor_surface_runs(groups,kind)]


def _preview_near_pair_counts(groups, kind, wavelength, axis_tol, max_elements):
    """Price geometric close pairs without quadrature or dense point matrices."""
    return _near_pair_counts([_mesh_generatrix(run, wavelength, max_elements, axis_tol)[0]
                              for run, _ in _bor_surface_runs(groups, kind)])


def _near_pair_counts(generatrices):
    """``{(i, j): pairs}`` that the solvers route to near integration, from meshed generatrices alone."""
    from ghost_backend.bor.solver import BorPecSolver, BorCrossOperators
    from ghost_backend.bor.kernels import Generatrix, ETA0
    surfaces = []
    counts = {}
    for i, points in enumerate(generatrices):
        surface = object.__new__(BorPecSolver)
        surface.gen, surface.near_span = Generatrix(points), 2
        surface.k, surface.eta = 1., ETA0  # Routing depends on geometry only.
        surface._configure_near_pair_routing()
        counts[i,i] = surface._near_pair_count
        surfaces.append(surface)
    for i, observer in enumerate(surfaces):
        for j, source in enumerate(surfaces[:i]):
            cross = BorCrossOperators(observer,source)
            counts[i,j] = counts[j,i] = len(cross.near_pairs)
    return counts


def _layout_families(kind, is_conductor):
    """``(efie, mfie, ibc)`` of one medium side of a preview surface."""
    if isinstance(kind, (tuple, list)):
        return tuple(bool(flag) for flag in kind)
    if kind is None or int(kind) == 2:
        # a conductor inside a material solver (EFIE + MFIE core) or an interface side (EFIE + rotated PV)
        return (True, True, False) if is_conductor else (True, False, True)
    if int(kind) == 1:
        return True, False, False
    if int(kind) == 3:
        return True, True, True
    raise ValueError('A BoR surface prepares one to three operator families.')


def _layout_storage(surface_layout, mode_cap, pair_counts=None, kinds=None, gauss_order=FAR_GAUSS_ORDER,
                    single_tables=False, streaming=False):
    """The run-time storage model (``bor_operator_storage_bytes``) on a preview layout.

    Records are what the solvers' own requirement tuples say, bounded from
    mesh counts: two medium sides per interface, every surface pair a cross
    operator, the larger of the geometric near-pair count and a stencil floor,
    every unknown a junction constraint, and the FFT workspace at its bound.
    Dense solves integrate one direction per surface pair and derive the
    reverse (``solver._reverse_cross``), which retains nothing of its own, so a
    pair is counted once; compressed solves build both directions.  The model
    is monotone, so the preview cannot fall below a gate that is priced from
    the constructed solvers of the same mesh.
    """
    from ghost_backend.bor.solver import BorCrossStorage, BorSurfaceStorage, bor_operator_storage_bytes
    table_bytes = 8 if single_tables else 16
    counts = pair_counts or {}
    both_directions = compressed_requested()
    surfaces, crosses = [], []
    for index, (elements, is_conductor) in enumerate(surface_layout):
        ne = max(1, int(elements))
        pairs = max(min(ne * ne, 8 * ne), counts.get((index, index), 0))
        families = _layout_families(None if kinds is None else kinds[index], is_conductor)
        surfaces += [BorSurfaceStorage(gauss_order * int(elements), ne + 1, table_bytes, pairs, *families, None)
                     ] * (1 if is_conductor else 2)
    for test_index, (test_elements, _) in enumerate(surface_layout):
        for source_index, (source_elements, _) in enumerate(surface_layout):
            if test_index == source_index or (source_index < test_index and not both_directions):
                continue
            nt, ns = max(1, int(test_elements)), max(1, int(source_elements))
            pairs = max(min(nt * ns, 8 * (nt + ns)), counts.get((test_index, source_index), 0),
                        0 if both_directions else counts.get((source_index, test_index), 0))
            crosses.append(BorCrossStorage(gauss_order * int(test_elements), gauss_order * int(source_elements),
                                           table_bytes, pairs, 192, None))
    unknowns = sum((2 if is_conductor else 4) * (int(elements) + 1) for elements, is_conductor in surface_layout)
    return bor_operator_storage_bytes(max(0, int(mode_cap)), surfaces, crosses, unknowns,
                                      streaming=streaming, compressed=compressed_requested())


def _mirror_stream_spill(plan, requested_workers: 'int', mode_cap: 'int',
                         per_mode_gb: 'float', exact: 'bool' = True,
                         surely_short: 'bool' = True, surface_layout=None):
    """A streamed ``(mode block, held GB, workers)`` plan after the spill decision.

    Every streamed BoR solve applies ``plan_stream_spill`` to its plan: when
    the budget cannot hold every mode and the temporary directory has the
    room, all modes are built once into memory-mapped files, only the modes
    being read stay resident, and every requested mode worker runs.  Returns
    ``(mode block, held GB, workers, spill GB, candidate GB, directory)``; the
    candidate is what the plan writes if it spills, which schedulers reserve
    across concurrent units.

    ``exact`` is False when ``per_mode_gb`` only approximates the solve's
    streams (junction layouts, see estimate_bor_resources): the solve may
    keep its own streams in memory where the plan spills, or spill where the
    directory is too small for the plan's size.  A spill is then priced only
    when ``surely_short`` (the streams the solve always builds cannot hold
    every mode either); otherwise a plan short of every mode keeps its
    in-memory block and prices every requested worker, the union of both
    outcomes.
    """
    from ghost_backend.bor.streaming import plan_stream_spill, stream_spill_candidate_gb
    mode_block, held_gb, workers = plan
    from ghost_backend.bor.compressed_far import far_compression_selected
    if surface_layout and any(far_compression_selected(int(elements) + 1)
                              for elements, _ in surface_layout):
        return mode_block, held_gb, workers, 0.0, 0.0, None
    mode_count = int(mode_cap) + 1
    candidate = stream_spill_candidate_gb(mode_block, mode_count, per_mode_gb)
    base, spilled_block, resident_gb = plan_stream_spill(mode_block, mode_count, per_mode_gb)
    if base is not None and (exact or surely_short):
        return spilled_block, resident_gb, int(requested_workers), candidate, candidate, str(base)
    if candidate > 0.0 and not exact:
        workers = int(requested_workers)
    return mode_block, held_gb, workers, 0.0, candidate, None


def _table_plan_peak_gb(far_tables_gb: 'float', surface_layout, mode_cap: 'int',
                        gauss_order: 'int' = FAR_GAUSS_ORDER) -> 'float':
    """Dense-table plan as the run-time gate prices it, from mesh counts alone.

    Mirrors ``solver.estimate_bor_operator_storage_gb``: 1.10 x the retained far
    tables and dense basis matrices of every medium side, plus the FFT build
    workspace, which the banded construction bounds (near contractions and
    junction projections are the separate auxiliary estimate). Measured process
    peaks at cap 20: CFIE sphere, 1.09 GB of tables, 1.34 GB; impedance sphere,
    1.95 GB, 2.21 GB; dielectric sphere, 2.17 GB, 2.45 GB; coated sphere,
    1.78 GB, 2.04 GB. The former blanket 3.5 x tables priced those at 3.8 to
    7.6 GB and made the preview two to four times the gate it predicts.
    """
    from ghost_backend.bor.solver import BOR_RETAINED_STORAGE_FACTOR
    parts = _layout_storage(surface_layout, mode_cap, gauss_order=gauss_order)
    return (BOR_RETAINED_STORAGE_FACTOR * (float(far_tables_gb) + parts['basis'] / 1.0e9)
            + parts['fft_workspace'] / 1.0e9)


def _estimate_multisurface_operator_gb(surface_layout, mode_cap: 'int',
                                       gauss_order: 'int' = FAR_GAUSS_ORDER,
                                       single_tables: 'bool' = False) -> 'float':
    """Conservative retained dense-table storage for nonconductor BoR.

    Each PMCHWT medium-side/self or cross operator retains one scalar modal
    table plus four rotated-principal-value tables.  Interface surfaces have
    two medium sides; conductor surfaces have one.  Counting every directed
    cross-surface pair is exact for a fully connected region and conservative
    for layered/banded topologies whose surfaces do not all share a region.
    """

    return _layout_storage(surface_layout, mode_cap, gauss_order=gauss_order,
                           single_tables=single_tables)['tables'] / 1.0e9


def _estimate_junction_auxiliary_gb(surface_layout, mode_cap: 'int', pair_counts=None,
                                    kinds=None) -> 'Dict[str, float]':
    """Conservative preview for retained junction and direct-near storage.

    Runtime planning uses the constructed solvers' exact near-pair lists and
    projection dimensions.  The scheduler preview intentionally assumes every
    surface has both EFIE and rotated-PV local contractions, every directed
    surface pair has a generous local near stencil, and every possible pair of
    surfaces meets at a graded junction.  This keeps partial, layered, and
    banded jobs from being admitted on far-block storage alone.

    ``kinds`` gives the operator families per surface where the caller knows
    the solve's own (``conductor_operator_kinds``: an impedance CFIE retains
    three, an open PEC shell one). Two per medium side is what every material
    solver prepares and the upper bound of their conductor surfaces. The bytes
    per family are the run-time gate's (``bor_near_cache_bytes``).
    """

    from ghost_backend.bor.solver import BOR_RETAINED_STORAGE_FACTOR
    parts = _layout_storage(surface_layout, mode_cap, pair_counts, kinds, streaming=True)
    retained_bytes = BOR_RETAINED_STORAGE_FACTOR * (parts['projection'] + parts['near'])
    return {
        "projection_gb": BOR_RETAINED_STORAGE_FACTOR * parts['projection'] / 1.0e9,
        "near_gb": BOR_RETAINED_STORAGE_FACTOR * parts['near'] / 1.0e9,
        "retained_gb": retained_bytes / 1.0e9,
        "peak_gb": (retained_bytes + parts['near_workspace']) / 1.0e9,
    }


def _enforce_total_element_limit(surface_layout, max_elements: 'int') -> 'int':
    total = sum(int(elements) for elements, _conductor in surface_layout)
    limit = int(max_elements)
    if total > limit:
        raise ValueError(
            f"BoR mesh would need approximately {total} total elements "
            f"across {len(surface_layout)} surface(s) (> max {limit}). "
            "Reduce frequency or density, or raise max_elements after "
            "reviewing the resource estimate."
        )
    return total


@profiled_solve
@reserve_output
@configured
def solve_monostatic_rcs_bor(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = 0.5,
    n_modes: 'Optional[int]' = None,
    mode_tol: 'float' = 1e-6,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'Optional[int]' = None,
    abort_event: 'Optional[threading.Event]' = None,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    expand_to_360: 'bool' = False,
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
) -> 'Dict[str, Any]':
    """
    Monostatic 3-D RCS (m^2 / dBsm) of an axisymmetric body described by a
    .geo geometry snapshot.  `elevations_deg` are ASPECT angles measured from
    the +z rotation axis (0 = nose-on, 90 = broadside, 180 = tail-on); the
    same argument name as the 2-D entry point is kept for a consistent UI.
    VV and HH are always co-solved and returned together; there is no channel
    selector in the production API.

    expand_to_360=True mirrors the samples about the axis to fill the full
    polar cut: sigma(360 - theta) = sigma(theta) -- EXACT for a body of
    revolution (rotating the problem 180 deg about z maps the body, the
    directions, and the polarization basis onto themselves), including the
    complex amplitudes.  The seam directions 0/360 and 180 are not
    duplicated.  Note this is a property of the axisymmetric MODEL: it does
    not conjure the effect of any non-axisymmetric feature the BoR cannot
    represent, and it is NOT the nose<->tail flip (theta -> 180 - theta),
    which is only valid for fore-aft symmetric bodies.
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one aspect angle is required.")
    cfie_alpha = float(cfie_alpha)
    if not math.isfinite(cfie_alpha) or not (0.0 < cfie_alpha < 1.0):
        raise ValueError(
            "BoR CFIE alpha must be finite and satisfy 0 < alpha < 1. "
            "Use the explicit EFIE formulation API when pure EFIE is intended."
        )
    frequencies = [float(f) for f in frequencies_ghz]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    mesh_ref_ghz = None
    if mesh_reference_ghz is not None:
        mesh_ref_ghz = float(mesh_reference_ghz)
        if (not math.isfinite(mesh_ref_ghz)) or mesh_ref_ghz <= 0.0:
            raise ValueError(
                "mesh_reference_ghz must be a positive finite GHz value."
            )
    aspects = [float(a) for a in elevations_deg]
    if any((not math.isfinite(a)) or a < 0.0 or a > 180.0 for a in aspects):
        raise ValueError("Aspect angles must lie in [0, 180] degrees from +z.")
    assembly_key = str(assembly).strip().lower()
    expected_assembly = "compressed" if compressed_requested() else assembly_key
    if assembly_key not in {"auto", "tables", "streaming"}:
        raise ValueError(
            "assembly must be 'auto', 'tables', or 'streaming'."
        )
    table_precision_key = str(table_precision).strip().lower()
    if table_precision_key not in {"auto", "single", "double"}:
        raise ValueError(
            "table_precision must be 'auto', 'single', or 'double'."
        )
    stream_budget = float(stream_budget_gb)
    if not math.isfinite(stream_budget) or stream_budget <= 0.0:
        raise ValueError("stream_budget_gb must be a positive finite value.")
    scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    _reject_unsupported_bor_ibc_interfaces(geometry_snapshot)
    base_dir, preflight, prepared_materials, scale = prepare_geometry(
        geometry_snapshot, base_dir, geometry_units)
    # Parsed models are immutable for this run. Notices belong to this solve:
    # sharing the library's warning lists would contaminate later frequencies
    # and the fine certification pass with earlier numerical warnings.
    materials = MaterialLibrary(prepared_materials.impedance_models,
                                prepared_materials.dielectric_models)


    mesh_control_frequencies = set(frequencies)
    if mesh_ref_ghz is not None:
        mesh_control_frequencies.add(mesh_ref_ghz)
    mesh_controls = {}
    for solve_frequency in frequencies:
        control_frequencies = {float(solve_frequency)}
        if mesh_ref_ghz is not None:
            control_frequencies.add(mesh_ref_ghz)
        mesh_controls[float(solve_frequency)] = (
            _conservative_mesh_wavelength_for_frequencies(
                geometry_snapshot,
                materials,
                control_frequencies,
            )
        )
    mesh_wavelength_m = min(value[0] for value in mesh_controls.values())
    mesh_max_refractive_index = max(
        value[1] for value in mesh_controls.values()
    )
    mesh_material_flags = sorted({
        flag
        for value in mesh_controls.values()
        for flag in value[2]
    })

    chains = _chains_from_snapshot(geometry_snapshot, scale)
    kind = _classify(chains)
    _validate_bor_far_controls(
        kind, assembly_key, table_precision_key, stream_budget
    )
    groups, tol, axis_tol = _prepare_bor_groups(chains, kind)

    def check_abort():
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    from ghost_backend.bor.samples import channel_buffers, finish_channels, nonfinite_sample_count
    samples_by_pol = channel_buffers(len(frequencies), aspects, expand_to_360)
    per_freq_meta: 'List[Dict[str, Any]]' = []
    formulation_label = ""
    total_steps = len(frequencies)
    negative_rcs_count = 0
    power_amplitude_inconsistent_count = 0
    nonfinite_expected_power_count = 0

    for fi, freq_ghz in enumerate(frequencies):
        check_abort()
        freq_hz = freq_ghz * 1e9
        current_mesh = mesh_controls[float(freq_ghz)]
        lam0 = float(current_mesh[0])
        graded_junctions = (_mark_impedance_junctions(groups[0], materials, freq_ghz)
                            if kind == "conductor" else 0)
        surface_layout = _bor_surface_layout(groups, kind, lam0)
        total_mesh_elements = _enforce_total_element_limit(
            surface_layout, max_elements
        )

        def report(modes_done, m_cap):
            if progress_callback is not None:
                try:
                    progress_callback(fi, total_steps,
                                     f"{freq_ghz:g} GHz: mode {modes_done}/{m_cap}")
                except Exception:
                    pass

        if kind in {"conductor", "sheet"}:
            ordered = groups[0]
            pts, elem_seg, elem_arc = _mesh_generatrix(ordered, lam0,
                                                       max_elements, axis_tol)
            zs_elem = np.zeros(len(pts) - 1, dtype=complex)
            for ei in range(len(zs_elem)):
                c = ordered[elem_seg[ei]]
                if c.ibc_flag > 0:
                    zs_elem[ei] = materials.get_impedance(
                        c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
            has_ibc = bool(np.any(np.abs(zs_elem) > 0.0))
            form = "efie" if kind == "sheet" else _conductor_formulation(zs_elem)
            out = solve_bor(pts, freq_hz, aspects, formulation=form,
                            cfie_alpha=cfie_alpha,
                            zs=zs_elem if has_ibc and kind != "sheet" else None,
                            sheet_zs=zs_elem if kind == "sheet" else None,
                            n_modes=n_modes, mode_tol=mode_tol,
                            workers=workers, progress=report,
                            check_abort=check_abort,
                            table_precision=table_precision_key,
                            assembly=assembly_key,
                            stream_budget_gb=stream_budget)
            actual_assembly = str(out.get("assembly", "")).strip().lower()
            actual_precision = str(
                out.get("table_precision", "")
            ).strip().lower()
            if (
                expected_assembly != "auto"
                and actual_assembly != expected_assembly
            ):
                raise RuntimeError(
                    "BoR conductor solver did not attest the requested "
                    f"assembly={expected_assembly!r}; reported "
                    f"{actual_assembly or 'missing'!r}."
                )
            if (
                table_precision_key != "auto"
                and actual_precision != table_precision_key
            ):
                raise RuntimeError(
                    "BoR conductor solver did not attest the requested "
                    f"table_precision={table_precision_key!r}; reported "
                    f"{actual_precision or 'missing'!r}."
                )
            formulation_label = ("BoR transmitting electric sheet / PEC EFIE" if kind == "sheet" else
                                 f"BoR-MoM IBC-{form.upper()} (Leontovich)" if has_ibc else f"BoR-MoM PEC {form.upper()}")
        elif kind == "dielectric":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts, _, _ = _mesh_generatrix(
                groups[0], lam0, max_elements, axis_tol
            )
            out = solve_bor_dielectric(pts, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort,
                                       table_precision=table_precision_key,
                                       assembly=assembly_key,
                                       stream_budget_gb=stream_budget)
            actual_assembly = str(out.get("assembly", "")).strip().lower()
            actual_precision = str(
                out.get("table_precision", "")
            ).strip().lower()
            if (
                expected_assembly != "auto"
                and actual_assembly != expected_assembly
            ):
                raise RuntimeError(
                    "BoR dielectric solver did not attest the requested "
                    f"assembly={expected_assembly!r}; reported "
                    f"{actual_assembly or 'missing'!r}."
                )
            if (
                table_precision_key != "auto"
                and actual_precision != table_precision_key
            ):
                raise RuntimeError(
                    "BoR dielectric solver did not attest the requested "
                    f"table_precision={table_precision_key!r}; reported "
                    f"{actual_precision or 'missing'!r}."
                )
            formulation_label = "BoR-MoM PMCHWT (homogeneous dielectric)"
        elif kind == "coated":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts_o, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            out = solve_bor_coated_pec(pts_o, pts_c, freq_hz, aspects, eps, mu,
                                       n_modes=n_modes, mode_tol=mode_tol,
                                       workers=workers, progress=report,
                                       check_abort=check_abort,
                                       table_precision=table_precision_key,
                                       assembly=assembly_key,
                                       stream_budget_gb=stream_budget)
            actual_assembly = str(out.get("assembly", "")).strip().lower()
            actual_precision = str(
                out.get("table_precision", "")
            ).strip().lower()
            if (
                expected_assembly != "auto"
                and actual_assembly != expected_assembly
            ):
                raise RuntimeError(
                    "BoR coated solver did not attest the requested "
                    f"assembly={expected_assembly!r}; reported "
                    f"{actual_assembly or 'missing'!r}."
                )
            if (
                table_precision_key != "auto"
                and actual_precision != table_precision_key
            ):
                raise RuntimeError(
                    "BoR coated solver did not attest the requested "
                    f"table_precision={table_precision_key!r}; reported "
                    f"{actual_precision or 'missing'!r}."
                )
            formulation_label = "BoR-MoM PMCHWT coated PEC (multi-region)"
        elif kind == "partial":
            eps, mu = materials.get_medium(groups[0][0].pos_mat, freq_ghz)
            pts_i, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            bare_pts = []
            bare_zs = []
            any_ibc = False
            for run in groups[2]:
                pts_b, elem_seg, elem_arc = _mesh_generatrix(
                    run, lam0, max_elements, axis_tol)
                zs_elem = np.zeros(len(pts_b) - 1, dtype=complex)
                for ei in range(len(zs_elem)):
                    c = run[elem_seg[ei]]
                    if c.ibc_flag > 0:
                        zs_elem[ei] = materials.get_impedance(
                            c.ibc_flag, freq_ghz, arc_s=float(elem_arc[ei]))
                bare_pts.append(pts_b)
                has = bool(np.any(np.abs(zs_elem) > 0.0))
                any_ibc |= has
                bare_zs.append(zs_elem if has else None)
            out = solve_bor_partial_coating(pts_i, pts_c, bare_pts, freq_hz,
                                            aspects, eps, mu, bare_zs=bare_zs,
                                            n_modes=n_modes,
                                            mode_tol=mode_tol, workers=workers,
                                            progress=report,
                                            check_abort=check_abort,
                                            table_precision=table_precision_key,
                                            assembly=assembly_key,
                                            stream_budget_gb=stream_budget)
            for w in out.get("warnings", []):
                materials.warn_once(w)
            formulation_label = ("BoR-MoM PMCHWT partial coating "
                                 f"({out['n_junctions']} junction(s)"
                                 f"{', IBC bare' if any_ibc else ''})")
        elif kind == "layered":
            outer_flag, inner_flag = groups[4]
            eps_o, mu_o = materials.get_medium(outer_flag, freq_ghz)
            eps_i, mu_i = materials.get_medium(inner_flag, freq_ghz)
            pts_p, _, _ = _mesh_generatrix(groups[0], lam0, max_elements, axis_tol)
            pts_m, _, _ = _mesh_generatrix(groups[1], lam0, max_elements, axis_tol)
            pts_c, _, _ = _mesh_generatrix(groups[3], lam0, max_elements, axis_tol)
            bare_pts = [_mesh_generatrix(run, lam0, max_elements, axis_tol)[0]
                        for run in groups[2]]
            if bare_pts:
                out = solve_bor_coating_patch(pts_p, pts_m, bare_pts, pts_c,
                                              freq_hz, aspects, eps_i, mu_i,
                                              eps_o, mu_o, n_modes=n_modes,
                                              mode_tol=mode_tol, workers=workers,
                                              progress=report,
                                              check_abort=check_abort,
                                              table_precision=table_precision_key,
                                              assembly=assembly_key,
                                              stream_budget_gb=stream_budget)
                formulation_label = ("BoR-MoM PMCHWT coating patch "
                                     f"({out['n_junctions']} junction(s))")
            else:
                out = solve_bor_coated2_pec(pts_p, pts_m, pts_c, freq_hz,
                                            aspects, eps_i, mu_i, eps_o, mu_o,
                                            n_modes=n_modes, mode_tol=mode_tol,
                                            workers=workers, progress=report,
                                            check_abort=check_abort,
                                            table_precision=table_precision_key,
                                            assembly=assembly_key,
                                            stream_budget_gb=stream_budget)
                formulation_label = "BoR-MoM PMCHWT two-layer coated PEC"
        elif kind == "layered_n":
            iface_groups, core_chains, flag_order = groups
            media = [materials.get_medium(fl, freq_ghz) for fl in flag_order]
            iface_pts = [_mesh_generatrix(g, lam0, max_elements, axis_tol)[0]
                         for g in iface_groups]
            pts_c, _, _ = _mesh_generatrix(core_chains, lam0, max_elements,
                                           axis_tol)

            eps_list = [media[i][0] for i in range(len(media) - 1, -1, -1)]
            mu_list = [media[i][1] for i in range(len(media) - 1, -1, -1)]
            out = solve_bor_coated_n_pec(iface_pts, pts_c, freq_hz, aspects,
                                         eps_list, mu_list, n_modes=n_modes,
                                         mode_tol=mode_tol, workers=workers,
                                         progress=report,
                                         check_abort=check_abort,
                                         table_precision=table_precision_key,
                                         assembly=assembly_key,
                                         stream_budget_gb=stream_budget)
            formulation_label = (f"BoR-MoM PMCHWT {len(iface_pts)}-layer "
                                 "coated PEC")
        elif kind == "banded":
            cov_runs, out_runs, wall_runs, bare_band_runs = groups

            def run_ends(run):
                return (run[0].pts[0], run[-1].pts[-1])

            def key(p):
                return (int(round(p[0] / tol)), int(round(p[1] / tol)))

            def lam_for(flags):


                del flags
                return lam0


            surfaces = []
            piece_ends = []
            piece_tag = []
            for flag, run in out_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for([flag]),
                                             max_elements, axis_tol)
                surfaces.append((pts, False))
                piece_ends.append(run_ends(run))
                piece_tag.append(("out", flag))
            for pair, run in wall_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for(list(pair)),
                                             max_elements, axis_tol)
                surfaces.append((pts, False))
                piece_ends.append(run_ends(run))
                piece_tag.append(("wall", pair))
            for flag, run in cov_runs:
                pts, _, _ = _mesh_generatrix(run, lam_for([flag]),
                                             max_elements, axis_tol)
                surfaces.append((pts, True))
                piece_ends.append(run_ends(run))
                piece_tag.append(("cov", flag))
            for run in bare_band_runs:
                pts, _, _ = _mesh_generatrix(run, lam0, max_elements, axis_tol)
                surfaces.append((pts, True))
                piece_ends.append(run_ends(run))
                piece_tag.append(("bare", None))


            regions = [{"medium": None, "exterior": True,
                        "bounds": [(i, +1) for i, tg in enumerate(piece_tag)
                                   if tg[0] in ("out", "bare")]}]

            for ci, tg in enumerate(piece_tag):
                if tg[0] != "cov":
                    continue
                flag = tg[1]
                cand = [i for i, t2 in enumerate(piece_tag)
                        if (t2[0] == "out" and t2[1] == flag)
                        or (t2[0] == "wall" and flag in t2[1])]
                comp = {ci}
                grew = True
                while grew:
                    grew = False
                    kset = {key(p) for i in comp for p in piece_ends[i]}
                    for i in cand:
                        if i not in comp and any(key(p) in kset
                                                 for p in piece_ends[i]):
                            comp.add(i)
                            grew = True
                bounds = []
                for i in sorted(comp):
                    t2 = piece_tag[i]
                    if t2[0] == "cov":
                        bounds.append((i, +1))
                    elif t2[0] == "out":
                        bounds.append((i, -1))
                    else:
                        bounds.append((i, +1 if t2[1][0] == flag else -1))
                if len(bounds) < 2:
                    raise ValueError(
                        f"Band (mat {flag}) covered piece has no attached "
                        "TYPE 3/5 boundary -- check junction coordinates "
                        "coincide exactly.")
                eps_b, mu_b = materials.get_medium(flag, freq_ghz)
                regions.append({"medium": (eps_b, mu_b), "bounds": bounds})

            # The configured entry, so a banded body gets the same automatic
            # mode-cap extension and admission handling as every other kind.
            out = solve_bor_banded_multiregion(
                surfaces, regions, freq_hz, aspects, n_modes=n_modes,
                mode_tol=mode_tol, workers=workers, progress=report,
                check_abort=check_abort, table_precision=table_precision_key,
                assembly=assembly_key, stream_budget_gb=stream_budget,
                formulation="pmchwt-banded", extra={},
            )
            formulation_label = (f"BoR-MoM PMCHWT banded coatings "
                                 f"({len(regions) - 1} band(s), "
                                 f"{out['n_junctions']} junction(s))")

        actual_assembly = str(out.get("assembly", "")).strip().lower()
        actual_precision = str(
            out.get("table_precision", "")
        ).strip().lower()
        if expected_assembly != "auto" and actual_assembly != expected_assembly:
            raise RuntimeError(
                f"BoR {kind} solver did not attest the requested "
                f"assembly={expected_assembly!r}; reported "
                f"{actual_assembly or 'missing'!r}."
            )
        if (
            table_precision_key != "auto"
            and actual_precision != table_precision_key
        ):
            raise RuntimeError(
                f"BoR {kind} solver did not attest the requested "
                f"table_precision={table_precision_key!r}; reported "
                f"{actual_precision or 'missing'!r}."
            )

        for w in out.get("warnings", []) or []:
            materials.warn_once(str(w))


        residual = float(out.get("linear_residual", math.nan))
        backward_error = float(
            out.get("linear_backward_error", math.nan)
        )
        for channel, sigma_key, amp_key in (
            ("VV", "sigma_vv", "amp_vv"),
            ("HH", "sigma_hh", "amp_hh"),
        ):
            sig = out[sigma_key]
            amp = out[amp_key]
            for ai, aspect in enumerate(aspects):
                raw_lin = float(sig[ai])
                a_val = complex(amp[ai])
                if math.isfinite(raw_lin) and raw_lin < 0.0:
                    negative_rcs_count += 1
                amp_abs2 = (
                    a_val.real * a_val.real + a_val.imag * a_val.imag
                )
                expected_lin = 4.0 * math.pi * amp_abs2
                if not math.isfinite(expected_lin):
                    nonfinite_expected_power_count += 1
                if (
                    math.isfinite(raw_lin)
                    and math.isfinite(expected_lin)
                    and abs(raw_lin - expected_lin)
                    > (
                        BOR_POWER_CONSISTENCY_RTOL
                        * max(raw_lin, expected_lin)
                        + np.finfo(float).tiny
                    )
                ):
                    power_amplitude_inconsistent_count += 1
                lin = max(raw_lin, 0.0)
                samples_by_pol[channel].append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(aspect),
                    "theta_scat_deg": float(aspect),
                    "rcs_linear": lin,


                    "rcs_db": 10.0 * math.log10(max(lin, 1e-300)),
                    "rcs_amp_real": float(a_val.real),
                    "rcs_amp_imag": float(a_val.imag),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(a_val))),
                    "linear_residual": residual,
                    "linear_backward_error": backward_error,
                })
        per_freq_meta.append({
            "modal_execution": out.get("modal_execution"),
            "near_preparation": out.get("near_preparation"),
            "bor_execution_options": current_options(),
            "frequency_ghz": float(freq_ghz),
            "graded_impedance_junctions": int(graded_junctions),
            "modes_used": int(out["modes_used"]),
            "mode_cap": int(out.get("mode_cap", out["modes_used"])),
            "mode_tail_start": int(out.get("mode_tail_start", 0)),
            "mode_converged": bool(out.get("mode_converged", False)),
            "mode_quiet_count": int(out.get("mode_quiet_count", 0)),
            "mode_last_relative_increment": float(
                out.get("mode_last_relative_increment", math.inf)
            ),
            "mode_last_absolute_increment": float(
                out.get("mode_last_absolute_increment", math.inf)
            ),
            "mode_tail_absolute_floor": float(
                out.get("mode_tail_absolute_floor", 0.0)
            ),
            "mode_worst_polarization": out.get("mode_worst_polarization"),
            "mode_worst_theta_deg": out.get("mode_worst_theta_deg"),
            "signed_mode_symmetry_used": bool(
                out.get("signed_mode_symmetry_used", False)
            ),
            "n_unknowns": int(out["n_unknowns"]),
            "linear_residual": residual,
            "linear_backward_error": backward_error,
            "linear_refinement_steps": int(
                out.get("linear_refinement_steps", 0)
            ),
            "max_cond": float(out["max_cond"]) if "max_cond" in out else None,
            "condition_est_computed": bool(
                out.get("condition_est_computed", "max_cond" in out)
            ),
            "condition_est_method": out.get("condition_est_method"),
            "condition_est_limit": float(
                out.get("condition_est_limit", BOR_CONDITION_EST_MAX)
            ),
            "assembly": (
                str(out.get("assembly", "") or "") or None
            ),
            "table_precision": (
                str(out.get("table_precision", "") or "") or None
            ),
            "stream_mode_block": out.get("stream_mode_block"),
            "stream_sweeps": out.get("stream_sweeps"),
            "stream_spill_gb": float(out.get("stream_spill_gb", 0.0) or 0.0),
            "stream_sampling_backend": out.get("stream_sampling_backend"),
            "stream_far_compression": out.get("stream_far_compression"),
            "near_quadrature": out.get("near_quadrature"),
            "mesh_elements_total": int(total_mesh_elements),
            "mesh_surface_count": int(len(surface_layout)),
            "mesh_elements_by_surface": [
                int(elements) for elements, _conductor in surface_layout
            ],
            "mesh_wavelength_m": float(current_mesh[0]),
            "mesh_max_refractive_index": float(current_mesh[1]),
            "mesh_material_flags": list(current_mesh[2]),
        })
        if progress_callback is not None:
            try:
                progress_callback(fi + 1, total_steps, f"Solved {freq_ghz:g} GHz")
            except Exception:
                pass

    samples = finish_channels(samples_by_pol, expand_to_360)

    residual_values = np.asarray(
        [row.get("linear_residual", math.nan) for row in per_freq_meta],
        dtype=float,
    )
    finite_residuals = residual_values[np.isfinite(residual_values)]
    residual_nonfinite_count = int(
        residual_values.size - finite_residuals.size
    )
    residual_max = (
        float(np.max(finite_residuals))
        if finite_residuals.size
        else math.nan
    )
    backward_error_values = np.asarray(
        [
            row.get("linear_backward_error", math.nan)
            for row in per_freq_meta
        ],
        dtype=float,
    )
    finite_backward_errors = backward_error_values[
        np.isfinite(backward_error_values)
    ]
    backward_error_nonfinite_count = int(
        backward_error_values.size - finite_backward_errors.size
    )
    backward_error_max = (
        float(np.max(finite_backward_errors))
        if finite_backward_errors.size
        else math.nan
    )
    condition_values = np.asarray(
        [
            row.get("max_cond", math.nan)
            if bool(row.get("condition_est_computed", False))
            else math.nan
            for row in per_freq_meta
        ],
        dtype=float,
    )
    finite_conditions = condition_values[np.isfinite(condition_values)]
    condition_missing_count = int(
        condition_values.size - finite_conditions.size
    )
    condition_max = (
        float(np.max(finite_conditions))
        if finite_conditions.size
        else math.nan
    )
    unconverged_modes = int(sum(
        not bool(row.get("mode_converged", False))
        for row in per_freq_meta
    ))
    nonfinite_samples = nonfinite_sample_count(samples_by_pol)
    quality_violations: 'List[str]' = []
    if residual_nonfinite_count:
        quality_violations.append(
            f"residual_nonfinite_count={residual_nonfinite_count} must be zero"
        )
    if backward_error_nonfinite_count:
        quality_violations.append(
            "backward_error_nonfinite_count="
            f"{backward_error_nonfinite_count} must be zero"
        )
    if (
        not math.isfinite(backward_error_max)
        or backward_error_max > BOR_LINEAR_BACKWARD_ERROR_MAX
    ):
        quality_violations.append(
            f"backward_error_max={backward_error_max:.6g} exceeds limit "
            f"{BOR_LINEAR_BACKWARD_ERROR_MAX:.6g}"
        )
    if condition_missing_count:
        quality_violations.append(
            "condition_missing_or_nonfinite_count="
            f"{condition_missing_count} must be zero"
        )
    if (
        not math.isfinite(condition_max)
        or condition_max > BOR_CONDITION_EST_MAX
    ):
        quality_violations.append(
            f"condition_est_max={condition_max:.6g} exceeds limit "
            f"{BOR_CONDITION_EST_MAX:.6g}"
        )
    if unconverged_modes:
        quality_violations.append(
            f"mode_unconverged_count={unconverged_modes} must be zero"
        )
    if nonfinite_samples:
        quality_violations.append(
            f"nonfinite_sample_count={nonfinite_samples} must be zero"
        )
    if negative_rcs_count:
        quality_violations.append(
            f"negative_rcs_count={negative_rcs_count} must be zero"
        )
    if power_amplitude_inconsistent_count:
        quality_violations.append(
            "power_amplitude_inconsistent_count="
            f"{power_amplitude_inconsistent_count} must be zero"
        )
    if nonfinite_expected_power_count:
        quality_violations.append(
            "nonfinite_expected_power_count="
            f"{nonfinite_expected_power_count} must be zero"
        )
    quality_gate = {
        "passed": not quality_violations,
        "thresholds": {
            "residual_norm_refinement_advisory": BOR_LINEAR_RESIDUAL_MAX,
            "backward_error_max": BOR_LINEAR_BACKWARD_ERROR_MAX,
            "condition_est_max": BOR_CONDITION_EST_MAX,
            "mode_unconverged_count": 0,
            "nonfinite_sample_count": 0,
            "negative_rcs_count": 0,
            "nonfinite_expected_power_count": 0,
            "power_amplitude_consistency_rtol": (
                BOR_POWER_CONSISTENCY_RTOL
            ),
        },
        "values": {
            "residual_norm_max": residual_max,
            "residual_nonfinite_count": residual_nonfinite_count,
            "backward_error_max": backward_error_max,
            "backward_error_nonfinite_count": (
                backward_error_nonfinite_count
            ),
            "condition_est_max": condition_max,
            "condition_missing_or_nonfinite_count":
                condition_missing_count,
            "mode_unconverged_count": unconverged_modes,
            "nonfinite_sample_count": nonfinite_samples,
            "negative_rcs_count": negative_rcs_count,
            "power_amplitude_inconsistent_count": (
                power_amplitude_inconsistent_count
            ),
            "nonfinite_expected_power_count": (
                nonfinite_expected_power_count
            ),
        },
        "violations": quality_violations,
        "reason": (
            "; ".join(quality_violations)
            if quality_violations
            else "BoR linear backward-error, conditioning, modal, and "
                 "field-consistency thresholds satisfied"
        ),
    }
    if quality_violations:
        raise RuntimeError(
            "BoR quality gate failed: " + quality_gate["reason"]
        )

    return {
        "solver": "bor_mom_rcs",
        "scattering_mode": "monostatic",
        "polarizations": ["VV", "HH"],
        "polarization_mapping": {"VV": "VV", "HH": "HH"},
        "rcs_log_unit": "dBsm",
        "rcs_linear_quantity": "sigma_3d",
        "samples": samples,


        "co_solved_samples": samples_by_pol,
        "metadata": {
            "formulation": formulation_label,
            "geometry_kind": kind,
            "frequency_count": int(len(frequencies)),
            "aspect_count": int(len(aspects)),
            "elevation_count": int(len(aspects)),
            "output_aspect_count": len(set(aspects) | (
                {360.0 - angle for angle in aspects if 0.0 < angle < 180.0}
                if expand_to_360 else set())),
            "expanded_to_360": bool(expand_to_360),
            "per_frequency": per_freq_meta,
            "residual_norm_max": residual_max,
            "residual_nonfinite_count": residual_nonfinite_count,
            "residual_norm_refinement_advisory": (
                BOR_LINEAR_RESIDUAL_MAX
            ),
            "backward_error_max": backward_error_max,
            "backward_error_nonfinite_count": (
                backward_error_nonfinite_count
            ),
            "condition_est_max": condition_max,
            "condition_missing_or_nonfinite_count":
                condition_missing_count,
            "condition_est_computed": bool(
                condition_missing_count == 0
                and len(per_freq_meta) > 0
            ),
            "condition_est_method": ("compressed_original_1norm_onenormest" if compressed_requested()
                                     else "lapack_gecon_1norm"),
            "quality_gate": quality_gate,
            "cfie_alpha": float(cfie_alpha),
            "workers": int(workers),
            "far_table_controls_applicable": bool(kind in ("conductor", "sheet")),
            "assembly_requested": expected_assembly,
            "table_precision_requested": table_precision_key,
            "stream_budget_gb": stream_budget,
            "mesh_reference_ghz": mesh_ref_ghz,
            "mesh_control_frequencies_ghz": sorted(mesh_control_frequencies),
            "mesh_wavelength_m": float(mesh_wavelength_m),
            "mesh_max_refractive_index": float(mesh_max_refractive_index),
            "mesh_material_flags": list(mesh_material_flags),
            "warnings": list(materials.warnings),
            "preflight": preflight,
        },
    }


def _bor_channel_result(
    result: 'Dict[str, Any]',
    polarization: 'str',
) -> 'Dict[str, Any]':
    """Expose one co-solved BoR polarization to the shared mesh comparator."""

    channels = result.get("co_solved_samples", {}) or {}
    samples = channels.get(polarization, [])
    if not samples:
        raise ValueError(
            "Certified BoR solve is missing co-solved "
            f"{polarization} samples."
        )
    return {"samples": samples}


@profiled_solve
@reserve_output
@configured
def solve_monostatic_rcs_bor_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = 0.5,
    n_modes: 'Optional[int]' = None,
    mode_tol: 'float' = 1e-6,
    max_elements: 'int' = MAX_ELEMENTS_DEFAULT,
    workers: 'Optional[int]' = None,
    abort_event: 'Optional[threading.Event]' = None,
    table_precision: 'str' = "auto",
    assembly: 'str' = "auto",
    expand_to_360: 'bool' = False,
    stream_budget_gb: 'float' = BOR_STREAM_BUDGET_GB_DEFAULT,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
) -> 'Dict[str, Any]':
    """Production BoR entry point.

    Solve the requested grid on the user mesh and one internally refined
    mesh.  Both co-solved VV/HH complex fields must pass the fixed production
    convergence policy.  Only the refined result is returned.
    """

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)

    common = dict(
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        mesh_reference_ghz=mesh_reference_ghz,
        cfie_alpha=cfie_alpha,
        n_modes=n_modes,
        mode_tol=mode_tol,
        max_elements=max_elements,
        workers=workers,
        abort_event=abort_event,
        table_precision=table_precision,
        assembly=assembly,
        expand_to_360=expand_to_360,
        stream_budget_gb=stream_budget_gb,
    )

    def phase_progress(phase):
        if progress_callback is None:
            return None

        def report(done, total, message):
            total_value = max(int(total), 1)
            done_value = max(0, min(int(done), total_value))
            progress_callback(
                phase * total_value + done_value,
                2 * total_value,
                ("Base mesh: " if phase == 0 else "Certified mesh: ")
                + str(message),
            )

        return report

    base_result = solve_monostatic_rcs_bor(
        geometry_snapshot=geometry_snapshot,
        progress_callback=phase_progress(0),
        **common
    )
    fine_snapshot = scale_snapshot_panel_density(
        geometry_snapshot, policy["fine_factor"]
    )
    fine_snapshot["_bor_certification_refinement_factor"] = float(
        policy["fine_factor"]
    )
    fine_snapshot["_bor_certification_base_segment_n"] = [
        (list(segment.get("properties", []) or []) + ["", ""])[1]
        for segment in list(geometry_snapshot.get("segments", []) or [])
    ]
    fine_result = solve_monostatic_rcs_bor(
        geometry_snapshot=fine_snapshot,
        progress_callback=phase_progress(1),
        **common
    )

    base_frequency_meshes = {
        round(float(row.get("frequency_ghz", 0.0)), 12): row
        for row in list(
            (base_result.get("metadata", {}) or {}).get("per_frequency", [])
            or []
        )
    }
    fine_frequency_meshes = {
        round(float(row.get("frequency_ghz", 0.0)), 12): row
        for row in list(
            (fine_result.get("metadata", {}) or {}).get("per_frequency", [])
            or []
        )
    }
    if set(base_frequency_meshes) != set(fine_frequency_meshes):
        raise RuntimeError(
            "Certified BoR mesh refinement failed: base and fine solves did "
            "not report the same frequency mesh inventory."
        )
    mesh_refinement_records = []
    for frequency in sorted(base_frequency_meshes):
        base_row = base_frequency_meshes[frequency]
        fine_row = fine_frequency_meshes[frequency]
        base_surfaces = [
            int(value)
            for value in list(base_row.get("mesh_elements_by_surface", []) or [])
        ]
        fine_surfaces = [
            int(value)
            for value in list(fine_row.get("mesh_elements_by_surface", []) or [])
        ]
        if not base_surfaces or len(base_surfaces) != len(fine_surfaces):
            raise RuntimeError(
                "Certified BoR mesh refinement failed: base/fine surface "
                f"inventories differ at {frequency:g} GHz."
            )
        stagnant = [
            index
            for index, (base_count, fine_count) in enumerate(
                zip(base_surfaces, fine_surfaces)
            )
            if fine_count <= base_count
        ]
        if stagnant:
            raise RuntimeError(
                "Certified BoR mesh refinement failed: the fine solve did "
                "not increase the realized element count of surface(s) "
                + ", ".join(str(index) for index in stagnant)
                + f" at {frequency:g} GHz (base={base_surfaces}, "
                f"fine={fine_surfaces})."
            )
        base_total = int(sum(base_surfaces))
        fine_total = int(sum(fine_surfaces))
        mesh_refinement_records.append({
            "frequency_ghz": float(frequency),
            "base_elements_by_surface": base_surfaces,
            "fine_elements_by_surface": fine_surfaces,
            "base_elements_total": base_total,
            "fine_elements_total": fine_total,
            "refinement_ratio_total": float(fine_total) / float(base_total),
        })

    per_polarization = {}
    violations = []
    for channel in ("VV", "HH"):
        gate = evaluate_mesh_convergence(
            _bor_channel_result(base_result, channel),
            _bor_channel_result(fine_result, channel),
            rms_limit_db=policy["rms_limit_db"],
            max_abs_limit_db=policy["max_abs_limit_db"],
            complex_rms_limit=policy["complex_rms_limit"],
            complex_max_limit=policy["complex_max_limit"],
            phase_rms_limit_deg=policy["phase_rms_limit_deg"],
            phase_max_limit_deg=policy["phase_max_limit_deg"],
            phase_floor_relative=policy["phase_floor_relative"],
        )
        per_polarization[channel] = gate
        if not bool(gate.get("passed", False)):
            violations.append(
                channel + ": "
                + str(gate.get("reason", "mesh convergence failed"))
            )

    mesh_gate = {
        "schema": "ghost.solver.mesh-convergence.v1",
        "passed": not violations,
        "fine_factor": policy["fine_factor"],
        "published_mesh": "fine",
        "geometry_model": "piecewise_linear_generatrix",
        "geometry_approximation_certified": False,
        "co_solved_polarizations": ["VV", "HH"],
        "policy": policy,
        "polarizations": per_polarization,
        "realized_mesh_refinement": mesh_refinement_records,
        "violations": violations,
        "reason": (
            "; ".join(violations)
            if violations
            else "BoR VV/HH complex-field mesh convergence passed"
        ),
    }
    if violations:
        raise RuntimeError(
            "Certified BoR mesh convergence failed: "
            + mesh_gate["reason"]
        )

    metadata = dict(fine_result.get("metadata", {}) or {})
    metadata["mesh_convergence"] = mesh_gate
    metadata["mesh_convergence_certified"] = True
    metadata["certified_entry_point"] = True
    mesh_gate["base_quality_gate"] = dict(
        (base_result.get("metadata", {}) or {}).get("quality_gate", {}) or {}
    )
    mesh_gate["fine_quality_gate"] = dict(
        metadata.get("quality_gate", {}) or {}
    )
    quality_gate = dict(metadata.get("quality_gate", {}) or {})
    quality_gate["mesh_convergence_certified"] = True
    quality_gate["certification_scope"] = (
        "discrete_linear_system_modal_truncation_and_mesh_convergence"
    )
    metadata["quality_gate"] = quality_gate
    fine_result["metadata"] = metadata
    return fine_result


@profiled_solve
@configured
def solve_monostatic_rcs_bor_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    **kwargs: 'Any',
) -> 'Dict[str, Any]':
    """Single-mesh BoR solve with explicit uncertified-result metadata."""

    result = solve_monostatic_rcs_bor(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        **kwargs,
    )
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence_certified"] = False
    metadata["certified_entry_point"] = False
    metadata["published_mesh"] = "base"
    metadata["survey_mode"] = True
    warning = (
        "SURVEY MODE: solved on the base BoR mesh only. No mesh-convergence "
        "certificate exists for this field."
    )
    warnings = metadata.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = False
        quality_gate["certification_scope"] = (
            "discrete_linear_system_and_modal_truncation_only"
        )
    return result
