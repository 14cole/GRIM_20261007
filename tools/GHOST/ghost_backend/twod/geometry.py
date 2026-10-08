"""2-D materials, geometry validation, and panel/linear-mesh construction."""

import cmath
import csv
import copy
import math
import os
from ghost_backend.execution.runtime import dataclass
import numpy as np
from ghost_backend.twod.formulations.thin_layer import ThinLayerDefinition
from ghost_backend.geometry.io import material_filename_from_row
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from ghost_backend.twod.constants import (
    C0,
    DEFAULT_PANELS_PER_WAVELENGTH,
    EPS,
    ETA0,
    MATERIAL_SINGULAR_TOL,
    MAX_PANELS_DEFAULT,
    MIN_EXPLICIT_PANELS_PER_WAVELENGTH,
    VIRTUAL_SHEET_REGION_START,
)
from ghost_backend.twod.special import _complex_hankel_backend_name


@dataclass
class Panel:
    """Single discretized boundary element used by the solver mesh builder."""

    name: 'str'
    seg_type: 'int'
    ibc_flag: 'int'
    pos_mat: 'int'
    neg_mat: 'int'
    p0: 'np.ndarray'
    p1: 'np.ndarray'
    center: 'np.ndarray'
    tangent: 'np.ndarray'
    normal: 'np.ndarray'
    length: 'float'


    arc_s_center: 'float' = 0.5
    primitive_key: 'str' = ''

@dataclass
class LinearNode:
    """Unique mesh node for a continuous piecewise-linear boundary discretization."""

    xy: 'np.ndarray'
    key: 'Tuple[int, int]'

@dataclass
class LinearElement:
    """Straight boundary element; endpoints precede any polynomial interior nodes."""

    name: 'str'
    seg_type: 'int'
    ibc_flag: 'int'
    pos_mat: 'int'
    neg_mat: 'int'
    node_ids: 'Tuple[int, ...]'
    p0: 'np.ndarray'
    p1: 'np.ndarray'
    center: 'np.ndarray'
    tangent: 'np.ndarray'
    normal: 'np.ndarray'
    length: 'float'
    panel_index: 'int'
    arc_s_center: 'float' = 0.5
    primitive_key: 'str' = ''

@dataclass
class LinearMesh:
    """Continuous linear boundary mesh assembled from boundary primitives."""

    nodes: 'List[LinearNode]'
    elements: 'List[LinearElement]'


def copy_linear_mesh(mesh: 'LinearMesh') -> 'LinearMesh':
    """A mesh with its own node list and element records over the same node
    and element data, so polynomial enrichment of the copy (``basis.enrich``
    appends nodes and rewrites ``node_ids``) leaves ``mesh`` untouched.  The
    geometric near-pair memo is shared with the copy: element centres and
    lengths are the same, so whichever of them counts first serves both."""
    from ghost_backend.twod.formulations.regions import near_pair_memo
    clone = LinearMesh(nodes=list(mesh.nodes), elements=[copy.copy(element) for element in mesh.elements])
    box = near_pair_memo(mesh)
    if box is not None:
        clone._near_pair_memo = box
    return clone


@dataclass
class PanelCoupledInfo:
    """
    Per-element material and interface bookkeeping for the coupled formulation.

    The unknown vector is [u_trace, q_minus]. This record maps each element's
    plus-side and minus-side constitutive data into the assembled system.
    """

    seg_type: 'int'
    plus_region: 'int'
    minus_region: 'int'
    plus_has_incident: 'bool'
    minus_has_incident: 'bool'
    eps_plus: 'complex'
    mu_plus: 'complex'
    eps_minus: 'complex'
    mu_minus: 'complex'
    k_plus: 'complex'
    k_minus: 'complex'
    q_plus_beta: 'complex'
    q_plus_gamma: 'complex'
    bc_kind: 'str'


    robin_impedance: 'complex'

# A frequency this close (relative) to a characterized end is that end: a sweep
# point computed as 0.8 + 4*0.1 = 1.2000000000000002 GHz must sample the 1.2 GHz
# row, not raise.
TABLE_RANGE_RELATIVE_TOLERANCE = 1.0e-12


def _table_sample_frequency(freq_ghz: 'float', freqs_ghz: 'np.ndarray') -> 'float':
    """Validate a table sample frequency, clamping round-off at the range ends."""

    freq_ghz = float(freq_ghz)
    if not math.isfinite(freq_ghz):
        raise ValueError("Material-table sample frequency must be finite.")
    fmin = float(freqs_ghz[0])
    fmax = float(freqs_ghz[-1])
    if fmin - TABLE_RANGE_RELATIVE_TOLERANCE * abs(fmin) <= freq_ghz < fmin:
        return fmin
    if fmax < freq_ghz <= fmax + TABLE_RANGE_RELATIVE_TOLERANCE * abs(fmax):
        return fmax
    if freq_ghz < fmin or freq_ghz > fmax:
        raise ValueError(
            f"Material-table sample frequency {freq_ghz:.17g} GHz is outside "
            f"the characterized range [{fmin:.17g}, {fmax:.17g}] GHz."
        )
    return freq_ghz


@dataclass
class ComplexTable:
    """Frequency-dependent complex scalar table with linear interpolation."""

    freqs_ghz: 'np.ndarray'
    values: 'np.ndarray'

    def sample(self, freq_ghz: 'float') -> 'complex':
        freq_ghz = _table_sample_frequency(freq_ghz, self.freqs_ghz)
        if len(self.freqs_ghz) == 1:
            return complex(self.values[0])
        real = np.interp(freq_ghz, self.freqs_ghz, self.values.real)
        imag = np.interp(freq_ghz, self.freqs_ghz, self.values.imag)
        return complex(real, imag)

@dataclass
class ImpedanceTaper:
    """Spatially tapered surface impedance along a segment.

    The segment is parametrized by arc_s in [0, 1], running from the segment's
    start endpoint (as drawn by the user) to the end endpoint.  At arc_s = 0
    the impedance equals z_start, at arc_s = 1 it equals z_end, with the
    interpolation weighting determined by ``kind``:

    - ``"linear"``   : w = s                               (straight ramp)
    - ``"cosine"``   : w = 0.5 * (1 - cos(pi * s))          (Hann/raised-cosine:
                                                             C^1 at both ends,
                                                             good for edge taper)
    - ``"exp"``      : log-space interpolation              (octave-per-length
                                                             ramp between two
                                                             nonzero impedances)

    For ``"linear"`` and ``"cosine"`` the endpoints may be zero (PEC) or
    arbitrary complex.  For ``"exp"`` both endpoints must be nonzero; zero
    endpoints are coerced to a tiny nonzero floor.

    This model is independent of frequency.  Combining a taper with a
    frequency-dependent material table would require a 2-D (s, f) model and is
    out of scope for the initial implementation.
    """

    kind: 'str'
    z_start: 'complex'
    z_end: 'complex'

    _ALLOWED_KINDS = ("constant", "linear", "cosine", "exp")

    def __post_init__(self) -> 'None':
        if self.kind not in self._ALLOWED_KINDS:
            raise ValueError(
                f"Unknown impedance taper kind '{self.kind}'. "
                f"Expected one of {self._ALLOWED_KINDS}."
            )
        self.z_start = _validate_passive_surface_impedance(
            self.z_start, "taper z_start"
        )
        if self.kind == "constant":


            self.z_end = self.z_start
        else:
            self.z_end = _validate_passive_surface_impedance(
                self.z_end, "taper z_end"
            )

    def evaluate(self, arc_s: 'float') -> 'complex':
        s = float(max(0.0, min(1.0, arc_s)))
        if self.kind == "constant":
            return self.z_start
        if self.kind == "linear":
            w = s
            return (1.0 - w) * self.z_start + w * self.z_end
        if self.kind == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * s))
            return (1.0 - w) * self.z_start + w * self.z_end

        z1 = self.z_start if abs(self.z_start) > EPS else complex(EPS, 0.0)
        z2 = self.z_end if abs(self.z_end) > EPS else complex(EPS, 0.0)
        return cmath.exp((1.0 - s) * cmath.log(z1) + s * cmath.log(z2))

@dataclass
class MediumTable:
    """Frequency-dependent (eps, mu) table with linear interpolation."""

    freqs_ghz: 'np.ndarray'
    eps_values: 'np.ndarray'
    mu_values: 'np.ndarray'

    def sample(self, freq_ghz: 'float') -> 'Tuple[complex, complex]':
        freq_ghz = _table_sample_frequency(freq_ghz, self.freqs_ghz)
        if len(self.freqs_ghz) == 1:
            return complex(self.eps_values[0]), complex(self.mu_values[0])
        eps_r = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.real)
        eps_i = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.imag)
        mu_r = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.real)
        mu_i = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.imag)
        return complex(eps_r, eps_i), complex(mu_r, mu_i)

class MaterialLibrary:
    """Material lookup facade for inline values and frequency tables."""

    def __init__(
        self,
        impedance_models: 'Dict[int, Union[complex, ComplexTable]]',
        dielectric_models: 'Dict[int, Union[Tuple[complex, complex], MediumTable]]',
    ):
        self.impedance_models = impedance_models
        self.dielectric_models = dielectric_models
        self.warnings: 'List[str]' = []
        self.information: 'List[str]' = []
        self._warning_seen: 'Set[str]' = set()

    @classmethod
    def from_entries(
        cls,
        ibcs_entries: 'List[List[str]]',
        dielectric_entries: 'List[List[str]]',
        base_dir: 'str',
    ) -> "MaterialLibrary":
        impedance_models: 'Dict[int, Union[complex, ComplexTable]]' = {}
        dielectric_models: 'Dict[int, Union[Tuple[complex, complex], MediumTable]]' = {}
        seen_impedance_flags: 'Set[int]' = set()
        seen_dielectric_flags: 'Set[int]' = set()

        for row in ibcs_entries:
            if not row:
                continue
            flag = _parse_material_definition_flag(
                row[0], "IBC material definition flag"
            )
            if flag in seen_impedance_flags:
                raise ValueError(f"Duplicate IBC material flag {flag}.")
            seen_impedance_flags.add(flag)
            if len(row) > 1 and str(row[1]).lower() == "thin_dielectric":
                impedance_models[flag] = ThinLayerDefinition.from_row(row)
                continue
            if len(row) == 2:
                filename = material_filename_from_row(row)
                if filename is None:
                    raise ValueError(
                        f"IBC flag {flag}: file-backed material definitions "
                        "must use 'flag filename.csv'."
                    )
                path = _resolve_material_file(base_dir, filename)
                impedance_models[flag] = _load_impedance_csv(path)
                continue


            tokens = [str(t).strip() for t in row[1:] if str(t).strip() != ""]
            if len(tokens) != 5:
                raise ValueError(
                    f"IBC flag {flag}: inline impedance requires "
                    "'<kind> R_start X_start R_end X_end' "
                    f"(got {len(tokens)} data tokens after the flag)."
                )
            kind = tokens[0].strip().lower()
            r_start = _parse_material_float(tokens[1], f"IBC flag {flag} R_start")
            x_start = _parse_material_float(tokens[2], f"IBC flag {flag} X_start")
            r_end = _parse_material_float(tokens[3], f"IBC flag {flag} R_end")
            x_end = _parse_material_float(tokens[4], f"IBC flag {flag} X_end")
            impedance_models[flag] = ImpedanceTaper(
                kind=kind,
                z_start=complex(r_start, x_start),
                z_end=complex(r_end, x_end),
            )

        for row in dielectric_entries:
            if not row:
                continue
            flag = _parse_material_definition_flag(
                row[0], "Dielectric material definition flag"
            )
            if flag in seen_dielectric_flags:
                raise ValueError(f"Duplicate dielectric material flag {flag}.")
            seen_dielectric_flags.add(flag)
            if len(row) == 2:
                filename = material_filename_from_row(row)
                if filename is None:
                    raise ValueError(
                        f"Dielectric flag {flag}: inline material requires "
                        "exactly 'flag eps_real eps_imag mu_real mu_imag'; "
                        "file-backed definitions require "
                        "'flag filename.csv'."
                    )
                path = _resolve_material_file(base_dir, filename)
                dielectric_models[flag] = _load_dielectric_csv(path)
                continue
            if len(row) != 5 or any(
                    str(token).strip() == "" for token in row):
                raise ValueError(
                    f"Dielectric flag {flag}: inline material requires exactly "
                    "'flag eps_real eps_imag mu_real mu_imag' with no blank "
                    f"fields (got {len(row)} fields).")
            eps_real = _parse_material_float(
                row[1], f"Dielectric flag {flag} epsilon real part")
            eps_imag = _parse_material_float(
                row[2], f"Dielectric flag {flag} epsilon imaginary part")
            mu_real = _parse_material_float(
                row[3], f"Dielectric flag {flag} mu real part")
            mu_imag = _parse_material_float(
                row[4], f"Dielectric flag {flag} mu imaginary part")


            eps_raw = _ensure_finite_complex(
                complex(eps_real, eps_imag),
                f"Dielectric flag {flag} epsilon",
            )
            mu_raw = _ensure_finite_complex(
                complex(mu_real, mu_imag),
                f"Dielectric flag {flag} mu",
            )
            eps, mu = _validate_passive_medium(
                eps_raw, mu_raw, f"Dielectric flag {flag}"
            )
            dielectric_models[flag] = (eps, mu)

        for model in impedance_models.values():
            if isinstance(model, ThinLayerDefinition) and model.dielectric_flag not in dielectric_models:
                raise ValueError(f"Thin layer references undefined dielectric flag {model.dielectric_flag}.")
        return cls(impedance_models=impedance_models, dielectric_models=dielectric_models)

    def get_impedance(self, flag: 'int', freq_ghz: 'float', arc_s: 'Optional[float]' = None) -> 'complex':
        if flag <= 0:
            return 0.0 + 0.0j
        model = self.impedance_models.get(flag)
        if model is None:
            raise ValueError(f"Undefined IBC flag {flag}.")
        if isinstance(model, ThinLayerDefinition):
            raise ValueError("A thin dielectric layer is a transmitting sheet, not an opaque IBC. Assign it only to TYPE 1.")
        if isinstance(model, ComplexTable):
            return _validate_passive_surface_impedance(
                model.sample(freq_ghz),
                f"IBC flag {flag} impedance sampled at {freq_ghz:g} GHz",
            )
        if isinstance(model, ImpedanceTaper):
            s = 0.5 if arc_s is None else float(arc_s)
            return _validate_passive_surface_impedance(
                model.evaluate(s),
                f"IBC flag {flag} tapered impedance at s={s:g}",
            )
        return _validate_passive_surface_impedance(
            model, f"IBC flag {flag} impedance"
        )

    def is_tapered_impedance(self, flag: 'int') -> 'bool':
        """True if the IBC flag is spatially tapered along the segment."""
        model = self.impedance_models.get(flag)
        return (
            isinstance(model, ImpedanceTaper)
            and model.kind != "constant"
        )

    def get_medium(self, flag: 'int', freq_ghz: 'float') -> 'Tuple[complex, complex]':
        if flag <= 0:
            return 1.0 + 0.0j, 1.0 + 0.0j
        model = self.dielectric_models.get(flag)
        if model is None:
            raise ValueError(f"Undefined dielectric flag {flag}.")
        if isinstance(model, MediumTable):
            eps, mu = model.sample(freq_ghz)
            return _validate_passive_medium(
                eps,
                mu,
                f"Dielectric flag {flag} sampled at {freq_ghz:g} GHz",
            )
        eps, mu = model
        return _validate_passive_medium(eps, mu, f"Dielectric flag {flag}")

    def _warn_once(self, message: 'str') -> 'None':
        if message in self._warning_seen:
            return
        self._warning_seen.add(message)
        self.warnings.append(message)

    def warn_once(self, message: 'str') -> 'None':
        self._warn_once(message)

    def inform_once(self, message: 'str') -> 'None':
        if message not in self.information:
            self.information.append(message)


def _parse_flag(token: 'Any') -> 'int':
    text = str(token).strip().lower()
    if not text:
        return 0
    if text.startswith("mat."):
        text = text.split("mat.", 1)[1]
    try:
        return int(float(text))
    except ValueError:
        return 0

def _parse_float(token: 'Any', default: 'float' = 0.0) -> 'float':
    try:
        return float(token)
    except (TypeError, ValueError):
        return default

def _parse_int(token: 'Any', default: 'int' = 0) -> 'int':
    try:
        return int(round(float(token)))
    except (TypeError, ValueError):
        return default

def _parse_geometry_float(token: 'Any', context: 'str') -> 'float':
    """Strict numeric parser for solver-facing geometry snapshots."""

    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a finite numeric value; got {token!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite; got {token!r}.")
    return value

def _parse_geometry_integer(
    token: 'Any',
    context: 'str',
    *,
    allow_mat_prefix: 'bool' = False,
) -> 'int':
    """Strict integral parser for TYPE/N/material flag fields."""

    text = str(token).strip()
    if allow_mat_prefix and text.lower().startswith("mat."):
        text = text[4:]
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer; got {token!r}.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{context} must be an integer; got {token!r}.")
    return int(value)

def _parse_material_definition_flag(token: 'Any', context: 'str') -> 'int':
    """Strict positive ID parser for user-supplied material-library rows."""

    value = _parse_geometry_integer(
        token, context, allow_mat_prefix=True
    )
    if value <= 0:
        raise ValueError(
            f"{context} must be a positive integer; got {token!r}."
        )
    return value

def _ensure_finite_complex(value: 'complex', context: 'str') -> 'complex':
    z = complex(value)
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        raise ValueError(f"{context} contains non-finite value {z!r}.")
    return z

def _parse_material_float(token: 'Any', context: 'str') -> 'float':
    """Parse one explicitly supplied material field without silent defaults."""

    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite numeric value; got {token!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite; got {token!r}.")
    return value

def _passivity_tolerance(value: 'complex') -> 'float':
    return 64.0 * np.finfo(float).eps * max(1.0, abs(complex(value)))

def _validate_passive_surface_impedance(value: 'complex', context: 'str') -> 'complex':
    """Validate a passive Leontovich impedance under the solver convention."""

    z = _ensure_finite_complex(value, context)
    if z.real < -_passivity_tolerance(z):
        raise ValueError(
            f"{context} has negative resistance Re(Zs)={z.real:g} ohm. "
            "Active/gain surface impedances are not supported."
        )
    return z

def _validate_passive_medium(
    eps: 'complex',
    mu: 'complex',
    context: 'str',
) -> 'Tuple[complex, complex]':
    """Validate passive constitutive values for the exp(+j*omega*t) convention.

    Require Im(eps) and Im(mu) <= 0. Singular or near-singular ENZ/MNZ values raise
    ValueError.
    """

    eps_eval = _ensure_finite_complex(eps, f"{context} epsilon")
    mu_eval = _ensure_finite_complex(mu, f"{context} mu")
    if abs(eps_eval) <= MATERIAL_SINGULAR_TOL:
        raise ValueError(
            f"{context} has unsupported singular/near-ENZ epsilon {eps_eval!r} "
            f"(|epsilon| <= {MATERIAL_SINGULAR_TOL:g}); it will not be replaced with free space."
        )
    if abs(mu_eval) <= MATERIAL_SINGULAR_TOL:
        raise ValueError(
            f"{context} has unsupported singular/near-MNZ mu {mu_eval!r} "
            f"(|mu| <= {MATERIAL_SINGULAR_TOL:g}); it will not be replaced with free space."
        )
    if eps_eval.imag > _passivity_tolerance(eps_eval):
        raise ValueError(
            f"{context} epsilon has gain-sign Im(epsilon)={eps_eval.imag:g}. "
            "For e^(+j*omega*t), passive media require Im(epsilon) <= 0; "
            "active/gain media are not supported."
        )
    if mu_eval.imag > _passivity_tolerance(mu_eval):
        raise ValueError(
            f"{context} mu has gain-sign Im(mu)={mu_eval.imag:g}. "
            "For e^(+j*omega*t), passive media require Im(mu) <= 0; "
            "active/gain media are not supported."
        )
    return eps_eval, mu_eval


def _resolve_material_file(base_dir: 'str', filename: 'str') -> 'str':
    """Resolve a validated material sidecar in the geometry directory only."""

    name = str(filename)
    folder = os.path.abspath(str(base_dir))
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Could not locate material file {name} in declared material directory "
        f"{folder}. Material tables are never searched in the process working "
        "directory.")


def _material_base_dir_for_snapshot(
    geometry_snapshot: 'Dict[str, Any]',
    material_base_dir: 'Optional[str]',
) -> 'str':
    """Return the single declared directory used for material sidecars.

    An explicit directory has priority.  Otherwise a file-backed snapshot
    inherits the directory containing ``source_path``.  Only a genuinely
    pathless, programmatic snapshot uses the process working directory as its
    documented default.  Thus changing cwd cannot change a loaded geometry's
    material model.
    """

    if material_base_dir is not None and str(material_base_dir).strip():
        return os.path.abspath(os.path.expanduser(str(material_base_dir)))
    source_path = str(geometry_snapshot.get("source_path", "") or "").strip()
    if source_path:
        return os.path.dirname(
            os.path.abspath(os.path.expanduser(source_path))
        )
    return os.path.abspath(os.getcwd())


def _read_csv_numeric_rows(
    path: 'str', expected_header: 'List[str]'
) -> 'List[List[float]]':
    """Read a headered, comma-separated material/IBC CSV with frequency in Hz.

    GHOST and FREDDY accept the same UTF-8 CSV contract: an optional BOM,
    blank lines and full-line # comments, exact headers (cell whitespace is
    ignored), and finite numeric data with positive, unique frequencies.
    """
    if not str(path).lower().endswith(".csv"):
        raise ValueError(f"Material/IBC file must use the .csv extension: {path}")
    rows: 'List[List[float]]' = []
    header_found = False
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        for lineno, raw_line in enumerate(csv_file, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                parts = [part.strip() for part in next(csv.reader([raw_line], strict=True))]
            except csv.Error as exc:
                raise ValueError(f"{path}: line {lineno} contains invalid CSV: {exc}") from exc
            if not header_found:
                if parts != expected_header:
                    raise ValueError(
                        f"{path}: line {lineno} must have comma-separated header "
                        f"{','.join(expected_header)} (frequency in Hz); "
                        f"found {','.join(parts)}."
                    )
                header_found = True
                continue
            if len(parts) != len(expected_header):
                raise ValueError(
                    f"{path}: line {lineno} must contain exactly "
                    f"{len(expected_header)} comma-separated columns; found {len(parts)}."
                )
            try:
                values = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path}: line {lineno} contains a non-numeric value.") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: line {lineno} contains a non-finite value.")
            if values[0] <= 0:
                raise ValueError(
                    f"{path}: line {lineno} has non-positive frequency {values[0]:g} Hz."
                )
            rows.append(values)
    if not header_found:
        raise ValueError(f"{path}: missing required header {','.join(expected_header)}.")
    if not rows:
        raise ValueError(f"{path}: no material data rows found after the header.")
    rows.sort(key=lambda row: row[0])
    for previous, current in zip(rows, rows[1:]):
        if current[0] == previous[0]:
            raise ValueError(f"{path}: duplicate frequency {current[0]:g} Hz.")
    return rows


def _load_impedance_csv(path: 'str') -> 'ComplexTable':
    """Load frequency_hz,resistance_ohm,reactance_ohm."""

    rows = _read_csv_numeric_rows(
        path,
        ["frequency_hz", "resistance_ohm", "reactance_ohm"],
    )
    freqs = np.asarray([row[0] / 1.0e9 for row in rows], dtype=float)
    values = np.asarray(
        [complex(row[1], row[2]) for row in rows],
        dtype=np.complex128,
    )
    for row_index, value in enumerate(values, start=1):
        _validate_passive_surface_impedance(
            value, f"Impedance CSV '{path}' data row {row_index}"
        )
    return ComplexTable(freqs_ghz=freqs, values=values)


def _load_dielectric_csv(path: 'str') -> 'MediumTable':
    """Load frequency_hz,eps_real,eps_imag,mu_real,mu_imag."""

    rows = _read_csv_numeric_rows(
        path,
        ["frequency_hz", "eps_real", "eps_imag", "mu_real", "mu_imag"],
    )
    freqs = np.asarray([row[0] / 1.0e9 for row in rows], dtype=float)
    eps_values = np.asarray(
        [complex(row[1], row[2]) for row in rows],
        dtype=np.complex128,
    )
    mu_values = np.asarray(
        [complex(row[3], row[4]) for row in rows],
        dtype=np.complex128,
    )
    for row_index, (eps, mu) in enumerate(
        zip(eps_values, mu_values), start=1
    ):
        _validate_passive_medium(
            eps, mu, f"Dielectric CSV '{path}' data row {row_index}"
        )
    return MediumTable(
        freqs_ghz=freqs,
        eps_values=eps_values,
        mu_values=mu_values,
    )


def _unit_scale_to_meters(units: 'str') -> 'float':
    value = (units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")

def _discretize_primitive(p0: 'np.ndarray', p1: 'np.ndarray', count: 'int') -> 'List[np.ndarray]':
    """Generate panel endpoints for a straight-line primitive.

    The first and last points are the primitive's own vertices, bit for bit.
    ``p0 + (p1 - p0) * 1.0`` can differ from ``p1`` in the last place, which is
    enough to put the two copies of a shared vertex on different sides of a
    node-snap rounding boundary and split a closed contour.
    """

    count = max(1, int(count))
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    points = [p0 + (p1 - p0) * (i / count) for i in range(count + 1)]
    points[0] = p0.copy()
    points[-1] = p1.copy()
    return points

def _primitive_length(p0: 'np.ndarray', p1: 'np.ndarray') -> 'float':
    return float(np.linalg.norm(p1 - p0))


# Where the surface impedance of an opaque conductor jumps, the Robin law
# q + alpha*u = 0 changes abruptly (TM alpha = j*k*eta/Zs, TE alpha =
# j*k*Zs/eta). Next to PEC the TM condition changes TYPE (u = 0 to a Robin law)
# and the density behaves like r**(-1/2): uniform linear panels converge at
# first order (0.97, 0.49, 0.24, 0.12 % at 104..832 panels on a PEC|75-20j
# circle). A large contrast does the same until the panels resolve 1/|alpha|:
# 1 ohm | 75-20j gives 0.36, 0.17, 0.089 % in TM and 1e5 ohm | 75-20j gives
# 2.0, 1.1, 0.57 % in TE. Splitting the panel on each side of the junction
# geometrically brings all of these to the accuracy of a smooth contour (0.24,
# 0.058, 0.015 %) for eight extra panels per junction. Below a contrast of
# about four the uniform mesh is already second order and within 1.5 times the
# graded error, so such junctions (a staircase approximating a taper) are left
# alone. The contrast is the complex jump |Z1 - Z2| against the larger
# magnitude: more than 1 - 1/4 of it, which is the factor of four for laws in
# phase and about 44 degrees for laws of equal magnitude (a reactive +75j
# against 75-20j stays second order but at 2.4 times the graded TM error).
# Both polarizations share the graded mesh, so their operators stay
# interchangeable.
JUNCTION_GRADING_LEVELS = 4
JUNCTION_GRADING_RATIO = 0.5
JUNCTION_GRADING_CONTRAST = 4.0
# Fixed levels only shrink the first-order term (the innermost panel is h/16):
# on the PEC|75-20j circle the TM error ratio per doubling decays 4.1, 3.5, 2.4,
# 2.0 from 104 to 1664 panels. One more level for every doubling of the density
# beyond this many panels per wavelength keeps the singular term under the
# smooth error: 0.0036 % and 0.0007 % at 832 and 1664 panels, the TE values.
JUNCTION_GRADING_DENSITY = 32.0
_OPAQUE_CONDUCTOR_TYPES = (2, 4)


def _junction_grading_levels(panel_length: 'float', wavelength) -> 'int':
    """Levels for a junction whose end panel has this length on a mesh of this wavelength."""
    levels = int(JUNCTION_GRADING_LEVELS)
    if levels <= 0 or not panel_length > 0.0:
        return max(levels, 0)
    if wavelength is not None and math.isfinite(wavelength) and wavelength > 0.0:
        density = float(wavelength) / float(panel_length)
        if density > JUNCTION_GRADING_DENSITY:
            levels += int(math.ceil(math.log2(density / JUNCTION_GRADING_DENSITY) - 1.0e-9))
    # the innermost panel stays two decades above the 1e-9 m node snap
    return max(0, min(levels, int(math.floor(math.log2(max(panel_length / 1.0e-7, 1.0))))))


def impedance_jump_is_graded(first: 'complex', second: 'complex') -> 'bool':
    """Whether two evaluated surface impedances (0 = PEC) differ enough to grade their junction."""
    larger = max(abs(first), abs(second))
    return larger > 0.0 and abs(first - second) > (1.0 - 1.0 / JUNCTION_GRADING_CONTRAST) * larger


def _inline_impedance_laws(geometry_snapshot) -> "MaterialLibrary":
    """Inline impedance laws of a snapshot, for callers that mesh without a material library.

    File-backed tables need the library (and frequencies) the field solve
    uses; without them such a law is graded only where it meets PEC.
    """

    models = {}
    for row in geometry_snapshot.get("ibcs", []) or []:
        if not row or len(row) <= 2 or str(row[1]).strip().lower() == "thin_dielectric":
            continue
        try:
            models.update(MaterialLibrary.from_entries([row], [], base_dir="").impedance_models)
        except (TypeError, ValueError):
            continue    # the material library proper reports a malformed row
    return MaterialLibrary(models, {})


def _end_impedance(materials, flag: 'int', freq_ghz, arc_s: 'float'):
    """``Zs`` at a conductor segment end as the field solve evaluates it.

    0 is PEC (flag 0 or a law within the solver's own ``EPS`` of zero), ``None``
    a law that cannot be evaluated here (a table without a frequency or outside
    its range; the field solve reports those).
    """

    if flag <= 0:
        return 0.0j
    model = materials.impedance_models.get(flag)
    if model is None or (freq_ghz is None and isinstance(model, ComplexTable)):
        return None
    try:
        value = complex(materials.get_impedance(flag, freq_ghz, arc_s=arc_s))
    except ValueError:
        return None
    return 0.0j if abs(value) <= EPS else value


def _boundary_condition_junctions(segments, meters_scale: 'float', materials,
                                  frequencies_ghz=None) -> 'Set[Tuple[int, int]]':
    """Snapped vertices where the boundary condition of an opaque conductor jumps.

    The condition is the evaluated one, not the flag: the law at that segment
    end (``arc_s`` 0 or 1 in the drawing direction, as a taper is sampled) at
    every frequency the mesh serves. A zero-ohm law is PEC exactly as flag 0 is,
    a table that reaches zero at one frequency of a fixed mesh grades that mesh,
    and two laws within ``JUNCTION_GRADING_CONTRAST`` of each other
    (``impedance_jump_is_graded``: magnitude and phase) do not.
    """

    frequencies = [float(f) for f in (frequencies_ghz if frequencies_ghz is not None else ())] or [None]
    ends: 'Dict[Tuple[int, int], List[List[Optional[float]]]]' = {}
    welder = _NodeWelder()
    for seg in segments:
        props = list(seg.get("properties", []) or [])
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2)
        )
        pairs = list(seg.get("point_pairs", []) or [])
        if seg_type not in _OPAQUE_CONDUCTOR_TYPES or not pairs:
            continue
        flag = _parse_flag(props[2] if len(props) > 2 else 0)
        for pair, names, arc_s in ((pairs[0], ("x1", "y1"), 0.0), (pairs[-1], ("x2", "y2"), 1.0)):
            key = welder.key(np.asarray([
                _parse_float(pair.get(names[0], 0.0), 0.0) * meters_scale,
                _parse_float(pair.get(names[1], 0.0), 0.0) * meters_scale,
            ], dtype=float))
            ends.setdefault(key, []).append(
                [_end_impedance(materials, flag, f, arc_s) for f in frequencies])
    junctions = _JunctionVertices(welder=welder)
    for key, laws in ends.items():
        for at_frequency in zip(*laws):
            known = [value for value in at_frequency if value is not None]
            # An unevaluated law is taken to be an impedance: it jumps next to PEC.
            jump = len(known) < len(at_frequency) and any(value == 0.0 for value in known)
            if jump or any(impedance_jump_is_graded(first, second)
                           for index, first in enumerate(known) for second in known[index + 1:]):
                junctions.add(key)
                break
    return junctions


def _grade_toward_junctions(pts, junctions, wavelength=None):
    """Split the end panels of a primitive geometrically toward junction vertices."""

    pts = [np.asarray(point, dtype=float) for point in pts]
    if len(pts) < 2 or not junctions:
        return pts
    if isinstance(junctions, _JunctionVertices):
        at_start, at_end = junctions.at(pts[0]), junctions.at(pts[-1])
    else:
        at_start = _linear_node_snap_key(pts[0]) in junctions
        at_end = _linear_node_snap_key(pts[-1]) in junctions
    if not (at_start or at_end):
        return pts
    if len(pts) == 2 and at_start and at_end:
        pts = [pts[0], 0.5 * (pts[0] + pts[1]), pts[1]]

    def fractions(first, second):
        levels = _junction_grading_levels(float(np.linalg.norm(second - first)), wavelength)
        return [JUNCTION_GRADING_RATIO ** level for level in range(levels, 0, -1)]
    if at_start:
        pts = [pts[0]] + [pts[0] + (pts[1] - pts[0]) * f for f in fractions(pts[0], pts[1])] + pts[1:]
    if at_end:
        pts = pts[:-1] + [pts[-1] + (pts[-2] - pts[-1]) * f for f in reversed(fractions(pts[-1], pts[-2]))] + [pts[-1]]
    return pts

# A wavelength-controlled primitive may exceed the target panel length by this
# fraction of a panel before it receives another one. A plain ceil doubled a
# regular 128-gon from 128 to 256 panels between 3.05 and 3.06 GHz (primitive
# 1.002 target lengths) and turned exact multiples into one extra panel through
# round-off; the largest panel now stays within 5 % of the target.
PANEL_COUNT_CEIL_TOLERANCE = 0.05


def _wavelength_panel_count(primitive_len: 'float', target: 'float') -> 'int':
    return max(1, int(math.ceil(primitive_len / target - PANEL_COUNT_CEIL_TOLERANCE)))


def _panel_count_from_n(n_prop: 'int', primitive_len: 'float', min_wavelength: 'float') -> 'int':
    """
    Convert geometry n property to panel count.

    n > 0: explicit panel count.
    n < 0: panels-per-wavelength style control.
    """

    if primitive_len <= EPS:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    if n_prop < 0:
        n_wave = max(1, abs(n_prop))
        target = min_wavelength / n_wave
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("The controlling mesh wavelength must be positive and finite.")
        return _wavelength_panel_count(primitive_len, target)

    if min_wavelength > EPS:
        target = min_wavelength / float(DEFAULT_PANELS_PER_WAVELENGTH)
        return _wavelength_panel_count(primitive_len, target)
    return max(1, int(math.ceil(primitive_len / (primitive_len / 10.0 + EPS))))


def _reverse_point_pairs(point_pairs: 'List[Dict[str, Any]]') -> 'List[Dict[str, Any]]':
    """Reverse a primitive chain: last primitive first, endpoints swapped."""

    reversed_pairs: 'List[Dict[str, Any]]' = []
    for pair in reversed(point_pairs):
        reversed_pairs.append({
            'x1': pair.get('x2', 0.0),
            'y1': pair.get('y2', 0.0),
            'x2': pair.get('x1', 0.0),
            'y2': pair.get('y1', 0.0),
        })
    return reversed_pairs


def _check_segment_orientation_or_raise(
    segments: 'List[Dict[str, Any]]',
    meters_scale: 'float' = 1.0,
) -> 'None':
    """
    Run the shared winding / air-side consistency checks (geometry_io) and
    raise on any ERROR finding.

    The TM formulations are winding-insensitive, but the TE MFIE/Robin rows
    carry a +-1/2 mass jump tied to the normal direction, so a wrong winding
    or inconsistent air side silently corrupts TE results (residuals stay
    tiny).  The solver deliberately refuses to run rather than silently
    reorienting the user's geometry.
    """

    from ghost_backend.geometry.io import (
        chains_from_snapshot_segments,
        check_orientation_consistency,
    )

    meters_scale = float(meters_scale)
    if not math.isfinite(meters_scale) or meters_scale <= 0.0:
        raise ValueError("meters_scale must be positive and finite.")
    chains = chains_from_snapshot_segments(segments)
    coordinates = [point for chain in chains for point in chain.points]
    if coordinates:
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        diagonal_m = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) * meters_scale
    else:
        diagonal_m = 0.0


    tolerance_m = max(1.0e-12, 1.0e-9 * max(diagonal_m, 1.0))
    findings = check_orientation_consistency(
        chains, tol=tolerance_m / meters_scale
    )
    errors = [msg for severity, _idx, msg in findings if severity == "ERROR"]
    if errors:
        raise ValueError(
            "Geometry orientation check failed:\n  - " + "\n  - ".join(errors)
        )


def _normalize_segment_orientation(
    seg_type: 'int',
    point_pairs: 'List[Dict[str, Any]]',
    meters_scale: 'float',
) -> 'List[Dict[str, Any]]':
    """
    Pass-through: the user's endpoint order is the source of truth.

    This routine does not reorient contours.  The user is responsible for
    drawing each segment so that the normal (computed from endpoint order)
    points in the physically intended direction.

    The per-panel-type convention mapping from user-facing geometry to
    solver-internal plus/minus assignments is handled separately in
    `_apply_user_convention_flip` (called from `_build_panels`).
    """

    return point_pairs


def _apply_user_convention_flip(
    seg_type: 'int',
    point_pairs: 'List[Dict[str, Any]]',
) -> 'List[Dict[str, Any]]':
    """
    Translate the user's drawing convention to the solver's internal convention.

    User-facing convention (this is what the user is asked to do when drawing
    geometry in the GUI or writing a .geo file):

        TYPE 2 (PEC / IBC body in air):
            Draw the boundary so the normal points INTO AIR, i.e., away
            from the conductor.  Example: on the top of a PEC body drawn
            left-to-right, the normal points UP.

        TYPE 3 (air / dielectric interface):
            Draw the boundary so the normal points INTO AIR, away from
            the dielectric region.  pos_mat names the dielectric material
            ON THE OPPOSITE SIDE OF THE NORMAL.  Example: on the top of a
            dielectric body drawn left-to-right, the normal points UP
            (into air), and pos_mat is the dielectric below.

        TYPE 4 (dielectric / PEC interface):
            No air is involved.  Draw the boundary so the normal points
            FROM THE PEC INTO THE DIELECTRIC (i.e., into the pos_mat region).
            Example: on the top of a PEC-backed dielectric coating drawn
            left-to-right, the normal points UP into the dielectric
            coating that sits above.

        TYPE 5 (dielectric / dielectric interface):
            No air is involved.  The normal points FROM neg_mat INTO pos_mat,
            i.e., pos_mat is on the normal side.  User chooses which
            dielectric to label pos_mat and which to label neg_mat based on
            the endpoint order they drew.

        TYPE 1 (free-floating resistive / reactive card):
            Both sides of a free card are air; the sheet impedance BC is
            symmetric.  Normal direction is physically irrelevant; the
            user's endpoint order is accepted as-is.

    Solver-internal convention (unchanged):
        - TYPE 1 sheet:  plus = virtual sheet region,  minus = air
        - TYPE 2 PEC:    plus = interior (-1),         minus = air
        - TYPE 3 diel:   plus = pos_mat dielectric,       minus = air
        - TYPE 4 coat:   plus = pos_mat dielectric,       minus = PEC interior
        - TYPE 5 d/d:    plus = pos_mat,                  minus = neg_mat

    The solver's "plus" side is always the side the stored panel normal points
    toward.  For TYPE 2 and TYPE 3 the user draws the normal pointing away
    from the plus side, so we reverse endpoint order to align conventions.
    For TYPE 4 and TYPE 5 the user already draws with the normal pointing
    toward the plus / pos_mat side, so no flip is needed.  TYPE 1 is symmetric.
    """

    if seg_type not in (2, 3):
        return point_pairs
    return _reverse_point_pairs(point_pairs)

def _snapshot_segments(geometry_snapshot: 'Dict[str, Any]') -> 'List[Dict[str, Any]]':
    return list(geometry_snapshot.get('segments', []) or [])

def _solver_point_key(x: 'float', y: 'float', tol: 'float') -> 'Tuple[int, int]':
    inv = 1.0 / max(tol, 1e-12)
    return int(round(float(x) * inv)), int(round(float(y) * inv))

def _points_close(a: 'Tuple[float, float]', b: 'Tuple[float, float]', tol: 'float') -> 'bool':
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) <= (tol * tol)

def _segment_intersects_strict(
    a1: 'Tuple[float, float]',
    a2: 'Tuple[float, float]',
    b1: 'Tuple[float, float]',
    b2: 'Tuple[float, float]',
    tol: 'float',
) -> 'bool':
    if _points_close(a1, b1, tol) or _points_close(a1, b2, tol) or _points_close(a2, b1, tol) or _points_close(a2, b2, tol):
        return False

    def orient(p, q, r):
        return (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1])) - (float(q[1]) - float(p[1])) * (float(r[0]) - float(p[0]))

    def on_seg(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - tol <= float(q[0]) <= max(float(p[0]), float(r[0])) + tol
            and min(float(p[1]), float(r[1])) - tol <= float(q[1]) <= max(float(p[1]), float(r[1])) + tol
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    # `orient` is a cross product (length^2), so comparing it directly against
    # `tol` (a length) makes the effective clearance tolerance tol/length --
    # coarse for long primitives, catastrophic for short ones.  Scale each
    # threshold by the length of the line it is measured against so the test
    # is a true perpendicular distance of `tol`.
    ta = tol * max(math.hypot(a2[0] - a1[0], a2[1] - a1[1]), EPS)
    tb = tol * max(math.hypot(b2[0] - b1[0], b2[1] - b1[1]), EPS)

    if ((o1 > ta and o2 < -ta) or (o1 < -ta and o2 > ta)) and ((o3 > tb and o4 < -tb) or (o3 < -tb and o4 > tb)):
        return True
    if abs(o1) <= ta and on_seg(a1, b1, a2):
        return True
    if abs(o2) <= ta and on_seg(a1, b2, a2):
        return True
    if abs(o3) <= tb and on_seg(b1, a1, b2):
        return True
    if abs(o4) <= tb and on_seg(b1, a2, b2):
        return True
    return False

def validate_geometry_snapshot_for_solver(
    geometry_snapshot: 'Dict[str, Any]',
    base_dir: 'str',
    meters_scale: 'float' = 1.0,
    material_library: 'Optional[MaterialLibrary]' = None,
) -> 'Dict[str, Any]':
    """
    Strict solver-side preflight for geometry/material consistency.

    This complements the GUI validator and protects headless solves / exports.
    Fatal problems raise before assembly begins.

    ``meters_scale`` converts snapshot coordinates to meters (pass the same
    unit scale the solver uses).  It is needed to detect "cracks": endpoint
    gaps small enough to look connected at drawing precision but larger than
    the mesh node-snap tolerance (1e-9 m absolute), which would silently
    mesh a closed body as an open contour.
    """

    meters_scale = float(meters_scale)
    if not math.isfinite(meters_scale) or meters_scale <= 0.0:
        raise ValueError("meters_scale must be positive and finite.")

    segments = _snapshot_segments(geometry_snapshot)
    if not segments:
        raise ValueError('Geometry snapshot contains no segments.')

    ibc_rows = [list(row) for row in (geometry_snapshot.get('ibcs', []) or []) if list(row)]
    diel_rows = [list(row) for row in (geometry_snapshot.get('dielectrics', []) or []) if list(row)]
    ibc_flags = {
        _parse_material_definition_flag(
            row[0], f"IBC definition row {idx + 1} flag"
        )
        for idx, row in enumerate(ibc_rows)
    }
    diel_flags = {
        _parse_material_definition_flag(
            row[0], f"Dielectric definition row {idx + 1} flag"
        )
        for idx, row in enumerate(diel_rows)
    }


    if material_library is None:
        MaterialLibrary.from_entries(ibc_rows, diel_rows, base_dir)

    warnings: 'List[str]' = []
    primitives: 'List[Tuple[int, int, str, Tuple[float, float], Tuple[float, float]]]' = []
    all_points: 'List[Tuple[float, float]]' = []
    chain_discontinuities: 'List[Tuple[str, int]]' = []

    for seg_idx, seg in enumerate(segments):
        props = list(seg.get('properties', []) or [])


        if len(props) < 5:
            props.extend([''] * (5 - len(props)))
        seg_name = str(seg.get('name', f'segment_{seg_idx + 1}'))
        header_type_token = seg.get('seg_type')
        property_type_token = props[0]
        has_header_type = (
            header_type_token is not None
            and bool(str(header_type_token).strip())
        )
        has_property_type = bool(str(property_type_token).strip())
        header_type = (
            _parse_geometry_integer(
                header_type_token, f"Segment '{seg_name}' header TYPE"
            )
            if has_header_type else None
        )
        property_type = (
            _parse_geometry_integer(
                property_type_token, f"Segment '{seg_name}' properties TYPE"
            )
            if has_property_type else None
        )
        if (
            header_type is not None
            and property_type is not None
            and header_type != property_type
        ):
            raise ValueError(
                f"Segment '{seg_name}' declares TYPE {header_type} in its "
                f"header but TYPE {property_type} in properties[0]. "
                "The two TYPE declarations must match."
            )
        seg_type = (
            property_type
            if property_type is not None
            else header_type if header_type is not None else 0
        )
        if str(props[1]).strip():
            _parse_geometry_integer(props[1], f"Segment '{seg_name}' N")
        ibc_flag = (
            _parse_geometry_integer(
                props[2], f"Segment '{seg_name}' IBC flag",
                allow_mat_prefix=True,
            )
            if str(props[2]).strip() else 0
        )
        pos_mat = (
            _parse_geometry_integer(
                props[3], f"Segment '{seg_name}' pos_mat flag",
                allow_mat_prefix=True,
            )
            if str(props[3]).strip() else 0
        )
        neg_mat = (
            _parse_geometry_integer(
                props[4], f"Segment '{seg_name}' neg_mat flag",
                allow_mat_prefix=True,
            )
            if str(props[4]).strip() else 0
        )
        point_pairs = list(seg.get('point_pairs', []) or [])

        if seg_type < 1 or seg_type > 5:
            raise ValueError(f"Segment '{seg_name}' has invalid TYPE '{props[0]}'; expected 1..5.")
        for field_name, flag in (
            ("IBC", ibc_flag),
            ("pos_mat", pos_mat),
            ("neg_mat", neg_mat),
        ):
            if flag < 0:
                raise ValueError(
                    f"Segment '{seg_name}' {field_name} flag must be "
                    f"non-negative; got {flag}."
                )
        if not point_pairs:
            raise ValueError(f"Segment '{seg_name}' has no primitives/point_pairs.")

        if ibc_flag > 0 and seg_type in (3, 5):
            raise ValueError(
                f"TYPE {seg_type} segment '{seg_name}' assigns IBC flag {ibc_flag} to a "
                "dielectric transmission interface. Surface impedance on TYPE 3/5 "
                "interfaces is not implemented by the 2D transmission formulations; "
                "remove the IBC flag or model the impedance on a supported TYPE 1, 2, "
                "or 4 boundary."
            )

        prev_end = None
        for prim_idx, pair in enumerate(point_pairs):
            context = f"Segment '{seg_name}' primitive {prim_idx + 1}"
            missing = [key for key in ('x1', 'y1', 'x2', 'y2') if key not in pair]
            if missing:
                raise ValueError(
                    f"{context} is missing coordinate field(s): "
                    + ", ".join(missing)
                )
            x1 = _parse_geometry_float(pair['x1'], f"{context} x1")
            y1 = _parse_geometry_float(pair['y1'], f"{context} y1")
            x2 = _parse_geometry_float(pair['x2'], f"{context} x2")
            y2 = _parse_geometry_float(pair['y2'], f"{context} y2")
            vals = [x1, y1, x2, y2]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} contains non-finite coordinates.")
            p1 = (x1 * meters_scale, y1 * meters_scale)
            p2 = (x2 * meters_scale, y2 * meters_scale)
            if _points_close(p1, p2, NODE_WELD_TOLERANCE_M):
                raise ValueError(
                    f"Segment '{seg_name}' primitive {prim_idx + 1} is "
                    f"{math.hypot(p2[0] - p1[0], p2[1] - p1[1]):.3g} m long, not longer "
                    f"than the {NODE_WELD_TOLERANCE_M:g} m mesh node tolerance, so its two "
                    "ends would be one mesh node (a degenerate element). Remove it or merge "
                    "it into a neighbouring primitive."
                )
            primitives.append((seg_idx, prim_idx, seg_name, p1, p2))
            all_points.extend([p1, p2])
            if prev_end is not None and not _points_close(prev_end, p1, 1e-9):


                chain_discontinuities.append((seg_name, prim_idx))
            prev_end = p2

        if ibc_flag > 0:
            if ibc_flag not in ibc_flags:
                raise ValueError(f"Segment '{seg_name}' references undefined IBC flag {ibc_flag}.")

        if seg_type == 3:
            if pos_mat <= 0:
                raise ValueError(f"TYPE 3 segment '{seg_name}' requires pos_mat > 0.")
            if pos_mat not in diel_flags:
                raise ValueError(f"TYPE 3 segment '{seg_name}' references undefined dielectric flag {pos_mat}.")
        elif seg_type == 4:
            if pos_mat <= 0:
                raise ValueError(f"TYPE 4 segment '{seg_name}' requires pos_mat > 0.")
            if pos_mat not in diel_flags:
                raise ValueError(f"TYPE 4 segment '{seg_name}' references undefined dielectric flag {pos_mat}.")
        elif seg_type == 5:
            if pos_mat <= 0 or neg_mat <= 0:
                raise ValueError(f"TYPE 5 segment '{seg_name}' requires pos_mat > 0 and neg_mat > 0.")
            if pos_mat == neg_mat:
                raise ValueError(
                    f"TYPE 5 segment '{seg_name}' assigns the same dielectric "
                    f"flag {pos_mat} to both sides. A same-medium interface is "
                    "physically redundant and must be removed."
                )
            for flag in (pos_mat, neg_mat):
                if flag not in diel_flags:
                    raise ValueError(f"TYPE 5 segment '{seg_name}' references undefined dielectric flag {flag}.")

    xs = [p[0] for p in all_points] if all_points else [0.0]
    ys = [p[1] for p in all_points] if all_points else [0.0]


    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    tol = max(1e-8, 1e-6 * diag)

    # Endpoints are one vertex exactly when the mesher welds them into one node.
    snap_tol_m = NODE_WELD_TOLERANCE_M
    welder = _NodeWelder(snap_tol_m)
    primitive_keys: 'List[Tuple[Tuple[int, int], Tuple[int, int]]]' = []
    vertex_point: 'Dict[Tuple[int, int], Tuple[float, float]]' = {}
    vertex_feature: 'Dict[Tuple[int, int], float]' = {}
    node_degree: 'Dict[Tuple[int, int], int]' = {}
    for _, _, _, p1, p2 in primitives:
        key1, key2 = welder.key(p1), welder.key(p2)
        primitive_keys.append((key1, key2))
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        for key, point in ((key1, p1), (key2, p2)):
            vertex_point.setdefault(key, point)
            vertex_feature[key] = min(vertex_feature.get(key, math.inf), length)
            node_degree[key] = node_degree.get(key, 0) + 1

    dangling_nodes = sum(1 for v in node_degree.values() if v == 1)
    high_degree_nodes = sum(1 for v in node_degree.values() if v > 2)
    if dangling_nodes > 0:
        warnings.append(f'Geometry contains {dangling_nodes} dangling endpoint node(s).')
    if high_degree_nodes > 0:
        warnings.append(f'Geometry contains {high_degree_nodes} high-degree node(s) (>2 connected primitives).')

    # The drawing-precision proximity tolerance is capped at half the local
    # feature size: the shortest primitive at a vertex or on a primitive. A
    # legitimate 2e-9..1e-6 m primitive is a feature, not a crack, and its two
    # ends (or its neighbours' ends) must not be compared at 1e-6 * diagonal.
    proximity_fraction = 0.5
    primitive_feature = [
        min(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), vertex_feature[key1], vertex_feature[key2])
        for (_, _, _, p1, p2), (key1, key2) in zip(primitives, primitive_keys)
    ]

    def local_tolerance(*features: 'float') -> 'float':
        return min(tol, proximity_fraction * min(features))

    joined = {frozenset(keys) for keys in primitive_keys}
    vertex_keys = list(vertex_point)
    from ghost_backend.geometry.spatial import overlapping_pairs
    endpoint_bounds = [(vertex_point[key][0], vertex_point[key][0],
                        vertex_point[key][1], vertex_point[key][1]) for key in vertex_keys]
    for i, j in overlapping_pairs(endpoint_bounds, tol):
        key_i, key_j = vertex_keys[i], vertex_keys[j]
        px, py = vertex_point[key_i]
        qx, qy = vertex_point[key_j]
        gap = math.hypot(qx - px, qy - py)
        if gap <= snap_tol_m or gap > local_tolerance(vertex_feature[key_i], vertex_feature[key_j]):
            continue
        if frozenset((key_i, key_j)) in joined:
            continue        # the two ends of one primitive
        raise ValueError(
            f"Geometry crack: endpoints ({px:.9g}, {py:.9g}) m and ({qx:.9g}, {qy:.9g}) m "
            f"are {gap:.3g} m apart -- close enough to look connected, but "
            "beyond the 1e-9 m mesh node-snap tolerance, so they would mesh as an OPEN "
            "gap. Make the endpoints exactly coincident (or separate them intentionally)."
        )


    primitive_bounds = []
    for _, _, _, p1, p2 in primitives:
        primitive_bounds.append((
            min(float(p1[0]), float(p2[0])),
            max(float(p1[0]), float(p2[0])),
            min(float(p1[1]), float(p2[1])),
            max(float(p1[1]), float(p2[1])),
        ))
    for i, j in overlapping_pairs(primitive_bounds, tol):
        seg_i, prim_i, name_i, a1, a2 = primitives[i]
        seg_j, prim_j, name_j, b1, b2 = primitives[j]
        pair_tol = local_tolerance(primitive_feature[i], primitive_feature[j])


        same_fwd = _points_close(a1, b1, pair_tol) and _points_close(a2, b2, pair_tol)
        same_rev = _points_close(a1, b2, pair_tol) and _points_close(a2, b1, pair_tol)
        if same_fwd or same_rev:
            raise ValueError(
                f"Duplicate primitive: '{name_i}' primitive {prim_i + 1} and "
                f"'{name_j}' primitive {prim_j + 1} have identical endpoints. "
                "Remove one -- a doubled boundary doubles the surface currents."
            )


        shared = None
        for pa, key_a in zip((a1, a2), primitive_keys[i]):
            for pb, key_b in zip((b1, b2), primitive_keys[j]):
                if key_a == key_b:
                    shared = (pa, pb)
                    break
            if shared:
                break
        if shared is not None:
            oa = a2 if shared[0] is a1 else a1
            ob = b2 if shared[1] is b1 else b1
            ux, uy = oa[0] - shared[0][0], oa[1] - shared[0][1]
            vx, vy = ob[0] - shared[1][0], ob[1] - shared[1][1]
            lu = math.hypot(ux, uy)
            lv = math.hypot(vx, vy)
            if lu > EPS and lv > EPS:
                cross = abs(ux * vy - uy * vx) / (lu * lv)
                dot = (ux * vx + uy * vy) / (lu * lv)
                if cross < 1.0e-7 and dot > 0.0 and min(lu, lv) > snap_tol_m:
                    raise ValueError(
                        f"Collinear overlapping primitives: '{name_i}' primitive {prim_i + 1} and "
                        f"'{name_j}' primitive {prim_j + 1} run along the same line from a shared "
                        f"endpoint, overlapping for {min(lu, lv):.3g} m. "
                        "Split or remove the overlapping span."
                    )
            continue

        if seg_i == seg_j and abs(prim_i - prim_j) <= 1:
            continue
        if _segment_intersects_strict(a1, a2, b1, b2, pair_tol):
            raise ValueError(
                f"Geometry contains an unsupported segment intersection between '{name_i}' primitive {prim_i + 1} and '{name_j}' primitive {prim_j + 1}."
            )

    if chain_discontinuities:
        seg_name, prim_idx = chain_discontinuities[0]
        raise ValueError(
            f"Segment '{seg_name}' has a disconnected primitive chain "
            f"between elements {prim_idx} and {prim_idx + 1}. "
            "Primitives within one segment must chain head-to-tail; "
            "put disconnected geometry in separate segments."
        )


    _check_segment_orientation_or_raise(segments, meters_scale)

    return {
        'segment_count': int(len(segments)),
        'primitive_count': int(len(primitives)),
        'dangling_nodes': int(dangling_nodes),
        'high_degree_nodes': int(high_degree_nodes),
        'warning_count': int(len(warnings)),
        'warnings': warnings,
    }

# Slow surface waves. A reactive opaque surface guides a bound wave: in TM (E_z)
# for X < 0 with index sqrt(1 + (eta/X)^2), in TE (H_z) for X > 0 with index
# sqrt(1 + (X/eta)^2). A 1-lambda square meshed at 20 panels per free-space
# wavelength had 12.7 % field error at -60j ohm (TM, index 6.4), 10.8 % at
# +2000j and 6.9 % at +600j (TE) against 0.12 / 0.56 % for PEC; 75-20j, whose
# bound wave is damped within a fraction of its own wavelength, had 0.06 %.
# The mesh wavelength of such a segment is the bound wavelength at every mesh
# frequency (both polarizations share one mesh), by at most this factor.
SURFACE_WAVE_MAX_DENSITY_FACTOR = 4.0
# A wave that needs more than this density factor is not resolved even at the
# cap (under 7 panels per bound wavelength) and the capped mesh did not help:
# -30j (index 12.6) went from 5.8 to 8.2 % and -20j (18.9) from 0.62 to 1.2 %
# at four times the panels, while +4000j (10.7) improved from 16 to 3.5 %.
# Such tightly bound waves keep the material mesh.
SURFACE_WAVE_MAX_UNRESOLVED_FACTOR = 12.0
# A tapered law is sampled at this many points along its segment.
SURFACE_WAVE_TAPER_SAMPLES = 9


def _bound_surface_wave_index(
    z_surface: 'complex',
    eps_medium: 'complex' = 1.0 + 0.0j,
    mu_medium: 'complex' = 1.0 + 0.0j,
) -> 'float':
    """Phase index (against free space) of the wave bound to a flat impedance surface.

    Under exp(+j*omega*t) the field above the surface varies as
    exp(-j*beta*x - kappa*y). The Robin laws of ``_surface_robin_alpha`` give
    kappa/k = -j*Zs/eta (TE) and -j*eta/Zs (TM), and beta/k = sqrt(1 +
    (kappa/k)^2). A wave counts when it is bound (Re kappa > 0) and propagates
    (its attenuation per radian is below one, |Im beta| <= Re beta); the larger
    of the two polarizations is returned, 0 when neither guides a wave.
    """

    z = complex(z_surface)
    if abs(z) <= EPS:
        return 0.0
    n_medium = _causal_medium_index(eps_medium, mu_medium)
    eta_medium = _medium_eta(eps_medium, mu_medium)
    best = 0.0
    for ratio in (z / eta_medium, eta_medium / z):
        kappa = -1j * ratio
        if kappa.real <= 0.0:
            continue
        beta = cmath.sqrt(1.0 - ratio * ratio)
        if abs(beta.imag) > beta.real:
            continue
        best = max(best, float(abs((n_medium * beta).real)))
    return best


def _segment_surface_wave_index(materials, seg_type: 'int', ibc_flag: 'int',
                                pos_mat: 'int', freq_ghz: 'float') -> 'float':
    """Largest bound-wave index of an opaque conductor segment's law at one frequency."""

    if seg_type not in _OPAQUE_CONDUCTOR_TYPES or ibc_flag <= 0:
        return 0.0
    model = materials.impedance_models.get(ibc_flag)
    if model is None or isinstance(model, ThinLayerDefinition):
        return 0.0
    try:
        eps, mu = (materials.get_medium(pos_mat, freq_ghz) if seg_type == 4
                   else (1.0 + 0.0j, 1.0 + 0.0j))
        if isinstance(model, ImpedanceTaper) and model.kind != "constant":
            positions = np.linspace(0.0, 1.0, SURFACE_WAVE_TAPER_SAMPLES)
        else:
            positions = (0.5,)
        values = [materials.get_impedance(ibc_flag, freq_ghz, arc_s=float(s)) for s in positions]
    except ValueError:
        return 0.0      # an unevaluable law or medium: the field solve reports it
    return max(_bound_surface_wave_index(z, eps, mu) for z in values)


def _surface_wave_mesh_wavelength(material_wavelength: 'float', materials, seg_type: 'int',
                                  ibc_flag: 'int', pos_mat: 'int', frequencies) -> 'float':
    """Mesh wavelength of a segment: its material wavelength or its bound surface wavelength."""

    if materials is None or not frequencies or seg_type not in _OPAQUE_CONDUCTOR_TYPES or ibc_flag <= 0:
        return material_wavelength
    shortest = material_wavelength
    for freq_ghz in frequencies:
        index = _segment_surface_wave_index(materials, seg_type, ibc_flag, pos_mat, freq_ghz)
        if index > 0.0:
            bound = C0 / (float(freq_ghz) * 1.0e9) / index
            if material_wavelength <= SURFACE_WAVE_MAX_UNRESOLVED_FACTOR * bound:
                shortest = min(shortest, bound)
    return max(shortest, material_wavelength / SURFACE_WAVE_MAX_DENSITY_FACTOR)


def _segment_material_wavelength(geometry_snapshot, segment, materials, frequencies) -> 'Optional[float]':
    """Shortest wavelength of the media on this segment's own sides (air included).

    None when a medium cannot be evaluated here (the field solve reports it).
    """

    local = dict(geometry_snapshot, segments=[segment])
    try:
        return min(_mesh_wavelength_for_snapshot(local, materials, float(f))[0] for f in frequencies)
    except ValueError:
        return None


def _spread_rank(index: 'int') -> 'float':
    """Base-2 radical inverse: spreads successive picks among equal panels along a chain."""

    result, weight = 0.0, 0.5
    while index:
        if index & 1:
            result += weight
        index >>= 1
        weight *= 0.5
    return result


def _certification_fine_counts(base_counts: 'Sequence[int]', lengths: 'Sequence[float]',
                               factor: 'float') -> 'List[int]':
    """Fine-mesh panel counts for one segment chain of the certification pair.

    The chain receives max(B + 1, ceil(factor * B)) panels for B base panels,
    added where the panels are longest (never fewer than the base count on any
    primitive). The former per-primitive max(n + 1, ceil(factor * n)) gave a
    densely drawn polyline twice its panels (8x the LU work) when every
    primitive carried a single panel.
    """

    counts = [max(1, int(c)) for c in base_counts]
    lengths = [float(length) for length in lengths]
    base_total = sum(counts)
    target = max(base_total + 1, int(math.ceil(base_total * float(factor))))
    if len(counts) == 1:
        return [target]

    def total(h: 'float') -> 'int':
        return sum(max(c, int(math.ceil(length / h))) for c, length in zip(counts, lengths))

    # Water-fill: the shortest panel-length threshold that stays within the budget.
    hi = max(length / c for c, length in zip(counts, lengths))
    if not hi > 0.0:
        counts[0] += target - base_total
        return counts
    lo = hi / max(2.0, float(factor)) / 2.0
    while total(lo) <= target and lo > EPS:
        lo *= 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if total(mid) <= target:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1.0e-12 * hi:
            break
    counts = [max(c, int(math.ceil(length / hi))) for c, length in zip(counts, lengths)]
    remainder = target - sum(counts)
    if remainder > 0:
        scale = max(length / c for c, length in zip(counts, lengths))
        key = [round(lengths[i] / counts[i] / scale * 1.0e9) for i in range(len(counts))]
        # Primitives with equal panel lengths (mirror images of a symmetric
        # body) are refined together, so a symmetric base mesh stays
        # symmetric: a group that does not fit the remaining budget is still
        # refined whole when that overshoots the target by at most 1% (at
        # least two panels), which only refines the chain further.  Only a
        # larger group (a densely drawn polyline) is refined in spread order.
        allowance = max(2, int(math.ceil(0.01 * target)))
        for value in sorted(set(key), reverse=True):
            if remainder <= 0:
                break
            group = [i for i in range(len(counts)) if key[i] == value]
            if len(group) > remainder + allowance:
                group = sorted(group, key=_spread_rank)[:remainder]
            for i in group:
                counts[i] += 1
            remainder -= len(group)
    return counts


# Faceting advisory. Refinement and certification keep the drawn vertices, so
# the chord error of a curve drawn with straight primitives is invisible to
# them: a 32-gon at ka = 30 is 0.53 / 0.82 dB (TM / TE) from the circle it
# approximates. Runs of at least this many consecutive vertices turning the
# same way by at most FACET_ADVISORY_MAX_TURN_DEG look like a drawn curve (the
# hp corner rule treats sharper turns as corners); an implied sagitta above
# 1/200 of the medium wavelength is reported as information, not a warning.
FACET_ADVISORY_MAX_TURN_DEG = 15.0
FACET_ADVISORY_MIN_RUN = 3
FACET_ADVISORY_SAGITTA_FRACTION = 1.0 / 200.0


def _faceting_sagitta(ends) -> 'float':
    """Largest implied sagitta of the primitives of a segment chain that is drawn as a curve.

    ``ends`` are (p0, p1, length) in chain order. At a vertex turning by
    theta between primitives of lengths L1 and L2 the implied radius is about
    (L1 + L2) / (2 * theta), so a primitive of length L on it departs from the
    arc by L**2 / (8 * R).
    """

    count = len(ends)
    if count < FACET_ADVISORY_MIN_RUN:
        return 0.0
    start_points = np.asarray([end[0] for end in ends], dtype=float)
    end_points = np.asarray([end[1] for end in ends], dtype=float)
    lengths = np.asarray([end[2] for end in ends], dtype=float)
    before = np.arange(count - 1)
    if count > 2 and np.hypot(*(end_points[-1] - start_points[0])) <= NODE_WELD_TOLERANCE_M:
        before = np.arange(count)          # closed chain: the closing vertex too
    after = (before + 1) % count
    with np.errstate(divide='ignore', invalid='ignore'):
        directions = (end_points - start_points) / lengths[:, None]
        u, v = directions[before], directions[after]
        theta = np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], np.sum(u * v, axis=1))
    connected = (np.hypot(*(end_points[before] - start_points[after]).T) <= NODE_WELD_TOLERANCE_M) \
        & (lengths[before] > EPS) & (lengths[after] > EPS)
    small = connected & (np.abs(theta) > 1.0e-6) & (np.abs(theta) <= math.radians(FACET_ADVISORY_MAX_TURN_DEG))
    turns = np.where(small, np.sign(theta), 0.0)
    sagitta = 0.0
    index, total = 0, len(turns)
    while index < total:
        if turns[index] == 0.0:
            index += 1
            continue
        stop = index
        while stop + 1 < total and turns[stop + 1] == turns[index]:
            stop += 1
        if stop - index + 1 >= FACET_ADVISORY_MIN_RUN:
            run = slice(index, stop + 1)
            la, lb = lengths[before[run]], lengths[after[run]]
            radius = 0.5 * (la + lb) / np.abs(theta[run])
            sagitta = max(sagitta, float(np.max(np.maximum(la, lb) ** 2 / (8.0 * radius))))
        index = stop + 1
    return sagitta


def _positive_frequencies(frequencies_ghz) -> 'List[float]':
    """Sorted distinct positive finite mesh frequencies (GHz); empty without frequencies."""

    if frequencies_ghz is None:
        return []
    values = {float(f) for f in frequencies_ghz}
    return sorted(f for f in values if math.isfinite(f) and f > 0.0)


def _segment_mesh_flags(seg) -> 'Tuple[int, int, int, int, int]':
    """(TYPE, N, IBC flag, pos_mat, neg_mat) of a segment as the mesher reads them."""

    props = list(seg.get("properties", []) or [])
    seg_type = _parse_flag(
        props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2)
    )
    n_prop = _parse_int(props[1] if len(props) > 1 else 0, 0)
    ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
    pos_mat = _parse_flag(props[3] if len(props) > 3 else 0)
    neg_mat = _parse_flag(props[4] if len(props) > 4 else 0)
    return seg_type, n_prop, ibc_flag, pos_mat, neg_mat


def _reference_panel_counts(
    geometry_snapshot: 'Dict[str, Any]',
    meters_scale: 'float',
    min_wavelength: 'float',
    materials=None,
    frequencies_ghz=None,
) -> 'List[Tuple[int, bool]]':
    """(panel count, explicit N) per primitive of the global P1 mesh, before junction grading.

    The counts ``_build_panels`` gives the same inputs on its global-wavelength
    path (surface-wave sizing included), without allocating any points.
    """

    law_library = materials if materials is not None else _inline_impedance_laws(geometry_snapshot)
    mesh_frequencies = _positive_frequencies(frequencies_ghz)
    result: 'List[Tuple[int, bool]]' = []
    for seg in geometry_snapshot.get("segments", []) or []:
        seg_type, n_prop, ibc_flag, pos_mat, _neg = _segment_mesh_flags(seg)
        wavelength = _surface_wave_mesh_wavelength(
            min_wavelength, law_library, seg_type, ibc_flag, pos_mat, mesh_frequencies)
        for pair in seg.get("point_pairs", []) or []:
            length = _primitive_length(
                np.asarray([_parse_float(pair.get("x1", 0.0), 0.0) * meters_scale,
                            _parse_float(pair.get("y1", 0.0), 0.0) * meters_scale], dtype=float),
                np.asarray([_parse_float(pair.get("x2", 0.0), 0.0) * meters_scale,
                            _parse_float(pair.get("y2", 0.0), 0.0) * meters_scale], dtype=float),
            )
            result.append((_panel_count_from_n(n_prop, length, wavelength), n_prop > 0))
    return result


def _build_panels(
    geometry_snapshot: 'Dict[str, Any]',
    meters_scale: 'float',
    min_wavelength: 'float',
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    segment_wavelengths=None,
    materials=None,
    frequencies_ghz=None,
    notices=None,
) -> 'List[Panel]':
    """
    Discretize all geometry primitives into oriented boundary elements.

    Normal direction follows endpoint ordering of each primitive.  Wrong
    winding is a hard preflight error (see
    `_check_segment_orientation_or_raise`), never silently corrected here.

    ``materials`` and ``frequencies_ghz`` (every frequency this mesh serves)
    let junction grading evaluate the boundary condition the field solve will
    apply, size reactive conductor segments for their bound surface wave, hold
    explicit panel counts to the segment's own material wavelength and report
    drawn faceting. Solves, backend selection and scheduling pass the same
    pair, so they count the same panels; without a library only inline laws
    are evaluated, and without frequencies no surface wave is sized.

    Advisories (information, never warnings) go to ``notices.inform_once``
    when a per-solve notice sink is given, else to ``materials.inform_once``.
    """

    from ghost_backend.twod.adaptive_geometry import protected_vertices, panel_parameters
    hp = '_2d_hp_coarsening' in geometry_snapshot
    protected = protected_vertices(geometry_snapshot, meters_scale) if hp else set()
    panels: 'List[Panel]' = []
    segments = geometry_snapshot.get("segments", []) or []
    refinement_factor = float(
        geometry_snapshot.get("_2d_certification_refinement_factor", 1.0)
        or 1.0
    )
    base_segment_n = list(
        geometry_snapshot.get("_2d_certification_base_segment_n", []) or []
    )
    if not math.isfinite(refinement_factor) or refinement_factor < 1.0:
        raise ValueError(
            "Internal 2-D certification refinement factor must be finite and >= 1."
        )
    if refinement_factor > 1.0 and len(base_segment_n) != len(segments):
        raise ValueError(
            "Internal 2-D certification refinement metadata does not match "
            "the geometry segment count."
        )

    if segment_wavelengths is not None:
        if len(segment_wavelengths) != len(segments) or any(not math.isfinite(v) or v < min_wavelength for v in segment_wavelengths):
            raise ValueError('Invalid local material mesh wavelengths.')
    law_library = materials if materials is not None else _inline_impedance_laws(geometry_snapshot)
    junctions = _boundary_condition_junctions(
        segments, meters_scale, law_library, frequencies_ghz)
    mesh_frequencies = _positive_frequencies(frequencies_ghz)
    panel_limit = max(1, int(max_panels))
    for seg_idx, seg in enumerate(segments):
        material_wavelength = min_wavelength if segment_wavelengths is None else segment_wavelengths[seg_idx]
        seg_type, n_prop, ibc_flag, pos_mat, neg_mat = _segment_mesh_flags(seg)
        name = str(seg.get("name", "segment"))
        wavelength = _surface_wave_mesh_wavelength(
            material_wavelength, law_library, seg_type, ibc_flag, pos_mat, mesh_frequencies)

        point_pairs = list(seg.get("point_pairs", []) or [])


        point_pairs = _normalize_segment_orientation(seg_type, point_pairs, meters_scale)
        point_pairs = _apply_user_convention_flip(seg_type, point_pairs)


        seg_start_idx = len(panels)

        ends = []
        for prim_idx, pair in enumerate(point_pairs):
            p0 = np.asarray([
                _parse_float(pair.get("x1", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y1", 0.0), 0.0) * meters_scale,
            ], dtype=float)
            p1 = np.asarray([
                _parse_float(pair.get("x2", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y2", 0.0), 0.0) * meters_scale,
            ], dtype=float)
            prim_len = _primitive_length(p0, p1)
            if EPS < prim_len <= NODE_WELD_TOLERANCE_M:
                raise ValueError(
                    f"Segment '{name}' has a primitive {prim_len:.3g} m long, not longer than "
                    f"the {NODE_WELD_TOLERANCE_M:g} m mesh node tolerance: its two ends would "
                    "be one mesh node. Remove it or merge it into a neighbouring primitive."
                )
            ends.append((p0, p1, prim_len))
        lengths = [end[2] for end in ends]

        if refinement_factor > 1.0:
            base_n_prop = _parse_int(base_segment_n[seg_idx], 0)
            base_counts = [_panel_count_from_n(base_n_prop, length, wavelength) for length in lengths]
            fine_total = max(sum(base_counts) + 1, int(math.ceil(sum(base_counts) * refinement_factor)))
            if not hp and len(panels) + fine_total > panel_limit:
                raise ValueError('Discretization exceeds the configured panel limit.')
            counts = _certification_fine_counts(base_counts, lengths, refinement_factor) if ends else []
        else:
            counts = [_panel_count_from_n(n_prop, length, wavelength) for length in lengths]

        local_wavelength = None
        sagitta = 0.0
        if materials is not None and mesh_frequencies and min_wavelength > EPS:
            sagitta = _faceting_sagitta(ends)
            if n_prop > 0 or sagitta > FACET_ADVISORY_SAGITTA_FRACTION * min_wavelength:
                local_wavelength = _segment_material_wavelength(
                    geometry_snapshot, seg, materials, mesh_frequencies)
        if n_prop > 0 and min_wavelength > EPS:
            # The gross floor of an explicit count uses the media this segment
            # touches: a distant high-index rod must not force an unrelated
            # air-only segment to a finer explicit count.
            floor_wavelength = local_wavelength if local_wavelength is not None else material_wavelength
            for prim_len, count in zip(lengths, counts):
                minimum_count = max(
                    1,
                    int(math.ceil(
                        prim_len
                        * float(MIN_EXPLICIT_PANELS_PER_WAVELENGTH)
                        / floor_wavelength
                    )),
                )
                if count < minimum_count:
                    raise ValueError(
                        f"Segment '{name}' explicit N={n_prop} under-resolves "
                        f"a {prim_len:.6g} m primitive at the segment's material "
                        f"wavelength {floor_wavelength:.6g} m: at least "
                        f"N={minimum_count} is required for the gross "
                        f"{MIN_EXPLICIT_PANELS_PER_WAVELENGTH}-panels-per-"
                        "wavelength safety floor. Increase N or use N=0 for "
                        "automatic material-wavelength meshing. Production "
                        "results still require base/fine mesh convergence."
                    )
        if local_wavelength is not None:
            if sagitta > FACET_ADVISORY_SAGITTA_FRACTION * local_wavelength:
                (notices if notices is not None else materials).inform_once(
                    f"Faceting advisory: segment '{name}' is drawn as a curve of straight "
                    f"primitives whose implied sagitta reaches {sagitta:.3g} m, more than "
                    "1/200 of the medium wavelength at a solved frequency. Mesh refinement "
                    "and certification keep the drawn vertices and do not measure this "
                    "geometric error; draw more primitives if the true surface is curved."
                )

        for (p0, p1, prim_len), count in zip(ends, counts):
            primitive_id = ''
            if hp:
                reference_n = _parse_int(base_segment_n[seg_idx], 0) if refinement_factor > 1 else n_prop
                reference_count = _panel_count_from_n(reference_n, prim_len, wavelength)
                primitive_id, pts = panel_parameters(geometry_snapshot, seg_idx, p0, p1,
                    reference_count, reference_n > 0, refinement_factor, protected,
                    max_panels=panel_limit - len(panels))
                count = len(pts) - 1
            else:
                if len(panels) + count > panel_limit:
                    raise ValueError('Discretization exceeds the configured panel limit.')
                if count > 1 and prim_len / count <= NODE_WELD_TOLERANCE_M:
                    raise ValueError(
                        f"Segment '{name}' N={n_prop} would split a {prim_len:.3g} m primitive "
                        f"into panels no longer than the {NODE_WELD_TOLERANCE_M:g} m mesh node "
                        "tolerance. Reduce N."
                    )
                pts = _discretize_primitive(p0, p1, count)
            if junctions:
                pts = _grade_toward_junctions(pts, junctions, wavelength)
                count = len(pts) - 1
            if len(panels) + count > panel_limit:
                raise ValueError('Discretization exceeds the configured panel limit.')

            for i in range(count):
                q0 = pts[i]
                q1 = pts[i + 1]
                vec = q1 - q0
                length = float(np.linalg.norm(vec))
                if length <= EPS:
                    continue
                tangent = vec / length


                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                center = 0.5 * (q0 + q1)
                panels.append(
                    Panel(
                        name=name,
                        seg_type=seg_type,
                        ibc_flag=ibc_flag,
                        pos_mat=pos_mat,
                        neg_mat=neg_mat,
                        p0=q0,
                        p1=q1,
                        center=center,
                        tangent=tangent,
                        normal=normal,
                        length=length,
                        arc_s_center=0.5,
                        primitive_key=primitive_id,
                    )
                )


        seg_panels = panels[seg_start_idx:]
        if seg_panels:
            total_len = sum(p.length for p in seg_panels)
            if total_len > EPS:
                cum = 0.0
                for p in seg_panels:
                    p.arc_s_center = (cum + 0.5 * p.length) / total_len
                    cum += p.length
            else:
                for p in seg_panels:
                    p.arc_s_center = 0.5


            if seg_type in (2, 3):
                for p in seg_panels:
                    p.arc_s_center = 1.0 - p.arc_s_center

    if not panels:
        raise ValueError("Geometry does not contain any valid discretized panels.")
    max_allowed = max(1, int(max_panels))
    if len(panels) > max_allowed:
        raise ValueError(
            f"Discretization produced {len(panels)} panels; limit is {max_allowed}. "
            "Reduce n/frequency range or increase max_panels."
        )
    return panels

def _linear_node_snap_key(xy: 'np.ndarray', tol: 'float' = 1.0e-9) -> 'Tuple[int, int]':
    scale = 1.0 / max(float(tol), EPS)
    return (int(round(float(xy[0]) * scale)), int(round(float(xy[1]) * scale)))


# Distinct vertices closer than this are one mesh node. Validation calls such
# endpoints connected (its crack scan and chain-continuity test use the same
# distance), so the mesher must never split them.
NODE_WELD_TOLERANCE_M = 1.0e-9


class _NodeWelder:
    """Tolerance-based vertex identity for the mesh builders.

    ``key(xy)`` returns the snap key of the node a point belongs to. A point
    joins the first registered node that shares its rounded snap key (the
    historical grid identity, so meshes without near-coincident vertices are
    numbered exactly as before) or that lies within ``tol`` of it (Euclidean),
    and otherwise opens a new node keyed by its own snap key. A fixed rounding
    grid alone splits two copies of one vertex that are 1e-12 m apart but
    straddle a grid line, meshing a closed contour as an open one.
    """

    def __init__(self, tol: 'float' = NODE_WELD_TOLERANCE_M) -> 'None':
        self.tol = float(tol)
        self._cell = 2.0 * max(self.tol, EPS)
        self._alias: 'Dict[Tuple[int, int], Tuple[int, int]]' = {}
        self._grid: 'Dict[Tuple[int, int], List[Tuple[float, float, Tuple[int, int]]]]' = {}

    def _nearest(self, x: 'float', y: 'float') -> 'Optional[Tuple[int, int]]':
        cx, cy = math.floor(x / self._cell), math.floor(y / self._cell)
        best = None
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for px, py, key in self._grid.get((i, j), ()):
                    distance = math.hypot(px - x, py - y)
                    if distance <= self.tol and (best is None or distance < best[0]):
                        best = (distance, key)
        return None if best is None else best[1]

    def lookup(self, xy) -> 'Optional[Tuple[int, int]]':
        """Key of the registered node this point belongs to, without registering it."""
        known = self._alias.get(_linear_node_snap_key(xy, self.tol))
        if known is not None:
            return known
        return self._nearest(float(xy[0]), float(xy[1]))

    def key(self, xy) -> 'Tuple[int, int]':
        raw = _linear_node_snap_key(xy, self.tol)
        known = self._alias.get(raw)
        if known is not None:
            return known
        x, y = float(xy[0]), float(xy[1])
        near = self._nearest(x, y)
        canonical = raw if near is None else near
        self._alias[raw] = canonical
        self._grid.setdefault(
            (math.floor(x / self._cell), math.floor(y / self._cell)), []
        ).append((x, y, canonical))
        return canonical


class _JunctionVertices(set):
    """Welded keys of graded junction vertices, queried with ``at(point)``."""

    def __init__(self, keys=(), welder: 'Optional[_NodeWelder]' = None) -> 'None':
        super().__init__(keys)
        self.welder = welder

    def at(self, point) -> 'bool':
        if not self:
            return False
        if self.welder is None:
            return _linear_node_snap_key(point) in self
        key = self.welder.lookup(point)
        return key is not None and key in self

def _linear_shape_values(xi: 'float') -> 'np.ndarray':
    x = float(xi)
    return np.asarray([1.0 - x, x], dtype=float)

def _build_linear_mesh(
    panels: 'List[Panel]',
    node_snap_tol: 'float' = 1.0e-9,
) -> 'LinearMesh':
    """
    Convert boundary elements into a continuous two-node linear boundary mesh.

    This is the stage-1 data-structure upgrade for the future linear Galerkin path.
    Each panel becomes one linear element, while shared endpoints are merged into
    unique global nodes by snapped coordinates.
    """

    node_index: 'Dict[Tuple[int, int], int]' = {}
    nodes: 'List[LinearNode]' = []
    elements: 'List[LinearElement]' = []
    welder = _NodeWelder(node_snap_tol)

    def get_node_id(xy: 'np.ndarray') -> 'int':
        key = welder.key(xy)
        idx = node_index.get(key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=key))
        return idx

    for pidx, panel in enumerate(panels):
        n0 = get_node_id(panel.p0)
        n1 = get_node_id(panel.p1)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                pos_mat=panel.pos_mat,
                neg_mat=panel.neg_mat,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
                arc_s_center=float(panel.arc_s_center),
                primitive_key=panel.primitive_key,
            )
        )

    if not elements:
        raise ValueError("Linear mesh construction requires at least one element.")
    from ghost_backend.twod.basis import enrich
    return enrich(LinearMesh(nodes=nodes, elements=elements))[0]

def _linear_panel_signature_from_info(
    panel: 'Panel',
    info: 'PanelCoupledInfo',
) -> 'Tuple[Any, ...]':
    """Topology signature used to decide when linear nodes may be shared safely.

    Only the material flags a TYPE uses enter it (TYPE 1/2: none, TYPE 3/4:
    pos_mat, TYPE 5: both): a stray neg_mat on one half of a closed TYPE 3
    contour split its nodes and moved the fields by 13-58 %.
    """

    seg_type = int(panel.seg_type)
    pos_mat = int(panel.pos_mat) if seg_type in (3, 4, 5) else 0
    neg_mat = int(panel.neg_mat) if seg_type == 5 else 0
    return (
        seg_type,
        int(panel.ibc_flag),
        pos_mat,
        neg_mat,
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )

def _closed_conductor_loop_panels(panels, infos, tolerance) -> 'Set[int]':
    """Panels of closed one-sided PEC/impedance loops, ignoring the IBC flag.

    A change of impedance flag along a closed conductor contour is a change of
    boundary law, not a break in the contour. Its density must stay continuous
    there: the combined ``S + gamma*D`` representation applies a hypersingular
    operator to that density, which has no meaning across a split node. Open
    chains and every other signature difference (type, materials, regions,
    boundary kind) keep their separate nodes.
    """
    groups: 'Dict[Tuple[Any, ...], List[int]]' = {}
    for index, (panel, info) in enumerate(zip(panels, infos)):
        one_sided = int(info.minus_region) < 0 or int(info.plus_region) < 0
        if str(info.bc_kind) != 'robin' or not one_sided or int(panel.seg_type) == 1:
            continue
        signature = _linear_panel_signature_from_info(panel, info)
        groups.setdefault((signature[0],) + signature[2:], []).append(index)
    closed: 'Set[int]' = set()
    welder = _NodeWelder(tolerance)
    for indices in groups.values():
        incidence: 'Dict[Tuple[int, int], List[int]]' = {}
        ends: 'Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]' = {}
        for index in indices:
            ends[index] = (welder.key(panels[index].p0), welder.key(panels[index].p1))
            for key in ends[index]:
                incidence.setdefault(key, []).append(index)
        remaining = set(indices)
        while remaining:
            component, stack = set(), [remaining.pop()]
            while stack:
                index = stack.pop()
                if index in component:
                    continue
                component.add(index)
                for key in ends[index]:
                    stack.extend(j for j in incidence[key] if j not in component)
            remaining -= component
            keys = {key for index in component for key in ends[index]}
            if all(len(incidence[key]) == 2 for key in keys):
                closed |= component
    return closed


def _build_linear_mesh_interface_aware(
    panels: 'List[Panel]',
    infos: 'List[PanelCoupledInfo]',
    node_snap_tol: 'float' = 1.0e-9,
    polarization=None,
) -> 'Tuple[LinearMesh, Dict[str, int]]':
    """
    Build a linear boundary mesh that only shares nodes across the *same* interface signature.

    This hardens the linear/Galerkin path for ordinary corners where distinct interface types
    touch at the same geometric coordinate. Those cases should not be forced to share a single
    nodal DOF, because that incorrectly imposes trace continuity across different interfaces.

    True branching nodes where more than two elements of the same interface signature meet are
    still reported separately by `_linear_coupled_node_report` for diagnostics.
    solver in production runs.
    """

    if len(panels) != len(infos):
        raise ValueError("Interface-aware linear mesh requires matching panels and panel infos.")

    continuous_sheet = str(polarization).upper() == 'TE'
    if continuous_sheet:
        panels = _orient_te_sheet_panels(panels, infos, node_snap_tol)

    node_index: 'Dict[Tuple[Tuple[int, int], Tuple[Any, ...]], int]' = {}
    nodes: 'List[LinearNode]' = []
    elements: 'List[LinearElement]' = []
    geometric_keys: 'Set[Tuple[int, int]]' = set()
    welder = _NodeWelder(node_snap_tol)

    def get_node_id(xy: 'np.ndarray', signature: 'Tuple[Any, ...]') -> 'int':
        # Nodes at one welded vertex share its key, including split interface
        # nodes, so every consumer that groups nodes by ``key`` sees one vertex.
        geom_key = welder.key(xy)
        geometric_keys.add(geom_key)
        full_key = (geom_key, signature)
        idx = node_index.get(full_key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[full_key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=geom_key))
        return idx

    closed_conductors = _closed_conductor_loop_panels(panels, infos, node_snap_tol)
    for pidx, (panel, info) in enumerate(zip(panels, infos)):
        sig = _linear_panel_signature_from_info(panel, info)
        if pidx in closed_conductors:
            # One continuous density around a closed PEC/impedance loop.
            sig = (sig[0], 0) + sig[2:]
        if continuous_sheet and int(panel.seg_type) == 1 and info.bc_kind != 'thin_layer':
            # Neither an impedance flag nor a name-derived virtual region is
            # a physical break in a connected TE sheet trace. Real material
            # regions and all non-sheet interface signatures stay distinct.
            regions = tuple(VIRTUAL_SHEET_REGION_START if rid >= VIRTUAL_SHEET_REGION_START
                            else rid for rid in sig[4:6])
            sig = (sig[0], 0) + sig[2:4] + regions + sig[6:]
        n0 = get_node_id(panel.p0, sig)
        n1 = get_node_id(panel.p1, sig)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                pos_mat=panel.pos_mat,
                neg_mat=panel.neg_mat,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
                arc_s_center=float(panel.arc_s_center),
                primitive_key=panel.primitive_key,
            )
        )

    if not elements:
        raise ValueError("Interface-aware linear mesh construction requires at least one element.")

    mesh = LinearMesh(nodes=nodes, elements=elements)
    geometric_count = int(len(geometric_keys))
    total_nodes = int(len(nodes))
    split_nodes = max(0, total_nodes - geometric_count)


    geo_key_counts: 'Dict[Tuple[int, int], int]' = {}
    for (gk, _sig), _nid in node_index.items():
        geo_key_counts[gk] = geo_key_counts.get(gk, 0) + 1
    multi_sig = sum(1 for c in geo_key_counts.values() if c > 1)

    stats = {
        "linear_geometric_node_count": geometric_count,
        "linear_interface_split_nodes": split_nodes,
        "shared_node_count": geometric_count,
        "split_node_count": split_nodes,
        "split_boundary_primitive_count": int(len(elements)),
        "multi_signature_node_count": multi_sig,
    }
    from ghost_backend.twod.basis import enrich
    return enrich(mesh, stats)


def _orient_te_sheet_panels(panels, infos, tolerance):
    """Orient each nonbranching sheet before sharing its endpoint DOFs.

    Keep element order/material records and enrich only after orientation, so
    p2/p3 basis nodes and taper coefficients stay attached to their elements.
    """
    from dataclasses import replace
    incidence = {}
    welder = _NodeWelder(tolerance)
    ends = {}
    for index, (panel, info) in enumerate(zip(panels, infos)):
        if int(panel.seg_type) != 1 or info.bc_kind == 'thin_layer':
            continue
        ends[index] = (welder.key(panel.p0), welder.key(panel.p1))
        for key in ends[index]:
            incidence.setdefault(key, []).append(index)
    if any(len(indices) > 2 for indices in incidence.values()):
        raise ValueError('TE sheet branching junctions require an unsupported junction condition.')
    remaining = set(index for indices in incidence.values() for index in indices)
    result = list(panels)
    while remaining:
        first = min(remaining)
        # Find the component, then start at a free end if it has one.
        component, stack = set(), [first]
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            for key in ends[index]:
                stack.extend(j for j in incidence[key] if j not in component)
        start = ends[first][0]
        for index in sorted(component):
            for key in ends[index]:
                if len(incidence[key]) == 1:
                    start = key
                    break
            else:
                continue
            break
        while True:
            choices = [index for index in incidence[start] if index in remaining]
            if not choices:
                break
            index = min(choices)
            panel = panels[index]
            head, tail = ends[index]
            if head != start:
                panel = replace(panel, p0=panel.p1.copy(), p1=panel.p0.copy(),
                                tangent=-panel.tangent, normal=-panel.normal)
                head, tail = tail, head
            result[index] = panel
            remaining.remove(index)
            start = tail
    return result


def _medium_eta(eps: 'complex', mu: 'complex') -> 'complex':
    eps, mu = _validate_passive_medium(eps, mu, "Medium")


    n = _causal_medium_index(eps, mu)
    return ETA0 * mu / n

def _medium_n(eps: 'complex', mu: 'complex') -> 'complex':
    eps, mu = _validate_passive_medium(eps, mu, "Medium")
    return cmath.sqrt(eps * mu)

def _safe_complex_div(num: 'complex', den: 'complex', fallback: 'complex') -> 'complex':
    if abs(den) <= EPS:
        return fallback
    return num / den


def _region_medium(materials: 'MaterialLibrary', region_flag: 'int', freq_ghz: 'float') -> 'Tuple[complex, complex]':


    if region_flag <= 0 or region_flag >= VIRTUAL_SHEET_REGION_START:
        return 1.0 + 0.0j, 1.0 + 0.0j
    return materials.get_medium(region_flag, freq_ghz)

def _causal_medium_index(eps: 'complex', mu: 'complex') -> 'complex':
    """
    Choose refractive-index branch consistent with passive media in e^{+jwt}.

    Passive attenuation requires Im(n) <= 0.  On the exactly lossless branch,
    select the sign that gives non-negative real wave impedance mu/n.  This is
    the limiting-absorption continuation and, in particular, makes a passive
    double-negative medium use Re(n) < 0 instead of discontinuously jumping to
    the positive-index root.
    """

    n = _medium_n(eps, mu)
    branch_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(n))
    if n.imag > branch_tol:
        n = -n
    elif abs(n.imag) <= branch_tol:
        eta_try = complex(mu) / n
        if eta_try.real < 0.0:
            n = -n
    return n

def _mesh_wavelength_for_snapshot(
    geometry_snapshot: 'Dict[str, Any]',
    materials: 'MaterialLibrary',
    frequency_ghz: 'float',
) -> 'Tuple[float, float, List[int]]':
    """
    Return the shortest wavelength that the boundary mesh must resolve.

    Air is always present as the exterior reference, so the controlling
    refractive-index magnitude is at least one. Only dielectric flags actually
    referenced by TYPE 3/4/5 boundaries participate; unused library rows do not
    over-refine a model. For lossy media ``|n|`` is a conservative spatial scale
    covering both phase variation and attenuation.
    """

    freq = float(frequency_ghz)
    if not math.isfinite(freq) or freq <= 0.0:
        raise ValueError(
            f"Mesh wavelength requires a positive finite frequency; got {frequency_ghz!r}."
        )

    material_flags: 'Set[int]' = set()
    for seg in _snapshot_segments(geometry_snapshot):
        props = list(seg.get("properties", []) or [])
        if len(props) < 5:
            props.extend([""] * (5 - len(props)))
        seg_type = _parse_flag(
            props[0] if str(props[0]).strip() else seg.get("seg_type", 0)
        )
        pos_mat = _parse_flag(props[3])
        neg_mat = _parse_flag(props[4])
        sheet_model = materials.impedance_models.get(_parse_flag(props[2]))
        if seg_type == 1 and isinstance(sheet_model, ThinLayerDefinition):
            material_flags.add(sheet_model.dielectric_flag)
        if seg_type in (3, 4) and pos_mat > 0:
            material_flags.add(pos_mat)
        elif seg_type == 5:
            if pos_mat > 0:
                material_flags.add(pos_mat)
            if neg_mat > 0:
                material_flags.add(neg_mat)

    max_index = 1.0
    for flag in sorted(material_flags):
        eps, mu = materials.get_medium(flag, freq)
        index_mag = float(abs(_causal_medium_index(eps, mu)))
        if not math.isfinite(index_mag) or index_mag <= 0.0:
            raise ValueError(
                f"Dielectric flag {flag} produced invalid refractive-index magnitude "
                f"{index_mag!r} at {freq:g} GHz."
            )
        max_index = max(max_index, index_mag)

    free_space_wavelength = C0 / (freq * 1.0e9)
    return (
        float(free_space_wavelength / max_index),
        float(max_index),
        sorted(material_flags),
    )

def _conservative_mesh_wavelength_for_frequencies(
    geometry_snapshot: 'Dict[str, Any]',
    materials: 'MaterialLibrary',
    frequencies_ghz,
) -> 'Tuple[float, float, List[int]]':
    """Shortest referenced-material wavelength over every supplied frequency."""

    values = [
        _mesh_wavelength_for_snapshot(geometry_snapshot, materials, float(freq))
        for freq in frequencies_ghz
    ]
    if not values:
        raise ValueError("At least one mesh-control frequency is required.")
    return (
        min(value[0] for value in values),
        max(value[1] for value in values),
        sorted({flag for value in values for flag in value[2]}),
    )

def _medium_wavenumber(
    k0: 'float',
    eps: 'complex',
    mu: 'complex',
) -> 'complex':
    """Complex medium wavenumber used directly inside integral kernels."""

    return complex(k0) * _causal_medium_index(eps, mu)

def _impedance_to_admittance(z_value: 'complex') -> 'complex':
    z_eval = _ensure_finite_complex(z_value, "Surface impedance")
    if abs(z_eval) <= EPS:
        return 0.0 + 0.0j
    return 1.0 / z_eval

def _surface_robin_alpha(
    pol: 'str',
    eps_medium: 'complex',
    mu_medium: 'complex',
    k_medium: 'complex',
    z_surface: 'complex',
) -> 'complex':
    """
    Return the scalar Robin coefficient alpha for q + alpha*u = 0.

    Physical SIBC boundary conditions for 2D scalar wave equation:

    TM (E_z, Dirichlet-like for PEC):
      E_z + Z_s * H_phi = 0
      -> du/dn + j*k*eta/Z_s * u = 0
      -> alpha = j * k * eta / Z_s
      Limits: Z_s->0 -> alpha->inf (u=0, PEC TM)
              Z_s->inf -> alpha->0 (q=0, PMC TM)

    TE (H_z, Neumann-like for PEC):
      -> du/dn + j*k*Z_s/eta * u = 0
      -> alpha = +j * k * Z_s / eta
      Limits: Z_s->0 -> alpha->0 (q=0, PEC TE)
              Z_s->inf -> alpha->inf (u=0, PMC TE)
      Sign pinned by the flat-interface reflection coefficient
      R_H = (eta - Z_s) / (eta + Z_s) (matched absorber Z_s = eta must
      absorb, not amplify) and validated against the impedance-cylinder
      Mie series; both alphas flip together with the normal, so
      alpha_TM * alpha_TE = -k^2 under any single normal convention.
    """

    if abs(z_surface) <= EPS:
        return 0.0 + 0.0j
    eta_medium = _medium_eta(eps_medium, mu_medium)
    if pol == "TM":
        return 1j * complex(k_medium) * _safe_complex_div(eta_medium, z_surface, 0.0 + 0.0j)
    return 1j * complex(k_medium) * _safe_complex_div(z_surface, eta_medium, 0.0 + 0.0j)

def _q_plus_beta(
    pol: 'str',
    eps_minus: 'complex',
    mu_minus: 'complex',
    eps_plus: 'complex',
    mu_plus: 'complex',
) -> 'complex':
    """
    Scaling between minus-side and plus-side raw normal derivatives across
    a transmission interface:  q_plus = beta * q_minus.

    For the 2D scalar Helmholtz reduction of Maxwell's equations under the
    e^{+jwt} convention:
      - TM (u = E_z axial):  the continuous flux quantity is (1/mu) du/dn,
        so beta = mu_plus / mu_minus.
      - TE (u = H_z axial):  the continuous flux quantity is (1/eps) du/dn,
        so beta = eps_plus / eps_minus.

    This matches the flux-scaling factor used in `_solve_dielectric_indirect`
    (see "factor = mu_ext/mu_int" for TM there) and the Mie reference in
    `ghost_backend.validation.cylinder`.
    """

    if pol == "TE":
        return _safe_complex_div(eps_plus, eps_minus, 1.0 + 0.0j)
    return _safe_complex_div(mu_plus, mu_minus, 1.0 + 0.0j)


def _build_coupled_panel_info(
    panels: 'List[Panel]',
    materials: 'MaterialLibrary',
    freq_ghz: 'float',
    pol: 'str',
    k0: 'float',
) -> 'List[PanelCoupledInfo]':
    """
    Translate geometry TYPE/IBC/IPN flags into coupled interface algebra per panel.

    Project convention:
    - the drawn panel normal points toward the pos_mat side,
    - TYPE 3: plus/pos_mat = dielectric, minus = air,
    - TYPE 5: plus/pos_mat, minus/neg_mat,
    - TYPE 4: plus/pos_mat = dielectric, minus = PEC/IBC side.

    The coupled assembly is allowed to use whichever side is the valid non-PEC side,
    so TYPE 4 remains solvable even though the PEC side is the minus side.
    """

    infos: 'List[PanelCoupledInfo]' = []
    sheet_region_by_name: 'Dict[str, int]' = {}
    next_sheet_region = VIRTUAL_SHEET_REGION_START


    medium_state = {}  # type: Dict[int, Tuple[complex, complex, complex]]

    def region_state(region):
        # type: (int) -> Tuple[complex, complex, complex]
        key = int(region)
        cached = medium_state.get(key)
        if cached is not None:
            return cached
        eps_value, mu_value = _region_medium(materials, key, freq_ghz)
        k_value = _medium_wavenumber(k0, eps_value, mu_value)
        if (
            abs(k_value.imag) > 1e-10
            and _complex_hankel_backend_name() == "unavailable"
        ):
            raise RuntimeError(
                "Lossy dielectric media require SciPy or mpmath for trustworthy "
                "complex-Hankel evaluation. Install one of those backends before "
                "running production dielectric solves."
            )
        cached = (eps_value, mu_value, k_value)
        medium_state[key] = cached
        return cached


    impedance_cache = {}  # type: Dict[int, complex]

    def panel_impedance(panel):
        # type: (Panel) -> complex
        flag = int(panel.ibc_flag)
        if flag <= 0:
            return 0.0 + 0.0j
        if isinstance(materials.impedance_models.get(flag), ThinLayerDefinition):
            if int(panel.seg_type) != 1:
                raise ValueError("Thin dielectric layers can be assigned only to a free TYPE 1 sheet.")
            return 0.0j
        if materials.is_tapered_impedance(flag):
            return materials.get_impedance(
                flag, freq_ghz, arc_s=float(panel.arc_s_center)
            )
        if flag not in impedance_cache:
            impedance_cache[flag] = materials.get_impedance(
                flag, freq_ghz, arc_s=float(panel.arc_s_center)
            )
        return impedance_cache[flag]

    for panel in panels:
        seg_type = panel.seg_type
        if seg_type == 3:
            if panel.pos_mat <= 0:
                raise ValueError(f"TYPE 3 panel '{panel.name}' requires pos_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = 0
            bc_kind = "transmission"
            plus_has_incident = False
            minus_has_incident = True
        elif seg_type == 5:
            if panel.pos_mat <= 0 or panel.neg_mat <= 0:
                raise ValueError(f"TYPE 5 panel '{panel.name}' requires pos_mat > 0 and neg_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = panel.neg_mat
            bc_kind = "transmission"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 4:
            if panel.pos_mat <= 0:
                raise ValueError(f"TYPE 4 panel '{panel.name}' requires pos_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = -1
            bc_kind = "robin"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 2:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False
        elif seg_type == 1:
            if panel.ibc_flag <= 0:
                raise ValueError(
                    f"TYPE 1 panel '{panel.name}' requires IBC > 0 in coupled dielectric mode."
                )
            sheet_name = panel.name.strip() or "__type1_sheet__"
            sheet_region = sheet_region_by_name.get(sheet_name)
            if sheet_region is None:
                sheet_region = next_sheet_region
                sheet_region_by_name[sheet_name] = sheet_region
                next_sheet_region += 1
            minus_region = 0
            plus_region = sheet_region
            bc_kind = "transmission"
            minus_has_incident = True
            plus_has_incident = False
        else:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False

        eps_minus, mu_minus, k_minus = region_state(minus_region)
        eps_plus, mu_plus, k_plus = region_state(plus_region)
        z_card = panel_impedance(panel)
        thin_layer = isinstance(materials.impedance_models.get(panel.ibc_flag), ThinLayerDefinition)
        if thin_layer:
            bc_kind = "thin_layer"
        if bc_kind == "transmission":
            if seg_type == 1:
                if abs(z_card) <= EPS:
                    raise ValueError(
                        f"TYPE 1 panel '{panel.name}' has zero impedance; provide non-zero IBC for sheet mode."
                    )
                q_plus_beta = -1.0 + 0.0j
                q_plus_gamma = _impedance_to_admittance(z_card)
            else:
                q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
                q_plus_gamma = _impedance_to_admittance(z_card)
        else:
            q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
            q_plus_gamma = 0.0 + 0.0j

        infos.append(
            PanelCoupledInfo(
                seg_type=seg_type,
                plus_region=plus_region,
                minus_region=minus_region,
                plus_has_incident=plus_has_incident,
                minus_has_incident=minus_has_incident,
                eps_plus=eps_plus,
                mu_plus=mu_plus,
                eps_minus=eps_minus,
                mu_minus=mu_minus,
                k_plus=k_plus,
                k_minus=k_minus,
                q_plus_beta=q_plus_beta,
                q_plus_gamma=q_plus_gamma,
                bc_kind=bc_kind,


                robin_impedance=(
                    z_card if bc_kind == "robin"
                    else (z_card if seg_type == 1 else 0.0 + 0.0j)
                ),
            )
        )

    return infos
