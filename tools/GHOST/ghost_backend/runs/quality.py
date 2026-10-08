import copy
import math
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-12

# The peak-normalised complex-field change is the primary convergence metric:
# it is defined everywhere, including at physical nulls.  dB and phase changes
# are pointwise-relative quantities that are undefined at a null, where a
# converged pattern still moves by several dB and many degrees for a complex
# change of 1e-4 of the peak.  They are therefore evaluated against a level
# floor relative to the per-frequency peak amplitude:
#   * dB deltas compare 20*log10(|A| + db_floor_relative*peak), which equals
#     the exact dB change well above the floor and stays bounded by
#     20*log10(1 + |dA|/(db_floor_relative*peak)) at and below it;
#   * phase deltas are compared only where both meshes are above
#     phase_floor_relative*peak.
# 3e-2 (-30 dB) keeps both checks meaningful over the pattern's upper 30 dB
# while a result with a 0.1% complex change can no longer fail them at a
# -75 dB null (it previously failed the tight 0.75 dB / 5 deg target).
MESH_CONVERGENCE_LEVEL_FLOOR = 3.0e-2

PRODUCTION_MESH_CONVERGENCE_DEFAULTS = {
    "fine_factor": 1.5,
    "rms_limit_db": 1.0,
    "max_abs_limit_db": 3.0,
    "complex_rms_limit": 0.02,
    "complex_max_limit": 0.05,
    "phase_rms_limit_deg": 5.0,
    "phase_max_limit_deg": 15.0,
    "phase_floor_relative": MESH_CONVERGENCE_LEVEL_FLOOR,
    "db_floor_relative": MESH_CONVERGENCE_LEVEL_FLOOR,
}


def accuracy_target_policy(target="standard"):
    """Concrete mesh tolerances; no claim about model-approximation error.

    Both targets share the dB/phase level floors of
    ``PRODUCTION_MESH_CONVERGENCE_DEFAULTS``; the complex-field limits are the
    primary criterion (tight: 1% maximum change relative to the peak).
    """
    if target == "standard":
        return validate_mesh_convergence_policy()
    if target == "tight":
        return validate_mesh_convergence_policy({
            "rms_limit_db": 0.25, "max_abs_limit_db": 0.75,
            "complex_rms_limit": 0.005, "complex_max_limit": 0.01,
            "phase_rms_limit_deg": 2.0, "phase_max_limit_deg": 5.0,
        })
    raise ValueError("Accuracy target must be standard or tight.")


def solver_report_text(metadata):
    """Readable evidence summary without conflating four error mechanisms."""
    lines = ["Numerical evidence (separate from material/Assembly approximation error)"]
    lines.append(str(metadata.get("formulation", "")))
    mesh = metadata.get("mesh_convergence", {})
    lines.append("Mesh refinement comparison: " + ("passed" if metadata.get("mesh_convergence_certified") else "not certified"))
    def measurements(value, key):
        if not isinstance(value, dict):
            return []
        found = [float(value[key])] if key in value and isinstance(value[key], (float, int)) and math.isfinite(value[key]) else []
        for name, child in value.items():
            if name not in {"limits", "thresholds", "policy"} and isinstance(child, dict):
                found.extend(measurements(child, key))
        return found
    for title, key, scale, unit in (("Largest normalized complex-field mesh change", "complex_max_normalized", 100., "%"),
                                   ("Worst RMS mesh phase change", "phase_rms_deg", 1., " deg")):
        values = measurements(mesh, key)
        if values:
            lines.append(f"{title}: {max(values)*scale:.4g}{unit}")
    for title, key in (("Maximum linear residual", "residual_norm_max"),
                       ("Maximum backward error", "backward_error_max"),
                       ("Condition estimate", "condition_est_max")):
        value = metadata.get(key)
        if isinstance(value, (float, int)) and math.isfinite(value):
            lines.append(f"{title}: {value:.4g}")
    frequencies = metadata.get("per_frequency", [])
    modal = [row["mode_converged"] for row in frequencies if "mode_converged" in row]
    if modal:
        lines.append(f"Modal convergence: {sum(bool(value) for value in modal)}/{len(modal)} frequencies passed")
    if any(row.get("near_quadrature") for row in frequencies):
        lines.append("Near-quadrature evidence is included in the per-frequency details.")
    quadrature = metadata.get('quadrature_comparison')
    if quadrature:
        worst = max((row.get('complex_max_normalized', 0.) for row in quadrature.get('samples', [])), default=0.)
        lines.append('Same-mesh integration comparison: {} (largest normalized complex-field change {:.4g}%).'.format(
            'passed' if quadrature.get('passed') else 'failed', 100.*worst))
    if modal:
        lines.append('Modal convergence covers the requested angles and polarizations; angular interpolation is not certified.')
    layers = metadata.get("thin_layer", [])
    if (any(layers.values()) if isinstance(layers, dict) else bool(layers)):
        lines.append("Thin layer: first-order thickness approximation. Mesh certification does not certify its difference from bulk material.")
    lines.append("Linear backend: "+str(metadata.get("linear_backend", "see per-frequency details")))
    selection = metadata.get('backend_selection')
    if selection:
        lines.append('Automatic backend: ' + str(selection['selected']) + '. ' + str(selection['reason']))
    if 'mesh_strategy_used' in metadata:
        lines.append('Mesh sizing: ' + str(metadata['mesh_strategy_used']))
    if metadata.get('polynomial_degree_min') != metadata.get('polynomial_degree_max'):
        lines.append('Boundary polynomial degrees: {}-{} (see frequency metadata).'.format(
            metadata['polynomial_degree_min'], metadata['polynomial_degree_max']))
    elif metadata.get('polynomial_degree', 1) > 1:
        lines.append('Boundary polynomial degree: ' + str(metadata['polynomial_degree']))
    adaptation = metadata.get('adaptive_mesh', {})
    if adaptation.get('steps'):
        lines.append('Adaptive accuracy checks: ' + str(len(adaptation['steps'])))
    if adaptation.get('fallback'):
        lines.append('Reference mesh used after adaptive rejection: ' + str(adaptation.get('reason', '')))
    if metadata.get('local_mesh_fallback'):
        lines.append('Frequencies that failed the local mesh comparison passed after retrying global sizing.')
    checkpoints = metadata.get('frequency_checkpoints')
    if checkpoints:
        lines.append('Completed frequencies: {}; saved checkpoints: {}; reused: {}.'.format(
            checkpoints['completed'], checkpoints.get('persisted', checkpoints['completed']), checkpoints['reused']))
        lines.append('Checkpoint directory: ' + str(checkpoints['directory']))
        lines.extend(checkpoints.get('write_warnings', []))
    execution = metadata.get('frequency_execution', {})
    if execution.get('persistent_solve_cache') is False:
        lines.append('Fresh calculation: {} frequencies computed; previous runs were not reused.'.format(
            execution.get('computed_frequencies', 0)))
    recovery = metadata.get('run_recovery')
    if recovery:
        lines.append('Run recovery: {} of {} frequencies saved in {}.'.format(
            recovery['completed'], recovery['requested'], recovery['directory']))
    if metadata.get("survey_mode"):
        lines.append("Survey: the mesh-refinement comparison was not run.")
    profile = metadata.get("runtime_profile", {})
    if profile:
        elapsed=profile.get('wall_seconds', metadata.get('execution_wall_seconds', 0.))
        lines.append(f"Elapsed: {elapsed:.3f} s")
        for name, value in sorted(profile.get("stage_seconds", {}).items()):
            lines.append(f"  {name.replace('_', ' ')}: {value:.3f} s")
        peak = profile.get("sampled_peak_process_rss_bytes")
        lines.append("Sampled peak process RAM: " + (
            f"{peak / 1024**3:.3f} GiB" if peak is not None else "unavailable"))
        lines.append("Stages may overlap. RAM includes other work in this process.")
        tree_peak = profile.get('sampled_peak_process_tree_rss_bytes')
        if tree_peak is not None:
            lines.append(f"Sampled RAM including worker processes: {tree_peak / 1024**3:.3f} GiB "
                         "(shared pages may be counted more than once).")
        private_peak = profile.get('sampled_peak_process_tree_private_bytes')
        if private_peak is not None:
            lines.append(f"Sampled private committed memory including workers: {private_peak / 1024**3:.3f} GiB.")
    return "\n".join(lines)


def _sample_key(row: 'Dict[str, Any]') -> 'Tuple[float, float, float]':
    return (
        round(float(row.get("frequency_ghz", 0.0)), 9),
        round(float(row.get("theta_inc_deg", 0.0)), 9),
        round(float(row.get("theta_scat_deg", 0.0)), 9),
    )


def _ensure_properties_len(props: 'List[Any]', n: 'int' = 6) -> 'List[str]':
    out = [str(p) for p in list(props or [])]
    if len(out) < n:
        out.extend([""] * (n - len(out)))
    return out


def scale_snapshot_panel_density(snapshot: 'Dict[str, Any]', fine_factor: 'float') -> 'Dict[str, Any]':
    """
    Return a deep-copied geometry snapshot with increased panel density.

    This scales the per-segment N/discretization property in `properties[1]` while
    preserving all geometric/material flags.

    When ``properties[1]`` is empty or zero (auto-density mode), the solver uses
    DEFAULT_PANELS_PER_WAVELENGTH.  The fine snapshot switches to an explicit
    panels-per-wavelength value (negative N) scaled by ``fine_factor`` so the
    convergence comparison is meaningful.
    """

    try:
        from ghost_backend.twod.solver import DEFAULT_PANELS_PER_WAVELENGTH
    except ImportError:
        DEFAULT_PANELS_PER_WAVELENGTH = 20

    factor = float(fine_factor)
    if factor <= 1.0:
        raise ValueError("fine_factor must be > 1.0.")

    out = copy.deepcopy(snapshot)
    segments = list(out.get("segments", []) or [])
    for seg in segments:
        props = _ensure_properties_len(seg.get("properties", []), 6)
        raw = str(props[1]).strip()
        try:
            base_n = int(round(float(raw or 0)))
        except Exception:
            base_n = 0

        if base_n == 0:


            fine_ppw = max(2, int(math.ceil(DEFAULT_PANELS_PER_WAVELENGTH * factor)))
            props[1] = str(-fine_ppw)
        elif base_n > 0:

            fine_n = max(base_n + 1, int(math.ceil(base_n * factor)))
            props[1] = str(fine_n)
        else:

            base_ppw = abs(base_n)
            fine_ppw = max(base_ppw + 1, int(math.ceil(base_ppw * factor)))
            props[1] = str(-fine_ppw)

        seg["properties"] = props
        seg["seg_type"] = props[0] if props[0] else seg.get("seg_type")
    out["segments"] = segments
    return out


def validate_mesh_convergence_policy(
    policy: 'Dict[str, Any]' = None,
) -> 'Dict[str, float]':
    """Return a complete, finite, fail-closed production mesh policy."""

    supplied = dict(policy or {})
    unknown = sorted(
        set(supplied) - set(PRODUCTION_MESH_CONVERGENCE_DEFAULTS)
    )
    if unknown:
        raise ValueError(
            "Unknown mesh convergence policy field(s): "
            + ", ".join(unknown)
        )
    complete = dict(PRODUCTION_MESH_CONVERGENCE_DEFAULTS)
    complete.update(supplied)
    validated = {}
    for name, raw_value in complete.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a finite number."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        validated[name] = value

    if validated["fine_factor"] <= 1.0:
        raise ValueError("fine_factor must be > 1.0.")
    for name in (
        "rms_limit_db",
        "max_abs_limit_db",
        "complex_rms_limit",
        "complex_max_limit",
        "phase_rms_limit_deg",
        "phase_max_limit_deg",
    ):
        if validated[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative.")
    for name in ("phase_floor_relative", "db_floor_relative"):
        if not 0.0 <= validated[name] < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    return validated


def evaluate_mesh_convergence(
    base_result: 'Dict[str, Any]',
    fine_result: 'Dict[str, Any]',
    rms_limit_db: 'float',
    max_abs_limit_db: 'float',
    complex_rms_limit: 'float' = 0.02,
    complex_max_limit: 'float' = 0.05,
    phase_rms_limit_deg: 'float' = 5.0,
    phase_max_limit_deg: 'float' = 15.0,
    phase_floor_relative: 'float' = MESH_CONVERGENCE_LEVEL_FLOOR,
    db_floor_relative: 'float' = MESH_CONVERGENCE_LEVEL_FLOOR,
) -> 'Dict[str, Any]':
    """
    Compare two solve results point-by-point in magnitude and complex field.

    Matching is done on (frequency_ghz, theta_inc_deg, theta_scat_deg).

    The primary metric is the complex-field change normalized by the peak
    amplitude of each frequency (over both meshes).  It remains meaningful at
    physical nulls, where a pointwise relative error and phase are undefined.

    dB and phase changes are secondary checks referred to the same
    per-frequency peak ``P``:

    * dB deltas compare ``20*log10(|A| + db_floor_relative*P)`` of the two
      meshes.  Well above the floor this is the exact dB change of the
      pattern; at and below it the delta is bounded by
      ``20*log10(1 + |dA|/(db_floor_relative*P))``, so the unbounded dB
      swing at a deep null of a converged pattern cannot fail the check,
      while a pattern that changes shape above the floor still does.
    * phase deltas are compared only where both meshes are above
      ``phase_floor_relative*P``.

    Both floors default to ``MESH_CONVERGENCE_LEVEL_FLOOR`` (3e-2, -30 dB).
    The unfloored dB deltas (from the samples' ``rcs_db``/``rcs_linear``) are
    reported for information only (``*_unfloored``).
    """

    policy = validate_mesh_convergence_policy({
        "rms_limit_db": rms_limit_db,
        "max_abs_limit_db": max_abs_limit_db,
        "complex_rms_limit": complex_rms_limit,
        "complex_max_limit": complex_max_limit,
        "phase_rms_limit_deg": phase_rms_limit_deg,
        "phase_max_limit_deg": phase_max_limit_deg,
        "phase_floor_relative": phase_floor_relative,
        "db_floor_relative": db_floor_relative,
    })
    phase_floor_fraction = policy["phase_floor_relative"]
    db_floor_fraction = policy["db_floor_relative"]

    base_samples = base_result.get("samples", []) or []
    fine_samples = fine_result.get("samples", []) or []
    if not base_samples or not fine_samples:
        raise ValueError("Both base_result and fine_result must contain samples.")

    from ghost_backend.runs.quality_samples import compact_comparison
    compact = compact_comparison(base_samples, fine_samples)
    if compact is not None:
        base_amp, fine_amp, unfloored_deltas, frequency_groups = compact
        sample_count = len(base_amp)
    else:
        def exact_sample_coordinates(row):
            return (
                float(row.get("frequency_ghz", 0.0)),
                float(row.get("theta_inc_deg", 0.0)),
                float(row.get("theta_scat_deg", 0.0)),
            )

        def group_by_key(samples, label):
            """Group samples that collide only because the tolerant key is rounded."""

            grouped = {}
            exact_coordinates = set()
            for row in samples:
                coordinates = exact_sample_coordinates(row)
                if coordinates in exact_coordinates:
                    raise ValueError(
                        f"Mesh convergence {label} contains duplicate exact sample "
                        f"coordinates {coordinates}."
                    )
                exact_coordinates.add(coordinates)
                grouped.setdefault(_sample_key(row), []).append(row)
            for rows in grouped.values():


                rows.sort(key=exact_sample_coordinates)
            return grouped

        base_by_key = group_by_key(base_samples, "base_result")
        fine_by_key = group_by_key(fine_samples, "fine_result")
        base_keys = set(base_by_key)
        fine_keys = set(fine_by_key)
        all_keys = base_keys | fine_keys
        missing_count = sum(
            max(0, len(base_by_key.get(key, ())) - len(fine_by_key.get(key, ())))
            for key in all_keys
        )
        extra_count = sum(
            max(0, len(fine_by_key.get(key, ())) - len(base_by_key.get(key, ())))
            for key in all_keys
        )
        if missing_count or extra_count:
            raise ValueError(
                "Mesh convergence sample grids differ: "
                f"{missing_count} missing and {extra_count} extra fine-result "
                "sample point(s)."
            )

        matched: 'List[Tuple[Dict[str, Any], Dict[str, Any]]]' = []
        for key in sorted(base_keys):
            matched.extend(zip(base_by_key[key], fine_by_key[key]))
        if not matched:
            raise ValueError("Mesh convergence comparison produced no overlapping samples.")

        def finite_float(row, key):
            if key not in row:
                raise ValueError(
                    "Mesh convergence requires authoritative complex amplitudes; "
                    f"sample {_sample_key(row)} is missing {key!r}."
                )
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Mesh convergence sample {_sample_key(row)} has invalid "
                    f"{key}={row[key]!r}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Mesh convergence sample {_sample_key(row)} has non-finite "
                    f"{key}={row[key]!r}."
                )
            return value

        def sample_db(row):
            if "rcs_db" in row:
                value = float(row["rcs_db"])
            else:
                linear = finite_float(row, "rcs_linear")
                if linear < 0.0:
                    raise ValueError(
                        f"Mesh convergence sample {_sample_key(row)} has negative "
                        f"rcs_linear={linear!r}."
                    )
                value = 10.0 * math.log10(max(linear, EPS))
            if not math.isfinite(value):
                raise ValueError(
                    f"Mesh convergence sample {_sample_key(row)} has non-finite "
                    f"rcs_db={value!r}."
                )
            return value

        base_amp = np.asarray([
            complex(
                finite_float(row, "rcs_amp_real"),
                finite_float(row, "rcs_amp_imag"),
            )
            for row, _other in matched
        ], dtype=np.complex128)
        fine_amp = np.asarray([
            complex(
                finite_float(other, "rcs_amp_real"),
                finite_float(other, "rcs_amp_imag"),
            )
            for _row, other in matched
        ], dtype=np.complex128)
        # Informational only: the display dB of each sample, without a level floor.
        # Reading them also keeps the fail-closed validation of rcs_db/rcs_linear.
        unfloored_deltas = np.asarray([
            sample_db(row) - sample_db(other)
            for row, other in matched
        ], dtype=float)

        frequency_groups = {}
        for index, (row, _other) in enumerate(matched):
            frequency = round(float(row.get("frequency_ghz", 0.0)), 9)
            frequency_groups.setdefault(frequency, []).append(index)
        sample_count = len(matched)
    complex_errors = np.zeros(sample_count, dtype=float)
    deltas = np.zeros(sample_count, dtype=float)
    above_db_floor = np.zeros(sample_count, dtype=bool)
    phase_error_groups = []
    peak_by_frequency = {}
    for frequency, raw_indices in sorted(frequency_groups.items()):
        indices = np.asarray(raw_indices, dtype=int)
        base_abs = np.abs(base_amp[indices])
        fine_abs = np.abs(fine_amp[indices])
        peak = float(max(np.max(base_abs), np.max(fine_abs)))
        peak_by_frequency[str(frequency)] = peak
        if not math.isfinite(peak) or peak <= 0.0:
            # Both meshes are identically zero at this frequency: no change.
            continue
        complex_errors[indices] = (
            np.abs(base_amp[indices] - fine_amp[indices]) / peak
        )
        level_floor = max(db_floor_fraction * peak, np.finfo(float).tiny)
        deltas[indices] = 20.0 * (
            np.log10(base_abs + level_floor) - np.log10(fine_abs + level_floor)
        )
        above_db_floor[indices] = np.maximum(base_abs, fine_abs) >= db_floor_fraction * peak
        phase_floor = phase_floor_fraction * peak
        phase_mask = (base_abs > phase_floor) & (fine_abs > phase_floor)
        if np.any(phase_mask):
            phase_error_groups.append(np.abs(np.degrees(np.angle(
                fine_amp[indices][phase_mask]
                * np.conj(base_amp[indices][phase_mask])
            ))))
    phase_errors_deg = (
        np.concatenate(phase_error_groups)
        if phase_error_groups
        else np.zeros(0, dtype=float)
    )
    peak_amp = float(max(peak_by_frequency.values()))

    abs_deltas = np.abs(deltas)
    rms_db = float(np.sqrt(np.mean(deltas * deltas)))
    max_abs_db = float(np.max(abs_deltas))
    complex_rms = float(np.sqrt(np.mean(complex_errors ** 2)))
    complex_max = float(np.max(complex_errors))
    if phase_errors_deg.size:
        phase_rms_deg = float(np.sqrt(np.mean(phase_errors_deg ** 2)))
        phase_max_deg = float(np.max(phase_errors_deg))
    else:
        phase_rms_deg = 0.0
        phase_max_deg = 0.0

    violations: 'List[str]' = []
    if rms_db > float(rms_limit_db):
        violations.append(f"RMS dB delta {rms_db:.6g} exceeds limit {float(rms_limit_db):.6g}")
    if max_abs_db > float(max_abs_limit_db):
        violations.append(
            f"Max |dB| delta {max_abs_db:.6g} exceeds limit {float(max_abs_limit_db):.6g}"
        )
    if complex_rms > float(complex_rms_limit):
        violations.append(
            f"RMS normalized complex-field delta {complex_rms:.6g} exceeds "
            f"limit {float(complex_rms_limit):.6g}"
        )
    if complex_max > float(complex_max_limit):
        violations.append(
            f"Max normalized complex-field delta {complex_max:.6g} exceeds "
            f"limit {float(complex_max_limit):.6g}"
        )
    if phase_rms_deg > float(phase_rms_limit_deg):
        violations.append(
            f"RMS phase delta {phase_rms_deg:.6g} deg exceeds limit "
            f"{float(phase_rms_limit_deg):.6g} deg"
        )
    if phase_max_deg > float(phase_max_limit_deg):
        violations.append(
            f"Max phase delta {phase_max_deg:.6g} deg exceeds limit "
            f"{float(phase_max_limit_deg):.6g} deg"
        )
    passed = not violations

    return {
        "passed": bool(passed),
        "sample_count": int(len(deltas)),
        "rms_db": rms_db,
        "max_abs_db": max_abs_db,
        "mean_db": float(np.mean(deltas)),
        "median_abs_db": float(np.median(abs_deltas)),
        "db_metric": "20*log10(|A| + db_floor_relative*peak) per frequency",
        "db_sample_count_above_floor": int(np.count_nonzero(above_db_floor)),
        "rms_db_unfloored": float(np.sqrt(np.mean(unfloored_deltas ** 2))),
        "max_abs_db_unfloored": float(np.max(np.abs(unfloored_deltas))),
        "complex_rms_normalized": complex_rms,
        "complex_max_normalized": complex_max,
        "complex_reference_peak_amplitude": peak_amp,
        "complex_reference_peak_amplitude_by_frequency":
            peak_by_frequency,
        "phase_sample_count": int(phase_errors_deg.size),
        "phase_rms_deg": phase_rms_deg,
        "phase_max_deg": phase_max_deg,
        "limits": {
            "rms_db": float(rms_limit_db),
            "max_abs_db": float(max_abs_limit_db),
            "complex_rms_normalized": float(complex_rms_limit),
            "complex_max_normalized": float(complex_max_limit),
            "phase_rms_deg": float(phase_rms_limit_deg),
            "phase_max_deg": float(phase_max_limit_deg),
            "phase_floor_relative": float(phase_floor_fraction),
            "db_floor_relative": float(db_floor_fraction),
        },
        "violations": violations,
        "reason": "; ".join(violations) if violations else "mesh convergence passed",
    }
