"""Shared Qt-free form values and loaded Assembly recipe records."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any, Mapping


LINE_RESPONSE_DEFAULTS = {
    "psi_tm_deg": 166.9,
    "psi_te_deg": -9.2,
    "grazing_taper_deg": 10.0,
}


def normalize_response_settings(value: Any, kind: str) -> dict[str, dict[str, Any]]:
    """Validate user-entered settings without constraining the source convention."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind.capitalize()} response settings must be an object.")
    result = {}
    for dataset_id, settings in value.items():
        if not isinstance(dataset_id, str) or not dataset_id.strip() or not isinstance(settings, Mapping):
            raise ValueError(f"{kind.capitalize()} response settings require a dataset ID and settings object.")
        if kind == "point":
            one_sided = settings.get("one_sided", True)
            if not isinstance(one_sided, bool):
                raise ValueError(f"Point response {dataset_id}: one_sided must be true or false.")
            result[dataset_id] = {"one_sided": one_sided}
        else:
            normalized = {}
            for key, default in LINE_RESPONSE_DEFAULTS.items():
                try:
                    number = float(settings.get(key, default))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Line response {dataset_id}: {key} must be a finite number.") from exc
                if not math.isfinite(number):
                    raise ValueError(f"Line response {dataset_id}: {key} must be a finite number.")
                if key == "grazing_taper_deg" and number < 0:
                    raise ValueError(f"Line response {dataset_id}: grazing taper must be nonnegative.")
                normalized[key] = number
            result[dataset_id] = normalized
    return result


@dataclass
class FeatureAssemblyValues:
    """User-editable values, independent of any GUI toolkit."""

    base_grim: str = ""
    output_grim: str = ""
    # Placement and mesh formats do not reliably encode length units.  Empty
    # defaults force a deliberate choice instead of silently assuming inches.
    coordinate_units: str = ""
    surface_mesh: str = ""
    surface_units: str = ""
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: float | None = None
    point_locations_csv: str = ""
    line_locations_csv: str = ""
    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0
    allow_legacy_base_metadata: bool = True
    require_feature_manifests: bool = False
    require_body_mesh_certification: bool = False
    study_frequencies_ghz: tuple[float, ...] | None = None
    study_azimuths_deg: tuple[float, ...] | None = None
    study_elevations_deg: tuple[float, ...] | None = None
    base_dir: str | None = None
    point_datasets: dict[str, str] = field(default_factory=dict)
    line_datasets: dict[str, str] = field(default_factory=dict)
    point_response_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    line_response_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Spatial feature-definition state. These stable CSV IDs are independent
    # from both preview visibility and whole-response dataset arithmetic.
    excluded_point_placement_ids: set[str] = field(default_factory=set)
    excluded_line_ids: set[str] = field(default_factory=set)

@dataclass(frozen=True)
class LoadedFeatureAssemblyRecipe:
    """One validated recipe plus non-fatal source-integrity observations."""

    path: Path
    name: str
    variant: str
    values: FeatureAssemblyValues
    source_warnings: tuple[str, ...] = ()
