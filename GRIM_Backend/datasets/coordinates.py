"""Wedge, SENTRi, and elevation-pair coordinate transforms."""
from __future__ import annotations

import copy
import json

import numpy as np

from GRIM_Backend.datasets.constants import (
    CONIC_VH_BASIS_CONVENTION,
    WEDGE_TURNTABLE_CONVENTION,
    _ADOPT_CLEAN_ARRAYS_TOKEN,
    _ANGLE_UNITS,
)


def wedge_to_conic_geometry_deg(phi_deg, tau_deg):
    """Map vertical-turntable/body-wedge coordinates to conic directions.

    ``phi`` is rotation about the fixed world ``+z`` turntable axis. ``tau``
    is a body pitch about body ``+y`` applied before that rotation, so the
    body-to-world attitude is ``Rz(phi) @ Ry(tau)``.  The returned longitude
    and latitude describe the same world line of sight in body coordinates.
    """

    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    tau = np.deg2rad(np.asarray(tau_deg, dtype=float))
    phi, tau = np.broadcast_arrays(phi, tau)
    direction = np.stack(
        (
            np.cos(tau) * np.cos(phi),
            -np.sin(phi),
            np.sin(tau) * np.cos(phi),
        ),
        axis=-1,
    )
    longitude = np.arctan2(direction[..., 1], direction[..., 0])
    latitude = np.arcsin(np.clip(direction[..., 2], -1.0, 1.0))
    return np.rad2deg(longitude), np.rad2deg(latitude)


def conic_to_wedge_geometry_deg(longitude_deg, latitude_deg):
    """Inverse of :func:`wedge_to_conic_geometry_deg` for |tau| <= 90 deg."""

    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    longitude, latitude = np.broadcast_arrays(longitude, latitude)
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    sin_phi = np.clip(-y, -1.0, 1.0)
    cos_phi_magnitude = np.sqrt(np.maximum(0.0, 1.0 - sin_phi * sin_phi))


    cos_phi = np.where(x < 0.0, -cos_phi_magnitude, cos_phi_magnitude)
    phi = np.arctan2(sin_phi, cos_phi)
    branch = np.where(cos_phi < 0.0, -1.0, 1.0)
    tau = np.arctan2(branch * z, branch * x)
    return np.rad2deg(phi), np.rad2deg(tau)


def wedge_to_conic_basis_change(phi_deg, tau_deg):
    """Return old-basis coordinates of the conic ``(V,H)`` basis.

    If ``S_w`` is a monostatic Jones matrix in the range's vertical/horizontal
    basis for the vertical-turntable wedge setup, the normal conic-range
    matrix is ``C.T @ S_w @ C``.  The last two axes of the result are ordered
    old ``(V,H)`` by new ``(V,H)``.
    """

    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    tau = np.deg2rad(np.asarray(tau_deg, dtype=float))
    phi, tau = np.broadcast_arrays(phi, tau)
    longitude_deg, latitude_deg = wedge_to_conic_geometry_deg(
        np.rad2deg(phi), np.rad2deg(tau)
    )
    longitude = np.deg2rad(longitude_deg)
    latitude = np.deg2rad(latitude_deg)

    wedge_v = np.stack(
        (-np.sin(tau), np.zeros_like(tau), np.cos(tau)), axis=-1
    )
    wedge_h = np.stack(
        (
            np.cos(tau) * np.sin(phi),
            np.cos(phi),
            np.sin(tau) * np.sin(phi),
        ),
        axis=-1,
    )
    conic_v = np.stack(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        ),
        axis=-1,
    )
    conic_h = np.stack(
        (-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)),
        axis=-1,
    )
    old_basis = np.stack((wedge_v, wedge_h), axis=-2)
    new_basis = np.stack((conic_v, conic_h), axis=-2)
    return np.einsum("...ia,...ja->...ij", old_basis, new_basis)


def rotate_wedge_jones_to_conic(jones, phi_deg, tau_deg):
    """Rotate monostatic Jones matrices from wedge-range to conic V/H."""

    matrix = np.asarray(jones)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("Jones data must end with a 2x2 (receive, transmit) matrix")
    change = wedge_to_conic_basis_change(phi_deg, tau_deg)
    return np.einsum("...ia,...ij,...jb->...ab", change, matrix, change)


def _jones_from_polarization_channels(
    field, polarizations, *, assume_missing_cross_pol_zero=False
):
    """Build ``[..., receive(V,H), transmit(V,H)]`` from named channels."""

    labels = [str(value).strip().upper() for value in polarizations]
    if len(set(labels)) != len(labels):
        raise ValueError("Wedge-to-Conic requires unique polarization labels")
    unsupported = sorted(set(labels) - {"VV", "VH", "HV", "HH"})
    if unsupported:
        raise ValueError(
            "Wedge-to-Conic Jones rotation supports VV/VH/HV/HH labels only; got "
            + ", ".join(unsupported)
        )
    index = {label: position for position, label in enumerate(labels)}
    if "VV" not in index or "HH" not in index:
        raise ValueError(
            "Wedge-to-Conic Jones rotation requires both VV and HH channels"
        )
    values = np.asarray(field)
    matrix = np.empty(values.shape[:-1] + (2, 2), dtype=values.dtype)
    matrix[..., 0, 0] = values[..., index["VV"]]
    matrix[..., 1, 1] = values[..., index["HH"]]
    if "VH" in index and "HV" in index:
        matrix[..., 0, 1] = values[..., index["VH"]]
        matrix[..., 1, 0] = values[..., index["HV"]]
        cross_note = "measured VH and HV"
    elif "VH" in index:
        matrix[..., 0, 1] = values[..., index["VH"]]
        matrix[..., 1, 0] = values[..., index["VH"]]
        cross_note = "monostatic reciprocity: HV=VH"
    elif "HV" in index:
        matrix[..., 1, 0] = values[..., index["HV"]]
        matrix[..., 0, 1] = values[..., index["HV"]]
        cross_note = "monostatic reciprocity: VH=HV"
    elif assume_missing_cross_pol_zero:
        matrix[..., 0, 1] = 0.0
        matrix[..., 1, 0] = 0.0
        cross_note = "explicit assumption: missing VH=HV=0"
    else:
        raise ValueError(
            "Wedge-to-Conic changes the V/H basis and cannot rotate VV/HH "
            "alone. Supply VH or HV (monostatic reciprocity supplies the "
            "other channel), or explicitly assume missing cross-pol is zero."
        )
    return matrix, labels, cross_note


def _polarization_channels_from_jones(matrix, labels):
    channel = {"VV": (0, 0), "VH": (0, 1), "HV": (1, 0), "HH": (1, 1)}
    return np.stack(
        [matrix[..., channel[label][0], channel[label][1]] for label in labels],
        axis=-1,
    )


def canonical_angular_coordinate_system(value):
    """Normalize scalar angular-coordinate metadata without guessing."""

    raw = value
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.reshape(-1)[0]
    text = str(raw or "").strip().lower().replace("-", "_")
    aliases = {
        "": "conic",
        "az_el": "conic",
        "azimuth_elevation": "conic",
        "spherical": "conic",
        "gc": "great_circle",
        "greatcircle": "great_circle",
    }
    return aliases.get(text, text)


_ANGULAR_DECLARATION_UNIT_KEYS = (
    "great_circle_coordinate_convention",
    "angular_roll_deg",
    "angular_tilt_deg",
)
_ANGULAR_DECLARATION_EXTRA_KEYS = (
    "great_circle_coordinate_convention",
    "angular_coordinate_declaration_json",
    "great_circle_conversion_assumption_json",
    "ptm_cut_type",
    "ptm_cut_type_source",
)


def strip_angular_coordinate_declarations(units, extra):
    """Drop great-circle and other non-azimuth/elevation chart declarations.

    GRIM treats every dataset's angle axes as azimuth/elevation. A plain
    ``conic`` tag is kept for downstream consumers that expect it; anything
    else (great-circle, wedge-turntable, ...) and its frame metadata is
    removed in place.
    """

    for container in (units, extra):
        declared = container.get("angular_coordinate_system")
        if (
            declared is not None
            and canonical_angular_coordinate_system(declared) != "conic"
        ):
            container.pop("angular_coordinate_system", None)
    for key in _ANGULAR_DECLARATION_UNIT_KEYS:
        units.pop(key, None)
    for key in _ANGULAR_DECLARATION_EXTRA_KEYS:
        extra.pop(key, None)


class GridCoordinatesMixin:
    """Wedge, SENTRi, and elevation-pair coordinate transforms."""

    def convert_wedge_to_conic(
        self,
        *,
        attest_wedge_axes=False,
        assume_missing_cross_pol_zero=False,
    ):
        """Convert a vertical-turntable/body-wedge acquisition to conic V/H.

        This produces a body-relative conic azimuth/elevation grid. Direction
        queries are inverse-mapped into the measured
        ``(turntable phi, body wedge tau)`` grid,
        interpolated as a full complex Jones matrix, and congruence-rotated
        into the conic spherical V/H basis. Unsupported parts of the normal
        conic grid remain NaN; they are never extrapolated.

        A single nonzero wedge tilt traces a great circle and cannot determine
        a constant-elevation normal azimuth cut, so at least two measured
        tilts are needed for regridding. A zero-tilt-only sweep is already the
        waterline conic cut: only its mechanical azimuth convention is changed,
        preserving all samples without needing phase or additional channels.
        Both paths require a complete turntable revolution.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        az_unit = self._canonical_unit(
            (self.units or {}).get("azimuth"), _ANGLE_UNITS, "deg"
        )
        el_unit = self._canonical_unit(
            (self.units or {}).get("elevation"), _ANGLE_UNITS, "deg"
        )
        if az_unit not in {"deg", "rad"} or el_unit not in {"deg", "rad"}:
            raise ValueError(
                "Wedge-to-Conic requires degree or radian angle axes; got "
                f"azimuth={az_unit!r}, elevation={el_unit!r}"
            )
        phi = np.asarray(self.azimuths, dtype=float)
        tau = np.asarray(self.elevations, dtype=float)
        if phi.size < 4 or not np.all(np.isfinite(phi)):
            raise ValueError(
                "Wedge-to-Conic requires at least four finite turntable angles"
            )
        if tau.size == 0 or not np.all(np.isfinite(tau)):
            raise ValueError("Wedge-to-Conic requires finite measured wedge tilts")
        phi_deg = np.rad2deg(phi) if az_unit == "rad" else phi
        tau_deg = np.rad2deg(tau) if el_unit == "rad" else tau
        waterline_only = tau_deg.size == 1 and tau_deg[0] == 0.0
        if tau.size == 1 and not waterline_only:
            raise ValueError(
                "One fixed wedge tilt away from waterline traces a great circle and cannot be "
                "converted into a normal constant-elevation azimuth sweep. "
                "Supply at least two measured wedge tilts."
            )
        if np.any(np.abs(tau_deg) >= 90.0 - 1.0e-9):
            raise ValueError(
                "Wedge-to-Conic requires body wedge tilts strictly between "
                "-90 and +90 degrees"
            )

        wrapped_phi = np.mod(phi_deg + 180.0, 360.0) - 180.0
        wrapped_phi[np.abs(wrapped_phi) <= 1.0e-12] = 0.0
        phi_order = np.argsort(wrapped_phi, kind="stable")
        wrapped_phi = wrapped_phi[phi_order]
        if np.any(np.diff(wrapped_phi) <= 1.0e-9):
            raise ValueError(
                "Wedge turntable axis contains duplicate or seam-alias angles"
            )
        circular_gaps = np.diff(
            np.concatenate((wrapped_phi, [wrapped_phi[0] + 360.0]))
        )
        typical_gap = float(np.median(circular_gaps))
        if (
            not np.isfinite(typical_gap)
            or typical_gap <= 0.0
            or float(np.max(circular_gaps)) > 2.5 * typical_gap + 1.0e-7
        ):
            raise ValueError(
                "Wedge-to-Conic normal-azimuth conversion requires a complete "
                "turntable revolution without a large unmeasured angular gap"
            )

        if waterline_only:
            return self._convert_wedge_waterline_to_conic(
                phi_deg, az_unit=az_unit, attest_wedge_axes=attest_wedge_axes
            )

        tau_order = np.argsort(tau_deg, kind="stable")
        tau_sorted = tau_deg[tau_order]
        if np.any(np.diff(tau_sorted) <= 1.0e-9):
            raise ValueError("Wedge tilt axis contains duplicate coordinates")
        if np.any(
            np.isfinite(self.rcs_power) & ~np.isfinite(self.rcs_phase)
        ):
            raise ValueError(
                "Wedge-to-Conic Jones rotation requires finite phase for every "
                "finite polarization sample; power-only data cannot be rotated"
            )

        source_field = np.asarray(self.rcs)[phi_order, ...][:, tau_order, ...]
        source_jones, labels, cross_note = _jones_from_polarization_channels(
            source_field,
            self.polarizations,
            assume_missing_cross_pol_zero=assume_missing_cross_pol_zero,
        )


        from scipy.interpolate import RegularGridInterpolator

        phi_interp = np.concatenate((wrapped_phi, [wrapped_phi[0] + 360.0]))
        jones_interp = np.concatenate(
            (source_jones, source_jones[:1, ...]), axis=0
        )
        interpolator = RegularGridInterpolator(
            (phi_interp, tau_sorted),
            jones_interp,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        target_lon = np.mod(-wrapped_phi + 180.0, 360.0) - 180.0
        target_lon[np.abs(target_lon) <= 1.0e-12] = 0.0
        target_lon = np.sort(target_lon, kind="stable")
        target_lat = np.array(tau_sorted, copy=True)
        lon_mesh, lat_mesh = np.meshgrid(target_lon, target_lat, indexing="ij")
        query_phi, query_tau = conic_to_wedge_geometry_deg(lon_mesh, lat_mesh)
        query_phi = (
            np.mod(query_phi - wrapped_phi[0], 360.0) + wrapped_phi[0]
        )
        query = np.column_stack((query_phi.ravel(), query_tau.ravel()))
        wedge_jones = interpolator(query)
        change = wedge_to_conic_basis_change(query[:, 0], query[:, 1])
        conic_jones = np.einsum(
            "qia,qfij,qjb->qfab", change, wedge_jones, change
        )
        conic_channels = _polarization_channels_from_jones(
            conic_jones, labels
        )
        output_shape = (
            target_lon.size,
            target_lat.size,
            self.frequencies.size,
            self.polarizations.size,
        )
        conic_channels = conic_channels.reshape(output_shape)

        converted_units = copy.deepcopy(self.units or {})
        converted_units["angular_coordinate_system"] = "conic"
        converted_units["polarization_basis"] = CONIC_VH_BASIS_CONVENTION
        converted_units.pop("wedge_coordinate_convention", None)
        output_lon = np.deg2rad(target_lon) if az_unit == "rad" else target_lon
        output_lat = np.deg2rad(target_lat) if el_unit == "rad" else target_lat

        converted_extra = {}
        original_shape = tuple(self.rcs_power.shape)
        stale = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
            "rcs_amp_real",
            "rcs_amp_imag",
        }
        for key, value in (self.extra or {}).items():
            if key in stale:
                continue
            array = np.asarray(value)
            if array.ndim >= 4 and tuple(array.shape[:4]) == original_shape:
                continue
            converted_extra[key] = copy.deepcopy(value)
        converted_extra.update(
            {
                "source_angular_coordinate_system": "wedge_turntable",
                "wedge_coordinate_convention": WEDGE_TURNTABLE_CONVENTION,
                "polarization_basis": CONIC_VH_BASIS_CONVENTION,
                "wedge_to_conic_mode": "complex_regrid",
                "wedge_to_conic_cross_pol_treatment": cross_note,
            }
        )
        converted_extra["wedge_axes_assumption_json"] = json.dumps(
            {
                "schema": "grim.wedge-axes-assumption.v1",
                "operation_requested": True,
                "source_coordinate_declaration_missing": True,
                "assumed_axes": WEDGE_TURNTABLE_CONVENTION,
                "legacy_user_attested": bool(attest_wedge_axes),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        geometric_supported = (
            (query_tau >= tau_sorted[0] - 1.0e-9)
            & (query_tau <= tau_sorted[-1] + 1.0e-9)
        )
        coverage = 100.0 * float(np.count_nonzero(geometric_supported)) / float(
            geometric_supported.size
        )
        history_entry = (
            "Wedge->Conic physical regrid: inverse direction map; complex "
            f"Jones C^T*S*C rotation ({cross_note}); no extrapolation; "
            f"normal-grid geometric coverage {coverage:.1f}%"
        )
        history_entry += "; source axes assumed from requested operation"
        history = (
            f"{self.history}\n{history_entry}" if self.history else history_entry
        )
        return RcsGrid(
            output_lon,
            output_lat,
            self.frequencies,
            self.polarizations,
            rcs=conic_channels,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=converted_units,
            extra=converted_extra,
        )

    def _convert_wedge_waterline_to_conic(self, phi_deg, *, az_unit, attest_wedge_axes):
        """Exactly relabel a validated zero-tilt sweep, including aligned data."""
        from GRIM_Backend.datasets.grid import RcsGrid

        longitude = np.mod(-phi_deg + 180.0, 360.0) - 180.0
        longitude[np.abs(longitude) <= 1.0e-12] = 0.0
        order = np.argsort(longitude, kind="stable")
        longitude = longitude[order]
        if az_unit == "rad":
            longitude = np.deg2rad(longitude)

        # At tau=0 the range and conic polarization bases coincide. Going
        # through complex amplitudes would discard valid power-only samples.
        power = np.take(self.rcs_power, order, axis=0)
        phase = np.take(self.rcs_phase, order, axis=0)
        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
            "assembly_angular_coordinate_contract",
        }
        extra = {}
        for key, value in (self.extra or {}).items():
            if key in stale_grid_metadata:
                continue
            array = np.asarray(value)
            if array.ndim >= 4 and tuple(array.shape[:4]) == original_shape:
                extra[key] = np.take(array, order, axis=0)
            else:
                extra[key] = copy.deepcopy(value)
        self._drop_malformed_raw_metadata(extra)
        self._invalidate_assembly_sampling_hash(extra, "wedge-waterline-to-conic")

        units = copy.deepcopy(self.units or {})
        units["angular_coordinate_system"] = "conic"
        units.pop("wedge_coordinate_convention", None)
        if set(self.polarizations).issubset({"VV", "VH", "HV", "HH"}):
            units["polarization_basis"] = CONIC_VH_BASIS_CONVENTION
            extra["polarization_basis"] = CONIC_VH_BASIS_CONVENTION
        extra.update({
            "source_angular_coordinate_system": "wedge_turntable",
            "wedge_coordinate_convention": WEDGE_TURNTABLE_CONVENTION,
            "wedge_to_conic_mode": "waterline_relabel",
            "wedge_to_conic_cross_pol_treatment": "not required: identity basis at waterline",
            "wedge_axes_assumption_json": json.dumps(
                {
                    "schema": "grim.wedge-axes-assumption.v1",
                    "operation_requested": True,
                    "source_coordinate_declaration_missing": True,
                    "assumed_axes": WEDGE_TURNTABLE_CONVENTION,
                    "legacy_user_attested": bool(attest_wedge_axes),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        })
        history_entry = (
            "Wedge->Conic exact waterline relabel: azimuth=-turntable angle "
            "wrapped to [-180,180) deg; elevation=0; stable-sorted sample arrays; "
            "no interpolation or polarization rotation; phase unchanged; "
            "source axes assumed from requested operation"
        )
        history = f"{self.history}\n{history_entry}" if self.history else history_entry
        return RcsGrid(
            longitude,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=units,
            extra=extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    def convert_sentri_elevation_to_grim(self):
        """Convert native SENTRi polar theta to GRIM signed elevation.

        SENTRi uses a polar angle measured down from the top look: 0 degrees is
        top-down, 90 degrees is waterline, and 180 degrees is bottom-up.  GRIM's
        conic elevation is positive above waterline, so the exact relabel is
        ``elevation = 90 - theta``.  The transformed elevation axis is
        stable-sorted and every grid-shaped sample/provenance array follows the
        same permutation.  SENTRi phi is also wrapped into GRIM's canonical
        [0, 360) degree azimuth axis and sorted.  There is no interpolation and
        no phase change.

        Explicitly incompatible elevation metadata is rejected. If convention
        metadata is absent, selecting this format-specific operation records
        the native-SENTRi assumption instead of blocking the conversion.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        native_tag = "sentri_theta_top_zero"
        grim_tag = "grim_elevation_waterline_zero_top_positive"
        units = copy.deepcopy(self.units)
        extra = dict(self.extra)
        convention = str(
            units.get(
                "elevation_coordinate_convention",
                extra.get("sentri_elevation_convention", ""),
            )
            or ""
        ).strip().lower()
        source_format = str(extra.get("source_format", "") or "").strip()

        if convention == grim_tag:
            raise ValueError("dataset already uses GRIM signed elevation")
        if convention and convention != native_tag:
            raise ValueError(
                "SENTRi coordinate conversion cannot override explicit "
                f"elevation convention {convention!r}"
            )
        assumed_native_convention = not bool(convention)

        elevation_unit = self._canonical_unit(
            units.get("elevation"), _ANGLE_UNITS, "deg"
        )
        if elevation_unit != "deg":
            raise ValueError(
                "SENTRi elevation conversion requires a degree-valued "
                f"elevation axis; got {units.get('elevation')!r}"
            )
        azimuth_unit = self._canonical_unit(
            units.get("azimuth"), _ANGLE_UNITS, "deg"
        )
        if azimuth_unit != "deg":
            raise ValueError(
                "SENTRi azimuth conversion requires a degree-valued azimuth "
                f"axis; got {units.get('azimuth')!r}"
            )

        native_theta = np.asarray(self.elevations, dtype=float)
        if np.any(~np.isfinite(native_theta)):
            raise ValueError("SENTRi theta axis must contain only finite values")
        tolerance = 1.0e-9
        if np.any(native_theta < -tolerance) or np.any(
            native_theta > 180.0 + tolerance
        ):
            raise ValueError(
                "native SENTRi theta must be within 0..180 degrees before "
                "conversion"
            )

        converted_elevation = 90.0 - native_theta
        converted_elevation[np.abs(converted_elevation) <= tolerance] = 0.0
        order = np.argsort(converted_elevation, kind="stable")
        converted_elevation = converted_elevation[order]
        if (
            converted_elevation.size > 1
            and np.any(np.diff(converted_elevation) <= tolerance)
        ):
            raise ValueError(
                "SENTRi elevation conversion would create duplicate or "
                "near-duplicate GRIM elevation coordinates within 1e-9 deg"
            )

        native_azimuth = np.asarray(self.azimuths, dtype=float)
        if np.any(~np.isfinite(native_azimuth)):
            raise ValueError("SENTRi phi axis must contain only finite values")
        converted_azimuth = np.mod(native_azimuth, 360.0)
        converted_azimuth[
            np.isclose(converted_azimuth, 360.0, atol=tolerance, rtol=0.0)
            | np.isclose(converted_azimuth, 0.0, atol=tolerance, rtol=0.0)
        ] = 0.0
        azimuth_order = np.argsort(converted_azimuth, kind="stable")
        converted_azimuth = converted_azimuth[azimuth_order]
        if (
            converted_azimuth.size > 1
            and np.any(np.diff(converted_azimuth) <= tolerance)
        ):
            raise ValueError(
                "SENTRi azimuth wrapping would create duplicate or "
                "near-duplicate coordinates within 1e-9 deg; reload the "
                "source with read_SENTRi so its seam-precedence policy can "
                "resolve the duplicate first"
            )

        power = np.take(self.rcs_power, azimuth_order, axis=0)
        power = np.take(power, order, axis=1)
        phase = np.take(self.rcs_phase, azimuth_order, axis=0)
        phase = np.take(phase, order, axis=1)
        original_shape = tuple(self.rcs_power.shape)
        stale_grid_metadata = {
            "solver_metadata_json",
            "production_mesh_certification_json",
            "source_body_mesh_certification_json",
            "requested_radar_grid_json",
        }
        converted_extra = {}
        for key, value in extra.items():
            if key in stale_grid_metadata:
                continue
            value_array = np.asarray(value)
            if (
                value_array.ndim >= 4
                and tuple(value_array.shape[:4]) == original_shape
            ):
                converted_value = np.take(value_array, azimuth_order, axis=0)
                converted_extra[key] = np.take(converted_value, order, axis=1)
            else:
                converted_extra[key] = value

        self._drop_malformed_raw_metadata(converted_extra)
        self._invalidate_assembly_sampling_hash(
            converted_extra, "convert-native-sentri-to-grim"
        )

        units["elevation_coordinate_convention"] = grim_tag
        converted_extra["sentri_elevation_convention"] = grim_tag
        converted_extra["assembly_angular_coordinate_contract"] = (
            "ghost.radar-azimuth-elevation.coming-from.deg.v1"
        )
        converted_extra["sentri_coordinate_mapping"] = (
            "GRIM elevation = 90 deg - native SENTRi theta; "
            "azimuth=wrapped phi"
        )
        if assumed_native_convention:
            converted_extra["sentri_coordinate_assumption_json"] = json.dumps(
                {
                    "schema": "grim.sentri-coordinate-assumption.v1",
                    "operation_requested": True,
                    "source_format": source_format or None,
                    "source_elevation_convention_missing": True,
                    "assumed_convention": native_tag,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        prior_history = str(self.history or "").strip()
        history_entry = (
            "Convert native SENTRi coordinates to GRIM conic angles "
            "(elevation=90-theta; azimuth=phi wrapped to [0,360) deg); "
            "stable-sorted axes and sample arrays; no interpolation or phase "
            "change"
        )
        if assumed_native_convention:
            history_entry += "; untagged source convention assumed from operation"
        history = (
            f"{prior_history}\n{history_entry}" if prior_history else history_entry
        )

        return RcsGrid(
            converted_azimuth,
            converted_elevation,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain=self.rcs_domain,
            source_path=self.source_path,
            history=history,
            units=units,
            extra=converted_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    def combine_elevation_pair_to_azimuth_360(
        self,
        elevation_lo: float | None = None,
        elevation_hi: float | None = None,
        *,
        azimuth_shift_deg: float = 180.0,
        tol: float = 1e-6,
        assumptions_attested: bool = False,
    ):
        """Stitch two elevation cuts into one 0-360 azimuth cut.

        The lower-elevation cut keeps its original azimuth values. The higher
        cut is shifted by the degree-valued `azimuth_shift_deg` and merged onto
        the same output elevation plane. On radian-native data the shift and
        tolerance are converted internally. Equivalent/complementary overlap
        bins merge; conflicting finite seam samples are rejected. Because this
        is an acquisition-specific relabel rather than a general spherical
        coordinate transform, untagged inputs require explicit attestation and
        the two elevations must be equal and opposite about zero.
        """

        if not isinstance(assumptions_attested, (bool, np.bool_)):
            raise TypeError("assumptions_attested must be True or False")

        el_axis = np.asarray(self.elevations, dtype=float)
        if el_axis.size < 2:
            raise ValueError("need at least 2 elevation values to combine into 360 azimuth")

        if elevation_lo is None or elevation_hi is None:
            finite = el_axis[np.isfinite(el_axis)]
            if finite.size < 2:
                raise ValueError("elevation axis has fewer than 2 finite values")
            lo_value = float(np.min(finite))
            hi_value = float(np.max(finite))
        else:
            lo_value = float(elevation_lo)
            hi_value = float(elevation_hi)

        if not np.isfinite(lo_value) or not np.isfinite(hi_value):
            raise ValueError("elevation pair values must be finite")
        try:
            tolerance_deg = float(tol)
        except (TypeError, ValueError) as exc:
            raise ValueError("combine tolerance must be finite and nonnegative") from exc
        if not np.isfinite(tolerance_deg) or tolerance_deg < 0.0:
            raise ValueError("combine tolerance must be finite and nonnegative")
        elevation_unit = self._supported_unit(
            "elevation", _ANGLE_UNITS, "deg"
        )
        azimuth_unit = self._supported_unit("azimuth", _ANGLE_UNITS, "deg")
        elevation_tol = (
            float(np.deg2rad(tolerance_deg))
            if elevation_unit == "rad"
            else tolerance_deg
        )
        azimuth_tol = (
            float(np.deg2rad(tolerance_deg))
            if azimuth_unit == "rad"
            else tolerance_deg
        )
        if np.isclose(lo_value, hi_value, atol=elevation_tol, rtol=0.0):
            raise ValueError("elevation pair values must be distinct")
        lo_deg = float(np.rad2deg(lo_value)) if elevation_unit == "rad" else lo_value
        hi_deg = float(np.rad2deg(hi_value)) if elevation_unit == "rad" else hi_value
        if not np.isclose(
            lo_deg,
            -hi_deg,
            atol=max(tolerance_deg, 1.0e-9),
            rtol=0.0,
        ):
            raise ValueError(
                "El->Az360 requires equal-and-opposite elevation cuts about "
                f"zero; got {lo_deg:.12g} and {hi_deg:.12g} deg"
            )
        if not np.isclose(
            float(azimuth_shift_deg), 180.0, atol=max(tolerance_deg, 1.0e-9), rtol=0.0
        ):
            raise ValueError(
                "El->Az360 requires a 180 degree second-half azimuth shift"
            )
        declared_contract = self._declared_scalar_metadata(
            "elevation_pair_azimuth_contract"
        )
        if not declared_contract and not bool(assumptions_attested):
            raise ValueError(
                "El->Az360 is an acquisition-specific relabel. Confirm the "
                "opposite-elevation/180-degree acquisition assumption before "
                "creating the result."
            )

        lo_matches = self._axis_value_match(
            self.elevations, lo_value, tol=elevation_tol
        )
        hi_matches = self._axis_value_match(
            self.elevations, hi_value, tol=elevation_tol
        )
        if lo_matches.size == 0 or hi_matches.size == 0:
            raise ValueError("requested elevation pair not found in dataset")

        lo_idx = int(lo_matches[0])
        hi_idx = int(hi_matches[0])
        az_shift = self._angle_value_from_degrees(
            azimuth_shift_deg, "azimuth"
        )

        az_base = np.asarray(self.azimuths, dtype=float)
        if az_base.size == 0:
            raise ValueError("dataset has no azimuth samples")

        az_lo = np.array(az_base, copy=True)
        az_hi = np.array(az_base, copy=True) + az_shift
        az_merged = self._axis_union([az_lo, az_hi], tol=azimuth_tol)
        if az_merged.size == 0:
            raise ValueError("combined azimuth axis is empty")

        out_shape = (len(az_merged), 1, len(self.frequencies), len(self.polarizations))
        out_power = np.full(out_shape, np.nan, dtype=self.rcs_power.dtype)
        out_phase = np.full(out_shape, np.nan, dtype=self.rcs_phase.dtype)
        raw_pair = self._complete_authoritative_raw_arrays()
        preserve_raw = raw_pair is not None
        if preserve_raw:
            raw_real = np.asarray(raw_pair[0], dtype=np.float64)
            raw_imag = np.asarray(raw_pair[1], dtype=np.float64)
            out_raw_real = np.full(out_shape, np.nan, dtype=np.float64)
            out_raw_imag = np.full(out_shape, np.nan, dtype=np.float64)

        lo_target_idx = self._indices_for_axis_values(
            az_merged, az_lo, tol=azimuth_tol
        )
        hi_target_idx = self._indices_for_axis_values(
            az_merged, az_hi, tol=azimuth_tol
        )
        if lo_target_idx is None or hi_target_idx is None:
            raise ValueError("failed to align azimuth bins during elevation combine")
        if (
            len(lo_target_idx) != az_lo.size
            or len(hi_target_idx) != az_hi.size
        ):
            raise ValueError(
                "cannot combine elevation cuts: the input azimuth axis contains "
                "coordinates closer than the matching tolerance "
                f"({tolerance_deg:g} deg); "
                "deduplicate the azimuth axis or use a smaller tolerance"
            )

        lo_power = self.rcs_power[:, lo_idx, :, :]
        lo_phase = self.rcs_phase[:, lo_idx, :, :]
        hi_power = self.rcs_power[:, hi_idx, :, :]
        hi_phase = self.rcs_phase[:, hi_idx, :, :]
        if preserve_raw:
            lo_raw_real = raw_real[:, lo_idx, :, :]
            lo_raw_imag = raw_imag[:, lo_idx, :, :]
            hi_raw_real = raw_real[:, hi_idx, :, :]
            hi_raw_imag = raw_imag[:, hi_idx, :, :]

        for label, target_indices, source_power, source_phase, source_real, source_imag in (
            (
                "lower elevation",
                lo_target_idx,
                lo_power,
                lo_phase,
                lo_raw_real if preserve_raw else None,
                lo_raw_imag if preserve_raw else None,
            ),
            (
                "shifted higher elevation",
                hi_target_idx,
                hi_power,
                hi_phase,
                hi_raw_real if preserve_raw else None,
                hi_raw_imag if preserve_raw else None,
            ),
        ):
            for src_idx, dst_idx in enumerate(target_indices):
                context = (
                    f"El->Az360 {label} at azimuth "
                    f"{az_merged[dst_idx]:.12g} {azimuth_unit}"
                )
                self._merge_equivalent_sample_blocks(
                    out_power[dst_idx, 0, :, :],
                    out_phase[dst_idx, 0, :, :],
                    source_power[src_idx, :, :],
                    source_phase[src_idx, :, :],
                    context=context,
                )
                if preserve_raw:
                    self._merge_equivalent_raw_blocks(
                        out_raw_real[dst_idx, 0, :, :],
                        out_raw_imag[dst_idx, 0, :, :],
                        source_real[src_idx, :, :],
                        source_imag[src_idx, :, :],
                        context=context,
                    )

        if preserve_raw:
            unmodeled = ~np.isfinite(out_power)
            out_raw_real[unmodeled] = np.nan
            out_raw_imag[unmodeled] = np.nan

        combined_extra = self._exact_transform_extra(
            coordinate_change="combine-elevation-pair-to-azimuth-360",
            preserve_angular_contract=False,
            preserve_raw=False,
        )
        if preserve_raw:
            combined_extra["rcs_amp_real"] = out_raw_real
            combined_extra["rcs_amp_imag"] = out_raw_imag
            combined_extra["raw_complex_amplitude_preserved"] = True
        combined_extra["elevation_pair_to_azimuth_json"] = json.dumps(
            {
                "schema": "grim.elevation-pair-to-azimuth.v1",
                "operation": "acquisition_specific_relabel",
                "elevation_pair_deg": [lo_deg, hi_deg],
                "azimuth_shift_deg": float(azimuth_shift_deg),
                "declared_contract": declared_contract or None,
                "user_assumptions_attested": bool(assumptions_attested),
                "interpolation": False,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return self._new_grid(
            az_merged,
            np.asarray([el_axis[lo_idx]], dtype=float),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=out_power,
            rcs_phase=out_phase,
            rcs_domain="power_phase",
            extra=combined_extra,
        )
