"""SENTRi table parsing and RCS grid conversion."""
from __future__ import annotations

import csv
import re

import numpy as np


class SentriFormatMixin:
    """SENTRi signature detection and complex-field table loading."""

    @classmethod
    def read_SENTRi(cls, path, *, max_output_bytes=None, _estimate_only=False):
        """Read either RCS table schema emitted by CREATE-RF SENTRi.

        Supported header families:

        * compact ``freq_MHz`` / ``theta_deg`` / ``rcs_pp_dBsm`` columns;
        * descriptive ``Frequency`` / ``Theta`` columns with either
          ``Scat-Inc`` RCS and legacy phase labels or explicit ``Inc-Scat``
          labels for both RCS and phase.

        SENTRi's reported polar ``Theta`` is stored unchanged so importing a
        file never silently changes its geometry.  The explicit
        :meth:`convert_sentri_elevation_to_grim` operation maps that native
        top-down convention to GRIM elevation when requested.  Phi sweeps
        contained in [0, 180] retain their positive endpoint; other sweeps
        use the signed [-180, 180) azimuth interval.  Reported
        E-field phase is stored with its original sign, so each sample is
        reconstructed as
        ``10**(dBsm/20) * exp(+1j*deg2rad(phase_deg))``.  The far-field phase convention is recorded for coherent Assembly;
        native theta still requires the explicit signed-elevation conversion.
        """
        from GRIM_Backend.datasets.constants import SENTRI_FAR_FIELD_METADATA, _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.cst import _cst_compact_header, _cst_dbsm_to_power, _DelimitedRows, _vendor_import_allocation, _vendor_estimate, _wrap_cst_azimuth_deg
        from GRIM_Backend.io.samples import _cst_samples_equivalent, construct_parsed_grid

        rows = _DelimitedRows(path)
        if not rows:
            raise ValueError("SENTRi table is empty")

        compact_schema = {
            "freqmhz": "frequency",
            "thetadeg": "theta",
            "phideg": "phi",
            "rcsppdbsm": "rcs_hh",
            "efieldphaseppdeg": "phase_hh",
            "rcsttdbsm": "rcs_vv",
            "efieldphasettdeg": "phase_vv",
            "rcsptdbsm": "rcs_hv",
            "efieldphaseptdeg": "phase_hv",
            "rcstpdbsm": "rcs_vh",
            "efieldphasetpdeg": "phase_vh",
        }
        descriptive_schema = {
            "frequency": "frequency",
            "theta": "theta",
            "phi": "phi",
            "rcsphiscatphiinc": "rcs_hh",
            "phasephiphi": "phase_hh",
            "rcsthetascatthetainc": "rcs_vv",
            "phasethetatheta": "phase_vv",
            "rcsphiscatthetainc": "rcs_hv",
            "phasephitheta": "phase_hv",
            "rcsthetascatphiinc": "rcs_vh",
            "phasethetaphi": "phase_vh",
            # Newer exports name the incident component first. GRIM keeps
            # scattered/received polarization first (HV = PhiScat/ThetaInc).
            "rcsphiincphiscat": "rcs_hh",
            "phasephiincphiscat": "phase_hh",
            "rcsthetaincthetascat": "rcs_vv",
            "phasethetaincthetascat": "phase_vv",
            "rcsthetaincphiscat": "rcs_hv",
            "phasethetaincphiscat": "phase_hv",
            "rcsphiincthetascat": "rcs_vh",
            "phasephiincthetascat": "phase_vh",
        }
        required = {
            "frequency", "theta", "phi",
            "rcs_vv", "phase_vv", "rcs_hv", "phase_hv",
            "rcs_vh", "phase_vh", "rcs_hh", "phase_hh",
        }

        header_idx = None
        columns = None
        frequency_scale = None
        schema_name = None
        for row_idx, row in enumerate(rows):
            normalized = [_cst_compact_header(cell) for cell in row]
            for aliases, scale, name in (
                (compact_schema, 1.0e-3, "compact MHz"),
                (descriptive_schema, 1.0e-9, "descriptive Hz"),
            ):
                mapped = {}
                for column_idx, token in enumerate(normalized):
                    key = aliases.get(token)
                    if key is not None and key not in mapped:
                        mapped[key] = column_idx
                if required.issubset(mapped):
                    header_idx = row_idx
                    columns = mapped
                    frequency_scale = scale
                    schema_name = name
                    break
            if header_idx is not None:
                break

        if header_idx is None or columns is None or frequency_scale is None:
            raise ValueError(
                "Could not find a complete SENTRi RCS header. Expected either "
                "freq_MHz/theta_deg/phi_deg with pp/tt/pt/tp magnitude and "
                "phase columns, or Frequency/Theta/Phi with the four "
                "Scat/Inc or Inc/Scat magnitude and phase pairs."
            )

        def _canonical_sentri_unit(raw_value):
            text = str(raw_value or "").strip().lower().replace("²", "2")
            if text == "°":
                return "deg"
            compact = re.sub(r"[^a-z0-9]+", "", text)
            aliases = {
                "hz": "hz",
                "hertz": "hz",
                "mhz": "mhz",
                "megahertz": "mhz",
                "deg": "deg",
                "degree": "deg",
                "degrees": "deg",
                "dbsm": "dbsm",
                "dbm2": "dbsm",
                "dbsqm": "dbsm",
                "dbsquaremeter": "dbsm",
                "dbsquaremetre": "dbsm",
            }
            return aliases.get(compact, compact)


        data_start_idx = header_idx + 1
        candidate = None
        for candidate_idx, candidate_row in enumerate(rows[data_start_idx:], start=data_start_idx):
            if candidate_row and any(str(cell).strip() for cell in candidate_row):
                candidate = candidate_row
                data_start_idx = candidate_idx
                break
        has_units_row = False
        if candidate is not None:
            frequency_cell_idx = columns["frequency"]
            frequency_cell = (
                str(candidate[frequency_cell_idx]).strip()
                if frequency_cell_idx < len(candidate)
                else ""
            )
            try:
                float(frequency_cell)
            except ValueError:
                expected_frequency_unit = (
                    "mhz" if schema_name == "compact MHz" else "hz"
                )
                expected_units = {
                    "frequency": expected_frequency_unit,
                    "theta": "deg",
                    "phi": "deg",
                    "rcs_vv": "dbsm",
                    "phase_vv": "deg",
                    "rcs_hv": "dbsm",
                    "phase_hv": "deg",
                    "rcs_vh": "dbsm",
                    "phase_vh": "deg",
                    "rcs_hh": "dbsm",
                    "phase_hh": "deg",
                }
                bad_units = []
                for key, expected_unit in expected_units.items():
                    column_idx = columns[key]
                    raw_unit = (
                        candidate[column_idx]
                        if column_idx < len(candidate)
                        else ""
                    )
                    actual_unit = _canonical_sentri_unit(raw_unit)
                    if actual_unit != expected_unit:
                        bad_units.append(
                            f"{key}={str(raw_unit).strip()!r} "
                            f"(expected {expected_unit})"
                        )
                if bad_units:
                    raise ValueError(
                        f"line {data_start_idx + 1}: invalid SENTRi units row: "
                        + "; ".join(bad_units)
                    )
                has_units_row = True
                data_start_idx += 1

        def _number(row, key, line_no, *, allow_negative_infinity=False):
            idx = columns[key]
            text = str(row[idx]).strip() if idx < len(row) else ""
            if not text:
                raise ValueError(f"line {line_no}: {key} is blank")
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            valid = np.isfinite(value) or (
                allow_negative_infinity and np.isneginf(value)
            )
            if not valid:
                expected = "finite or -Inf" if allow_negative_infinity else "finite"
                raise ValueError(f"line {line_no}: {key} must be {expected}")
            return float(value)

        channel_specs = (
            ("VV", "rcs_vv", "phase_vv"),
            ("HV", "rcs_hv", "phase_hv"),
            ("VH", "rcs_vh", "phase_vh"),
            ("HH", "rcs_hh", "phase_hh"),
        )
        def _records(*, coordinates_only=False):
            for row_idx, row in enumerate(
                rows[data_start_idx:], start=data_start_idx + 1
            ):
                if not row or all(not str(cell).strip() for cell in row):
                    continue
                raw_frequency = _number(row, "frequency", row_idx)
                frequency_ghz = raw_frequency * frequency_scale
                if not np.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
                    raise ValueError(f"line {row_idx}: frequency must be positive")
                theta_deg = _number(row, "theta", row_idx)
                coordinate_tolerance = 1.0e-9
                if (
                    theta_deg < -coordinate_tolerance
                    or theta_deg > 180.0 + coordinate_tolerance
                ):
                    raise ValueError(
                        f"line {row_idx}: SENTRi theta must be in [0, 180] deg"
                    )


                if abs(theta_deg) <= coordinate_tolerance:
                    theta_deg = 0.0
                elif abs(theta_deg - 180.0) <= coordinate_tolerance:
                    theta_deg = 180.0
                elevation_deg = float(theta_deg)
                raw_phi_deg = _number(row, "phi", row_idx)
                if abs(raw_phi_deg) <= coordinate_tolerance:
                    raw_phi_deg = 0.0
                elif abs(raw_phi_deg - 360.0) <= coordinate_tolerance:
                    raw_phi_deg = 360.0
                elif abs(raw_phi_deg + 180.0) <= coordinate_tolerance:
                    raw_phi_deg = -180.0
                elif abs(raw_phi_deg - 180.0) <= coordinate_tolerance:
                    raw_phi_deg = 180.0
                azimuth_deg = _wrap_cst_azimuth_deg(raw_phi_deg)
                if abs(azimuth_deg) <= coordinate_tolerance:
                    azimuth_deg = 0.0
                elif abs(azimuth_deg + 180.0) <= coordinate_tolerance:
                    azimuth_deg = -180.0

                if coordinates_only:
                    yield (row_idx, azimuth_deg, elevation_deg, float(frequency_ghz),
                           None, None, None, raw_phi_deg)
                    continue
                for polarization, magnitude_key, phase_key in channel_specs:
                    magnitude_dbsm = _number(
                        row, magnitude_key, row_idx, allow_negative_infinity=True
                    )
                    reported_phase_deg = _number(row, phase_key, row_idx)
                    power = _cst_dbsm_to_power(
                        magnitude_dbsm,
                        context=f"line {row_idx} {magnitude_key}",
                    )
                    phase = float(np.deg2rad(reported_phase_deg))
                    yield (row_idx, azimuth_deg, elevation_deg, float(frequency_ghz),
                           polarization, power, phase, raw_phi_deg)

        az_values, el_values, freq_values, raw_phi_values = set(), set(), set(), set()
        for _, azimuth, elevation, frequency, _, _, _, raw_phi in _records(coordinates_only=True):
            az_values.add(azimuth)
            el_values.add(elevation)
            freq_values.add(frequency)
            raw_phi_values.add(raw_phi)
        if not az_values:
            raise ValueError("SENTRi table contains no data rows")
        positive_half_sweep = all(0.0 <= value <= 180.0 for value in raw_phi_values)

        def _axis_phi(phi):
            value = _wrap_cst_azimuth_deg(phi)
            if abs(value) <= 1.0e-9:
                value = 0.0
            elif abs(value + 180.0) <= 1.0e-9:
                value = -180.0
            return 180.0 if positive_half_sweep and value == -180.0 else value

        if positive_half_sweep and -180.0 in az_values:
            az_values.remove(-180.0)
            az_values.add(180.0)
        azimuths = np.asarray(sorted(az_values), dtype=float)
        elevations = np.asarray(sorted(el_values), dtype=float)
        frequencies = np.asarray(sorted(freq_values), dtype=float)
        polarizations = np.asarray([spec[0] for spec in channel_specs])
        shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
        # Most sweeps have one source phi per output azimuth. Only aliased
        # source coordinates need additional typed power/phase history to keep
        # exact same-source diagnostics when a preferred seam row replaces it.
        source_phis = {}
        for raw_phi in sorted(raw_phi_values):
            source_phis.setdefault(_axis_phi(raw_phi), []).append(raw_phi)
        alias_phis = [raw_phi for group in source_phis.values() if len(group) > 1
                      for raw_phi in group]
        alias_azimuths = [value for value, group in source_phis.items() if len(group) > 1]
        alias_index = {value: index for index, value in enumerate(alias_phis)}
        alias_az_index = {value: index for index, value in enumerate(alias_azimuths)}
        trailing_cells = int(np.prod(shape[1:], dtype=object))
        alias_bytes = trailing_cells * (24 * len(alias_phis) + 8 * len(alias_azimuths))
        allocation = _vendor_import_allocation(
            shape, source=f"SENTRi import {path}", max_output_bytes=max_output_bytes,
            extra_working_bytes=alias_bytes + 128 * len(raw_phi_values),
        )
        if _estimate_only:
            return _vendor_estimate(allocation, f"SENTRi {schema_name} RCS table")
        del az_values, el_values, freq_values, raw_phi_values, source_phis
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        first_line = np.zeros(shape, dtype=np.uint64)
        alias_shape = (len(alias_phis), *shape[1:])
        source_power = np.full(alias_shape, np.nan, dtype=np.float64)
        source_phase = np.full(alias_shape, np.nan, dtype=np.float64)
        source_line = np.zeros(alias_shape, dtype=np.uint64)
        selected_phi = np.full((len(alias_azimuths), *shape[1:]), np.nan)
        az_index = {value: idx for idx, value in enumerate(azimuths.tolist())}
        el_index = {value: idx for idx, value in enumerate(elevations.tolist())}
        freq_index = {value: idx for idx, value in enumerate(frequencies.tolist())}
        pol_index = {value: idx for idx, value in enumerate(polarizations.tolist())}
        used_zero_360_precedence = False
        used_signed_180_precedence = False
        for row_idx, azimuth, elevation, frequency, polarization, sample_power, sample_phase, raw_phi in _records():
            azimuth = _axis_phi(raw_phi)
            tail = (el_index[elevation], freq_index[frequency], pol_index[polarization])
            index = (az_index[azimuth], *tail)
            if raw_phi in alias_index:
                source_index = (alias_index[raw_phi], *tail)
                prior_line = source_line[source_index]
                prior_power, prior_phase = source_power[source_index], source_phase[source_index]
            else:
                source_index = None
                prior_line = first_line[index]
                prior_power, prior_phase = power[index], phase[index]
            if prior_line:
                if _cst_samples_equivalent(prior_power, prior_phase, sample_power, sample_phase):
                    continue
                raise ValueError(
                    f"line {row_idx}: conflicting duplicate SENTRi sample "
                    f"at source phi={raw_phi:g}; first defined on line {int(prior_line)}"
                )
            if source_index is not None:
                source_line[source_index] = row_idx
                source_power[source_index], source_phase[source_index] = sample_power, sample_phase
            alias_coordinate = ((alias_az_index[azimuth], *tail)
                                if azimuth in alias_az_index else None)
            if first_line[index]:
                prior_raw_phi = selected_phi[alias_coordinate]
                zero_360_pair = azimuth == 0.0 and {prior_raw_phi, raw_phi} == {0.0, 360.0}
                signed_180_pair = azimuth == -180.0 and {prior_raw_phi, raw_phi} == {-180.0, 180.0}
                if zero_360_pair or signed_180_pair:
                    if zero_360_pair:
                        used_zero_360_precedence = True
                        current_is_authoritative = raw_phi == 360.0
                    else:
                        used_signed_180_precedence = True
                        current_is_authoritative = raw_phi == 180.0
                    if not current_is_authoritative:
                        continue
                elif _cst_samples_equivalent(power[index], phase[index], sample_power, sample_phase):
                    continue
                else:
                    raise ValueError(
                        f"line {row_idx}: conflicting duplicate SENTRi sample "
                        f"after azimuth wrapping; first defined on line {int(first_line[index])}"
                    )
            first_line[index] = row_idx
            power[index], phase[index] = sample_power, sample_phase
            if alias_coordinate is not None:
                selected_phi[alias_coordinate] = raw_phi
        del first_line, source_power, source_phase, source_line, selected_phi

        phi_mapping = (
            "phi retained in [0, 180]" if positive_half_sweep
            else "phi wrapped to [-180, 180)"
        )
        mapping = (
            f"elevation=theta; {phi_mapping}; "
            "VV=tt/theta-theta, HV=pt/phi-theta, "
            "VH=tp/theta-phi, HH=pp/phi-phi; "
            "stored phase=reported E-field phase"
        )
        return construct_parsed_grid(cls,
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=str(path),
            history=f"Loaded SENTRi {schema_name} RCS table; {mapping}: {path}",
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={
                "source_format": f"SENTRi {schema_name} RCS table",
                **SENTRI_FAR_FIELD_METADATA,
                "sentri_coordinate_mapping": (
                    "elevation=theta; azimuth=phi retained in [0, 180]"
                    if positive_half_sweep
                    else "elevation=theta; azimuth=wrapped phi"
                ),
                "sentri_elevation_convention": "sentri_theta_top_zero",
                "sentri_zero_360_seam_policy": (
                    "source phi=360 supplies canonical azimuth 0 when both "
                    "phi=0 and phi=360 are present"
                ),
                "sentri_zero_360_precedence_used": bool(
                    used_zero_360_precedence
                ),
                "sentri_signed_180_seam_policy": (
                    "source phi=+180 supplies canonical azimuth -180 when "
                    "both phi=-180 and phi=+180 are present"
                ),
                "sentri_signed_180_precedence_used": bool(
                    used_signed_180_precedence
                ),
                "sentri_polarization_mapping": (
                    "VV=tt/theta-theta; HV=pt/phi-theta; "
                    "VH=tp/theta-phi; HH=pp/phi-phi"
                ),
                "sentri_phase_mapping": (
                    "GRIM complex amplitude = 10^(dBsm/20) "
                    "* exp(+j*deg2rad(reported_phase_deg))"
                ),
                "sentri_units_row_present": bool(has_units_row),
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def has_SENTRi_signature(cls, path):
        """Return whether a delimited file has a supported SENTRi header.

        A matching signature selects the SENTRi parser; malformed data raises a
        format error.
        """
        from GRIM_Backend.io.cst import _cst_compact_header


        with open(path, "r", newline="", encoding="utf-8-sig") as stream:
            sample = stream.read(8192)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = max((",", "\t", ";"), key=sample.count)
            rows = []
            for row_index, row in enumerate(csv.reader(stream, delimiter=delimiter)):
                rows.append(row)
                if row_index >= 255:
                    break
        compact_required = {
            "freqmhz",
            "thetadeg",
            "phideg",
            "rcsppdbsm",
            "efieldphaseppdeg",
            "rcsttdbsm",
            "efieldphasettdeg",
            "rcsptdbsm",
            "efieldphaseptdeg",
            "rcstpdbsm",
            "efieldphasetpdeg",
        }
        descriptive_required = {
            "frequency",
            "theta",
            "phi",
            "rcsphiscatphiinc",
            "phasephiphi",
            "rcsthetascatthetainc",
            "phasethetatheta",
            "rcsphiscatthetainc",
            "phasephitheta",
            "rcsthetascatphiinc",
            "phasethetaphi",
        }
        for row in rows:
            tokens = {_cst_compact_header(cell) for cell in row}
            if compact_required.issubset(tokens) or descriptive_required.issubset(tokens):
                return True
            compact_family = {"freqmhz", "thetadeg", "phideg"}.issubset(tokens) and any(
                token.startswith(("rcspp", "rcstt", "rcspt", "rcstp"))
                or token.startswith("efieldphase")
                for token in tokens
            )
            descriptive_family = {"frequency", "theta", "phi"}.issubset(tokens) and any(
                token.startswith((
                    "rcsphiscat", "rcsthetascat", "rcsphiinc", "rcsthetainc",
                ))
                for token in tokens
            )
            if compact_family or descriptive_family:
                return True
        return False
