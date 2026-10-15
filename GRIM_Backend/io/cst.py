"""CST far-field table parsing and RCS grid conversion."""
from __future__ import annotations

import csv
import itertools
import sys
import os
import re

import numpy as np

from GRIM_Backend.io.samples import construct_parsed_grid, parsed_sample_memory


class _DelimitedRows:
    """Repeatable CSV rows with bounded text memory and stable source checks."""

    def __init__(self, path):
        self.path = path
        self.signature = self._signature()
        with open(path, "r", newline="", encoding="utf-8-sig") as stream:
            sample = stream.read(8192)
        try:
            self.delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            self.delimiter = max((",", "\t", ";"), key=sample.count)

    def _signature(self):
        stat = os.stat(self.path)
        return stat.st_size, stat.st_mtime_ns

    def check_unchanged(self):
        if self._signature() != self.signature:
            raise ValueError("table changed during import; reload the completed file")

    def __iter__(self):
        self.check_unchanged()
        with open(self.path, "r", newline="", encoding="utf-8-sig") as stream:
            yield from csv.reader(stream, delimiter=self.delimiter)
        self.check_unchanged()

    def __getitem__(self, selection):
        if isinstance(selection, slice):
            return itertools.islice(iter(self), selection.start or 0,
                                    selection.stop, selection.step or 1)
        return next(itertools.islice(iter(self), selection, selection + 1))

    def __bool__(self):
        return next(iter(self), None) is not None


def _read_cst_delimited_rows(path):
    """Compatibility materializer; production imports use repeatable rows."""
    return list(_DelimitedRows(path))


def _vendor_import_allocation(shape, *, source, max_output_bytes,
                              extra_working_bytes=0, sample_dtype=np.float64,
                              duplicate_bytes_per_cell=8):
    """Include typed duplicate tracking and sample canonicalization in planning."""
    from GRIM_Backend.datasets.memory import _checked_dense_import_allocation

    dense_bytes, peak_bytes = parsed_sample_memory(
        shape, sample_dtype=sample_dtype,
        duplicate_bytes_per_cell=duplicate_bytes_per_cell,
        extra_working_bytes=extra_working_bytes,
    )
    resident = peak_bytes - dense_bytes
    return _checked_dense_import_allocation(
        shape, (sample_dtype, sample_dtype), source=source,
        max_output_bytes=max_output_bytes, resident_bytes=resident,
    )


def _vendor_estimate(allocation, source_format):
    return {"shape": allocation["shape"],
            "retained_bytes": allocation["dense_bytes"],
            "estimated_peak_bytes": allocation["peak_bytes"],
            "source_format": source_format}


def estimate_vendor_text_import(path):
    """Plan recognized SENTRi/CST files without retaining rows or sample arrays.

    Return None for other table formats. Unsupported or malformed vendor tables
    keep their normal format errors instead of using a misleading file-size cap.
    """
    from GRIM_Backend.datasets.grid import RcsGrid

    if RcsGrid.has_SENTRi_signature(path):
        return RcsGrid.read_SENTRi(path, max_output_bytes=sys.maxsize,
                                  _estimate_only=True)
    recognized = str(path).lower().endswith(".cst_data")
    if not recognized:
        for row in itertools.islice(_DelimitedRows(path), 256):
            tokens = {_cst_compact_header(value) for value in row}
            frequency = any(_cst_frequency_unit(value) for value in row)
            spherical = any("theta" in token for token in tokens) and any(
                "phi" in token for token in tokens)
            flat = any(token.startswith("elevation") for token in tokens) and any(
                token.startswith("azimuth") for token in tokens)
            physics = any(token in {"iq", "complexiq", "complexsample", "complexamplitude"}
                          or (("rcs" in token or "magnitude" in token)
                              and ("dbsm" in token or "dbm2" in token))
                          for token in tokens)
            if frequency and physics and (spherical or flat):
                recognized = True
                break
    if not recognized and not str(path).lower().endswith(".csv"):
        legacy_required = {"thetadeg", "phideg", "absthetadbm2", "phasethetadeg",
                           "absphidbm2", "phasephideg"}
        with open(path, "r", encoding="utf-8-sig") as stream:
            for line in itertools.islice(stream, 256):
                tokens = {_cst_compact_header(token).replace("dbsm", "dbm2")
                          for token in re.split(r"[,\s]+", line.strip())}
                if legacy_required.issubset(tokens):
                    return RcsGrid.load_theta_phi_txt(
                        path, max_output_bytes=sys.maxsize, _estimate_only=True)
    if not recognized:
        return None
    return RcsGrid.read_CST(path, max_output_bytes=sys.maxsize,
                            _estimate_only=True)


def _cst_compact_header(value):
    """Normalize CST/MATLAB table headings without losing their unit text."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _cst_frequency_unit(value):
    """Return the supported CST frequency unit matching the header."""

    compact = _cst_compact_header(value)
    for prefix in ("frequency", "freq"):
        if compact.startswith(prefix):
            suffix = compact[len(prefix):]
            return suffix if suffix in {"hz", "khz", "mhz", "ghz"} else None
    return None


def _cst_frequency_scale_to_ghz(value):
    unit = _cst_frequency_unit(value)
    scales = {"hz": 1.0e-9, "khz": 1.0e-6, "mhz": 1.0e-3, "ghz": 1.0}
    if unit is None:
        raise ValueError(
            "CST frequency header must explicitly end in exactly Hz, kHz, "
            "MHz, or GHz; other prefixes and unit guessing are unsupported"
        )
    return scales[unit], unit


def _wrap_cst_azimuth_deg(value):
    """Use GRIM's canonical half-open azimuth interval, [-180, 180)."""

    wrapped = float(np.mod(float(value) + 180.0, 360.0) - 180.0)
    return 0.0 if abs(wrapped) < 1.0e-12 else wrapped


def _parse_cst_iq(value):
    """Parse common Python/MATLAB spellings of one complex IQ sample."""

    text = str(value or "").strip()
    if not text:
        return None
    token = text.replace(" ", "").strip("()[]{}")
    token = token.replace("*", "").replace("I", "j").replace("i", "j")
    token = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", token)
    try:
        result = complex(token)
    except ValueError as exc:
        raise ValueError(f"unsupported IQ value {text!r}") from exc
    if not (np.isfinite(result.real) and np.isfinite(result.imag)):
        raise ValueError(f"IQ value must be finite, got {text!r}")
    return result


def _cst_dbsm_to_power(value, *, context="CST magnitude"):
    """Convert dBsm to finite float64 power with an actionable overflow error."""

    value = float(value)
    if np.isneginf(value):
        return 0.0
    if not np.isfinite(value):
        raise ValueError(f"{context} must be finite or -Inf, got {value!r}")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = float(np.power(10.0, value / 10.0))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            f"{context}={value:g} dBsm overflows finite linear power"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{context}={value:g} dBsm does not produce finite linear power"
        )
    return result


def _cst_iq_to_power(value, *, context="CST IQ"):
    """Return finite |IQ|^2 without allowing float64 overflow."""

    amplitude = float(abs(value))
    if amplitude > float(np.sqrt(np.finfo(np.float64).max)):
        raise ValueError(f"{context} magnitude overflows finite linear power")
    result = amplitude * amplitude
    if not np.isfinite(result):
        raise ValueError(f"{context} does not produce finite linear power")
    return result


class CstFormatMixin:
    """CST and theta/phi table readers for RcsGrid."""

    @classmethod
    def read_CST(cls, path, *, max_output_bytes=None, _estimate_only=False):
        """Read a supported CST RCS table into a physically tagged grid.

        Two schemas are recognized:

        * CST's wide spherical table (frequency/theta/phi and one magnitude /
          phase pair per spherical polarization component).
        * The ``.cst_data`` flat table
          (elevation/azimuth/frequency/polarity/magnitude/phase/IQ).

        Standard CST theta is a colatitude, so the wide form is converted to
        GRIM elevation with ``elevation = 90 - theta``.  Both forms use GRIM's
        canonical azimuth interval ``[-180, 180)``.
        """
        from GRIM_Backend.io.cst import _cst_compact_header, _cst_frequency_unit

        rows = _DelimitedRows(path)
        if not rows:
            raise ValueError("CST table is empty")

        def _flat_key(cell_value):
            compact = _cst_compact_header(cell_value)
            if compact.startswith("elevation") and (
                "deg" in compact or "degree" in compact
            ):
                return "elevation"
            if compact.startswith("azimuth") and (
                "deg" in compact or "degree" in compact
            ):
                return "azimuth"
            if compact in {"pol", "polarity", "polarization"}:
                return "polarization"
            if compact in {"iq", "complexiq", "complexsample", "complexamplitude"}:
                return "iq"
            if _cst_frequency_unit(cell_value) is not None:
                return "frequency"
            if "magnitude" in compact and (
                "dbsm" in compact or "dbm2" in compact
            ):
                return "magnitude_dbsm"
            if compact.startswith("rcs") and (
                "dbsm" in compact or "dbm2" in compact
            ):
                return "magnitude_dbsm"
            if "phase" in compact and (
                "deg" in compact or "degree" in compact
            ):
                return "phase_deg"
            return None

        required = {"elevation", "azimuth", "frequency", "polarization"}
        for header_idx, row in enumerate(rows):
            mapped = {}
            tokens = {}
            for column_idx, cell in enumerate(row):
                key = _flat_key(cell)
                if key is not None and key not in mapped:
                    mapped[key] = column_idx
                    tokens[key] = str(cell)
            if required.issubset(mapped) and (
                "magnitude_dbsm" in mapped or "iq" in mapped
            ):
                return cls._read_cst_flat_rows(
                    path,
                    rows,
                    header_idx,
                    mapped,
                    tokens,
                    max_output_bytes=max_output_bytes,
                    _estimate_only=_estimate_only,
                )
            # A spherical header selects the wide parser immediately instead
            # of searching every numeric data row for a later flat header.
            compact_tokens = [_cst_compact_header(cell) for cell in row]
            if (any(_cst_frequency_unit(cell) for cell in row)
                    and any(token.startswith("theta") for token in compact_tokens)
                    and any(token.startswith("phi") for token in compact_tokens)
                    and any("rcs" in token for token in compact_tokens)):
                break

        if str(path).lower().endswith(".cst_data"):
            raise ValueError(
                "Could not find the .cst_data header. Need elevation, azimuth, "
                "frequency, polarity, and magnitude(dBsm) and/or IQ columns."
            )
        return cls._read_cst_theta_phi_csv(
            path, rows=rows, max_output_bytes=max_output_bytes,
            _estimate_only=_estimate_only,
        )

    @classmethod
    def load_theta_phi_csv(cls, path, *, max_output_bytes=None):
        """Compatibility name for :meth:`read_CST`."""

        return cls.read_CST(path, max_output_bytes=max_output_bytes)

    @classmethod
    def _read_cst_flat_rows(
        cls,
        path,
        rows,
        header_idx,
        col_idx,
        header_tokens,
        *,
        max_output_bytes=None,
        _estimate_only=False,
    ):
        """Parse the row-per-polarization ``.cst_data`` schema."""
        from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.cst import _cst_dbsm_to_power, _cst_frequency_scale_to_ghz, _cst_iq_to_power, _parse_cst_iq, _wrap_cst_azimuth_deg
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        def _cell(row, key):
            idx = col_idx.get(key, -1)
            if idx < 0 or idx >= len(row):
                return ""
            return str(row[idx]).strip()

        def _required_float(row, key, line_no):
            text = _cell(row, key)
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"line {line_no}: {key} must be finite")
            return float(value)

        def _optional_float(row, key, line_no):
            text = _cell(row, key)
            if not text:
                return None
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_no}: invalid {key} value {text!r}"
                ) from exc
            if np.isnan(value):
                return None
            if np.isposinf(value):
                raise ValueError(f"line {line_no}: {key} cannot be +Inf")
            return float(value)

        frequency_scale, _ = _cst_frequency_scale_to_ghz(
            header_tokens.get("frequency", "")
        )
        counters = dict(iq_validated=0, iq_only=0, iq_unparsed=0)

        def _records(*, coordinates_only=False):
            for row_idx, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
                if not row or all(not str(value).strip() for value in row):
                    continue

                elevation = _required_float(row, "elevation", row_idx)
                azimuth = _wrap_cst_azimuth_deg(
                    _required_float(row, "azimuth", row_idx)
                )
                raw_frequency = _required_float(row, "frequency", row_idx)
                if raw_frequency <= 0.0:
                    raise ValueError(f"line {row_idx}: frequency must be positive")
                polarization = _cell(row, "polarization").upper()
                if not polarization:
                    raise ValueError(f"line {row_idx}: polarity is blank")
                frequency = float(raw_frequency * frequency_scale)
                if not np.isfinite(frequency) or frequency <= 0.0:
                    raise ValueError(
                        f"line {row_idx}: frequency conversion to GHz did not "
                        "produce a positive finite value"
                    )
                if coordinates_only:
                    yield row_idx, azimuth, elevation, frequency, polarization, None, None
                    continue

                magnitude_dbsm = _optional_float(
                    row, "magnitude_dbsm", row_idx
                ) if "magnitude_dbsm" in col_idx else None
                phase_deg = _optional_float(
                    row, "phase_deg", row_idx
                ) if "phase_deg" in col_idx else None

                iq_text = _cell(row, "iq") if "iq" in col_idx else ""
                iq_value = None
                if iq_text:
                    try:
                        iq_value = _parse_cst_iq(iq_text)
                    except ValueError as exc:
                        if magnitude_dbsm is None:
                            raise ValueError(f"line {row_idx}: {exc}") from exc
                        counters["iq_unparsed"] += 1

                if magnitude_dbsm is None and iq_value is None:
                    raise ValueError(
                        f"line {row_idx}: need a finite magnitude(dBsm) or parsable IQ sample"
                    )

                if magnitude_dbsm is None:
                    power = _cst_iq_to_power(
                        iq_value, context=f"line {row_idx} IQ"
                    )
                else:
                    power = _cst_dbsm_to_power(
                        magnitude_dbsm, context=f"line {row_idx} Magnitude(dBsm)"
                    )

                if phase_deg is not None and not np.isfinite(phase_deg):
                    raise ValueError(f"line {row_idx}: phase_deg must be finite")
                phase = (
                    float(np.deg2rad(phase_deg))
                    if phase_deg is not None
                    else float(np.angle(iq_value)) if iq_value is not None
                    else float("nan")
                )

                if iq_value is not None and magnitude_dbsm is not None:
                    iq_power = _cst_iq_to_power(
                        iq_value, context=f"line {row_idx} IQ"
                    )
                    if power == 0.0:
                        magnitude_matches = iq_power <= 1.0e-20
                    elif iq_power == 0.0:
                        magnitude_matches = False
                    else:
                        iq_dbsm = 10.0 * np.log10(iq_power)
                        magnitude_matches = abs(iq_dbsm - magnitude_dbsm) <= 0.05
                    if not magnitude_matches:
                        raise ValueError(
                            f"line {row_idx}: IQ magnitude disagrees with "
                            "Magnitude(dBsm) by more than 0.05 dB"
                        )

                if (
                    iq_value is not None
                    and phase_deg is not None
                    and abs(iq_value) > 1.0e-15
                ):
                    phase_error = np.angle(np.exp(1j * (np.angle(iq_value) - phase)))
                    if abs(float(np.rad2deg(phase_error))) > 0.5:
                        raise ValueError(
                            f"line {row_idx}: IQ phase disagrees with Phase(deg) "
                            "by more than 0.5 deg"
                        )

                if iq_value is not None:


                    power = _cst_iq_to_power(
                        iq_value, context=f"line {row_idx} IQ"
                    )
                    phase = float(np.angle(iq_value))
                    if magnitude_dbsm is None and phase_deg is None:
                        counters["iq_only"] += 1
                    else:
                        counters["iq_validated"] += 1
                yield row_idx, azimuth, elevation, frequency, polarization, power, phase

        az_values, el_values, freq_values = set(), set(), set()
        pol_order = []
        for _, azimuth, elevation, frequency, polarization, _, _ in _records(coordinates_only=True):
            az_values.add(azimuth)
            el_values.add(elevation)
            freq_values.add(frequency)
            if polarization not in pol_order:
                pol_order.append(polarization)
        if not az_values:
            raise ValueError("CST flat table contains no data rows")
        azimuths = np.asarray(sorted(az_values), dtype=float)
        elevations = np.asarray(sorted(el_values), dtype=float)
        frequencies = np.asarray(sorted(freq_values), dtype=float)
        polarizations = np.asarray(pol_order, dtype=object)
        del az_values, el_values, freq_values
        shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
        allocation = _vendor_import_allocation(
            shape, source=f"CST flat import {path}", max_output_bytes=max_output_bytes,
        )
        if _estimate_only:
            return _vendor_estimate(allocation, "CST flat cst_data")
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        first_line = np.zeros(shape, dtype=np.uint64)
        az_index = {value: index for index, value in enumerate(azimuths.tolist())}
        el_index = {value: index for index, value in enumerate(elevations.tolist())}
        freq_index = {value: index for index, value in enumerate(frequencies.tolist())}
        pol_index = {str(value): index for index, value in enumerate(polarizations.tolist())}
        counters = dict(iq_validated=0, iq_only=0, iq_unparsed=0)
        for row_idx, azimuth, elevation, frequency, polarization, sample_power, sample_phase in _records():
            index = (az_index[azimuth], el_index[elevation], freq_index[frequency],
                     pol_index[polarization])
            if first_line[index]:
                if _cst_samples_equivalent(power[index], phase[index], sample_power, sample_phase):
                    continue
                raise ValueError(
                    f"line {row_idx}: conflicting duplicate CST sample after "
                    f"azimuth wrapping; first defined on line {int(first_line[index])}"
                )
            first_line[index] = row_idx
            power[index], phase[index] = sample_power, sample_phase
        del first_line
        iq_validated, iq_only, iq_unparsed = (
            counters["iq_validated"], counters["iq_only"], counters["iq_unparsed"]
        )

        iq_summary = (
            f"IQ validated={iq_validated}, IQ-only={iq_only}, "
            f"IQ-unparsed fallback={iq_unparsed}"
        )
        return construct_parsed_grid(cls,
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=(
                f"Loaded CST flat cst_data; explicit elevation; azimuth wrapped "
                f"to [-180, 180); {iq_summary}: {path}"
            ),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra={
                "source_format": "CST flat cst_data",
                "cst_angle_mapping": (
                    "explicit elevation; azimuth wrapped to [-180, 180)"
                ),
                "cst_polarization_mapping": "labels supplied by Polarity column",
                "cst_iq_rows_validated": iq_validated,
                "cst_iq_only_rows": iq_only,
                "cst_iq_unparsed_fallback_rows": iq_unparsed,
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def _read_cst_theta_phi_csv(
        cls, path, *, rows=None, max_output_bytes=None, _estimate_only=False
    ):
        """Load a theta/phi scattering CSV into an RcsGrid.

        Expected layout:
            - Two header rows total (or any leading metadata rows), with one row
              containing column names like:
              frequency(hz), theta(deg), phi(deg),
              rcs theta-theta(dbsm), rcs phi-theta(dbsm),
              rcs theta-phi(dbsm), rcs phi-phi,
              phase theta-theta(...), phase phi-theta(...),
              phase theta-phi(...), phase phi-phi(...)

        Conventions applied:
            - phi(deg), wrapped to [-180, 180), -> azimuth axis
            - standard CST theta colatitude -> elevation = 90 - theta
            - theta -> V, phi -> H
              rcs theta-theta -> VV
              rcs phi-theta   -> HV
              rcs theta-phi   -> VH
              rcs phi-phi     -> HH
            - RCS columns are interpreted as dBsm and converted to linear power.
            - Phase columns are interpreted as degrees and converted to radians.
        """
        from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.cst import _cst_dbsm_to_power, _cst_frequency_scale_to_ghz, _wrap_cst_azimuth_deg
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        def _norm(text: str) -> str:
            s = str(text).strip().lower()
            for ch in (" ", "_", "\t"):
                s = s.replace(ch, "")
            return s

        def _infer_freq_scale_to_ghz(freq_header_token: str) -> tuple[float, str]:
            scale, unit = _cst_frequency_scale_to_ghz(freq_header_token)
            labels = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
            return scale, labels[unit]

        alias_to_key = {
            "frequency(hz)": "frequency",
            "frequencyhz": "frequency",
            "frequency(ghz)": "frequency",
            "frequencyghz": "frequency",
            "frequency(mhz)": "frequency",
            "frequencymhz": "frequency",
            "frequency(khz)": "frequency",
            "frequencykhz": "frequency",
            "frequency": "frequency",
            "theta(deg)": "theta_deg",
            "phi(deg)": "phi_deg",
            "rcstheta-theta(dbsm)": "rcs_vv_dbsm",
            "rcstheta-thetadbsm": "rcs_vv_dbsm",
            "rcstheta-theta(dbm^2)": "rcs_vv_dbsm",
            "rcstheta-thetadbm2": "rcs_vv_dbsm",
            "rcsphi-theta(dbsm)": "rcs_hv_dbsm",
            "rcsphi-thetadbsm": "rcs_hv_dbsm",
            "rcsphi-theta(dbm^2)": "rcs_hv_dbsm",
            "rcsphi-thetadbm2": "rcs_hv_dbsm",
            "rcstheta-phi(dbsm)": "rcs_vh_dbsm",
            "rcstheta-phidbsm": "rcs_vh_dbsm",
            "rcstheta-phi(dbm^2)": "rcs_vh_dbsm",
            "rcstheta-phidbm2": "rcs_vh_dbsm",
            "rcsphi-phi(dbsm)": "rcs_hh_dbsm",
            "rcsphi-phidbsm": "rcs_hh_dbsm",
            "rcsphi-phi(dbm^2)": "rcs_hh_dbsm",
            "rcsphi-phidbm2": "rcs_hh_dbsm",
            "phasetheta-theta(deg)": "phase_vv_deg",
            "phasephi-theta(deg)": "phase_hv_deg",
            "phasetheta-phi(deg)": "phase_vh_deg",
            "phasephi-phi(deg)": "phase_hh_deg",
        }

        if rows is None:
            rows = _DelimitedRows(path)
        if not rows:
            raise ValueError("CST theta/phi table is empty")

        def _classify_fuzzy_header(cell_value: str) -> str | None:
            raw = str(cell_value or "").strip().lower()
            if raw == "":
                return None

            key = alias_to_key.get(_norm(raw))
            if key is not None:
                return key

            compact = re.sub(r"[^a-z0-9]+", "", raw)
            if compact in {"f", "freq"} or "frequency" in compact:
                return "frequency"
            if (
                "theta" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
                and ("deg" in compact or "degree" in compact)
            ):
                return "theta_deg"
            if (
                "phi" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
                and ("deg" in compact or "degree" in compact)
            ):
                return "phi_deg"

            has_phase = "phase" in compact and (
                "deg" in compact or "degree" in compact
            )
            has_explicit_rcs_quantity = (
                "rcs" in compact
                or "radarcrosssection" in compact
                or "sigma" in compact
            )
            has_explicit_rcs_unit = "dbsm" in compact or "dbm2" in compact
            has_mag = (
                has_explicit_rcs_quantity
                and has_explicit_rcs_unit
                and not has_phase
            )
            if not has_phase and not has_mag:
                return None

            pair_key: str | None = None
            theta_count = len(re.findall("theta", raw))
            phi_count = len(re.findall("phi", raw))
            if "phi-theta" in raw or re.search(r"phi[^a-z0-9]+theta", raw):
                pair_key = "hv"
            elif "theta-phi" in raw or re.search(r"theta[^a-z0-9]+phi", raw):
                pair_key = "vh"
            elif theta_count >= 2:
                pair_key = "vv"
            elif phi_count >= 2:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 0:
                pair_key = "vv"
            elif phi_count == 1 and theta_count == 0:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 1:
                pair_key = "hv" if raw.find("phi") < raw.find("theta") else "vh"

            if pair_key is None:
                return None
            if has_phase:
                return f"phase_{pair_key}_deg"
            return f"rcs_{pair_key}_dbsm"

        header_idx = None
        data_start_idx = 0
        col_idx: dict[str, int] = {}
        header_tokens: dict[str, str] = {}
        required_axes = {"frequency", "theta_deg", "phi_deg"}
        for i, row in enumerate(rows):
            mapped: dict[str, int] = {}
            mapped_tokens: dict[str, str] = {}
            ambiguous_physics_headers: list[str] = []
            for j, cell in enumerate(row):
                key = _classify_fuzzy_header(cell)
                if key is not None and key not in mapped:
                    mapped[key] = j
                    mapped_tokens[key] = str(cell)
                    continue
                compact = re.sub(
                    r"[^a-z0-9]+", "", str(cell or "").strip().lower()
                )
                mentions_basis = "theta" in compact or "phi" in compact
                mentions_rcs = (
                    "rcs" in compact
                    or "radarcrosssection" in compact
                    or "sigma" in compact
                )
                if mentions_basis and ("phase" in compact or mentions_rcs):
                    ambiguous_physics_headers.append(str(cell))
            has_any_rcs = any(k.startswith("rcs_") for k in mapped.keys())
            if required_axes.issubset(mapped.keys()) and has_any_rcs:
                if ambiguous_physics_headers:
                    raise ValueError(
                        "Ambiguous CST wide-table physics header(s): "
                        + ", ".join(repr(value) for value in ambiguous_physics_headers)
                        + ". RCS magnitudes must state dBsm/dBm^2 and phases "
                        "must state degrees."
                    )
                header_idx = i
                data_start_idx = i + 1
                col_idx = mapped
                header_tokens = mapped_tokens
                break

        if header_idx is None:
            raise ValueError(
                "Could not find an explicit CST RCS header. Need frequency "
                "with units, theta/phi axes, and at least one RCS magnitude "
                "column explicitly labeled dBsm or dBm^2. Headerless/order-"
                "guessed and generic Abs(field) tables are not accepted."
            )

        def _records():
            for row_index, row in enumerate(rows[data_start_idx:], start=data_start_idx):
                line_no = row_index + 1
                if not row or all(str(cell).strip() == "" for cell in row):
                    continue

                def _axis_cell(key: str) -> float:
                    idx = col_idx[key]
                    raw = row[idx] if idx < len(row) else ""
                    text = str(raw).strip()
                    if not text:
                        raise ValueError(f"line {line_no}: {key} is blank")
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(
                            f"line {line_no}: invalid {key} value {text!r}"
                        ) from exc
                    if not np.isfinite(value):
                        raise ValueError(f"line {line_no}: {key} must be finite")
                    return value

                f_hz = _axis_cell("frequency")
                if f_hz <= 0.0:
                    raise ValueError(f"line {line_no}: frequency must be positive")
                theta_deg = _axis_cell("theta_deg")
                phi_deg = _axis_cell("phi_deg")

                def _cell(key: str) -> float:
                    idx = col_idx.get(key, -1)
                    if idx < 0 or idx >= len(row):
                        return float("nan")
                    text = str(row[idx]).strip()
                    if not text:
                        return float("nan")
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(
                            f"line {line_no}: invalid {key} value {text!r}"
                        ) from exc
                    if key.startswith("phase_") and not np.isfinite(value):
                        raise ValueError(f"line {line_no}: {key} must be finite")
                    if key.startswith("rcs_") and not (
                        np.isfinite(value) or np.isneginf(value)
                    ):
                        raise ValueError(
                            f"line {line_no}: {key} must be finite or -Inf"
                        )
                    return value

                yield (
                        float(f_hz),
                        float(theta_deg),
                        float(phi_deg),
                        _cell("rcs_vv_dbsm"),
                        _cell("rcs_hv_dbsm"),
                        _cell("rcs_vh_dbsm"),
                        _cell("rcs_hh_dbsm"),
                        _cell("phase_vv_deg"),
                        _cell("phase_hv_deg"),
                        _cell("phase_vh_deg"),
                        _cell("phase_hh_deg"),
                        int(line_no),
                    )


        freq_scale_to_ghz, _ = _infer_freq_scale_to_ghz(header_tokens.get("frequency", ""))
        channel_specs = (
            ("VV", 3, 7, "theta-theta"), ("HV", 4, 8, "phi-theta"),
            ("VH", 5, 9, "theta-phi"), ("HH", 6, 10, "phi-phi"),
        )

        def _normalized_records():
            for record in _records():
                f_ghz = float(record[0] * freq_scale_to_ghz)
                if not np.isfinite(f_ghz) or f_ghz <= 0.0:
                    raise ValueError(
                        f"line {int(record[11])}: frequency conversion to GHz "
                        "did not produce a positive finite value"
                    )
                theta_deg = float(record[1])
                if theta_deg < -1.0e-9 or theta_deg > 180.0 + 1.0e-9:
                    raise ValueError(
                        f"line {int(record[11])}: standard CST theta must be "
                        f"within [0, 180] deg, got {theta_deg:g}"
                    )
                yield (f_ghz, float(90.0 - theta_deg),
                       _wrap_cst_azimuth_deg(record[2]), int(record[11]), record)

        freq_values, el_values, az_values, present_channels = set(), set(), set(), set()
        data_rows = 0
        for f_ghz, elevation, azimuth, source_line, record in _normalized_records():
            data_rows += 1
            channels = [spec[0] for spec in channel_specs if not np.isnan(record[spec[1]])]
            if channels:
                freq_values.add(f_ghz)
                el_values.add(elevation)
                az_values.add(azimuth)
                present_channels.update(channels)
        if not data_rows:
            raise ValueError("CSV contains no data rows after the header")
        present_specs = [spec for spec in channel_specs if spec[0] in present_channels]
        if not present_specs:
            raise ValueError(
                "CST theta/phi table parsed, but no finite RCS magnitude values were found"
            )
        freqs = np.asarray(sorted(freq_values), dtype=float)
        elevs = np.asarray(sorted(el_values), dtype=float)
        azims = np.asarray(sorted(az_values), dtype=float)
        pols = np.asarray([spec[0] for spec in present_specs], dtype=object)
        del freq_values, el_values, az_values, present_channels
        shape = (len(azims), len(elevs), len(freqs), len(pols))
        allocation = _vendor_import_allocation(
            shape, source=f"CST wide import {path}", max_output_bytes=max_output_bytes,
        )
        if _estimate_only:
            return _vendor_estimate(allocation, "CST wide theta/phi table")
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        first_line = np.zeros(shape, dtype=np.uint64)
        f_idx = {float(value): index for index, value in enumerate(freqs.tolist())}
        el_idx = {float(value): index for index, value in enumerate(elevs.tolist())}
        az_idx = {float(value): index for index, value in enumerate(azims.tolist())}
        pol_idx = {str(value): index for index, value in enumerate(pols.tolist())}
        for f_ghz, elevation_deg, azimuth_deg, source_line_no, source_record in _normalized_records():
            if not any(not np.isnan(source_record[spec[1]]) for spec in present_specs):
                continue
            for pol_label, magnitude_index, phase_index, component_name in present_specs:
                magnitude = source_record[magnitude_index]
                if np.isnan(magnitude):
                    continue
                sample_power = _cst_dbsm_to_power(
                    magnitude, context=f"line {source_line_no} CST {component_name} magnitude"
                )
                sample_phase = float(np.deg2rad(source_record[phase_index]))
                index = (az_idx[azimuth_deg], el_idx[elevation_deg], f_idx[f_ghz], pol_idx[pol_label])
                if first_line[index]:
                    if _cst_samples_equivalent(power[index], phase[index], sample_power, sample_phase):
                        continue
                    raise ValueError(
                        f"line {source_line_no}: conflicting duplicate CST theta/phi sample after "
                        "coordinate conversion: "
                        f"az={azimuth_deg:g}, el={elevation_deg:g}, "
                        f"f={f_ghz:g} GHz, component={component_name}; "
                        f"first defined on line {int(first_line[index])}"
                    )
                first_line[index] = source_line_no
                power[index], phase[index] = sample_power, sample_phase
        del first_line

        return construct_parsed_grid(cls,
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=(
                "Loaded CST theta/phi table; standard theta converted with "
                f"elevation=90-theta; phi wrapped to [-180, 180): {path}"
            ),
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra={
                "source_format": "CST wide theta/phi table",
                "cst_angle_mapping": (
                    "elevation=90-theta; phi wrapped to [-180, 180)"
                ),
                "cst_polarization_mapping": (
                    "theta=V, phi=H; component pair mapped in written order"
                ),
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )

    @classmethod
    def load_theta_phi_txt(
        cls,
        path,
        *,
        frequency_ghz=None,
        max_output_bytes=None,
        _estimate_only=False,
    ):
        """Load whitespace-delimited theta/phi TXT format into an RcsGrid.

        Expected columns in an explicit unit-bearing header row:
            theta(deg), phi(deg), abs(rcs)(dbm^2), abs(theta)(dbm^2),
            phase(theta)(deg), abs(phi)(dbm^2), phase(phi)(deg), ax.ratio(db)

        Axis/polarization mapping:
            - theta(deg) -> azimuth
            - phi(deg)   -> elevation
            - theta -> V, phi -> H
              abs(theta), phase(theta) -> VV
              abs(phi),   phase(phi)   -> HH
            - abs(rcs) is loaded as a third polarization channel: TOTAL
        """
        from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.cst import _cst_dbsm_to_power
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        def _norm_token(text: str) -> str:
            normalized = (
                str(text).strip().lower().replace("²", "2").replace("°", "deg")
            )
            return re.sub(r"[^a-z0-9]+", "", normalized)

        def _frequency_from_filename_ghz(file_path: str) -> float | None:
            name = os.path.basename(str(file_path))
            match = re.search(
                r"(?:^|[^a-z0-9])f\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-z]+)?",
                name,
                flags=re.IGNORECASE,
            )
            if match is None:
                return None
            try:
                raw_value = float(match.group(1))
            except (TypeError, ValueError):
                return None
            if not np.isfinite(raw_value):
                return None

            unit = (match.group(2) or "").strip().lower()
            if unit == "ghz":
                scale = 1.0
            elif unit == "mhz":
                scale = 1.0e-3
            elif unit == "khz":
                scale = 1.0e-6
            elif unit == "hz":
                scale = 1.0e-9
            else:
                return None
            converted = float(raw_value * scale)
            return converted if np.isfinite(converted) and converted > 0.0 else None

        alias_to_key = {
            "thetadeg": "theta_deg",
            "phideg": "phi_deg",
            "absrcsdbm2": "abs_rcs_dbm2",
            "absrcsdbsm": "abs_rcs_dbm2",
            "absthetadbm2": "abs_theta_dbm2",
            "absthetadbsm": "abs_theta_dbm2",
            "phasethetadeg": "phase_theta_deg",
            "absphidbm2": "abs_phi_dbm2",
            "absphidbsm": "abs_phi_dbm2",
            "phasephideg": "phase_phi_deg",
            "axratiodb": "ax_ratio_db",
        }

        header_idx = None
        col_idx: dict[str, int] = {}
        required = {
            "theta_deg",
            "phi_deg",
            "abs_theta_dbm2",
            "phase_theta_deg",
            "abs_phi_dbm2",
            "phase_phi_deg",
        }

        def _tokenize(text: str) -> list[str]:
            return [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]

        saw_line = False


        with open(path, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                saw_line = True
                tokens = _tokenize(line)
                mapped: dict[str, int] = {}
                for j, token in enumerate(tokens):
                    key = alias_to_key.get(_norm_token(token))
                    if key is not None and key not in mapped:
                        mapped[key] = j
                if required.issubset(mapped.keys()):
                    header_idx = i
                    col_idx = mapped
                    break

        if not saw_line:
            raise ValueError("TXT is empty")

        if header_idx is None:
            raise ValueError(
                "Could not parse legacy theta/phi TXT: an explicit header with "
                "theta(deg), phi(deg), abs(theta)(dBm^2), phase(theta)(deg), "
                "abs(phi)(dBm^2), and phase(phi)(deg) is required. Headerless "
                "column-order guessing is not physically safe."
            )

        inferred_frequency = _frequency_from_filename_ghz(path)
        if frequency_ghz is None:
            if inferred_frequency is None:
                raise ValueError(
                    "legacy theta/phi TXT requires an explicit frequency with "
                    "unit in the filename (for example 'f=10GHz') or the "
                    "frequency_ghz= loader argument"
                )
            selected_frequency = inferred_frequency
            frequency_source = "unit-qualified filename"
        else:
            if isinstance(frequency_ghz, (bool, np.bool_)):
                raise ValueError("frequency_ghz must be a positive finite value")
            try:
                selected_frequency = float(frequency_ghz)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("frequency_ghz must be a positive finite value") from exc
            if not np.isfinite(selected_frequency) or selected_frequency <= 0.0:
                raise ValueError("frequency_ghz must be a positive finite value")
            if inferred_frequency is not None and not np.isclose(
                selected_frequency,
                inferred_frequency,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(
                    f"frequency_ghz={selected_frequency:g} conflicts with the "
                    f"unit-qualified filename value {inferred_frequency:g} GHz"
                )
            frequency_source = "explicit frequency_ghz argument"

        def _required_value(
            tokens,
            line_index,
            key: str,
            *,
            allow_negative_infinity=False,
        ) -> float:
            idx = col_idx.get(key, -1)
            if idx < 0 or idx >= len(tokens):
                raise ValueError(f"line {line_index}: {key} is missing")
            text = str(tokens[idx]).strip()
            if not text:
                raise ValueError(f"line {line_index}: {key} is blank")
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"line {line_index}: invalid {key} value {text!r}"
                ) from exc
            if not np.isfinite(value) and not (
                allow_negative_infinity and np.isneginf(value)
            ):
                expected = "finite or -Inf" if allow_negative_infinity else "finite"
                raise ValueError(
                    f"line {line_index}: {key} must be {expected}"
                )
            return float(value)

        def _optional_value(
            tokens,
            line_index,
            key: str,
            *,
            allow_negative_infinity=False,
        ):
            idx = col_idx.get(key, -1)
            if idx < 0 or idx >= len(tokens) or not str(tokens[idx]).strip():
                return float("nan")
            return _required_value(
                tokens,
                line_index,
                key,
                allow_negative_infinity=allow_negative_infinity,
            )

        def _iter_data_tokens():
            with open(path, "r", encoding="utf-8-sig") as stream:
                for line_index, line in enumerate(stream, start=1):
                    if line_index <= header_idx + 1:
                        continue
                    stripped = line.strip()
                    if not stripped or stripped.startswith(("#", "!", "%")):
                        continue
                    yield line_index, _tokenize(line)

        def _parse_data_tokens(tokens, line_index):
            theta_deg = _required_value(tokens, line_index, "theta_deg")
            phi_deg = _required_value(tokens, line_index, "phi_deg")
            abs_theta_db = _required_value(
                tokens,
                line_index,
                "abs_theta_dbm2",
                allow_negative_infinity=True,
            )
            phase_theta_deg = _required_value(
                tokens, line_index, "phase_theta_deg"
            )
            abs_phi_db = _required_value(
                tokens,
                line_index,
                "abs_phi_dbm2",
                allow_negative_infinity=True,
            )
            phase_phi_deg = _required_value(
                tokens, line_index, "phase_phi_deg"
            )
            abs_rcs_db = _optional_value(
                tokens,
                line_index,
                "abs_rcs_dbm2",
                allow_negative_infinity=True,
            )


            _optional_value(tokens, line_index, "ax_ratio_db")
            sample_values = (
                (
                    "VV",
                    _cst_dbsm_to_power(
                        abs_theta_db, context=f"line {line_index} abs(theta)"
                    ),
                    float(np.deg2rad(phase_theta_deg)),
                ),
                (
                    "HH",
                    _cst_dbsm_to_power(
                        abs_phi_db, context=f"line {line_index} abs(phi)"
                    ),
                    float(np.deg2rad(phase_phi_deg)),
                ),
                (
                    "TOTAL",
                    (
                        _cst_dbsm_to_power(
                            abs_rcs_db, context=f"line {line_index} abs(rcs)"
                        )
                        if not np.isnan(abs_rcs_db)
                        else float("nan")
                    ),
                    float("nan"),
                ),
            )
            return float(theta_deg), float(phi_deg), sample_values


        azimuth_values = set()
        elevation_values = set()
        data_row_count = 0
        for line_index, tokens in _iter_data_tokens():
            theta_deg, phi_deg, _ = _parse_data_tokens(tokens, line_index)
            data_row_count += 1
            azimuth_values.add(theta_deg)
            elevation_values.add(phi_deg)

        if data_row_count == 0:
            raise ValueError("TXT contains no data rows after header")

        azims = np.asarray(sorted(azimuth_values), dtype=float)
        elevs = np.asarray(sorted(elevation_values), dtype=float)
        freqs = np.asarray([float(selected_frequency)], dtype=float)
        pols = np.asarray(["VV", "HH", "TOTAL"], dtype=str)
        del azimuth_values, elevation_values

        shape = (len(azims), len(elevs), 1, len(pols))
        allocation = _vendor_import_allocation(
            shape, source=f"legacy theta/phi TXT import {path}",
            max_output_bytes=max_output_bytes, sample_dtype=np.float32,
            duplicate_bytes_per_cell=24,
        )
        if _estimate_only:
            return _vendor_estimate(allocation, "legacy theta/phi TXT")
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)
        # Keep first-row values at original precision for duplicate comparisons.
        # Output stays float32 and keeps the last equivalent row, as before.
        first_power = np.full(shape, np.nan, dtype=np.float64)
        first_phase = np.full(shape, np.nan, dtype=np.float64)
        first_line = np.zeros(shape, dtype=np.uint64)
        el_idx = {float(v): i for i, v in enumerate(elevs.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azims.tolist())}
        pol_idx = {str(v): i for i, v in enumerate(pols.tolist())}


        for line_index, tokens in _iter_data_tokens():
            theta_deg, phi_deg, sample_values = _parse_data_tokens(
                tokens, line_index
            )
            ai = az_idx[theta_deg]
            ei = el_idx[phi_deg]
            for polarization, sample_power, sample_phase in sample_values:
                if not np.isfinite(sample_power):
                    continue
                pi = pol_idx[polarization]
                index = (ai, ei, 0, pi)
                if first_line[index]:
                    if not _cst_samples_equivalent(
                        first_power[index], first_phase[index], sample_power, sample_phase
                    ):
                        raise ValueError(
                            f"line {line_index}: conflicting duplicate legacy "
                            f"TXT sample at theta={theta_deg:g}, phi={phi_deg:g}, "
                            f"polarization={polarization}; first defined on line "
                            f"{int(first_line[index])}"
                        )
                else:
                    first_line[index] = line_index
                    first_power[index], first_phase[index] = sample_power, sample_phase
                power[index], phase[index] = sample_power, sample_phase
        del first_line, first_power, first_phase

        if not np.isfinite(power).any():
            raise ValueError("TXT parsed, but no finite magnitude values were found")

        return construct_parsed_grid(cls,
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded theta/phi TXT: {path}",
            units={
                "azimuth": "deg", "elevation": "deg", "frequency": "GHz",
                "rcs_log_unit": "dBsm", "rcs_linear_quantity": "sigma_3d",
            },
            extra={
                "source_format": "legacy theta/phi TXT",
                "legacy_txt_frequency_source": frequency_source,
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )
