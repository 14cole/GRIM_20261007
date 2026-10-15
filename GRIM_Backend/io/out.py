"""OUT table loading into frequency and azimuth RCS grids."""
from __future__ import annotations

from array import array
import math
import os
import re

import numpy as np


class OutFormatMixin:
    """OUT table loading into frequency and azimuth RCS grids."""

    @classmethod
    def load_out(cls, path):
        """Load whitespace-delimited `.out` data into an RcsGrid.

        Expected columns per non-comment line:
            frequency_ghz  azimuth_deg  rcs_dbke  phase_deg

        Parsing rules:
            - Lines starting with `#` (or text after `#`) are ignored.
            - Values are whitespace-delimited.
            - Polarization is inferred from filename (`HH` or `VV`);
              if not present, polarization is `NA`.
            - The third column is interpreted as absolute dBke and converted to
              linear 2D scattering width using sigma_2d = (lambda / 2pi) * 10^(dBke/10).

        Output mapping:
            - azimuth axis   <- angle column
            - elevation axis <- single value [0.0]
            - frequency axis <- frequency_ghz column
            - polarization   <- inferred filename polarization
            - stored power   <- linear 2D scattering width (matches .grim storage)
        """
        from GRIM_Backend.datasets.constants import C0
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        file_name = os.path.basename(str(path))
        stem_upper = os.path.splitext(file_name)[0].upper()
        pol_matches = set(
            re.findall(r"(?<![A-Z0-9])(HH|VV)(?![A-Z0-9])", stem_upper)
        )
        if len(pol_matches) > 1:
            raise ValueError(
                f"OUT filename {file_name!r} ambiguously declares both HH and VV"
            )
        pol_label = next(iter(pol_matches)) if pol_matches else "NA"

        # Parsing stays a plain Python loop (the format is text), but the
        # per-row NumPy scalar calls, errstate contexts and element-wise
        # grid writes of the former implementation are gone: validation
        # uses float comparisons, and the grid is filled with array indexing.
        # Every message and the order in which problems are reported are
        # unchanged: the first offending line wins, whatever its problem.
        freq_column = array("d")
        azimuth_column = array("d")
        dbke_column = array("d")
        phase_column = array("d")
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"line {line_no}: expected exactly 4 columns "
                        "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                    )
                try:
                    freq_ghz = float(parts[0])
                    azimuth_deg = float(parts[1])
                    rcs_dbke = float(parts[2])
                    phase_deg = float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc

                if not (freq_ghz > 0.0 and math.isfinite(freq_ghz)):
                    raise ValueError(
                        f"line {line_no}: frequency_ghz must be positive and finite"
                    )
                if not math.isfinite(azimuth_deg):
                    raise ValueError(
                        f"line {line_no}: azimuth_deg must be finite"
                    )
                if math.isnan(rcs_dbke) or rcs_dbke == math.inf:
                    raise ValueError(
                        f"line {line_no}: rcs_dbke must be finite or -Inf"
                    )
                if math.isinf(phase_deg):
                    raise ValueError(
                        f"line {line_no}: phase_deg must be finite or NaN"
                    )
                freq_column.append(freq_ghz)
                azimuth_column.append(azimuth_deg)
                dbke_column.append(rcs_dbke)
                phase_column.append(phase_deg)

        record_count = len(freq_column)
        if not record_count:
            raise ValueError("OUT contains no data rows")

        frequencies = np.asarray(sorted(set(freq_column)), dtype=float)
        azimuths = np.asarray(sorted(set(azimuth_column)), dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        polarizations = np.asarray([pol_label], dtype=object)

        # Each record value is one of the sorted unique axis values, so its
        # insertion point is its axis index.
        az_index = np.searchsorted(azimuths, np.frombuffer(azimuth_column, dtype=np.float64))
        f_index = np.searchsorted(frequencies, np.frombuffer(freq_column, dtype=np.float64))

        # The dBke -> sigma_2d arithmetic is the former per-row expression on
        # Python floats, so the stored values are identical.  A conversion
        # failure is remembered rather than raised at once: a conflicting
        # duplicate on an earlier line is still reported first, as before.
        two_pi = 2.0 * np.pi
        sigma = np.empty(record_count, dtype=np.float64)
        failure = None
        for row, (freq_ghz, azimuth_deg, rcs_dbke) in enumerate(
            zip(freq_column, azimuth_column, dbke_column)
        ):
            if rcs_dbke == -math.inf:
                sigma[row] = 0.0
                continue
            lambda_m = C0 / (freq_ghz * 1.0e9)
            try:
                sigma_2d = (lambda_m / two_pi) * (10.0 ** (rcs_dbke / 10.0))
            except OverflowError:
                failure = (
                    row,
                    "OUT dBke magnitude overflows finite linear power at "
                    f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg",
                )
                break
            if not math.isfinite(sigma_2d):
                failure = (
                    row,
                    "OUT dBke magnitude does not produce finite linear power at "
                    f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg",
                )
                break
            sigma[row] = sigma_2d
        incoming_phase = np.deg2rad(np.frombuffer(phase_column, dtype=np.float64))

        limit = record_count if failure is None else failure[0]
        cells = az_index[:limit] * len(frequencies) + f_index[:limit]
        order = np.argsort(cells, kind="stable")
        sorted_cells = cells[order]
        first_of_cell = np.ones(order.size, dtype=bool)
        first_of_cell[1:] = sorted_cells[1:] != sorted_cells[:-1]
        first_rows = order[first_of_cell]

        shape = (len(azimuths), 1, len(frequencies), 1)
        power = np.full(shape, np.nan, dtype=np.float64)
        phase = np.full(shape, np.nan, dtype=np.float64)
        power[az_index[first_rows], 0, f_index[first_rows], 0] = sigma[first_rows]
        phase[az_index[first_rows], 0, f_index[first_rows], 0] = incoming_phase[first_rows]

        # Repeated coordinates are rare.  Replay them in file order against
        # the stored sample, exactly as the former row loop did: the first
        # record is kept, a later record may only supply a missing phase.
        for row in np.sort(order[~first_of_cell]).tolist():
            ai = int(az_index[row])
            fi = int(f_index[row])
            existing_phase = float(phase[ai, 0, fi, 0])
            incoming = float(incoming_phase[row])
            if not _cst_samples_equivalent(
                float(power[ai, 0, fi, 0]),
                existing_phase,
                float(sigma[row]),
                incoming,
            ):
                raise ValueError(
                    "conflicting duplicate OUT sample at "
                    f"frequency={freq_column[row]:g} GHz, "
                    f"azimuth={azimuth_column[row]:g} deg"
                )
            if not np.isfinite(existing_phase) and np.isfinite(incoming):
                phase[ai, 0, fi, 0] = incoming
        if failure is not None:
            raise ValueError(failure[1])

        if not np.isfinite(power).any():
            raise ValueError("OUT parsed, but no finite RCS magnitude values were found")

        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded OUT (dBke -> linear sigma_2d): {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_log_unit": "dBke"},
        )
