"""Coherent and incoherent arithmetic and statistical reductions."""
from __future__ import annotations

import json

import numpy as np

from GRIM_Backend.datasets.result_storage import (
    allocate_result, copy_result, resident_result_bytes,
)

from GRIM_Backend.datasets.operations import dataset_operation

from GRIM_Backend.datasets.audit import _physical_grid_content_sha256, _support_reference_qa
from GRIM_Backend.datasets.constants import _COHERENT_OPERATION_BLOCK_CELLS
from GRIM_Backend.datasets.memory import (
    _bounded_grid_selections,
    _coherent_working_set_limit_bytes,
    _real_storage_dtype,
)


def _bounded_sum(grids, *, coherent=False):
    """Accumulate in input order, deriving only the current sample block."""
    if coherent:
        readers_and_types = [grid._bounded_complex_slice_reader() for grid in grids]
        readers = [reader for reader, _dtype in readers_and_types]
        dtype = np.result_type(*(np.result_type(dtype, np.complex64)
                                 for _reader, dtype in readers_and_types))
    else:
        readers = [grid.power_slice for grid in grids]
        dtype = np.result_type(*(grid.power.dtype for grid in grids))
        # Writable legacy setters can supply integer or float16 power arrays.
        # Match the constructor's real storage dtype after (not before) the
        # sequential sum, preserving historical intermediate arithmetic.
        dtype = np.dtype(np.float64 if dtype.kind == "f" and dtype.itemsize > 4 else np.float32)
    result = allocate_result(grids[0].shape, dtype)
    for selection in _bounded_grid_selections(result.shape, _COHERENT_OPERATION_BLOCK_CELLS):
        block = readers[0](selection)
        for read in readers[1:]:
            # Do not promote the first inputs to the final output precision:
            # preserve the historical sequential mixed-dtype rounding.
            block = block + read(selection)
        # The output is newly owned. Apply the constructor's normalization
        # here so it can adopt the completed array without a grid-sized copy.
        if coherent:
            block[~np.isfinite(block.real) | ~np.isfinite(block.imag)] = np.nan + 1j * np.nan
        else:
            block = np.asarray(block, dtype=dtype)
            block[~np.isfinite(block)] = np.nan
            np.maximum(block, 0.0, out=block)
        result[selection] = block
    return result


def _finite_complex_count(values):
    return sum(int(np.count_nonzero(np.isfinite(values[selection])))
               for selection in _bounded_grid_selections(
                   values.shape, _COHERENT_OPERATION_BLOCK_CELLS))


class GridArithmeticMixin:
    """Coherent and incoherent arithmetic and statistical reductions."""

    @dataset_operation("coherent_add")
    def coherent_add(self, other, *, metadata_attested=False):
        """Coherently add two grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            other: Another RcsGrid with identical axes.
            metadata_attested: Record a user attestation when supplied.
                Missing or conflicting conventions are advisory; samples are used as supplied.

        Returns:
            New RcsGrid with rcs = self.rcs + other.rcs.
        """
        self._assert_compatible(
            other,
            coherent=True,
            coherent_metadata_attested=metadata_attested,
            _scan_phase_samples=False,
        )
        rcs_out = _bounded_sum((self, other), coherent=True)
        usable_count = _finite_complex_count(rcs_out)
        if usable_count == 0:
            raise ValueError(
                "coherent addition has no common usable complex samples"
            )
        history, attestation_extra = self._coherent_attestation_provenance(
            (other,),
            operation="coherent-add",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            (other,),
            operation="coherent-add",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-add",
                "total_sample_count": int(rcs_out.size),
                "usable_sample_count": usable_count,
                "masked_sample_count": int(rcs_out.size - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
            _adopt_clean_arrays=True,
        )

    @dataset_operation("coherent_add_many")
    def coherent_add_many(self, *grids, metadata_attested=False):
        """Coherently add multiple grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            *grids: One or more RcsGrid instances.
            metadata_attested: Record a user attestation when supplied.
                Missing or conflicting conventions are advisory; samples are used as supplied.

        Returns:
            New RcsGrid with rcs = self.rcs + sum(grid.rcs).
        """
        if not grids:
            return self
        for grid in grids:
            self._assert_compatible(
                grid,
                coherent=True,
                coherent_metadata_attested=metadata_attested,
                _scan_phase_samples=False,
            )
        total = _bounded_sum((self, *grids), coherent=True)
        usable_count = _finite_complex_count(total)
        if usable_count == 0:
            raise ValueError(
                "coherent addition has no common usable complex samples"
            )
        history, attestation_extra = self._coherent_attestation_provenance(
            grids,
            operation="coherent-add-many",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            grids,
            operation="coherent-add-many",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-add-many",
                "total_sample_count": int(total.size),
                "usable_sample_count": usable_count,
                "masked_sample_count": int(total.size - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            total,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
            _adopt_clean_arrays=True,
        )

    @dataset_operation("coherent_subtract")
    def coherent_subtract(
        self,
        other,
        *,
        metadata_attested=False,
        maximum_working_bytes=None,
    ):
        """Coherently subtract two grids (complex difference).

        Use when phases are aligned and you want field-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.
            metadata_attested: Record a user attestation when supplied.
                Missing or conflicting conventions are advisory; samples are used as supplied.
            maximum_working_bytes: Optional cap for the newly retained result
                arrays plus bounded arithmetic/QA scratch. By default GRIM uses
                half of currently available physical memory (or the reviewed
                fallback); ``GRIM_COHERENT_WORKING_SET_MB`` can set a process-
                wide cap.

        Returns:
            New RcsGrid with rcs = self.rcs - other.rcs.
        """
        self._assert_compatible(
            other,
            coherent=True,
            coherent_metadata_attested=metadata_attested,
            _scan_phase_samples=False,
        )
        def response_precision_upper_bound(grid):
            raw_real = (grid.extra or {}).get("rcs_amp_real")
            raw_imag = (grid.extra or {}).get("rcs_amp_imag")
            if raw_real is not None and raw_imag is not None:
                if (
                    np.asarray(raw_real).shape == grid.power.shape
                    and np.asarray(raw_imag).shape == grid.power.shape
                ):


                    return np.dtype(np.float64)
            return np.dtype(_real_storage_dtype(grid.power, grid.phase))

        left_real_dtype = response_precision_upper_bound(self)
        right_real_dtype = response_precision_upper_bound(other)
        real_dtype = np.dtype(
            np.float64
            if max(left_real_dtype.itemsize, right_real_dtype.itemsize) > 4
            else np.float32
        )
        complex_dtype = np.dtype(
            np.complex128 if real_dtype == np.dtype(np.float64) else np.complex64
        )
        cell_count = int(self.power.size)
        block_cells = min(cell_count, _COHERENT_OPERATION_BLOCK_CELLS)
        retained_bytes = 2 * real_dtype.itemsize * cell_count


        scratch_bytes = block_cells * (
            12 * complex_dtype.itemsize + 8 * real_dtype.itemsize
        )
        result_ram_bytes = resident_result_bytes(retained_bytes)
        estimated_peak_bytes = result_ram_bytes + scratch_bytes
        limit_bytes = _coherent_working_set_limit_bytes(maximum_working_bytes)
        if (
            retained_bytes > np.iinfo(np.intp).max
            or estimated_peak_bytes > np.iinfo(np.intp).max
        ):
            raise MemoryError(
                "coherent subtraction result exceeds this Python/NumPy build's "
                "addressable allocation size"
            )
        if estimated_peak_bytes > limit_bytes:
            raise MemoryError(
                "coherent subtraction needs an estimated "
                f"{estimated_peak_bytes / 1024**2:.1f} MiB working set "
                f"({result_ram_bytes / 1024**2:.1f} MiB result RAM plus "
                f"{scratch_bytes / 1024**2:.1f} MiB bounded scratch), above "
                f"the {limit_bytes / 1024**2:.1f} MiB limit. Crop the common "
                "grid or deliberately raise maximum_working_bytes / "
                "GRIM_COHERENT_WORKING_SET_MB on a machine with verified "
                "headroom."
            )


        read_left, _left_reader_dtype = self._bounded_complex_slice_reader()
        read_right, _right_reader_dtype = other._bounded_complex_slice_reader()
        field_out = allocate_result(self.power.shape, complex_dtype)
        usable_count = 0
        for selection in _bounded_grid_selections(
            self.power.shape, _COHERENT_OPERATION_BLOCK_CELLS
        ):
            left = np.asarray(read_left(selection), dtype=complex_dtype)
            if not left.flags.writeable or not left.flags.owndata:
                left = np.array(left, dtype=complex_dtype, copy=True)
            right = np.asarray(read_right(selection), dtype=complex_dtype)
            np.subtract(left, right, out=left)
            power_block = np.empty(left.shape, dtype=real_dtype)
            with np.errstate(invalid="ignore", over="ignore"):
                np.hypot(left.real, left.imag, out=power_block)
                np.multiply(power_block, power_block, out=power_block)
            invalid = ~np.isfinite(left.real)
            invalid |= ~np.isfinite(left.imag)
            invalid |= ~np.isfinite(power_block)
            left[invalid] = np.nan + 1j * np.nan
            field_out[selection] = left
            usable_count += int(np.count_nonzero(~invalid))

        if usable_count == 0:
            raise ValueError(
                "coherent subtraction has no common usable complex samples"
            )

        history, attestation_extra = self._coherent_attestation_provenance(
            (other,),
            operation="coherent-subtract",
            metadata_attested=metadata_attested,
        )
        extra = self._derived_response_extra(
            (other,),
            operation="coherent-subtract",
            coherent=True,
            attestation_extra=attestation_extra,
        )
        extra["coherent_sample_qa_json"] = json.dumps(
            {
                "schema": "grim.coherent-sample-qa.v1",
                "operation": "coherent-subtract",
                "total_sample_count": int(cell_count),
                "usable_sample_count": int(usable_count),
                "masked_sample_count": int(cell_count - usable_count),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs=field_out,
            rcs_domain="power_phase",
            history=history,
            extra=extra,
            _adopt_clean_arrays=True,
        )

    @dataset_operation("support_referenced_difference")
    def support_referenced_difference(
        self,
        support_reference,
        *,
        metadata_attested=False,
        assumptions_attested=False,
        target_label=None,
        support_label=None,
        maximum_working_bytes=None,
    ):
        """Return an exact target-plus-support minus support-only field.

        This guided wrapper intentionally delegates all numerical work and
        compatibility enforcement to :meth:`coherent_subtract`.  It adds role-
        explicit, content-bound provenance and QA diagnostics so the result
        cannot be mistaken for a generic operand-order subtraction.

        The result is a *support-referenced difference*.  It is not guaranteed
        to equal the target's free-space response because target/support
        coupling, shadowing, and multiple-bounce terms are not recoverable from
        two measurements by subtraction.
        """
        from GRIM_Backend.datasets.grid import RcsGrid

        for option_name, option_value in (
            ("metadata_attested", metadata_attested),
            ("assumptions_attested", assumptions_attested),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise TypeError(f"{option_name} must be True or False")
        if not isinstance(support_reference, RcsGrid):
            raise TypeError("support_reference must be an RcsGrid")
        if support_reference is self:
            raise ValueError(
                "target+support and support-only roles must use different datasets"
            )
        chained_inputs = []
        if "support_reference_difference_json" in (self.extra or {}):
            chained_inputs.append("target_plus_support")
        if "support_reference_difference_json" in (support_reference.extra or {}):
            chained_inputs.append("support_only_reference")

        support_metadata_contract = (
            self._assert_support_reference_metadata_compatible(support_reference)
        )


        difference = self.coherent_subtract(
            support_reference,
            metadata_attested=bool(metadata_attested),
            maximum_working_bytes=maximum_working_bytes,
        )
        target_name = str(
            target_label or self.source_path or "target+support acquisition"
        )
        support_name = str(
            support_label
            or support_reference.source_path
            or "support-only reference"
        )
        qa = _support_reference_qa(self, support_reference, difference)
        if int(qa["common_finite_sample_count"]) == 0:
            raise ValueError(
                "support-referenced difference has no common finite complex "
                "samples after exact subtraction"
            )
        content_namespace = "grim.physical-grid-content.v1"
        target_sha256 = _physical_grid_content_sha256(
            self, namespace=content_namespace
        )
        support_sha256 = _physical_grid_content_sha256(
            support_reference, namespace=content_namespace
        )
        result_sha256 = _physical_grid_content_sha256(
            difference, namespace=content_namespace
        )
        provenance = {
            "schema": "grim.support-reference-difference.v1",
            "mode": "exact_complex_subtraction",
            "formula": "A_difference=A_target_plus_support-A_support_reference",
            "axis_policy": (
                "identical_axes_units_quantities_and_noncontradictory_explicit_"
                "acquisition_metadata; no_interpolation"
            ),
            "target_plus_support": target_name,
            "target_plus_support_content_sha256": target_sha256,
            "support_only_reference": support_name,
            "support_only_reference_content_sha256": support_sha256,
            "result_content_sha256": result_sha256,
            "content_hash_schema": content_namespace,
            "operation_selected_as_assumption_of_compatible_acquisition": True,
            "user_assumptions_attested": bool(assumptions_attested),
            "metadata_attestation_used": bool(metadata_attested),
            "chained_support_difference_input_roles": chained_inputs,
            "support_metadata_contract": support_metadata_contract,
            "interpretation": "support_referenced_complex_difference",
            "not_free_space_target": True,
            "unrecoverable_effects": [
                "target_support_coupling",
                "support_shadowing",
                "target_support_multiple_bounce_scattering",
                "acquisition_drift_or_misregistration",
            ],
            "qa": qa,
        }
        difference.extra = dict(difference.extra or {})
        for key, value in support_metadata_contract[
            "matching_explicit_declarations"
        ].items():
            if key != "complex_field_domain":
                difference.extra[key] = value
        difference.extra["support_reference_difference_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        difference.extra["complex_field_domain"] = (
            "support_referenced_complex_difference"
        )
        history_entry = (
            "Support-referenced difference (exact complex subtraction): "
            f"target+support={target_name}; support-only={support_name}; "
            "identical axes, no interpolation; QA/content hashes recorded; "
            "not a reconstructed free-space target response"
        )
        if chained_inputs:
            history_entry += (
                "; warning: chained support-difference input role(s)="
                + ",".join(chained_inputs)
            )
        difference.history = (
            f"{difference.history}\n{history_entry}"
            if difference.history
            else history_entry
        )
        difference.source_path = None
        return difference

    @dataset_operation("incoherent_add")
    def incoherent_add(self, other):
        """Incoherently add two grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.power + other.power.
        """
        self._assert_compatible(other)
        power_sum = _bounded_sum((self, other))
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_sum,
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                (other,), operation="incoherent-add", coherent=False
            ),
            _adopt_clean_arrays=True,
        )

    @dataset_operation("incoherent_add_many")
    def incoherent_add_many(self, *grids):
        """Incoherently add multiple grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with linear power = self.power + sum(grid.power).
        """
        if not grids:
            return self
        for grid in grids:
            self._assert_compatible(grid)
        total = _bounded_sum((self, *grids))
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=total,
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                grids, operation="incoherent-add-many", coherent=False
            ),
            _adopt_clean_arrays=True,
        )

    @dataset_operation("incoherent_subtract")
    def incoherent_subtract(self, other):
        """Incoherently subtract two grids (magnitude difference).

        Use when phases are unrelated and you want power-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.power - other.power.

        A physically negative power result is rejected.  Only a negative
        residual consistent with floating-point subtraction roundoff is
        replaced by exact zero; this prevents a materially invalid
        subtraction from being silently clipped into a plausible dataset.
        """
        self._assert_compatible(other)
        left_dtype, right_dtype = self.power.dtype, other.power.dtype
        calculation_dtype = np.result_type(left_dtype, right_dtype, np.float64)
        power_diff = allocate_result(self.shape, calculation_dtype)
        input_epsilons = [
            np.finfo(dtype).eps
            for dtype in (left_dtype, right_dtype)
            if np.issubdtype(dtype, np.floating)
        ]
        input_epsilon = max(input_epsilons, default=np.finfo(np.float64).eps)
        count, minimum = 0, float("inf")
        for selection in _bounded_grid_selections(self.shape, _COHERENT_OPERATION_BLOCK_CELLS):
            left = np.asarray(self.power[selection], dtype=calculation_dtype)
            right = np.asarray(other.power[selection], dtype=calculation_dtype)
            block = left - right
            roundoff_limit = 8.0 * float(input_epsilon) * np.maximum(np.abs(left), np.abs(right))
            finite_negative = np.isfinite(block) & (block < 0.0)
            material_negative = finite_negative & (block < -roundoff_limit)
            if np.any(material_negative):
                count += int(np.count_nonzero(material_negative))
                minimum = min(minimum, float(np.min(block[material_negative])))
            block[finite_negative] = 0.0
            power_diff[selection] = block
        if count:
            raise ValueError(
                "incoherent subtraction would produce materially negative "
                f"linear power in {count} cell(s); minimum difference is "
                f"{minimum:.17g}"
            )
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_diff,
            rcs_domain="power_phase",
            extra=self._derived_response_extra(
                (other,), operation="incoherent-subtract", coherent=False
            ),
        )

    @dataset_operation("arithmetic_db_subtract")
    def arithmetic_db_subtract(self, other):
        """Return the dimensionless power ratio represented by a dB difference.

        Returns a grid whose dB display equals ``self_dB - other_dB``. For two
        constant lines at 30 and 25 dBsm, the result displays as 5 dB. Phase is
        meaningless for this magnitude-domain operation and is set to NaN.

        Both grids must share the same ``default_log_unit`` (dBsm or dBke).
        """
        self._assert_compatible(other)
        unit_a = self.default_log_unit()
        unit_b = other.default_log_unit()
        if unit_a != unit_b:
            raise ValueError(
                f"dB arithmetic requires matching log units; got {unit_a} vs {unit_b}"
            )


        output_dtype = np.result_type(
            self.power.dtype, other.power.dtype, np.float64
        )
        output_power = allocate_result(self.shape, output_dtype, fill=np.nan)
        for selection in _bounded_grid_selections(self.shape, _COHERENT_OPERATION_BLOCK_CELLS):
            numerator = np.asarray(self.power[selection], dtype=output_dtype)
            denominator = np.asarray(other.power[selection], dtype=output_dtype)
            valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
            block = output_power[selection]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                np.divide(numerator, denominator, out=block, where=valid)
            block[~np.isfinite(block)] = np.nan
        ratio_units = dict(self.units)
        ratio_units["rcs_log_unit"] = "dB"
        ratio_units["rcs_linear_quantity"] = "power_ratio"

        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=output_power,
            rcs_domain="power_phase",
            units=ratio_units,
            extra=self._derived_response_extra(
                (other,), operation="arithmetic-db-subtract", coherent=False
            ),
        )

    @dataset_operation("statistics_dataset")
    def statistics_dataset(
        self,
        statistic="mean",
        axes=("azimuth", "elevation", "frequency"),
        *,
        domain="magnitude",
        percentile=50.0,
        broadcast_reduced=False,
    ):
        """Compute a statistic over selected axes and return a dataset."""
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2, "polarization": 3}
        axis_alias = {
            "azimuths": "azimuth",
            "elevations": "elevation",
            "frequencies": "frequency",
            "polarizations": "polarization",
            "az": "azimuth",
            "el": "elevation",
            "freq": "frequency",
            "pol": "polarization",
        }

        axes_list = self._as_list(axes)
        if axes_list is None:
            raise ValueError("axes must include at least one axis")
        reduce_axes = []
        for axis_name in axes_list:
            key = str(axis_name).strip().lower()
            key = axis_alias.get(key, key)
            if key not in axis_map:
                raise ValueError(f"unknown axis: {axis_name}")
            idx = axis_map[key]
            if idx not in reduce_axes:
                reduce_axes.append(idx)
        if not reduce_axes:
            raise ValueError("axes must include at least one axis")
        reduce_axes = tuple(sorted(reduce_axes))

        if domain not in {"complex", "magnitude", "db", "dbsm", "dbke"}:
            raise ValueError("domain must be 'complex', 'magnitude', 'dbsm', or 'dbke'")

        stat_key = str(statistic).strip().lower()
        if stat_key.startswith("p") and stat_key[1:].replace(".", "", 1).isdigit():
            percentile = float(stat_key[1:])
            stat_key = "percentile"

        if domain == "complex" and stat_key == "percentile":
            raise ValueError("percentile on complex values is not supported; use magnitude, dbsm, or dbke domain")
        if stat_key == "std" and domain in {"db", "dbsm", "dbke"}:
            raise ValueError(
                "standard deviation in a logarithmic domain is a dB spread, "
                "not an absolute RCS dataset. Use domain='magnitude' for a "
                "linear-power RCS standard deviation."
            )

        reducers = {"mean": np.nanmean, "median": np.nanmedian,
                    "min": np.nanmin, "max": np.nanmax, "std": np.nanstd,
                    "percentile": np.nanpercentile}
        if stat_key not in reducers:
            raise ValueError(
                "statistic must be mean, median, min, max, std, percentile, or pXX (for percentile XX)"
            )
        from .reduction_workspace import reduce_dataset
        reduced = reduce_dataset(self, domain, reduce_axes, stat_key, percentile)

        axis_values = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        if broadcast_reduced:


            reduced = copy_result(np.broadcast_to(reduced, self.shape))
        else:
            for axis_idx in reduce_axes:
                original = axis_values[axis_idx]
                if axis_idx == 3:
                    axis_values[axis_idx] = np.asarray(["ALL"])
                else:
                    numeric = np.asarray(original, dtype=float)
                    rep = float(np.nanmean(numeric)) if numeric.size else 0.0
                    axis_values[axis_idx] = np.asarray([rep], dtype=float)

        statistics_coherent = domain == "complex"
        statistics_extra = self._derived_response_extra(
            operation=f"statistics-{stat_key}-{domain}",
            coherent=statistics_coherent,
        )
        axis_names = ("azimuth", "elevation", "frequency", "polarization")
        axis_summaries = {}
        source_axes = (
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
        )
        for axis_idx in reduce_axes:
            axis_name = axis_names[axis_idx]
            original = np.asarray(source_axes[axis_idx])
            summary = {"source_count": int(original.size)}
            if axis_idx == 3:
                summary["representative"] = (
                    None if broadcast_reduced else "ALL"
                )
            else:
                numeric = np.asarray(original, dtype=float)
                finite = numeric[np.isfinite(numeric)]
                summary.update(
                    source_min=(float(np.min(finite)) if finite.size else None),
                    source_max=(float(np.max(finite)) if finite.size else None),
                    representative=(
                        None
                        if broadcast_reduced
                        else float(axis_values[axis_idx][0])
                    ),
                )
            axis_summaries[axis_name] = summary
        statistics_extra["statistics_reduction_json"] = json.dumps(
            {
                "schema": "grim.statistics-reduction.v1",
                "statistic": stat_key,
                "percentile": (
                    float(percentile) if stat_key == "percentile" else None
                ),
                "domain": domain,
                "reduced_axes": [axis_names[index] for index in reduce_axes],
                "broadcast_reduced": bool(broadcast_reduced),
                "coordinate_semantics": (
                    "original coordinates with repeated aggregate value"
                    if broadcast_reduced
                    else "representative aggregate label; not an observed sample"
                ),
                "axes": axis_summaries,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        source_role = self._declared_scalar_metadata(
            "assembly_response_role"
        ).strip().casefold()
        if source_role == "features_only_delta":
            statistics_extra["assembly_response_role"] = (
                "coherent_field_sum"
                if statistics_coherent
                else "incoherent_power_sum"
            )
        if statistics_coherent:
            self._invalidate_assembly_sampling_hash(
                statistics_extra, f"statistics-{stat_key}-{domain}"
            )

        if domain == "complex":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                reduced,
                rcs_domain="power_phase",
                extra=statistics_extra,
            )
        if domain == "magnitude":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                rcs_power=copy_result(reduced, dtype=self.power.dtype),
                rcs_domain="power_phase",
                extra=statistics_extra,
            )

        reduced_linear = allocate_result(reduced.shape, self.power.dtype)
        for selection in _bounded_grid_selections(reduced.shape, _COHERENT_OPERATION_BLOCK_CELLS):
            block = np.asarray(reduced[selection], dtype=float)
            if domain == "dbke":
                freq_grid = np.asarray(axis_values[2][selection[2]], dtype=float).reshape(1, 1, -1, 1)
                reduced_linear[selection] = self.dbke_to_linear(block, freq_grid)
            else:
                reduced_linear[selection] = 10.0 ** (block / 10.0)
        return self._new_grid(
            axis_values[0],
            axis_values[1],
            axis_values[2],
            axis_values[3],
            rcs_power=reduced_linear,
            rcs_domain="power_phase",
            extra=statistics_extra,
        )
