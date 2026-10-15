"""Canonical sample storage and bounded, read-only derived array access.

The historical ``rcs_power``/``rcs_phase`` API exposes mutable NumPy arrays.
Legacy exposure retains exact original fields until a bounded content check
detects an edit, including mutations made through ordinary NumPy views.
Application code uses ``power``/``phase`` and the bounded slice APIs instead.
"""
from __future__ import annotations

import math
import hashlib
import weakref
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from GRIM_Backend.datasets.constants import _RAW_COMPLEX_VALIDATION_BLOCK_CELLS
from GRIM_Backend.datasets.memory import _bounded_grid_selections


def _readonly(value):
    array = np.asarray(value)
    array.flags.writeable = False
    return array


class _FlatSamples(NDArrayOperatorsMixin):
    def __init__(self, component):
        self.component = component

    def __len__(self):
        return self.component.size

    def __getitem__(self, key):
        count = self.component.size
        if isinstance(key, slice):
            indices = np.arange(*key.indices(count), dtype=np.intp)
        else:
            indices = np.asarray(key)
            if indices.dtype.kind == "b":
                if indices.shape != (count,):
                    raise IndexError("boolean flat index has the wrong shape")
                indices = np.flatnonzero(indices)
            indices = np.asarray(indices, dtype=np.intp)
            indices = np.where(indices < 0, indices + count, indices)
            if np.any((indices < 0) | (indices >= count)):
                raise IndexError("flat sample index out of range")
        coordinates = np.unravel_index(indices, self.component.shape)
        return self.component[coordinates]

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.component, dtype=dtype).reshape(-1)
        if copy:
            return result.copy()
        return _readonly(result)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(item, (_FlatSamples, SampleComponent))
               for item in kwargs.get("out", ())):
            raise ValueError("derived samples are read-only; use set_power_phase()")
        return getattr(ufunc, method)(
            *(np.asarray(item) if isinstance(item, (_FlatSamples, SampleComponent)) else item
              for item in inputs), **kwargs
        )

    def __setitem__(self, key, value):
        raise ValueError("derived samples are read-only; use set_power_phase()")


class SampleComponent(NDArrayOperatorsMixin):
    """An array-like power or phase component that derives only indexed cells."""
    __array_priority__ = 1000

    def __init__(self, grid, component):
        self._grid = grid
        self._component = component

    @property
    def shape(self):
        return self._grid.shape

    @property
    def dtype(self):
        return self._grid.real_dtype

    @property
    def size(self):
        return self._grid.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def flat(self):
        return _FlatSamples(self)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        reader = self._grid.power_slice if self._component == "power" else self._grid.phase_slice
        result = reader(selection)
        return result[()] if np.ndim(result) == 0 else result

    def __setitem__(self, selection, value):
        raise ValueError("derived samples are read-only; use set_power_phase()")

    def __array__(self, dtype=None, copy=None):
        result = self[...]
        if dtype is not None:
            result = np.asarray(result, dtype=dtype)
        if copy:
            return result.copy()
        return _readonly(result)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(item, (SampleComponent, _FlatSamples))
               for item in kwargs.get("out", ())):
            raise ValueError("derived samples are read-only; use set_power_phase()")
        converted = [np.asarray(item) if isinstance(item, (SampleComponent, _FlatSamples))
                     else item for item in inputs]
        return getattr(ufunc, method)(*converted, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __repr__(self):
        return f"SampleComponent({self._component!r}, shape={self.shape}, dtype={self.dtype})"


class GridStorageMixin:
    def __copy__(self):
        """Retain historical shallow-copy sharing when legacy arrays emerge.

        GUI rows make shallow dataset copies for their own metadata. Their
        sample storage remains shared, including a later request for writable
        legacy arrays. Explicit sample replacement still affects only the
        instance on which it is performed.
        """
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        peers = getattr(self, "_storage_peers", None)
        if peers is None:
            peers = weakref.WeakSet([self])
            self._storage_peers = peers
        peers.add(result)
        result._storage_peers = peers
        return result

    @property
    def shape(self):
        source = self._power if self._legacy_samples else (self._complex if self._complex is not None else self._power)
        return source.shape

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def real_dtype(self):
        if self._complex is not None and not self._legacy_samples:
            if self._storage_domain == "raw_field":
                return np.dtype(np.float64)
            if ("rcs_amp_real" in self.extra and "rcs_amp_imag" in self.extra
                    and self._raw_normalization_defined()
                    and self._complete_authoritative_raw_arrays() is not None):
                return np.dtype(np.float64)
            return self._complex.real.dtype
        return np.result_type(self._power.dtype, self._phase.dtype) if self._phase is not None else self._power.dtype

    @property
    def storage_kind(self):
        return self._storage_kind

    @property
    def storage_domain(self):
        return self._storage_domain

    def phase_coverage(self):
        """Report the stored phase as ``"complete"``, ``"partial"`` or ``"none"``.

        A sample carries a phase when both its power and its phase are
        finite.  ``"complete"`` means every finite-power sample does,
        ``"partial"`` that only some do and ``"none"`` that no sample does.
        Canonical complex storage is complete and power-only storage has
        none, so both answer without reading a sample.  Only a
        ``power_phase`` grid (exceptional phase information that complex
        storage cannot encode) or the legacy writable arrays are scanned,
        in bounded blocks that stop as soon as the verdict is known.  A grid
        without any finite sample reports ``"none"``.
        """
        self._ensure_legacy_source_current()
        if self._storage_kind == "complex":
            return "complete"
        if self._phase is None:
            return "none"
        power, phase = np.asarray(self._power), np.asarray(self._phase)
        has_phase = has_unknown = False
        if power.size:
            for selection in _bounded_grid_selections(power.shape, _RAW_COMPLEX_VALIDATION_BLOCK_CELLS):
                finite_power = np.isfinite(power[selection])
                finite_phase = np.isfinite(phase[selection])
                has_phase = has_phase or bool(np.any(finite_power & finite_phase))
                has_unknown = has_unknown or bool(np.any(finite_power & ~finite_phase))
                if has_phase and has_unknown:
                    break
        if not has_phase:
            return "none"
        return "partial" if has_unknown else "complete"

    def _unknown_phase_count(self):
        """Count finite-power samples without a phase, in bounded blocks."""
        self._ensure_legacy_source_current()
        if self._storage_kind == "complex":
            return 0
        power = np.asarray(self._power)
        phase = None if self._phase is None else np.asarray(self._phase)
        count = 0
        if power.size:
            for selection in _bounded_grid_selections(power.shape, _RAW_COMPLEX_VALIDATION_BLOCK_CELLS):
                unknown = np.isfinite(power[selection])
                if phase is not None:
                    unknown &= ~np.isfinite(phase[selection])
                count += int(np.count_nonzero(unknown))
        return count

    @property
    def power(self):
        return self._power if self._legacy_samples else SampleComponent(self, "power")

    @property
    def phase(self):
        return self._phase if self._legacy_samples else SampleComponent(self, "phase")

    def _clear_raw_storage_metadata(self):
        for key in ("rcs_amp_real", "rcs_amp_imag", "raw_complex_amplitude_preserved"):
            self.extra.pop(key, None)

    def _install_complex(self, field, *, domain="physical", copy=True):
        self._complex = np.array(field, copy=copy)
        _readonly(self._complex)
        self._power = self._phase = None
        self._legacy_samples = False
        self._storage_kind = "complex"
        self._storage_domain = domain
        self._legacy_fingerprint = None
        self._raw_coverage_cache = None
        if domain == "raw_field":
            self.extra["rcs_amp_real"] = self._complex.real
            self.extra["rcs_amp_imag"] = self._complex.imag
            self.extra["raw_complex_amplitude_preserved"] = True

    def _canonicalize_samples(self, complex_input=None, *, prefer_complex=False):
        """Choose one representation, retaining exceptional phase information."""
        power, phase = self._power, self._phase
        # The coverage scan is grid-sized; skip it when the quantity cannot
        # normalise a raw pair at all.
        pair = (
            self._complete_authoritative_raw_arrays()
            if self.linear_quantity() in {"sigma_2d", "sigma_3d"} else None
        )
        if pair is not None:
            report = self._raw_complex_consistency_report(
                expected_shape=power.shape, frequencies=self.frequencies,
                rcs_power=power, rcs_phase=phase, units=self.units, extra=self.extra,
            )
            exceptional_zero = (power == 0) & (~np.isfinite(phase) | (phase != 0))
            if report["issues"] or np.any(exceptional_zero):
                # Preserve contradictory inputs for audit/save validation;
                # canonicalization must not silently repair their evidence.
                self._power, self._phase = _readonly(power), _readonly(phase)
                self._storage_kind = "power_phase"
                return
            # Coverage is checked above; normalization must also be defined.
            normalized_probe = self._authoritative_raw_amplitude_from_pair(pair, (0, 0, 0, 0))
            if normalized_probe is not None:
                dtype = np.result_type(pair[0].dtype, pair[1].dtype, np.complex64)
                raw = np.empty(power.shape, dtype=dtype)
                raw.real, raw.imag = pair
                self._install_complex(raw, domain="raw_field", copy=False)
                return
        finite_power = np.isfinite(power)
        if not np.any(np.isfinite(phase) & finite_power):
            self._power = _readonly(power)
            self._phase = None
            self._storage_kind = "power"
            return
        recoverable = self._recoverable_phase(power, phase)
        if recoverable:
            if prefer_complex and complex_input is not None:
                field = np.array(complex_input, copy=True)
                invalid = ~finite_power | ~np.isfinite(field.real) | ~np.isfinite(field.imag)
                field[invalid] = np.nan + 1j * np.nan
            else:
                field = self._complex_from_power_phase(power, phase)
            self._install_complex(field, copy=False)
            return
        self._power = _readonly(power)
        self._phase = _readonly(phase)
        self._storage_kind = "power_phase"

    def _recoverable_phase(self, power, phase):
        # Zero amplitude cannot encode either unknown phase or a supplied
        # nonzero phase. Unwrapped turn counts also require polar storage.
        if np.any(np.isfinite(power) & ~np.isfinite(phase)):
            return False
        if np.any((power == 0) & (phase != 0)):
            return False
        wrap = str(self.units.get("phase_wrap", "")).strip()
        if wrap == "0_360":
            return bool(np.all(~np.isfinite(phase) | ((phase >= 0) & (phase < 2 * np.pi))))
        return bool(np.all(~np.isfinite(phase) | ((phase >= -np.pi) & (phase <= np.pi))))

    def _wrapped_phase_values(self, phase):
        phase = np.array(phase, copy=True)
        finite = np.isfinite(phase)
        wrap = str(self.units.get("phase_wrap", "")).strip()
        if wrap == "0_360":
            outside = finite & ((phase < 0) | (phase >= 2 * np.pi))
            phase[outside] = np.remainder(phase[outside], 2 * np.pi)
            phase[phase >= 2 * np.pi] = 0
        elif wrap == "-180_180":
            outside = finite & ((phase < -np.pi) | (phase >= np.pi))
            phase[outside] = np.remainder(phase[outside] + np.pi, 2 * np.pi) - np.pi
        return phase

    @staticmethod
    def _squared_magnitude(field):
        """Return ``|field|**2`` as ``re*re + im*im`` in the field's real precision.

        This avoids ``hypot`` (a square root that is squared straight away) and
        works on a read-only view, so a power read allocates only the result
        and one same-sized scratch array instead of copying the complex field.
        """
        real, imag = field.real, field.imag
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.multiply(real, real)
            if isinstance(result, np.ndarray):
                np.add(result, np.multiply(imag, imag), out=result)
            else:
                result = result + imag * imag
        return np.asarray(result)

    def power_slice(self, selection=Ellipsis):
        if self._legacy_samples or self._complex is None:
            return _readonly(np.array(self._power[selection], copy=True))
        return _readonly(self._squared_magnitude(
            self._complex_samples(selection, owned=False)
        ))

    def phase_slice(self, selection=Ellipsis):
        if self._phase is not None:
            return _readonly(np.array(self._phase[selection], copy=True))
        if self._complex is None:
            return _readonly(np.full(np.shape(self._power[selection]), np.nan, dtype=self.real_dtype))
        field = self._complex_samples(selection, owned=False)
        wrap = str(self.units.get("phase_wrap", "")).strip()
        # ``arctan2`` on the component views is what ``np.angle`` computes
        # and allocates only the result; one boolean scratch array then
        # serves every mask and the wrap is applied in place.  NaN
        # components already propagate; the first mask also hides
        # infinities.  Scalar selections become 0-d arrays, as before.
        result = np.asarray(np.arctan2(field.imag, field.real))
        scratch = np.asarray(np.isfinite(field))
        np.logical_not(scratch, out=scratch)
        np.copyto(result, np.nan, where=scratch)
        if wrap == "0_360":
            np.less(result, 0, out=scratch)
            np.add(result, 2 * np.pi, out=result, where=scratch)
            np.greater_equal(result, 2 * np.pi, out=scratch)
            np.copyto(result, 0.0, where=scratch)
        elif wrap == "-180_180":
            np.greater_equal(result, np.pi, out=scratch)
            np.copyto(result, -np.pi, where=scratch)
        return _readonly(result.astype(self.real_dtype, copy=False))

    def _expose_legacy_samples(self):
        if self._legacy_samples:
            return
        # No derived cache survives mutable-array exposure. NumPy can then
        # freely produce writable views without an invalidation protocol.
        power = np.array(self.power, copy=True)
        phase = np.array(self.phase, copy=True)
        # A shallow copy previously shared its actual power/phase arrays.
        # Materializing those arrays must preserve that sharing relationship;
        # otherwise editing an engineer's source object leaves its GUI row
        # looking at the pre-edit canonical field.
        peers = list(getattr(self, "_storage_peers", (self,)))
        sharing = [peer for peer in peers
                   if not peer._legacy_samples and peer._complex is self._complex
                   and peer._power is self._power and peer._phase is self._phase
                   and peer._storage_domain == self._storage_domain]
        if self not in sharing:
            sharing.append(self)
        for peer in sharing:
            peer._power, peer._phase = power, phase
            peer._storage_kind = "legacy_power_phase"
            peer._legacy_samples = True
        # Preserve exact source I/Q for callers that only read the legacy
        # arrays. Once either array changes, the exposed arrays become the
        # sole authority. Hashing is deliberately confined to legacy access.
        fingerprint = self._sample_fingerprint()
        for peer in sharing:
            peer._legacy_fingerprint = fingerprint

    def _sample_fingerprint(self):
        digest = hashlib.sha256()
        for value in (self._power, self._phase):
            array = np.asarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            if array.dtype.hasobject:
                digest.update(repr(array.tolist()).encode("utf-8"))
                continue
            for block in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                                   op_flags=["readonly"], order="C", buffersize=65536):
                digest.update(np.ascontiguousarray(block).tobytes())
        return digest.digest()

    def _ensure_legacy_source_current(self):
        if self._legacy_samples and self._complex is not None:
            if self._sample_fingerprint() != self._legacy_fingerprint:
                self._complex = None
                self._storage_domain = "physical"
                self._legacy_fingerprint = None
                self._clear_raw_storage_metadata()

    @property
    def rcs_power(self):
        """Legacy mutable power array; use ``power`` for efficient read access."""
        self._expose_legacy_samples()
        return self._power

    @rcs_power.setter
    def rcs_power(self, value):
        self._expose_legacy_samples()
        self._power = value

    @property
    def rcs_phase(self):
        """Legacy mutable phase array; use ``phase`` for efficient read access."""
        self._expose_legacy_samples()
        return self._phase

    @rcs_phase.setter
    def rcs_phase(self, value):
        self._expose_legacy_samples()
        self._phase = value

    def _sample_kwargs(self, selection=None, transform=None):
        """Return constructor samples for an exact rearrangement of this grid.

        Every returned array is freshly owned, so a caller may edit it in place
        and the constructor may adopt it (``_adopt_clean_arrays``) without a
        further copy.  A raw solver field is returned unnormalized together
        with ``rcs_storage_domain``; the constructor then re-derives the raw
        metadata views instead of validating a normalized field against them.
        """
        self._ensure_legacy_source_current()
        selection = Ellipsis if selection is None else selection

        def changed(values):
            result = np.asarray(values if transform is None else transform(values))
            if result.flags.owndata and result.flags.writeable:
                return result
            return np.array(result, copy=True)

        if self._complex is not None:
            if self._storage_domain == "raw_field":
                return {"rcs": changed(self._complex[selection]),
                        "rcs_storage_domain": "raw_field"}
            return {"rcs": changed(self.rcs_slice(selection))}
        result = {"rcs_power": changed(self._power[selection])}
        if self._phase is not None:
            result["rcs_phase"] = changed(self._phase[selection])
        return result

    def native_sample_payload(self):
        self._ensure_legacy_source_current()
        if (self._complex is not None and self._storage_domain == "physical"
                and ("rcs_amp_real" in self.extra or "rcs_amp_imag" in self.extra)
                and self._raw_normalization_defined()):
            pair = self._complete_authoritative_raw_arrays()
            if pair is not None:
                raw = np.empty(pair[0].shape, dtype=np.result_type(pair[0].dtype, pair[1].dtype, np.complex64))
                raw.real, raw.imag = pair
                return {"storage_kind": "complex", "storage_domain": "raw_field", "rcs_complex": _readonly(raw)}
        if self._complex is not None:
            return {"storage_kind": "complex", "storage_domain": self._storage_domain,
                    "rcs_complex": self._complex}
        result = {"storage_kind": "power" if self._phase is None else "power_phase",
                  "storage_domain": "physical", "rcs_power": self._power}
        if self._phase is not None:
            result["rcs_phase"] = self._phase
        return result

    def set_power_phase(self, selection=Ellipsis, *, power=None, phase=None):
        """Atomically replace selected physical samples and recanonicalize."""
        if power is None and phase is None:
            return
        self._ensure_legacy_source_current()
        dtype = np.result_type(self.real_dtype, *(np.asarray(value).dtype for value in (power, phase) if value is not None))
        if self._complex is not None:
            field = self.rcs
            complex_dtype = np.complex128 if np.dtype(dtype).itemsize > 4 else np.complex64
            selected = np.asarray(field[selection], dtype=complex_dtype)
            # Derive in the promoted precision as well as storing in it;
            # float32 hypot followed by float64 scaling retains its old error.
            selected_power = np.array(np.abs(selected) ** 2, dtype=dtype, copy=True)
            selected_phase = self._wrapped_phase_values(np.angle(selected).astype(dtype, copy=False))
            if power is not None:
                selected_power[...] = power
            if phase is not None:
                selected_phase[...] = phase
            selected_power = self._clean_power(selected_power)
            selected_phase = self._wrapped_phase_values(self._clean_phase(selected_phase))
            selected_phase[~np.isfinite(selected_power)] = np.nan
            if self._recoverable_phase(selected_power, selected_phase):
                field = field.astype(complex_dtype, copy=False)
                replacement = self._complex_from_power_phase(selected_power, selected_phase)
                if phase is None:
                    # A magnitude edit preserves the original complex
                    # direction without an atan2 -> exp round trip.
                    amplitude = np.abs(selected)
                    usable = (amplitude > 0) & np.isfinite(amplitude) & np.isfinite(selected_power)
                    scale = np.ones(np.shape(amplitude), dtype=dtype)
                    np.divide(np.sqrt(selected_power), amplitude, out=scale, where=usable)
                    replacement = np.where(usable, selected * scale, replacement)
                field[selection] = replacement
                self._clear_raw_storage_metadata()
                self._install_complex(field, copy=False)
                return
        new_power = np.array(self.power, dtype=dtype, copy=True)
        new_phase = np.array(self.phase, dtype=dtype, copy=True)
        if power is not None:
            new_power[selection] = power
        if phase is not None:
            new_phase[selection] = phase
        new_power = self._clean_power(new_power)
        new_phase = self._wrapped_phase_values(self._clean_phase(new_phase))
        new_phase[~np.isfinite(new_power)] = np.nan
        self._power, self._phase = new_power, new_phase
        self._complex = None
        self._legacy_samples = False
        self._storage_domain = "physical"
        self._clear_raw_storage_metadata()
        self._canonicalize_samples()

    def set_complex(self, selection, values):
        """Replace selected samples with known complex fields."""
        values = np.asarray(values)
        field = self.rcs
        field = field.astype(np.result_type(field.dtype, values.dtype), copy=False)
        field[selection] = values
        invalid = ~np.isfinite(field.real) | ~np.isfinite(field.imag)
        field[invalid] = np.nan + 1j * np.nan
        if self._complex is not None:
            self._clear_raw_storage_metadata()
            self._install_complex(field, copy=False)
            return
        new_power = np.array(self.power, dtype=field.real.dtype, copy=True)
        new_phase = np.array(self.phase, dtype=field.real.dtype, copy=True)
        new_power[selection] = np.abs(values) ** 2
        new_phase[selection] = np.angle(values)
        self._power, self._phase = self._clean_power(new_power), self._clean_phase(new_phase)
        self._phase[~np.isfinite(self._power)] = np.nan
        self._complex = None
        self._legacy_samples = False
        self._storage_domain = "physical"
        self._clear_raw_storage_metadata()
        self._canonicalize_samples(field, prefer_complex=True)
