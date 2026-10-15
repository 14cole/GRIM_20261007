"""Standalone native GRIM codec and sliceable compatibility component views."""
from __future__ import annotations
import itertools
import json
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

C0 = 299_792_458.0
DECODE_BLOCK_CELLS = 262_144
SAMPLES = {"rcs_complex", "rcs_power", "rcs_phase", "rcs_amp_real", "rcs_amp_imag"}
FRAMING = {"grim_format_version", "storage_kind", "storage_domain"}


def grid_blocks(shape, max_cells=DECODE_BLOCK_CELLS):
    if not shape or any(length <= 0 for length in shape):
        raise ValueError("Native sample grid axes must be nonempty")
    sizes = list(shape)
    while int(np.prod(sizes, dtype=np.int64)) > max_cells:
        axis = max(range(len(sizes)), key=lambda i: sizes[i])
        sizes[axis] = max(1, (sizes[axis] + 1) // 2)
    for starts in itertools.product(*(range(0, n, step) for n, step in zip(shape, sizes))):
        yield tuple(slice(start, min(n, start + step)) for start, n, step in zip(starts, shape, sizes))


def units_and_scale(payload, label="GRIM dataset"):
    try:
        text = np.asarray(payload.get("units", "{}")).item()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        units = json.loads(text) if text else {}
        if not isinstance(units, dict):
            raise ValueError("units must be an object")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: invalid native units metadata.") from exc
    quantity = str(units.get("rcs_linear_quantity", "")).strip().lower()
    if not quantity:
        quantity = "sigma_2d" if str(units.get("rcs_log_unit", "dBsm")).lower() == "dbke" else "sigma_3d"
    if str(payload.get("power_domain", "")) == "delta_amp_sq" and str(payload.get("rcs_domain", "")) == "delta":
        scale = 1.
    elif quantity == "sigma_3d":
        scale = np.sqrt(4. * np.pi)
    elif quantity == "sigma_2d":
        factor = {"hz": 1., "khz": 1.e3, "mhz": 1.e6, "ghz": 1.e9}.get(str(units.get("frequency", "GHz")).lower())
        frequency = np.asarray(payload["frequencies"], dtype=float)
        if factor is None or np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError(f"{label}: invalid native frequency units or values.")
        scale = (1. / (2. * np.sqrt(2. * np.pi * frequency * factor / C0)))[None, None, :, None]
    else:
        scale = None
    return units, scale


class SampleComponent(NDArrayOperatorsMixin):
    """Derive bounded read values; item assignment detaches this component.

    Slices are computed values, not writable views into a derived full grid.
    Explicit ``np.asarray(component)`` materializes a writable legacy array;
    ``np.array(component, copy=True)`` requests an independent full copy.
    """
    __array_priority__ = 1000

    def __init__(self, field, component, scale=1., wrap="-180_180", dtype=None):
        self.field, self.component, self.scale, self.wrap = field, component, scale, wrap
        self._detached = None
        self.shape, self.ndim, self.size = field.shape, field.ndim, field.size
        self.dtype = np.dtype(dtype or np.result_type(field.real.dtype, float))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        if self._detached is not None:
            return self._detached[selection]
        field = self.field[selection]
        if self.component == "missing_phase":
            value = np.full(np.shape(field), np.nan)
        else:
            scale = np.broadcast_to(self.scale, self.shape)[selection] if np.ndim(self.scale) else self.scale
            if self.component in {"power", "physical_power"}:
                with np.errstate(over="ignore", invalid="ignore"):
                    value = (np.abs(field)**2 if self.component == "physical_power" else
                             (field.real * field.real + field.imag * field.imag) * np.asarray(scale)**2)
            elif self.component == "phase":
                value = np.angle(field)
                value = np.where((field.real == 0.) & (field.imag == 0.), 0., value)
                if self.wrap == "0_360":
                    value = np.mod(value, 2. * np.pi)
            else:
                field = field * scale if np.ndim(scale) or scale != 1. else field
                value = field.real if self.component == "raw_real" else field.imag
        # Raw components must have the same value semantics as derived ones.
        # Returning field.real/imag directly would bypass detachment on edits.
        return np.array(value, dtype=self.dtype, copy=True)

    def __array__(self, dtype=None, copy=None):
        if copy is True:
            return np.array(self[...], dtype=dtype, copy=True)
        return np.asarray(self._write_array(), dtype=dtype)

    def _write_array(self):
        if self._detached is None:
            self._detached = self[...]
        return self._detached

    def __setitem__(self, selection, value):
        self._write_array()[selection] = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "at" and inputs and isinstance(inputs[0], SampleComponent):
            inputs = (inputs[0]._write_array(),) + inputs[1:]
        inputs = tuple(np.asarray(x) if isinstance(x, SampleComponent) else x for x in inputs)
        original_outputs = kwargs.get("out")
        if original_outputs is not None:
            kwargs["out"] = tuple(x._write_array() if isinstance(x, SampleComponent) else x for x in original_outputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if original_outputs is not None:
            # In-place operators must return the wrapper, just as ndarray
            # operators return their original output. Returning the detached
            # array would lose the explicit-edit state in payload[key] += x.
            values = (result,) if ufunc.nout == 1 else result
            outputs = tuple(original if isinstance(original, SampleComponent) else value
                            for original, value in zip(original_outputs, values))
            return outputs[0] if ufunc.nout == 1 else outputs
        return result

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in {"fill", "put", "sort", "partition", "setfield", "itemset"}:
            return getattr(self._write_array(), name)
        if name == "byteswap":
            def byteswap(inplace=False):
                return self._write_array().byteswap(True) if inplace else np.asarray(self).byteswap()
            return byteswap
        return getattr(np.asarray(self), name)


def legacy_components(payload, field, *, domain="raw_field", component_dtype=None):
    result = {k: v for k, v in payload.items() if k not in SAMPLES | FRAMING}
    units, scale = units_and_scale(result)
    if domain == "raw_field" and scale is None:
        raise ValueError("GRIM dataset: unsupported raw field normalization")
    physical_scale = scale if domain == "raw_field" else 1.
    result["rcs_power"] = SampleComponent(field, "power" if domain == "raw_field" else "physical_power", physical_scale, dtype=component_dtype)
    result["rcs_phase"] = SampleComponent(field, "phase", physical_scale, units.get("phase_wrap", "-180_180"), component_dtype)
    if scale is not None:
        raw_scale = 1. if domain == "raw_field" else 1. / scale
        result["rcs_amp_real"] = SampleComponent(field, "raw_real", raw_scale)
        result["rcs_amp_imag"] = SampleComponent(field, "raw_imag", raw_scale)
        result["raw_complex_amplitude_preserved"] = np.asarray(True)
    return result


def raw_amplitude(payload, selection=Ellipsis):
    """Return the coherent field, sharing its unchanged native authority.

    Numerical consumers read this field and allocate their result. A caller
    that intends to edit it independently should explicitly make a copy.
    """
    real, imag = payload["rcs_amp_real"], payload["rcs_amp_imag"]
    if (isinstance(real, SampleComponent) and isinstance(imag, SampleComponent)
            and real.field is imag.field and real._detached is imag._detached is None
            and real.component == "raw_real" and imag.component == "raw_imag"
            and np.array_equal(real.scale, imag.scale)):
        scale = np.broadcast_to(real.scale, real.shape)[selection] if np.ndim(real.scale) else real.scale
        return real.field[selection] * scale if np.ndim(scale) or scale != 1. else real.field[selection]
    r, q = np.asarray(real[selection]), np.asarray(imag[selection])
    result = np.empty(r.shape, dtype=np.result_type(r.dtype, q.dtype, np.complex128))
    result.real, result.imag = r, q
    return result


def decode_native_samples(payload, *, label="GRIM dataset"):
    if "grim_format_version" not in payload:
        return payload
    result = dict(payload)
    def scalar(key):
        if key not in result or np.asarray(result[key]).size != 1:
            raise ValueError(f"{label}: missing or nonscalar native metadata {key!r}.")
        value = np.asarray(result[key]).reshape(-1)[0]
        return value.item() if isinstance(value, np.generic) else value
    version = scalar("grim_format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
        raise ValueError(f"{label}: unsupported grim_format_version {version!r}.")
    if version == 1:
        result.pop("grim_format_version")
        return result
    kind, domain = scalar("storage_kind"), scalar("storage_domain")
    if kind not in {"complex", "power", "power_phase"}:
        raise ValueError(f"{label}: unsupported native storage_kind {kind!r}.")
    if domain not in {"physical", "raw_field"} or (domain == "raw_field" and kind != "complex"):
        raise ValueError(f"{label}: unsupported native storage_domain {domain!r}.")
    expected = {"rcs_complex"} if kind == "complex" else {"rcs_power"} if kind == "power" else {"rcs_power", "rcs_phase"}
    if set(result) & {"rcs_complex", "rcs_power", "rcs_phase"} != expected:
        raise ValueError(f"{label}: conflicting native sample arrays.")
    shape = tuple(np.asarray(result[k]).size for k in ("azimuths", "elevations", "frequencies", "polarizations"))
    if kind == "complex":
        field = np.asarray(result.pop("rcs_complex"))
        if field.dtype.kind != "c" or field.shape != shape:
            raise ValueError(f"{label}: invalid native complex shape or dtype.")
        for selection in grid_blocks(shape):
            block = field[selection]
            valid = (np.isfinite(block.real) & np.isfinite(block.imag)) | (np.isnan(block.real) & np.isnan(block.imag))
            if np.any(~valid):
                raise ValueError(f"{label}: native complex samples contain infinity or one-sided NaN.")
        raw = {k: result[k] for k in ("rcs_amp_real", "rcs_amp_imag") if k in result}
        if domain == "raw_field" and raw:
            raise ValueError(f"{label}: duplicate native raw complex samples.")
        # Keep the established GHOST compatibility precision for raw-field
        # files, while retaining only the original complex authority in RAM.
        result = legacy_components(result, field, domain=domain,
                                   component_dtype=np.float32 if domain == "raw_field" else field.real.dtype)
        if domain == "physical" and raw:
            result.update(raw)
        if str(result.get("rcs_domain", "")) == "complex_amplitude":
            result["rcs_domain"] = np.asarray("power_phase")
    elif kind == "power":
        power = np.asarray(result["rcs_power"])
        if power.shape != shape or power.dtype.kind not in "iuf":
            raise ValueError(f"{label}: invalid native power shape or dtype.")
        result["rcs_phase"] = SampleComponent(power, "missing_phase")
    for key in FRAMING:
        result.pop(key, None)
    return result


def encode_native_samples(payload):
    """Compact current values, retaining exceptional/edited power-phase values."""
    data = decode_native_samples(payload)
    result = {k: v for k, v in data.items() if k not in SAMPLES | FRAMING | {"_amp", "_amp_from_power_phase", "_pol_primary"}}
    power, phase = data["rcs_power"], data.get("rcs_phase")
    units, scale = units_and_scale(data)
    shape = tuple(len(data[k]) for k in ("azimuths", "elevations", "frequencies", "polarizations"))
    raw_keys = {"rcs_amp_real", "rcs_amp_imag"} & set(data)
    if raw_keys and len(raw_keys) != 2:
        raise ValueError("Native raw amplitude requires both real and imaginary components")
    has_raw = "rcs_amp_real" in data and "rcs_amp_imag" in data
    field, domain = (raw_amplitude(data), "raw_field") if has_raw else (None, "physical")
    canonical, phase_absent = phase is not None, True
    edited_components = False
    for selection in grid_blocks(shape):
        p = np.asarray(power[selection])
        q = np.asarray(phase[selection]) if phase is not None else np.full(p.shape, np.nan)
        phase_absent &= bool(np.all(np.isnan(q)))
        low, high = (0., 2. * np.pi) if units.get("phase_wrap") == "0_360" else (-np.pi, np.pi)
        in_range = (q >= low) & ((q < high) if units.get("phase_wrap") == "0_360" else (q <= high))
        canonical &= bool(np.all(np.isfinite(q) & in_range))
        canonical &= bool(np.all(np.isfinite(p) & (p >= 0)))
        canonical &= bool(np.all((p != 0) | (q == 0)))
        if field is not None:
            if scale is None:
                canonical = False
                continue
            factor = np.broadcast_to(scale, shape)[selection] if np.ndim(scale) else scale
            selected = field[selection]
            expected = (selected.real * selected.real + selected.imag * selected.imag) * np.asarray(factor)**2
            # Round only to the supplied component precision. A compatibility
            # edit must not disappear merely because it is physically close.
            precision = np.finfo(p.dtype if p.dtype.kind == "f" else float)
            tolerance = 4. * precision.eps * np.maximum(expected, p) + precision.tiny
            expected_phase = np.angle(selected)
            expected_phase = np.where((selected.real == 0.) & (selected.imag == 0.), 0., expected_phase)
            if units.get("phase_wrap") == "0_360":
                expected_phase = np.mod(expected_phase, 2. * np.pi)
            phase_precision = np.finfo(q.dtype if q.dtype.kind == "f" else float)
            phase_tolerance = 4. * phase_precision.eps * np.maximum(1., np.abs(q))
            canonical &= bool(np.all(np.abs(expected - p) <= tolerance)
                              and np.all(np.abs(q - expected_phase) <= phase_tolerance))
            # A materialized compatibility component may have been edited.
            # Require exact stored-precision reconstruction for those values,
            # so small deliberate edits never disappear behind the tolerance.
            if isinstance(power, SampleComponent) and power._detached is not None:
                exact = np.array_equal(expected.astype(p.dtype), p)
                canonical &= exact
                edited_components |= not exact
            if isinstance(phase, SampleComponent) and phase._detached is not None:
                exact = np.array_equal(expected_phase.astype(q.dtype), q)
                canonical &= exact
                edited_components |= not exact
    if canonical:
        if field is None:
            field = np.sqrt(np.asarray(power)) * np.exp(1j * np.asarray(phase))
        result.update(grim_format_version=np.int16(2), storage_kind="complex", storage_domain=domain, rcs_complex=field)
        if domain == "raw_field":
            result["raw_complex_amplitude_preserved"] = np.asarray(True)
        return result
    if edited_components:
        # The physical values were explicitly changed without updating the
        # raw field. Retaining that pair would let a tolerant reader promote
        # stale raw samples and silently undo the edit.
        has_raw = False
        result["raw_complex_amplitude_preserved"] = np.asarray(False)
    result.update(grim_format_version=np.int16(2), storage_kind="power" if phase_absent and not has_raw else "power_phase",
                  storage_domain="physical", rcs_power=np.asarray(power))
    if result["storage_kind"] == "power_phase":
        result["rcs_phase"] = np.asarray(phase) if phase is not None else np.full(shape, np.nan)
    if has_raw:
        result["rcs_amp_real"], result["rcs_amp_imag"] = np.asarray(data["rcs_amp_real"]), np.asarray(data["rcs_amp_imag"])
    return result
