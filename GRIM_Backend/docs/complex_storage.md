# Complex dataset storage

GRIM keeps one authoritative sample representation per dataset. Known-phase
data use complex samples, `I + 1j*Q`. Data with unknown phase use linear power.
Datasets with a mixture of known and unknown phases retain power and phase so
that a valid magnitude is not lost when its phase is unavailable. Exceptional
phase information that a complex number cannot encode, such as an unwrapped
turn count or an unknown phase at zero power, is retained when supplied.

For physical complex samples, linear RCS is `I**2 + Q**2`. The stored quantity
still matters: this can be RCS in square meters, 2-D scattering width, or a
dimensionless ratio. Logarithmic displays retain the dataset's corresponding
unit. Phase is in radians internally. Complex64 retains two float32 components;
complex128 retains two float64 components. Converting power and phase to I/Q
can change the last few floating-point bits of values derived afterward.

Raw solver fields keep their original normalization and precision. Their raw
real and imaginary metadata arrays are views of a single complex field, not
additional sample copies. GRIM applies the existing quantity-dependent scale
when providing physical samples. Raw solver I/Q should therefore not be
interpreted as physical RCS I/Q without that scale.

## Dataset operations

The plotting controls and operation buttons keep their mathematical meaning.
Coherent addition, subtraction, phase operations, I/Q analysis, and ISAR can
read complex samples directly. Operations that depend on power, including
statistics and incoherent sums, read or produce linear power. Unknown phase
remains unknown; GRIM does not assign a zero-degree phase to a nonzero
magnitude-only measurement.

Axis selection and exact sample rearrangement preserve the complex samples.
Operations that actually change values retain their existing interpolation,
normalization, compatibility, and provenance rules. This change does not make
every operation faster. Power plots still have to calculate power from a
complex input, while complex operations avoid repeated polar-to-Cartesian
conversion. A complex64 array takes the same sample space as two float32
arrays; savings come from removing redundant representations and repeated
temporary allocations.

## Python access

To correct a reversed stored phase sign, select datasets and use
**Dataset Operations → Transform → Flip Phase**. Each selection produces a new
dataset with the same magnitude and coordinates: I is unchanged, Q is negated,
and phase becomes its negative. Unknown phase stays unknown. Applying it twice
restores the original samples. The operation records the correction in history
and Python replay; save the result to retain it in a file.

This is an explicit correction of the samples, with the declared time convention
left unchanged. It is different from adding a constant 180-degree phase shift
or converting both the samples and their convention metadata. It can correct a
phase-sign mismatch that mirrors a range or ISAR image; it does not alter the
PTM or PIO loader's interpretation for other files.

```python
from GRIM_Backend.scripting.workspace import flip_phase
corrected = flip_phase(dataset)
```

Prefer the new read accessors for analysis:

```python
power = dataset.power[azimuth_slice, elevation_index, frequency_index, pol_index]
phase = dataset.phase[azimuth_slice, elevation_index, frequency_index, pol_index]
field = dataset.rcs_slice((azimuth_slice, elevation_index, frequency_index, pol_index))
```

`power` and `phase` expose array-like views that calculate only the requested
slice. `numpy.asarray(dataset.power)` explicitly requests the full power
array. `dataset.rcs` returns the full physical complex array; `rcs_slice`
avoids that full-grid allocation. Derived full-grid caches are not retained.

`dataset.phase_coverage()` reports whether every finite sample carries a
phase (`"complete"`), only some do (`"partial"`), or none does (`"none"`).
Complex and power-only storage answer without reading a sample; a mixed
power/phase pair is scanned in bounded blocks. Use it instead of testing
`numpy.isfinite` on the full phase array.

Use explicit methods when changing samples:

```python
dataset.set_power_phase(selection, power=new_power, phase=new_phase)
dataset.set_complex(selection, new_complex_samples)
```

The existing `rcs_power` and `rcs_phase` attributes remain available for scripts
that expect mutable NumPy arrays. Accessing them materializes the legacy
arrays. Changes to those arrays become authoritative before subsequent complex
access or saving, and stale raw field metadata is discarded. Retaining legacy
writable arrays costs additional memory and requires change detection; use the
new read accessors for efficient new scripts.

## Native file compatibility

Current GRIM reads existing `.grim` files and writes version 2 native archives.
The archives remain pickle-free NumPy ZIP files. They include
`grim_format_version=2`, `storage_kind`, and `storage_domain`:

| Storage kind | Sample arrays | Meaning |
| --- | --- | --- |
| `complex` | `rcs_complex` | Known complex samples; domain is `physical` or `raw_field`. |
| `power` | `rcs_power` | Linear power with unknown phase. |
| `power_phase` | `rcs_power`, `rcs_phase` | Mixed or exceptional phase information. |

Axes, units, history, provenance, and ancillary solver metadata remain in the
archive. A raw-field complex archive does not also save redundant raw real and
imaginary arrays; the loader restores their metadata views. Embedded dataset
payloads in assemblies use the same representation.

**Newly saved files require the updated GRIM.** Older GRIM versions and scripts
that read `rcs_power` or `rcs_phase` directly from the ZIP may need updating.
Use `RcsGrid.load()` to read either native version. Original files are not
rewritten simply by loading them.
