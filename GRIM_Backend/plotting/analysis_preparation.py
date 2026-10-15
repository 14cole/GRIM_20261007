"""Captured native-cut inputs usable without reading any Qt controls."""
from dataclasses import dataclass, replace
import numpy as np

from GRIM_Backend.plotting.engineering_tools import dataset_cut
from GRIM_Backend.plotting.modes.common import axis_unit, convert_axis_values


@dataclass(frozen=True)
class CutRequest:
    dataset: object
    label: str
    axis: str
    fixed: tuple
    polarization: str
    source: object = None
    selected: tuple | None = None
    contextual: bool = False

    @property
    def cache_key(self):
        return (id(self.dataset), self.label, self.axis, self.fixed, self.polarization,
                id(self.source), id(self.selected), self.contextual,
                *(id(getattr(self.dataset, name, None)) for name in ('_complex', '_power', '_phase')),
                repr(getattr(self.dataset, 'units', {})))

    def read(self):
        fixed = dict(self.fixed)
        if self.source is not None and self.source is not self.dataset:
            for key in ('azimuth', 'elevation', 'frequency'):
                if key == self.axis:
                    continue
                value = self.source.get_axis(key)[fixed[key + '_index']]
                converted = float(convert_axis_values([value], key, axis_unit(self.source, key),
                                                       axis_unit(self.dataset, key))[0])
                values = np.asarray(self.dataset.get_axis(key), dtype=float)
                valid = np.flatnonzero(np.isfinite(values))
                if not np.isfinite(converted) or not len(valid):
                    raise ValueError(f'Reference comparison requires a finite {key} coordinate.')
                nearest = int(valid[np.argmin(np.abs(values[valid] - converted))])
                tolerance = np.finfo(float).eps * max(abs(converted), float(np.max(np.abs(values[valid]))), np.finfo(float).tiny) * 32
                if abs(values[nearest] - converted) > tolerance:
                    raise ValueError(f'{self.label} has no matching {key}={converted:.9g} {axis_unit(self.dataset, key)} cut.')
                fixed[key + '_index'] = nearest
        cut = dataset_cut(self.dataset, self.axis, polarization=self.polarization,
                          label=self.label, sweep_indices=self.selected, **fixed)
        if not self.contextual:
            return cut
        context = (f'Main selection: {len(cut.x)} of {len(self.dataset.get_axis(self.axis))} native {self.axis} samples, captured at Update.'
                   if self.selected is not None else 'Full native sweep; all sweep-axis samples.')
        return replace(cut, notes=[*cut.notes, context])


def capture_cut(panel, *, reference=False, polarization=None, dataset_index=None,
                selected=None, contextual=False):
    index = (panel.reference_combo.currentIndex() if reference else panel.dataset_combo.currentIndex()) if dataset_index is None else dataset_index
    if not 0 <= index < len(panel._datasets):
        raise ValueError('Select a dataset and refresh the analysis sources.')
    name, dataset = panel._datasets[index]
    source_index = panel.dataset_combo.currentIndex()
    source = panel._datasets[source_index][1] if 0 <= source_index < len(panel._datasets) and index != source_index else None
    fixed = tuple((key + '_index', int(panel.fixed[key].currentData() or 0)) for key in ('azimuth', 'elevation', 'frequency'))
    return CutRequest(dataset, name, str(panel.axis_combo.currentData()), fixed,
                      polarization or panel.fixed['polarization'].currentText(), source, selected, contextual)


def cached_cut(request, cache, maximum_bytes=64 * 1024**2):
    # The public legacy arrays permit arbitrary in-place edits. They have no
    # cheap mutation counter, so never serve a cached cut from mutable exposure.
    def mutable_samples(dataset):
        return (getattr(dataset, '_legacy_samples', False) or
                (getattr(dataset, '_storage_domain', None) != 'raw_field' and
                 any(name in getattr(dataset, 'extra', {}) for name in ('rcs_amp_real', 'rcs_amp_imag'))))
    if mutable_samples(request.dataset) or mutable_samples(request.source):
        return request.read()
    key = request.cache_key
    if key in cache:
        return cache[key]
    cut = request.read()
    def size(value):
        return sum(getattr(getattr(value, name), 'nbytes', 0) for name in ('x', 'power', 'field', 'db', 'native_indices'))
    needed = size(cut)
    while cache and (len(cache) >= 8 or sum(size(value) for value in cache.values()) + needed > maximum_bytes):
        cache.pop(next(iter(cache)))
    if needed <= maximum_bytes:
        cache[key] = cut
    return cut
