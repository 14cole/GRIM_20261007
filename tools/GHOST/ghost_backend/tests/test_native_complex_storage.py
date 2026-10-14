"""Current GRIM archives remain usable by standalone GHOST services."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ghost_backend" / "data_tools"))

from ghost_backend.assembly.fields import _load_grim, _load_grim_sample, C0
from ghost_backend.assembly.expand_wing_sections import _load_base
from ghost_backend.hpc.common import read_unit_grims
from cem_tools.grim_native import load_grim
from cem_tools.native_samples import decode_native_samples


def payload_for(domain="raw_field", quantity="sigma_3d", fortran=False):
    field = np.arange(24, dtype=float).reshape(3, 2, 2, 2).astype(complex) + 1.e-13j
    field[1, 1, 1, 1] = -2. - 3.e-12j
    if fortran:
        field = np.asfortranarray(field)
    frequencies = np.array([2., 3.])
    scale = (np.sqrt(4. * np.pi) if quantity == "sigma_3d" else
             (1. / (2. * np.sqrt(2. * np.pi * frequencies * 1.e9 / C0)))[None, None, :, None])
    physical = field * scale
    stored = field if domain == "raw_field" else physical
    return dict(azimuths=np.array([0., 1., 2.]), elevations=np.array([0., 1.]),
                frequencies=frequencies, polarizations=np.array(["VV", "HH"]),
                units=json.dumps({"frequency": "GHz", "rcs_linear_quantity": quantity}),
                grim_format_version=2, storage_kind="complex", storage_domain=domain,
                rcs_complex=np.asfortranarray(stored) if fortran else stored,
                rcs_domain="complex_amplitude", power_domain="linear_rcs"), field, physical


def save(path, payload):
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


@pytest.mark.parametrize("domain", ["raw_field", "physical"])
@pytest.mark.parametrize("quantity", ["sigma_3d", "sigma_2d"])
@pytest.mark.parametrize("fortran", [False, True])
def test_full_and_bounded_assembly_readers_preserve_native_complex(tmp_path, domain, quantity, fortran):
    payload, raw, physical = payload_for(domain, quantity, fortran)
    path = tmp_path / "compact.grim"
    save(path, payload)
    whole = _load_grim(path)
    single = _load_grim_sample(path, frequency=3., azimuth=1., elevation=1.)
    np.testing.assert_allclose(whole["_amp"], raw, rtol=2.e-15, atol=1.e-28)
    np.testing.assert_allclose(single["_amp"], raw[1:2, 1:2, 1:2], rtol=2.e-15, atol=1.e-28)
    np.testing.assert_allclose(whole["rcs_power"], np.abs(physical) ** 2, rtol=2.e-15)
    assert "rcs_complex" not in whole and "grim_format_version" not in whole


def test_standalone_data_tools_reads_compact_authority(tmp_path):
    payload, raw, _ = payload_for()
    path = tmp_path / "compact.grim"
    save(path, payload)
    restored = load_grim(path)
    np.testing.assert_array_equal(restored["rcs_amp_real"], raw.real)
    np.testing.assert_array_equal(restored["rcs_amp_imag"], raw.imag)
    assert "grim_format_version" not in restored


def test_wing_base_reader_accepts_compact_complex(tmp_path):
    payload, raw, _ = payload_for()
    path = tmp_path / "body.grim"
    save(path, payload)
    restored, field, channels = _load_base(path)
    np.testing.assert_array_equal(field, raw)
    assert channels == {"vv": 0, "hh": 1}
    assert "grim_format_version" not in restored


def test_hpc_result_reader_accepts_compact_dual_channel(tmp_path):
    payload, raw, _ = payload_for()
    payload["elevations"] = np.array([0.])
    payload["frequencies"] = np.array([3.])
    payload["rcs_complex"] = raw[:, :1, 1:2]
    payload["solver_metadata_json"] = json.dumps({"source": "solver"})
    path = tmp_path / "3GHz_compact.grim"
    save(path, payload)
    results = read_unit_grims(tmp_path)
    assert len(results) == 1
    np.testing.assert_array_equal(results[0]["amp"], raw[:, 0, 1:2])
    assert results[0]["solver_audit"] == {"source": "solver"}


def test_native_power_only_adapter_does_not_invent_phase():
    payload, _, _ = payload_for()
    field = payload.pop("rcs_complex")
    payload.update(storage_kind="power", storage_domain="physical", rcs_power=np.abs(field) ** 2)
    restored = decode_native_samples(payload)
    assert np.isnan(restored["rcs_phase"]).all()
    assert "rcs_amp_real" not in restored


@pytest.mark.parametrize("edit", [
    {"grim_format_version": 3}, {"storage_kind": "invalid"},
    {"storage_domain": "invalid"}, {"rcs_power": np.ones((3, 2, 2, 2))},
    {"rcs_complex": np.full((3, 2, 2, 2), complex(np.nan, 0.))},
])
def test_invalid_compact_framing_is_rejected(edit):
    payload, _, _ = payload_for()
    payload.update(edit)
    with pytest.raises(ValueError):
        decode_native_samples(payload)
