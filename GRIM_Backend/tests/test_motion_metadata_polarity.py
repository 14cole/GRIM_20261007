"""Motion declarations retain their key polarity during ISAR preflight."""

import numpy as np
import pytest

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.plotting.modes import isar_mode


MOTION_ASSUMPTION = "a stable or motion-compensated phase center"
POSITIVE_KEYS = (
    "motion_compensation", "motion_compensated", "phase_center_stability",
    "range_alignment",
)


def point_grid(key="phase_center_motion", value="none", container="extra"):
    """Unit stationary point at the origin: its far-field phase is zero."""
    azimuths = np.linspace(-4.0, 4.0, 17)
    frequencies = np.linspace(8.0e9, 10.0e9, 17)
    units = {
        "azimuth": "deg", "elevation": "deg", "frequency": "Hz",
        "time_convention": "exp(+jwt)",
    }
    extra = {
        "phase_reference": "fixed origin",
        "measurement_geometry": "far-field monostatic",
        "range_phase_convention": "S~exp(-j*2*k*R)",
    }
    (units if container == "units" else extra)[key] = value
    return RcsGrid(
        azimuths, [0.0], frequencies, ["HH"],
        rcs=np.ones((17, 1, 17, 1), dtype=np.complex128),
        units=units, extra=extra,
    )


@pytest.mark.parametrize("container", ["units", "extra"])
@pytest.mark.parametrize("value", [False, 0, "false", "no", "none", " NONE "])
def test_false_motion_is_an_explicit_stationary_declaration(container, value):
    assumptions = []
    assert isar_mode._isar_preflight_error(
        point_grid(value=value, container=container), undeclared_out=assumptions,
    ) is None
    assert MOTION_ASSUMPTION not in assumptions


@pytest.mark.parametrize("key", POSITIVE_KEYS)
@pytest.mark.parametrize("value", [False, 0, "false", "no", "none"])
def test_false_positive_safety_declaration_remains_blocked(key, value):
    assert "motion-compensated phase center" in isar_mode._isar_preflight_error(
        point_grid(key=key, value=value), legacy_metadata_attested=True,
    )


@pytest.mark.parametrize("key", POSITIVE_KEYS)
@pytest.mark.parametrize("value", [True, "yes"])
def test_true_positive_safety_declaration_is_explicitly_safe(key, value):
    assumptions = []
    assert isar_mode._isar_preflight_error(
        point_grid(key=key, value=value), undeclared_out=assumptions,
    ) is None
    assert MOTION_ASSUMPTION not in assumptions


@pytest.mark.parametrize("value", [True, 1, "true", "yes", "moving"])
def test_present_motion_remains_blocked(value):
    assert "motion-compensated phase center" in isar_mode._isar_preflight_error(
        point_grid(value=value), legacy_metadata_attested=True,
    )


@pytest.mark.parametrize("value", [
    "no motion", "without motion", "no drift", "fixed", "static", "stable",
])
def test_known_stationary_text_remains_explicitly_safe(value):
    assumptions = []
    assert isar_mode._isar_preflight_error(
        point_grid(value=value), undeclared_out=assumptions,
    ) is None
    assert MOTION_ASSUMPTION not in assumptions


@pytest.mark.parametrize("value", [
    "none; moving", "false but phase center drifting", "no motion; uncompensated",
    "fixed but unstable", "not fixed", "not compensated", "not aligned",
])
def test_false_or_safe_words_do_not_hide_unsafe_free_text(value):
    assert "motion-compensated phase center" in isar_mode._isar_preflight_error(
        point_grid(value=value), legacy_metadata_attested=True,
    )


def test_conflicting_scalar_sources_remain_blocked():
    grid = point_grid(value="none")
    grid.units["phase_center_motion"] = "moving"
    with pytest.raises(ValueError, match="contradictory phase_center_motion"):
        isar_mode.form_isar(grid)


def test_other_unsafe_key_is_not_overridden_by_absent_motion():
    grid = point_grid(value="none")
    grid.extra["motion_compensated"] = False
    with pytest.raises(ValueError, match="motion-compensated phase center"):
        isar_mode.form_isar(grid, legacy_metadata_attested=True)


@pytest.mark.parametrize("value", ["N/A", "producer-specific description"])
def test_unknown_or_placeholder_motion_remains_an_assumption(value):
    assumptions = []
    assert isar_mode._isar_preflight_error(
        point_grid(value=value), undeclared_out=assumptions,
    ) is None
    assert MOTION_ASSUMPTION in assumptions


@pytest.mark.parametrize("container", ["units", "extra"])
@pytest.mark.parametrize("reconstruction", ["fast", "accurate"])
def test_none_metadata_forms_stationary_point_without_changing_samples(
    container, reconstruction,
):
    grid = point_grid(value="none", container=container)
    original_field = grid.rcs.copy()
    bands, _elapsed = isar_mode.form_isar(
        grid, reconstruction=reconstruction, aperture_mode="coherent",
        window="Rectangular", retain_complex=True,
    )
    assert len(bands) == 1
    band = bands[0]
    magnitude = band["magnitude"]
    peak = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    assert band["x_range"][peak[0]] == pytest.approx(0.0, abs=1.0e-12)
    assert band["y_range"][peak[1]] == pytest.approx(0.0, abs=1.0e-12)
    assert magnitude[peak] == pytest.approx(1.0, abs=2.0e-3)
    assert np.all(np.isfinite(magnitude))
    assert MOTION_ASSUMPTION not in band["isar_contract_undeclared_fields"]
    assert not band["isar_contract_user_assumed"]
    np.testing.assert_array_equal(grid.rcs, original_field)
    assert getattr(grid, container)["phase_center_motion"] == "none"


def test_cache_does_not_allow_later_unsafe_metadata():
    grid = point_grid(value="none")
    isar_mode.form_isar(grid)
    grid.extra["phase_center_motion"] = "moving"
    with pytest.raises(ValueError, match="motion-compensated phase center"):
        isar_mode.form_isar(grid, legacy_metadata_attested=True)
