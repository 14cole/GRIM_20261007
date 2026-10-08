"""Material junctions, thin clearances, and solver-aware BoR validation."""
import sys
from pathlib import Path
from unittest import mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.geometry.io import Segment, parse_geometry
from ghost_backend.geometry.validation import GeometryAudit
DIELECTRICS = [["1", "2.5", "0", "1", "0"], ["2", "4", "0", "1", "0"]]


def segment(name, points, kind=2, pos=0, neg=0, ibc=0):
    pairs = list(zip(points, points[1:]))
    return Segment(name, str(kind), [str(kind), "1", str(ibc), str(pos), str(neg)],
                   [x for a, b in pairs for x, _ in (a, b)],
                   [y for a, b in pairs for _, y in (a, b)])


def audit(segments, mode="2d", ibcs=None):
    return GeometryAudit(segments, ibcs or [], DIELECTRICS, ".", mode=mode).run()


def errors(findings):
    return [message for severity, _, message in findings if severity == "ERROR"]


def junctions(findings):
    return [message for message in errors(findings) if "material-side mismatch" in message]


def test_connected_material_mismatch_marks_both_rows():
    pieces = [segment("upper", [(-1, 0), (0, 0)], 3, 1),
              segment("lower", [(0, 0), (1, 0)], 3, 2)]
    findings, rows = audit(pieces)
    assert junctions(findings)
    assert {0, 1} <= rows
    assert "material 1" in junctions(findings)[0]
    assert "material 2" in junctions(findings)[0]


def test_valid_three_material_junction_and_wrong_coating():
    pieces = [segment("air coating", [(0, 0), (1, 0)], 3, 1),
              segment("bare conductor", [(-.5, .866), (0, 0)], 2),
              segment("covered conductor", [(0, 0), (-.5, -.866)], 4, 1)]
    assert not junctions(audit(pieces)[0])
    pieces[2].properties[3] = "2"
    findings, rows = audit(pieces)
    assert junctions(findings)
    assert {0, 2} <= rows


def test_reversed_type5_requires_swapping_material_sides():
    pieces = [segment("first", [(-1, 0), (0, 0)], 5, 1, 2),
              segment("second", [(1, 0), (0, 0)], 5, 1, 2)]
    assert junctions(audit(pieces)[0])
    pieces[1].properties[3:5] = ["2", "1"]
    assert not junctions(audit(pieces)[0])


def test_endpoint_weld_handles_rounding_cell_boundary():
    pieces = [segment("first", [(-1, 0), (.99e-9, 0)], 3, 1),
              segment("second", [(1.01e-9, 0), (1, 0)], 3, 2)]
    assert junctions(audit(pieces)[0])


def test_thin_clearances_are_not_welded_at_large_drawing_scale():
    pieces = [segment("near a", [(0, 0), (1, 0)], 1),
              segment("near b", [(0, .01), (1, .01)], 1),
              segment("far", [(20000, 0), (20001, 0)], 1)]
    assert not errors(audit(pieces)[0])


@pytest.mark.parametrize("end", [(1, 0), (.5, 0)])
def test_duplicate_and_partial_overlap_detected_with_shared_tip(end):
    pieces = [segment("first", [(0, 0), (1, 0)], 1),
              segment("second", [(0, 0), end], 1)]
    findings, rows = audit(pieces)
    assert any("overlapping or duplicate" in message for message in errors(findings))
    assert rows == {0, 1}


def test_valid_bor_axis_to_axis_profile_and_2d_warning():
    pieces = [segment("body", [(0, 2), (1, 2), (1, -2), (0, -2)])]
    findings, rows = audit(pieces, "bor")
    assert not errors(findings)
    assert not any(severity == "WARN" for severity, _, _ in findings)
    assert not rows
    assert any("solver geometry preflight" in message for _, _, message in findings)
    assert any("dangling" in message for _, _, message in audit(pieces, "2d")[0])


def test_split_bor_profile_does_not_require_each_row_to_reach_axis():
    pieces = [segment("top", [(0, 2), (1, 2)]),
              segment("wall", [(1, 2), (1, -2)]),
              segment("bottom", [(1, -2), (0, -2)])]
    findings, _ = audit(pieces, "bor")
    assert not errors(findings)
    assert not any(severity == "WARN" for severity, _, _ in findings)


@pytest.mark.parametrize("points, expected", [
    ([(0, 2), (-1, 2), (-1, -2), (0, -2)], "crosses the rotation axis"),
    ([(.2, 2), (1, 2), (1, -2), (0, -2)], "endpoints must lie ON"),
    ([(0, -2), (1, -2), (1, 2), (0, 2)], "bottom-to-top"),
])
def test_bor_uses_real_solver_preflight(points, expected):
    findings, _ = audit([segment("body", points)], "bor")
    assert any(expected in message for message in errors(findings))


def test_bor_sheet_allows_free_rim():
    findings, _ = audit([segment("sheet", [(1, 1), (1, -1)], 1)], "bor")
    assert not errors(findings)
    assert not any(severity == "WARN" for severity, _, _ in findings)


def test_bor_rejects_unsupported_impedance_on_dielectric():
    pieces = [segment("body", [(0, 2), (1, 2), (1, -2), (0, -2)], 3, 1, ibc=1)]
    findings, _ = audit(pieces, "bor", [["1", "constant", "10", "0", "10", "0"]])
    assert any("not implemented by the BoR" in message for message in errors(findings))


def test_bor_cancellation_is_not_reported_as_geometry_error():
    pieces = [segment("body", [(0, 2), (1, 2), (1, -2), (0, -2)])]
    with mock.patch("ghost_backend.bor.dispatch._prepare_bor_groups", side_effect=InterruptedError("stop")):
        with pytest.raises(InterruptedError):
            audit(pieces, "bor")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_coordinates_report_findings_instead_of_crashing(bad):
    findings, rows = audit([segment("bad", [(bad, 0), (1, 0)])])
    assert any("coordinates must be finite" in message for message in errors(findings))
    assert rows == {0}


def test_validation_does_not_pad_input_properties_in_place():
    item = segment("sheet", [(0, 0), (1, 0)], 1)
    item.properties = ["1", "1"]
    audit([item])
    assert item.properties == ["1", "1"]


def test_fractional_material_flag_is_not_silently_truncated():
    item = segment("wrong", [(0, 0), (1, 0)], 3, 1)
    item.properties[3] = "1.5"
    findings, _ = audit([item])
    assert any("pos_mat must be a nonnegative integer" in message for message in errors(findings))


def test_bundled_bor_cylinder_has_no_false_hanging_warnings():
    geometry = Path(__file__).resolve().parents[1] / "geometry/geometries/BOR/pec_cylinder_r4_h10_in.geo"
    _, segments, ibcs, materials = parse_geometry(geometry.read_text())
    findings, rows = GeometryAudit(segments, ibcs, materials, str(geometry.parent), mode="bor").run()
    assert not errors(findings)
    assert not any(severity == "WARN" for severity, _, _ in findings)
    assert not rows


@pytest.mark.parametrize("points, expected", [
    ([(-1, 1), (-1, -1)], "rho coordinates must be >= 0"),
    ([(0, 1), (0, -1)], "zero surface area"),
])
def test_bor_sheet_surface_safety(points, expected):
    findings, _ = audit([segment("sheet", points, 1)], "bor")
    assert any(expected in message for message in errors(findings))


def test_partial_bor_coating_accepts_valid_off_axis_material_junctions():
    from test_bor_physics_regression import _partial_coating_snapshot
    from ghost_backend.geometry.io import snapshot_to_geometry_text
    snapshot = _partial_coating_snapshot()
    for item in snapshot["segments"]:
        item["name"] = item["name"].replace(" ", "_")
    _, segments, ibcs, materials = parse_geometry(snapshot_to_geometry_text(snapshot))
    findings, rows = GeometryAudit(segments, ibcs, materials, ".", mode="bor").run()
    assert not errors(findings)
    assert not any(severity == "WARN" for severity, _, _ in findings)
    assert not rows


@pytest.mark.parametrize("mode", ["2d", "bor"])
def test_sheet_to_pure_pec_transition_is_supported(mode):
    pieces = [segment("sheet", [(1, 1), (1, 0)], 1, ibc=1),
              segment("pec screen", [(1, 0), (1, -1)], 2)]
    findings, _ = audit(pieces, mode, [["1", "constant", "100", "0", "100", "0"]])
    assert not errors(findings)
    if mode == "bor":
        assert not any(level == "WARN" for level, _, _ in findings)


@pytest.mark.parametrize("kind", [4, 5])
def test_nested_boundaries_must_agree_on_shared_material(kind):
    def square(radius):
        return [(radius, radius), (radius, -radius), (-radius, -radius),
                (-radius, radius), (radius, radius)]
    pieces = [segment("outer coating", square(2), 3, 1),
              segment("inner", square(1), kind, 2, 1 if kind == 5 else 0)]
    findings, rows = audit(pieces)
    assert any("surrounding region is material 1" in message for message in errors(findings))
    assert {0, 1} <= rows
    pieces[1].properties[3] = "1"
    if kind == 5:
        pieces[1].properties[4] = "2"
    assert not errors(audit(pieces)[0])
