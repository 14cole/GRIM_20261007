"""File-to-output checks for per-response Assembly modeling controls."""
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BACKEND), str(BACKEND.parent)]
from ghost_backend.assembly import components, fields, workflow
from ghost_backend.assembly.inspector import ContributionInspector
from test_feature_production_contracts import manifest


def _save(path, amplitude, azimuths, elevations, polarizations, quantity):
    amplitude = np.asarray(amplitude, complex)
    k = 2 * np.pi * 3.e9 / fields.C0
    scale = 4 * np.pi if quantity == "sigma_3d" else 1 / (4 * k)
    payload = dict(
        azimuths=np.asarray(azimuths, float),
        elevations=np.asarray(elevations, float), frequencies=np.asarray([3.]),
        polarizations=np.asarray(polarizations),
        rcs_power=scale * abs(amplitude) ** 2, rcs_phase=np.angle(amplitude),
        rcs_amp_real=amplitude.real, rcs_amp_imag=amplitude.imag,
        raw_complex_amplitude_preserved=True,
        units=json.dumps(dict(azimuth="deg", elevation="deg", frequency="GHz",
            rcs_linear_quantity=quantity,
            rcs_log_unit="dBsm" if quantity == "sigma_3d" else "dBke")),
        rcs_domain="power_phase" if path.name == "body.grim" else "delta",
        power_domain="linear_rcs", history="Analytic response settings fixture",
        source_path="",
    )
    if path.name == "body.grim":
        payload.update(combine_role="coherent",
            phase_reference=components.COMPONENT_PHASE_REFERENCE,
            amplitude_convention=components.COMPONENT_AMPLITUDE_CONVENTION,
            complex_field_domain=components.COMPONENT_COMPLEX_FIELD_DOMAIN)
    elif quantity == "sigma_3d":
        payload.update(fields.point_pattern_convention_metadata())
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _fixture(root, kind):
    elevations = [-30., 5., 30., 90.]
    body = np.zeros((1, len(elevations), 1, 3), complex)
    body[..., :2] = .07 + .03j
    _save(root / "body.grim", body, [0.], elevations, ["VV", "HH", "VH"], "sigma_3d")
    (root / "plane.stl").write_text(
        "solid plane\nfacet normal 0 0 1\nouter loop\n"
        "vertex -1 -1 0\nvertex 1 -1 0\nvertex 1 1 0\n"
        "endloop\nendfacet\nfacet normal 0 0 1\nouter loop\n"
        "vertex -1 -1 0\nvertex 1 1 0\nvertex -1 1 0\n"
        "endloop\nendfacet\nendsolid plane\n", encoding="utf-8")
    options = {}
    if kind == "line":
        amplitude = np.full((181, 1, 1, 2), 2 + 1j, complex)
        _save(root / "line.grim", amplitude, np.arange(181), [0.], ["TM", "TE"], "sigma_2d")
        (root / "lines.csv").write_text(
            "line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z\n"
            "line_1,line,1,-.115,0,0,.115,0,0,0,0,1,0,0,1\n", encoding="utf-8")
        options.update(line_locations_csv=root / "lines.csv", line_datasets={"line": root / "line.grim"})
    else:
        azimuths, local_elevations = np.arange(0., 361., 5.), np.arange(-90., 91., 5.)
        amplitude = np.zeros((len(azimuths), len(local_elevations), 1, 3), complex)
        amplitude[..., :2] = .02 + .01j
        _save(root / "point.grim", amplitude, azimuths, local_elevations, ["VV", "HH", "VH"], "sigma_3d")
        (root / "points.csv").write_text(
            "placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z\n"
            "point_1,point,0,0,0,0,0,1,1,0,0\n", encoding="utf-8")
        options.update(point_locations_csv=root / "points.csv", point_datasets={"point": root / "point.grim"})
    return workflow.FeatureAssemblyRequest(
        root / "body.grim", root / "total.grim", coordinate_units="meters",
        surface_mesh=root / "plane.stl", surface_units="meters", shadow=False, **options)


class ResponseSettingsTests(unittest.TestCase):
    def test_common_phase_and_taper_changes_reach_saved_output_and_inspector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _fixture(root, "line")
            results = []
            settings_list = [
                dict(psi_tm_deg=0., psi_te_deg=0., grazing_taper_deg=0.),
                dict(psi_tm_deg=45., psi_te_deg=45., grazing_taper_deg=0.),
                dict(psi_tm_deg=0., psi_te_deg=0., grazing_taper_deg=10.),
            ]
            for index, settings in enumerate(settings_list):
                current = replace(request, output_grim=root / f"total_{index}.grim",
                    line_response_settings={"line": settings})
                plan = workflow.prepare_feature_assembly(current)
                inspector = ContributionInspector()
                samples = [inspector.evaluate(plan, 3., 0., elevation)
                    for elevation in (-30., 5., 30., 90.)]
                workflow.execute_feature_assembly(plan)
                stored = fields._load_grim(str(current.output_grim))
                for sample_index, sample in enumerate(samples):
                    np.testing.assert_allclose(stored["_amp"][0, sample_index, 0],
                        sample["body"] + sample["fields"].sum(axis=0), atol=2e-14, rtol=2e-13)
                results.append(np.asarray([sample["fields"][0] for sample in samples]))
                provenance = json.loads(str(stored["feature_provenance_json"]))[-1]
                self.assertEqual(provenance["details"]["line_response_settings"]["line"], settings)
            np.testing.assert_allclose(results[1], results[0] * np.exp(1j * np.pi / 4), atol=2e-14)
            # The cosine taper at half its width is exactly one-half in field.
            np.testing.assert_allclose(results[2][1], .5 * results[0][1], atol=2e-14)
            np.testing.assert_allclose(results[2][2:], results[0][2:], atol=2e-14)
            np.testing.assert_allclose(abs(results[0][-1, :2]), abs((2 + 1j) * .23 / (4 * np.pi)), rtol=2e-13)

    def test_two_sided_point_retains_supplied_backside_without_changing_front(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _fixture(root, "point")
            results = []
            for one_sided in (True, False):
                current = replace(request, output_grim=root / f"point_{one_sided}.grim",
                    point_response_settings={"point": {"one_sided": one_sided}})
                plan = workflow.prepare_feature_assembly(current)
                inspector = ContributionInspector()
                samples = [inspector.evaluate(plan, 3., 0., elevation) for elevation in (-30., 30.)]
                workflow.execute_feature_assembly(plan)
                stored = fields._load_grim(str(current.output_grim))
                for grid_index, sample in zip((0, 2), samples):
                    np.testing.assert_allclose(stored["_amp"][0, grid_index, 0],
                        sample["body"] + sample["fields"].sum(axis=0), atol=2e-14)
                results.append(np.asarray([sample["fields"][0] for sample in samples]))
                provenance = json.loads(str(stored["feature_provenance_json"]))[-1]
                self.assertEqual(provenance["details"]["point_response_settings"]["point"],
                    {"one_sided": one_sided})
            np.testing.assert_allclose(results[0][0], 0., atol=1e-14)
            np.testing.assert_allclose(results[1][0, :2], .02 + .01j, atol=2e-14)
            np.testing.assert_allclose(results[1][1], results[0][1], atol=2e-14)

    def test_omitted_settings_keep_legacy_behavior_and_new_settings_are_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _fixture(root, "line")
            legacy = workflow.prepare_feature_assembly(request)
            default = dict(psi_tm_deg=workflow.PSI_HH_DEG, psi_te_deg=workflow.PSI_VV_DEG,
                grazing_taper_deg=workflow.GRAZING_TAPER_DEG)
            explicit = workflow.prepare_feature_assembly(replace(request, line_response_settings={"line": default}))
            inspector = ContributionInspector()
            np.testing.assert_array_equal(inspector.evaluate(legacy, 3., 0., 5.)["fields"],
                inspector.evaluate(explicit, 3., 0., 5.)["fields"])
            changed = workflow.prepare_feature_assembly(replace(request,
                line_response_settings={"line": dict(default, psi_tm_deg=0.)}))
            self.assertNotEqual(legacy.prepared_plan_sha256, changed.prepared_plan_sha256)
            snapshot = workflow._execution_plan_snapshot(changed)
            changed.request.line_response_settings["line"]["psi_tm_deg"] = 33.
            self.assertEqual(snapshot.request.line_response_settings["line"]["psi_tm_deg"], 0.)
            with self.assertRaisesRegex(ValueError, "changed"):
                inspector.evaluate(changed, 3., 0., 5.)

    def test_nonfinite_phase_and_negative_taper_are_rejected_but_zero_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            request = _fixture(Path(directory), "line")
            for settings in ({"psi_tm_deg": float("nan")}, {"psi_te_deg": float("inf")},
                    {"grazing_taper_deg": -1.}, {"grazing_taper_deg": float("nan")}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    workflow.prepare_feature_assembly(replace(request, line_response_settings={"line": settings}))
            plan = workflow.prepare_feature_assembly(replace(request,
                line_response_settings={"line": {"grazing_taper_deg": 0.}}))
            self.assertEqual(plan.line_placements[0]["grazing_taper_deg"], 0.)

    def test_nonlegacy_manifest_phase_and_taper_are_preserved(self):
        from ghost_backend.assembly.create_feature_manifest import _validate_cli_manifest
        value = manifest("line", "line")
        value["line_phase_calibration"].update(tm_deg=0., te_deg=31., grazing_taper_deg=0.)
        result = workflow.validate_feature_library_manifest(value, dataset_id="line", feature_kind="line")
        self.assertEqual(result["line_phase_calibration"], value["line_phase_calibration"])
        cli = _validate_cli_manifest(value, dataset_id="line", feature_kind="line", label="test")
        self.assertEqual(cli["line_phase_calibration"], value["line_phase_calibration"])

    def test_different_response_controls_combine_and_keep_per_response_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _fixture(root, "line")
            with (root / "lines.csv").open("a", encoding="utf-8") as stream:
                stream.write("line_2,other,1,-.115,.25,0,.115,.25,0,0,0,1,0,0,1\n")
            controls = {
                "line": dict(psi_tm_deg=0., psi_te_deg=0., grazing_taper_deg=0.),
                "other": dict(psi_tm_deg=90., psi_te_deg=90., grazing_taper_deg=10.),
            }
            request = replace(request,
                line_datasets={"line": root / "line.grim", "other": root / "line.grim"},
                line_response_settings=controls)
            plan = workflow.prepare_feature_assembly(request)
            sample = ContributionInspector().evaluate(plan, 3., 0., 90.)
            np.testing.assert_allclose(sample["fields"][1], 1j * sample["fields"][0], atol=2e-14)
            workflow.execute_feature_assembly(plan)
            result = fields._load_grim(str(request.output_grim))
            np.testing.assert_allclose(result["_amp"][0, -1, 0],
                sample["body"] + sample["fields"].sum(axis=0), atol=2e-14)
            provenance = json.loads(str(result["feature_provenance_json"]))[-1]
            self.assertEqual(provenance["line_phase_mapping_deg"], {"TM": None, "TE": None})
            self.assertIsNone(provenance["line_grazing_taper_deg"])
            self.assertEqual(provenance["details"]["line_response_settings"], controls)

    def test_manifest_mismatch_is_advisory_and_selected_values_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _fixture(root, "line")
            value = manifest("line", "line",
                response_content_sha256=workflow.feature_response_content_sha256(root / "line.grim"))
            value["line_phase_calibration"].update(tm_deg=0., te_deg=0., grazing_taper_deg=0.)
            (root / "line.feature.json").write_text(json.dumps(value), encoding="utf-8")
            settings = dict(psi_tm_deg=45., psi_te_deg=45., grazing_taper_deg=0.)
            plan = workflow.prepare_feature_assembly(replace(request, line_response_settings={"line": settings}))
            self.assertTrue(any("differs from its library validation settings" in text for text in plan.validation_warnings))
            self.assertFalse(plan.line_records[0]["response_settings_match_library_validation"])
            self.assertEqual(plan.line_records[0]["response_settings"], settings)
            sample = ContributionInspector().evaluate(plan, 3., 0., 90.)
            np.testing.assert_allclose(sample["fields"][0, :2],
                (2 + 1j) * .23 / (4 * np.pi) * np.exp(1j * np.pi / 4), atol=2e-14)


if __name__ == "__main__":
    unittest.main()
