"""Family-first vehicle authoring, backed by the existing strict CSV workflow.

Edits are staged in a new managed file and parsed before the live form changes.
Imported placements are source documents and are never modified by these actions.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from .model import (
    FeatureAssemblyFormModel, LINE_PLACEMENT_COLUMNS, POINT_PLACEMENT_COLUMNS,
    UNIT_SCALE_M, _resolved_user_path, coerce_feature_workflow,
)

MAX_AUTHORED_ROWS = 10_000
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def _kind(kind):
    if kind not in ("point", "line"):
        raise ValueError("Feature kind must be point or line.")
    return kind


def _unique_name(name, used):
    stem = re.sub(r"[^\w-]+", "_", str(name).strip()).strip("_") or "feature"
    candidate, suffix = stem, 2
    while candidate in used:
        candidate, suffix = f"{stem}_{suffix}", suffix + 1
    return candidate


def _read_rows(values, kind):
    raw = getattr(values, f"{kind}_locations_csv")
    if not raw:
        return []
    source = _resolved_user_path(raw, base_dir=values.base_dir)
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("The placement editor supports files up to 16 MiB.")
    columns = POINT_PLACEMENT_COLUMNS if kind == "point" else LINE_PLACEMENT_COLUMNS
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != columns:
            raise ValueError(f"The {kind} placement CSV has an unsupported header.")
        rows = list(reader)
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"The {kind} placement CSV contains an incomplete row.")
    if len(rows) > MAX_AUTHORED_ROWS:
        raise ValueError("The placement editor supports at most 10,000 rows.")
    return rows


@dataclass(frozen=True)
class VehicleFeatureEdit:
    model: FeatureAssemblyFormModel
    dataset_id: str
    created_path: Path | None
    instance_count: int


def _stage_rows(values, kind, rows, *, directory, service, dataset_id, instance_count):
    """Create and validate a candidate without touching the supplied values."""
    created = None
    candidate = FeatureAssemblyFormModel(copy.deepcopy(values))
    try:
        if rows:
            if len(rows) > MAX_AUTHORED_ROWS:
                raise ValueError("This addition exceeds the 10,000-row placement editor limit.")
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            columns = POINT_PLACEMENT_COLUMNS if kind == "point" else LINE_PLACEMENT_COLUMNS
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                    prefix=f"vehicle_{kind}_", suffix=".csv", dir=directory,
                    delete=False) as stream:
                created = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            setattr(candidate.values, f"{kind}_locations_csv", str(created.resolve()))
        else:
            setattr(candidate.values, f"{kind}_locations_csv", "")
        if candidate.values.point_locations_csv or candidate.values.line_locations_csv:
            candidate.discover_dataset_ids(coerce_feature_workflow(service))
        else:
            candidate.update_dataset_requirements({"point_dataset_ids": (), "line_dataset_ids": ()})
        return VehicleFeatureEdit(candidate, dataset_id, created, instance_count)
    except Exception:
        if created is not None:
            created.unlink(missing_ok=True)
        raise


def prepare_vehicle_feature(values, definition, *, directory, service):
    """Stage one new response family, assigning collision-free stable IDs."""
    kind = _kind(definition.get("kind"))
    name = str(definition.get("name", "")).strip()
    if not name:
        raise ValueError("Name this feature family.")
    units = str(definition.get("units", ""))
    if units not in UNIT_SCALE_M:
        raise ValueError("Choose placement units before adding a feature.")
    if values.coordinate_units and units != values.coordinate_units:
        raise ValueError("New features must use the Assembly placement units.")
    if (values.point_locations_csv or values.line_locations_csv) and not values.coordinate_units:
        raise ValueError("Choose the units of the existing placements before adding a feature.")
    response_text = str(definition.get("response", "")).strip()
    if not response_text:
        raise ValueError("Choose a response file for this feature.")
    response = _resolved_user_path(response_text, base_dir=values.base_dir)
    if response.suffix.lower() != ".grim" or not response.is_file():
        raise ValueError("Choose an existing .grim response file for this feature.")
    previous = _read_rows(values, kind)
    columns = POINT_PLACEMENT_COLUMNS if kind == "point" else LINE_PLACEMENT_COLUMNS
    added = [list(map(str, row)) for row in definition.get("rows", ())]
    if not added or any(len(row) != len(columns) for row in added):
        raise ValueError(f"Provide complete {kind} placements for this feature.")
    mappings = dict(getattr(values, f"{kind}_datasets"))
    dataset_id = _unique_name(name, set(mappings) | {row[1] for row in previous})
    used_ids = {row[0] for row in previous}
    translated = {}
    for index, row in enumerate(added):
        original_id = row[0]
        if not original_id.strip():
            raise ValueError("Every placement needs an identifier.")
        if kind == "point" or original_id not in translated:
            new_id = _unique_name(original_id, used_ids)
            used_ids.add(new_id)
            if kind == "line":
                translated[original_id] = new_id
        else:
            new_id = translated[original_id]
        row[0], row[1] = new_id, dataset_id
    candidate_values = copy.deepcopy(values)
    candidate_values.coordinate_units = units
    mappings[dataset_id] = str(response)
    setattr(candidate_values, f"{kind}_datasets", mappings)
    return _stage_rows(candidate_values, kind, previous + added, directory=directory,
                       service=service, dataset_id=dataset_id,
                       instance_count=len({row[0] for row in added}))


def prepare_vehicle_feature_removal(values, kind, dataset_id, *, directory, service):
    """Remove a whole response family from a newly staged placement document."""
    kind = _kind(kind)
    rows = _read_rows(values, kind)
    removed = [row for row in rows if row[1] == dataset_id]
    if not removed:
        raise ValueError("Select a feature family that has placements.")
    candidate_values = copy.deepcopy(values)
    getattr(candidate_values, f"{kind}_datasets").pop(dataset_id, None)
    exclusions = (candidate_values.excluded_point_placement_ids if kind == "point"
                  else candidate_values.excluded_line_ids)
    exclusions.difference_update(row[0] for row in removed)
    return _stage_rows(candidate_values, kind, [row for row in rows if row[1] != dataset_id],
                       directory=directory, service=service, dataset_id=dataset_id,
                       instance_count=len({row[0] for row in removed}))


def portable_vehicle_values(values, recipe_target, *, draft_directory):
    """Copy managed placement documents beside a recipe; leave imports intact.

    Returns a separate values object so saving does not discard a validated plan.
    Unique files preserve earlier recipe variants and support safe Save As.
    """
    result = copy.deepcopy(values)
    target = Path(recipe_target).expanduser().resolve()
    draft_directory = Path(draft_directory).resolve()
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Assembly recipe folder does not exist: {target.parent}")
    created = []
    try:
        for kind in ("point", "line"):
            raw = getattr(result, f"{kind}_locations_csv")
            if not raw:
                continue
            source = _resolved_user_path(raw, base_dir=values.base_dir)
            if source.parent != draft_directory or target.parent == draft_directory:
                continue
            with tempfile.NamedTemporaryFile(prefix=f"{target.name.removesuffix('.assembly.json')}_{kind}_",
                    suffix=".csv", dir=target.parent, delete=False) as stream:
                destination = Path(stream.name)
            created.append(destination)
            shutil.copyfile(source, destination)
            setattr(result, f"{kind}_locations_csv", str(destination))
        return result
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


class VehicleBuilderMixin:
    def clear_all(self):
        """Start an empty vehicle without modifying any source or output file."""
        from PySide6.QtCore import QSignalBlocker
        from .model import AssemblyWorkEstimate
        from .values import FeatureAssemblyValues

        # A finished thread may still have queued results awaiting delivery.
        # Wait until the normal completion handler has released it as well.
        if self._thread is not None:
            self.status_changed.emit("Wait for the current Assembly operation before clearing.")
            return False
        try:
            self._ensure_vehicle_editable()
        except ValueError as exc:
            self.status_changed.emit(str(exc))
            return False

        self._geometry_preview_timer.stop()
        self._geometry_preview_pending = False
        self._calculate_pending = False
        # A library search belongs to the old mapping table. Its completion
        # checks cancellation before presenting suggestions.
        for mapping in (self.point_mapping, self.line_mapping):
            scan = getattr(mapping, "_library_scan", None)
            if scan is not None:
                scan.cancel.set()
                progress = getattr(mapping, "_library_progress", None)
                if progress is not None:
                    progress.close()

        # Do not pull/validate the old form: Clear all must also recover from
        # malformed entries. Suppress intermediate signals and preview jobs.
        sections = (self.body_geometry_section, self.feature_selection_section,
                    self.advanced_section, self.readiness_section, self.study_section,
                    self.model_scope_section, self.preview_guide)
        # Block only stable controls: mapping-row children are deleted while
        # clearing their tables and must not be retained by QSignalBlocker.
        controls_to_block = (
            self.coordinate_units, self.surface_units, self.flip_normals,
            self.shadow, self.skin_tol, self.phase_tol, self.normal_tol,
            self.validation_profile, self.point_mapping, self.line_mapping,
            self.spatial_feature_filter, self.spatial_feature_tree,
            self.recipe_name_edit, self.recipe_variant_edit,
            self.validation_warning_ack, self.point_format_button,
            self.line_format_button, self.workflow_tabs,
            self.wing_table, self.wing_corner_table,
            *(section.header for section in sections),
            *(controls["advanced"].header for controls in self.vehicle_feature_controls.values()),
        )
        blockers = [QSignalBlocker(control) for control in controls_to_block]
        self._loading_recipe = True
        try:
            values = FeatureAssemblyValues()
            self.model = FeatureAssemblyFormModel(values)
            for picker in (self.base_picker, self.surface_picker, self.output_picker,
                           self.point_csv_picker, self.line_csv_picker, self.wing_output_picker):
                picker.set_path("")
            self.coordinate_units.setCurrentIndex(self.coordinate_units.findData(""))
            self.surface_units.setCurrentIndex(self.surface_units.findData(""))
            self.flip_normals.setChecked(False)
            self.shadow.setChecked(False)
            self.shadow_bias.clear()
            self.skin_tol.setValue(values.skin_tol_m / UNIT_SCALE_M["inches"])
            self.phase_tol.setValue(values.skin_phase_tol_deg)
            self.normal_tol.setValue(values.normal_tol_deg)
            self._set_validation_profile_from_values(values)
            for control in (*self.study_fields.values(), *self.wing_grid_fields.values()):
                control.clear()
            self.point_mapping.set_dataset_ids(())
            self.line_mapping.set_dataset_ids(())
            self.spatial_feature_filter.clear()
            self.wing_table.setRowCount(0)
            self.wing_corner_table.setRowCount(0)
            self.wing_geometry_units.setCurrentIndex(0)
            self.wing_angle_step.setValue(0.5)
            self.wing_mirror.setChecked(False)
            self.wing_oblique.setChecked(False)
            self.wing_shadow.setChecked(True)
            self.wing_use_as_body.setChecked(False)
            self._recipe_path = None
            self._recipe_dirty = False
            self._recipe_source_warnings = ()
            self._draft_path = self._draft_path.with_name(uuid.uuid4().hex + ".assembly.json")
            self.recipe_name_edit.setText("Vehicle feature assembly")
            self.recipe_variant_edit.setText("Baseline")
            self.recipe_dialog.hide()
            self._discovery_paths = None
            self._preview_is_current = False
            self._validated_plan_current = False
            self._surface_binding_checked_key = None
            self._surface_binding_checked = None
            self._surface_binding_error_key = None
            self._surface_binding_error = ""
            self._surface_dimensions_key = None
            self._surface_dimensions_text = ""
            self._geometry_needed = False
            self._current_work_estimate = AssemblyWorkEstimate(available=False)
            self.membership_notice.clear()
            self.membership_notice.hide()
            self._clear_validation_qa("Choose a body response to begin.")
            self.point_format_button.setChecked(False)
            self.line_format_button.setChecked(False)
            self._toggle_schema_help("point", False)
            self._toggle_schema_help("line", False)
            for section in sections:
                section.header.setChecked(False)
                section._sync(False)
            for controls in self.vehicle_feature_controls.values():
                controls["advanced"].header.setChecked(False)
                controls["advanced"]._sync(False)
                controls["signature"] = None
            self.workflow_tabs.setCurrentIndex(0)
            self.workflow_tabs.setTabVisible(4, False)
            self.operation_progress.setValue(0)
            self.operation_progress.hide()
            self.cancel_operation_button.hide()
            self.cancel_operation_button.setEnabled(False)
            self._refresh_spatial_feature_tree()
        finally:
            self._loading_recipe = False
            blockers.clear()
        self._update_recipe_status()
        self._update_workflow_readiness()
        self.assembly_cleared.emit()
        self.status_changed.emit("Assembly cleared. Choose a body response to begin.")
        return True

    def _ensure_vehicle_editable(self):
        if self.job_is_running():
            raise ValueError("Wait for the current Assembly operation before editing features.")
        if getattr(self, "_placement_editors", {}):
            raise ValueError("Apply or close the open placement editor before editing feature families.")

    def _add_vehicle_feature(self, kind):
        from PySide6.QtWidgets import QDialog
        from .feature_composer import AddFeatureDialog
        try:
            self._ensure_vehicle_editable()
            mapping = self.point_mapping if kind == "point" else self.line_mapping
            dialog = AddFeatureDialog(kind, units=str(self.coordinate_units.currentData() or ""),
                                      dataset_choices=mapping.mapping(), parent=self)
            if dialog.exec() == QDialog.Accepted:
                self.add_vehicle_feature(dialog.feature_definition())
        except Exception as exc:
            self._show_error(str(exc))

    def _adopt_vehicle_edit(self, edit, kind):
        # Discovery already completed on a separate model. Avoid the regular
        # CSV-setter reset, which intentionally clears exclusions on imports.
        self.model.__dict__.update(edit.model.__dict__)
        picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
        picker.set_path(getattr(self.model.values, f"{kind}_locations_csv"))
        self.coordinate_units.blockSignals(True)
        try:
            self.coordinate_units.setCurrentIndex(self.coordinate_units.findData(self.model.values.coordinate_units))
        finally:
            self.coordinate_units.blockSignals(False)
        self._apply_requirements_to_tables()
        self._mark_preview_stale()
        self._schedule_geometry_preview()

    def add_vehicle_feature(self, definition):
        """Add a family and its response in one operation; return its stable ID."""
        self._ensure_vehicle_editable()
        self._pull_values()
        edit = prepare_vehicle_feature(self.model.values, definition,
            directory=self._draft_path.parent, service=self._service)
        self._adopt_vehicle_edit(edit, definition["kind"])
        self.workflow_tabs.setCurrentIndex(1 if definition["kind"] == "point" else 2)
        controls = getattr(self, "vehicle_feature_controls", {}).get(definition["kind"], {})
        table = controls.get("table")
        if table is not None:
            from PySide6.QtCore import Qt
            for row in range(table.rowCount()):
                item = table.item(row, 1)
                if item is not None and (item.text() == edit.dataset_id or item.data(Qt.UserRole) == edit.dataset_id):
                    table.selectRow(row)
                    break
        self.status_changed.emit(f"Added {edit.dataset_id}: {edit.instance_count} placement(s), with its response assigned.")
        return edit.dataset_id

    def remove_vehicle_feature(self, kind, dataset_id):
        self._ensure_vehicle_editable()
        self._pull_values()
        edit = prepare_vehicle_feature_removal(self.model.values, kind, dataset_id,
            directory=self._draft_path.parent, service=self._service)
        self._adopt_vehicle_edit(edit, kind)
        self.status_changed.emit(f"Removed {dataset_id} and its {edit.instance_count} placement(s) from this vehicle.")

    def recipe_values_with_vehicle_placements(self, recipe_target):
        return portable_vehicle_values(self.model.values, recipe_target,
                                       draft_directory=self._draft_path.parent)
