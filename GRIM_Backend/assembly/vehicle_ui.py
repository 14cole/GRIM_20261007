"""Task-oriented controls for the vehicle assembly form.

The existing form/model remains the single authority for response mappings,
membership and validation. These controls present the same state as named
features, with CSV import as the primary point and line placement workflow.
"""
from __future__ import annotations

from pathlib import Path


class VehicleAssemblyUiMixin:
    def _build_vehicle_feature_page(self, kind, page, layout):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QAbstractItemView, QHeaderView, QHBoxLayout, QLabel, QPushButton,
            QTableWidget, QVBoxLayout, QWidget,
        )
        from .panel import _DisclosureSection

        is_point = kind == "point"
        title = QLabel("Point features" if is_point else "Line features", page)
        title.setObjectName("featurePanelIntro")
        layout.addWidget(title)
        description = QLabel(
            "Import a CSV of fastener, inlet, antenna, or other point placements. "
            "Positions and orientations come from the file; choose a response for each feature type."
            if is_point else
            "Import a CSV of panel gaps or other line paths. Segment endpoints "
            "and normals come from the file; choose a line response for each feature type.", page,
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        # Keep manual authoring available without putting coordinate-entry
        # controls in the CSV workflow. Hidden buttons retain controller hooks.
        add = QPushButton("Add manually…", page)
        add.clicked.connect(lambda: self._add_vehicle_feature(kind))
        add.hide()
        add_action = self.manual_placement_menu.addAction(
            "Add point feature manually…" if is_point else "Add line path manually…", add.click,
        )
        import_button = QPushButton("Import point CSV…" if is_point else "Import line CSV…", page)
        import_button.setObjectName("featureWorkflowAction")
        import_button.setProperty("primaryAction", True)
        import_button.setToolTip(
            f"Load {kind} positions and orientations from CSV. "
            f"Choosing another CSV replaces the current {kind} placement list."
        )
        import_button.clicked.connect(lambda: self._import_vehicle_placements(kind))
        layout.addWidget(import_button)
        empty = QLabel(
            "No point features yet. Import your CSV above; no coordinate typing is needed."
            if is_point else
            "No line features yet. Import your CSV above; no vertex entry is needed.", page,
        )
        empty.setWordWrap(True)
        empty.setObjectName("featureHint")
        layout.addWidget(empty)

        table = QTableWidget(0, 4, page)
        table.setObjectName(f"vehicle{kind.title()}Features")
        table.setHorizontalHeaderLabels(["Use", "Feature", "Placed", "Response"])
        table.verticalHeader().hide()
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(80)
        table.setMaximumHeight(180)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.setToolTip("Use includes this feature in the calculation. Uncheck to compare a variant.")
        table.itemChanged.connect(lambda item: self._vehicle_membership_changed(kind, item))
        table.itemSelectionChanged.connect(lambda: self._update_vehicle_selection_actions(kind))
        table.cellDoubleClicked.connect(lambda *_: self._change_vehicle_response(kind))
        layout.addWidget(table)
        actions = QWidget(page)
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        edit = QPushButton("Edit placements…", actions)
        edit.clicked.connect(lambda: self._edit_vehicle_placements(kind))
        edit.hide()
        edit_action = self.manual_placement_menu.addAction(
            "Edit selected point placements…" if is_point else "Edit selected line placements…", edit.click,
        )
        response = QPushButton("Change response…", actions)
        response.clicked.connect(lambda: self._change_vehicle_response(kind))
        remove = QPushButton("Remove", actions)
        remove.clicked.connect(lambda: self._remove_vehicle_feature_clicked(kind))
        for button in (response, remove):
            row.addWidget(button)
        layout.addWidget(actions)
        hint = QLabel(
            "Use controls the calculation. Visibility in the 3-D view is separate. "
            "Double-click a feature to assign its response. Coordinates come from your CSV.", page,
        )
        hint.setWordWrap(True)
        hint.setObjectName("featureHint")
        layout.addWidget(hint)
        advanced = _DisclosureSection(
            "CSV details / response mappings",
            page, expanded=False,
        )
        legacy = QWidget(advanced)
        legacy_layout = QVBoxLayout(legacy)
        legacy_layout.setContentsMargins(0, 0, 0, 0)
        advanced.addWidget(legacy)
        layout.addWidget(advanced)
        layout.addStretch(1)
        if not hasattr(self, "vehicle_feature_controls"):
            self.vehicle_feature_controls = {}
        self.vehicle_feature_controls[kind] = {
            "table": table, "empty": empty, "add": add, "actions": actions,
            "edit": edit, "response": response, "remove": remove,
            "advanced": advanced, "hint": hint, "signature": None,
            "description": description,
            "import": import_button,
            "add_action": add_action, "edit_action": edit_action,
        }
        self._update_vehicle_selection_actions(kind)
        return legacy, legacy_layout

    def _vehicle_selected_dataset(self, kind):
        from PySide6.QtCore import Qt
        table = self.vehicle_feature_controls[kind]["table"]
        item = table.item(table.currentRow(), 1)
        return item.data(Qt.UserRole) if item is not None else None

    def _update_vehicle_selection_actions(self, kind):
        controls = self.vehicle_feature_controls[kind]
        selected = bool(self._vehicle_selected_dataset(kind))
        unlocked = not self.job_is_running() and not self._placement_editors
        editable = selected and unlocked
        for key in ("edit", "response", "remove"):
            controls[key].setEnabled(editable)
        controls["add"].setEnabled(unlocked)
        controls["add_action"].setEnabled(unlocked)
        controls["edit_action"].setEnabled(editable)
        if controls["import"] is not None:
            controls["import"].setEnabled(unlocked)
        controls["table"].setEnabled(unlocked)

    def _import_vehicle_placements(self, kind):
        """Use the existing CSV parser and mapping workflow from the main action."""
        from PySide6.QtWidgets import QFileDialog
        if self.job_is_running() or self._placement_editors:
            return
        picker = self.point_csv_picker if kind == "point" else self.line_csv_picker
        path, _ = QFileDialog.getOpenFileName(self, picker.caption, picker.path(), picker.file_filter)
        if not path:
            return
        self.workflow_tabs.setCurrentIndex(1 if kind == "point" else 2)
        self.vehicle_feature_controls[kind]["advanced"].header.setChecked(True)
        picker.set_path(path)
        picker.editing_finished.emit()

    def _refresh_vehicle_ui(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTableWidgetItem
        if not hasattr(self, "vehicle_feature_controls"):
            return
        values = self.model.values
        counts = []
        for kind, controls in self.vehicle_feature_controls.items():
            instances = self.model.point_instances if kind == "point" else self.model.line_instances
            excluded = values.excluded_point_placement_ids if kind == "point" else values.excluded_line_ids
            ids = self.model.point_dataset_ids if kind == "point" else self.model.line_dataset_ids
            mapping = self.point_mapping.mapping() if kind == "point" else self.line_mapping.mapping()
            members_by_id = {}
            for instance in instances:
                members_by_id.setdefault(instance[1], []).append(instance)
            signature = (instances, tuple(ids), tuple(sorted(excluded)), tuple(sorted(mapping.items())))
            counts.append(sum(instance[0] not in excluded for instance in instances))
            if controls["signature"] != signature:
                controls["signature"] = signature
                table = controls["table"]
                selected = self._vehicle_selected_dataset(kind)
                table.blockSignals(True)
                try:
                    table.setRowCount(len(ids))
                    for row, dataset_id in enumerate(ids):
                        members = members_by_id.get(dataset_id, ())
                        enabled = sum(item[0] not in excluded for item in members)
                        use = QTableWidgetItem()
                        use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                        use.setData(Qt.UserRole, dataset_id)
                        use.setCheckState(Qt.Unchecked if members and not enabled else
                                          Qt.PartiallyChecked if enabled < len(members) else Qt.Checked)
                        table.setItem(row, 0, use)
                        name = QTableWidgetItem(dataset_id.replace("_", " "))
                        name.setData(Qt.UserRole, dataset_id)
                        name.setToolTip(f"Response ID: {dataset_id}")
                        table.setItem(row, 1, name)
                        count = QTableWidgetItem(f"{enabled}/{len(members)}" if enabled != len(members) else str(len(members)))
                        count.setToolTip("Included / total placements" if kind == "point" else "Included / total paths")
                        table.setItem(row, 2, count)
                        path = mapping.get(dataset_id, "")
                        response = QTableWidgetItem(Path(path).name if path else "Choose response…")
                        response.setToolTip(path or "Select this feature, then Change response.")
                        table.setItem(row, 3, response)
                        if dataset_id == selected:
                            table.selectRow(row)
                    if table.currentRow() < 0 and ids:
                        table.selectRow(0)
                finally:
                    table.blockSignals(False)
                # Feature actions should stay next to their rows. Large imported
                # libraries scroll within the table instead of pushing actions
                # below the page fold.
                content_height = (table.horizontalHeader().height()
                                  + sum(table.rowHeight(row) for row in range(min(5, len(ids))))
                                  + 2 * table.frameWidth())
                table.setFixedHeight(max(80, min(180, content_height)))
                for key in ("table", "actions", "hint"):
                    controls[key].setVisible(bool(ids))
                controls["empty"].setVisible(not ids)
                controls["description"].setVisible(not ids)
            self._update_vehicle_selection_actions(kind)
        if hasattr(self, "vehicle_summary_label"):
            body = Path(values.base_grim).name if values.base_grim else "Choose body response"
            point_label = "point" if counts[0] == 1 else "points"
            path_label = "path" if counts[1] == 1 else "paths"
            self.vehicle_summary_label.setText(f"{body}  ·  {counts[0]} {point_label}  ·  {counts[1]} {path_label}")
            self.vehicle_summary_label.setToolTip(values.base_grim)

    def _vehicle_membership_changed(self, kind, item):
        from PySide6.QtCore import Qt
        if item.column() != 0:
            return
        if self.job_is_running() or self._placement_editors:
            # Also protect programmatic/queued changes arriving as a job starts.
            # Reset the view so an ignored click cannot misrepresent membership.
            self.vehicle_feature_controls[kind]["signature"] = None
            self._refresh_vehicle_ui()
            return
        dataset_id = item.data(Qt.UserRole)
        included = item.checkState() != Qt.Unchecked
        instances = self.model.point_instances if kind == "point" else self.model.line_instances
        for instance in instances:
            if instance[1] == dataset_id:
                self.model.set_feature_instance_enabled(kind, instance[0], included)
        self._refresh_spatial_feature_tree()
        self._mark_preview_stale()
        self._schedule_geometry_preview()

    def _edit_vehicle_placements(self, kind):
        if self.job_is_running() or self._placement_editors:
            return
        dataset_id = self._vehicle_selected_dataset(kind)
        if not dataset_id:
            return
        self._edit_placements(kind)
        editor = self._placement_editors.get(kind)
        if editor is not None:
            for row, values in enumerate(editor.rows()):
                if values[1] == dataset_id:
                    editor.table.selectRow(row)
                    editor.table.scrollToItem(editor.table.item(row, 0))
                    break

    def _change_vehicle_response(self, kind):
        if self.job_is_running() or self._placement_editors:
            return
        dataset_id = self._vehicle_selected_dataset(kind)
        if dataset_id:
            mapping = self.point_mapping if kind == "point" else self.line_mapping
            mapping._browse(dataset_id)

    def _remove_vehicle_feature_clicked(self, kind):
        dataset_id = self._vehicle_selected_dataset(kind)
        if dataset_id:
            try:
                self.remove_vehicle_feature(kind, dataset_id)
            except Exception as exc:
                self._show_error(str(exc))

    def _show_wing_tools(self):
        self.workflow_tabs.setTabVisible(4, True)
        self.workflow_tabs.setCurrentIndex(4)

    def _vehicle_tab_changed(self, index):
        if index != 4:
            self.workflow_tabs.setTabVisible(4, False)
