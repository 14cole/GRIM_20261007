"""Small workflow actions layered onto the existing validated assembly form."""
from __future__ import annotations
import os
from pathlib import Path


def suggest_response_mapping(dataset_ids, folder, existing=None, *, cancel_check=lambda: False, progress_callback=None):
    """Exact filename-stem suggestions only; physical compatibility is not inferred."""
    existing = existing or {}
    by_stem = {}
    wanted = {value.casefold() for value in dataset_ids if not existing.get(value)}
    scanned = 0
    for root, _dirs, files in os.walk(folder, followlinks=False):
        if cancel_check():
            raise InterruptedError("Library search cancelled.")
        for filename in files:
            if cancel_check():
                raise InterruptedError("Library search cancelled.")
            path = Path(root) / filename
            scanned += 1
            if progress_callback is not None and scanned % 128 == 0:
                progress_callback(scanned)
            if path.suffix.casefold() == '.grim' and path.stem.casefold() in wanted and path.is_file():
                by_stem.setdefault(path.stem.casefold(), []).append(str(path.resolve()))
    unique, ambiguous, missing = {}, {}, []
    for dataset_id in dataset_ids:
        if str(existing.get(dataset_id, '')).strip():
            continue
        matches = sorted(by_stem.get(dataset_id.casefold(), []))
        if len(matches) == 1:
            unique[dataset_id] = matches[0]
        elif matches:
            ambiguous[dataset_id] = matches
        else:
            missing.append(dataset_id)
    return unique, ambiguous, missing


class MappingWorkflowMixin:
    def _suggest_library_folder(self):
        from PySide6.QtWidgets import QFileDialog, QProgressDialog
        from .library_scan import LibraryScan
        if getattr(self, '_library_scan', None) is not None:
            self._library_scan.cancel.set()
            return
        folder = QFileDialog.getExistingDirectory(self, 'Suggest response files from a library folder')
        if not folder: return
        worker = LibraryScan(self._required_ids, folder, self.mapping())
        self._library_scan = worker
        self._library_progress = QProgressDialog('Searching response files…', 'Cancel', 0, 0, self)
        self._library_progress.setWindowTitle('Library search')
        self._library_progress.canceled.connect(worker.cancel.set)
        self.destroyed.connect(worker.cancel.set)
        worker.signals.done.connect(self._library_scan_done)
        worker.signals.progress.connect(self._library_scan_progress)
        self._library_progress.show()
        worker.start()

    def _library_scan_progress(self, count):
        if getattr(self, '_library_progress', None) is not None:
            self._library_progress.setLabelText(f'Searched {count:,} files…')

    def _library_scan_done(self, payload):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QComboBox, QDialogButtonBox
        worker, result, error = payload
        if worker is not getattr(self, '_library_scan', None):
            return
        self._library_scan = None
        self._library_progress.close()
        self._library_progress.deleteLater()
        self._library_progress = None
        if worker.cancel.is_set():
            return
        if error:
            self.catalog_notice.emit(error)
            return
        if tuple(self._required_ids) != worker.identifiers or self.mapping() != worker.mapping:
            self.catalog_notice.emit('Response mappings changed during search; search the current selection again.')
            return
        unique, ambiguous, missing = result
        dialog = QDialog(self)
        dialog.setWindowTitle('Review response mappings')
        dialog.resize(750,400)
        layout = QVBoxLayout(dialog)
        label = QLabel('Filename matches are suggestions. Verify each response; Validate placements still checks compatibility. Existing mappings are preserved.')
        label.setWordWrap(True)
        layout.addWidget(label)
        keys = list(unique) + list(ambiguous)
        table = QTableWidget(len(keys),2)
        table.setHorizontalHeaderLabels(['Dataset ID', 'Response file'])
        choices = {}
        for i,key in enumerate(keys):
            table.setItem(i,0,QTableWidgetItem(key))
            combo = QComboBox()
            combo.addItem('Leave unmapped', '')
            for path in [unique[key]] if key in unique else ambiguous[key]:
                combo.addItem(path,path)
            if key in unique: combo.setCurrentIndex(1)
            choices[key] = combo
            table.setCellWidget(i,1,combo)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        notice = QLabel(f'{len(unique)} unique match(es); {len(ambiguous)} ambiguous ID(s). Choose ambiguous files explicitly.\nNo filename match: ' + (', '.join(missing) or 'none'))
        notice.setWordWrap(True)
        layout.addWidget(notice)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.Accepted:
            for key,combo in choices.items():
                if combo.currentData(): self.set_path(key,combo.currentData())


class AssemblyWorkflowMixin:
    def create_variant_path(self, path, name):
        from GRIM_Backend.assembly.panel import write_feature_assembly_recipe, _recipe_target_path
        if self.job_is_running():
            raise ValueError('Wait for the current assembly operation before creating a variant.')
        name = str(name).strip()
        if not name: raise ValueError('Enter a variant name.')
        target = _recipe_target_path(path)
        if target.exists():
            raise ValueError('Choose a new recipe filename to preserve existing variants.')
        self._pull_values()
        values = self.recipe_values_with_vehicle_placements(target)
        stem = target.name.removesuffix('.assembly.json').removesuffix('.json')
        output = target.with_name(stem + '_total.grim')
        suffix = 2
        while output.exists() or output.with_name(output.stem + '_features_only.grim').exists():
            output = target.with_name(f'{stem}_{suffix}_total.grim')
            suffix += 1
        if values.output_grim and output.resolve() == Path(values.output_grim).resolve():
            output = target.with_name(f'{stem}_variant_total.grim')
            if output.exists(): raise ValueError('Choose another variant filename; its output already exists.')
        values.output_grim = str(output)
        saved = write_feature_assembly_recipe(values, target, name=self.recipe_name_edit.text(), variant=name)
        self.recipe_variant_edit.setText(name)
        self.output_picker.set_path(str(output))
        self._recipe_path = saved
        self._output_path_changed()
        self._recipe_dirty = False
        self._recipe_source_warnings = ()
        self._update_recipe_status()
        self.status_changed.emit(f'Created variant {name}: {saved.name}. Output: {output.name}. Adjust enabled features and validate before building.')
        return saved

    def _create_variant(self):
        from PySide6.QtWidgets import QInputDialog, QFileDialog
        if self.job_is_running(): return
        name,ok = QInputDialog.getText(self,'Create Assembly variant','Variant name (copies current settings and enabled features)')
        if not ok or not name.strip(): return
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name.strip())
        default = Path(self._recipe_default_path()).with_name(safe+'.assembly.json')
        path,_ = QFileDialog.getSaveFileName(self,'Save new variant',str(default),'Assembly recipe (*.assembly.json)')
        if path:
            try: self.create_variant_path(path,name)
            except Exception as exc: self._show_error(str(exc))

    def _go_to_requirement(self, item, *_):
        from GRIM_Backend.assembly.panel import _resolved_user_path
        label = item.text(0)
        point_missing = bool(self.point_mapping.missing_ids())
        for key, path in self.point_mapping.mapping().items():
            if key in self.model.active_point_dataset_ids() and path:
                try:
                    point_missing |= not _resolved_user_path(path,base_dir=self.model.values.base_dir).is_file()
                except OSError:
                    point_missing = True
        feature_tab = 1 if point_missing or (self.point_csv_picker.path() and not self.line_csv_picker.path()) else 2
        feature_mapping = self.point_mapping if feature_tab == 1 else self.line_mapping
        routes = {
            'GHOST feature backend': (0,self.base_picker),
            'Body-only baseline when no features enabled': (0,self.base_picker),
            'Clean-body response': (0,self.base_picker), 'Surface mesh': (0,self.surface_picker),
            'Surface mesh units': (0,self.surface_units), 'Reviewed solve ↔ mesh binding': (0,self.check_surface_binding_button),
            'Placement coordinate units': (feature_tab,self.coordinate_units),
            'Placement CSV selected': (feature_tab,self.point_csv_picker if feature_tab==1 else self.line_csv_picker),
            'Placement CSV read': (3,self.scan_button),
            'Every dataset_id mapped': (feature_tab,feature_mapping.table),
            'Mapped response files available': (feature_tab,feature_mapping.table),
            'Output response selected': (3,self.output_picker),
            'Validation warnings reviewed': (3,self.validation_warning_ack),
            'Advanced settings valid': (3,self.shadow_bias),
            'Placements validated': (3,self.preview_button),
            'Body mesh certificate': (3,self.preview_button),
        }
        tab, widget = routes.get(label, (3,self.readiness_checklist))
        self.workflow_tabs.setCurrentIndex(tab)
        # QAbstractItemView.edit is a method, unlike the path picker's widget.
        from PySide6.QtWidgets import QWidget
        editor = getattr(widget, 'edit', None)
        if isinstance(editor, QWidget):
            widget = editor
        # Scroll the containing page to the control when the step is compact.
        from PySide6.QtWidgets import QScrollArea, QAbstractButton
        parent = widget.parentWidget()
        while parent:
            header=getattr(parent,'header',None)
            if isinstance(header,QAbstractButton) and hasattr(parent,'body'):
                header.setChecked(True)
            if isinstance(parent,QScrollArea):
                parent.ensureWidgetVisible(widget)
                break
            parent = parent.parentWidget()
        widget.setFocus()

    def _fix_next_requirement(self):
        from PySide6.QtCore import Qt
        for index in range(self.readiness_checklist.topLevelItemCount()):
            group = self.readiness_checklist.topLevelItem(index)
            for n in range(group.childCount()):
                item = group.child(n)
                if item.data(1,Qt.UserRole) and not item.data(0,Qt.UserRole):
                    self._go_to_requirement(item)
                    return
