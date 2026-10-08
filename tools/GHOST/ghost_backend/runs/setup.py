"""Run requests, setup checks and output notes for the GHOST solver tab."""
import math
from pathlib import Path

DEFAULT_QUALITY = {'residual_norm_max': 1e-6, 'condition_est_max': 1e6, 'warnings_max': 10}
REQUEST_SCHEMA = 'grim.2d-run-request'


def two_d_request(frequencies_ghz, angles_deg, units, mesh_certification, accuracy,
                  scattering, observation_angles_deg):
    """The physical choices of a 2D run; execution settings are always automatic."""
    from ghost_backend.execution.options import automatic_run
    request = dict(schema=REQUEST_SCHEMA, frequencies_ghz=[float(f) for f in frequencies_ghz],
                   angles_deg=[float(a) for a in angles_deg], units=units,
                   mesh_certification=bool(mesh_certification), accuracy=accuracy,
                   scattering=scattering,
                   observation_angles_deg=[float(a) for a in (observation_angles_deg or [])])
    if not request['frequencies_ghz'] or any(not math.isfinite(f) or f <= 0 for f in request['frequencies_ghz']):
        raise ValueError('Frequencies must be positive GHz values.')
    if not request['angles_deg'] or any(not math.isfinite(a) for a in request['angles_deg']):
        raise ValueError('Supply at least one finite incident angle.')
    if units not in ('inches', 'meters') or accuracy not in ('standard', 'tight'):
        raise ValueError('Unsupported units or accuracy target.')
    if scattering == 'bistatic' and not request['observation_angles_deg']:
        raise ValueError('Bistatic mode requires at least one observation angle.')
    request.update(automatic_run(scattering))
    return request


def geometry_dimensions(snapshot, units):
    points = [(float(pair[x]), float(pair[y])) for seg in snapshot.get('segments', [])
              for pair in seg.get('point_pairs', []) for x,y in [('x1','y1'), ('x2','y2')]]
    if not points or any(not math.isfinite(v) for p in points for v in p):
        raise ValueError('Load geometry with finite coordinates to inspect its dimensions.')
    scale = .0254 if units == 'inches' else 1.
    width = (max(x for x,y in points) - min(x for x,y in points)) * scale
    height = (max(y for x,y in points) - min(y for x,y in points)) * scale
    return f'X span {width/.0254:g} in \u00d7 Y span {height/.0254:g} in'


def validate_material_coverage(snapshot, library, frequencies, checkpoint=None):
    """Validate every requested frequency without rebuilding invariant flag sets."""
    from ghost_backend.twod.formulations.thin_layer import ThinLayerDefinition, validate_thin_layer
    used_ibcs = {int(seg['properties'][2]) for seg in snapshot['segments']
                 if int(seg['properties'][2]) > 0}
    used_media = {int(seg['properties'][i]) for seg in snapshot['segments']
                  for i in (3, 4) if int(seg['properties'][i]) > 0}
    for freq in frequencies:
        if checkpoint is not None:
            checkpoint()
        for flag in used_ibcs:
            model = library.impedance_models[flag]
            if isinstance(model, ThinLayerDefinition):
                eps, mu = library.get_medium(model.dielectric_flag, freq)
                validate_thin_layer(eps, mu, model.thickness_m, 2*math.pi*freq*1e9/299792458.)
            else:
                library.get_impedance(flag, freq, arc_s=0.)
                library.get_impedance(flag, freq, arc_s=1.)
        for flag in used_media:
            library.get_medium(flag, freq)


class RunSetupMixin:
    def _build_run_setup_controls(self, form):
        try:
            from PySide6.QtWidgets import QLabel, QPushButton
        except ImportError:
            from PySide2.QtWidgets import QLabel, QPushButton
        self.run_setup_notice = QLabel()
        self.run_setup_notice.setWordWrap(True)
        form.addRow(self.run_setup_notice)
        self.run_output_notice = QLabel('Output: automatic unique GRIM file after solving.')
        self.run_output_notice.setWordWrap(True)
        self.run_preflight_button = QPushButton('Check geometry and run setup')
        self.run_preflight_button.clicked.connect(self._check_run_setup)
        form.addRow(self.run_preflight_button)
        self.cmb_units.currentTextChanged.connect(self._update_run_dimensions)
        self.edit_geo_path.textChanged.connect(self._update_run_dimensions)

    def _setup_busy(self):
        return self._job_is_active()

    def _capture_run_setup(self):
        solver = self.cmb_solver_kind
        is_bor = solver.currentData() == 'bor'
        freq, angles = self._collect_frequency_values(), self._collect_elevation_values()
        if is_bor:
            from ghost_backend.runs.bor_setup import validate_bor_setup
            from ghost_backend.assembly.fields import radar_grid_aspects
            grid = self._collect_bor_radar_grid()
            aspects = radar_grid_aspects(grid['azimuths_deg'], grid['elevations_deg'],
                                         grid['axis_az_deg'], grid['axis_el_deg'])
            return validate_bor_setup(dict(schema='grim.bor-run-setup', version=1,
                frequencies_ghz=freq, aspects_deg=aspects.tolist(), radar_grid=grid,
                units=self.cmb_units.currentText(),
                mesh_certification=self.chk_mesh_certification.isChecked(),
                accuracy=self.cmb_accuracy_target.currentData(),
                cfie_alpha=float(self.edit_cfie_alpha.text()), bor_options=self.bor_options_widget.value()))
        scattering = self.cmb_scatter_mode.currentData()
        return two_d_request(freq, angles, self.cmb_units.currentText(),
            self.chk_mesh_certification.isChecked(), self.cmb_accuracy_target.currentData(), scattering,
            [] if scattering == 'monostatic' else self._parse_list(self.edit_obs_angles.text(), 'Observation angles'))

    def _update_run_dimensions(self, *_):
        try:
            snapshot,_,_ = self._load_geometry_for_solver()
            self.lbl_run_dimensions.setText(geometry_dimensions(snapshot,self.cmb_units.currentText()))
        except Exception as exc:
            self.lbl_run_dimensions.setText(str(exc))

    def _run_setup_summary(self, snapshot, base_dir, value, checkpoint=None,
                           forecast_frequencies=None):
        if value['schema'] == 'grim.bor-run-setup':
            from ghost_backend.runs.bor_setup import resource_summary
            return resource_summary(snapshot, base_dir, value, checkpoint, forecast_frequencies)
        from ghost_backend.twod.preparation import prepare_geometry
        _, result, library, _ = prepare_geometry(snapshot, base_dir, value['units'])
        validate_material_coverage(snapshot, library, value['frequencies_ghz'], checkpoint)
        forecast_frequencies = (value['frequencies_ghz'] if forecast_frequencies is None
                                else list(forecast_frequencies))
        selection_note = ''
        if not forecast_frequencies:
            selection_note = 'Matching checkpoints found for every frequency; no new solve forecast needed.\n'
        elif value['execution_options']['factorization'] == 'adaptive':
            from ghost_backend.execution.selection import select_backend
            from ghost_backend.runs.quality import accuracy_target_policy
            arguments = dict(geometry_snapshot=snapshot, material_base_dir=base_dir,
                geometry_units=value['units'], frequencies_ghz=value['frequencies_ghz'],
                elevations_deg=value['angles_deg'], solver_method=value['solver_method'], max_panels=100000,
                mesh_convergence_policy=accuracy_target_policy(value['accuracy']))
            # Execution finishes both channels per frequency. Forecast those
            # same requests so its run-scoped cache can serve the actual solve.
            selections = [select_backend(dict(arguments, frequencies_ghz=[frequency]),
                value['execution_options'], value['mesh_certification'], checkpoint)
                for frequency in forecast_frequencies]
            selection = dict(selected='/'.join(sorted({s['selected'] for s in selections})),
                dense_peak_gib=max(s['dense_peak_gib'] for s in selections),
                admission_budget_gib=min(s['admission_budget_gib'] for s in selections))
            selection_note = 'Planned backend: {}. Dense peak forecast {:.2f} GiB; admission budget {:.2f} GiB.\n'.format(
                selection['selected'], selection['dense_peak_gib'], selection['admission_budget_gib'])
        warnings = list(result['warnings']) + list(library.warnings)
        count = len(value['frequencies_ghz'])*len(value['angles_deg'])
        if value['scattering']=='bistatic': count *= len(value['observation_angles_deg'])
        return (f"{result['segment_count']} segments; {result['primitive_count']} primitives. "
                + geometry_dimensions(snapshot,value['units']) + '\n'
                + f"{len(value['frequencies_ghz'])} frequencies \u00d7 {len(value['angles_deg'])} incident angles; {count} samples per channel, VV + HH. "
                + ('Base/fine mesh comparison' if value['mesh_certification'] else 'Uncertified single mesh')
                + f"; {value['accuracy']} target.\n"
                + selection_note
                + ('Warnings: ' + '; '.join(warnings) if warnings else 'Geometry and material checks passed.')
                + '\nSolver quality and convergence are evaluated during the run.')

    def _check_run_setup(self):
        if self._setup_busy(): return
        try:
            value = self._capture_run_setup()
            snapshot,_,base_dir = self._load_geometry_for_solver()
            self._start_setup_check(snapshot,base_dir,value)
            self._update_run_dimensions()
            self._update_run_output_note()
        except Exception as exc:
            self.run_setup_notice.setText(f'Correct before running: {exc}')

    def _update_run_output_note(self):
        if not self.chk_export_after_solve.isChecked():
            self.run_output_notice.setText('Output: retained in GHOST; use Export Last Result when ready.')
            return
        raw=self.edit_output.text().strip()
        if raw:
            path=Path(raw).expanduser()
            if not path.is_absolute():
                _,source,base=self._load_geometry_for_solver()
                path=(Path(base) if source else self._documents_output_dir())/path
            if path.suffix.lower()!='.grim': path=Path(str(path)+'.grim')
            details=str(path.resolve())
            note=f'Output: {path.name} in {path.parent.name}.'
            if path.exists(): note+=' Existing output: replacement will require review at export.'
        else:
            _,source,base=self._load_geometry_for_solver()
            folder=Path(base) if source else self._documents_output_dir()
            details=str(folder)
            note=f'Output: a unique timestamped GRIM file in {folder.name}. Hover here for the full folder path.'
        if self.cmb_scatter_mode.currentData()=='bistatic':
            note+=' Bistatic runs write a separate file for each incident angle.'
        self.run_output_notice.setText(note)
        self.run_output_notice.setToolTip(details)

    def _start_setup_check(self, snapshot, base_dir, value):
        from ghost_backend.ui.solver import _SolveWorker, QThread
        import threading
        self._abort_event=threading.Event()
        self._solve_run_serial += 1
        run_id=self._solve_run_serial
        self._active_solve_run_id=run_id
        self._pending_solve_context=None
        self._set_solving_state(True)
        self.run_setup_notice.setText('Checking the current geometry and setup\u2026')
        thread=QThread(self)
        is_bor = value['schema'] == 'grim.bor-run-setup'
        worker=_SolveWorker(snapshot,'',base_dir,value['frequencies_ghz'],value['aspects_deg'] if is_bor else value['angles_deg'],value['units'],DEFAULT_QUALITY,
                            abort_event=self._abort_event,preflight_setup=value,preflight_only=True,
                            execution_options=value.get('execution_options'), solver_method=value.get('solver_method', 'direct'),
                            lu_precision=value.get('lu_precision', 'double'), solver_kind='bor' if is_bor else '2d',
                            bor_options=value.get('bor_options'))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_solver_progress)
        worker.setup_checked.connect(self.run_setup_notice.setText)
        worker.telemetry.connect(self._on_execution_progress)
        worker.finished.connect(self._setup_check_finished)
        worker.error.connect(self._setup_check_failed)
        worker.canceled.connect(self._on_solver_canceled)
        for signal in (worker.finished,worker.error,worker.canceled):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._on_solver_thread_finished(run_id))
        self._solve_thread,self._solve_worker=thread,worker
        thread.start()

    def _setup_check_finished(self, *_):
        self._set_solving_state(False)
        self.lbl_status.setText('Setup check complete. Review the summary, then run. Geometry and materials are checked again when solving.')

    def _setup_check_failed(self, message):
        self._set_solving_state(False)
        self.run_setup_notice.setText('Correct before running: '+message)
