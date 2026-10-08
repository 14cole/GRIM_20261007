"""Export solver fields in the same native format from the GUI or a worker."""


def export_solver_result(result, output_path, *, source_path='', history='', context=None):
    is_bor = (str(result.get('solver', '')).lower() == 'bor_mom_rcs'
              or str(result.get('rcs_linear_quantity', '')).lower() == 'sigma_3d')
    if not is_bor:
        from ghost_backend.io.grim import export_result_to_grim
        return export_result_to_grim(result, output_path, source_path=source_path,
                                     history=history, preserve_raw_complex_amplitude=True)
    if not isinstance(context, dict) or context.get('solver_kind') != 'bor':
        raise ValueError('The BoR solve context is unavailable; re-run the body before exporting it.')
    from ghost_backend.assembly.fields import (
        bodies_from_bor_solver_result, bor_solver_diagnostics_by_frequency,
        bor_output_profile, bor_output_profile_metadata, save_monostatic_grim,
    )
    grid = context.get('radar_grid')
    if grid is None:
        raise ValueError('The BoR radar grid is unavailable; re-run the body before exporting it.')
    path = save_monostatic_grim(
        bodies_from_bor_solver_result(result),
        bor_output_profile(context['snapshot'], str(context['units'])), output_path,
        azimuths_deg=grid['azimuths_deg'], elevations_deg=grid['elevations_deg'],
        axis_az_deg=grid['axis_az_deg'], axis_el_deg=grid['axis_el_deg'], roll_deg=grid['roll_deg'],
        source_path=source_path, history=history,
        solver_diagnostics=bor_solver_diagnostics_by_frequency(result),
        artifact_metadata=bor_output_profile_metadata(context['snapshot']),
    )
    return [path]
