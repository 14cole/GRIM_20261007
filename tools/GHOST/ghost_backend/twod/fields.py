"""Shared bounded RHS solve and field projection for every 2-D formulation."""
import numpy as np
from ghost_backend.execution.cpu import current_state, configured_batch_size
from ghost_backend.linalg.dense import DenseFactor


def solve_fields(mesh, matrix, k0, angles, rhs_builder, diagnostics, label, **kwargs):
    from ghost_backend.execution.options import linear_algebra_threads
    with linear_algebra_threads(matrix.shape[0]):
        return _solve_fields(mesh, matrix, k0, angles, rhs_builder, diagnostics, label, **kwargs)


def _solve_fields(mesh, matrix, k0, angles, rhs_builder, diagnostics, label,
                 potential='SLP', density_builder=None, observation_angles=None,
                 element_mask=None, order=8, return_density=False, project=True,
                 second_potential=None, coordinates=None, second_density_builder=None, adaptive_routes=None,
                 second_element_mask=None):
    import ghost_backend.twod.solver as rcs
    angles = np.asarray(angles, dtype=float).reshape(-1)
    observations = angles if observation_angles is None else np.asarray(observation_angles, float).reshape(-1)
    state = current_state()
    from ghost_backend.twod.assembly.session import current_session
    session = current_session()
    checkpoint = state.checkpoint if state is not None else session.checkpoint if session is not None else None
    requested, threshold = rcs._requested_dense_backend()
    from ghost_backend.linalg.hierarchical import factor_mode
    hierarchical = factor_mode() != 'dense'
    from ghost_backend.compressed.operator import StreamedOperator
    compressed=isinstance(matrix,StreamedOperator)
    if compressed:coordinates=matrix.coordinates
    if requested == 'gpu' and hierarchical:
        raise ValueError('Hierarchical factorization requires the CPU dense backend.')
    if requested == 'gpu' and rcs.requested_precision() == 'mixed':
        raise ValueError('Mixed-precision LU is a CPU method; select the CPU dense backend.')


    gpu = (not hierarchical and state is None and diagnostics is None and rcs.requested_precision() != 'mixed'
           and (requested == 'gpu' or requested == 'auto' and len(matrix) >= threshold))
    batch_size = len(angles) if gpu else configured_batch_size()
    if coordinates is None and len(matrix) in (len(mesh.nodes), 2*len(mesh.nodes)):
        xy = np.zeros((len(mesh.nodes), 2))
        for element in mesh.elements:
            xy[list(element.node_ids)] = [mesh.nodes[i].xy for i in element.node_ids]
        coordinates = np.tile(xy, (len(matrix)//len(mesh.nodes), 1))
    from ghost_backend.compressed.factor import CompressedFactor
    factor_class=CompressedFactor if compressed else DenseFactor
    if checkpoint is not None:
        checkpoint()
    # Restore disk-backed TE coefficients into their owned buffer before the
    # next TM assembly consumes it. No second dense RAM matrix is needed.
    pending = session.pending if session is not None else None
    retained = pending is not None and isinstance(pending[1],tuple) and pending[1][0] is matrix
    ownership = {} if compressed else dict(owned_matrix=True)
    if compressed:
        identity=getattr(matrix,'recycling_identity',None)
        ownership=dict(recycling_key=(identity,label) if identity is not None else None,
                       recycling_frequency=float(k0))
    try:
        factor = None if gpu else factor_class(matrix, diagnostics, label,
            evidence=state.systems if state is not None else None, checkpoint=checkpoint,
            force_double=state is not None, coordinates=coordinates, **ownership)
    except BaseException:
        if retained:
            session.pending = None
        raise
    completed = False
    # Monostatic batches observe along their incidence directions: the RHS keeps
    # its polynomial plane-wave loads for the far-field projection of the batch.
    loads = session.plane_wave_loads if session is not None else None
    if loads is not None:
        loads['monostatic'] = bool(project and observation_angles is None)
    try:
        amplitude = np.zeros(len(angles) if observation_angles is None else (len(angles), len(observations)), complex)
        densities = np.empty((len(mesh.nodes), len(angles)), complex) if return_density else None
        max_residual = 0.0
        from ghost_backend.linalg.sweep import (
            SweepBasis,
            solve as solve_sweep,
            mode as compression_mode,
        )
        sweep_basis = (SweepBasis(batch_size) if factor is not None and len(angles) >= 32
                       and compression_mode() != 'off' else None)
        from ghost_backend.twod.assembly.kernels import GridProjection
        projection_plan = (GridProjection(mesh, k0, observations)
                           if project and observation_angles is not None else None)
        for start in range(0, len(angles), batch_size):
            if checkpoint is not None:
                checkpoint()
            stop = min(len(angles), start + batch_size)
            rhs = rhs_builder(angles[start:stop])
            if factor is None:
                evidence = {}
                solution = rcs._solve_dense_system(matrix, rhs, diagnostics, label, residual_diagnostics=evidence)
                relative = evidence['relative_residual']
            else:
                solution = solve_sweep(factor, rhs, sweep_basis,
                    hint=session.compression_hint if session is not None else None)
                relative = factor.relative_residual
            max_residual = max(max_residual, float(np.max(relative)))
            density = solution[:len(mesh.nodes)] if density_builder is None else density_builder(solution)
            from ghost_backend.twod.adaptivity import observe
            observe(mesh, solution, density, adaptive_routes)
            if densities is not None:
                densities[:, start:stop] = density
            if project:
                obs = angles[start:stop] if observation_angles is None else observations
                projection = 'matched' if observation_angles is None else 'grid'
                field = rcs._farfield_linear_density_many(mesh, density, k0, obs, potential,
                    element_mask=element_mask, projection=projection, order=order,
                    prepared_projection=projection_plan)
                if second_potential is not None:
                    second_density=(solution[len(mesh.nodes):] if second_density_builder is None
                                    else second_density_builder(solution))
                    field += rcs._farfield_linear_density_many(mesh, second_density, k0, obs,
                        second_potential, projection=projection, order=order, element_mask=second_element_mask,
                        prepared_projection=projection_plan)
                amplitude[start:stop] = field
            if loads is not None:
                loads.pop('latest', None)
            if state is not None:
                state.checkpoint(stop, len(angles))


            solution = density = rhs = None
        completed = True
        return rcs._rcs_sigma_from_amp(amplitude, k0), amplitude, max_residual, densities
    finally:
        if loads is not None:
            loads.pop('latest', None)
            loads.pop('monostatic', None)
        if retained and not completed:
            session.pending = None
        if factor is not None and compressed and completed:
            factor.retain_preconditioner()
        if factor is not None and not compressed:
            try:
                if retained and completed:
                    factor.restore_original()
            except BaseException:
                # A canceled or failed disk read may leave a partly restored
                # buffer; it must not be reused by the next polarization.
                session.pending = None
                raise
            finally:
                factor.close()
