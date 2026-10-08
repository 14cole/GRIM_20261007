"""Experimental sampled far tiles; only complete coefficient checks certify them.

Sampling proposes a low-rank matrix. It does not bound the error in unqueried
entries. ``FarProposal.validate`` therefore compares EVERY coefficient before
the result may enter a checked operator, and includes that difference in its
entrywise error budget. The normal production assembly does not call this module.
"""
import time
import numpy as np
import scipy.linalg as la


class FarProposal:
    """An unverified interpolative proposal, never an accuracy certificate."""
    def __init__(self, left, right, evidence=None):
        self.left = np.asarray(left, complex)
        self.right = np.asarray(right, complex)
        if (self.left.ndim != 2 or self.right.ndim != 2
                or self.left.shape[1] != self.right.shape[0]
                or not np.all(np.isfinite(self.left))
                or not np.all(np.isfinite(self.right))):
            raise ValueError('Invalid sampled far-tile factors.')
        self.evidence = dict(evidence or {})
        self.evidence.update(certified=False, validation='not_performed')

    def reconstruct(self):
        return self.left @ self.right

    def validate(self, exact, tail, tolerance=1e-14):
        """Return tile_payload's tuple, retaining exact coefficients on rejection.

        The reference must be the original oracle's completed tile, including
        all material routes and their existing entrywise truncation bounds.
        """
        exact, tail = np.asarray(exact), np.asarray(tail)
        if (exact.ndim != 2 or tail.shape != exact.shape
                or exact.shape != (len(self.left), self.right.shape[1])
                or not np.all(np.isfinite(exact)) or not np.all(np.isfinite(tail))
                or np.any(tail < 0) or not np.isfinite(tolerance)
                or not 0 < tolerance < 1):
            raise ValueError('Invalid complete far-tile validation inputs.')
        started = time.perf_counter()
        candidate = self.reconstruct()
        difference = abs(exact - candidate)
        norm = max(float(np.linalg.norm(exact)), 1e-300)
        relative = float(np.linalg.norm(difference)) / norm
        accepted = bool(np.all(np.isfinite(candidate)) and relative <= tolerance
                        and self.left.nbytes + self.right.nbytes < exact.nbytes)
        self.evidence.update(validation='all_coefficients', certified=accepted,
                             full_relative_error=relative, accepted=accepted,
                             validation_seconds=time.perf_counter() - started)
        if not accepted:
            return (exact, None), exact, tail, False
        # Match the qualified compression path: physical residual bounds include
        # every measured coefficient difference, never just the sampled check.
        return (self.left, self.right), candidate, tail + difference, True


def _sources(oracle):
    return list(getattr(oracle, 'oracles', (oracle,)))


def _node_ids(source, dofs):
    if hasattr(source, 'layout'):
        mapping = getattr(source, '_fast_far_node_ids', None)
        if mapping is None:
            mapping = np.full(source.n, -1, int)
            for (interface, _), (offset, count) in source.layout['dof_map'].items():
                mapping[offset:offset+count] = source.layout['ifaces'][interface]['nodes']
            if np.any(mapping < 0):
                raise ValueError('Incomplete regional degree-of-freedom mapping.')
            mapping.flags.writeable = False
            source._fast_far_node_ids = mapping
        return mapping[dofs]
    # The nonlocal thin-layer mass inverse can couple distant support. It is
    # deliberately excluded even when its nodal coordinates look separated.
    if getattr(source, 'kind', None) == 'thin':
        return None
    nn = len(source.mesh.nodes)
    if source.n not in (nn, 2*nn):
        return None
    return dofs % nn


def separated_supports(oracle, rows, cols):
    """Prove all integration supports in a tile belong to the far-pair class.

    Bounding boxes include both endpoints of EVERY panel touching a requested
    basis node. Nodal separation alone is insufficient at a corner or thin gap.
    """
    sources = _sources(oracle)
    if not sources or not hasattr(sources[0], 'geometry'):
        return False
    source = sources[0]
    row_nodes, column_nodes = _node_ids(source, rows), _node_ids(source, cols)
    if row_nodes is None or column_nodes is None:
        return False
    geometry = source.geometry
    boxes, longest = [], 0.
    for nodes in (row_nodes, column_nodes):
        mask = geometry.elements_touching(np.unique(nodes))
        if not np.any(mask):
            return False
        endpoints = np.concatenate((geometry.p0[mask], geometry.p0[mask] + geometry.segments[mask]))
        boxes.append((np.min(endpoints, axis=0), np.max(endpoints, axis=0)))
        longest = max(longest, float(np.max(geometry.lengths[mask])))
    a, b = boxes
    gap = float(np.linalg.norm(np.maximum(0., np.maximum(a[0]-b[1], b[0]-a[1]))))
    diameter = max(float(np.linalg.norm(a[1]-a[0])), float(np.linalg.norm(b[1]-b[0])))
    return gap > max(3. * longest, .5 * diameter)


def _spread_order(count):
    """Nested dyadic spread, then remaining indices, with no repeated column."""
    order, seen, intervals = [], set(), [(0, count-1)]
    for value in (0, count-1):
        if value not in seen:
            order.append(value); seen.add(value)
    while intervals:
        next_intervals = []
        for lo, hi in intervals:
            if hi-lo <= 1:
                continue
            middle = (lo+hi)//2
            if middle not in seen:
                order.append(middle); seen.add(middle)
            next_intervals.extend(((lo, middle), (middle, hi)))
        intervals = next_intervals
    return np.asarray(order, int)


def propose(oracle, rows, cols, tolerance=1e-14, maximum_samples=96):
    """Propose far tiles using exact sampled columns and pivot-selected rows.

    Returns one FarProposal for a single oracle, a list for a paired oracle, or
    None when supports are not separated or the proposal does not pay. The
    held-out columns only reject bad proposals; they never certify acceptance.
    """
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    if (rows.ndim != 1 or cols.ndim != 1 or not np.isfinite(tolerance)
            or not 0 < tolerance < 1):
        raise ValueError('Invalid sampled far-tile query.')
    if (len(np.unique(rows)) != len(rows) or len(np.unique(cols)) != len(cols)
            or np.any(rows < 0) or np.any(cols < 0)
            or np.any(rows >= oracle.n) or np.any(cols >= oracle.n)):
        raise ValueError('Far-tile indices must be unique valid degrees of freedom.')
    m, n = len(rows), len(cols)
    limit = min(int(maximum_samples), min(m, n)//4)
    if limit < 12 or not separated_supports(oracle, rows, cols):
        return None
    started = time.perf_counter()
    paired = hasattr(oracle, 'oracles')
    count = len(_sources(oracle))
    queried_entries, calls = 0, 0

    def query(rr, cc):
        nonlocal queried_entries, calls
        from ghost_backend.compressed.runtime import checkpoint
        checkpoint()
        values = oracle.get_with_error(rr, cc)
        queried_entries += len(rr)*len(cc)
        calls += 1
        return values if paired else [values]

    order = _spread_order(n)
    # Deterministic independent columns disjoint from each candidate's fit set.
    probe_order = np.random.default_rng(41923).permutation(n)
    matrices = [np.empty((m, 0), complex) for _ in range(count)]
    previous, samples = 0, min(12, limit)
    while True:
        selected = order[:samples]
        new = query(rows, cols[order[previous:samples]])
        matrices = [np.concatenate((old, item[0]), axis=1) for old, item in zip(matrices, new)]
        bases, pivot_rows, failed = [], [], False
        for matrix in matrices:
            if not np.all(np.isfinite(matrix)):
                return None
            q, r, _ = la.qr(matrix, mode='economic', pivoting=True, check_finite=False)
            diagonal = abs(np.diag(r))
            threshold = .1*tolerance*max(float(np.linalg.norm(matrix)), 1e-300)
            rank = int(np.count_nonzero(diagonal > threshold))
            if rank >= samples-2 or rank == 0:
                failed = True
                break
            q = q[:, :rank]
            _, _, pivots = la.qr(q.T, mode='economic', pivoting=True, check_finite=False)
            bases.append(q); pivot_rows.append(np.asarray(pivots[:rank]))
        if not failed:
            union = np.unique(np.concatenate(pivot_rows))
            exact_rows = query(rows[union], cols)
            proposals = []
            for base, pivots, (row_matrix, _) in zip(bases, pivot_rows, exact_rows):
                # Q is well scaled; a pivoted QR selects a stable interpolation
                # minor. Solve its transpose instead of forming an inverse.
                try:
                    left = la.solve(base[pivots].T, base.T, check_finite=False).T
                except la.LinAlgError:
                    return None
                right = row_matrix[np.searchsorted(union, pivots)]
                proposals.append(FarProposal(left, right))
            probes = probe_order[~np.isin(probe_order, selected)][:8]
            checks = query(rows, cols[probes])
            errors = []
            for proposal, (reference, _) in zip(proposals, checks):
                approximation = proposal.left @ proposal.right[:, probes]
                error = float(np.linalg.norm(reference-approximation)) / max(float(np.linalg.norm(reference)), 1e-300)
                errors.append(error)
            if max(errors) <= tolerance:
                elapsed = time.perf_counter()-started
                for proposal, error in zip(proposals, errors):
                    proposal.evidence.update(method='experimental_sampled_interpolation',
                        fit_columns=samples, rank=proposal.left.shape[1],
                        probe_columns=len(probes), sampled_relative_error=error,
                        queried_entries=queried_entries, exact_tile_entries=m*n,
                        oracle_calls=calls, proposal_seconds=elapsed,
                        full_validation_required=True)
                return proposals if paired else proposals[0]
        if samples >= limit:
            return None
        previous, samples = samples, min(limit, 2*samples)
