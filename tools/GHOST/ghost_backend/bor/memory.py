"""Memory admission in the decimal GB used by BoR forecasts."""
_GIB_PER_GB = 1.e9 / 1024**3


def solve_memory_limit_gb():
    """Convert the shared GiB limit, including reservations, to decimal GB."""
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    return _solve_memory_limit_gb() / _GIB_PER_GB


def memory_gate_message(required_gb, limit_gb, context, details="",
                        remedies="Reduce the solve size."):
    """Report both detected availability and the solve limit in GiB."""
    from ghost_backend.twod.solver import _memory_gate_message
    return _memory_gate_message(required_gb * _GIB_PER_GB,
        limit_gb * _GIB_PER_GB, context, details, remedies, unit="GiB")
