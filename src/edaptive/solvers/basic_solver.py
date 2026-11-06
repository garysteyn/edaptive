from edaptive.core.solver import BaseSolver


class BasicSolver(BaseSolver):
    """Runs the optimizer with static CPs: everything comes from ``fixed_CPs``."""
