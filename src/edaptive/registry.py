"""Names accepted on the command line for solvers, optimizers and adapters.

``experiments/scripts/tune.py`` and ``execute.py`` look up their ``--solver``,
``--optimizer`` and ``--adapter`` arguments here. Register new classes below.
"""

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.adapters.r_cga import RcGA
from edaptive.adapters.r_pbil import RPBIL
from edaptive.adapters.r_umda import RUMDA
from edaptive.adapters.uniform_random import UniformRandomAdapter
from edaptive.optimizers.pso import PSO
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.pso.sg_pso import StabilityGuidedPSO
from edaptive.solvers.pso.tvac_pso import TVACPSO
from edaptive.solvers.uniform_random_solver import UniformRandomSolver

SOLVERS = {
    cls.__name__: cls
    for cls in (
        BasicSolver,
        EDASolver,
        UniformRandomSolver,
        StabilityGuidedPSO,
        TVACPSO,
    )
}
# Old names still accepted on the command line.
SOLVERS["stability_guided_PSO"] = StabilityGuidedPSO

OPTIMIZERS = {cls.__name__: cls for cls in (PSO,)}

ADAPTERS = {
    cls.__name__: cls for cls in (BetaEDA, UniformRandomAdapter, RUMDA, RPBIL, RcGA)
}

_REGISTRIES = {"solver": SOLVERS, "optimizer": OPTIMIZERS, "adapter": ADAPTERS}


def resolve(kind, name):
    """Return the class registered as ``name`` for ``kind`` (solver, optimizer or
    adapter), or ``None`` if ``name`` is ``None``."""
    if name is None:
        return None
    registry = _REGISTRIES[kind]
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} {name!r}; choose from {sorted(registry)}"
        ) from None
