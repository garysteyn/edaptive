"""PSO with time-varying acceleration coefficients (PSO-TVAC).

Ratnaweera, Halgamuge & Watson (2004), "Self-organizing hierarchical particle swarm
optimizer with time-varying acceleration coefficients", IEEE Transactions on
Evolutionary Computation 8(3):240-255.
"""

import numpy as np

from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO
from edaptive.utils.timing import timed


class TVACPSO(BaseSolver):
    """PSO-TVAC: all particles share ``w``, ``c_1`` and ``c_2``, which change
    linearly with the iteration counter ``iter`` from their initial to their final
    values over ``MAXITR = max_iterations`` iterations:

        w   = w_i   + (w_f   - w_i)   * iter / MAXITR
        c_1 = c_1i  + (c_1f  - c_1i)  * iter / MAXITR
        c_2 = c_2i  + (c_2f  - c_2i)  * iter / MAXITR

    The defaults are the paper's: ``w`` from 0.9 to 0.4 (time-varying inertia
    weight), ``c_1`` from 2.5 to 0.5 and ``c_2`` from 0.5 to 2.5.

    Step ``t`` (``t = 0, ..., max_iterations - 1``) uses ``iter = t``, so the first
    step uses the initial values and the last uses ``iter = MAXITR - 1``.

    The paper clamps velocities to ``V_max`` (the dynamic range of each variable);
    ``PSO`` has no velocity clamping or boundary handling, so neither does this
    solver. Infeasible particles cannot become personal or global bests.
    """

    fixed_optimizer_type = PSO

    adapted_CP_names = ("w", "c_1", "c_2")
    initial_CPs = (0.9, 2.5, 0.5)
    final_CPs = (0.4, 0.5, 2.5)

    def run(self, max_iterations=5000):
        # The schedule needs MAXITR, which is only known once the run starts.
        self.max_iterations = max_iterations
        return super().run(max_iterations)

    def initial_adapted_CPs(self):
        return dict(zip(self.adapted_CP_names, self.initial_CPs, strict=True))

    def adapt(self, t):
        # Set the CPs for the next step, t + 1.
        CPs = self.scheduled_CPs(iteration=t + 1, max_iterations=self.max_iterations)
        self._set_adapted_CPs(self.adapted_CP_names, CPs)

    @timed
    def scheduled_CPs(self, iteration, max_iterations):
        """The scheduled ``(w, c_1, c_2)`` at ``iteration``, one row per particle."""
        initial = np.asarray(self.initial_CPs, dtype=float)
        final = np.asarray(self.final_CPs, dtype=float)
        CPs = initial + (final - initial) * iteration / max_iterations
        return np.tile(CPs, (self.n_s, 1))
