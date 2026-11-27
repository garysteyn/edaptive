"""Belief-space-guided self-adaptive PSO (BS-SAPSO): von Eschwege & Engelbrecht
(2024), "A belief space-guided self-adaptive particle swarm optimizer", Swarm
Intelligence 18:31-78.

The reference implementation is calypso's ``scripts/pso/calypso.py``
(``setTrigger``, ``setBeliefSpace``, ``setControlParam``); where it differs from
the paper, this follows the code. Only calypso's ``"fixed"`` trigger and its
``elitist`` and ``improve`` selections are implemented.
"""

import numpy as np

from edaptive.core.parameters import ParameterType
from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO
from edaptive.utils.timing import timed


def masked_fitness(X, f_X, problem_lb, problem_ub):
    """``f_X`` of particles inside the problem bounds (inclusive), ``+inf`` for
    the others (calypso does not evaluate infeasible particles), and the
    feasibility mask."""
    feasible = np.all((X >= problem_lb) & (X <= problem_ub), axis=1)
    return np.where(feasible, f_X, np.inf), feasible


def one_iteration_change(masked_f_X, previous_masked_f_X):
    """``f_t - f_{t-1}`` of the masked fitness, ``+inf`` where it is NaN
    (infeasible in both steps) or ``-inf`` (infeasible to feasible). A change
    between feasible values of ``+-max_float`` overflows to ``+-inf``; ``-inf`` is
    reset to ``+inf`` too, as in calypso."""
    with np.errstate(invalid="ignore", over="ignore"):
        change = masked_f_X - previous_masked_f_X
    change[np.isnan(change) | np.isneginf(change)] = np.inf
    return change


class BSSAPSO(BaseSolver):
    """BS-SAPSO: the CPs in ``config.adapted_CP_space`` (continuous) are sampled
    per particle, uniformly and independently, from a box, the belief space, which
    is refitted to the min/max of a selected group of particles' CPs.

    **Settings come from ``config.adapter_params``** (exactly ``update_freq``,
    ``selection``, ``n_e``, ``n_e_min``). This solver builds no adapter; the
    config's ``"adapter"`` section is only where its settings live.

    - ``update_freq``: the box is updated after step ``t`` when
      ``(t + 1) % update_freq == 0``, so update ``k`` (1-based) comes after step
      ``k * update_freq - 1``. **The paper's (and calypso's) interval ``p``
      corresponds to ``update_freq = p + 1``**: calypso updates in iteration ``i``
      (``= t + 1``) when ``i - last_update > p``, i.e. at ``i = p + 1, 2(p + 1),
      ...``. So the published ``p = 200`` is ``update_freq = 201``, not 200.
    - ``selection``: ``"elitist"`` and ``"improvement"`` are the paper's elitist
      and improve selections. ``"elitist"`` selects the ``n_e`` lowest
      fitnesses, then ``n_e`` decreases by one, down to ``n_e_min``, so update
      ``k`` selects ``max(n_e - (k - 1), n_e_min)`` particles (at most ``n_s``).
      ``"improvement"`` selects the ``n_e`` lowest one-iteration changes
      ``f_t - f_{t-1}`` (``one_iteration_change``) at every update (the paper says
      ``n_e`` decreases; the code does not). This is *not* BetaEDA's window-based
      ``elite_signal = "improvement"``.

    Timeline. The box starts at the CP bounds, and the initial CPs are sampled
    from it (``initial_adapted_CPs``, before the PSO is built). After every step
    ``t``:

    1. once the belief space has collapsed, nothing happens for the rest of the
       run (no update, no resampling);
    2. the masked fitness (``masked_fitness``; ``+inf`` outside the bounds) is
       computed and kept for the next step's ``f_{t-1}`` (the initial swarm's is
       kept at construction);
    3. collapse check: if the box's widths, summed over the CPs, are below
       ``EPSILON``, the belief space has collapsed;
    4. if ``(t + 1) % update_freq == 0``: select from the CPs in use (those that
       produced ``f_t``), set the box to the min/max of the selected CPs, and
       resample the CPs of all particles in it. They are used from step ``t + 1``.

    The update that collapses the box still resamples once, in the collapsed box;
    the collapse is detected after the next step, and the CPs never change after
    that. So a run that collapses records fewer updates than its schedule has:
    with ``n`` recorded updates, the collapse was detected after step
    ``n * update_freq``.

    Selection: mechanism, consequences, and the contrast with the other EDAs
    (calypso's behaviour, kept on purpose).

    - Mechanism: every particle is ranked, infeasible ones last (all at
      ``+inf``), and ``n_e`` are always taken.
    - Consequences: infeasible particles are tied, however far outside the bounds
      they are. When fewer than ``n_e`` particles are feasible, elitist takes
      every feasible particle regardless of fitness and fills the remaining
      places with infeasible ones (lowest index first here, numpy's unstable
      tie order in calypso): selection then reflects feasibility rather than
      quality. With ``n_e = n_s`` every particle is selected, so improvement is
      fitness-blind throughout (the box is the range of the current samples),
      and so is elitist's first update. Elitist's ``n_e`` decreases by one per
      update only: with ``update_freq = 201`` a 5000-step run has 24 updates and
      never selects fewer than 7 particles.
    - Contrast: BetaEDA, the multi-valued EDAs and the EMNA adapters rank feasible
      particles only and, when fewer than they want are feasible, select only
      those. Here infeasible particles' CPs can enter the box.

    Ties are broken by particle index (stable sort). calypso uses numpy's default,
    unstable sort; particle indices are exchangeable, so the distribution of the
    selected group is the same.

    The collapse threshold ``EPSILON`` applies to the sum of the raw widths, over
    CPs with different ranges (here ``w`` in [0, 1], ``c_1``, ``c_2`` in [0, 4]),
    as in calypso. It is a test of the belief space's bounds, not of swarm
    convergence.

    RNG order (one generator, ``rnd_seed``): the initial CPs' ``rng.random`` (in
    ``initial_adapted_CPs``, before the PSO is built), the PSO's initialisation,
    then after each update one ``rng.random((n_s, n_CPs))``.

    ``history["belief_space"]`` records, per update, the new box (``belief_lbs``,
    ``belief_ubs``, rows in ``adapted_CP_space`` order) and the number of feasible
    particles (``n_feasible``): lists during the run, arrays (shapes
    ``(n_updates, n_CPs)`` and ``(n_updates,)``) after ``run``. Recording it is not
    timed.

    Timed: ``_get_required_data``, ``_select``, ``_update_belief_space``,
    ``_sample`` and ``_set_adapted_CPs``.
    """

    fixed_optimizer_type = PSO

    # Collapse threshold on the sum of the belief space's widths (calypso's
    # epsilon).
    EPSILON = 1e-9

    setting_names = ("update_freq", "selection", "n_e", "n_e_min")
    selections = ("elitist", "improvement")
    required_data_names = (
        "problem_lb",
        "problem_ub",
        "X",
        "f_X",
        "adaptable_CPs",
        "adaptable_CP_names",
    )

    def __init__(
        self, optimizer_type=None, adapter_type=None, *, problem, config, rnd_seed=0
    ):
        settings = self._validated_settings(config.adapter_params)
        space = config.adapted_CP_space
        if space is None:
            raise ValueError(f"{type(self).__name__} needs an adapted CP space")
        for parameter in space:
            if parameter.type != ParameterType.CONTINUOUS:
                raise ValueError(
                    f"{type(self).__name__} needs continuous CPs, but "
                    f"{parameter.name!r} is {parameter.type.value}"
                )

        self.update_freq = int(settings["update_freq"])
        self.selection = settings["selection"]
        self.n_e_min = int(settings["n_e_min"])
        self.CP_names = tuple(parameter.name for parameter in space)
        self.n_CPs = len(self.CP_names)

        self._n_e = int(settings["n_e"])
        self._belief_lbs = np.array(space.lower_bounds, dtype=float)
        self._belief_ubs = np.array(space.upper_bounds, dtype=float)
        self._collapse_step = None

        super().__init__(
            optimizer_type,
            adapter_type,
            problem=problem,
            config=config,
            rnd_seed=rnd_seed,
        )

        self._history = {"belief_lbs": [], "belief_ubs": [], "n_feasible": []}
        self.optimizer.history["belief_space"] = self._history

        data = self._get_required_data()
        self._previous_masked_f_X, _ = masked_fitness(
            data["X"], data["f_X"], data["problem_lb"], data["problem_ub"]
        )

    @classmethod
    def _validated_settings(cls, adapter_params):
        missing = sorted(set(cls.setting_names) - set(adapter_params))
        unexpected = sorted(set(adapter_params) - set(cls.setting_names))
        if missing or unexpected:
            raise ValueError(
                f"{cls.__name__} takes exactly the settings {list(cls.setting_names)} "
                f"in adapter_params; missing {missing}, unexpected {unexpected}"
            )
        update_freq = adapter_params["update_freq"]
        selection = adapter_params["selection"]
        n_e = adapter_params["n_e"]
        n_e_min = adapter_params["n_e_min"]
        if selection not in cls.selections:
            raise ValueError(
                f"selection must be one of {cls.selections}, got {selection!r} "
                "(calypso's other selections are not implemented)"
            )
        if int(update_freq) != update_freq or update_freq < 1:
            raise ValueError(f"update_freq must be an integer >= 1, got {update_freq}")
        if int(n_e_min) != n_e_min or n_e_min < 2:
            raise ValueError(f"n_e_min must be an integer >= 2, got {n_e_min}")
        if int(n_e) != n_e or n_e < n_e_min:
            raise ValueError(f"n_e must be an integer >= n_e_min, got {n_e}")
        return dict(adapter_params)

    def initial_adapted_CPs(self):
        samples = self._sample(self.n_s)
        return {name: samples[:, k] for k, name in enumerate(self.CP_names)}

    def run(self, max_iterations=5000):
        f_best, history = super().run(max_iterations)
        history["belief_space"] = {
            "belief_lbs": np.array(self._history["belief_lbs"], dtype=float).reshape(
                -1, self.n_CPs
            ),
            "belief_ubs": np.array(self._history["belief_ubs"], dtype=float).reshape(
                -1, self.n_CPs
            ),
            "n_feasible": np.array(self._history["n_feasible"], dtype=int),
        }
        return f_best, history

    def adapt(self, t):
        # After the collapse (see _select) the CPs never change again.
        if self._collapse_step is not None:
            return

        selected_CPs, n_feasible = self._select(t, self._get_required_data())
        if selected_CPs is None:
            return

        self._update_belief_space(selected_CPs)
        self._set_adapted_CPs(self.CP_names, self._sample(self.n_s))

        self._history["belief_lbs"].append(self._belief_lbs.copy())
        self._history["belief_ubs"].append(self._belief_ubs.copy())
        self._history["n_feasible"].append(n_feasible)

    @timed
    def _get_required_data(self):
        return self.optimizer.get_required_data(self.required_data_names)

    @timed
    def _select(self, t, data):
        """Steps 2-4 of the timeline up to the selection: the selected particles'
        CPs (best first) and the number of feasible particles if the box is
        updated after step ``t``, else ``(None, None)``."""
        masked_f_X, feasible = masked_fitness(
            data["X"], data["f_X"], data["problem_lb"], data["problem_ub"]
        )
        previous_masked_f_X = self._previous_masked_f_X
        self._previous_masked_f_X = masked_f_X

        # Collapse: the box's widths, summed over the CPs (raw widths, as in
        # calypso), are below EPSILON. From then on the box is a point and the CPs
        # never change. This tests the belief space's bounds, not swarm
        # convergence.
        if np.sum(self._belief_ubs - self._belief_lbs) < self.EPSILON:
            self._collapse_step = t
            return None, None

        if (t + 1) % self.update_freq != 0:
            return None, None

        if self.selection == "elitist":
            scores = masked_f_X
        else:
            scores = one_iteration_change(masked_f_X, previous_masked_f_X)
        selected = np.argsort(scores, kind="stable")[: self._n_e]

        if self.selection == "elitist" and self._n_e > self.n_e_min:
            self._n_e -= 1

        columns = [data["adaptable_CP_names"].index(name) for name in self.CP_names]
        return data["adaptable_CPs"][selected][:, columns], int(
            np.count_nonzero(feasible)
        )

    @timed
    def _update_belief_space(self, selected_CPs):
        self._belief_lbs = selected_CPs.min(axis=0)
        self._belief_ubs = selected_CPs.max(axis=0)

    @timed
    def _sample(self, N):
        return self._belief_lbs + self.rng.random((N, self.n_CPs)) * (
            self._belief_ubs - self._belief_lbs
        )
