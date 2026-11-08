"""Elitist iterated F-race (I/F-race).

Follows the elitist iterated racing of irace as described in

    M. López-Ibáñez, J. Dubois-Lacoste, L. Pérez Cáceres, M. Birattari, T. Stützle,
    "The irace package: Iterated racing for automatic algorithm configuration",
    Operations Research Perspectives 3 (2016) 43-58.

Section and equation numbers in the docstrings refer to that paper. Deliberate
deviations: the first race samples a scrambled Sobol' sequence instead of uniform
random points, and parameters flagged ``log_scale_tuning`` are sampled, compared and
restarted on a log scale. Other known differences from the paper are listed in
CLAUDE.md ("Tuning: elitist I/F-race").

Rank 0 runs the algorithm; with MPI, the other ranks only run solver jobs handed out by
``IFRace._evaluate``, which is the only place that communicates.
"""

import logging
import pickle
import uuid
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from mpi4py import MPI
from numpy import floor, log2
from scikit_posthocs import posthoc_conover_friedman
from scipy.stats import friedmanchisquare, qmc, rankdata, truncnorm

from edaptive.core.parameters import ParameterType

CONFIG_ID = "_config_id"
NUMERICAL_TYPES = (ParameterType.CONTINUOUS, ParameterType.INTEGER)

_debug_log = logging.getLogger(__name__)


def _ceil_to_multiple(x, d):
    """``nm(x, d)`` in eq. 5: the smallest multiple of ``d`` not less than ``x``."""
    return int(np.ceil(x / d) * d)


def _numerical_range(param):
    """Width of a numerical parameter's range (log space for log-scale ones)."""
    if param.log_scale_tuning:
        return np.log(param.upper) - np.log(param.lower)
    return param.upper - param.lower


def _mean_ranks(data):
    """Mean rank of each configuration (columns) over the instances (rows)."""
    return rankdata(data, axis=1).mean(axis=0)


class IFRaceSamplingModel:
    """The sampling model of iterated racing (Sections 3.2 and 3.3).

    Every configuration has its own sampling distributions: a standard deviation per
    numerical parameter and a probability vector per categorical parameter. Each
    configuration is a dict of parameter values plus a ``CONFIG_ID`` key that indexes
    this state. New configurations inherit the distributions of their parent elite.
    """

    def __init__(self, parameter_space, n_races, rng, digits=4):
        self._space = parameter_space
        # N_iter in eqs. 3-4, fixed when the model is created (see CLAUDE.md).
        self._n_races = n_races
        self._rng = rng
        self._digits = digits
        self._n_configs_created = 0

        self._sigmas = {}  # config id -> {parameter name: standard deviation}
        self._probabilities = {}  # config id -> {parameter name: probabilities}

        self._elites = None
        self._elite_weights = None
        self._parent_indices = ()
        self._soft_restarted_elites = ()
        self._categorical_update_weight = None

    # ------------------------------------------------------------------ state

    def sigmas(self, config_id):
        """Copy of the standard deviations of a configuration's numerical parameters."""
        return dict(self._sigmas[config_id])

    def categorical_probabilities(self, config_id):
        """Copy of the probabilities of a configuration's categorical parameters."""
        return {name: p.copy() for name, p in self._probabilities[config_id].items()}

    @property
    def parent_indices(self):
        """Elite index of the parent of each configuration from the last ``sample``."""
        return self._parent_indices

    @property
    def soft_restarted_elites(self):
        """Indices of the elites soft-restarted by the last ``sample``."""
        return self._soft_restarted_elites

    @property
    def categorical_update_weight(self):
        """``(j - 1) / N_iter`` used by the last ``update`` (eqs. 3-4)."""
        return self._categorical_update_weight

    # --------------------------------------------------------------- sampling

    def sample_initial(self, n_configs):
        """Configurations for the first race (Algorithm 1, line 1).

        Deliberate deviation: points come from an Owen-scrambled Sobol' sequence
        instead of uniform random sampling; log-scale parameters are spread uniformly
        in log space. Standard deviations start at half the (log) range (Section 3.2)
        and categorical distributions start uniform.
        """
        sobol = qmc.Sobol(d=self._space.dimension, scramble=True, rng=self._rng)
        points = sobol.random(n=n_configs)

        configurations = []
        for point in points:
            configuration = {
                param.name: self._initial_value(param, x)
                for param, x in zip(self._space, point, strict=True)
            }
            config_id = self._next_config_id()
            configuration[CONFIG_ID] = config_id
            configurations.append(configuration)

            self._probabilities[config_id] = {
                param.name: np.full(param.n_classes, 1.0 / param.n_classes)
                for param in self._space
                if param.type is ParameterType.CATEGORICAL
            }
            self._sigmas[config_id] = {
                param.name: _numerical_range(param) / 2.0
                for param in self._space
                if param.type in NUMERICAL_TYPES
            }

        return configurations

    def update(self, elites, elite_weights, race_index, N_new):
        """Update the elites' distributions before sampling race ``race_index + 1``.

        Numerical: ``sigma *= (1 / N_new)^(1 / N_param)`` (eq. 2). Categorical: move
        ``(j - 1) / N_iter`` of the probability mass to the elite's value (eqs. 3-4),
        cap every probability at ``0.2^(1 / N_param)`` and renormalise (eq. 6).
        """
        self._elites = elites
        self._elite_weights = elite_weights
        self._categorical_update_weight = None

        if race_index <= 0 or len(elites) == 0:
            return

        N_param = len(self._space)
        weight = race_index / self._n_races
        self._categorical_update_weight = weight
        max_probability = 0.2 ** (1.0 / N_param)

        for elite in elites:
            elite_id = elite[CONFIG_ID]
            for param in self._space:
                if param.type in NUMERICAL_TYPES:
                    self._sigmas[elite_id][param.name] *= (1.0 / N_new) ** (
                        1.0 / N_param
                    )
                elif param.type is ParameterType.CATEGORICAL:
                    probabilities = self._probabilities[elite_id][param.name]
                    probabilities *= 1.0 - weight
                    probabilities[param.classes.index(elite[param.name])] += weight
                    probabilities = np.minimum(probabilities, max_probability)
                    probabilities /= probabilities.sum()
                    self._probabilities[elite_id][param.name] = probabilities

    def sample(self, N_new):
        """``N_new`` new configurations sampled from the elites (Section 3.2).

        If two candidates are at distance zero (Section 3.3), the elites involved are
        soft-restarted and the whole set is sampled once more; this happens at most
        once per call.
        """
        if not self._elites:
            raise ValueError("Cannot sample configurations without previous elites.")

        configurations, parents = self._sample_once(N_new)

        converged = self._find_converged_elites(configurations, parents)
        self._soft_restarted_elites = tuple(sorted(converged))
        if converged:
            self._soft_restart(self._soft_restarted_elites, N_new)
            configurations, parents = self._sample_once(N_new)

        self._parent_indices = tuple(parents)
        return self._register_children(configurations, parents)

    def _sample_once(self, N_new):
        """Choose a parent per eq. 1 for each new configuration, then sample from it."""
        elite_indices = np.arange(len(self._elites))
        configurations = []
        parents = []
        for _ in range(N_new):
            parent = self._rng.choice(elite_indices, p=self._elite_weights)
            elite = self._elites[parent]
            parents.append(parent)
            configurations.append(
                {param.name: self._sample_value(param, elite) for param in self._space}
            )
        return configurations, parents

    def _sample_value(self, param, elite):
        """Sample one parameter value from the elite's distribution (Section 3.2)."""
        elite_id = elite[CONFIG_ID]

        if param.type is ParameterType.CATEGORICAL:
            probabilities = self._probabilities[elite_id][param.name]
            index = self._rng.choice(len(param.classes), p=probabilities)
            return param.classes[index]

        if param.type not in NUMERICAL_TYPES:
            raise ValueError(f"Unsupported parameter type: {param.type}")

        sigma = self._sigmas[elite_id][param.name]
        if sigma <= 0:
            raise ValueError("Sampling standard deviation must be positive.")

        if param.log_scale_tuning:
            mean = np.log(elite[param.name])
            lower, upper = np.log(param.lower), np.log(param.upper)
        else:
            mean = elite[param.name]
            lower, upper = param.lower, param.upper

        if param.type is ParameterType.INTEGER and not param.log_scale_tuning:
            # Truncated normal on [lower, upper + 1), centred on mean + 0.5, then
            # floored: avoids rounding's bias against the bounds (footnote 2).
            loc = mean + 0.5
            value = truncnorm.rvs(
                (lower - loc) / sigma,
                (upper + 1.0 - loc) / sigma,
                loc=loc,
                scale=sigma,
                random_state=self._rng,
            )
            value = int(np.floor(value))
        else:
            value = truncnorm.rvs(
                (lower - mean) / sigma,
                (upper - mean) / sigma,
                loc=mean,
                scale=sigma,
                random_state=self._rng,
            )
            if param.log_scale_tuning:
                value = np.exp(value)
            if param.type is ParameterType.INTEGER:
                value = int(np.round(value))

        if param.type is ParameterType.INTEGER:
            value = max(param.lower, min(value, param.upper))
        return value

    def _initial_value(self, param, x):
        """Map a Sobol' coordinate ``x`` in [0, 1) to a parameter value."""
        if param.type is ParameterType.CONTINUOUS:
            if param.log_scale_tuning:
                return param.lower * (param.upper / param.lower) ** x
            return param.lower + x * (param.upper - param.lower)

        if param.type is ParameterType.INTEGER:
            if param.log_scale_tuning:
                value = int(np.round(param.lower * (param.upper / param.lower) ** x))
            else:
                n_values = param.upper - param.lower + 1
                value = param.lower + int(np.floor(x * n_values))
            return max(param.lower, min(value, param.upper))

        if param.type is ParameterType.CATEGORICAL:
            return param.classes[
                min(int(np.floor(x * param.n_classes)), param.n_classes - 1)
            ]

        raise ValueError(f"Unsupported parameter type: {param.type}")

    # ----------------------------------------------------------- soft restart

    def _distance(self, config1, config2):
        """Distance between two configurations (Section 3.3).

        The maximum over parameters of: the normalised absolute difference for
        numerical parameters (zero if at most ``10^-digits``; in log space for
        log-scale parameters), and 0/1 inequality for categorical parameters.
        """
        max_distance = 0.0
        threshold = 10.0**-self._digits

        for param in self._space:
            value1, value2 = config1[param.name], config2[param.name]
            if param.type in NUMERICAL_TYPES:
                if param.log_scale_tuning:
                    distance = abs(np.log(value1) - np.log(value2))
                else:
                    distance = abs(value1 - value2)
                distance /= _numerical_range(param)
                if distance > threshold:
                    max_distance = max(max_distance, distance)
            elif param.type is ParameterType.CATEGORICAL:
                if value1 != value2:
                    max_distance = max(max_distance, 1.0)

        return max_distance

    def _find_converged_elites(self, configurations, parents):
        """Elites involved in a zero-distance pair (Section 3.3).

        A pair is two new configurations, or a new configuration and an elite. Both
        new configurations' parents count, and so does the elite itself in the second
        case (deviation: the paper restarts only the parents).
        """
        converged = set()
        for i, child in enumerate(configurations):
            for j in range(i + 1, len(configurations)):
                if self._distance(child, configurations[j]) == 0:
                    converged.add(parents[i])
                    converged.add(parents[j])
            for j, elite in enumerate(self._elites):
                if self._distance(child, elite) == 0:
                    converged.add(parents[i])
                    converged.add(j)
        return converged

    def _soft_restart(self, elite_indices, N_new):
        """Partially reinitialise the elites' distributions (Section 3.3).

        Categorical: ``p = 0.9 p + 0.1 max(p)``, renormalised. Numerical: the standard
        deviation is brought back two iterations, capped at its second-iteration value:
        ``min(sigma * N_new^(2/N_param), range/2 * (1/N_new)^(1/N_param))``.
        """
        N_param = len(self._space)
        for index in elite_indices:
            elite_id = self._elites[index][CONFIG_ID]
            for param in self._space:
                if param.type is ParameterType.CATEGORICAL:
                    probabilities = self._probabilities[elite_id][param.name]
                    probabilities = 0.9 * probabilities + 0.1 * np.max(probabilities)
                    probabilities /= probabilities.sum()
                    self._probabilities[elite_id][param.name] = probabilities
                elif param.type in NUMERICAL_TYPES:
                    sigma = self._sigmas[elite_id][param.name]
                    back_two_races = float(N_new) ** (2.0 / N_param)
                    second_race = (1.0 / float(N_new)) ** (1.0 / N_param)
                    self._sigmas[elite_id][param.name] = min(
                        sigma * back_two_races,
                        (_numerical_range(param) / 2.0) * second_race,
                    )

    # ---------------------------------------------------------------- helpers

    def _register_children(self, configurations, parents):
        """Give new configurations ids and their parents' distributions (Sec. 3.2)."""
        for configuration, parent in zip(configurations, parents, strict=True):
            parent_id = self._elites[parent][CONFIG_ID]
            child_id = self._next_config_id()
            configuration[CONFIG_ID] = child_id
            self._probabilities[child_id] = {
                name: p.copy() for name, p in self._probabilities[parent_id].items()
            }
            self._sigmas[child_id] = dict(self._sigmas[parent_id])
        return configurations

    def _next_config_id(self):
        self._n_configs_created += 1
        return f"config_{self._n_configs_created}"


class IFRace:
    """Elitist iterated F-race (Algorithm 1 with the elitist race of Section 3.4).

    ``I`` holds problem classes; an instance is a ``(problem class, seed)`` pair and its
    cost is the best fitness of one ``solver.run(n_iter_per_run)``. ``B`` is the budget
    in solver runs. ``L`` (N_iter) and ``n_min`` (N_min) default to
    ``floor(2 + log2(N_param))``, ``mu`` to ``T_first``.
    """

    def __init__(
        self,
        I,
        n_x,
        B,
        L=None,
        n_min=None,
        mu=None,
        n_iter_per_run=5000,
        friedman_alpha=0.05,
        conover_alpha=0.05,
        # instances seen before the first statistical test
        T_first=5,
        # instances between later statistical tests
        T_each=1,
        # minimum number of new instances raced first in each race after the first
        T_new=1,
        # consecutive tests without elimination that end a race (Section 3.4)
        T_max=2,
        # numerical values closer than 10^-digits count as equal (soft restart)
        digits=4,
        results_directory="results/tuning_results",
    ):
        self.I = list(I)
        self.n_x = n_x
        self.B = B
        self.n_iter_per_run = n_iter_per_run
        self.friedman_alpha = friedman_alpha
        self.conover_alpha = conover_alpha
        self.T_first = T_first
        self.T_each = T_each
        self.T_new = T_new
        self.T_max = T_max
        self.mu = T_first if mu is None else mu
        self.digits = digits
        self.results_directory = Path(results_directory)

        self._L_setting = L
        self._n_min_setting = n_min

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._reset_state()

    def _reset_state(self):
        self._L = self._L_setting
        self._n_min = self._n_min_setting
        self._model = None
        self._elites = None
        self._elite_weights = None
        self._elite_histories = None  # per elite: {(instance, seed): result}
        self._best_configuration = None
        self._race_history = []
        self._log = _TuningLogger(None)

    # ------------------------------------------------------------------ state

    @property
    def L(self):
        """Number of races (N_iter); grows while budget remains."""
        return self._L

    @property
    def n_min(self):
        """Survivors at which a race stops, and the maximum number of elites (N_min)."""
        return self._n_min

    @property
    def elites(self):
        """Elite configurations after the last race (best first)."""
        return None if self._elites is None else list(self._elites)

    @property
    def elite_weights(self):
        """Parent-selection probabilities of the elites (eq. 1)."""
        return None if self._elite_weights is None else self._elite_weights.copy()

    @property
    def best_configuration(self):
        return self._best_configuration

    @property
    def race_history(self):
        """``{"race", "instance", "count"}`` per instance (instance 0: race start)."""
        return [dict(entry) for entry in self._race_history]

    @property
    def sampling_model(self):
        return self._model

    # -------------------------------------------------------------- algorithm

    def tune(
        self,
        solver_type,
        optimizer_type,
        adapter_type,
        config_template,
        tuning_parameter_space,
        rnd_seed=0,
    ):
        """Run elitist I/F-race and return the best configuration (on every rank)."""
        self._solver_type = solver_type
        self._optimizer_type = optimizer_type
        self._adapter_type = adapter_type
        self._config_template = config_template
        self._space = tuning_parameter_space
        self._reset_state()

        N_param = len(tuning_parameter_space)
        if self._n_min is None:
            self._n_min = int(floor(2 + log2(N_param)))
        if self._L is None:
            self._L = int(floor(2 + log2(N_param)))

        if self._rank == 0:
            self._rng = np.random.default_rng(rnd_seed)
            output_directory = self.results_directory / config_template.identifier
            self._log = _TuningLogger(output_directory / "logs")
            try:
                self._log.tuning_started(self, rnd_seed)
                self._iterated_racing()
                # Save the result before plotting, so a plotting failure can't lose it.
                self._save_best_configuration(output_directory)
                plot_race_history(
                    self._race_history,
                    self._n_min,
                    output_directory / "candidate_trajectory.png",
                )
            finally:
                self._stop_workers()
                self._log.close()
        else:
            self._worker_loop()

        final_state = (self._best_configuration, self._L, self._race_history)
        self._best_configuration, self._L, self._race_history = self._comm.bcast(
            final_state if self._rank == 0 else None, root=0
        )
        return self._best_configuration

    def _iterated_racing(self):
        """The outer loop of Algorithm 1 (Section 3.2).

        Race ``l`` gets ``B_l = (B - B_used) / (L - l + 1)``. Tuning stops when the
        budget is used up or no new configuration fits (``N_l <= N_elite``); if budget
        remains after race ``L``, ``L`` is increased.
        """
        B_used = 0
        l = 0
        while B_used < self.B:
            l += 1
            B_l = (self.B - B_used) / (self._L - l + 1)
            N_new = self._n_new_configurations(l, B_l)
            if N_new <= 0:
                self._log.no_new_configurations(l, N_new)
                break

            self._log.race_started(l, N_new, B_l, B_used, self._elites)
            C = self._sample_configurations(l, N_new)
            C, results, n_runs, instances = self._race(l, C, B_l)
            B_used += n_runs

            if not instances:
                done = True
            else:
                self._best_configuration = self._select_best(C, results)
                done = B_used >= self.B
                if not done:
                    self._select_elites(C, results, instances)

            self._log.race_finished(n_runs, B_used, len(C), len(instances))
            if done:
                self._log.budget_exhausted(l)
                break

            if l == self._L and B_used < self.B:
                self._L += 1
                self._log.races_extended(self._L)

        self._log.tuning_finished(self._best_configuration, B_used)

    def _n_new_configurations(self, l, B_l):
        """``N_new = N_l - N_elite`` with ``N_l`` from eq. 5 (Section 3.4).

        ``N_l = floor((B_l + N_elite * e) / max(mu + T_each * min(5, l),
        nm(T_new + e, T_each)))``, at least ``N_elite``; ``e`` is the longest elite
        history.
        """
        if l == 1 or self._elites is None:
            N_elite = 0
            e = 0
        else:
            N_elite = len(self._elites)
            e = max(len(history) for history in self._elite_histories)

        denominator = max(
            self.mu + self.T_each * min(5, l),
            _ceil_to_multiple(self.T_new + e, self.T_each),
        )
        N_l = int(np.floor((B_l + N_elite * e) / denominator))
        N_l = max(N_l, N_elite)
        return N_l - N_elite

    def _sample_configurations(self, l, N_new):
        """The configurations raced in race ``l`` (Algorithm 1, lines 1, 6 and 7).

        The first race samples the space; later races update the model, sample
        ``N_new`` configurations from the elites and race them together with the
        elites.
        """
        if l == 1:
            self._model = IFRaceSamplingModel(
                self._space, self._L, self._rng, self.digits
            )
            C = self._model.sample_initial(N_new)
            self._log.initial_configurations(l, C)
            return C

        self._model.update(self._elites, self._elite_weights, l - 1, N_new)
        self._log.model_updated(l, self._model, self._elites)
        new_configurations = self._model.sample(N_new)
        self._log.configurations_sampled(l, self._model, new_configurations)
        return self._elites + new_configurations

    def _instance_sequence(self, l):
        """The ``(instance, seed)`` order for race ``l`` and the elites' ``e``.

        Race 1: all of ``I`` in random order, each with a fresh seed. Later races
        (Section 3.4): ``T_new`` unseen instances, then the instances of the elite with
        the longest history in random order (``e`` of them), then the remaining unseen
        instances.
        """
        if l == 1:
            order = self._permuted(self.I)
            problems = [(instance, self._new_seed()) for instance in order]
            self._log.first_instance_order(problems)
            return problems, 0

        longest = int(np.argmax([len(history) for history in self._elite_histories]))
        seen_problems = self._permuted(list(self._elite_histories[longest]))
        e = len(seen_problems)

        seen = {
            instance for history in self._elite_histories for instance, _ in history
        }
        order = self._permuted(self.I)
        new_instances = [instance for instance in order if instance not in seen]
        new_instances = new_instances[: self.T_new]
        new_problems = [(instance, self._new_seed()) for instance in new_instances]

        used = seen | set(new_instances)
        remaining = [instance for instance in order if instance not in used]
        remaining_problems = [(instance, self._new_seed()) for instance in remaining]

        self._log.instance_order(e, new_instances, seen_problems, remaining)
        return new_problems + seen_problems + remaining_problems, e

    def _race(self, l, C, B_l):
        """Race the configurations ``C`` over the instances (Sections 3.2 and 3.4).

        On each instance every surviving configuration is evaluated (an elite reuses
        its cached result once the ``T_new`` new instances are done). After ``T_first``
        instances, and every ``T_each`` instances after that, a test may eliminate
        configurations. The race stops when the next instance doesn't fit in ``B_l``,
        when at most ``n_min`` configurations survive, or after ``T_max`` consecutive
        tests without elimination (counted once past ``e + T_new`` instances).

        Returns the survivors, their results per instance, the number of solver runs
        and the instances raced.
        """
        C = list(C)
        results = [[] for _ in C]
        n_runs = 0
        n_tests_without_elimination = 0
        instances = []

        self._race_history.append({"race": l, "instance": 0, "count": len(C)})
        problems, e = self._instance_sequence(l)

        for k, (instance, seed) in enumerate(problems, start=1):
            cached, to_run = self._split_cached(l, k, C, instance, seed)
            can_evaluate = n_runs + len(to_run) <= B_l
            self._log.instance_cost(k, len(to_run), n_runs, B_l, can_evaluate)
            if not can_evaluate:
                break

            outputs = self._evaluate([(C[i], instance, seed) for i in to_run])
            new_results = dict(cached)
            for i, (result, _timings) in zip(to_run, outputs, strict=True):
                new_results[i] = result
            for i in range(len(C)):
                results[i].append(new_results[i])
            self._log.instance_results(l, k, C, seed, new_results, cached)

            n_runs += len(to_run)
            instances.append((instance, seed))
            self._log.instance_completed(k, n_runs)

            if k >= self.T_first and (k - self.T_first) % self.T_each == 0:
                eliminated = self._test_and_eliminate(l, k, C, results)

                if l == 1 or k >= e + self.T_new:
                    if eliminated:
                        n_tests_without_elimination = 0
                    else:
                        n_tests_without_elimination += 1
                self._log.tests_without_elimination(
                    n_tests_without_elimination, self.T_max
                )

                for j in reversed(eliminated):
                    del C[j]
                    del results[j]
                self._log.survivors(len(C))

            self._race_history.append({"race": l, "instance": k, "count": len(C)})

            if len(C) <= self._n_min:
                self._log.race_stopped(
                    k, f"minimum configurations reached ({len(C)} <= {self._n_min})"
                )
                break
            if n_tests_without_elimination >= self.T_max:
                self._log.race_stopped(
                    k,
                    f"T_max reached ({n_tests_without_elimination} >= {self.T_max})",
                )
                break

        return C, results, n_runs, instances

    def _split_cached(self, l, k, C, instance, seed):
        """Cached elite results ``{index: result}`` and the indices that must run.

        An elite reuses its result on an ``(instance, seed)`` it has already seen, but
        only after the ``T_new`` new instances of the race.
        """
        cached = {}
        to_run = []
        histories = self._elite_histories_by_id() if l > 1 and k > self.T_new else {}
        for i, c in enumerate(C):
            history = histories.get(c[CONFIG_ID])
            if history is not None and (instance, seed) in history:
                cached[i] = history[(instance, seed)]
            else:
                to_run.append(i)
        return cached, to_run

    def _test_and_eliminate(self, l, k, C, results):
        """Indices of the configurations eliminated after instance ``k``.

        Friedman test; if significant, the Conover post-hoc test compares every
        configuration with the best (lowest mean rank) and eliminates the
        significantly worse ones (Section 3.2). An elite with ``e_j`` instances in its
        history is protected until ``k >= e_j + T_new`` (Section 3.4).
        """
        statistic, p_value = friedmanchisquare(*results)
        self._log.friedman_test(k, statistic, p_value)

        eliminated = []
        if p_value < self.friedman_alpha:
            data = np.asarray(results).T
            mean_ranks = _mean_ranks(data)
            best = np.argmin(mean_ranks)
            self._log.race_leader(best, C[best], mean_ranks[best])

            p_values = posthoc_conover_friedman(data)
            histories = self._elite_histories_by_id() if l > 1 else {}

            for j in range(len(C)):
                if j == best:
                    continue
                history = histories.get(C[j][CONFIG_ID])
                if history is not None and k < len(history) + self.T_new:
                    continue
                if p_values.iloc[best, j] < self.conover_alpha:
                    eliminated.append(j)

            self._log.eliminations(C, eliminated)

        return eliminated

    def _select_best(self, C, results):
        """The configuration with the lowest mean rank in the race just finished."""
        return C[np.argmin(_mean_ranks(np.asarray(results).T))]

    def _select_elites(self, C, results, instances):
        """Keep the ``min(N_surv, n_min)`` best survivors as elites (Section 3.2).

        Elites are ranked by mean rank in the race. Each elite's history (all
        ``(instance, seed)`` results it has, including earlier races) is kept, and the
        parent-selection weights follow eq. 1.
        """
        n_survivors = len(C)
        n_elites = min(n_survivors, self._n_min)

        ranking = np.argsort(_mean_ranks(np.asarray(results).T), kind="stable")
        elite_indices = ranking[:n_elites]

        previous_histories = (
            self._elite_histories_by_id() if self._elites is not None else {}
        )
        elites = []
        histories = []
        for i in elite_indices:
            history = dict(previous_histories.get(C[i][CONFIG_ID], {}))
            for problem, result in zip(instances, results[i], strict=True):
                history[problem] = result
            elites.append(C[i])
            histories.append(history)

        ranks = np.arange(1, n_elites + 1)
        weights = (n_elites - ranks + 1) / (n_elites * (n_elites + 1) / 2)

        self._elites = elites
        self._elite_weights = weights
        self._elite_histories = histories
        self._log.elites_selected(elite_indices, elites, weights, histories)

    def _elite_histories_by_id(self):
        return {
            elite[CONFIG_ID]: history
            for elite, history in zip(self._elites, self._elite_histories, strict=True)
        }

    def _permuted(self, items):
        return [items[i] for i in self._rng.permutation(len(items))]

    def _new_seed(self):
        return self._rng.integers(0, 2**32 - 1)

    # ------------------------------------------------------------- evaluation

    def _evaluate(self, jobs):
        """Run ``(configuration, instance, seed)`` jobs, spread over the MPI ranks.

        Called on rank 0 only; every other rank serves it from ``_worker_loop``.
        Returns ``(result, timings)`` per job, in job order.
        """
        self._comm.bcast(jobs, root=0)
        gathered = self._comm.gather(self._run_share(jobs), root=0)
        outputs = [None] * len(jobs)
        for rank_outputs in gathered:
            for index, output in rank_outputs:
                outputs[index] = output
        return outputs

    def _worker_loop(self):
        """Ranks other than 0: run each job list from ``_evaluate`` until ``None``."""
        while True:
            jobs = self._comm.bcast(None, root=0)
            if jobs is None:
                return
            self._comm.gather(self._run_share(jobs), root=0)

    def _stop_workers(self):
        self._comm.bcast(None, root=0)

    def _run_share(self, jobs):
        return [
            (index, self._run_solver(*jobs[index]))
            for index in range(self._rank, len(jobs), self._size)
        ]

    def _run_solver(self, c, instance, seed):
        solver = self._solver_type(
            optimizer_type=self._optimizer_type,
            adapter_type=self._adapter_type,
            problem=instance(dimensions=self.n_x),
            config=self.create_config(c),
            rnd_seed=seed,
        )
        result, history = solver.run(max_iterations=self.n_iter_per_run)
        _debug_log.debug(
            "[rank %d] %s, seed %s, configuration %s: result=%s",
            self._rank,
            instance.__name__,
            seed,
            c,
            result,
        )
        return result, history["timings"]

    def create_config(self, c):
        """The SolverConfig for configuration ``c`` (tuned values filled in)."""
        return self._config_template.with_values(
            {k: v for k, v in c.items() if k != CONFIG_ID}
        )

    def _save_best_configuration(self, output_directory):
        path = output_directory / "best_configuration.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.create_config(self._best_configuration).to_dict(), f)
        self._log.best_configuration_saved(path)


def plot_race_history(race_history, n_min, path):
    """Plot the number of surviving configurations over every race and instance.

    Uses a bare ``Figure`` rather than ``pyplot``, so no GUI backend or window is
    involved (TkAgg segfaults in this environment). It only writes to ``path``.
    """
    if not race_history:
        return

    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()
    x = np.arange(len(race_history))
    counts = [entry["count"] for entry in race_history]
    labels = [
        f"R{entry['race']}-I{entry['instance']}"
        if entry["instance"] > 0
        else f"R{entry['race']}-Start"
        for entry in race_history
    ]

    ax.plot(
        x,
        counts,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#1f77b4",
        label="Surviving Configurations",
    )
    for index in range(1, len(race_history)):
        if race_history[index]["race"] != race_history[index - 1]["race"]:
            ax.axvline(x=index - 0.5, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(
        y=n_min,
        color="red",
        linestyle=":",
        label=f"Min. Candidates Threshold ($N_{{min}}={n_min}$)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Race & Instance Step (Race-Instance)", fontsize=11)
    ax.set_ylabel("Number of Candidates", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)


def _names(instances):
    return [instance.__name__ for instance in instances]


class _TuningLogger:
    """All of IFRace's log output: one method per event, five log files.

    Writes ``tuning``, ``races``, ``evaluations``, ``elites`` and ``sampling`` logs into
    ``log_directory``. With ``log_directory=None`` (ranks other than 0) it does
    nothing.
    """

    FILES = ("tuning", "evaluations", "races", "elites", "sampling")

    def __init__(self, log_directory):
        self._loggers = {}
        if log_directory is None:
            return

        log_directory.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        for name in self.FILES:
            logger = logging.getLogger(f"edaptive.ifrace.{run_id}.{name}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.FileHandler(log_directory / f"{name}.log")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self._loggers[name] = logger

    def close(self):
        for logger in self._loggers.values():
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        self._loggers = {}

    def _write(self, name, *messages):
        logger = self._loggers.get(name)
        if logger is not None:
            for message in messages:
                logger.info(message)

    # ---------------------------------------------------------------- tuning

    def tuning_started(self, tuner, rnd_seed):
        self._write(
            "tuning",
            "=" * 60,
            "Elitist I/F-RACE TUNING",
            "=" * 60,
            f"Solver: {tuner._solver_type.__name__}",
            f"Optimizer: {tuner._optimizer_type.__name__}",
            *(
                [f"Adapter: {tuner._adapter_type.__name__}"]
                if tuner._adapter_type
                else []
            ),
            f"Budget: {tuner.B}",
            f"Number of races: {tuner.L}",
            f"Minimum configurations: {tuner.n_min}",
            f"T_first: {tuner.T_first}",
            f"T_each: {tuner.T_each}",
            f"T_new: {tuner.T_new}",
            f"T_max: {tuner.T_max}",
            f"mu: {tuner.mu}",
            f"Random seed: {rnd_seed}",
            f"Number of tuning parameters: {len(tuner._space)}",
            f"Number of MPI processes: {tuner._size}",
            "",
        )

    def no_new_configurations(self, l, N_new):
        self._write("tuning", f"Race {l} stopping: N_new ({N_new}) <= 0.")

    def race_started(self, l, N_new, B_l, B_used, elites):
        self._write(
            "tuning",
            "",
            "-" * 60,
            f"RACE {l}",
            "-" * 60,
            f"New configurations generated: {N_new}",
            f"Race budget: {B_l}",
            f"Budget used before race: {B_used}",
        )
        if l > 1:
            self._write("tuning", f"Elite configurations carried over: {len(elites)}")
        self._write("races", "", "=" * 60, f"RACE {l}", "=" * 60, f"Race budget: {B_l}")
        if l > 1:
            self._write("races", f"Previous elites: {len(elites)}")
            for i, elite in enumerate(elites, start=1):
                self._write("races", f"Elite {i}: {elite}")

    def race_finished(self, n_runs, B_used, n_survivors, n_instances):
        self._write(
            "tuning",
            f"Race evaluations: {n_runs}",
            f"Cumulative budget used: {B_used}",
            f"Configurations remaining: {n_survivors}",
            f"Instances evaluated: {n_instances}",
        )

    def budget_exhausted(self, l):
        self._write(
            "tuning", f"Tuning completed after race {l}: global budget exhausted."
        )

    def races_extended(self, L):
        self._write("tuning", f"Budget remains. Extended total iterations (L) to {L}.")

    def tuning_finished(self, best_configuration, B_used):
        self._write(
            "tuning",
            "Tuning completed.",
            f"Final best configuration: {best_configuration}",
            f"Total evaluations: {B_used}",
        )

    def best_configuration_saved(self, path):
        self._write("tuning", f"Saved best configuration to {path}")

    # -------------------------------------------------------------- sampling

    def initial_configurations(self, l, C):
        self._write("sampling", f"Race {l}: initial configurations={len(C)}")
        self._write("races", f"Initial configurations: {len(C)}")

    def model_updated(self, l, model, elites):
        if model.categorical_update_weight is not None:
            self._write(
                "sampling",
                f"Race {l}: categorical update weight="
                f"{model.categorical_update_weight:.6f}",
            )
        self._write(
            "sampling",
            f"Race {l}: sampling distributions updated using {len(elites)} elite "
            f"configurations carried forward from race {l - 1}",
        )
        for index, elite in enumerate(elites, start=1):
            config_id = elite[CONFIG_ID]
            for name, sigma in model.sigmas(config_id).items():
                self._write(
                    "sampling", f"Race {l}: sigma [elite {index}][{name}]={sigma}"
                )
            for name, p in model.categorical_probabilities(config_id).items():
                self._write(
                    "sampling",
                    f"Race {l}: categorical probabilities [elite {index}][{name}]={p}",
                )

    def configurations_sampled(self, l, model, configurations):
        if model.soft_restarted_elites:
            elites = [int(index) + 1 for index in model.soft_restarted_elites]
            self._write(
                "sampling",
                f"Race {l}: soft restart of elites {elites}; configurations resampled",
            )
        for i, (parent, configuration) in enumerate(
            zip(model.parent_indices, configurations, strict=True), start=1
        ):
            self._write(
                "sampling",
                f"Race {l}: new configuration {i} (parent elite {int(parent) + 1}): "
                f"{configuration}",
            )
        self._write("races", f"New configurations: {len(configurations)}")

    # ----------------------------------------------------------------- races

    def first_instance_order(self, problems):
        self._write(
            "races",
            f"Initial instance order: {_names(instance for instance, _ in problems)}",
        )

    def instance_order(self, e, new_instances, seen_problems, remaining):
        self._write(
            "races",
            f"Maximum elite history (e): {e}",
            f"New instances: {_names(new_instances)}",
            "Previous elite instances: "
            f"{_names(instance for instance, _ in seen_problems)}",
            f"Remaining instances: {_names(remaining)}",
        )

    def instance_cost(self, k, cost, n_runs, B_l, can_evaluate):
        self._write(
            "races",
            f"Instance {k}: evaluation cost={cost}, run_counter={n_runs}, "
            f"B_l={B_l}, can_evaluate={can_evaluate}",
        )

    def instance_results(self, l, k, C, seed, results, cached):
        for i, c in enumerate(C):
            if i in cached:
                message = f"cached result reused = {results[i]}"
            else:
                message = f"evaluated, result={results[i]}, seed={seed}"
            self._write(
                "evaluations", f"Race {l}, instance {k}, configuration {c}: {message}"
            )

    def instance_completed(self, k, n_runs):
        self._write("races", f"Instance {k} completed: evaluations used={n_runs}")

    def friedman_test(self, k, statistic, p_value):
        self._write(
            "races",
            f"Statistical test at instance {k}: Friedman statistic={statistic:.6f}, "
            f"p-value={p_value:.6g}",
        )

    def race_leader(self, index, configuration, mean_rank):
        self._write(
            "races",
            f"Best configuration: index={index}, configuration={configuration}, "
            f"mean rank={mean_rank:.6f}",
        )

    def eliminations(self, C, eliminated):
        self._write(
            "races", f"Configurations marked for elimination: {len(eliminated)}"
        )
        for j in eliminated:
            self._write("races", f"Eliminating configuration index={j}: {C[j]}")

    def tests_without_elimination(self, count, T_max):
        self._write("races", f"No-elimination tests: {count}/{T_max}")

    def survivors(self, n_survivors):
        self._write(
            "races", f"Configurations remaining after elimination: {n_survivors}"
        )

    def race_stopped(self, k, reason):
        self._write("races", f"Race stopping at instance {k}: {reason}")

    # ---------------------------------------------------------------- elites

    def elites_selected(self, elite_indices, elites, weights, histories):
        for rank, history in enumerate(histories, start=1):
            self._write(
                "elites",
                f"Elite {rank}: history length={len(history)}",
                f"Elite {rank}: instances={_names(i for i, _ in history)}",
                f"Elite {rank}: results={list(history.values())}",
            )
        self._write("elites", f"Selected {len(elites)} elite configurations")
        for rank, (index, configuration, weight) in enumerate(
            zip(elite_indices, elites, weights, strict=True), start=1
        ):
            self._write(
                "elites",
                f"Elite {rank}: configuration index={index}, weight={weight:.6f}, "
                f"configuration={configuration}",
            )
