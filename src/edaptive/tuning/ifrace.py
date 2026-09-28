import numpy as np
from numpy import round, log2
from scipy.stats import friedmanchisquare, rankdata
from copy import deepcopy
from mpi4py import MPI
from scikit_posthocs import posthoc_conover_friedman

import logging
from datetime import datetime
from pathlib import Path

def set_parameters(config: dict, updates: dict) -> None:
    for key, value in config.items():

        if key in updates:
            config[key] = updates[key]

        elif isinstance(value, dict):
            set_parameters(value, updates)

class IFRace():
    def __init__(
            self,
            I,
            n_x,
            sampling_model,
            B, # budget
            L=None, # number of iterations
            n_min=None,
            mu=None,
            n_iter_per_run = 5000,
            friedman_alpha=0.05,
            conover_alpha=0.05,
            T_first=5, # no. initial instances evaluated before statistical tests are performed
            T_each=1, # frequency of statistical tests after the T_first initial evaluations
            T_new=1, # minimum number of new instances introduced at the beginning of each subsequent race
            T_max=2 # maximum number of consecutive statistical tests with no elimination before the race terminates
    ):

        self.I = I
        self.n_x = n_x
        self.n_iter_per_run = n_iter_per_run
        self.sampling_model = sampling_model()
        self.L = L
        self.B = B

        self.friedman_alpha = friedman_alpha
        self.conover_alpha = conover_alpha

        self.T_first = T_first
        self.T_each = T_each
        self.T_new = T_new
        self.T_max = T_max

        self.L = L
        self.n_min = n_min

        if mu is None:
            self.mu = T_first
        else:
            self.mu = mu

        self.solver = None

        self.best_configuration = None

        self.prev_elites = None
        self.prev_weights = None
        self.prev_elite_results = None
        self.prev_elite_instances = None

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def nm(self, x, d):
        return int(np.ceil(x / d) * d)

    def tune(self, solver_type, optimizer_type, adapter_type, config_template, tuning_parameter_space, rnd_seed=0):
        self.solver_type = solver_type
        self.optimizer_type = optimizer_type
        self.adapter_type = adapter_type
        self.config_template = config_template
        self.tuning_parameter_space = tuning_parameter_space

        if self.n_min is None:
            self.n_min = 2 + int(round(log2(len(tuning_parameter_space))))

        if self.L is None:
            self.L = 2 + int(round(log2(len(tuning_parameter_space))))

        self.master_rng = np.random.default_rng(rnd_seed)

        self.setup_loggers()

        self.log("tuning", "=" * 60)
        self.log("tuning", "Elitist I/F-RACE TUNING")
        self.log("tuning", "=" * 60)

        self.log("tuning", f"Solver: {solver_type}")
        self.log("tuning", f"Optimizer: {optimizer_type}")
        self.log("tuning", f"Adapter: {adapter_type}")
        self.log("tuning", f"Budget: {self.B}")
        self.log("tuning", f"Number of races: {self.L}")
        self.log("tuning", f"Minimum configurations: {self.n_min}")
        self.log("tuning", f"T_first: {self.T_first}")
        self.log("tuning", f"T_each: {self.T_each}")
        self.log("tuning", f"T_new: {self.T_new}")
        self.log("tuning", f"T_max: {self.T_max}")
        self.log("tuning", f"mu: {self.mu}")
        self.log("tuning", f"Random seed: {rnd_seed}")
        self.log("tuning", f"Number of tuning parameters: {len(tuning_parameter_space)}")
        self.log("tuning", f"Number of MPI processes: {self.size}")
        self.log("tuning", "")

        B_used = 0

        for l in range(1, self.L + 1):
            B_l = (self.B - B_used) / (self.L - l + 1)

            N_l = self.calc_num_candidate_configs(l, B_l)

            self.log("tuning", "")
            self.log("tuning", "-" * 60)
            self.log("tuning", f"RACE {l}")
            self.log("tuning", "-" * 60)
            self.log("tuning", f"Configurations generated: {N_l}")
            self.log("tuning", f"Race budget: {B_l}")
            self.log("tuning", f"Budget used before race: {B_used}")

            if l > 1:
                self.log(
                    "tuning",
                    f"Elite configurations carried over: {len(self.prev_elites)}"
                )

            C = self.generate_configs(l, N_l)

            C, results, run_counter, instances = self.execute_race(
                l,
                C,
                B_l
            )

            B_used += run_counter

            stop = self.finalize_race(B_used, l, C, results, run_counter, instances)

            self.log("tuning", f"Race evaluations: {run_counter}")
            self.log("tuning", f"Cumulative budget used: {B_used}")
            self.log("tuning", f"Configurations remaining: {len(C)}")
            self.log("tuning", f"Instances evaluated: {len(instances)}")
            self.log("tuning", f"Race stopped: {stop}")

            if stop:
                self.log(
                    "tuning",
                    f"Tuning completed after race {l}"
                )
                self.log(
                    "tuning",
                    f"Final best configuration: {self.best_configuration}"
                )
                self.log(
                    "tuning",
                    f"Total evaluations: {B_used}"
                )

                return self.best_configuration

        self.log(
            "tuning",
            f"Tuning completed after race {self.L}"
        )
        self.log(
            "tuning",
            f"Final best configuration: {self.best_configuration}"
        )
        self.log(
            "tuning",
            f"Total evaluations: {B_used}"
        )

        return self.best_configuration

    def finalize_race(self, B_used, l, C, results, run_counter, instances):
        if self.rank == 0:
            # No instances were evaluated within the budget
            if len(instances) == 0:
                stop = True
            else:
                    # Determine best configuration from this race
                self.best_configuration = self.select_best(
                        C,
                        results
                    )
                
                    # Determine whether tuning should stop
                stop = (
                        (l == self.L) or (B_used >= self.B)
                    )

                if not stop:
                    (
                            self.prev_elites,
                            self.prev_weights,
                            self.prev_elite_instances,
                            self.prev_elite_results
                        ) = self.select_elites(
                            C,
                            results,
                            instances
                        )
                        # print(self.prev_elite_results)
        else:
            stop = None


            # ---------------------------------------------------------
            # Broadcast stopping decision
            # ---------------------------------------------------------

        stop = self.comm.bcast(
                stop,
                root=0
            )

            # ---------------------------------------------------------
            # Broadcast best configuration
            # ---------------------------------------------------------

        self.best_configuration = self.comm.bcast(
                self.best_configuration if self.rank == 0 else None,
                root=0
            )
        return stop

    def generate_configs(self, l, N_l):
        if self.rank == 0:
            if l == 1:
                C = self.sampling_model.initialize(
                        self.tuning_parameter_space,
                        N_l,
                        self.L,
                        self.master_rng
                    )

                self.log(
                    "sampling",
                    f"Race {l}: initial configurations={len(C)}"
                )
            else:
                self.sampling_model.update(
                        prev_elites=self.prev_elites,
                        prev_weights=self.prev_weights,
                        prev_l = l - 1,
                        N_l=N_l
                    )

                if self.sampling_model.categorical_update_elite is not None:
                    self.log(
                        "sampling",
                        f"Race {l}: categorical update elite="
                        f"{self.sampling_model.categorical_update_elite + 1}, "
                        f"weight={self.sampling_model.categorical_update_weight:.6f}"
                    )

                self.log(
                    "sampling",
                    f"Race {l}: numerical sampling distributions updated "
                    f"using {len(self.prev_elites)} elite configurations "
                    f"carried forward from race {l - 1}"
                )

                for parameter, sigma in self.sampling_model.sigma.items():
                    self.log(
                        "sampling",
                        f"Race {l}: sigma[{parameter}]={sigma}"
                    )

                for parameter, probabilities in (
                    self.sampling_model.categorical_probabilities.items()
                ):
                    self.log(
                        "sampling",
                        f"Race {l}: categorical probabilities "
                        f"[{parameter}]={probabilities}"
                    )

                n_new = N_l - len(self.prev_elites)

                self.log(
                    "sampling",
                    f"Race {l}: total configurations={N_l}, "
                    f"carried-over elites={len(self.prev_elites)}, "
                    f"new configurations={n_new}"
                )

                new_configurations = self.sampling_model.sample(n_new)

                for i, elite_index in enumerate(
                    self.sampling_model.sampled_elites,
                    start=1
                ):
                    self.log(
                        "sampling",
                        f"Race {l}: new configuration {i} "
                        f"sampled from elite {elite_index + 1}"
                    )

                for i, configuration in enumerate(new_configurations, start=1):
                    self.log(
                        "sampling",
                        f"Race {l}: new configuration {i}: {configuration}"
                    )

                C = self.prev_elites + new_configurations

        C = self.comm.bcast(
            C if self.rank == 0 else None,
            root=0
            )
        
        return C

    def calc_num_candidate_configs(self, l, B_l):
        if l == 1:
            N_elite = 0
            e = 0
        else:
            N_elite = len(self.prev_elites)
            e = max(
                    len(instances)
                    for instances in self.prev_elite_instances
                )
        denominator = max(
                self.mu + self.T_each * min(5, l),
                self.nm(self.T_new + e, self.T_each)
            )

        N_l = int(
                np.floor(
                    (B_l + N_elite * e) / denominator
                )
            )

        N_l = max(N_l, N_elite)
        
        return N_l

    def select_best(self, configurations, results):
        data = np.asarray(results).T
        mean_ranks = self.calculate_mean_ranks(data)
        best_index = np.argmin(mean_ranks)

        return configurations[best_index]

    def select_elites(self, configurations, results, instances):
        n_survive = len(configurations)

        n_elites = min(n_survive, self.n_min)

        data = np.asarray(results).T
        mean_ranks = self.calculate_mean_ranks(data)

        ranking = np.argsort(mean_ranks)

        elite_indices = ranking[:n_elites]

        elite_configurations = [
            configurations[i]
            for i in elite_indices
        ]

        elite_instances = []
        elite_results = []

        for elite_rank, i in enumerate(elite_indices, start=1):

            if (
                self.prev_elites is not None
                and configurations[i] in self.prev_elites
            ):
                previous_index = self.prev_elites.index(
                    configurations[i]
                )

                history = dict(
                    zip(
                        self.prev_elite_instances[previous_index],
                        self.prev_elite_results[previous_index]
                    )
                )
            else:
                history = {}

            for instance, result in zip(instances, results[i]):
                history[instance] = result

            elite_instances.append(
                list(history.keys())
            )

            elite_results.append(
                list(history.values())
            )

            self.log(
                "elites",
                f"Elite {elite_rank}: history length={len(history)}"
            )

            self.log(
                "elites",
                f"Elite {elite_rank}: instances={list(history.keys())}"
            )

            self.log(
                "elites",
                f"Elite {elite_rank}: results={list(history.values())}"
            )

        ranks = np.arange(1, n_elites + 1)

        weights = (
            n_elites - ranks + 1
        ) / (
            n_elites * (n_elites + 1) / 2
        )

        self.log(
            "elites",
            f"Selected {n_elites} elite configurations"
        )

        for rank, (index, configuration, weight) in enumerate(
            zip(elite_indices, elite_configurations, weights),
            start=1
        ):
            self.log(
                "elites",
                f"Elite {rank}: "
                f"configuration index={index}, "
                f"weight={weight:.6f}, "
                f"configuration={configuration}"
            )

        return (
            elite_configurations,
            weights,
            elite_instances,
            elite_results
        )

    def calculate_mean_ranks(self, data):
        ranks = np.array([
            rankdata(instance_results)
            for instance_results in data
        ])
        return ranks.mean(axis=0)

    def execute_race(self, l, C, B_l):

        self.log("races", "")
        self.log("races", "=" * 60)
        self.log("races", f"RACE {l}")
        self.log("races", "=" * 60)
        self.log("races", f"Initial configurations: {len(C)}")
        self.log("races", f"Race budget: {B_l}")

        if l > 1:
            self.log(
                "races",
                f"Previous elites: {len(self.prev_elites)}"
            )

            for i, elite in enumerate(self.prev_elites):
                self.log(
                    "races",
                    f"Elite {i + 1}: {elite}"
                )
                self.log(
                    "races",
                    f"Elite {i + 1} history length: "
                    f"{len(self.prev_elite_instances[i])}"
                )

        run_counter = 0
        no_elimination_tests = 0
        evaluated_instances = []

        if self.rank == 0:
            if l == 1:
                problems = self.master_rng.permutation(self.I)
            else:
                reference_elite_index = max(
                    range(len(self.prev_elite_instances)),
                    key=lambda i: len(self.prev_elite_instances[i])
                )

                e = len(
                    self.prev_elite_instances[reference_elite_index]
                )

                previous_instances = list(
                    self.master_rng.permutation(
                        self.prev_elite_instances[reference_elite_index]
                    )
                )

                previous_elite_results = [
                    dict(
                        zip(
                            self.prev_elite_instances[elite_index],
                            self.prev_elite_results[elite_index]
                        )
                    )
                    for elite_index in range(len(self.prev_elites))
                ]

                all_elite_instances = set().union(
                    *self.prev_elite_instances
                )

                instance_order = self.master_rng.permutation(self.I)

                new_instances = [
                    instance
                    for instance in instance_order
                    if instance not in all_elite_instances
                ][:self.T_new]

                used_instances = set(new_instances) | set(previous_instances)

                remaining_instances = [
                    instance
                    for instance in instance_order
                    if instance not in used_instances
                ]

                problems = np.concatenate([
                    new_instances,
                    previous_instances,
                    remaining_instances
                ])

                self.log(
                    "races",
                    f"Maximum elite history (e): {e}"
                )
                self.log(
                    "races",
                    f"New instances: {new_instances}"
                )
                self.log(
                    "races",
                    f"Previous elite instances: {previous_instances}"
                )
                self.log(
                    "races",
                    f"Remaining instances: {remaining_instances}"
                )

            if l == 1:
                self.log(
                    "races",
                    f"Initial instance order: {list(problems)}"
                )

            results = [[] for _ in C]

        else:
            problems = None
            results = None

        # Everyone needs to know whether to continue.
        problems = self.comm.bcast(
            problems,
            root=0
        )

        if self.rank == 0:
            if l == 1:
                previous_elite_results = None

        previous_elite_results = self.comm.bcast(
            previous_elite_results if self.rank == 0 else None,
            root=0
        )

        for k, I in enumerate(problems, start=1):

            # ---------------------------------------------------------
            # Determine whether another complete instance can be
            # evaluated within the remaining budget.
            # ---------------------------------------------------------

            if self.rank == 0:
                if l > 1 and k > self.T_new:

                    evaluation_cost = 0

                    for c in C:
                        if c not in self.prev_elites:
                            evaluation_cost += 1
                            continue

                        elite_index = self.prev_elites.index(c)

                        if problems[k - 1] not in previous_elite_results[elite_index]:
                            evaluation_cost += 1

                else:
                    evaluation_cost = len(C)

                can_evaluate = run_counter + evaluation_cost <= B_l
            else:
                can_evaluate = None

            # Everyone needs to know whether to continue.
            can_evaluate = self.comm.bcast(
                can_evaluate,
                root=0
            )

            if self.rank == 0:
                self.log(
                    "races",
                    f"Instance {k}: evaluation cost={evaluation_cost}, "
                    f"run_counter={run_counter}, B_l={B_l}, "
                    f"can_evaluate={can_evaluate}"
                )

            if not can_evaluate:
                break

            if self.rank == 0:
                iteration_seed = self.master_rng.integers(0, 2**32 - 1)
            else:
                iteration_seed = None

            iteration_seed = self.comm.bcast(
                iteration_seed,
                root=0
            )

            # ---------------------------------------------------------
            # Evaluate configurations in parallel
            # ---------------------------------------------------------

            local_results = []
            local_run_count = 0

            for i in range(self.rank, len(C), self.size):

                c = C[i]

                if (
                    l > 1
                    and c in self.prev_elites
                    and k > self.T_new
                ):
                    elite_index = self.prev_elites.index(c)

                    instance = problems[k - 1]
                    cached_results = previous_elite_results[elite_index]

                    if instance in cached_results:
                        result = cached_results[instance]

                        self.log(
                            "evaluations",
                            f"Race {l}, instance {k}, configuration {c}: "
                            f"cached result reused = {result}"
                        )
                    else:
                        config = self.create_config_dict(c)
                        solver = self.solver_type(
                            optimizer_type=self.optimizer_type,
                            adapter_type=self.adapter_type,
                            problem=I(dimensions=self.n_x),
                            hyper_params=config,
                            rnd_seed=iteration_seed
                            )

                        result, _ = solver.run(
                            max_iterations=self.n_iter_per_run
                        )
                        local_run_count += 1

                        self.log(
                            "evaluations",
                            f"Race {l}, instance {k}, configuration {c}: "
                            f"evaluated, result={result}, seed={iteration_seed}"
                        )
                else:
                    config = self.create_config_dict(c)
                    solver = self.solver_type(
                        optimizer_type=self.optimizer_type,
                        adapter_type=self.adapter_type,
                        problem=I(dimensions=self.n_x),
                        hyper_params=config,
                        rnd_seed=iteration_seed
                        )

                    result, _ = solver.run(
                        max_iterations=self.n_iter_per_run
                    )
                    local_run_count += 1

                    self.log(
                        "evaluations",
                        f"Race {l}, instance {k}, configuration {c}: "
                        f"evaluated, result={result}, seed={iteration_seed}"
                    )

                local_results.append((i, result))

                print(
                    f"[Rank {self.rank}] "
                    f"Instance {k}, configuration {c}, "
                    f"result={result}"
                )

            # ---------------------------------------------------------
            # Gather results from all MPI ranks
            # ---------------------------------------------------------

            gathered_results = self.comm.gather(
                local_results,
                root=0
            )


            # ---------------------------------------------------------
            # Rank 0 reconstructs the complete results
            # ---------------------------------------------------------

            if self.rank == 0:
                for rank_results in gathered_results:
                    for i, result in rank_results:
                        results[i].append(result)

            run_counter += self.comm.allreduce(
                local_run_count,
                op=MPI.SUM
            )

            if self.rank == 0:
                evaluated_instances.append(I)

                self.log(
                    "races",
                    f"Instance {k} completed: "
                    f"evaluations used={run_counter}"
                )

                # -------------------------------------------------------------
                # Determine whether elites are still protected
                # -------------------------------------------------------------

                # -------------------------------------------------------------
                # Statistical test
                # -------------------------------------------------------------

                if (
                    k >= self.T_first
                    and (k - self.T_first) % self.T_each == 0
                ):
                    statistic, p_value = friedmanchisquare(*results)

                    self.log(
                        "races",
                        f"Statistical test at instance {k}: "
                        f"Friedman statistic={statistic:.6f}, "
                        f"p-value={p_value:.6g}"
                    )

                    eliminate = []

                    if p_value < self.friedman_alpha:

                        data = np.asarray(results).T
                        mean_ranks = self.calculate_mean_ranks(data)

                        # Best configuration
                        best_index = np.argmin(mean_ranks)

                        self.log(
                            "races",
                            f"Best configuration: index={best_index}, "
                            f"configuration={C[best_index]}, "
                            f"mean rank={mean_ranks[best_index]:.6f}"
                        )

                        p_values = posthoc_conover_friedman(data)

                        # # Determine configurations to eliminate

                        for j in range(len(C)):

                            if j == best_index:
                                continue

                            if l > 1 and C[j] in self.prev_elites:
                                elite_index = self.prev_elites.index(C[j])
                                e_j = len(self.prev_elite_instances[elite_index])

                                if k < e_j + self.T_new:
                                    continue

                            if p_values.iloc[best_index, j] < self.conover_alpha:
                                eliminate.append(j)

                        self.log(
                            "races",
                            f"Configurations marked for elimination: {len(eliminate)}"
                        )

                        for j in eliminate:
                            self.log(
                                "races",
                                f"Eliminating configuration index={j}: {C[j]}"
                            )

                    # ---------------------------------------------------------
                    # Update consecutive no-elimination counter
                    # ---------------------------------------------------------

                    if l == 1 or k >= e + self.T_new:
                        if eliminate:
                            no_elimination_tests = 0
                        else:
                            no_elimination_tests += 1

                    self.log(
                        "races",
                        f"No-elimination tests: {no_elimination_tests}/{self.T_max}"
                    )

                    # ---------------------------------------------------------
                    # Remove configurations
                    # ---------------------------------------------------------

                    print("number to eliminate", len(eliminate))

                    for j in reversed(eliminate):
                        del C[j]
                        del results[j]

                    self.log(
                        "races",
                        f"Configurations remaining after elimination: {len(C)}"
                    )

                # -------------------------------------------------------------
                # Check whether the race should stop
                # -------------------------------------------------------------

                stop = (
                    len(C) <= self.n_min
                    or no_elimination_tests >= self.T_max
                )

                if stop:
                    if len(C) <= self.n_min:
                        stop_reason = (
                            f"minimum configurations reached "
                            f"({len(C)} <= {self.n_min})"
                        )
                    else:
                        stop_reason = (
                            f"T_max reached "
                            f"({no_elimination_tests} >= {self.T_max})"
                        )

                    self.log(
                        "races",
                        f"Race stopping at instance {k}: {stop_reason}"
                    )
            else:
                stop = None

            # ---------------------------------------------------------
            # Broadcast the updated configuration set
            # ---------------------------------------------------------

            C = self.comm.bcast(
                C if self.rank == 0 else None,
                root=0
            )

            # ---------------------------------------------------------
            # Broadcast whether the race should stop
            # ---------------------------------------------------------

            stop = self.comm.bcast(
                stop,
                root=0
            )

            if stop:
                break

        return C, results, run_counter, evaluated_instances
    
    def create_config_dict(self, c):
        config = deepcopy(self.config_template)
        set_parameters(config, c)
        return config

    def setup_loggers(self, log_directory="logs"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        run_name = (
            f"{self.solver_type}_"
            f"{self.optimizer_type}_"
            f"{self.adapter_type}_"
            f"{timestamp}"
        )

        self.log_directory = Path(log_directory) / run_name

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.loggers = {}

        log_files = {
            "tuning": "tuning.log",
            "evaluations": "evaluations.log",
            "races": "races.log",
            "elites": "elites.log",
            "sampling": "sampling.log",
        }

        for name, filename in log_files.items():
            logger = logging.getLogger(
                f"Elitist IFRace.{name}.{timestamp}"
            )
            logger.setLevel(logging.INFO)
            logger.propagate = False

            handler = logging.FileHandler(
                self.log_directory / filename
            )
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )

            logger.addHandler(handler)
            self.loggers[name] = logger

    def log(self, name, message):
        if self.rank == 0:
            self.loggers[name].info(message)