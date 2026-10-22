import numpy as np
from numpy import round, log2, floor
from scipy.stats import friedmanchisquare, rankdata
from copy import deepcopy
from mpi4py import MPI
from scikit_posthocs import posthoc_conover_friedman
import matplotlib.pyplot as plt

import logging
from datetime import datetime
from pathlib import Path
import pickle

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
            T_max=2, # maximum number of consecutive statistical tests with no elimination before the race terminates
            digits=4, # influences the leniency of the threshold used to trigger the soft-restart mechanism (threshold = 10^(digits))
            results_directory="tuning_results"
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

        self.digits = digits

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
        self.race_history = []
        self.results_directory = results_directory

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def nm(self, x, d):
        return int(np.ceil(x / d) * d)

    def tune(self, solver_type, optimizer_type, adapter_type, config_template, tuning_parameter_space, rnd_seed=0):
        self.master_rng = np.random.default_rng(rnd_seed)

        self.solver_type = solver_type
        self.optimizer_type = optimizer_type
        self.adapter_type = adapter_type
        self.config_template = config_template
        self.tuning_parameter_space = tuning_parameter_space
        self.race_history = []
 
        N_param = len(tuning_parameter_space)
        
        if self.n_min is None:
            self.n_min = int(floor(2 + log2(N_param)))
            
        if self.L is None:
            self.L = int(floor(2 + log2(N_param)))

        self.setup_loggers()

        self.log("tuning", "=" * 60)
        self.log("tuning", "Elitist I/F-RACE TUNING")
        self.log("tuning", "=" * 60)

        self.log("tuning", f"Solver: {solver_type.__name__}")
        self.log("tuning", f"Optimizer: {optimizer_type.__name__}")
        if adapter_type:
            self.log("tuning", f"Adapter: {adapter_type.__name__}")
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
        l = 0
        
        while B_used < self.B:
            l += 1
            B_l = (self.B - B_used) / (self.L - l + 1)

            N_new = self.calc_num_candidate_configs(l, B_l)

            if N_new <= 0:
                self.log("tuning", f"Race {l} stopping: N_new ({N_new}) <= 0.")
                break

            self.log("tuning", "")
            self.log("tuning", "-" * 60)
            self.log("tuning", f"RACE {l}")
            self.log("tuning", "-" * 60)
            self.log("tuning", f"New configurations generated: {N_new}")
            self.log("tuning", f"Race budget: {B_l}")
            self.log("tuning", f"Budget used before race: {B_used}")

            if l > 1:
                self.log("tuning", f"Elite configurations carried over: {len(self.prev_elites)}")

            C = self.generate_configs(l, N_new)

            C, results, run_counter, instances = self.execute_race(
                l,
                C,
                B_l
            )

            B_used += run_counter

            global_stop = self.finalize_race(B_used, l, C, results, run_counter, instances)

            self.log("tuning", f"Race evaluations: {run_counter}")
            self.log("tuning", f"Cumulative budget used: {B_used}")
            self.log("tuning", f"Configurations remaining: {len(C)}")
            self.log("tuning", f"Instances evaluated: {len(instances)}")

            if global_stop:
                self.log("tuning", f"Tuning completed after race {l}: global budget exhausted.")
                break
                
            if l == self.L and B_used < self.B:
                self.L += 1
                self.log("tuning", f"Budget remains. Extended total iterations (L) to {self.L}.")

        self.log("tuning", "Tuning completed.")
        self.log("tuning", f"Final best configuration: {self.best_configuration}")
        self.log("tuning", f"Total evaluations: {B_used}")

        self.plot_candidate_counts()

        # Save pickle of best config dict
        if self.rank == 0:
            best_config_dict = self.create_config_dict(self.best_configuration)
            config_path = self.results_directory / "best_configuration.pkl"

            with open(config_path, "wb") as f:
                pickle.dump(best_config_dict, f)
            self.log("tuning", f"Saved best configuration to {config_path}")

        return self.best_configuration

    def finalize_race(self, B_used, l, C, results, run_counter, instances):
        if self.rank == 0:
            if len(instances) == 0:
                global_stop = True
            else:
                self.best_configuration = self.select_best(C, results)
                global_stop = B_used >= self.B
                
                if not global_stop:
                    (
                        self.prev_elites,
                        self.prev_weights,
                        self.prev_elite_instances,
                        self.prev_elite_results
                    ) = self.select_elites(C, results, instances)
        else:
            global_stop = None

        global_stop = self.comm.bcast(global_stop, root=0)
        self.best_configuration = self.comm.bcast(self.best_configuration if self.rank == 0 else None, root=0)

        if not global_stop:
            self.prev_elites = self.comm.bcast(self.prev_elites if self.rank == 0 else None, root=0)
            self.prev_weights = self.comm.bcast(self.prev_weights if self.rank == 0 else None, root=0)
            self.prev_elite_instances = self.comm.bcast(self.prev_elite_instances if self.rank == 0 else None, root=0)
            self.prev_elite_results = self.comm.bcast(self.prev_elite_results if self.rank == 0 else None, root=0)

        return global_stop

    def generate_configs(self, l, N_new):
        if self.rank == 0:
            if l == 1:
                C = self.sampling_model.initialize(
                        self.tuning_parameter_space,
                        N_new,
                        self.L,
                        self.master_rng,
                        self.digits
                    )
                self.log("sampling", f"Race {l}: initial configurations={len(C)}")
            else:
                self.sampling_model.update(
                        prev_elites=self.prev_elites,
                        prev_weights=self.prev_weights,
                        prev_l=l - 1,
                        N_new=N_new
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

                for elite_index, elite in enumerate(self.prev_elites, start=1):
                    elite_id = elite.get('_config_id')
                    if elite_id in self.sampling_model.numeric_sigma:
                        numeric_dists = self.sampling_model.numeric_sigma[elite_id]
                        for parameter, sigma in numeric_dists.items():
                            self.log("sampling", f"Race {l}: sigma [elite {elite_index}][{parameter}]={sigma}")

                    distributions = self.sampling_model.categorical_probabilities[elite_id]
                    for parameter, probabilities in distributions.items():
                        self.log("sampling", f"Race {l}: categorical probabilities [elite {elite_index}][{parameter}]={probabilities}")

                N_l = N_new + len(self.prev_elites)

                self.log(
                    "sampling",
                    f"Race {l}: total configurations={N_l}, "
                    f"carried-over elites={len(self.prev_elites)}, "
                    f"new configurations={N_new}"
                )

                new_configurations = self.sampling_model.sample(N_new)

                for i, elite_index in enumerate(self.sampling_model.sampled_elites, start=1):
                    self.log("sampling", f"Race {l}: new configuration {i} sampled from elite {elite_index + 1}")

                for i, configuration in enumerate(new_configurations, start=1):
                    self.log("sampling", f"Race {l}: new configuration {i}: {configuration}")

                C = self.prev_elites + new_configurations

        C = self.comm.bcast(C if self.rank == 0 else None, root=0)
        return C

    def calc_num_candidate_configs(self, l, B_l):
        if l == 1 or self.prev_elites is None:
            N_elite = 0
            e = 0
        else:
            N_elite = len(self.prev_elites)
            e = max([len(hist) for hist in self.prev_elite_instances]) if self.prev_elite_instances else 0

        denominator = max(
                self.mu + self.T_each * min(5, l),
                self.nm(self.T_new + e, self.T_each)
            )

        N_l = int(np.floor((B_l + N_elite * e) / denominator))
        N_l = max(N_l, N_elite)
        N_new = N_l - N_elite
        
        return N_new

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

        ranking = np.argsort(mean_ranks, kind='stable')
        elite_indices = ranking[:n_elites]

        elite_configurations = [configurations[i] for i in elite_indices]

        elite_instances = []
        elite_results = []

        for elite_rank, i in enumerate(elite_indices, start=1):

            if (self.prev_elites is not None and configurations[i] in self.prev_elites):
                previous_index = self.prev_elites.index(configurations[i])
                history = dict(
                    zip(
                        self.prev_elite_instances[previous_index],
                        self.prev_elite_results[previous_index]
                    )
                )
            else:
                history = {}

            for instance_seed_tuple, result in zip(instances, results[i]):
                history[instance_seed_tuple] = result

            elite_instances.append(list(history.keys()))
            elite_results.append(list(history.values()))

            self.log("elites", f"Elite {elite_rank}: history length={len(history)}")
            self.log("elites", f"Elite {elite_rank}: instances={[inst.__name__ for inst, s in history.keys()]}")
            self.log("elites", f"Elite {elite_rank}: results={list(history.values())}")

        ranks = np.arange(1, n_elites + 1)
        weights = (n_elites - ranks + 1) / (n_elites * (n_elites + 1) / 2)

        self.log("elites", f"Selected {n_elites} elite configurations")

        for rank, (index, configuration, weight) in enumerate(zip(elite_indices, elite_configurations, weights), start=1):
            self.log("elites", f"Elite {rank}: configuration index={index}, weight={weight:.6f}, configuration={configuration}")

        return (elite_configurations, weights, elite_instances, elite_results)

    def calculate_mean_ranks(self, data):
        ranks = rankdata(data, axis=1)
        return ranks.mean(axis=0)

    def execute_race(self, l, C, B_l):

        self.log("races", "")
        self.log("races", "=" * 60)
        self.log("races", f"RACE {l}")
        self.log("races", "=" * 60)
        self.log("races", f"Initial configurations: {len(C)}")
        self.log("races", f"Race budget: {B_l}")

        if l > 1:
            self.log("races", f"Previous elites: {len(self.prev_elites)}")
            for i, elite in enumerate(self.prev_elites):
                self.log("races", f"Elite {i + 1}: {elite}")
                self.log("races", f"Elite {i + 1} history length: {len(self.prev_elite_instances[i])}")

        run_counter = 0
        no_elimination_tests = 0
        evaluated_instances = []

        if self.rank == 0:
            # Record initial state of race l before instance evaluations
            self.race_history.append({'race': l, 'instance': 0, 'count': len(C)})

            if l == 1:
                e = 0
                instance_order = list(self.master_rng.permutation(self.I))
                problems = [(inst, self.master_rng.integers(0, 2**32 - 1)) for inst in instance_order]
            else:
                max_idx = np.argmax([len(hist) for hist in self.prev_elite_instances])
                previous_problems = list(self.master_rng.permutation(self.prev_elite_instances[max_idx]))
                e = len(previous_problems)

                all_seen_insts = {inst for hist in self.prev_elite_instances for inst, seed in hist}
                
                instance_order = list(self.master_rng.permutation(self.I))
                new_insts = [inst for inst in instance_order if inst not in all_seen_insts][:self.T_new]
                new_problems = [(inst, self.master_rng.integers(0, 2**32 - 1)) for inst in new_insts]
                
                used_insts = all_seen_insts.union(set(new_insts))
                remaining_insts = [inst for inst in instance_order if inst not in used_insts]
                remaining_problems = [(inst, self.master_rng.integers(0, 2**32 - 1)) for inst in remaining_insts]
                
                problems = new_problems + previous_problems + remaining_problems

                previous_elite_results = [
                    dict(zip(self.prev_elite_instances[elite_index], self.prev_elite_results[elite_index]))
                    for elite_index in range(len(self.prev_elites))
                ]

                self.log("races", f"Maximum elite history (e): {e}")
                self.log("races", f"New instances: {[inst.__name__ for inst in new_insts]}")
                self.log("races", f"Previous elite instances: {[inst.__name__ for inst, s in previous_problems]}")
                self.log("races", f"Remaining instances: {[inst.__name__ for inst in remaining_insts]}")

            if l == 1:
                self.log("races", f"Initial instance order: {[inst.__name__ for inst, s in problems]}")

            results = [[] for _ in C]

        else:
            problems = None
            results = None

        problems = self.comm.bcast(problems, root=0)

        if self.rank == 0:
            if l == 1:
                previous_elite_results = None

        previous_elite_results = self.comm.bcast(previous_elite_results if self.rank == 0 else None, root=0)

        for k, (I_inst, iteration_seed) in enumerate(problems, start=1):

            if self.rank == 0:
                if l > 1 and k > self.T_new:
                    evaluation_cost = 0
                    for c in C:
                        if c not in self.prev_elites:
                            evaluation_cost += 1
                            continue

                        elite_index = self.prev_elites.index(c)
                        if (I_inst, iteration_seed) not in previous_elite_results[elite_index]:
                            evaluation_cost += 1
                else:
                    evaluation_cost = len(C)

                can_evaluate = run_counter + evaluation_cost <= B_l
            else:
                can_evaluate = None

            can_evaluate = self.comm.bcast(can_evaluate, root=0)

            if self.rank == 0:
                self.log(
                    "races",
                    f"Instance {k}: evaluation cost={evaluation_cost}, "
                    f"run_counter={run_counter}, B_l={B_l}, "
                    f"can_evaluate={can_evaluate}"
                )

            if not can_evaluate:
                break

            local_results = []
            local_run_count = 0

            for i in range(self.rank, len(C), self.size):
                c = C[i]

                if l > 1 and c in self.prev_elites and k > self.T_new:
                    elite_index = self.prev_elites.index(c)
                    cached_results = previous_elite_results[elite_index]

                    if (I_inst, iteration_seed) in cached_results:
                        result = cached_results[(I_inst, iteration_seed)]
                        is_cached = True
                    else:
                        config = self.create_config_dict(c)
                        solver = self.solver_type(
                            optimizer_type=self.optimizer_type,
                            adapter_type=self.adapter_type,
                            problem=I_inst(dimensions=self.n_x),
                            hyper_params=config,
                            rnd_seed=iteration_seed
                        )

                        result, _ = solver.run(max_iterations=self.n_iter_per_run)
                        local_run_count += 1
                        is_cached = False
                else:
                    config = self.create_config_dict(c)
                    solver = self.solver_type(
                        optimizer_type=self.optimizer_type,
                        adapter_type=self.adapter_type,
                        problem=I_inst(dimensions=self.n_x),
                        hyper_params=config,
                        rnd_seed=iteration_seed
                    )

                    result, _ = solver.run(max_iterations=self.n_iter_per_run)
                    local_run_count += 1
                    is_cached = False

                local_results.append((i, result, is_cached))

                print(f"[Rank {self.rank}] Instance {k}, configuration {c}, result={result}")

            gathered_results = self.comm.gather(local_results, root=0)

            if self.rank == 0:
                for rank_results in gathered_results:
                    for i, result, is_cached in rank_results:
                        results[i].append(result)
                        c = C[i]
                        if is_cached:
                            self.log("evaluations", f"Race {l}, instance {k}, configuration {c}: cached result reused = {result}")
                        else:
                            self.log("evaluations", f"Race {l}, instance {k}, configuration {c}: evaluated, result={result}, seed={iteration_seed}")

            run_counter += self.comm.allreduce(local_run_count, op=MPI.SUM)

            if self.rank == 0:
                evaluated_instances.append((I_inst, iteration_seed))

                self.log("races", f"Instance {k} completed: evaluations used={run_counter}")

                if k >= self.T_first and (k - self.T_first) % self.T_each == 0:
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
                        best_index = np.argmin(mean_ranks)

                        self.log(
                            "races",
                            f"Best configuration: index={best_index}, "
                            f"configuration={C[best_index]}, "
                            f"mean rank={mean_ranks[best_index]:.6f}"
                        )

                        p_values = posthoc_conover_friedman(data)

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

                        self.log("races", f"Configurations marked for elimination: {len(eliminate)}")
                        for j in eliminate:
                            self.log("races", f"Eliminating configuration index={j}: {C[j]}")

                    if l == 1 or k >= e + self.T_new:
                        if eliminate:
                            no_elimination_tests = 0
                        else:
                            no_elimination_tests += 1

                    self.log("races", f"No-elimination tests: {no_elimination_tests}/{self.T_max}")
                    print("number to eliminate", len(eliminate))

                    for j in reversed(eliminate):
                        del C[j]
                        del results[j]

                    self.log("races", f"Configurations remaining after elimination: {len(C)}")

                # Record remaining candidates after instance evaluation & eliminations
                self.race_history.append({'race': l, 'instance': k, 'count': len(C)})

                stop = (len(C) <= self.n_min or no_elimination_tests >= self.T_max)

                if stop:
                    if len(C) <= self.n_min:
                        stop_reason = f"minimum configurations reached ({len(C)} <= {self.n_min})"
                    else:
                        stop_reason = f"T_max reached ({no_elimination_tests} >= {self.T_max})"
                    
                    self.log("races", f"Race stopping at instance {k}: {stop_reason}")
            else:
                stop = None

            C = self.comm.bcast(C if self.rank == 0 else None, root=0)
            stop = self.comm.bcast(stop, root=0)

            if stop:
                break

        return C, results, run_counter, evaluated_instances

    def plot_candidate_counts(self):
        if self.rank != 0 or not self.race_history:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        x_indices = np.arange(len(self.race_history))
        counts = [entry['count'] for entry in self.race_history]

        labels = [
            f"R{entry['race']}-I{entry['instance']}" if entry['instance'] > 0 else f"R{entry['race']}-Start"
            for entry in self.race_history
        ]

        ax.plot(
            x_indices,
            counts,
            marker='o',
            linewidth=2,
            markersize=6,
            color='#1f77b4',
            label='Surviving Configurations'
        )

        current_race = None
        for idx, entry in enumerate(self.race_history):
            if current_race is not None and entry['race'] != current_race:
                ax.axvline(x=idx - 0.5, color='gray', linestyle='--', alpha=0.6)
            current_race = entry['race']

        ax.axhline(y=self.n_min, color='red', linestyle=':', label=f'Min. Candidates Threshold ($N_{{min}}={self.n_min}$)')

        ax.set_xticks(x_indices)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel("Race & Instance Step (Race-Instance)", fontsize=11)
        ax.set_ylabel("Number of Candidates", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')

        plt.tight_layout()

        output_file = self.results_directory / "candidate_trajectory.png"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)

    def create_config_dict(self, c):
        config = deepcopy(self.config_template)
        updates = {k: v for k, v in c.items() if k != '_config_id'}
        set_parameters(config, updates)
        return config

    def setup_loggers(self):
        if self.rank == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if self.adapter_type:
                run_name = (
                    f"{self.solver_type.__name__}_"
                    f"{self.optimizer_type.__name__}_"
                    f"{self.adapter_type.__name__}"
                )
            else:
                run_name = (
                    f"{self.solver_type.__name__}_"
                    f"{self.optimizer_type.__name__}"
                )


            # Cast the base string to a Path object before using the '/' operator
            self.results_directory = Path(self.results_directory) / run_name
            self.log_directory = self.results_directory / "logs"

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
        else:
            self.loggers = {}
            self.log_directory = None
            self.results_directory = None

    def log(self, name, message):
        if self.rank == 0 and name in self.loggers:
            self.loggers[name].info(message)