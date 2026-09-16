import numpy as np
from numpy import round, log2
from scipy.stats import friedmanchisquare, rankdata
from copy import deepcopy
from mpi4py import MPI
from scikit_posthocs import posthoc_conover_friedman

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
            n_iter_per_run = 5000,
            friedman_alpha=0.05,
            conover_alpha=0.05
    ):
        self.I = I
        self.n_x = n_x
        self.n_iter_per_run = n_iter_per_run
        self.sampling_model = sampling_model()
        self.L = L
        self.B = B

        self.friedman_alpha = friedman_alpha
        self.conover_alpha = conover_alpha

        self.n_min = n_min
        self.L = L

        self.solver = None
        self.best_configuration = None

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


    def tune(self, solver, config_template, tuning_parameter_space, rnd_seed=0):
        self.solver = solver
        self.config_template = config_template
        self.tuning_parameter_space = tuning_parameter_space

        if self.n_min is None:
            self.n_min = 2 + int(round(log2(len(tuning_parameter_space))))

        if self.L is None:
            self.L = 2 + int(round(log2(len(tuning_parameter_space))))

        self.master_rng = np.random.default_rng(rnd_seed)

        B_used = 0

        for l in range(1, self.L + 1):
            self.l = l
            B_l = (self.B - B_used) / (self.L - l + 1)
            mu_l = 5 + l
            N_l = int(np.floor(B_l / mu_l))
            
            # ---------------------------------------------------------
            # Generate configurations on rank 0
            # ---------------------------------------------------------

            if self.rank == 0:

                print(
                    f"Race {l}: "
                    f"N_l={N_l}, B_l={B_l}"
                )

                if l == 1:
                    self.C = self.sampling_model.initialize(
                        tuning_parameter_space,
                        N_l,
                        self.L,
                        self.master_rng
                    )
                else:
                    self.sampling_model.update(
                        prev_elites=self.prev_elites,
                        prev_weights=self.prev_weights,
                        prev_l = l - 1,
                        N_l=N_l
                    )

                    # TODO: Check whether the following line can ever resolve to a negative value (n_new)
                    n_new = N_l - len(self.prev_elites)
                    new_configurations = self.sampling_model.sample(n_new)
                    self.C = self.prev_elites + new_configurations

            # ---------------------------------------------------------
            # Broadcast configurations to all ranks
            # ---------------------------------------------------------

            self.C = self.comm.bcast(
                self.C if self.rank == 0 else None,
                root=0
            )

            # ---------------------------------------------------------
            # Execute race
            # ---------------------------------------------------------

            self.C, results, run_counter = self.execute_race(self.C, B_l)
            B_used += run_counter

            # ---------------------------------------------------------
            # Rank 0 makes decisions
            # ---------------------------------------------------------

            if self.rank == 0:
                # No further evaluations were possible within the budget
                if run_counter == 0:
                    stop = True
                else:
                    # Determine best configuration from this race
                    self.best_configuration = self.select_best(
                        self.C,
                        results
                    )
                
                    # Determine whether tuning should stop
                    stop = (
                        (l == self.L) or (B_used >= self.B)
                    )

                    if not stop:
                        self.prev_elites, self.prev_weights = (
                            self.select_elites(
                                self.C,
                                results
                            )
                        )

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

            if stop:
                return self.best_configuration
        
        return self.best_configuration

    def select_best(self, configurations, results):
        data = np.asarray(results).T
        mean_ranks = self.calculate_mean_ranks(data)
        best_index = np.argmin(mean_ranks)

        return configurations[best_index]

    def select_elites(self, configurations, results):
        n_survive = len(configurations)

        n_elites = min(n_survive, self.n_min)
        
        data = np.asarray(results).T
        mean_ranks = self.calculate_mean_ranks(data)

        ranking = np.argsort(mean_ranks)

        # Select the best configurations
        elite_indices = ranking[:n_elites]

        elite_configurations = [
            configurations[i]
            for i in elite_indices
        ]

        # Rank-based weights
        ranks = np.arange(1, n_elites + 1)

        weights = (
            n_elites - ranks + 1
        ) / (
            n_elites * (n_elites + 1) / 2
        )

        return elite_configurations, weights

    def calculate_mean_ranks(self, data):
        ranks = np.array([
            rankdata(instance_results)
            for instance_results in data
        ])
        return ranks.mean(axis=0)

    def execute_race(self, C, B_l):

        # Only rank 0 needs the complete result history.
        if self.rank == 0:
            results = [[] for _ in C]
        else:
            results = None

        run_counter = 0

        # TODO: Change to use stratified sampling
        if self.rank == 0:
            problems = self.master_rng.permutation(self.I)
        else:
            problems = None
        # print(problems)

        # Everyone needs to know whether to continue.
        problems = self.comm.bcast(
            problems,
            root=0
        )
        
        for k, I in enumerate(problems, start=1):

            # ---------------------------------------------------------
            # Determine whether another complete instance can be
            # evaluated within the remaining budget.
            # ---------------------------------------------------------

            if self.rank == 0:
                can_evaluate = run_counter + len(C) <= B_l
            else:
                can_evaluate = None

            # Everyone needs to know whether to continue.
            can_evaluate = self.comm.bcast(
                can_evaluate,
                root=0
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

            for i in range(self.rank, len(C), self.size):
                c = C[i]
                config = self.generate_configuration(c)
                self.solver.initialize(I(dimensions=self.n_x), hyper_params=config, rnd_seed=iteration_seed)
                # TODO: make max_iterations a parameter of I/F-race
                result, _ = self.solver.run(max_iterations=self.n_iter_per_run)

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
            run_counter += len(C)

            if self.rank == 0:
                if k >= 2:
                    # Friedman test
                    statistic, p_value = friedmanchisquare(*results)
                    if p_value < self.friedman_alpha:

                        data = np.asarray(results).T
                        mean_ranks = self.calculate_mean_ranks(data)

                        # Best configuration
                        best_index = np.argmin(mean_ranks)

                        p_values = posthoc_conover_friedman(data)
                        # Determine configurations to eliminate
                        eliminate = []

                        for j in range(len(C)):
                            if j == best_index:
                                continue

                            if p_values.iloc[best_index, j] < self.conover_alpha:
                                eliminate.append(j)

                        # Remove configurations
                        print("number to eliminate", len(eliminate))
                        for j in reversed(eliminate):
                            del C[j]
                            del results[j]
                
                # -----------------------------------------------------
                # Check whether the race should stop
                # -----------------------------------------------------

                stop = len(C) <= self.n_min
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

        return C, results, run_counter
    
    def generate_configuration(self, c):
        config = deepcopy(self.config_template)
        set_parameters(config, c)
        return config