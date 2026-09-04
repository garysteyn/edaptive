from abc import ABC, abstractmethod
import numpy as np
from edaptive.core.parameters import ParameterType
from scipy.stats import qmc
from scipy.stats import truncnorm

class IFRaceSamplingModel(ABC):
    @abstractmethod
    def initialize(self, configuration_space, n_configurations):
        pass

    @abstractmethod
    def update(self, elite_configurations, weights, iteration):
        pass

    @abstractmethod
    def sample(self, n_configurations):
        pass

class IFRaceDefaultSamplingModel(IFRaceSamplingModel):
    def __init__(self):
        self.configuration_space = None
        self.prev_elites = None
        self.prev_weights = None
        self.sigma = {}
        self.categorical_probabilities = {}

    def initialize(self, configuration_space, n_configurations, L, master_rng):
        self.configuration_space = configuration_space
        self.L = L
        self.master_rng = master_rng

        sobol_scrambled = qmc.Sobol(d=configuration_space.dimension, scramble=True, rng=master_rng)
        points = sobol_scrambled.random(n=n_configurations)
        # Initialize categorical distributions uniformly
        self.categorical_probabilities = {}

        for param in configuration_space:

            if param.type is ParameterType.CATEGORICAL:
                self.categorical_probabilities[param.name] = np.full(
                    param.n_classes,
                    1.0 / param.n_classes,
                )

        configurations = []

        # Transform Sobol points into configurations
        for point in points:

            configuration = {}

            for i, param in enumerate(configuration_space):

                x = point[i]

                if param.type is ParameterType.CONTINUOUS:

                    if param.log_scale_tuning:
                        value = (
                            param.lower
                            * (param.upper / param.lower) ** x
                        )
                    else:
                        value = (
                            param.lower
                            + x * (param.upper - param.lower)
                        )

                elif param.type is ParameterType.INTEGER:

                    n_values = param.upper - param.lower + 1

                    value = (
                        param.lower
                        + int(np.floor(x * n_values))
                    )

                elif param.type is ParameterType.CATEGORICAL:

                    index = min(
                        int(np.floor(x * param.n_classes)),
                        param.n_classes - 1,
                    )

                    value = param.classes[index]

                configuration[param.name] = value

            configurations.append(configuration)

        # for i, param in enumerate(configuration_space):
        #     if param.type is ParameterType.CONTINUOUS:
        #         if param.log_scale_tuning:
        #             points_scrambled[:, i] = param.lower * (param.upper / param.lower)**(points_scrambled[:, i])
        #         else:
        #             points_scrambled[:, i] = param.lower + points_scrambled[:, i] * (param.upper - param.lower)
        #     elif param.type is ParameterType.INTEGER:
        #         N = param.upper - param.lower + 1
        #         points_scrambled[:, i] = param.lower + np.floor(points_scrambled[:, i] * N)
        #     elif param.type is ParameterType.CATEGORICAL:
        #         points_scrambled[:, i] = np.floor(points_scrambled[:, i] * param.n_classes)

        # configurations = []
        # for p in points_scrambled:
        #     configuration = {param.name : p[i] for i, param in enumerate(configuration_space)}
        #     configurations.append(configuration)

        return configurations
    
    def update(
        self,
        prev_elites,
        prev_weights,
        prev_l,
        N_l,
    ):
        self.prev_elites = prev_elites
        self.prev_weights = prev_weights

        n_parameters = len(self.configuration_space)

        # Update numerical sampling distributions
        self.sigma = {}

        for param in self.configuration_space:
            if param.type in (
                ParameterType.CONTINUOUS,
                ParameterType.INTEGER,
            ):
                self.sigma[param.name] = (
                    param.range
                    * (1 / N_l)
                    ** (prev_l / n_parameters)
                )

        # Update categorical distributions
        if prev_l < self.L:

            # Select ONE elite according to its rank-based weight
            elite_index = self.master_rng.choice(
                len(prev_elites),
                p=prev_weights,
            )

            elite = prev_elites[elite_index]

            weight = prev_l / self.L

            for param in self.configuration_space:

                if param.type is ParameterType.CATEGORICAL:
                    probabilities = self.categorical_probabilities[
                        param.name
                    ]

                    elite_value = elite[param.name]

                    elite_value_index = param.classes.index(
                        elite_value
                    )

                    probabilities *= (1.0 - weight)
                    probabilities[elite_value_index] += weight

    def sample(self, N_l):
        configurations = []

        prev_elites_indices = np.arange(len(self.prev_elites))

        for _ in range(N_l):

            # Select an elite according to its rank-based weight
            elite_index = self.master_rng.choice(
                prev_elites_indices,
                p=self.prev_weights,
            )

            elite = self.prev_elites[elite_index]

            configuration = {}

            for parameter in self.configuration_space:

                if parameter.type is ParameterType.CATEGORICAL:
                    # Sample from the categorical probability
                    # distribution P_l(F_i)
                    probabilities = self.categorical_probabilities[
                        parameter.name
                    ]

                    value = self.master_rng.choice(
                        parameter.classes,
                        p=probabilities,
                    )
                elif parameter.type in (
                    ParameterType.CONTINUOUS,
                    ParameterType.INTEGER,
                ):
                    mean = elite[parameter.name]
                    sigma = self.sigma[parameter.name]

                    a = (
                        parameter.lower - mean
                    ) / sigma

                    b = (
                        parameter.upper - mean
                    ) / sigma

                    value = truncnorm.rvs(
                        a,
                        b,
                        loc=mean,
                        scale=sigma,
                        random_state=self.master_rng
                    )

                    if parameter.type is ParameterType.INTEGER:
                        value = int(round(value))

                configuration[parameter.name] = value

            configurations.append(configuration)

        return configurations