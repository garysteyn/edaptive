from abc import ABC, abstractmethod

import numpy as np
from edaptive.core.parameters import ParameterType
from scipy.stats import qmc, truncnorm


class IFRaceSamplingModel(ABC):
    @abstractmethod
    def initialize(self, configuration_space, n_configurations, L, master_rng):
        pass

    @abstractmethod
    def update(self, prev_elites, prev_weights, prev_l, N_new):
        pass

    @abstractmethod
    def sample(self, N_l):
        pass

class IFRaceDefaultSamplingModel(IFRaceSamplingModel):
    def __init__(self):
        self.digits = None
        self.configuration_space = None
        self.prev_elites = None
        self.prev_weights = None
        self._config_counter = 0

        # Maps each configuration's explicit ID to its numerical standard deviations.
        self.numeric_sigma = {}

        # Maps each configuration's explicit ID to its categorical probability distributions.
        self.categorical_probabilities = {}

        self.sampled_elites = []
        self.categorical_update_elite = None
        self.categorical_update_weight = None

    def _next_config_id(self):
        self._config_counter += 1
        return f"config_{self._config_counter}"

    def _initialize_categorical_distributions(self, configuration):
        """Create uniform categorical distributions for a configuration."""
        return {
            param.name: np.full(
                param.n_classes,
                1.0 / param.n_classes,
            )
            for param in self.configuration_space
            if param.type is ParameterType.CATEGORICAL
        }

    def _distance(self, config1, config2):
        """Calculate the normalized distance between two configurations for soft restart."""
        max_dist = 0.0
        threshold = 10.0 ** -self.digits
        
        for param in self.configuration_space:
            val1 = config1.get(param.name)
            val2 = config2.get(param.name)

            if val1 is None and val2 is None:
                continue
            if (val1 is None) != (val2 is None):
                max_dist = max(max_dist, 1.0)
                continue

            if param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                if param.log_scale_tuning:
                    rng = np.log(param.upper) - np.log(param.lower)
                    dist = abs(np.log(val1) - np.log(val2)) / rng
                else:
                    rng = param.upper - param.lower
                    dist = abs(val1 - val2) / rng
                
                if dist > threshold:
                    max_dist = max(max_dist, dist)
            elif param.type == ParameterType.CATEGORICAL:
                if val1 != val2:
                    max_dist = max(max_dist, 1.0)

        return max_dist

    def initialize(
        self,
        configuration_space,
        n_configurations,
        L,
        master_rng,
        digits=4
    ):
        self.configuration_space = configuration_space
        self.L = L
        self.master_rng = master_rng
        self.digits = digits
        self._config_counter = 0

        self.numeric_sigma = {}
        self.categorical_probabilities = {}

        sobol_scrambled = qmc.Sobol(
            d=configuration_space.dimension,
            scramble=True,
            rng=master_rng,
        )

        points = sobol_scrambled.random(n=n_configurations)
        configurations = []

        for point in points:
            configuration = {}

            for i, param in enumerate(configuration_space):
                x = point[i]

                if param.type is ParameterType.CONTINUOUS:
                    if param.log_scale_tuning:
                        value = param.lower * (param.upper / param.lower) ** x
                    else:
                        value = param.lower + x * (param.upper - param.lower)

                elif param.type is ParameterType.INTEGER:
                    if param.log_scale_tuning:
                        value = param.lower * (param.upper / param.lower) ** x
                        value = int(np.round(value))
                    else:
                        n_values = param.upper - param.lower + 1
                        value = param.lower + int(np.floor(x * n_values))

                    value = max(param.lower, min(value, param.upper))

                elif param.type is ParameterType.CATEGORICAL:
                    index = min(
                        int(np.floor(x * param.n_classes)),
                        param.n_classes - 1,
                    )
                    value = param.classes[index]

                else:
                    raise ValueError(
                        f"Unsupported parameter type: {param.type}"
                    )

                configuration[param.name] = value

            config_id = self._next_config_id()
            configuration['_config_id'] = config_id

            configurations.append(configuration)

            self.categorical_probabilities[config_id] = (
                self._initialize_categorical_distributions(configuration)
            )

            # Initialize numerical sampling distributions (range / 2)
            self.numeric_sigma[config_id] = {}
            for param in self.configuration_space:
                if param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                    if param.log_scale_tuning:
                        rng_size = np.log(param.upper) - np.log(param.lower)
                    else:
                        rng_size = param.upper - param.lower
                    self.numeric_sigma[config_id][param.name] = rng_size / 2.0

        return configurations

    def update(
        self,
        prev_elites,
        prev_weights,
        prev_l,
        N_new,
    ):
        self.prev_elites = prev_elites
        self.prev_weights = prev_weights

        self.categorical_update_elite = None
        self.categorical_update_weight = None

        n_parameters = len(self.configuration_space)

        if prev_l > 0 and len(prev_elites) > 0:
            weight = prev_l / self.L
            self.categorical_update_weight = weight
            max_categorical_p = 0.2 ** (1.0 / n_parameters)

            for elite in prev_elites:
                elite_id = elite['_config_id']

                for param in self.configuration_space:
                    if param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                        self.numeric_sigma[elite_id][param.name] *= (
                            (1.0 / N_new) ** (1.0 / n_parameters)
                        )
                    
                    elif param.type is ParameterType.CATEGORICAL:
                        elite_distributions = self.categorical_probabilities[elite_id]
                        probabilities = elite_distributions[param.name]
                        elite_value_index = param.classes.index(elite[param.name])

                        probabilities *= 1.0 - weight
                        probabilities[elite_value_index] += weight
                        
                        probabilities = np.minimum(probabilities, max_categorical_p)
                        probabilities /= probabilities.sum()
                        elite_distributions[param.name] = probabilities

    def sample(self, N_l):
        if self.prev_elites is None or len(self.prev_elites) == 0:
            raise ValueError(
                "Cannot sample configurations without previous elites."
            )

        prev_elites_indices = np.arange(len(self.prev_elites))
        n_parameters = len(self.configuration_space)

        for attempt in range(2):
            configurations = []
            self.sampled_elites = []
            parents = []

            for _ in range(N_l):
                elite_index = self.master_rng.choice(
                    prev_elites_indices,
                    p=self.prev_weights,
                )

                elite = self.prev_elites[elite_index]
                elite_id = elite['_config_id']

                self.sampled_elites.append(elite_index)
                parents.append(elite)

                configuration = {}

                for param in self.configuration_space:
                    if param.type is ParameterType.CATEGORICAL:
                        probabilities = self.categorical_probabilities[elite_id][param.name]
                        idx = self.master_rng.choice(len(param.classes), p=probabilities)
                        value = param.classes[idx]

                    elif param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                        mean_val = elite[param.name]
                        sigma = self.numeric_sigma[elite_id][param.name]

                        if sigma <= 0:
                            raise ValueError(f"Sampling standard deviation must be positive.")

                        if param.log_scale_tuning:
                            mean = np.log(mean_val)
                            lower_bound = np.log(param.lower)
                            upper_bound = np.log(param.upper)
                        else:
                            mean = mean_val
                            lower_bound = param.lower
                            upper_bound = param.upper

                        if param.type == ParameterType.INTEGER and not param.log_scale_tuning:
                            loc = mean + 0.5
                            a = (lower_bound - loc) / sigma
                            b = (upper_bound + 1.0 - loc) / sigma
                            
                            val = truncnorm.rvs(a, b, loc=loc, scale=sigma, random_state=self.master_rng)
                            value = int(np.floor(val))
                        else:
                            loc = mean
                            a = (lower_bound - loc) / sigma
                            b = (upper_bound - loc) / sigma
                            
                            val = truncnorm.rvs(a, b, loc=loc, scale=sigma, random_state=self.master_rng)
                            if param.log_scale_tuning:
                                val = np.exp(val)
                            
                            if param.type == ParameterType.INTEGER:
                                value = int(np.round(val))
                            else:
                                value = val

                        if param.type == ParameterType.INTEGER:
                            value = max(param.lower, min(value, param.upper))

                    else:
                        raise ValueError(f"Unsupported parameter type: {param.type}")

                    configuration[param.name] = value

                configurations.append(configuration)

            if attempt == 0:
                affected_elite_indices = set()
                
                # Pairwise soft-restart check (children vs children)
                for i, child1 in enumerate(configurations):
                    for j in range(i + 1, len(configurations)):
                        if self._distance(child1, configurations[j]) == 0:
                            affected_elite_indices.add(self.sampled_elites[i])
                            affected_elite_indices.add(self.sampled_elites[j])

                    # Pairwise soft-restart check (children vs all previous elites)
                    for j, prev_elite in enumerate(self.prev_elites):
                        if self._distance(child1, prev_elite) == 0:
                            affected_elite_indices.add(self.sampled_elites[i])
                            affected_elite_indices.add(j)

                if affected_elite_indices:
                    print("soft-restart")
                    for idx in sorted(list(affected_elite_indices)):
                        elite_id = self.prev_elites[idx]['_config_id']
                        
                        for param in self.configuration_space:
                            if param.type == ParameterType.CATEGORICAL:
                                probs = self.categorical_probabilities[elite_id][param.name]
                                max_p = np.max(probs)
                                probs = 0.9 * probs + 0.1 * max_p
                                probs /= probs.sum()
                                self.categorical_probabilities[elite_id][param.name] = probs
                                
                            elif param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                                sigma = self.numeric_sigma[elite_id][param.name]
                                factor1 = float(N_l) ** (2.0 / n_parameters)
                                factor2 = (1.0 / float(N_l)) ** (1.0 / n_parameters)
                                
                                if param.log_scale_tuning:
                                    rng = np.log(param.upper) - np.log(param.lower)
                                else:
                                    rng = param.upper - param.lower
                                    
                                new_sigma = min(sigma * factor1, (rng / 2.0) * factor2)
                                self.numeric_sigma[elite_id][param.name] = new_sigma
                    continue 

            for i, configuration in enumerate(configurations):
                parent_elite_id = parents[i]['_config_id']
                child_id = self._next_config_id()
                configuration['_config_id'] = child_id

                self.categorical_probabilities[child_id] = {
                    param.name: self.categorical_probabilities[parent_elite_id][param.name].copy()
                    for param in self.configuration_space
                    if param.type is ParameterType.CATEGORICAL
                }
                
                self.numeric_sigma[child_id] = {
                    param.name: self.numeric_sigma[parent_elite_id][param.name]
                    for param in self.configuration_space
                    if param.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER)
                }

            return configurations
        
        return configurations