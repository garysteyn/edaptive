"""PSO whose CPs are set by a trained Soft Actor-Critic policy (SAC-SAPSO).

von Eschwege & Engelbrecht (2024), "A Soft Actor-Critic Approach to Self-Adaptive
Particle Swarm Optimisation", Mathematics 12:3481. The policy is trained in
calypso (``calypso/scripts/rl/calypsoTrainSAC.py``) and exported to a numpy file
by ``calypso/scripts/rl/exportPolicy.py``; this module needs numpy only.

Policy file (``policy.npz``, written by ``save_policy``):

- ``hidden_kernel_<k>``, ``hidden_bias_<k>`` (``k = 0, 1, ...``): the actor's
  hidden Dense layers (ReLU), in order;
- ``projection_kernel``, ``projection_bias``: the projection layer, whose
  outputs are ``[means, raw stds]`` (3 each);
- ``low``, ``high``: the action bounds (``w``, ``c_1``, ``c_2``);
- ``metadata``: a JSON string with at least ``policy_type`` (``"stochastic"`` or
  ``"greedy"``), ``j``, ``N`` and ``max_iter_cal``.
"""

import json

import numpy as np

from edaptive.core.solver import BaseSolver
from edaptive.optimizers.pso import PSO, stable_particle_proportion
from edaptive.utils.timing import timed

POLICY_TYPES = ("stochastic", "greedy")


def save_policy(path, hidden_layers, projection, low, high, metadata):
    """Write a policy file. ``hidden_layers`` is a list of ``(kernel, bias)``,
    ``projection`` one ``(kernel, bias)``; arrays are stored as float64."""
    arrays = {}
    for k, (kernel, bias) in enumerate(hidden_layers):
        arrays[f"hidden_kernel_{k}"] = np.asarray(kernel, dtype=np.float64)
        arrays[f"hidden_bias_{k}"] = np.asarray(bias, dtype=np.float64)
    arrays["projection_kernel"] = np.asarray(projection[0], dtype=np.float64)
    arrays["projection_bias"] = np.asarray(projection[1], dtype=np.float64)
    arrays["low"] = np.asarray(low, dtype=np.float64)
    arrays["high"] = np.asarray(high, dtype=np.float64)
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


def load_policy(path):
    """Read a policy file: ``{"hidden_layers", "projection", "low", "high",
    "metadata"}`` with float64 arrays."""
    with np.load(path, allow_pickle=False) as file:

        def read(key):
            return file[key].astype(np.float64)

        n_hidden = sum(1 for key in file.files if key.startswith("hidden_kernel_"))
        policy = {
            "hidden_layers": [
                (read(f"hidden_kernel_{k}"), read(f"hidden_bias_{k}"))
                for k in range(n_hidden)
            ],
            "projection": (read("projection_kernel"), read("projection_bias")),
            "low": read("low"),
            "high": read("high"),
            "metadata": json.loads(str(file["metadata"])),
        }
    if policy["metadata"].get("policy_type") not in POLICY_TYPES:
        raise ValueError(
            f"policy_type must be one of {POLICY_TYPES}, "
            f"got {policy['metadata'].get('policy_type')!r}"
        )
    return policy


def policy_distribution(policy, observation):
    """Mean and std (float64, length 3) of the pre-tanh normal for ``observation``.

    tf-agents ``ActorDistributionNetwork`` + ``TanhNormalProjectionNetwork``:
    hidden Dense layers with ReLU, then a linear projection split into
    ``[means, raw stds]``, and ``std = exp(raw std)`` (no clipping).
    """
    h = np.asarray(observation, dtype=np.float64)
    for kernel, bias in policy["hidden_layers"]:
        h = np.maximum(h @ kernel + bias, 0.0)
    kernel, bias = policy["projection"]
    out = h @ kernel + bias
    mean, raw_std = np.split(out, 2, axis=-1)
    return mean, np.exp(raw_std)


def policy_action(policy, observation, rng):
    """The action (float64 ``w, c_1, c_2``) for ``observation``.

    Stochastic policy: one ``rng.standard_normal(3)`` call, ``eps``;
    ``z = mean + std * eps``. Greedy policy: ``z = mean``, no RNG.
    Then ``action = (high + low)/2 + (high - low)/2 * tanh(z)`` (tf-agents'
    ``SquashToSpecNormal``), clipped to ``[low, high]`` (``ActorPolicy`` clips to
    the action spec).
    """
    mean, std = policy_distribution(policy, observation)
    if policy["metadata"]["policy_type"] == "stochastic":
        eps = rng.standard_normal(mean.shape[-1])
        z = mean + std * eps
    else:
        z = mean
    low, high = policy["low"], policy["high"]
    action = (high + low) / 2 + (high - low) / 2 * np.tanh(z)
    return np.clip(action, low, high)


def build_observation(
    t, max_iterations, infeasible_proportion, CPs, V, problem_lb, problem_ub
):
    """calypso's observation after the block whose last step is ``t`` (float64).

    ``[(t + 2)/max_iterations, infeasible_proportion, stable proportion,
    tanh(mean_j |V_ij| / (ub_j - lb_j)) for each particle i]``. The stable
    proportion is Poli's criterion on the shared ``CPs = (w, c_1, c_2)``, so 0 or
    1. A NaN mean velocity becomes ``+inf`` (1 after tanh).
    """
    w, c_1, c_2 = CPs
    velocity = np.mean(np.abs(V / (problem_ub - problem_lb)), axis=1)
    velocity[np.isnan(velocity)] = np.inf
    head = np.array(
        [
            (t + 2) / max_iterations,
            infeasible_proportion,
            stable_particle_proportion(w, c_1, c_2),
        ],
        dtype=np.float64,
    )
    return np.concatenate((head, np.tanh(velocity)))


def reset_observation(n_s, max_iterations):
    """calypso's first observation, before any iteration:
    ``[1/max_iterations, 0, 0.5, 0, ..., 0]``."""
    observation = np.zeros(n_s + 3, dtype=np.float64)
    observation[:3] = (1 / max_iterations, 0.0, 0.5)
    return observation


def infeasible_proportion(X, problem_lb, problem_ub):
    """Proportion of particles outside the bounds in any dimension."""
    return np.mean(np.any((X < problem_lb) | (X > problem_ub), axis=1))


class SACSAPSO(BaseSolver):
    """SAC-SAPSO at test time: a trained SAC policy sets ``w``, ``c_1`` and
    ``c_2``, shared by all particles, every ``nt`` iterations.

    **Settings come from ``config.adapter_params``** (``policy_path``, ``nt``,
    ``max_iterations``, exactly these keys). This solver builds no adapter; the
    config's ``"adapter"`` section is only where its own settings live.
    ``nt``, ``n_s`` and ``max_iterations`` must equal the policy file's ``j``,
    ``N`` and ``max_iter_cal``, and ``run`` must be called with
    ``max_iterations``.

    Replicates calypso's env (``calypsoEnv`` / ``CALYPSO.iter``). The first action
    comes from the reset observation (``reset_observation``) and is used from step
    0. After step ``t`` with ``(t + 1) % nt == 0`` and ``t + 1 < max_iterations``
    (calypso never acts after the last iteration), the policy sees
    ``build_observation``. Its infeasible proportion is that of the positions
    *before* step ``t``'s move (calypso's ``storeMetrics`` reads ``P[i-1]``), and
    the velocities are those after step ``t`` (post-clamping). The new action is
    used from step ``t + 1``.

    Actions are drawn as in ``policy_action``: a stochastic policy (what calypso
    validates and saves, ``agent.policy``) samples, a greedy one takes the mode.

    RNG order (one generator, ``rnd_seed``): for a stochastic policy, the first
    action's ``standard_normal(3)`` (in ``initial_adapted_CPs``, before the PSO is
    built), then the PSO's initialisation; afterwards each decision's
    ``standard_normal(3)`` follows the draws of the step that precedes it.

    Timed: ``_load_policy``, ``_swarm_data``, ``_observation``,
    ``_policy_action`` and ``_set_adapted_CPs``, so the forward pass counts
    toward the runtime.
    """

    fixed_optimizer_type = PSO

    adapted_CP_names = ("w", "c_1", "c_2")
    setting_names = ("policy_path", "nt", "max_iterations")

    def __init__(
        self, optimizer_type=None, adapter_type=None, *, problem, config, rnd_seed=0
    ):
        self.settings = self._validated_settings(config.adapter_params)
        super().__init__(
            optimizer_type,
            adapter_type,
            problem=problem,
            config=config,
            rnd_seed=rnd_seed,
        )
        data = self._swarm_data()
        # Positions before the next move: those of the initial swarm.
        self._infeasible_proportion = infeasible_proportion(
            data["X"], data["problem_lb"], data["problem_ub"]
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
        return dict(adapter_params)

    @property
    def nt(self):
        return self.settings["nt"]

    @property
    def max_iterations(self):
        return self.settings["max_iterations"]

    def initial_adapted_CPs(self):
        self.policy = self._load_policy(self.settings["policy_path"])
        metadata = self.policy["metadata"]
        expected = {"j": self.nt, "N": self.n_s, "max_iter_cal": self.max_iterations}
        for name, value in expected.items():
            if str(metadata[name]) != str(value):
                raise ValueError(
                    f"The policy was trained with {name} = {metadata[name]}, "
                    f"but this run has {value}"
                )
        n_inputs = self.policy["hidden_layers"][0][0].shape[0]
        if n_inputs != self.n_s + 3:
            raise ValueError(
                f"The policy expects {n_inputs} observations, not n_s + 3 = "
                f"{self.n_s + 3}"
            )
        self.action = self._policy_action(
            reset_observation(self.n_s, self.max_iterations)
        )
        return {
            name: np.full(self.n_s, float(value))
            for name, value in zip(self.adapted_CP_names, self.action, strict=True)
        }

    def run(self, max_iterations=5000):
        if max_iterations != self.max_iterations:
            raise ValueError(
                f"{type(self).__name__} was configured for max_iterations = "
                f"{self.max_iterations}, not {max_iterations}"
            )
        return super().run(max_iterations)

    def adapt(self, t):
        data = self._swarm_data()
        before_move = self._infeasible_proportion
        self._infeasible_proportion = infeasible_proportion(
            data["X"], data["problem_lb"], data["problem_ub"]
        )
        if (t + 1) % self.nt != 0 or t + 1 >= self.max_iterations:
            return
        observation = self._observation(t, before_move, data)
        self.action = self._policy_action(observation)
        self._set_adapted_CPs(
            self.adapted_CP_names, np.tile(self.action.astype(float), (self.n_s, 1))
        )

    @timed
    def _load_policy(self, path):
        return load_policy(path)

    @timed
    def _swarm_data(self):
        return self.optimizer.get_required_data(("X", "V", "problem_lb", "problem_ub"))

    @timed
    def _observation(self, t, infeasible_before_move, data):
        return build_observation(
            t,
            self.max_iterations,
            infeasible_before_move,
            self.action,
            data["V"],
            data["problem_lb"],
            data["problem_ub"],
        )

    @timed
    def _policy_action(self, observation):
        return policy_action(self.policy, observation, self.rng)
