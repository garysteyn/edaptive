"""Regenerate ``sac_sapso_policy.npz``: a small fixed random SAC policy for the
SACSAPSO tests (``n_s = 10``, ``nt = 10``, 100 iterations).

Run from the repo root: ``python -m tests.fixtures.make_sac_sapso_policy``.
"""

from pathlib import Path

import numpy as np

from edaptive.solvers.pso.sac_sapso import save_policy

POLICY_PATH = Path(__file__).with_name("sac_sapso_policy.npz")
N_S = 10
NT = 10
MAX_ITERATIONS = 100
N_HIDDEN_UNITS = 8


def make_policy(path=POLICY_PATH, policy_type="stochastic"):
    rng = np.random.default_rng(2024)
    sizes = [N_S + 3, N_HIDDEN_UNITS, N_HIDDEN_UNITS]
    hidden_layers = [
        (rng.normal(0, 0.5, (n_in, n_out)), rng.normal(0, 0.1, n_out))
        for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    projection = (rng.normal(0, 0.5, (N_HIDDEN_UNITS, 6)), rng.normal(0, 0.1, 6))
    save_policy(
        path,
        hidden_layers=hidden_layers,
        projection=projection,
        low=[0, 0, 0],
        high=[1, 4, 4],
        metadata={
            "policy_type": policy_type,
            "j": NT,
            "N": N_S,
            "max_iter_cal": MAX_ITERATIONS,
        },
    )


if __name__ == "__main__":
    make_policy()
