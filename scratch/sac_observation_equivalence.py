"""SAC-SAPSO observation check: what the policy sees in calypso vs SACSAPSO.

Run from the repo root in the TF venv, after training and export:

    python scratch/sac_observation_equivalence.py --exp vc_sac_j50 [--seeds 30]
        [--workers 10]

Same runs as ``sac_sapso_equivalence.py`` (functions, seeds from ``trial_seed``,
saved policy, TF seeded once before the calypso episodes, which run in order in
this process; see that script for why TF cannot be seeded per episode), but at
every decision it records the observation and the action on both sides. The
calypso env builds float64 time steps; they are cast to the saved policy's spec
dtype at the policy call only (``policy_time_step``; a float32 policy has no
float64 signature, and the cast goes after the float64 retrain). The SACSAPSO
runs go to a process pool (they use no TF).

Observation components: ``time`` (index 0), ``infeasible`` (1), ``stable`` (2),
``velocity`` (mean over particles of indices 3:). Per run, each is averaged over
decisions 2..n (decision 1 is the reset observation, identical on both sides),
and compared across seeds with a two-sided Mann-Whitney U test. Per-decision
means over seeds are written to the JSON file.

Actions are split into two parts, with the numpy policy (``policy.npz``):

- ``expected``: E[action | observation] under the policy's own noise
  (Gauss-Hermite quadrature per dimension, then clipped), the part of the action
  that the observation determines;
- ``eps``: the standard-normal noise the action used, recovered as
  ``(atanh((a - mid)/half) - mean)/std``.

Per action dimension, the run means of the action, ``expected`` and ``eps`` are
compared with MWU. ``eps`` should be N(0, 1) on both sides. The script also
reports, per side, the correlation across seed index ``s`` of the run-mean ``eps``
between functions. It should be near 0: SACSAPSO draws the noise from the run's
generator at the same stream position on every function, so it would be 1 if the
functions shared seeds.

Holm adjusts over all MWU tests. Writes
``results/sac_training/<exp>/observation_equivalence.json``.
"""

import argparse
import json
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
CALYPSO_ROOT = REPO / "calypso"
sys.path.insert(0, str(CALYPSO_ROOT))
sys.path.insert(0, str(REPO / "scratch"))

from sac_sapso_equivalence import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    TF_NOISE_NOTE,
    function_seeds,
    holm,
    policy_time_step,
    saved_policy_dir,
    v_max_fraction,
)

from edaptive.core.config import SolverConfig  # noqa: E402
from edaptive.solvers.pso.sac_sapso import (  # noqa: E402
    SACSAPSO,
    load_policy,
    policy_distribution,
)

ACTION_NAMES = ("w", "c_1", "c_2")
COMPONENTS = ("time", "infeasible", "stable", "velocity")
PRINT_DECISIONS = (1, 2, 3, 5, 10, 25, 50, 75, 100)  # 1-based
GH_X, GH_W = np.polynomial.hermite.hermgauss(64)


class RecordingSACSAPSO(SACSAPSO):
    def _policy_action(self, observation):
        action = super()._policy_action(observation)
        self.__dict__.setdefault("records", []).append(
            (np.array(observation, dtype=np.float64), np.array(action, dtype=float))
        )
        return action


def run_edaptive(args):
    problem_type, seed, policy_path, metadata = args
    config = SolverConfig(
        fixed_CPs={"n_s": metadata["N"], "V_max_fraction": v_max_fraction(metadata)},
        adapter_params={
            "policy_path": str(policy_path),
            "nt": metadata["j"],
            "max_iterations": metadata["max_iter_cal"],
        },
    )
    solver = RecordingSACSAPSO(
        problem=problem_type(dimensions=metadata["D"]), config=config, rnd_seed=seed
    )
    f_final, _ = solver.run(metadata["max_iter_cal"])
    observations = np.stack([o for o, _ in solver.records])
    actions = np.stack([a for _, a in solver.records])
    return float(f_final), observations, actions


def run_calypso(problem_type, seed, policy, metadata, obs_dtype):
    from scripts.rl.calypsoEnv import calypsoEnv

    params = {
        "N": metadata["N"],
        "D": metadata["D"],
        "j": str(metadata["j"]),
        "gamma": 0.99,
        "velocityClamping": metadata["velocity_clamping"],
        "seed": seed,
        "max_iter_cal": metadata["max_iter_cal"],
    }
    env = calypsoEnv(
        envName="envObservationEquivalence",
        exp="observation_equivalence",
        params=params,
        funcs=[problem_type],
        varsPath=tempfile.mkdtemp(),
        evalEnv=False,
    )
    time_step = env.reset()
    observations, actions = [], []
    while not time_step.is_last():
        observations.append(np.array(time_step.observation, dtype=np.float64))
        action = policy.action(policy_time_step(time_step, obs_dtype)).action
        actions.append(np.array(action, dtype=float))
        time_step = env.step(action)
    return float(env.lsbCurr), np.stack(observations), np.stack(actions)


def components(observations):
    """seeds x decisions x (n_s + 3) -> {component: seeds x decisions}."""
    return {
        "time": observations[..., 0],
        "infeasible": observations[..., 1],
        "stable": observations[..., 2],
        "velocity": observations[..., 3:].mean(axis=-1),
    }


def decompose(policy, observations, actions):
    """Expected action given the observation, and the recovered noise ``eps``.
    Both seeds x decisions x 3."""
    mean, std = policy_distribution(policy, observations)
    low, high = policy["low"], policy["high"]
    mid, half = (high + low) / 2, (high - low) / 2
    z = mean[..., None] + std[..., None] * np.sqrt(2) * GH_X  # ... x 3 x nodes
    squashed = np.clip(mid[:, None] + half[:, None] * np.tanh(z), low[:, None],
                       high[:, None])
    expected = (squashed * GH_W).sum(axis=-1) / np.sqrt(np.pi)
    u = np.clip((actions - mid) / half, -1 + 1e-12, 1 - 1e-12)
    eps = (np.arctanh(u) - mean) / std
    saturated = np.abs((actions - mid) / half) >= 1 - 1e-6
    return expected, eps, saturated


def mwu(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.concatenate([a, b])
    p = 1.0 if np.ptp(both) == 0 else float(mannwhitneyu(a, b).pvalue)
    return {
        "median_calypso": float(np.median(a)),
        "median_edaptive": float(np.median(b)),
        "mean_calypso": float(np.mean(a)),
        "mean_edaptive": float(np.mean(b)),
        "p": p,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    out_dir = REPO / "results" / "sac_training" / args.exp
    policy_npz = (args.policy_npz or out_dir / "policy.npz").resolve()
    policy = load_policy(policy_npz)
    metadata = policy["metadata"]
    v_max_fraction(metadata)
    policy_dir = saved_policy_dir(out_dir, args.exp, metadata["step"])
    seeds_by_function = {
        f.__name__: function_seeds(f, metadata["D"], args.seed_base, args.seeds)
        for f in DEFAULT_FUNCTIONS
    }

    # SACSAPSO runs first, in worker processes (no TF imported yet).
    with Pool(args.workers, maxtasksperchild=1) as pool:
        edaptive = pool.map(
            run_edaptive,
            [
                (f, s, policy_npz, metadata)
                for f in DEFAULT_FUNCTIONS
                for s in seeds_by_function[f.__name__]
            ],
        )

    import tensorflow as tf
    from tf_agents.policies import py_tf_eager_policy

    tf.keras.utils.set_random_seed(0)  # resets the policy's noise stream; see TF_NOISE_NOTE
    saved_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        str(policy_dir), load_specs_from_pbtxt=True
    )
    obs_dtype = saved_policy.time_step_spec.observation.dtype
    print(f"Saved policy {policy_dir} (step {metadata['step']}, {obs_dtype})")

    os.chdir(CALYPSO_ROOT)
    report, per_decision, eps_run_means = {}, {}, {"calypso": {}, "edaptive": {}}
    for k, problem_type in enumerate(DEFAULT_FUNCTIONS):
        name = problem_type.__name__
        calypso = [
            run_calypso(problem_type, s, saved_policy, metadata, obs_dtype)
            for s in seeds_by_function[name]
        ]
        ed = edaptive[k * args.seeds:(k + 1) * args.seeds]
        sides = {"calypso": calypso, "edaptive": ed}
        obs = {s: np.stack([o for _, o, _ in r]) for s, r in sides.items()}
        act = {s: np.stack([a for _, _, a in r]) for s, r in sides.items()}
        if obs["calypso"].shape != obs["edaptive"].shape:
            raise RuntimeError(f"{name}: decision counts differ")
        comp = {s: components(obs[s]) for s in sides}
        dec = {s: decompose(policy, obs[s], act[s]) for s in sides}

        tests = {"f_final": mwu([f for f, _, _ in calypso], [f for f, _, _ in ed])}
        for c in COMPONENTS:
            tests[f"obs.{c}"] = mwu(comp["calypso"][c][:, 1:].mean(axis=1),
                                    comp["edaptive"][c][:, 1:].mean(axis=1))
        for d, a in enumerate(ACTION_NAMES):
            tests[f"{a}.action"] = mwu(act["calypso"][..., d].mean(axis=1),
                                       act["edaptive"][..., d].mean(axis=1))
            tests[f"{a}.expected"] = mwu(dec["calypso"][0][..., d].mean(axis=1),
                                         dec["edaptive"][0][..., d].mean(axis=1))
            tests[f"{a}.eps"] = mwu(dec["calypso"][1][..., d].mean(axis=1),
                                    dec["edaptive"][1][..., d].mean(axis=1))
        for s in sides:
            eps_run_means[s][name] = dec[s][1].mean(axis=1)  # seeds x 3

        report[name] = {
            "n_decisions": int(obs["calypso"].shape[1]),
            "tests": tests,
            "eps_pooled": {
                s: {
                    a: {
                        "mean": float(dec[s][1][..., d].mean()),
                        "std": float(dec[s][1][..., d].std()),
                        "n_saturated": int(dec[s][2][..., d].sum()),
                    }
                    for d, a in enumerate(ACTION_NAMES)
                }
                for s in sides
            },
            "max_abs_obs_diff_decision_1": float(
                np.abs(obs["calypso"][:, 0] - obs["edaptive"][:, 0]).max()
            ),
        }
        per_decision[name] = {
            s: {
                **{c: comp[s][c].mean(axis=0).tolist() for c in COMPONENTS},
                **{f"{a}.action": act[s][..., d].mean(axis=0).tolist()
                   for d, a in enumerate(ACTION_NAMES)},
                **{f"{a}.expected": dec[s][0][..., d].mean(axis=0).tolist()
                   for d, a in enumerate(ACTION_NAMES)},
            }
            for s in sides
        }
        print(f"{name} done", flush=True)

    keys = [(f, q) for f in report for q in report[f]["tests"]]
    for (f, q), p in zip(keys, holm([report[f]["tests"][q]["p"] for f, q in keys]),
                         strict=True):
        report[f]["tests"][q]["p_holm"] = float(p)

    names = [f.__name__ for f in DEFAULT_FUNCTIONS]
    eps_correlation = {}
    for s in ("calypso", "edaptive"):
        eps_correlation[s] = {}
        for d, a in enumerate(ACTION_NAMES):
            m = np.stack([eps_run_means[s][n][:, d] for n in names])  # funcs x seeds
            r = np.corrcoef(m)
            eps_correlation[s][a] = float(r[np.triu_indices(len(names), 1)].mean())

    for name in names:
        entry = report[name]
        print(f"\n{name} ({entry['n_decisions']} decisions; max |obs diff| at "
              f"decision 1: {entry['max_abs_obs_diff_decision_1']:.2e})")
        print(f"  {'quantity':<16}{'mean calypso':>14}{'mean edaptive':>15}"
              f"{'p':>9}{'p Holm':>9}")
        for q, t in entry["tests"].items():
            print(f"  {q:<16}{t['mean_calypso']:>14.6g}{t['mean_edaptive']:>15.6g}"
                  f"{t['p']:>9.4f}{t['p_holm']:>9.4f}")
        for s, e in entry["eps_pooled"].items():
            print(f"  eps pooled {s:<9}" + "  ".join(
                f"{a}: {v['mean']:+.4f} (sd {v['std']:.3f}, sat {v['n_saturated']})"
                for a, v in e.items()))
        print(f"  per-decision means (calypso / edaptive) at {PRINT_DECISIONS}:")
        for c in (*COMPONENTS, "c_1.expected", "c_2.expected"):
            cal = np.array(per_decision[name]["calypso"][c])
            eda = np.array(per_decision[name]["edaptive"][c])
            cells = " ".join(f"{cal[k - 1]:.4g}/{eda[k - 1]:.4g}"
                             for k in PRINT_DECISIONS if k <= len(cal))
            print(f"    {c:<13}{cells}")

    print("\nMean pairwise correlation across seed index of run-mean eps between "
          "functions:")
    for s, v in eps_correlation.items():
        print(f"  {s:<9}" + "  ".join(f"{a}: {r:+.3f}" for a, r in v.items()))

    with open(out_dir / "observation_equivalence.json", "w") as file:
        json.dump(
            {
                "seeds": seeds_by_function,
                "seed_base": args.seed_base,
                "tf_noise": TF_NOISE_NOTE,
                "saved_policy": str(policy_dir),
                "policy_obs_dtype": str(obs_dtype),
                "n_tests": len(keys),
                "functions": report,
                "eps_run_mean_correlation_between_functions": eps_correlation,
                "per_decision_means": per_decision,
            },
            file,
            indent=2,
        )
    print(f"\n{len(keys)} tests; wrote {out_dir / 'observation_equivalence.json'}")


if __name__ == "__main__":
    main()
