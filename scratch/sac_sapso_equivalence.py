"""SAC-SAPSO equivalence check: the selected policy in calypso vs edaptive's SACSAPSO.

Run from the repo root in the TF venv, after training and export:

    python scratch/sac_sapso_equivalence.py --exp vc_sac_j50 [--seeds 30]

For each function (default: 5 training functions, one BBOB) and seed, it runs one
calypso episode (``calypsoEnv`` + the saved TF policy, as ``calypsoEvalSAC.py``
does) and one ``SACSAPSO`` run (``policy.npz``), then reports:

- per function, the final gbest of both (median) and a two-sided Mann-Whitney U
  test;
- the action trajectories: per action dimension, the mean over seeds at every
  decision for both, their largest absolute difference, and a KS test on all
  actions pooled.

The two are not expected to agree bit-for-bit (calypso is float32, draws its
own RNG streams and does not evaluate infeasible particles). Report the result;
do not tune anything to make them agree.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

REPO = Path(__file__).resolve().parents[1]
CALYPSO = REPO / "calypso"
sys.path.insert(0, str(CALYPSO))

from edaptive.core.config import SolverConfig  # noqa: E402
from edaptive.problems.ela_benchmark_suite import (  # noqa: E402
    BBOB_FID17_IID2,
    Brown,
    GeneralizedEggCrate,
    Qing,
    Rosenbrock,
)
from edaptive.solvers.pso.sac_sapso import SACSAPSO, load_policy  # noqa: E402

DEFAULT_FUNCTIONS = [Rosenbrock, Qing, GeneralizedEggCrate, Brown, BBOB_FID17_IID2]


def run_calypso(problem_type, seed, policy_dir, metadata):
    from tf_agents.policies import py_tf_eager_policy

    from scripts.rl.calypsoEnv import calypsoEnv

    params = {
        "N": metadata["N"],
        "D": metadata["D"],
        "j": str(metadata["j"]),
        "gamma": 0.99,
        "velocityClamping": metadata["velocity_clamping"],
        "seed": seed,  # episode run 0 uses seed + 0
        "max_iter_cal": metadata["max_iter_cal"],
    }
    env = calypsoEnv(
        envName="envEquivalence",
        exp="equivalence",
        params=params,
        funcs=[problem_type],
        varsPath=tempfile.mkdtemp(),  # the env writes its run counter here
        evalEnv=False,
    )
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_dir, load_specs_from_pbtxt=True
    )
    time_step = env.reset()
    actions = []
    while not time_step.is_last():
        action = policy.action(time_step).action
        actions.append(np.asarray(action, dtype=float))
        time_step = env.step(action)
    return float(env.lsbCurr), np.array(actions)


def run_edaptive(problem_type, seed, policy_path, metadata):
    config = SolverConfig(
        fixed_CPs={"n_s": metadata["N"], "V_max_fraction": 1.0},
        adapter_params={
            "policy_path": str(policy_path),
            "nt": metadata["j"],
            "max_iterations": metadata["max_iter_cal"],
        },
    )
    solver = SACSAPSO(
        problem=problem_type(dimensions=metadata["D"]), config=config, rnd_seed=seed
    )
    actions = [solver.action.astype(float)]
    original = solver._policy_action

    def recording(observation):
        action = original(observation)
        actions.append(action.astype(float))
        return action

    solver._policy_action = recording
    f_final, _ = solver.run(metadata["max_iter_cal"])
    return float(f_final), np.array(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    args = parser.parse_args()

    out_dir = REPO / "results" / "sac_training" / args.exp
    policy_npz = args.policy_npz or out_dir / "policy.npz"
    metadata = load_policy(policy_npz)["metadata"]
    with open(out_dir / "selected_policy.json") as file:
        selected = json.load(file)
    policy_dir = os.path.join(
        CALYPSO, "data", args.exp, "model", "policies", "validated",
        f"step_{metadata['step']}",
    )
    print(f"Saved policy {policy_dir} (selected step {selected['step']})")

    os.chdir(CALYPSO)  # calypso's const.py paths
    seeds = [args.seed_base + k for k in range(args.seeds)]
    report = {}
    for problem_type in DEFAULT_FUNCTIONS:
        name = problem_type.__name__
        calypso_runs = [run_calypso(problem_type, s, policy_dir, metadata) for s in seeds]
        edaptive_runs = [run_edaptive(problem_type, s, policy_npz, metadata) for s in seeds]

        f_calypso = np.array([f for f, _ in calypso_runs])
        f_edaptive = np.array([f for f, _ in edaptive_runs])
        a_calypso = np.stack([a for _, a in calypso_runs])  # seeds x decisions x 3
        a_edaptive = np.stack([a for _, a in edaptive_runs])

        mean_difference = np.abs(a_calypso.mean(axis=0) - a_edaptive.mean(axis=0))
        report[name] = {
            "median_f_calypso": float(np.median(f_calypso)),
            "median_f_edaptive": float(np.median(f_edaptive)),
            "mann_whitney_p": float(mannwhitneyu(f_calypso, f_edaptive).pvalue),
            "n_decisions": [a_calypso.shape[1], a_edaptive.shape[1]],
            "max_mean_action_difference": mean_difference.max(axis=0).tolist(),
            "action_ks": [
                {
                    "statistic": float(r.statistic),
                    "p": float(r.pvalue),
                }
                for r in (
                    ks_2samp(a_calypso[..., d].ravel(), a_edaptive[..., d].ravel())
                    for d in range(3)
                )
            ],
        }
        print(name, json.dumps(report[name], indent=2))

    with open(out_dir / "equivalence.json", "w") as file:
        json.dump({"seeds": seeds, "functions": report}, file, indent=2)
    print(f"Wrote {out_dir / 'equivalence.json'}")


if __name__ == "__main__":
    main()
