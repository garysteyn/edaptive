"""SAC-SAPSO equivalence check: the selected policy in calypso vs edaptive's SACSAPSO.

Run from the repo root in the TF venv, after training and export:

    python scratch/sac_sapso_equivalence.py --exp vc_sac_j50 [--seeds 30] [--tf-seed 0]

For each function (default: 5 training functions, one BBOB) and seed, it runs one
calypso episode (``calypsoEnv`` + the saved TF policy, as ``calypsoEvalSAC.py``
does) and one ``SACSAPSO`` run (``policy.npz``). The saved policy is the export's
copy, ``results/sac_training/<exp>/saved_policy/``, or, if that is missing,
``calypso/data/<exp>/model/policies/validated/step_<step>``. It is loaded once,
and TF's random seed is set once (``--tf-seed``) before any calypso episode.

Every comparison is a two-sided Mann-Whitney U test across seeds on a per-run
quantity, reported with both medians:

- the final gbest;
- per action dimension (``w``, ``c_1``, ``c_2``): the run's mean action, the run's
  standard deviation of actions, and the actions at decisions 1, 25, 50 and 100
  (1-based; decision 1 is the action from the reset observation).

Holm-adjusted p-values over all tests (all functions and quantities) are reported
next to the raw ones. Per action dimension it also reports the decision with the
largest |difference of mean actions| (means over seeds), the raw gap, and the gap
divided by the pooled seed-to-seed standard error at that decision.

The two are not expected to agree bit-for-bit (calypso draws its own RNG
streams, from numpy's global generator). Report the result; do not tune anything
to make them agree.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

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
ACTION_NAMES = ("w", "c_1", "c_2")
DECISIONS = (1, 25, 50, 100)  # 1-based


def run_calypso(problem_type, seed, policy, metadata):
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
    time_step = env.reset()
    actions = []
    while not time_step.is_last():
        action = policy.action(time_step).action
        actions.append(np.asarray(action, dtype=float))
        time_step = env.step(action)
    return float(env.lsbCurr), np.array(actions)


def v_max_fraction(metadata):
    """calypso's ``--vc`` clamps each velocity component to the full range."""
    if not metadata["velocity_clamping"]:
        raise ValueError(
            "The policy was trained without velocity clamping; SACSAPSO has no "
            "equivalent setting to compare against"
        )
    return 1.0


def run_edaptive(problem_type, seed, policy_path, metadata):
    config = SolverConfig(
        fixed_CPs={"n_s": metadata["N"], "V_max_fraction": v_max_fraction(metadata)},
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


def mwu(x_calypso, x_edaptive):
    return {
        "median_calypso": float(np.median(x_calypso)),
        "median_edaptive": float(np.median(x_edaptive)),
        "p": float(mannwhitneyu(x_calypso, x_edaptive).pvalue),
    }


def per_run_tests(f_calypso, f_edaptive, a_calypso, a_edaptive):
    """``a_*``: seeds x decisions x 3. Returns ``{quantity: mwu result}``."""
    tests = {"f_final": mwu(f_calypso, f_edaptive)}
    n_decisions = a_calypso.shape[1]
    for d, name in enumerate(ACTION_NAMES):
        c, e = a_calypso[..., d], a_edaptive[..., d]
        tests[f"{name}.mean"] = mwu(c.mean(axis=1), e.mean(axis=1))
        tests[f"{name}.std"] = mwu(c.std(axis=1), e.std(axis=1))
        for k in DECISIONS:
            if k <= n_decisions:
                tests[f"{name}.decision_{k}"] = mwu(c[:, k - 1], e[:, k - 1])
    return tests


def largest_mean_action_gaps(a_calypso, a_edaptive):
    """Per action dimension, the decision with the largest |gap| between the mean
    actions over seeds, and that gap over the pooled standard error."""
    n_c, n_e = a_calypso.shape[0], a_edaptive.shape[0]
    gap = a_calypso.mean(axis=0) - a_edaptive.mean(axis=0)  # decisions x 3
    pooled_var = (
        (n_c - 1) * a_calypso.var(axis=0, ddof=1)
        + (n_e - 1) * a_edaptive.var(axis=0, ddof=1)
    ) / (n_c + n_e - 2)
    se = np.sqrt(pooled_var * (1 / n_c + 1 / n_e))
    result = {}
    for d, name in enumerate(ACTION_NAMES):
        k = int(np.argmax(np.abs(gap[:, d])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = gap[k, d] / se[k, d]
        result[name] = {
            "decision": k + 1,
            "gap": float(gap[k, d]),
            "pooled_se": float(se[k, d]),
            "gap_over_se": float(ratio),
        }
    return result


def holm(p_values):
    """Holm step-down adjusted p-values (NaNs stay NaN and are not counted)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for i, index in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[index]))
        adjusted[index] = running
    return adjusted


def saved_policy_dir(out_dir, exp, step):
    copy = out_dir / "saved_policy"
    if copy.is_dir():
        return copy
    return CALYPSO / "data" / exp / "model" / "policies" / "validated" / f"step_{step}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=5_000_000)
    parser.add_argument("--tf-seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = REPO / "results" / "sac_training" / args.exp
    policy_npz = (args.policy_npz or out_dir / "policy.npz").resolve()
    metadata = load_policy(policy_npz)["metadata"]
    v_max_fraction(metadata)  # fail before any run
    policy_dir = saved_policy_dir(out_dir, args.exp, metadata["step"])
    print(f"Saved policy {policy_dir} (step {metadata['step']})")

    import tensorflow as tf
    from tf_agents.policies import py_tf_eager_policy

    tf.keras.utils.set_random_seed(args.tf_seed)
    saved_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        str(policy_dir), load_specs_from_pbtxt=True
    )

    os.chdir(CALYPSO)  # calypso's const.py paths
    seeds = [args.seed_base + k for k in range(args.seeds)]
    report = {}
    for problem_type in DEFAULT_FUNCTIONS:
        name = problem_type.__name__
        calypso_runs = [
            run_calypso(problem_type, s, saved_policy, metadata) for s in seeds
        ]
        edaptive_runs = [
            run_edaptive(problem_type, s, policy_npz, metadata) for s in seeds
        ]
        f_calypso = np.array([f for f, _ in calypso_runs])
        f_edaptive = np.array([f for f, _ in edaptive_runs])
        a_calypso = np.stack([a for _, a in calypso_runs])  # seeds x decisions x 3
        a_edaptive = np.stack([a for _, a in edaptive_runs])
        if a_calypso.shape != a_edaptive.shape:
            raise RuntimeError(
                f"{name}: calypso made {a_calypso.shape[1]} decisions, "
                f"SACSAPSO {a_edaptive.shape[1]}"
            )
        report[name] = {
            "n_decisions": a_calypso.shape[1],
            "tests": per_run_tests(f_calypso, f_edaptive, a_calypso, a_edaptive),
            "largest_mean_action_gap": largest_mean_action_gaps(a_calypso, a_edaptive),
        }
        print(f"{name} done")

    keys = [(f, q) for f in report for q in report[f]["tests"]]
    adjusted = holm([report[f]["tests"][q]["p"] for f, q in keys])
    for (f, q), p_holm in zip(keys, adjusted, strict=True):
        report[f]["tests"][q]["p_holm"] = float(p_holm)

    for name, entry in report.items():
        print(f"\n{name} ({entry['n_decisions']} decisions)")
        print(f"  {'quantity':<18}{'median calypso':>16}{'median edaptive':>17}"
              f"{'p':>10}{'p Holm':>10}")
        for quantity, t in entry["tests"].items():
            print(f"  {quantity:<18}{t['median_calypso']:>16.6g}"
                  f"{t['median_edaptive']:>17.6g}{t['p']:>10.4f}{t['p_holm']:>10.4f}")
        for action, g in entry["largest_mean_action_gap"].items():
            print(f"  largest mean-action gap {action}: decision {g['decision']}, "
                  f"gap {g['gap']:+.4f}, gap/SE {g['gap_over_se']:+.2f}")

    n_tests = len(keys)
    with open(out_dir / "equivalence.json", "w") as file:
        json.dump(
            {
                "seeds": seeds,
                "tf_seed": args.tf_seed,
                "saved_policy": str(policy_dir),
                "policy_npz": str(policy_npz),
                "decisions": list(DECISIONS),
                "n_tests": n_tests,
                "functions": report,
            },
            file,
            indent=2,
        )
    print(f"\n{n_tests} tests; wrote {out_dir / 'equivalence.json'}")


if __name__ == "__main__":
    main()
