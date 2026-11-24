"""SACSAPSO's policy noise under different seed bases.

Run from the repo root:

    python scratch/sac_sapso_seed_bases.py --exp vc_sac_j50 [--seeds 30]
        [--n-bases 10] [--workers 18] [--calypso-json <file>]

It reproduces the diagnosis of the c_1/c_2 shift between calypso and SACSAPSO, so
it intentionally uses the old shared-seed scheme of the original equivalence runs:
seed ``base + s`` on every function, not ``trial_seed``. With it, SACSAPSO draws
the same policy noise on every function for a given seed (the draw counts per step
don't depend on the problem).

calypso's means come from ``--calypso-json`` (relative to
``results/sac_training/<exp>/``), by default
``observation_equivalence_shared_seeds.json``: the
``sac_observation_equivalence.py`` output of the shared-seed runs (seeds
5 000 000 + s), kept under that name.

1. Replays each seed's generator stream (``standard_normal(3)`` for the first
   action, the PSO's ``uniform`` draws, a ``standard_normal(3)`` per decision)
   without running the PSO, and reports the mean noise per action dimension for
   the bases ``5 000 000 + 1 000 000 b``.
2. Runs SACSAPSO on the 5 functions for every base and reports the 30-seed mean
   of the run-mean actions, the mean over all bases minus calypso's mean (from
   ``observation_equivalence.json``), the standard deviation of the base means, and
   the correlation over bases between the noise mean and the mean action.

Writes ``results/sac_training/<exp>/seed_bases.json``.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scratch"))

from sac_observation_equivalence import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    run_edaptive,
)

from edaptive.solvers.pso.sac_sapso import load_policy  # noqa: E402

ACTION_NAMES = ("w", "c_1", "c_2")


def replay_eps(seed, n_s, n_x, max_iterations, nt):
    """The policy noise SACSAPSO draws for ``seed``: decisions x 3."""
    # Hard-codes the order of SACSAPSO's and PSO's random-number draws (the first
    # action's standard_normal(3) before the PSO is built, PSO.initialize's
    # uniform positions, PSO.iterate's r_1 and r_2, then one standard_normal(3)
    # per decision). Update it if either changes.
    rng = np.random.default_rng(seed)
    eps = [rng.standard_normal(3)]
    rng.uniform(size=n_s * n_x)
    for t in range(max_iterations):
        rng.uniform(size=(n_s, n_x))
        rng.uniform(size=(n_s, n_x))
        if (t + 1) % nt == 0 and t + 1 < max_iterations:
            eps.append(rng.standard_normal(3))
    return np.array(eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-bases", type=int, default=10)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument(
        "--calypso-json", default="observation_equivalence_shared_seeds.json"
    )
    args = parser.parse_args()

    out_dir = REPO / "results" / "sac_training" / args.exp
    npz = out_dir / "policy.npz"
    metadata = load_policy(npz)["metadata"]
    shape = (metadata["N"], metadata["D"], metadata["max_iter_cal"], metadata["j"])
    bases = [5_000_000 + 1_000_000 * b for b in range(args.n_bases)]
    names = [f.__name__ for f in DEFAULT_FUNCTIONS]

    eps_means = np.stack(
        [
            np.stack([replay_eps(base + s, *shape) for s in range(args.seeds)]).mean(
                axis=(0, 1)
            )
            for base in bases
        ]
    )  # bases x 3
    n_draws = args.seeds * (metadata["max_iter_cal"] // metadata["j"])
    print(f"Replayed noise mean per base (z over {n_draws} draws per dimension):")
    for base, m in zip(bases, eps_means, strict=True):
        z = m * np.sqrt(n_draws)
        print(f"  base {base}: " + "  ".join(
            f"{a} {v:+.4f} (z {zv:+.2f})" for a, v, zv in zip(ACTION_NAMES, m, z,
                                                              strict=True)))

    jobs = [
        (f, base + s, npz, metadata)
        for base in bases
        for f in DEFAULT_FUNCTIONS
        for s in range(args.seeds)
    ]
    with Pool(args.workers, maxtasksperchild=4) as pool:
        runs = pool.map(run_edaptive, jobs)
    means = (
        np.array([a.mean(axis=0) for _, _, a in runs])
        .reshape(len(bases), len(names), args.seeds, 3)
        .mean(axis=2)
    )  # bases x functions x 3

    calypso_json = out_dir / args.calypso_json
    calypso_file = json.loads(calypso_json.read_text())
    calypso = {
        n: [float(np.mean(calypso_file["per_decision_means"][n]["calypso"][f"{a}.action"]))
            for a in ACTION_NAMES]
        for n in names
    }

    print(f"\ncalypso mean of run-mean actions ({calypso_json.name}):")
    for n in names:
        print(f"  {n:<20}" + "  ".join(
            f"{a} {v:.4f}" for a, v in zip(ACTION_NAMES, calypso[n], strict=True)))
    print("\nSACSAPSO mean of run-mean actions per base (c_1/c_2 per function):")
    for b, base in enumerate(bases):
        print(f"  base {base}: " + "  ".join(
            f"{means[b, i, 1]:.3f}/{means[b, i, 2]:.3f}" for i in range(len(names))))
    print(f"\nSACSAPSO over all {len(bases)} bases minus calypso (sd of base means):")
    pooled = {}
    for i, n in enumerate(names):
        pooled[n] = {
            a: {
                "edaptive_minus_calypso": float(means[:, i, d].mean() - calypso[n][d]),
                "sd_of_base_means": float(means[:, i, d].std(ddof=1)),
            }
            for d, a in enumerate(ACTION_NAMES)
        }
        print(f"  {n:<20}" + "  ".join(
            f"{a} {v['edaptive_minus_calypso']:+.4f} (sd {v['sd_of_base_means']:.4f})"
            for a, v in pooled[n].items()))
    correlation = {
        a: float(np.corrcoef(eps_means[:, d], means[:, :, d].mean(axis=1))[0, 1])
        for d, a in enumerate(ACTION_NAMES)
    }
    print("\nCorrelation over bases of noise mean with mean action: " + "  ".join(
        f"{a} {r:+.3f}" for a, r in correlation.items()))

    out = out_dir / "seed_bases.json"
    with open(out, "w") as file:
        json.dump(
            {
                "bases": bases,
                "n_seeds": args.seeds,
                "seed_scheme": "base + s on every function",
                "functions": names,
                "eps_mean_per_base": eps_means.tolist(),
                "edaptive_mean_action_per_base": means.tolist(),
                "calypso_json": str(calypso_json),
                "calypso_seeds": calypso_file["seeds"],
                "calypso_mean_action": calypso,
                "pooled_vs_calypso": pooled,
                "correlation_eps_vs_action_over_bases": correlation,
            },
            file,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
