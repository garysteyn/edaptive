"""Run PSO with a multi-valued EDA adapter and plot how its model evolves.

Run from the repo root, e.g.

    python scratch/r_eda_demo.py --adapter RUMDA --problem Rosenbrock --mu 5

Prints the best fitness and the final frequency matrix, and saves a figure with
the f_best curve and a heatmap of each CP's frequencies over time to
results/scratch/.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from edaptive.core.config import SolverConfig  # noqa: E402
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType  # noqa: E402
from edaptive.optimizers.pso import PSO  # noqa: E402
from edaptive.problems import ela_benchmark_suite  # noqa: E402
from edaptive.registry import resolve  # noqa: E402
from edaptive.solvers.eda_solver import EDASolver  # noqa: E402

# Adapter name -> its own parameters (names of command-line arguments).
ADAPTER_PARAMS = {
    "RUMDA": ("mu",),
}

ADAPTED_CPS = {"w": (0.0, 1.0), "c_1": (0.0, 4.0), "c_2": (0.0, 4.0)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", choices=sorted(ADAPTER_PARAMS), default="RUMDA")
    parser.add_argument("--problem", default="Rosenbrock")
    parser.add_argument("--n_x", type=int, default=10)
    parser.add_argument("--n_s", type=int, default=30)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_values", type=int, default=10)
    parser.add_argument("--update_freq", type=int, default=1)
    parser.add_argument(
        "--elite_signal", choices=("fitness", "improvement"), default="fitness"
    )
    parser.add_argument("--mu", type=int, default=5)
    parser.add_argument("--output", default="results/scratch")
    return parser.parse_args()


def main():
    args = parse_args()

    adapted_CP_space = ParameterSpace(
        [
            Parameter(name=name, type=ParameterType.CONTINUOUS, lower=lb, upper=ub)
            for name, (lb, ub) in ADAPTED_CPS.items()
        ]
    )
    adapter_params = {
        "n_values": args.n_values,
        "update_freq": args.update_freq,
        "elite_signal": args.elite_signal,
    } | {name: getattr(args, name) for name in ADAPTER_PARAMS[args.adapter]}

    config = SolverConfig(
        identifier=f"{args.adapter}_demo",
        fixed_CPs={"n_s": args.n_s},
        adapted_CP_space=adapted_CP_space,
        adapter_params=adapter_params,
    )
    problem = getattr(ela_benchmark_suite, args.problem)(dimensions=args.n_x)
    solver = EDASolver(
        PSO,
        resolve("adapter", args.adapter),
        problem=problem,
        config=config,
        rnd_seed=args.seed,
    )
    adapter = solver.adapter

    # The loop of BaseSolver.run, recording the model after every iteration.
    p_history = [adapter.p.copy()]
    for t in range(args.max_iter):
        solver.optimizer.step(t)
        solver.adapt(t)
        p_history.append(adapter.p.copy())
    p_history = np.array(p_history)  # (iterations + 1, n_CPs, n_values)
    f_best = np.array(solver.optimizer.history["f_best"])

    print(f"{args.adapter} on {args.problem} (n_x={args.n_x}, seed={args.seed})")
    print(f"best fitness: {f_best.min():.6g}")
    print(f"borders: [{adapter.lower_border:.4f}, {adapter.upper_border:.4f}]")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    for i, name in enumerate(adapter.CP_names):
        print(f"{name}: values {adapter.values[i]}")
        print(f"{' ' * len(name)}  p      {adapter.p[i]}")

    figure, axes = plt.subplots(
        1 + adapter.n_CPs, 1, figsize=(8, 2.5 * (1 + adapter.n_CPs)), sharex=True
    )
    axes[0].plot(f_best)
    if f_best.min() > 0:
        axes[0].set_yscale("log")
    axes[0].set_ylabel("f_best")
    axes[0].set_title(f"{args.adapter} on {args.problem}")
    for i, name in enumerate(adapter.CP_names):
        image = axes[i + 1].imshow(
            p_history[:, i, :].T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=adapter.upper_border,
            extent=(0, args.max_iter, -0.5, adapter.n_values - 0.5),
        )
        axes[i + 1].set_yticks(range(adapter.n_values))
        axes[i + 1].set_yticklabels(
            [f"{v:.4g}" if isinstance(v, float) else str(v) for v in adapter.values[i]]
        )
        axes[i + 1].set_ylabel(name)
        figure.colorbar(image, ax=axes[i + 1], label="p")
    axes[-1].set_xlabel("iteration")
    figure.tight_layout()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.adapter}_{args.problem}_seed{args.seed}.png"
    figure.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
