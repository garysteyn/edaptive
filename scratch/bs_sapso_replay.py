"""Exact replay for Known issue 17 (CLAUDE.md): BSSAPSO driven by calypso's own random
draws.

Run from the repo root:

    python scratch/bs_sapso_replay.py

A stand-in generator (CalypsoDraws) hands BSSAPSO calypso's draws in calypso's order
(MT19937 via RandomState, CPs as one rand(N, 1) per CP, initial positions with
calypso's arithmetic). It is swapped in for np.random.default_rng during
construction only; BSSAPSO itself is unchanged. Compares the gbest at every
iteration and the final positions with calypso, for improvement and elitist
(update_freq 201) on Rosenbrock and BBOB_FID17_IID2, 3 seeds each (seed base 7
million). Prints the result; writes nothing.
"""
import contextlib, io, sys
import numpy as np
sys.path.insert(0, "scratch")
import bs_sapso_bbob_trajectories as inv
import edaptive.core.solver as core_solver
from edaptive.solvers.pso.bs_sapso import BSSAPSO
from edaptive.utils.seeding import trial_seed


class CalypsoDraws:
    """np.random.Generator stand-in that returns calypso's draws in calypso's order:
    CPs as one rand(N, 1) per CP, positions as (rand(N, D) - 0.5) * range + mid,
    r_1 and r_2 as rand(N, D)."""

    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)

    def random(self, size):
        n, n_CPs = size
        return np.column_stack([self.rs.rand(n, 1) for _ in range(n_CPs)])

    def uniform(self, low=None, high=None, size=None):
        if low is None:
            return self.rs.rand(*size)
        low, high = np.asarray(low), np.asarray(high)
        return (self.rs.rand(*size) - 0.5) * (high - low) + (high + low) / 2


def replay_edaptive(name, selection, update_freq, seed):
    original = core_solver.np.random.default_rng
    core_solver.np.random.default_rng = lambda s: CalypsoDraws(s)
    try:
        solver = BSSAPSO(problem=inv.problem(name), config=inv.eq.edaptive_config(selection, update_freq, inv.N_S), rnd_seed=seed)
    finally:
        core_solver.np.random.default_rng = original
    _, h = solver.run(inv.T)
    return np.array(h["f_best"]), solver


def calypso_run(name, selection, update_freq, seed):
    from scripts.pso.calypso import CALYPSO
    p = inv.problem(name)
    kw, _ = inv.eq.calypso_kwargs(inv.eq.execute_argv(inv.eq.CALYPSO_SELECTIONS[selection], update_freq - 1, inv.N_X, inv.N_S, inv.T), 0, p, (p.lower, p.upper), name)
    c = CALYPSO(**kw, seed=seed, savePath=False)
    with contextlib.redirect_stdout(io.StringIO()):
        _, f, _ = c.iterate()
    return np.array(list(c.data["globalBests"]) + [f], dtype=float), c


if __name__ == "__main__":
    for name in ("Rosenbrock", "BBOB_FID17_IID2"):
        for selection in ("improvement", "elitist"):
            for s in range(3):
                seed = trial_seed(name, inv.N_X, s, base=7_000_000)
                fc, c = calypso_run(name, selection, 201, seed)
                fe, e = replay_edaptive(name, selection, 201, seed)
                diff = np.flatnonzero(fc != fe)
                first = int(diff[0]) if len(diff) else None
                X_same = np.array_equal(c.P[-1], e.optimizer.X)
                print(f"{name:16s} {selection:11s} seed {s}: final {fc[-1]:.6f} / {fe[-1]:.6f}  "
                      f"gbest identical at all {len(fc)} iterations: {first is None}  first difference: {first}  final X identical: {X_same}", flush=True)
                if first is not None:
                    k = first
                    print(f"    iter {k}: calypso {fc[k]!r} edaptive {fe[k]!r}; iter {k-1}: {fc[k-1]!r} {fe[k-1]!r}")
