"""BBOB_FID17_IID2 diagnostic for Known issue 17 (CLAUDE.md): is the final-gbest shift
there in fresh seed bases?

Run from the repo root:

    nohup taskset -c 0,2,4,6 python scratch/bs_sapso_bbob_seed_bases.py \\
        > results/bs_sapso_equivalence_seed7000000/bbob_seed_bases.log 2>&1 &

BSSAPSO improvement (update_freq 201) vs calypso's vc_fixed200_improve30, seed bases
6-11 million, 30 seeds each per side. Initial and final gbest per run; MWU per seed
base and pooled; Spearman correlation of initial and final gbest. Writes
results/bs_sapso_equivalence_seed7000000/bbob_seed_bases.json.
"""
import json, sys
from multiprocessing import Pool
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, "scratch")
import bs_sapso_bbob_trajectories as inv
from edaptive.utils.seeding import trial_seed

BASES = [6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000, 11_000_000]
NAME = "BBOB_FID17_IID2"

def task(t):
    side, base, s = t
    seed = trial_seed(NAME, inv.N_X, s, base=base)
    traj = inv.bs_calypso(NAME, seed, s) if side == "calypso" else inv.bs_edaptive(NAME, seed)
    return (side, base, s), (traj[0], traj[-1])

if __name__ == "__main__":
    tasks = [(side, b, s) for b in BASES for s in range(30) for side in ("calypso", "edaptive")]
    tasks.sort(key=lambda t: t[0] != "calypso")
    with Pool(4, maxtasksperchild=40) as pool:
        res = dict(pool.imap_unordered(task, tasks))
    out = {}
    print(f"{'seed base':>10} {'init c':>8} {'init e':>8} {'p':>6} {'final c':>8} {'final e':>8} {'p':>6}")
    for b in BASES + ["all"]:
        bs = BASES if b == "all" else [b]
        c = np.array([res[("calypso", x, s)] for x in bs for s in range(30)])
        e = np.array([res[("edaptive", x, s)] for x in bs for s in range(30)])
        row = {k: float(v) for k, v in dict(
            init_c=np.median(c[:, 0]), init_e=np.median(e[:, 0]), p_init=mannwhitneyu(c[:, 0], e[:, 0]).pvalue,
            final_c=np.median(c[:, 1]), final_e=np.median(e[:, 1]), p_final=mannwhitneyu(c[:, 1], e[:, 1]).pvalue).items()}
        out[str(b)] = row
        print(f"{str(b):>10} {row['init_c']:8.3f} {row['init_e']:8.3f} {row['p_init']:6.3f} {row['final_c']:8.3f} {row['final_e']:8.3f} {row['p_final']:6.3f}")
    allr = np.array([res[k] for k in res])
    rho = spearmanr(allr[:, 0], allr[:, 1])
    out["spearman_initial_vs_final_all_runs"] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
    print("Spearman initial vs final gbest (all 360 runs):", round(rho.correlation, 3), "p", rho.pvalue)
    out["per_run"] = {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in res.items()}
    p = "/workspaces/claude-sandbox/edaptive/results/bs_sapso_equivalence_seed7000000/bbob_seed_bases.json"
    open(p, "w").write(json.dumps(out, indent=1) + "\n")
    print("wrote", p)
