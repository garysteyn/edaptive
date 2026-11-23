"""Train SAC-SAPSO in calypso: the configuration procedure of the SAC-SAPSO baseline.

The authors' original-repo hyperparameters, the shared training set and a budget of
2000 PSO runs (as I/F-race's B). Run from the repo root in the TF venv:

    python experiments/scripts/train_sac.py

The experiment is written to calypso/data/vc_sac_j50/. Before starting, the
wrapper deletes calypso/data and calypso/log, so export a finished run first
(calypso/scripts/rl/exportPolicy.py) and never run two trainings at once.
"""

import os

# One thread per process, so each process's recorded times are its own compute effort
# (inherited by the calypso subprocess; set before numpy is imported; a value from
# the shell wins).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# CPU only, so CPU time covers all compute.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from edaptive.integrations.calypso.sac_pso import CalypsoWrapper
from edaptive.problems.ela_benchmark_suite import TRAINING_SET


def main():
    wrapper = CalypsoWrapper(
        dimensions=30,
        N=30,
        max_iter_cal=5000,
        train_problems=list(TRAINING_SET),
        validation_problems=list(TRAINING_SET),
        j=50,
        i=200000,  # safety cap on SAC collect steps; the budget stops training
        n=256,
        gamma=0.99,
        rsf=1,
        ilr=0.0001,
        dcr=1,
        dcs=100,
        ics=5,
        psi=10,
        ei=10,
        nef=1,
        vc=True,
        seed=123,
        budget_runs=2000,
        eval_seed_offset=1_000_000,
    )
    wrapper.train()


if __name__ == "__main__":
    main()
