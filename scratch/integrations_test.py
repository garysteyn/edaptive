# Smoke test of the calypso integrations: a tiny SAC training run (3 functions,
# one BBOB; 5-D, 200 iterations, budget 18 runs). The real training is
# experiments/scripts/train_sac.py. Deletes calypso/data and calypso/log.
from edaptive.integrations.calypso.bs_pso import BS_PSO  # noqa: F401
from edaptive.integrations.calypso.sac_pso import CalypsoWrapper
from edaptive.problems.ela_benchmark_suite import BBOB_FID2_IID1, Ripple25, Rosenbrock

problems = [Rosenbrock, Ripple25, BBOB_FID2_IID1]

wrapper = CalypsoWrapper(
    dimensions=5,
    N=10,
    max_iter_cal=200,
    train_problems=problems,
    validation_problems=problems,
    j=50,
    i=1000,
    n=32,
    gamma=0.99,
    rsf=1,
    ilr=0.0001,
    dcr=1,
    dcs=100,
    ics=1,
    psi=1,
    ei=1,
    nef=1,
    vc=True,
    seed=123,
    budget_runs=18,
)

wrapper.train()

# wrapper = BS_PSO(
#     dimensions=30,
#     N=30,
#     trials=10,
#     max_iter_cal=5000,
#     problems=problems,
#     b=None,
#     tc="always",
#     sc=["elitist", "20"],
#     sg=None,
#     temp=None,
#     imitate=None,
#     vc=True,
#     seed=123,
# )

# wrapper.run()
