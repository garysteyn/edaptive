import numpy as np
from edaptive.problems.ela_benchmark_suite import *
from edaptive.integrations.calypso.sac_pso import CalypsoWrapper

problems = [
    Schwefel1,
    # Ripple25,
    # Exponential,
    # NeedleEye,
    # Step3,
    # GeneralizedGiunta,
    # GeneralizedPaviani,
    # Brown,
    # CosineMixture,
    # Mishra07,
    # Mishra01,
    # GeneralizedPrice2,
    # GeneralizedEggCrate,
    # Rosenbrock,
    # Pinter2,
    # Qing,
    # BBOB_FID2_IID1,
    # BBOB_FID6_IID1,
    # BBOB_FID16_IID1,
    # BBOB_FID17_IID2,
    # DropWave,
    # BonyadiMichalewicz,
    # Discus,
    # Elliptic
]

wrapper = CalypsoWrapper(
    dimensions=30,
    N=30,
    max_iter_cal=5000,
    train_problems=problems,
    validation_problems=problems,
    j=50,
    i=1,
    n=256,
    gamma=0.99,
    rsf=1,
    ilr=0.0001,
    dcr=1,
    dcs=100,
    ics=1,
    psi=10,
    ei=10,
    nef=1,
    vc=True,
    seed=123
)

wrapper.train()