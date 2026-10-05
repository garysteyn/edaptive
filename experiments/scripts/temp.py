import numpy as np
from edaptive.problems.ela_benchmark_suite import *

problems = [
    Schwefel1,
    Ripple25,
    Exponential,
    NeedleEye,
    Step3,
    GeneralizedGiunta,
    GeneralizedPaviani,
    Brown,
    CosineMixture,
    Mishra07,
    Mishra01,
    GeneralizedPrice2,
    GeneralizedEggCrate,
    Rosenbrock,
    Pinter2,
    Qing,
    BBOB_FID2_IID1,
    BBOB_FID6_IID1,
    BBOB_FID16_IID1,
    BBOB_FID17_IID2,
    DropWave,
    BonyadiMichalewicz,
    Discus,
    Elliptic
]

np.random.seed(1001)

dim = 30
x = np.random.uniform(3, 5, size=[30, dim])

print(len(problems))

for problem in problems:
    p = problem(dimensions=dim)
    print(problem, p(x).mean())