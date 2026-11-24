"""``trial_seed`` is reproducible and unrelated across problems, dimensions and
trials."""

import itertools
import os
import subprocess
import sys

from edaptive.utils.seeding import trial_seed

PROBLEMS = ["Rosenbrock", "Qing", "BBOB_FID17_IID2", "CEC_2013_FID3"]
DIMENSIONS = [2, 10, 30]
TRIALS = range(20)


def test_same_across_calls():
    assert trial_seed("Rosenbrock", 30, 7) == trial_seed("Rosenbrock", 30, 7)
    assert trial_seed("Rosenbrock", 30, 7, base=3) == trial_seed(
        "Rosenbrock", 30, 7, base=3
    )


def test_same_across_processes():
    """Also with different ``PYTHONHASHSEED``s, which change ``hash(str)``."""
    code = (
        "from edaptive.utils.seeding import trial_seed; "
        "print(trial_seed('Rosenbrock', 30, 7), trial_seed('Qing', 10, 0, base=5))"
    )
    expected = f"{trial_seed('Rosenbrock', 30, 7)} {trial_seed('Qing', 10, 0, base=5)}"
    for hash_seed in ("0", "1", "12345"):
        env = os.environ | {"PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == expected


def test_differs_across_problems_dimensions_trials_and_bases():
    seeds = [
        trial_seed(p, n_x, t)
        for p, n_x, t in itertools.product(PROBLEMS, DIMENSIONS, TRIALS)
    ]
    assert len(set(seeds)) == len(seeds)
    assert trial_seed("Rosenbrock", 30, 0, base=0) != trial_seed(
        "Rosenbrock", 30, 0, base=1
    )


def test_independent_of_problem_list_order():
    def seeds(problems):
        return {p: [trial_seed(p, 30, t) for t in TRIALS] for p in problems}

    assert seeds(PROBLEMS) == seeds(list(reversed(PROBLEMS)))
    assert seeds(PROBLEMS)["Qing"] == seeds(["Qing"])["Qing"]


def test_range_fits_legacy_numpy_seeding():
    for p, n_x, t in itertools.product(PROBLEMS, DIMENSIONS, TRIALS):
        assert 0 <= trial_seed(p, n_x, t) < 2**32
