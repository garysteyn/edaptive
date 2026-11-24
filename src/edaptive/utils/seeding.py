import zlib

import numpy as np


def trial_seed(problem_name, n_x, trial, base=0):
    """The seed of one experiment run: trial ``trial`` of problem ``problem_name``
    in ``n_x`` dimensions.

    It depends only on these values and ``base``, so it is the same for every
    algorithm and does not depend on the order in which problems are listed.
    Different problems, dimensions and trials get unrelated seeds, so runs on
    different problems are independent. The problem name is hashed with CRC-32,
    which, unlike ``hash``, is the same in every process.

    Returns an int in ``[0, 2**32)``, which numpy's legacy ``np.random.seed`` (used
    by calypso) also accepts.
    """
    entropy = [base, zlib.crc32(problem_name.encode()), n_x, trial]
    return int(np.random.SeedSequence(entropy).generate_state(1, np.uint32)[0])
