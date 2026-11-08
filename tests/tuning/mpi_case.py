"""Run a characterisation tuning case under MPI (used by ``test_ifrace_mpi.py``).

``mpirun -n 2 python -m tests.tuning.mpi_case <case> <results dir> <output json>``.
Every rank checks that it got the same best configuration; rank 0 writes the result.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")

from mpi4py import MPI  # noqa: E402

from tests.characterisation import cases  # noqa: E402


def main():
    case_name, results_directory, output_path = sys.argv[1:4]
    result = cases.run_tuning_case(case_name, results_directory)

    comm = MPI.COMM_WORLD
    all_results = comm.gather(result, root=0)
    if comm.Get_rank() == 0:
        assert all(r == result for r in all_results), "ranks disagree"
        with open(output_path, "w") as f:
            json.dump(result | {"n_ranks": comm.Get_size()}, f)


if __name__ == "__main__":
    main()
