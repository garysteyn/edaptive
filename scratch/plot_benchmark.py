"""Scratch driver: plots a 2-D benchmark function as a 3D surface."""

from edaptive.problems.ela_benchmark_suite import BBOB_FID17_IID2, plot_benchmark_3d


def main():
    plot_benchmark_3d(BBOB_FID17_IID2)


if __name__ == "__main__":
    main()
