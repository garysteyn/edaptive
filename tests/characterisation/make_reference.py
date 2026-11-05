"""Regenerate ``reference.json`` from the current code.

Run from the repo root:

    python -m tests.characterisation.make_reference

Only do this on purpose: when a change is *meant* to alter results (state why in
the commit message), or when the cases in ``cases.py`` change. Refactors that
claim to keep behaviour must pass the tests against the existing file instead.
"""

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tests.characterisation import cases  # noqa: E402

REFERENCE_PATH = Path(__file__).with_name("reference.json")


def build_reference():
    reference = {"solvers": {}, "tuning": {}}

    for case_name in cases.SOLVER_CASES:
        for problem_name in cases.PROBLEMS:
            key = f"{case_name}/{problem_name}"
            print(f"solver {key}")
            reference["solvers"][key] = cases.run_solver_case(case_name, problem_name)

    with tempfile.TemporaryDirectory() as results_directory:
        for case_name in cases.TUNING_CASES:
            print(f"tuning {case_name}")
            reference["tuning"][case_name] = cases.run_tuning_case(
                case_name, Path(results_directory) / case_name
            )

    return reference


def main():
    reference = build_reference()
    REFERENCE_PATH.write_text(json.dumps(reference, indent=1) + "\n")
    print(f"wrote {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
