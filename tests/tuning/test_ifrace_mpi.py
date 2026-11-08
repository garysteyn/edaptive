"""The MPI-parallel tuner must give exactly the serial (pinned) results.

Runs a characterisation tuning case under ``mpirun -n 2``. Skipped when ``mpirun`` isn't
available.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.characterisation.test_characterisation import REFERENCE, assert_matches

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
@pytest.mark.parametrize("case_name", ["basic", "eda"])
def test_mpi_tuning_matches_serial_reference(case_name, tmp_path):
    output = tmp_path / "result.json"
    command = [
        "mpirun",
        "-n",
        "2",
        sys.executable,
        "-m",
        "tests.tuning.mpi_case",
        case_name,
        str(tmp_path / "results"),
        str(output),
    ]
    env = os.environ | {"OMP_NUM_THREADS": "1"}
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    result = json.loads(output.read_text())
    assert result.pop("n_ranks") == 2
    assert_matches(result, REFERENCE["tuning"][case_name])
