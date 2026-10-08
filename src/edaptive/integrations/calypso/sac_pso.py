from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

class CalypsoWrapper:
    def __init__(
        self,
        dimensions=None,
        N=None,
        max_iter_cal=None,
        train_problems=None,
        validation_problems=None,
        j=None,
        i=None,
        n=None,
        gamma=None,
        rsf=None,
        ilr=None,
        dcr=None,
        dcs=None,
        ics=None,
        psi=None,
        ei=None,
        nef=None,
        vc=None,
        seed=None,
    ):
        self.dimensions = dimensions
        self.N = N
        self.max_iter_cal = max_iter_cal
        self.train_problems = train_problems
        self.validation_problems = validation_problems

        self.j = j
        self.i = i
        self.n = n
        self.gamma = gamma
        self.rsf = rsf

        self.ilr = ilr
        self.dcr = dcr
        self.dcs = dcs

        self.ics = ics
        self.psi = psi
        self.ei = ei
        self.nef = nef

        self.vc = vc

        self.seed = seed

    def train(self):
        calypso_root = Path(__file__).resolve().parents[4] / "calypso"

        train_problems = [
            f"{problem.__module__}.{problem.__qualname__}"
            for problem in self.train_problems
        ]

        eval_problems = [
            f"{problem.__module__}.{problem.__qualname__}"
            for problem in self.validation_problems
        ]

        command = [
            sys.executable,
            "-m",
            "scripts.rl.calypsoTrainSAC",
        ]

        parameters = {
            "dimensions": self.dimensions,
            "N": self.N,
            "max-iter-cal": self.max_iter_cal,
            "j": self.j,
            "i": self.i,
            "n": self.n,
            "gamma": self.gamma,
            "rsf": self.rsf,
            "ilr": self.ilr,
            "dcr": self.dcr,
            "dcs": self.dcs,
            "ics": self.ics,
            "psi": self.psi,
            "ei": self.ei,
            "nef": self.nef,
            "seed": self.seed,
        }
        print(parameters)
        for name, value in parameters.items():
            if value is not None:
                command.extend([f"--{name}", str(value)])

        if train_problems:
            command.extend(["--train-problems", *train_problems])

        if eval_problems:
            command.extend(["--eval-problems", *eval_problems])

        if self.vc:
            command.append("--vc")

        problems = list(dict.fromkeys(
            (self.train_problems or []) + (self.validation_problems or [])
        ))

        func_ranges = pd.DataFrame(
            {
                "low": np.inf,
                "high": -np.inf,
                "delta": np.nan,
            },
            index=[problem.__name__ for problem in problems],
        )

        func_ranges_path = calypso_root / "misc" / "pkl" / "funcRanges.pkl"
        func_ranges.to_pickle(func_ranges_path)

        subprocess.run(
            command,
            cwd=calypso_root,
            check=True,
        )