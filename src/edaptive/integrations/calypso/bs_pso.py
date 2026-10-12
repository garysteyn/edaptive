from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import shutil

class BS_PSO:
    def __init__(
        self,
        dimensions=None,
        N=None,
        trials=None,
        max_iter_cal=None,
        problems=None,
        b=None,
        tc=None,
        sc=None,
        sg=None,
        temp=None,
        imitate=None,
        vc=None,
        seed=None,
    ):
        self.dimensions = dimensions
        self.N = N
        self.trials=trials
        self.max_iter_cal = max_iter_cal
        self.problems = problems
        self.b = b
        self.tc = tc
        self.sc = sc
        self.sg = sg
        self.temp = temp
        self.imitate = imitate
        self.vc = vc
        self.seed = seed

    def _clear_previous_run(self, calypso_root):
        for directory_name in ("data", "log"):
            directory = calypso_root / directory_name

            if directory.is_dir():
                shutil.rmtree(directory)

    def run (self):
        calypso_root = Path(__file__).resolve().parents[4] / "calypso"

        problems = [
            f"{problem.__module__}.{problem.__qualname__}"
            for problem in self.problems
        ]

        command = [
            sys.executable,
            "-m",
            "scripts.pso.execute",
        ]

        parameters = {
            "dimensions": self.dimensions,
            "N": self.N,
            "runs": self.trials,
            "max-iter-cal": self.max_iter_cal,
            "b": self.b,
            "seed": self.seed,
        }

        for name, value in parameters.items():
            if value is not None:
                command.extend([f"--{name}", str(value)])

        if self.tc:
            command.extend(["--tc", *map(str, self.tc)])

        if self.sc:
            command.extend(["--sc", *map(str, self.sc)])

        if problems:
            command.extend(["--problems", *problems])

        if self.sg:
            command.append("--sg")

        if self.temp:
            command.append("--temp")

        if self.imitate:
            command.append("--imitate")

        if self.vc:
            command.append("--vc")

        problem_list = list(dict.fromkeys(self.problems or []))

        func_ranges = pd.DataFrame(
            {
                "low": np.inf,
                "high": -np.inf,
                "delta": np.nan,
            },
            index=[problem.__name__ for problem in problem_list],
        )

        func_ranges_path = calypso_root / "misc" / "pkl" / "funcRanges.pkl"
        func_ranges.to_pickle(func_ranges_path)

        self._clear_previous_run(calypso_root)

        subprocess.run(
            command,
            cwd=calypso_root,
            check=True,
        )

    # def __init__(
    #     self,
    #     dimensions=None,
    #     N=None,
    #     max_iter_cal=None,
    #     problems=None,
    #     b=None,
    #     tc=None,
    #     sc=None,
    #     sg=None,
    #     temp=None,
    #     imitate=None,
    #     vc=None,
    #     seed=None,
    # ):