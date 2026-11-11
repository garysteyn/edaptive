"""Seeded runs whose outputs are pinned in ``reference.json``.

Each ``run_*`` function returns a JSON-serialisable dict of the quantities to pin.
``test_characterisation.py`` compares them against ``reference.json``, and
``make_reference.py`` regenerates that file.

Refactors may need to change how this module *builds* solvers and tuners (e.g.
renamed keyword arguments). The pinned outputs must not change unless the step
is meant to change behaviour, and the commit says so.
"""

import numpy as np

from edaptive.adapters.beta_eda import BetaEDA
from edaptive.adapters.multi_valued_eda import MultiValuedEDA
from edaptive.adapters.r_umda import RUMDA
from edaptive.optimizers.pso import PSO
from edaptive.problems.ela_benchmark_suite import (
    Brown,
    Discus,
    Exponential,
    GeneralizedEggCrate,
    Qing,
    Rosenbrock,
)
from edaptive.solvers.basic_solver import BasicSolver
from edaptive.solvers.eda_solver import EDASolver
from edaptive.solvers.pso.sg_pso import StabilityGuidedPSO
from edaptive.solvers.pso.tvac_pso import TVACPSO
from edaptive.solvers.uniform_random_solver import UniformRandomSolver
from edaptive.tuning.ifrace import IFRace
from edaptive.utils.experiment_utils import parse_config

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

N_X = 5
N_S = 10
MAX_ITER = 100
SOLVER_SEED = 7
PROBLEMS = {"Rosenbrock": Rosenbrock, "GeneralizedEggCrate": GeneralizedEggCrate}

_ADAPTED = {
    "w": {"type": "continuous", "lower": 0, "upper": 1},
    "c_1": {"type": "continuous", "lower": 0, "upper": 4},
    "c_2": {"type": "continuous", "lower": 0, "upper": 4},
}


def _fixed(name, value, type_):
    return {name: {"tuned": False, "type": type_, "value": value}}


def _eda_config(elite_weighting, update_freq=1, copula="None", elite_signal="fitness"):
    return {
        "identifier": "char_eda",
        "adapter": (
            _fixed("eta", 0.1, "continuous")
            | _fixed("rho_min", 0.02, "continuous")
            | _fixed("n_elites", 5, "integer")
            | _fixed("update_freq", update_freq, "integer")
            | _fixed("copula", copula, "categorical")
            | _fixed("elite_weighting", elite_weighting, "categorical")
            | _fixed("elite_signal", elite_signal, "categorical")
        ),
        "optimizer": {"fixed": _fixed("n_s", N_S, "integer"), "adapted": _ADAPTED},
    }


def _multi_valued_config(
    identifier,
    params,
    update_freq=1,
    elite_signal="fitness",
    n_values=5,
    adapted=_ADAPTED,
):
    adapter = (
        _fixed("n_values", n_values, "integer")
        | _fixed("update_freq", update_freq, "integer")
        | _fixed("elite_signal", elite_signal, "categorical")
    )
    for name, (value, type_) in params.items():
        adapter |= _fixed(name, value, type_)
    return {
        "identifier": identifier,
        "adapter": adapter,
        "optimizer": {"fixed": _fixed("n_s", N_S, "integer"), "adapted": adapted},
    }


_R_UMDA_PARAMS = {"mu": (3, "integer")}

# One CP of each type, each with 5 values.
_MIXED_ADAPTED = {
    "w": {"type": "categorical", "classes": [0.4, 0.55, 0.7298, 0.85, 1.0]},
    "c_1": {"type": "continuous", "lower": 0, "upper": 4},
    "c_2": {"type": "integer", "lower": 0, "upper": 4},
}


_BASIC_CONFIG = {
    "identifier": "char_basic",
    "optimizer": {
        "fixed": (
            _fixed("n_s", N_S, "integer")
            | _fixed("w", 0.7298, "continuous")
            | _fixed("c_1", 1.49618, "continuous")
            | _fixed("c_2", 1.49618, "continuous")
        )
    },
}

_UNIFORM_RANDOM_CONFIG = {
    "identifier": "char_uniform_random",
    "optimizer": {"fixed": _fixed("n_s", N_S, "integer"), "adapted": _ADAPTED},
}

_SG_PSO_CONFIG = {
    "identifier": "char_sg_pso",
    "optimizer": {"fixed": _fixed("n_s", N_S, "integer")},
}

_TVAC_PSO_CONFIG = {
    "identifier": "char_tvac_pso",
    "optimizer": {"fixed": _fixed("n_s", N_S, "integer")},
}

# name -> (solver_type, adapter_type, JSON-style config)
SOLVER_CASES = {
    "basic": (BasicSolver, None, _BASIC_CONFIG),
    "uniform_random": (UniformRandomSolver, None, _UNIFORM_RANDOM_CONFIG),
    "sg_pso": (StabilityGuidedPSO, None, _SG_PSO_CONFIG),
    "tvac_pso": (TVACPSO, None, _TVAC_PSO_CONFIG),
    "eda_uniform_weights": (EDASolver, BetaEDA, _eda_config("None")),
    "eda_rank": (EDASolver, BetaEDA, _eda_config("rank")),
    "eda_log_rank": (EDASolver, BetaEDA, _eda_config("log-rank")),
    "eda_log_rank_freq5": (EDASolver, BetaEDA, _eda_config("log-rank", 5)),
    "eda_log_rank_copula": (
        EDASolver,
        BetaEDA,
        _eda_config("log-rank", copula="gaussian"),
    ),
    "eda_log_rank_freq5_improvement": (
        EDASolver,
        BetaEDA,
        _eda_config("log-rank", 5, elite_signal="improvement"),
    ),
    "r_umda": (EDASolver, RUMDA, _multi_valued_config("char_r_umda", _R_UMDA_PARAMS)),
    "r_umda_freq5_improvement": (
        EDASolver,
        RUMDA,
        _multi_valued_config(
            "char_r_umda", _R_UMDA_PARAMS, update_freq=5, elite_signal="improvement"
        ),
    ),
    "r_umda_mixed": (
        EDASolver,
        RUMDA,
        _multi_valued_config("char_r_umda", _R_UMDA_PARAMS, adapted=_MIXED_ADAPTED),
    ),
}


def build_solver(solver_type, adapter_type, config, problem_type, seed):
    solver_config, _ = parse_config(config)
    return solver_type(
        PSO,
        adapter_type,
        problem=problem_type(dimensions=N_X),
        config=solver_config,
        rnd_seed=seed,
    )


def _floats(values):
    return [float(v) for v in values]


def run_solver(solver_type, adapter_type, config, problem_type, seed):
    solver = build_solver(solver_type, adapter_type, config, problem_type, seed)
    f_final, history = solver.run(max_iterations=MAX_ITER)

    out = {
        "f_final": float(f_final),
        "f_best": _floats(history["f_best"]),
        "prop_feasible": _floats(history["prop_feasible"]),
        "prop_stable": _floats(history["prop_stable"]),
        "X": solver.optimizer.X.tolist(),
        "adaptable_CPs": solver.optimizer.adaptable_CPs.tolist(),
    }
    adapter = getattr(solver, "adapter", None)
    if isinstance(adapter, BetaEDA):
        out["adapter_a"] = _floats(adapter.a)
        out["adapter_b"] = _floats(adapter.b)
        if adapter.copula == "gaussian":
            out["adapter_R"] = np.asarray(adapter.R).tolist()
    if isinstance(adapter, MultiValuedEDA):
        out["adapter_p"] = adapter.p.tolist()
    return out


def run_solver_case(case_name, problem_name, seed=SOLVER_SEED):
    solver_type, adapter_type, config = SOLVER_CASES[case_name]
    return run_solver(solver_type, adapter_type, config, PROBLEMS[problem_name], seed)


def solver_case_ids():
    return [f"{case}/{problem}" for case in SOLVER_CASES for problem in PROBLEMS]


# ---------------------------------------------------------------------------
# Elitist I/F-race
# ---------------------------------------------------------------------------

TUNING_N_X = 2
TUNING_SEED = 0
TUNING_INSTANCES = [Rosenbrock, Brown, Qing, Discus, Exponential, GeneralizedEggCrate]

_BASIC_TUNING_CONFIG = {
    "identifier": "char_tune_basic",
    "optimizer": {
        "fixed": (
            _fixed("n_s", N_S, "integer")
            | {
                "w": {"tuned": True, "type": "continuous", "lower": 0, "upper": 1},
                "c_1": {"tuned": True, "type": "continuous", "lower": 0, "upper": 4},
                "c_2": {"tuned": True, "type": "continuous", "lower": 0, "upper": 4},
            }
        )
    },
}

_EDA_TUNING_CONFIG = {
    "identifier": "char_tune_eda",
    "adapter": {
        "eta": {
            "tuned": True,
            "log_scale_tuning": True,
            "type": "continuous",
            "lower": 1e-4,
            "upper": 0.2,
        },
        "rho_min": {
            "tuned": True,
            "log_scale_tuning": True,
            "type": "continuous",
            "lower": 1e-4,
            "upper": 0.5,
        },
        "n_elites": {"tuned": True, "type": "integer", "lower": 2, "upper": N_S},
        "elite_weighting": {
            "tuned": True,
            "type": "categorical",
            "classes": ["None", "rank", "log-rank"],
        },
    }
    | _fixed("update_freq", 1, "integer")
    | _fixed("copula", "None", "categorical")
    | _fixed("elite_signal", "fitness", "categorical"),
    "optimizer": {"fixed": _fixed("n_s", N_S, "integer"), "adapted": _ADAPTED},
}

# name -> (solver_type, adapter_type, JSON-style config, extra IFRace kwargs)
TUNING_CASES = {
    "basic": (BasicSolver, None, _BASIC_TUNING_CONFIG, {"B": 300}),
    "eda": (EDASolver, BetaEDA, _EDA_TUNING_CONFIG, {"B": 300}),
}
TUNING_ITERATIONS_PER_RUN = 30


def _plain(obj):
    """numpy scalars/arrays → plain Python, recursively (for JSON)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_tuning_case(case_name, results_directory, seed=TUNING_SEED):
    solver_type, adapter_type, config, ifrace_kwargs = TUNING_CASES[case_name]
    config_template, tuning_space = parse_config(config)

    tuner = IFRace(
        I=list(TUNING_INSTANCES),
        n_x=TUNING_N_X,
        n_iter_per_run=TUNING_ITERATIONS_PER_RUN,
        results_directory=str(results_directory),
        **ifrace_kwargs,
    )
    best = tuner.tune(
        solver_type=solver_type,
        optimizer_type=PSO,
        adapter_type=adapter_type,
        config_template=config_template,
        tuning_parameter_space=tuning_space,
        rnd_seed=seed,
    )
    return {
        "best_configuration": _plain(best),
        "race_history": _plain(tuner.race_history),
        "L": int(tuner.L),
    }
