# CLAUDE.md — edaptive

Context for Claude Code sessions working on this repository.

## Project purpose

`edaptive` is PhD research code (author: Gary Steyn) for experimenting with **self-adaptive metaheuristics**.
The central research idea is to use **Estimation of Distribution Algorithms (EDAs) as generic, optimizer-agnostic
mechanisms for adapting the control parameters (CPs) of other metaheuristics** (currently PSO), and to benchmark them
against existing (self-)adaptive methods.

Results feed into a thesis/papers, so **reproducibility, correct seeding, and faithful experimental protocol matter
more than speed or code elegance**. Don't change algorithm behaviour silently: if a change alters
numerical results (RNG consumption order, update rules, defaults), say so explicitly.

## Maintaining this file

This file is the main source of project context for future sessions, so keep it accurate:

- If a change you make leaves anything here inaccurate, update the relevant section **as part of the same task**,
  not as a follow-up. That includes the repository layout, class interfaces or constructor signatures, the
  `SolverConfig` or JSON config schema, run or setup commands, output paths, conventions, and the calypso/COCO integration.
- When you fix an item listed under "Known issues", remove it. When you find a new issue but don't fix it, add it there.
- "Roadmap" holds planned improvements. Tick sub-tasks off as they're done. Once an item is finished, record the facts
  it established in the relevant sections and remove the item. New improvement ideas go into the Roadmap.
- Record facts about the code as it stands, not a history of changes (git keeps that). Keep it concise.
- When you finish a task, tell the user whether you updated this file and which sections changed.

## Repository layout

```
edaptive/
├── pyproject.toml            # package "edaptive" (setuptools, src layout), Python >= 3.10; `dev` extra; ruff config
├── requirements.txt          # pinned deps: edaptive + calypso (TensorFlow 2.9 / tf-agents 0.13 ⇒ Python 3.10, numpy<2)
├── src/edaptive/
│   ├── const.py              # max_float = np.finfo(float).max
│   ├── core/                 # abstract base classes + parameter-space types
│   │   ├── adapter.py        # BaseAdapter, EDAAdapter
│   │   ├── optimizer.py      # BaseOptimizer
│   │   ├── solver.py         # BaseSolver: shared construction + run loop
│   │   ├── config.py         # SolverConfig (frozen dataclass)
│   │   └── parameters.py     # ParameterType, Parameter, ParameterSpace
│   ├── adapters/             # generic (optimizer-agnostic) adapters
│   │   ├── beta_eda.py       # BetaEDA (beta marginals, optional Gaussian copula) + commented-out legacy variants
│   │   └── uniform_random.py # UniformRandomAdapter (EDAAdapter; uniform resampling baseline)
│   ├── optimizers/
│   │   └── pso.py            # PSO (gbest, per-particle CPs), swarm_diversity, stable_particle_proportion
│   ├── solvers/              # wrappers that run optimizer (+ adapter)
│   │   ├── basic_solver.py   # BasicSolver: static CPs, no adapter
│   │   ├── eda_solver.py     # EDASolver: optimizer + any EDAAdapter
│   │   ├── uniform_random_solver.py # UniformRandomSolver: optimizer + UniformRandomAdapter
│   │   └── pso/
│   │       ├── sg_pso.py     # StabilityGuidedPSO: PSO-specific adaptation implemented as a solver
│   │       └── tvac_pso.py   # empty placeholder
│   ├── problems/
│   │   ├── benchmark_problem.py   # BenchmarkProblem ABC
│   │   ├── ela_benchmark_suite.py # ~24 hand-written functions + 4 BBOB wrappers (the tuning/training set)
│   │   ├── bbob.py                # BBOBProblem (COCO cocoex)
│   │   ├── cec.py                 # CECProblem (opfunu)
│   │   ├── custom_test_set.py     # evaluation set: BBOB_FIDx_IIDy and CEC_yyyy_FIDx classes
│   │   └── misc_problems.py       # empty
│   ├── tuning/
│   │   └── ifrace.py               # IFRace (elitist iterated F-race, MPI-parallel) + IFRaceSamplingModel
│   ├── integrations/calypso/       # subprocess wrappers around the external calypso repo
│   │   ├── bs_pso.py               # BS_PSO   → calypso `scripts.pso.execute` (belief-space SA-PSO)
│   │   └── sac_pso.py              # CalypsoWrapper.train() → calypso `scripts.rl.calypsoTrainSAC` (SAC SA-PSO)
│   ├── registry.py                 # CLI names → solver/optimizer/adapter classes (`resolve(kind, name)`)
│   └── utils/
│       ├── experiment_utils.py     # parse_config: JSON config → (SolverConfig, tuning ParameterSpace)
│       └── timing.py               # @timed (wall + CPU time per method), core_time(timings)
├── experiments/
│   ├── configs/*.json        # algorithm configs (see "Config schema")
│   └── scripts/
│       ├── tune.py           # run I/F-race for a solver/optimizer/adapter + config
│       └── execute.py        # run final experiments (uses tuned config if present), MPI-parallel
├── scratch/                  # ad-hoc driver scripts (kept in git, excluded from ruff, not tests)
├── tests/                    # pytest (`testpaths` set in pyproject.toml); see "Tests"
│   ├── conftest.py           # forces the Agg matplotlib backend
│   ├── core/                 # unit tests: adaptable-CP mapping, SolverConfig/parse_config
│   ├── solvers/              # unit tests: subset / reordered "adapted" spaces, construction, timings
│   ├── utils/                # unit tests: @timed, core_time
│   ├── tuning/               # unit tests of IFRace pieces; mpirun -n 2 vs serial reference
│   └── characterisation/     # R8a: pinned seeded results of the current code
│       ├── cases.py          # case definitions (solver/tuner builds + what gets pinned)
│       ├── reference.json    # the pinned outputs
│       ├── make_reference.py # regenerates reference.json
│       └── test_characterisation.py
├── scripts/                  # environment setup + patches for third-party code
│   ├── setup_coco.sh                       # clone COCO into .coco/, pin commit, apply patches, build, pip install cocoex
│   ├── bbob_dimensions.patch               # BBOB suite dims → {2,3,5,10,20,30,40,50,60}
│   ├── bbob_support_high_dimensions.patch  # heap-allocate rotation buffers (removes fixed-size array asserts)
│   ├── update_bbob_dimensions.sh           # regenerate bbob_dimensions.patch for a new dim list
│   ├── setup_calypso.sh                    # clone calypso-clean into ./calypso, apply integration patch, pip install, patch tf_agents
│   ├── calypso-integration.patch           # makes calypso accept edaptive problem classes / CLI args / seeds
│   └── patch_tf_agents.py                  # np.bool → np.bool_ in tf_agents/typing/types.py
├── calypso/   (gitignored; external clone, patched)
├── .coco/     (gitignored; external clone, patched + built)
└── venv/      (gitignored; machine-specific)
```

Gitignored and generated: `results/`, `data/`, `calypso/`, `.coco/`, `venv/`, `.vscode/`.

## Core architecture

Three roles, connected by keyword-argument dictionaries:

### Optimizer (`core/optimizer.py: BaseOptimizer`)
- `__init__(**kwargs)` → sets `_timings`, `history = {"f_best": []}`, calls `initialize(**kwargs)`, then `init_history()`.
- Abstract: `initialize(**kwargs)`, `init_history()`, `step(t)`.
- **Adaptable CPs** are the per-individual control parameters that may change during a run (PSO: `w`, `c_1`,
  `c_2`). Other CPs, such as `n_s`, are fixed when the optimizer is built. Whether a solver actually adapts them is
  set by its config (`"fixed"` vs `"adapted"`).
- Class attribute `adaptable_CP_names` (column order) and read-only property `adaptable_CPs`, an
  `(n_s, len(adaptable_CP_names))` array with one row per individual.
- Columns are addressed **by name**: `column_indices(names)` (raises `ValueError` for unknown names) and
  `set_adaptable_CPs(names, values)`, which writes only the named columns into a fresh copy of the array. Solvers use
  this, so an `"adapted"` space may be any subset of the adaptable CPs, in any order.
- `get_required_data(names)` returns `{name: getattr(self, name)}`. This is how adapters pull state from the optimizer
  generically, so attribute names on optimizers form an implicit interface (`problem_lb`, `problem_ub`, `t`, `X`,
  `f_X`, `adaptable_CPs`, `adaptable_CP_names` for BetaEDA).

### PSO (`optimizers/pso.py`)
- `adaptable_CP_names = ("w", "c_1", "c_2")`. `initialize(problem, rng, n_s, w, c_1, c_2)`: each of `w`, `c_1`,
  `c_2` may be a scalar or a length-`n_s` array; each is broadcast to `n_s` and stacked into `_adaptable_CPs`.
  PSO's own code indexes the columns by position.
- Notation follows Engelbrecht: `X` positions, `V` velocities, `Y`/`f_Y` personal bests, `Y_hat`/`f_Y_hat`
  global best, `n_s` swarm size, `n_x` dimension, `r_1`, `r_2` uniform coefficients.
- `iterate()` order: **evaluate X → update pbest/gbest (feasible particles only) → update V → X += V**.
  So after `step(t)`, `self.f_X` holds the fitness of the *pre-move* positions while `self.X` holds the *post-move* positions.
- Boundary handling: none (no clamping). Infeasible particles are evaluated but can't become pbest/gbest.
  Inf/NaN in V and X are replaced with ±`max_float`.
- History: `f_best` (gbest per iteration), `prop_feasible`, `prop_stable` (Poli stability criterion
  `c1 + c2 < 24(1 - w²)/(7 - 5w)`); `velocities` and `diversity` are currently only recorded at init.
  `stable_particle_proportion` raises if any `w ∉ [-1, 1]`.
- `@timed` on `initialize` and `iterate` (fitness evaluation is part of `iterate`). `update_history` (performance
  measures) is not timed.

### Adapter (`core/adapter.py`)
- `BaseAdapter.__init__(CP_space, **kwargs)`: `CP_space` is a `ParameterSpace` of the **adapted** CPs.
  Sets `CP_space`, `CP_lbs`, `CP_ubs`, `n_CPs` (= number of adapted CPs), `CP_names` (in `CP_space` order), then
  `initialize(**kwargs)`.
- `select_own_CPs(adaptable_CPs, adaptable_CP_names)` picks this adapter's columns (in `CP_space` order) out of an
  optimizer's `adaptable_CPs`. Solvers write the adapter's samples back with
  `optimizer.set_adaptable_CPs(adapter.CP_names, samples)`.
- Generic contract (`BaseAdapter`, which assumes nothing about how the adapter updates, so e.g. scheduled or
  deterministic adapters fit): `required_data()` names the optimizer attributes it reads; `check_update` (called with those
  attributes as keyword arguments) holds the criteria for whether an update occurs in this iteration and returns the
  input for `update` if so, otherwise `None`; `update` (called with that input) changes its state.
  Defaults: `[]`, `check_update(self)` → `None`, `update(self)` no-op; subclasses that update override both with
  the arguments they need.
- `EDAAdapter` adds abstract `sample(N)`. `EDASolver` after each step:
  `data = optimizer.get_required_data(adapter.required_data())` → `elite_CPs = adapter.check_update(**data)` → if
  not `None`: `update(elite_CPs)`, then `sample(n_s)`. `BetaEDA` overrides all of them; `UniformRandomAdapter`
  only `sample`.
- **Adapters are meant to be optimizer-agnostic.** Adaptation schemes tailored to one optimizer
  (e.g. stability-guided PSO) are implemented directly as **solvers** instead.

### BetaEDA (`adapters/beta_eda.py`)
`initialize(eta, rho_min, n_elites, update_freq, copula, elite_weighting, rng)`
- Model: independent **Beta(a_j, b_j) marginals** on CPs scaled to [0,1]; optional **Gaussian copula**
  (`copula == "gaussian"`) with correlation matrix `R`. Any other value (e.g. `None` / `"None"`) means independent marginals.
- Init: `a = b = 1` (uniform), `R = I`.
- `check_update`: only every `update_freq` iterations (`(t+1) % update_freq == 0`); filters to particles whose
  `X` is inside problem bounds; needs ≥ 2 feasible; selects its own columns with `select_own_CPs`;
  elites = CPs of the `n_elites` best `f_X`.
- `update`: scale elites to [0,1], clip to [1e-6, 1-1e-6]; weighted mean/variance (weights: `None` = uniform with
  ddof=1 variance, `"rank"` = linear, `"log-rank"` = log(n+1) - log(rank)); method-of-moments concentration
  `kappa = mu(1-mu)/var - 1`, clipped to `[eps, kappa_max]` with `kappa_max = 1/rho_min - 1`
  (`rho = 1/(kappa+1) = var/(mu(1-mu))` is the normalised variance, so `rho_min` is a floor on it).
  Then `a, b ← (1-eta)·old + eta·(mu·kappa, (1-mu)·kappa)` (exponential smoothing with learning rate `eta`).
  Copula: transform elites through the *updated* beta CDFs → `norm.ppf` → `corrcoef` → EMA into `R` with the same `eta`.
- `sample(N)`: copula path = Cholesky of `R + 1e-10·I`, correlated normals (`self.rng.standard_normal`) →
  `norm.cdf` → `beta.ppf`; otherwise `beta.rvs(..., random_state=self.rng)`. Output is rescaled to `[CP_lbs, CP_ubs]`.
- Older separate classes (`BetaMarginalsEDA`, rank/log-rank variants, `BetaMarginalsGaussianCopulaEDA`) remain
  commented out at the bottom of the file; `BetaEDA` replaces them.

### Solvers (`core/solver.py: BaseSolver`)
- Every solver has the same constructor (used by IFRace, execute.py and the tests):
  `Solver(optimizer_type, adapter_type, problem=<instance>, config=<SolverConfig>, rnd_seed=<int>)`.
- Type validation via class attributes: `fixed_optimizer_type` / `fixed_adapter_type` (the solver always uses that
  type; callers may pass `None` or the same type) and `uses_adapter`. Mismatches raise `ValueError`:
  `BasicSolver` takes no adapter, `EDASolver` needs one, `UniformRandomSolver` fixes `UniformRandomAdapter`,
  `StabilityGuidedPSO` fixes `PSO` and takes no adapter.
- Construction (and RNG consumption) order: **one** `rng = np.random.default_rng(rnd_seed)` shared by adapter,
  solver and optimizer → `_build_adapter()` (`CP_space=config.adapted_CP_space`, `**config.adapter_params`, `rng`)
  → `initial_adapted_CPs()` (hook; default: `adapter.sample(n_s)` by name, `{}` without an adapter) →
  `_build_optimizer()` (`problem`, `rng`, `**(config.fixed_CPs | initial_adapted_CPs)`). `n_s` is read from
  `config.fixed_CPs["n_s"]`.
- `run(max_iterations=5000)`: `for t: optimizer.step(t); self.adapt(t)`, then stores `self.timings()` in
  `history["timings"]` and returns `(min(history["f_best"]), history)`. `adapt(t)` is a no-op hook by default.
- Solvers exchange data with the optimizer through the timed helpers `_get_required_data()` and
  `_set_adapted_CPs(names, values)`, so this interaction lands in the solver's own timings.
- `BasicSolver`: no overrides; all CPs come from `fixed_CPs` (which must include w, c_1, c_2).
- `EDASolver.adapt`: `check_update` → if elites: `update`, resample the adapted CPs of **all** `n_s` particles
  (`set_adaptable_CPs`). CPs not in `adapted_CP_space` come from `fixed_CPs` and never change.
- `UniformRandomSolver.adapt`: resamples the adapted CPs uniformly every iteration.
- `StabilityGuidedPSO`: class attributes `adapted_CP_names = ("w", "c_1", "c_2")`, `n_adapted_CPs` and
  bounds `adapted_CP_lbs/ubs` = `[0,1]×[0,4]×[0,4]`; overrides `initial_adapted_CPs` and `adapt` to resample them
  uniformly with rejection sampling (`sample_adapted_CPs`, timed) until every particle satisfies Poli's stability
  criterion. `is_stable` unpacks the sampled columns in `adapted_CP_names` order.

### Timing (`utils/timing.py`)
- `@timed` accumulates `self._timings[method] = {"wall": s, "cpu": s}` per call (`perf_counter` and the
  process's `process_time`). `BaseOptimizer`, `BaseAdapter` and `BaseSolver` each own a `_timings` dict. Only
  leaf methods are timed, so nothing is counted twice. Each timed call adds about 0.5 µs of clock overhead.
- `history["timings"]` = `{"optimizer": {...}, "adapter": {...}, "solver": {...}}` (adapter `{}` without one).
  Timed: optimizer `initialize`/`iterate`; adapter `initialize`/`check_update`/`update`/`sample`
  (where implemented); solver `_get_required_data`/`_set_adapted_CPs`, plus `sample_adapted_CPs` in
  `StabilityGuidedPSO`. Measured cost of the interaction helpers: about 3.5 µs per iteration, roughly 2% of
  core time (`n_s=n_x=30`, BetaEDA).
- `core_time(timings)` → `{"wall": total, "cpu": total}` over all components.
- Runs execute in parallel (MPI ranks) and per-run times are summed afterwards, so each process must be
  single-threaded: `execute.py` and `tune.py` set `OMP_NUM_THREADS=1` by default (a shell value wins). Use no more
  ranks than physical cores, or wall times include waiting; CPU time is robust to that.

### SolverConfig (`core/config.py`)
Frozen dataclass `SolverConfig(identifier, fixed_CPs, adapted_CP_space, adapter_params)`:
- `fixed_CPs`: optimizer kwargs fixed for a run (`n_s`; `w`/`c_1`/`c_2` when not adapted). `adapted_CP_space`: a
  `ParameterSpace` of the CPs the solver adapts (or `None`). `adapter_params`: e.g. `eta`, `rho_min`, `n_elites`,
  `update_freq`, `copula`, `elite_weighting`. A CP in both `fixed_CPs` and `adapted_CP_space` raises `ValueError`.
- Tuned entries hold `None` until `with_values({name: value})` returns a filled copy (names are looked up in
  `fixed_CPs` and `adapter_params`; unknown or ambiguous names raise `ValueError`).
- `to_dict()` / `from_dict()` convert to and from the plain dict stored in `best_configuration.pkl`:
```python
{
  "identifier": "name",
  "adapter":   {"eta": 0.01, "rho_min": 0.02, "n_elites": 10, "update_freq": 1,
                "copula": None, "elite_weighting": "log-rank"},   # omitted when there are no adapter params
  "optimizer": {
      "fixed":   {"n_s": 30},                 # or w/c_1/c_2 too for BasicSolver
      "adapted": ParameterSpace([...]),       # omitted when nothing is adapted
  },
}
```

### Parameters (`core/parameters.py`)
`Parameter` (frozen dataclass: `name`, `type` ∈ {continuous, integer, categorical}, `lower`, `upper`, `classes`,
`log_scale_tuning`) with validation; `ParameterSpace(parameters)` with `dimension`, `lower_bounds`,
`upper_bounds`, `get(name)`, iteration. The same types describe both **adapted CP spaces** and **tuning spaces**.
Adapters currently assume all adapted CPs are continuous.

## Problems

- `BenchmarkProblem(dimensions)`: subclasses set `self.lower`, `self.upper` (arrays) and implement `_evaluate(x)`.
  `__call__` accepts **batched input `(N, D)` → `(N,)`** (and 1-D), suppresses FP warnings, and maps
  NaN/±inf to ±`max_float`. New problems must be vectorised along the last axis (`axis=-1`).
- Problems are passed around **as classes** and instantiated as `ProblemClass(dimensions=n_x)`.
  Seeds are supplied separately via `rnd_seed`.
- `ela_benchmark_suite.py`: hand-coded functions (Schwefel1, Ripple25, Exponential, NeedleEye, Step3,
  GeneralizedGiunta, GeneralizedPaviani, Brown, CosineMixture(_OG), Mishra01/07, GeneralizedPrice2,
  GeneralizedEggCrate, Rosenbrock, Pinter2(_OG), Qing, DropWave, BonyadiMichalewicz, Discus, Elliptic) plus
  `BBOB_FID{2,6,16,17}_IID*`, with optional `global_optimum` metadata. This set is used for **tuning / SAC training**
  (presumably chosen for ELA-feature coverage). `plot_benchmark_3d` is a plotting helper.
- `custom_test_set.py`: the **evaluation** set (BBOB FID/IID combos + CEC 2013/2015/2017 functions via opfunu).
- `BBOBProblem` uses `cocoex.Suite("bbob", "", "")` and loops over rows for batched input (slow but correct).
  It works only for dimensions included in the patched suite: **2, 3, 5, 10, 20, 30, 40, 50, 60**.
  Changing that requires `scripts/update_bbob_dimensions.sh` + rerunning `setup_coco.sh`.

## Tuning: elitist I/F-race (`tuning/ifrace.py`)

Reference: López-Ibáñez et al. 2016, "The irace package", Operations Research Perspectives 3:43–58 (a local copy
may be at `scratch/irace.pdf`, untracked). Docstrings cite its sections/equations.

`IFRace(I, n_x, B, L=None, n_min=None, mu=None, n_iter_per_run=5000, friedman_alpha=0.05, conover_alpha=0.05,
T_first=5, T_each=1, T_new=1, T_max=2, digits=4, results_directory="results/tuning_results")`
→ `.tune(solver_type, optimizer_type, adapter_type, config_template, tuning_parameter_space, rnd_seed=0)` returns the
best configuration on every rank. `L` and `n_min` default to `floor(2 + log2(N_param))` and `mu` defaults to
`T_first`. These defaults are recomputed on each `tune()` call, and every call starts from a fresh state.

- An "instance" is a `(ProblemClass, seed)` pair. A configuration is a dict of tuned values plus a `'_config_id'`
  key (`CONFIG_ID`), which indexes its sampling state. Its cost on an instance is the best fitness from
  `solver.run(n_iter_per_run)`, and `B` counts solver runs.
- How the methods map onto the paper:
  - `_iterated_racing` (Alg. 1): race budget `B_l = (B-B_used)/(L-l+1)`. It stops when the budget is used up or
    `N_new <= 0`, and extends `L` while budget remains.
  - `_n_new_configurations`: `N_new = N_l − N_elite`, with `N_l` from eq. 5. `e` is the longest elite history.
  - `_sample_configurations`: race 1 calls `sample_initial`. Later races call `update`, then `sample`, then
    prepend the elites.
  - `_instance_sequence` (§3.4): `T_new` unseen instances, then the shuffled history of the longest-history elite,
    then the remaining unseen instances. Each new instance gets a fresh seed.
  - `_race`: after the `T_new` new instances, elites reuse their cached results on `(instance, seed)` pairs they
    have already seen; `_split_cached` handles this. The race stops when:
    - the next instance doesn't fit in `B_l`;
    - `≤ n_min` configurations survive;
    - or `T_max` consecutive tests eliminate nothing. These tests are counted only once `k ≥ e + T_new`.
  - `_test_and_eliminate`: a Friedman test at `T_first`, then every `T_each` instances. If it is significant,
    Conover post-hoc tests compare each configuration with the best (lowest mean rank). Elite `j` is protected
    until `k ≥ e_j + T_new`.
  - `_select_elites`: the `min(N_surv, n_min)` survivors with the best mean rank become elites. Each keeps its
    merged `{(instance, seed): result}` history, and eq. 1 gives the weights.
- `IFRaceSamplingModel(space, n_races, rng, digits)`, created by `IFRace`. Each configuration carries its own
  sigmas and categorical probabilities, and children inherit their parent's.
  - `sample_initial`: an Owen-scrambled Sobol' sequence. Sigma starts at half the range (log range for log-scale
    parameters), and categorical probabilities start uniform.
  - `update`: eq. 2 sigma shrink, eqs. 3–4 categorical update, and the eq. 6 cap.
  - `sample`: eq. 1 parent choice, then a truncated normal. Integers use `+0.5` and a floor; log-scale values are
    sampled in log space. If two candidates, or a candidate and an elite, are at distance 0 (§3.3), it soft-restarts
    the elites involved and resamples, at most once.
- Read-only state:
  - `IFRace`: `L`, `n_min`, `elites`, `elite_weights`, `best_configuration`, `race_history`, `sampling_model`,
    `tuning_times`.
  - Model: `sigmas(id)`, `categorical_probabilities(id)` (copies), `parent_indices`, `soft_restarted_elites`,
    `categorical_update_weight`.
- **MPI.** Rank 0 runs the whole algorithm and owns `rng`. `_evaluate(jobs)` is the only code that communicates:
  - It broadcasts the `(config, instance, seed)` jobs and runs `jobs[rank::size]` on each rank.
  - It gathers `(result, history["timings"])` back into job order.
  - The other ranks sit in `_worker_loop` until `_stop_workers` sends `None`. This happens in a `finally` block,
    so an exception on rank 0 doesn't deadlock the workers.
  - Cached elite results never go to the workers.
- **Tuning time** (`tuning_times`), wall and CPU per race and in total, each as `solver`, `tuner` and `total`:
  - `solver` = Σ `core_time` over the runs executed in the race, summed over all ranks (cached elite results add 0).
  - `tuner` = rank 0's elapsed race time minus its time inside `_evaluate` (solver runs and MPI) and inside
    logging. That leaves sampling, tests, elimination, elite selection and bookkeeping. `self._log` is wrapped in
    `_TimedCalls`, which times every logger call, message formatting included.
  - `total` = `solver + tuner`. Lines go to `tuning.log`; everything to `tuning_times.json`. Serial and
    `mpirun -n 2` give the same `solver` sums (per-process times are summed, not elapsed).
- **Logging** lives entirely in `_TuningLogger`, with one method per event and nothing on ranks other than 0.
  - Each run gets uniquely named loggers, and their handlers are closed at the end.
  - Per-run solver results go to the module logger at DEBUG level.
  - `plot_race_history` draws the plot on a bare `matplotlib.figure.Figure` (no pyplot), so it works whatever
    the backend. TkAgg segfaults here: Pillow 9.2.0's `_imagingtk` vs the uv Python's Tk 9.0. The pickle is
    saved before the plot is drawn.
- `config_template` is a `SolverConfig`. `create_config(c)` = `config_template.with_values(c without '_config_id')`,
  so tuned parameter names must be unique across `fixed_CPs` and `adapter_params`.
- Outputs (relative to CWD): `results/tuning_results/<identifier>/logs/{tuning,evaluations,races,elites,sampling}.log`,
  `candidate_trajectory.png`, `best_configuration.pkl` (`SolverConfig.to_dict()` of the best configuration) and
  `tuning_times.json`.

**Fidelity to the paper.** Checked against §3.2–3.4.
- These match the paper:
  - `N_iter`, `B_j` and eq. 5, including `nm`;
  - `N_new = N_j − N_elite`, and stopping when `N_j ≤ N_elite`;
  - eqs. 1–4 and 6;
  - integer `+0.5`/floor sampling;
  - the soft-restart distance, the categorical reset, the sigma reset, and the at-most-once rule;
  - per-elite `e_j + T_new` protection, and counting `T_max` only after `T_new + e`;
  - `mu = T_first`;
  - the test schedule and the race stopping rules;
  - elites chosen by mean rank.
- Deliberate deviations:
  - Sobol' initialisation instead of uniform sampling.
  - Log-scale parameters: init, sigma, sampling, distance and soft restart all work in log space. This is sound,
    apart from deviation 7 below.
- Deviations found and not yet fixed (R4 phase 2):
  1. **The model keeps the initial `L` as `N_iter`.** Once `L` is extended past it, the eq. 3 weight `(j−1)/N_iter`
     reaches 1 and the elite's categorical distribution becomes one-hot. The eq. 6 cap can't undo this, because the
     other entries are 0. If the weight goes above 1 after a soft restart has put mass back, the probabilities turn
     negative and `choice` raises `ValueError` (reproduced with `digits=0`, `B=300` on the `eda` tuning case).
  2. **Instances never repeat with new seeds.** Once every class in `I` has been seen, races have no new instances
     and can't go beyond `e`. irace draws fresh `(instance, seed)` pairs from a cycling stream. With the 6-instance
     test cases this already happens in race 2.
  3. **Only the longest-history elite's instances are replayed.** An instance seen only by other elites is never
     used again.
  4. **The soft restart also resets an elite that a child matched**, not only the children's parents (§3.3 resets
     only the parents).
  5. **Real values aren't rounded to `digits` decimals** as in irace (Appendix A). The proposal is to keep this as a
     deliberate deviation, because log-scale values near `1e-4` would collapse. This still needs the author's
     decision.
  6. **`L` is extended whenever `B_used < B`.** The paper extends only when the budget is "enough for a new race".
     The `N_new <= 0` stop makes this equivalent in effect, so it is only documented.
  7. **Log-scale integers use `round`** (Sobol' init and sampling), which reintroduces the endpoint bias that §3.2
     footnote 2 removes. The fix is to sample `[log lb, log(ub+1))` and take the floor.
  8. **`Parameter` accepts `log_scale_tuning` with `lower <= 0`.** It should be rejected.

## Config schema (`experiments/configs/*.json`) and `utils/experiment_utils.parse_config`

```json
{
  "identifier": "my_alg",
  "adapter": {                                   // omit for adapter-free solvers
    "eta":     {"tuned": true,  "type": "continuous", "lower": 1e-4, "upper": 0.2, "log_scale_tuning": true},
    "n_elites": {"tuned": true, "type": "integer",    "lower": 2,    "upper": 30},
    "copula":  {"tuned": false, "type": "categorical", "value": "None"}
  },
  "optimizer": {
    "fixed":   {"n_s": {"tuned": false, "type": "integer", "value": 30},
                "w":   {"tuned": true,  "type": "continuous", "lower": 0, "upper": 1}},
    "adapted": {"w": {"type": "continuous", "lower": 0, "upper": 1}, ...}   // CP space for the adapter
  }
}
```
`parse_config` returns `(SolverConfig, tuning_parameter_space)`: tuned entries become `None` placeholders; untuned
entries take `value`; `optimizer.adapted` becomes `adapted_CP_space` (with or without an `adapter` section).
`tuning_parameter_space` is `None` when nothing is tuned. Current configs: `basic.json` (static PSO, tunes w/c_1/c_2), `test.json` (BetaEDA-PSO),
`sg_pso.json` (no tuning). `test2.json` is stale (uses `kappa_max` and categorical `n_elites`, which `BetaEDA` doesn't accept).

## Running things

Run from the **repo root**, since result paths are relative to CWD. Solver/optimizer/adapter are given as class names
and looked up in `edaptive/registry.py` (unknown names raise an error listing the valid ones). Register new classes
there. The old name `stability_guided_PSO` is still accepted as an alias.

```bash
# Tuning (MPI optional)
mpirun -n 8 python experiments/scripts/tune.py --solver EDASolver --optimizer PSO --adapter BetaEDA \
    --config experiments/configs/test.json
python experiments/scripts/tune.py --solver BasicSolver --optimizer PSO --config experiments/configs/basic.json

# Experiments: uses results/tuning_results/<identifier>/best_configuration.pkl if present, else --config
mpirun -n 8 python experiments/scripts/execute.py --solver EDASolver --optimizer PSO --adapter BetaEDA \
    --identifier testname --n_x 30 --max_iter 5000 --n_trials 50
python experiments/scripts/execute.py --solver StabilityGuidedPSO --identifier sg_pso \
    --config experiments/configs/sg_pso.json
```
`execute.py` writes `results/experiment_results/<identifier>.pkl` as `{problem_name: {trial: history_dict}}`
(trial index = seed; `history_dict["timings"]` as in "Timing").

### Environment setup
- Python **3.10** (TensorFlow 2.9 for calypso), `numpy<2`. Install with `pip install -e '.[dev]'` (`dev` adds ruff and pytest).
- Two environments share this folder:
  - **Host**: `venv/` in the repo, which is machine-specific (`.vscode/settings.json` hardcodes
    `/home/gary/Desktop/edaptive`). Never recreate or modify `venv/` from inside the dev container.
  - **Dev container** (config in `../.devcontainer/`, outside this repo): Python 3.10 comes from the devcontainers
    Python feature. The venv is at `/home/vscode/.venvs/edaptive` on a Docker volume that persists across rebuilds,
    and `post-create.sh` creates it and runs `pip install -e`. `remoteEnv` puts it first on `PATH` and sets
    `VIRTUAL_ENV`/`VENV_DIR`. OpenMPI is apt-installed for mpi4py.
- COCO: `scripts/setup_coco.sh` (installs into `$VENV_DIR`, defaulting to `venv/`; the venv must exist first;
  pinned commit `0a7b447f…`). It resets and rebuilds the shared `.coco/` checkout.
- calypso: `scripts/setup_calypso.sh` (clones `git@github.com:garysteyn/calypso-clean.git` into `./calypso`, **deletes any
  existing `./calypso` first**, applies `calypso-integration.patch`, installs requirements, patches tf_agents).
- `scratch/` holds manual driver scripts, not tests.

### Tests
`python -m pytest` from the repo root (about 30 s). `tests/characterisation/` pins seeded outputs of the current code
(not their correctness). `tests/core/`, `tests/solvers/`, `tests/utils/` and `tests/tuning/` hold unit tests (R8b
grows these). `tests/tuning/test_ifrace_mpi.py` runs both tuning cases under `mpirun -n 2`
(`tests/tuning/mpi_case.py`) and requires exactly the serial reference. It is skipped without `mpirun`.
- Every solver (`BasicSolver`, `UniformRandomSolver`, `StabilityGuidedPSO`, `EDASolver`+`BetaEDA` with each
  elite weighting, with `update_freq=5`, and with the Gaussian copula) on Rosenbrock and GeneralizedEggCrate
  (`n_x=5`, `n_s=10`, 100 iterations). Pins the `f_best`/`prop_feasible`/`prop_stable` histories, final `X` and
  `adaptable_CPs`, and BetaEDA's `a`/`b` (and `R`).
- Same seed reproduces; different seeds differ; no solver touches numpy's global RNG (its state is unchanged,
  and reseeding it doesn't change results).
- Two small seeded `IFRace.tune` runs (BasicSolver tuning w/c_1/c_2; EDASolver+BetaEDA tuning eta/rho_min/n_elites
  and the categorical elite_weighting) on 6 cheap ELA instances at `n_x=2`, `B=300`, 30 iterations per run. Pins
  `L`, the race history and the best configuration.

Comparison is exact (bit-for-bit). The reference was generated in the dev container; another numpy/scipy/libm
build (e.g. the host venv) may not match even when nothing is wrong. A behaviour-preserving change must pass
unchanged. A change meant to alter results regenerates the reference with
`python -m tests.characterisation.make_reference` and explains the difference in the commit message. If a refactor
changes how solvers or tuners are built, update `cases.py`, not the reference.

## calypso integration (Eschwege & Engelbrecht)

`calypso/` contains research code for two self-adaptive PSOs used as **comparison baselines**:
belief-space-guided SA-PSO (`scripts/pso/calypso.py`, run via `scripts.pso.execute`) and Soft Actor-Critic SA-PSO
(`scripts/rl/calypsoTrainSAC.py`, `calypsoEnv.py`, `softActorCritic.py`). The patch:
- makes paths in `const.py` relative to the calypso root and creates output dirs;
- adds CLI args `--problems/--train-problems/--eval-problems` (dotted import paths to **edaptive problem classes**),
  `--dimensions`, `--N`, `--max-iter-cal`, `--runs`, `--seed`;
- makes calypso call edaptive problems directly (`loss = problem`, `lim = (problem.lower, problem.upper)`), so
  calypso must run inside the same venv with `edaptive` installed;
- adds TF determinism (`set_random_seed`, `enable_op_determinism`) and seeds the SAC agent.

The edaptive wrappers (`BS_PSO.run()`, `CalypsoWrapper.train()`) build the CLI, write a **placeholder**
`misc/pkl/funcRanges.pkl` (low=inf, high=-inf, delta=NaN, so calypso's normalised "GBN" printouts are meaningless),
**delete `calypso/data` and `calypso/log`**, then `subprocess.run(..., cwd=calypso_root)`. Results stay in calypso's
own output dirs, and runs aren't safe to execute concurrently. To change calypso behaviour, edit the patched clone and
regenerate `scripts/calypso-integration.patch` (`git -C calypso diff > scripts/calypso-integration.patch`).

## Conventions

- Notation/variable names follow PSO/EC literature: `CP` control parameter (adaptable CPs: see "Optimizer"), `n_s`,
  `n_x`, `X`, `V`, `Y`, `Y_hat`,
  `f_*` fitness, `w`, `c_1`, `c_2`, `eta`, `kappa`, `rho`, `I` instances, `B` budget, `L` races.
- Minimisation everywhere.
- Constructors take explicit keyword arguments that are spread from config dicts (`**(a | b | c)`), so
  **dict keys must exactly match `initialize()` parameter names**.
- All randomness should flow through the single `np.random.Generator` created by the solver from `rnd_seed`.
- Numpy-vectorised over the population; avoid per-particle Python loops.
- Formatting/linting: ruff, configured in `pyproject.toml` (line length 88, target py310). Run `ruff format .` and
  `ruff check .` from the repo root. Rules: `E, W, F, I, N, B`, with `N802/N803/N806/N815/N816` ignored so
  literature notation (`X`, `Y_hat`, `B`, `L`, `T_first`, …) stays. Per-file ignores: `N801` in `problems/`
  (class names such as `BBOB_FID2_IID1` are keys in result pickles and calypso's dotted paths) and `E741` in
  `ifrace.py` (the paper's race index `l`). `*.md` files, `scratch/`, `calypso/`, `.coco/` and `venv/` are excluded.
  `ruff format --check .` and `ruff check .` both pass; keep it that way. Otherwise snake_case, with some
  descriptive comments.
  Formatting-only commits are listed in `.git-blame-ignore-revs` (enable locally with
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`). Add any future formatting-only commit there.
- The author plans substantial refactoring, so check intent with the user before large structural changes.

## Roadmap (planned improvements)

Agreed with the author. Implement each item as its own task and confirm the design before any structural change.
Decisions already made:
- **Naming:** use descriptive words. `n_x` means the problem dimension only.
- **Plurals:** a name is plural wherever it refers to more than one thing. That covers arrays of values (`CPs`,
  `elite_CPs`, `CP_lbs`) and counts (`n_CPs`, `n_elites`). A qualifier stays singular (`CP_space`,
  `adaptable_CP_names`). Exceptions are literature symbols: `n_s`, `n_x`, and the irace paper's `N_param`,
  `N_elite`, `N_new`, `N_l`.
- **PEP 8:** enforce it with ruff, but keep literature notation (`X`, `Y_hat`, `B`, `L`, `T_first`, …) by exempting
  the naming rules.
- **I/F-race fidelity:** follow the original elitist irace (López-Ibáñez et al. 2016 / the irace R package).
  Deliberate deviations: Sobol' initialisation and log-scale parameter handling. Conditional parameters and other
  irace extras are out of scope.

Suggested order (each step is checked against the R8a reference results):
1. R7a: ruff config, `dev` extra, one formatting-only commit. Later diffs then show only real changes.
2. R8a: characterisation tests (done; see "Tests"). They protect all later steps.
3. Known issue 1 (seeded copula RNG; done, copula runs are now pinned). This changed copula results.
4. R2 (done): `adaptable_CPs`, name-based CP mapping, plurals. Only `BasicSolver`'s pinned `adaptable_CPs` changed
   shape (`(1, 3)` → `(n_s, 3)`, same values).
5. R3 (done): `SolverConfig`, shared `BaseSolver` construction/run loop. Results unchanged.
6. R5a–d (done): per-component wall/CPU timings, see "Timing". Results unchanged.
7. R4 phase 1 (done): pure restructuring and the fidelity audit. Results unchanged.
8. R5e (done): tuning time (per-process times summed). Results unchanged.
9. R4 phase 2: fidelity fixes from the audit, one behaviour change per commit, each with its difference from the
   reference explained.
10. R7b (done): the code passes `ruff check`.

R8b is not a separate step: add each component's unit tests in the step that touches it.
Known issue 2 changes EDASolver behaviour, so resolve it before running EDASolver tuning/experiments. Steps 1–6
don't depend on it.

**R4 phase 2: I/F-race fidelity fixes.** Phase 1 (restructure and audit) is done; see "Tuning". Make one commit
per fix. Each commit regenerates the reference and explains the change against it.
- [ ] Deviation 1: pass the current `L` to `IFRaceSamplingModel.update` for the eq. 3 weight. This fixes the
  `ValueError` after `L` has been extended.
- [ ] Deviation 2: a cycling stream of `(instance, seed)` pairs (shuffled `I`, fresh seeds), so new instances never
  run out.
- [ ] Deviation 3: replay the union of the elites' histories, in a defined order, instead of only the longest one.
- [ ] Deviation 4: soft-restart only the parents of zero-distance candidates.
- [ ] Deviation 7: log-scale integers on `[log lb, log(ub+1))` with a floor, at init and when sampling.
- [ ] Deviation 8: `Parameter` rejects `log_scale_tuning` with `lower <= 0`. This changes no results.
- [ ] Deviation 5 (rounding reals to `digits`): the author decides whether to keep it as a deliberate deviation.
- Out of scope: conditional parameters, forbidden configurations, capping, recovery files.

**R8b Unit tests** (pytest, `tests/` mirroring `src/edaptive/`; R8a, the characterisation tests, is done).
- [ ] (b) Unit tests: `Parameter`/`ParameterSpace` validation; `parse_config`; `BenchmarkProblem` batching and
  NaN/inf mapping; PSO shapes, feasibility-gated pbest/gbest and the stability criterion; `BetaEDA` (uniform init,
  moment fit recovers a known Beta, `rho_min` floor, bounds, seeded copula);
  IFRace pieces (N_l/budget formulas, elite selection, sampling bounds including log scale, soft restart).
  Skip COCO/opfunu tests if they aren't installed. Add an optional `mpirun -n 2` test that must equal the serial run.

## Known issues / things to verify (found by reading the code, not yet confirmed or fixed)

2. **Possible credit-assignment / timing mismatch in `EDASolver`**: after `PSO.step`, `f_X` is the fitness of the
   pre-move positions but `X` (used for the feasibility filter) is post-move, and `adaptable_CPs` is the matrix used for the move
   just made, not the one that produced the positions in `f_X`. With `update_freq=1` and full resampling, the elites
   selected may not be the CPs responsible for the observed fitness. Confirm the intended semantics with the author.
3. `BetaEDA.update`: weighted variance is not bias-corrected while the unweighted path uses `ddof=1`, and
   `np.maximum(var, eps)` is applied twice.
4. `execute.py`: iterates over `tuning_problems` (the `evaluation_problems` list is defined but unused); has a hardcoded
   debug print `results["CosineMixture_OG"][k] for k in range(0, 50)` (fails if `n_trials < 50`); `config` is undefined
   when there's neither a tuned pkl nor `--config`.
5. `CECProblem._evaluate` forwards batched `(N, D)` input straight to opfunu's `evaluate`, which is likely 1-D only.
   Verify before using CEC problems with PSO.
6. `create_adapted_parameter_space` ignores integer/categorical specifics; `UniformRandomAdapter`/`BetaEDA` assume continuous CPs.
8. PSO history keys `velocities`/`diversity` are only populated at initialisation.
9. Empty placeholders: `solvers/pso/tvac_pso.py`, `problems/misc_problems.py`.
10. Git: branches `main` and `elitist-ifrace` (also on origin).
13. `BBOB_FID17_IID2` is defined twice, as two separate classes: in `ela_benchmark_suite.py` (tuning set) and in
    `custom_test_set.py` (evaluation set). They compute the same function, but result pickles and calypso paths
    identify problems by class name, so the tuning and evaluation versions are indistinguishable there.
    `execute.py` imports the evaluation-set version.
