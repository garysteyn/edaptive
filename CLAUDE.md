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
  hyper_params or JSON config schema, run or setup commands, output paths, conventions, and the calypso/COCO integration.
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
│   │   ├── solver.py         # BaseSolver
│   │   └── parameters.py     # ParameterType, Parameter, ParameterSpace
│   ├── adapters/             # generic (optimizer-agnostic) adapters
│   │   ├── beta_eda.py       # BetaEDA (beta marginals, optional Gaussian copula) + commented-out legacy variants
│   │   └── uniform_random.py # UniformRandomAdapter (EDAAdapter; uniform resampling baseline)
│   ├── optimizers/
│   │   └── pso.py            # PSO (gbest, per-particle CPs), swarm_diversity, stable_particle_proportion
│   ├── solvers/              # wrappers that run optimizer (+ adapter)
│   │   ├── basic_solver.py   # BasicSolver: static CPs, no adapter
│   │   ├── eda_solver.py     # EDASolver: optimizer + any EDAAdapter
│   │   ├── uniform_random_solver.py # UniformRandomSolver: optimizer + UniformRandomAdapter (also has a scratch main())
│   │   └── pso/
│   │       ├── sg_pso.py     # stability_guided_PSO: PSO-specific adaptation implemented as a solver
│   │       └── tvac_pso.py   # empty placeholder
│   ├── problems/
│   │   ├── benchmark_problem.py   # BenchmarkProblem ABC
│   │   ├── ela_benchmark_suite.py # ~24 hand-written functions + 4 BBOB wrappers (the tuning/training set)
│   │   ├── bbob.py                # BBOBProblem (COCO cocoex)
│   │   ├── cec.py                 # CECProblem (opfunu)
│   │   ├── custom_test_set.py     # evaluation set: BBOB_FIDx_IIDy and CEC_yyyy_FIDx classes
│   │   └── misc_problems.py       # empty
│   ├── tuning/
│   │   ├── ifrace.py               # IFRace: elitist iterated F-race (MPI-parallel)
│   │   └── ifrace_sampling_model.py# IFRaceSamplingModel ABC + IFRaceDefaultSamplingModel (irace-style)
│   ├── integrations/calypso/       # subprocess wrappers around the external calypso repo
│   │   ├── bs_pso.py               # BS_PSO   → calypso `scripts.pso.execute` (belief-space SA-PSO)
│   │   └── sac_pso.py              # CalypsoWrapper.train() → calypso `scripts.rl.calypsoTrainSAC` (SAC SA-PSO)
│   │   (integrations/test.py is a scratch driver script)
│   └── utils/
│       ├── experiment_utils.py     # JSON config → (config_template, tuning ParameterSpace)
│       └── timing.py               # @timed decorator → obj._timings[func_name] accumulates seconds
├── experiments/
│   ├── configs/*.json        # algorithm configs (see "Config schema")
│   └── scripts/
│       ├── tune.py           # run I/F-race for a solver/optimizer/adapter + config
│       ├── execute.py        # run final experiments (uses tuned config if present), MPI-parallel
│       └── test*.py, temp.py, sobol_owen.py   # ad-hoc scratch scripts, not tests
├── tests/                    # pytest (`testpaths` set in pyproject.toml); see "Tests"
│   ├── conftest.py           # forces the Agg matplotlib backend
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
- `CP` property: an `(n_s, n_cp)` array of **per-individual control parameters** (each particle has its own CPs).
  Adapters/solvers write new CPs with `optimizer.CP = new_array`.
- `get_required_data(names)` returns `{name: getattr(self, name)}`. This is how adapters pull state from the optimizer
  generically, so attribute names on optimizers form an implicit interface (`problem_lb`, `problem_ub`, `t`, `X`,
  `f_X`, `CP` for BetaEDA).

### PSO (`optimizers/pso.py`)
- `initialize(problem, rng, n_s, w, c_1, c_2)`. `w`, `c_1`, `c_2` may be scalars or length-`n_s` arrays;
  `_CP = column_stack((w, c_1, c_2))`, so **CP column order is (w, c_1, c_2)**.
- Notation follows Engelbrecht: `X` positions, `V` velocities, `Y`/`f_Y` personal bests, `Y_hat`/`f_Y_hat`
  global best, `n_s` swarm size, `n_x` dimension, `r_1`, `r_2` uniform coefficients.
- `iterate()` order: **evaluate X → update pbest/gbest (feasible particles only) → update V → X += V**.
  So after `step(t)`, `self.f_X` holds the fitness of the *pre-move* positions while `self.X` holds the *post-move* positions.
- Boundary handling: none (no clamping). Infeasible particles are evaluated but can't become pbest/gbest.
  Inf/NaN in V and X are replaced with ±`max_float`.
- History: `f_best` (gbest per iteration), `prop_feasible`, `prop_stable` (Poli stability criterion
  `c1 + c2 < 24(1 - w²)/(7 - 5w)`); `velocities` and `diversity` are currently only recorded at init.
  `stable_particle_proportion` raises if any `w ∉ [-1, 1]`.
- `@timed` on `initialize` and `iterate`; solvers copy `_timings` into `history["timings"]`.

### Adapter (`core/adapter.py`)
- `BaseAdapter.__init__(param_space, **kwargs)`: `param_space` is a `ParameterSpace` of the **adapted** CPs.
  Sets `param_space`, `CP_lb`, `CP_ub`, `n_params` (= number of adapted CPs), then `initialize(**kwargs)`.
- `required_data()` lists optimizer attributes needed for an update. `EDAAdapter` adds abstract `sample(N)`.
- Contract used by `EDASolver`: `check_update(**data) -> elites | None`, then `update(elites)`, then `sample(N)`.
  (`check_update` isn't declared on the ABC yet.)
- **Adapters are meant to be optimizer-agnostic.** Adaptation schemes tailored to one optimizer
  (e.g. stability-guided PSO) are implemented directly as **solvers** instead.

### BetaEDA (`adapters/beta_eda.py`)
`initialize(eta, rho_min, n_elite, update_freq, copula, elite_weighting, rng)`
- Model: independent **Beta(a_j, b_j) marginals** on CPs scaled to [0,1]; optional **Gaussian copula**
  (`copula == "gaussian"`) with correlation matrix `R`. Any other value (e.g. `None` / `"None"`) means independent marginals.
- Init: `a = b = 1` (uniform), `R = I`.
- `check_update`: only every `update_freq` iterations (`(t+1) % update_freq == 0`); filters to particles whose
  `X` is inside problem bounds; needs ≥ 2 feasible; elites = CPs of the `n_elite` best `f_X`.
- `update`: scale elites to [0,1], clip to [1e-6, 1-1e-6]; weighted mean/variance (weights: `None` = uniform with
  ddof=1 variance, `"rank"` = linear, `"log-rank"` = log(n+1) - log(rank)); method-of-moments concentration
  `kappa = mu(1-mu)/var - 1`, clipped to `[eps, kappa_max]` with `kappa_max = 1/rho_min - 1`
  (`rho = 1/(kappa+1) = var/(mu(1-mu))` is the normalised variance, so `rho_min` is a floor on it).
  Then `a, b ← (1-eta)·old + eta·(mu·kappa, (1-mu)·kappa)` (exponential smoothing with learning rate `eta`).
  Copula: transform elites through the *updated* beta CDFs → `norm.ppf` → `corrcoef` → EMA into `R` with the same `eta`.
- `sample(N)`: copula path = Cholesky of `R + 1e-10·I`, correlated normals (`self.rng.standard_normal`) →
  `norm.cdf` → `beta.ppf`; otherwise `beta.rvs(..., random_state=self.rng)`. Output is rescaled to `[CP_lb, CP_ub]`.
- Older separate classes (`BetaMarginalsEDA`, rank/log-rank variants, `BetaMarginalsGaussianCopulaEDA`) remain
  commented out at the bottom of the file; `BetaEDA` replaces them.

### Solvers (`core/solver.py: BaseSolver`)
- `__init__(optimizer_type, adapter_type=None, **kwargs)` → `initialize(**kwargs)`; `run(max_iterations=5000)`
  returns `(min(history["f_best"]), history)`.
- Uniform construction convention (used by IFRace and execute.py):
  `Solver(optimizer_type=..., adapter_type=..., problem=<instance>, hyper_params=<dict>, rnd_seed=<int>)`.
- Each solver creates **one** `rng = np.random.default_rng(rnd_seed)` and passes it to both optimizer and adapter.
- `BasicSolver`: optimizer gets `hyper_params["optimizer"]["fixed"]` (which must include w, c_1, c_2).
- `EDASolver`: adapter built from `hyper_params["adapter"]` + rng over `hyper_params["optimizer"]["adapted"]`
  (a `ParameterSpace`); initial CPs = `adapter.sample(n_s)`, passed to the optimizer by parameter name. Each iteration:
  `optimizer.step(t)` → `check_update` → if elites: `update`, resample **all** `n_s` CPs.
- `UniformRandomSolver`: resamples all CPs uniformly every iteration (hardwires `UniformRandomAdapter`).
- `stability_guided_PSO`: hardcodes PSO and CP bounds `[0,1]×[0,4]×[0,4]`; resamples all CPs uniformly every
  iteration with rejection sampling until every particle satisfies Poli's stability criterion. Its constructor
  takes no `optimizer_type`/`adapter_type`.

### hyper_params dict (runtime form)
```python
{
  "identifier": "name",                       # present when built from JSON
  "adapter":   {"eta": 0.01, "rho_min": 0.02, "n_elite": 10, "update_freq": 1,
                "copula": None, "elite_weighting": "log-rank"},
  "optimizer": {
      "fixed":   {"n_s": 30},                 # or w/c_1/c_2 too for BasicSolver
      "adapted": ParameterSpace([...]),       # CPs controlled by the adapter
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

`IFRace(I, n_x, sampling_model, B, L=None, n_min=None, mu=None, n_iter_per_run=5000, friedman_alpha, conover_alpha,
T_first=5, T_each=1, T_new=1, T_max=2, digits=4, results_directory="results/tuning_results")`
→ `.tune(solver_type, optimizer_type, adapter_type, config_template, tuning_parameter_space, rnd_seed=0)`.

- Follows irace's elitist variant: budget `B` split over `L = floor(2 + log2(N_param))` races (extended if budget
  remains); `n_min` survivors become elites carried into the next race with their instance/result history cached
  (elites aren't re-run on already-seen `(instance, seed)` pairs); new instances (`T_new`) are raced first;
  Friedman test + Conover post-hoc for elimination after `T_first` instances; stop on `n_min` survivors or `T_max`
  consecutive tests with no elimination; elites can't be eliminated before they've seen `e_j + T_new` instances.
- An "instance" is a `(ProblemClass, seed)` pair. Results are final best fitness from `solver.run(n_iter_per_run)`.
- `IFRaceDefaultSamplingModel`: race 1 = Owen-scrambled Sobol over the tuning space (log-scale aware);
  later races sample from elites (rank-weighted) with truncated normals whose sigma shrinks by
  `(1/N_new)^(1/n_params)` per race, categorical probabilities pulled toward the elite value, and irace-style
  **soft restart** when sampled configs are too close (`10^-digits`) to each other or to elites.
  Each config carries a `'_config_id'` key that indexes per-config sigma/probability state.
- **MPI**: rank 0 owns all decisions and logging; configurations are evaluated round-robin across ranks
  (`range(rank, len(C), size)`) and synced with `bcast`/`gather`/`allreduce`. Any rank-0-only change must keep
  the broadcast sequence identical on all ranks, or it will deadlock.
- Configs are materialised with `create_config_dict` → deep-copies `config_template` and `set_parameters` overwrites
  **any key with a matching name at any nesting depth**, so parameter names must be unique across adapter and optimizer.
- Outputs (relative to CWD): `results/tuning_results/<identifier>/logs/{tuning,evaluations,races,elites,sampling}.log`,
  `candidate_trajectory.png`, `best_configuration.pkl` (a full hyper_params dict).

## Config schema (`experiments/configs/*.json`) and `utils/experiment_utils.parse_config`

```json
{
  "identifier": "my_alg",
  "adapter": {                                   // omit for adapter-free solvers
    "eta":     {"tuned": true,  "type": "continuous", "lower": 1e-4, "upper": 0.2, "log_scale_tuning": true},
    "n_elite": {"tuned": true,  "type": "integer",    "lower": 2,    "upper": 30},
    "copula":  {"tuned": false, "type": "categorical", "value": "None"}
  },
  "optimizer": {
    "fixed":   {"n_s": {"tuned": false, "type": "integer", "value": 30},
                "w":   {"tuned": true,  "type": "continuous", "lower": 0, "upper": 1}},
    "adapted": {"w": {"type": "continuous", "lower": 0, "upper": 1}, ...}   // CP space for the adapter
  }
}
```
`parse_config` returns `(config_template, tuning_parameter_space)`: tuned entries become `None` placeholders in the
template; untuned entries take `value`; `adapted` becomes a `ParameterSpace`. `tuning_parameter_space` is `None` when
nothing is tuned. Current configs: `basic.json` (static PSO, tunes w/c_1/c_2), `test.json` (BetaEDA-PSO),
`sg_pso.json` (no tuning). `test2.json` is stale (uses `kappa_max` and categorical `n_elite`, which `BetaEDA` doesn't accept).

## Running things

Run from the **repo root**, since result paths are relative to CWD. Solver/optimizer/adapter are given as class names
and resolved with `eval` against the script's imports.

```bash
# Tuning (MPI optional)
mpirun -n 8 python experiments/scripts/tune.py --solver EDASolver --optimizer PSO --adapter BetaEDA \
    --config experiments/configs/test.json
python experiments/scripts/tune.py --solver BasicSolver --optimizer PSO --config experiments/configs/basic.json

# Experiments: uses results/tuning_results/<identifier>/best_configuration.pkl if present, else --config
mpirun -n 8 python experiments/scripts/execute.py --solver EDASolver --optimizer PSO --adapter BetaEDA \
    --identifier testname --n_x 30 --max_iter 5000 --n_trials 50
python experiments/scripts/execute.py --solver stability_guided_PSO --identifier sg_pso \
    --config experiments/configs/sg_pso.json
```
`execute.py` writes `results/experiment_results/<identifier>.pkl` as `{problem_name: {trial: history_dict}}`
(trial index = seed).

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
- `experiments/scripts/test*.py` are manual smoke scripts, not tests.

### Tests
`python -m pytest` from the repo root (about 6 s). `tests/characterisation/` pins seeded outputs of the current code
(not their correctness):
- Every solver (`BasicSolver`, `UniformRandomSolver`, `stability_guided_PSO`, `EDASolver`+`BetaEDA` with each
  elite weighting, with `update_freq=5`, and with the Gaussian copula) on Rosenbrock and GeneralizedEggCrate (`n_x=5`, `n_s=10`, 100
  iterations). Pins the `f_best`/`prop_feasible`/`prop_stable` histories, final `X` and `CP`, and BetaEDA's `a`/`b` (and `R`).
- Same seed reproduces; different seeds differ; no solver touches numpy's global RNG (its state is unchanged,
  and reseeding it doesn't change results).
- Two small seeded `IFRace.tune` runs (BasicSolver tuning w/c_1/c_2; EDASolver+BetaEDA tuning eta/rho_min/n_elite
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

- Notation/variable names follow PSO/EC literature: `CP` control parameters, `n_s`, `n_x`, `X`, `V`, `Y`, `Y_hat`,
  `f_*` fitness, `w`, `c_1`, `c_2`, `eta`, `kappa`, `rho`, `I` instances, `B` budget, `L` races.
- Minimisation everywhere.
- Constructors take explicit keyword arguments that are spread from config dicts (`**(a | b | c)`), so
  **dict keys must exactly match `initialize()` parameter names**.
- All randomness should flow through the single `np.random.Generator` created by the solver from `rnd_seed`.
- Numpy-vectorised over the population; avoid per-particle Python loops.
- Formatting/linting: ruff, configured in `pyproject.toml` (line length 88, target py310). Run `ruff format .` and
  `ruff check .` from the repo root. Rules: `E, W, F, I, N, B`, with `N802/N803/N806/N815/N816` ignored so
  literature notation (`X`, `Y_hat`, `B`, `L`, `T_first`, …) stays. `*.md` files, `calypso/`, `.coco/` and `venv/`
  are excluded. The code is ruff-formatted; lint violations remain (see R7b). Otherwise snake_case, with some
  descriptive comments.
  Formatting-only commits are listed in `.git-blame-ignore-revs` (enable locally with
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`). Add any future formatting-only commit there.
- The author plans substantial refactoring, so check intent with the user before large structural changes.

## Roadmap (planned improvements)

Agreed with the author. Implement each item as its own task and confirm the design before any structural change.
Decisions already made:
- **Naming:** use descriptive words. `n_x` means the problem dimension only.
- **PEP 8:** enforce it with ruff, but keep literature notation (`X`, `Y_hat`, `B`, `L`, `T_first`, …) by exempting
  the naming rules.
- **I/F-race fidelity:** follow the original elitist irace (López-Ibáñez et al. 2016 / the irace R package).
  Deliberate deviations: Sobol' initialisation and log-scale parameter handling. Conditional parameters and other
  irace extras are out of scope.

Suggested order (each step is checked against the R8a reference results):
1. R7a: ruff config, `dev` extra, one formatting-only commit. Later diffs then show only real changes.
2. R8a: characterisation tests (done; see "Tests"). They protect all later steps.
3. Known issue 1 (seeded copula RNG; done, copula runs are now pinned). This changed copula results.
4. R2 renaming pass. Pinned results must stay unchanged. Settle the `"adapted"` key question here.
5. R3, together with the `StabilityGuidedPSO` rename from R7b. Results stay unchanged.
6. R5a–d. These need the shared run loop from R3.
7. R4 phase 1: pure restructuring. Its output must be bit-identical to the R8a reference.
8. R5e: serial tuning time.
9. R4 phase 2: fidelity audit, one behaviour change per commit, each with its difference from the reference
   explained.
10. The rest of R7b.

R8b is not a separate step: add each component's unit tests in the step that touches it.
Known issue 2 changes EDASolver behaviour, so resolve it before running EDASolver tuning/experiments. Steps 1–6
don't depend on it.

**R2 Separate adaptive control parameters from fixed ones (e.g. `n_s`).** Must not change numerical results.
- [ ] `BaseOptimizer.CP`/`_CP` → `adaptive_params`/`_adaptive_params`, shape `(n_s, n_adaptive)`.
- [ ] Optimizers declare `adaptive_param_names` (PSO: `("w", "c_1", "c_2")`). Solvers then map columns by name instead of
  by position (this removes the hardcoded indexing in `sg_pso.py` and the order assumption in the solvers).
- [ ] `CP_lb`/`CP_ub` → `param_lb`/`param_ub`; in `BetaEDA`, `elite_CPs`/`get_elite_CPs`/`scaled_CP` → `elite_params`/…;
  the `required_data()` entry `"CP"` → `"adaptive_params"`.
- [ ] Solver locals: `CP_init`/`CP_new`/`sample_CP` → `initial_params`/`new_params`/`sample_params`.
- [ ] Decide whether the hyper_params/JSON key `"adapted"` → `"adaptive"`. Renaming it breaks existing configs and
  `best_configuration.pkl` files, so it needs a migration or a backward-compatible loader.

**R3 Cleaner solver construction (`hyper_params` unpacking).** This is a structural change, so confirm the design first.
- [ ] Replace the nested `hyper_params["optimizer"]["fixed"]` access with a small frozen dataclass, e.g.
  `SolverConfig(identifier, fixed_params, adaptive_space, adapter_params)`, produced by `parse_config`, with
  `from_dict()` for old pickles.
- [ ] Move the shared logic into `BaseSolver`: RNG creation, `_build_optimizer(problem, initial_params=None)`,
  `_build_adapter()`, a common `run()` loop with a per-iteration hook `adapt(t)`, and timing collection
  (covers Known issue 7).
- [ ] Give every solver the same constructor signature, `stability_guided_PSO` included (fixes the constructor parts of
  Known issue 4).
- Constraint: keep the RNG consumption order (in `EDASolver`, the adapter samples initial params before PSO draws `X`).

**R4 Rewrite the elitist I/F-race tuner for readability and fidelity.** Behaviour changes are allowed where they move
the code toward irace, but each one must be stated explicitly.
- [ ] Merge `ifrace_sampling_model.py` into `tuning/ifrace.py` and drop the `IFRaceSamplingModel` ABC.
- [ ] Structure the code to mirror the published algorithm: `tune()` is the outer loop (budget split, `L`, extending
  `L`). Separate methods cover: configurations per race, sampling/model update, the race over instances, the
  Friedman + Conover test and elimination, and elite selection. Docstrings cite the paper's sections/equations.
- [ ] Isolate the parallelism in one `_evaluate(jobs)` that spreads `(config, instance, seed)` jobs over ranks and
  gathers `(result, history timings)`. Rank 0 makes all decisions, and the state is broadcast once per step. The
  algorithm code contains no `comm.*` calls.
- [ ] Use explicit read-only properties for the tuner and sampling-model state (elites, weights, sigma per config,
  categorical probabilities, sampled parent indices, best configuration, race history). The tuner must no longer
  read model internals such as `numeric_sigma` and `categorical_update_elite`.
- [ ] Separate logging from the logic (so R5e can exclude it). Replace `print` calls with logging. Rename `nm()`.
- [ ] Fidelity audit against irace, recording every deviation here: the N_l formula (`mu + T_each·min(5,l)`), the
  budget split and extension of `L`, the per-race sigma reduction, the categorical update and cap, soft restart,
  elite protection (`e_j + T_new`), the instance order (new instances, then seen ones, then the rest), the counter of
  tests without elimination, and instance seeding.
- [ ] Verify that the log-scale handling is consistent across Sobol' init, sigma init (half the log range), truncated
  normal sampling in log space, integer rounding, the distance check and soft restart. Keep it if it is sound.
- [ ] Out of scope: conditional parameters, forbidden configurations, capping, recovery files.
- The R8a tuning cases are the seeded reference, so every change in results can be explained against it.

**R5 Timing of the core algorithm components** (exclude logging and performance measures).
- [ ] (a) Adapters: `BaseAdapter._timings`; `@timed` on `check_update`/`update`/`sample` of `BetaEDA` and the random
  adapter.
- [ ] (b) Tailored adaptation logic: time the rejection sampling in `stability_guided_PSO`.
- [ ] (c) Optimizer↔adapter interaction: time `get_required_data` and the adaptive-params setter in the solvers.
  They pass references, so they are probably negligible. Measure, then keep or drop.
- [ ] (d) `history["timings"]` → `{"optimizer": {...}, "adapter": {...}, "solver": {...}}` plus a helper
  `core_time(timings)` that sums them. It excludes `update_history`/`prop_stable` etc. Use `OMP_NUM_THREADS=1` for
  honest serial times.
- [ ] (e) IFRace serial tuning time for each race: `solver_time` = Σ `core_time` over the runs actually executed
  (cached elite results add 0); `tuner_time` = sampling, tests, elimination and elite selection on rank 0 (excluding
  logging and MPI); `total` = their sum. Write to `tuning.log` and `results/tuning_results/<id>/tuning_times.json`,
  including the grand total. Depends on R4.

**R7 PEP 8 compliance.**
- [x] (a) ruff config, `dev` extra, `ruff format` applied (the AST is identical before and after).
- [ ] (b) Fix the remaining `ruff check` violations (about 280): wildcard imports (`F403/F405`, mostly in
  `experiments/scripts`), unused/unsorted imports (`F401/I001`, auto-fixable), `E501` long lines (mostly
  `ifrace.py`), `E741` (the race index `l` in `ifrace.py`: rename it or ignore the rule there), `B905/B027/B008`,
  `stability_guided_PSO` → `StabilityGuidedPSO` (update the commands above), and the scratch `main()` blocks
  (Known issue 9). Decide on `N801` for the problem classes (`BBOB_FID2_IID1`, `CEC_2013_FID13`,
  `CosineMixture_OG`, …). Their names are used as keys in result pickles and in calypso's dotted paths, so a
  per-file ignore for `problems/` is probably better than renaming them.

**R8b Unit tests** (pytest, `tests/` mirroring `src/edaptive/`; R8a, the characterisation tests, is done).
- [ ] (b) Unit tests: `Parameter`/`ParameterSpace` validation; `parse_config`; `BenchmarkProblem` batching and
  NaN/inf mapping; PSO shapes, feasibility-gated pbest/gbest and the stability criterion; `BetaEDA` (uniform init,
  moment fit recovers a known Beta, `rho_min` floor, bounds, seeded copula); timing
  structure; IFRace pieces (N_l/budget formulas, elite selection, sampling bounds including log scale, soft restart).
  Skip COCO/opfunu tests if they aren't installed. Add an optional `mpirun -n 2` test that must equal the serial run.

## Known issues / things to verify (found by reading the code, not yet confirmed or fixed)

2. **Possible credit-assignment / timing mismatch in `EDASolver`**: after `PSO.step`, `f_X` is the fitness of the
   pre-move positions but `X` (used for the feasibility filter) is post-move, and `CP` is the matrix used for the move
   just made, not the one that produced the positions in `f_X`. With `update_freq=1` and full resampling, the elites
   selected may not be the CPs responsible for the observed fitness. Confirm the intended semantics with the author.
3. `BetaEDA.update`: weighted variance is not bias-corrected while the unweighted path uses `ddof=1`, and
   `np.maximum(var, eps)` is applied twice.
4. `execute.py`: iterates over `tuning_problems` (the `evaluation_problems` list is defined but unused); has a hardcoded
   debug print `results["CosineMixture_OG"][k] for k in range(0, 50)` (fails if `n_trials < 50`); `config` is undefined
   when there's neither a tuned pkl nor `--config`; untuned branch doesn't pass `adapter_type` (breaks `EDASolver`);
   tuned branch passes `optimizer_type`/`adapter_type`, which `stability_guided_PSO` doesn't accept (duplicate kwarg).
   The same constructor mismatch means `tune.py` can't drive `stability_guided_PSO`. (Constructor parts → R3.)
5. `CECProblem._evaluate` forwards batched `(N, D)` input straight to opfunu's `evaluate`, which is likely 1-D only.
   Verify before using CEC problems with PSO.
6. `create_adapted_parameter_space` ignores integer/categorical specifics; `UniformRandomAdapter`/`BetaEDA` assume continuous CPs.
7. `BaseAdapter`'s abstract interface doesn't declare `check_update`; `BaseSolver` lacks shared run-loop logic
   (duplicated across solvers). (→ R3.)
8. PSO history keys `velocities`/`diversity` are only populated at initialisation.
9. Empty placeholders: `solvers/pso/tvac_pso.py`, `problems/misc_problems.py`. Scratch `main()` blocks live in
   `uniform_random_solver.py` and `ela_benchmark_suite.py`. (→ R7b.)
10. Git: branches `main` and `elitist-ifrace` (also on origin).
11. `parse_config` only builds `optimizer.adapted` when the config has an `"adapter"` section, so
    `UniformRandomSolver` (adapted space, no adapter params) can't be built from a JSON config. The
    characterisation tests work around it with `create_adapted_parameter_space`. (Confirmed.)
12. `IFRace.setup_loggers` names its loggers by a timestamp with one-second resolution. Two `tune()` calls
    started within the same second in one process share loggers, so the second run also writes into the first
    run's log files. Logs only; results are unaffected.
