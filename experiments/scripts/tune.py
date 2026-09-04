import json
from edaptive.utils.experiment_utils import parse_args, parse_config
from edaptive.solvers.eda_solver import EDASolver
from edaptive.adapters.beta_eda import *
from edaptive.optimizers.pso import PSO 
from edaptive.tuning.ifrace import IFRace
from edaptive.tuning.ifrace_sampling_model import IFRaceDefaultSamplingModel
from edaptive.problems.ela_benchmark_suite import CosineMixture_OG, GeneralizedPrice2, Brown, Mishra07, Step3

def main():
    args = parse_args()

    solver_type = eval(args.solver)
    optimizer_type = eval(args.optimizer)
    adapter_type = eval(args.adapter)

    with args.config.open("r") as file:
        config = json.load(file)

    config_template, tuning_parameter_space = parse_config(config)

    solver = solver_type(optimizer_type=optimizer_type, adapter_type=adapter_type)
    
    ifrace = IFRace(
        I=[
            CosineMixture_OG,
            GeneralizedPrice2,
            Brown,
            Mishra07,
            Step3
        ] * 10,
        n_x=30,
        sampling_model=IFRaceDefaultSamplingModel,
        B=1000
    )

    ifrace.tune(
        solver=solver,
        config_template=config_template,
        tuning_parameter_space=tuning_parameter_space
    )

if __name__ == "__main__":
    main()