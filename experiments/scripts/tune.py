import json
from edaptive.utils.experiment_utils import parse_args, parse_config
from edaptive.solvers.eda_solver import EDASolver
from edaptive.adapters.beta_eda import *
from edaptive.optimizers.pso import PSO 
from edaptive.tuning.ifrace import IFRace
from edaptive.tuning.ifrace_sampling_model import IFRaceDefaultSamplingModel
from edaptive.problems.ela_benchmark_suite import *

def main():
    args = parse_args()

    solver_type = eval(args.solver)
    optimizer_type = eval(args.optimizer)
    adapter_type = eval(args.adapter)

    with args.config.open("r") as file:
        config = json.load(file)

    config_template, tuning_parameter_space = parse_config(config)

    # solver = solver_type(optimizer_type=optimizer_type, adapter_type=adapter_type)
    
    ifrace = IFRace(
        I=[
            Schwefel1,
            Ripple25,
            Exponential,
            NeedleEye,
            Step3,
            GeneralizedGiunta,
            GeneralizedPaviani,
            Brown,
            CosineMixture_OG,
            CosineMixture,
            Mishra07,
            Mishra01,
            GeneralizedPrice2,
            BBOB_FID2_IID1,
            BBOB_FID6_IID1,
            BBOB_FID16_IID1,
            BBOB_FID17_IID2
],
        n_x=30,
        sampling_model=IFRaceDefaultSamplingModel,
        B=100
    )

    ifrace.tune(
        solver_type=solver_type,
        optimizer_type=optimizer_type,
        adapter_type=adapter_type,
        config_template=config_template,
        tuning_parameter_space=tuning_parameter_space
    )

if __name__ == "__main__":
    main()