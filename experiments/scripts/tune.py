import argparse
import json
from pathlib import Path

from edaptive.solvers.eda_solver import EDASolver
from edaptive.adapters.beta_eda import *
from edaptive.optimizers.pso import PSO 
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace
from edaptive.tuning.ifrace import IFRace
from edaptive.tuning.ifrace_sampling_model import IFRaceDefaultSamplingModel
from edaptive.problems.ela_benchmark_suite import CosineMixture_OG, GeneralizedPrice2, Brown, Mishra07, Step3
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--solver",
        type=str,
        required=True,
        help="Solver to run."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer to run."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Adapter to run."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file."
    )

    return parser.parse_args()

# TODO: Change the following to account for categorical parameters
def create_adapted_parameter_space(
    config: dict
) -> ParameterSpace:

    parameters = []

    parameter_configs = config["optimizer"]["adapted"]

    for name, parameter_config in parameter_configs.items():
        parameters.append(
            Parameter(
                name=name,
                lower=parameter_config["lower"],
                upper=parameter_config["upper"],
                type=ParameterType(
                    parameter_config["type"]
                )
            )
        )

    return ParameterSpace(parameters)

def create_tuning_parameter_space(config: dict) -> ParameterSpace:
    parameters = []

    for name, spec in config["adapter"].items():
        if spec.get("tuned", False):
            parameters.append(Parameter.from_config(name, spec))

    for name, spec in config["optimizer"]["fixed"].items():
        if spec.get("tuned", False):
            parameters.append(Parameter.from_config(name, spec))

    return ParameterSpace(parameters)

def parse_config(config):
    adapted_parameter_space = create_adapted_parameter_space(config=config)
    config_template = {"adapter" : dict(), "optimizer" : {"fixed" : dict(), "adapted" : adapted_parameter_space}}

    for name, spec in config["adapter"].items():
        if spec["tuned"]:
            config_template["adapter"][name] = None
        else:
            config_template["adapter"][name] = spec["value"]
    
    for name, spec in config["optimizer"]["fixed"].items():
        if spec["tuned"]:
            config_template["optimizer"]["fixed"][name] = None
        else:
            config_template["optimizer"]["fixed"][name] = spec["value"]

    tuning_parameter_space = create_tuning_parameter_space(config=config)
    # print(tuning_parameter_space)
    return (config_template, tuning_parameter_space)

def update_parameters(config: dict, updates: dict) -> None:
    for key, value in config.items():

        if key in updates:
            config[key] = updates[key]

        elif isinstance(value, dict):
            update_parameters(value, updates)

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