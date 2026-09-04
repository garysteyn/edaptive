import argparse
from pathlib import Path
from edaptive.core.parameters import ParameterType, Parameter, ParameterSpace

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