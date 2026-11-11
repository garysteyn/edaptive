from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace


def create_adapted_parameter_space(config: dict) -> ParameterSpace:

    parameters = []

    parameter_configs = config["optimizer"]["adapted"]

    for name, parameter_config in parameter_configs.items():
        parameters.append(Parameter.from_config(name, parameter_config))

    return ParameterSpace(parameters)


def create_tuning_parameter_space(config: dict) -> ParameterSpace:
    parameters = []

    if "adapter" in config:
        for name, spec in config["adapter"].items():
            if spec.get("tuned", False):
                parameters.append(Parameter.from_config(name, spec))

    for name, spec in config["optimizer"]["fixed"].items():
        if spec.get("tuned", False):
            parameters.append(Parameter.from_config(name, spec))

    if len(parameters) > 0:
        return ParameterSpace(parameters)


def parse_config(config):
    """JSON config → ``(SolverConfig, tuning ParameterSpace or None)``.

    Tuned entries become ``None`` placeholders in the ``SolverConfig``; untuned
    entries take their ``value``.
    """

    def values(section):
        return {
            name: None if spec["tuned"] else spec["value"]
            for name, spec in section.items()
        }

    adapted_CP_space = None
    if "adapted" in config["optimizer"]:
        adapted_CP_space = create_adapted_parameter_space(config=config)

    solver_config = SolverConfig(
        identifier=config["identifier"],
        fixed_CPs=values(config["optimizer"]["fixed"]),
        adapted_CP_space=adapted_CP_space,
        adapter_params=values(config.get("adapter", {})),
    )

    return (solver_config, create_tuning_parameter_space(config=config))
