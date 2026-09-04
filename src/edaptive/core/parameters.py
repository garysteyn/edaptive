import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

class ParameterType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"

@dataclass(frozen=True)
class Parameter:
    name: str
    type: ParameterType

    lower: float | None = None
    upper: float | None = None
    classes: tuple = ()
    log_scale_tuning: bool = False

    @classmethod
    def from_config(cls, name: str, spec: dict) -> "Parameter":
        parameter_type = ParameterType(spec["type"])
        if parameter_type == ParameterType.CATEGORICAL:
            return cls(
                name=name,
                type=parameter_type,
                classes=tuple(spec["classes"]),
            )

        use_log_scale_tuning = False
        if spec.get("log_scale_tuning", False):
            use_log_scale_tuning = True

        return cls(
            name=name,
            type=parameter_type,
            lower=spec["lower"],
            upper=spec["upper"],
            log_scale_tuning=use_log_scale_tuning
        )

    @property
    def range(self) -> float:
        if self.type == ParameterType.CATEGORICAL:
            raise AttributeError(
                "Categorical parameters do not have a numeric range"
            )
        return self.upper - self.lower

    @property
    def n_classes(self) -> int:
        if self.type != ParameterType.CATEGORICAL:
            raise AttributeError(
                "Only categorical parameters have classes"
            )
        return len(self.classes)

    def __post_init__(self):
        if not self.name:
            raise ValueError("name must not be empty")

        if self.type == ParameterType.CONTINUOUS:
            if self.lower is None or self.upper is None:
                raise ValueError(
                    "Continuous parameters require lower and upper bounds"
                )

            if self.lower > self.upper:
                raise ValueError(
                    "lower must not be greater than upper"
                )

        elif self.type == ParameterType.INTEGER:
            if self.lower is None or self.upper is None:
                raise ValueError(
                    "Integer parameters require lower and upper bounds"
                )

            if not float(self.lower).is_integer() or \
                not float(self.upper).is_integer():
                raise ValueError(
                    "Integer parameters require integer bounds"
                )

            if self.lower > self.upper:
                raise ValueError(
                    "lower must not be greater than upper"
                )

        elif self.type == ParameterType.CATEGORICAL:
            if not self.classes:
                raise ValueError(
                    "Categorical parameters require at least one class"
                )

        else:
            raise ValueError(
                f"Unsupported parameter type: {self.type}"
            )

@dataclass
class ParameterSpace:
    parameters: list[Parameter]

    def __post_init__(self):
        if not self.parameters:
            raise ValueError("Parameter space cannot be empty")

        names = [parameter.name for parameter in self.parameters]

        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def lower_bounds(self) -> np.ndarray:
        return np.array(
            [parameter.lower for parameter in self.parameters]
        )

    @property
    def upper_bounds(self) -> np.ndarray:
        return np.array(
            [parameter.upper for parameter in self.parameters]
        )

    def get(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter

        raise KeyError(f"Unknown parameter: {name}")

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self.parameters)