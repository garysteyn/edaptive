"""Unit tests for ``Parameter`` validation (``core/parameters.py``)."""

import pytest

from edaptive.core.parameters import Parameter, ParameterType


@pytest.mark.parametrize(
    "param_type", [ParameterType.CONTINUOUS, ParameterType.INTEGER]
)
@pytest.mark.parametrize("lower", [0, -1])
def test_log_scale_requires_positive_lower_bound(param_type, lower):
    with pytest.raises(ValueError, match="positive lower bound"):
        Parameter("p", param_type, lower=lower, upper=10, log_scale_tuning=True)


@pytest.mark.parametrize(
    "param_type", [ParameterType.CONTINUOUS, ParameterType.INTEGER]
)
def test_log_scale_with_positive_lower_bound_is_accepted(param_type):
    param = Parameter("p", param_type, lower=1, upper=10, log_scale_tuning=True)
    assert param.log_scale_tuning


def test_non_positive_lower_bound_is_fine_without_log_scale():
    Parameter("p", ParameterType.CONTINUOUS, lower=-1.0, upper=1.0)
