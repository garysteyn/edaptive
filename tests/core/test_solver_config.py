import pytest

from edaptive.core.config import SolverConfig
from edaptive.core.parameters import Parameter, ParameterSpace, ParameterType
from edaptive.utils.experiment_utils import parse_config


def w_space():
    return ParameterSpace(
        [Parameter(name="w", type=ParameterType.CONTINUOUS, lower=0.0, upper=1.0)]
    )


JSON_CONFIG = {
    "identifier": "eda",
    "adapter": {
        "eta": {"tuned": True, "type": "continuous", "lower": 1e-4, "upper": 0.2},
        "copula": {"tuned": False, "type": "categorical", "value": "None"},
    },
    "optimizer": {
        "fixed": {
            "n_s": {"tuned": False, "type": "integer", "value": 30},
            "c_1": {"tuned": True, "type": "continuous", "lower": 0, "upper": 4},
        },
        "adapted": {"w": {"type": "continuous", "lower": 0, "upper": 1}},
    },
}


def test_parse_config_builds_solver_config_with_placeholders():
    config, tuning_space = parse_config(JSON_CONFIG)
    assert config.identifier == "eda"
    assert config.fixed_CPs == {"n_s": 30, "c_1": None}
    assert config.adapter_params == {"eta": None, "copula": "None"}
    assert [p.name for p in config.adapted_CP_space] == ["w"]
    assert [p.name for p in tuning_space] == ["eta", "c_1"]


def test_parse_config_builds_adapted_space_without_adapter_section():
    json_config = {k: v for k, v in JSON_CONFIG.items() if k != "adapter"}
    config, _ = parse_config(json_config)
    assert config.adapter_params == {}
    assert [p.name for p in config.adapted_CP_space] == ["w"]


def test_parse_config_builds_integer_and_categorical_adapted_CPs():
    json_config = {k: v for k, v in JSON_CONFIG.items() if k != "adapter"}
    json_config["optimizer"] = dict(JSON_CONFIG["optimizer"])
    json_config["optimizer"]["adapted"] = {
        "w": {"type": "categorical", "classes": [0.4, 0.7298, 0.9]},
        "c_2": {"type": "integer", "lower": 0, "upper": 4},
    }
    config, _ = parse_config(json_config)
    w, c_2 = config.adapted_CP_space
    assert w.type == ParameterType.CATEGORICAL
    assert w.classes == (0.4, 0.7298, 0.9)
    assert (c_2.type, c_2.lower, c_2.upper) == (ParameterType.INTEGER, 0, 4)


def test_with_values_fills_fixed_CPs_and_adapter_params():
    config, _ = parse_config(JSON_CONFIG)
    filled = config.with_values({"eta": 0.01, "c_1": 1.5})
    assert filled.fixed_CPs == {"n_s": 30, "c_1": 1.5}
    assert filled.adapter_params == {"eta": 0.01, "copula": "None"}
    # The original is unchanged.
    assert config.fixed_CPs["c_1"] is None
    assert config.adapter_params["eta"] is None


def test_with_values_rejects_unknown_name():
    config, _ = parse_config(JSON_CONFIG)
    with pytest.raises(ValueError, match="'rho_min' is neither"):
        config.with_values({"rho_min": 0.1})


def test_with_values_rejects_ambiguous_name():
    config = SolverConfig(fixed_CPs={"x": 1}, adapter_params={"x": 2})
    with pytest.raises(ValueError, match="'x' is both"):
        config.with_values({"x": 3})


def test_rejects_CP_that_is_both_fixed_and_adapted():
    with pytest.raises(ValueError, match=r"\['w'\] are both fixed and adapted"):
        SolverConfig(fixed_CPs={"n_s": 10, "w": 0.5}, adapted_CP_space=w_space())


def test_dict_round_trip():
    config, _ = parse_config(JSON_CONFIG)
    config = config.with_values({"eta": 0.01, "c_1": 1.5})
    assert SolverConfig.from_dict(config.to_dict()) == config


def test_from_dict_reads_best_configuration_pickle_dicts():
    stored = {
        "adapter": {"eta": 0.01, "n_elites": 5},
        "optimizer": {"fixed": {"n_s": 30}, "adapted": w_space()},
        "identifier": "eda",
    }
    config = SolverConfig.from_dict(stored)
    assert config == SolverConfig(
        identifier="eda",
        fixed_CPs={"n_s": 30},
        adapted_CP_space=w_space(),
        adapter_params={"eta": 0.01, "n_elites": 5},
    )
    assert config.to_dict() == stored

    basic = {"optimizer": {"fixed": {"n_s": 30, "w": 0.7}}, "identifier": "basic"}
    assert SolverConfig.from_dict(basic).to_dict() == basic
