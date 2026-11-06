from dataclasses import dataclass, field, replace

from edaptive.core.parameters import ParameterSpace


@dataclass(frozen=True)
class SolverConfig:
    """Everything a solver needs besides the problem and the seed.

    - ``fixed_CPs``: optimizer keyword arguments that stay fixed for a run (e.g.
      ``n_s``; ``w``/``c_1``/``c_2`` when they aren't adapted).
    - ``adapted_CP_space``: the adaptable CPs this solver adapts, with their bounds.
    - ``adapter_params``: the adapter's own parameters (e.g. ``eta``, ``n_elites``).

    Entries that are tuned hold ``None`` until ``with_values`` fills them in. Treat
    the dicts as read-only; ``with_values`` returns a new config.
    """

    identifier: str | None = None
    fixed_CPs: dict = field(default_factory=dict)
    adapted_CP_space: ParameterSpace | None = None
    adapter_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.adapted_CP_space is not None:
            both = [p.name for p in self.adapted_CP_space if p.name in self.fixed_CPs]
            if both:
                raise ValueError(f"CPs {both} are both fixed and adapted")

    def with_values(self, values):
        """Return a copy with the named entries of ``fixed_CPs`` / ``adapter_params``
        replaced by ``values``."""
        fixed_CPs = dict(self.fixed_CPs)
        adapter_params = dict(self.adapter_params)
        for name, value in values.items():
            in_fixed = name in fixed_CPs
            in_adapter = name in adapter_params
            if in_fixed and in_adapter:
                raise ValueError(
                    f"{name!r} is both a fixed CP and an adapter parameter"
                )
            if in_fixed:
                fixed_CPs[name] = value
            elif in_adapter:
                adapter_params[name] = value
            else:
                raise ValueError(
                    f"{name!r} is neither a fixed CP nor an adapter parameter"
                )
        return replace(self, fixed_CPs=fixed_CPs, adapter_params=adapter_params)

    def to_dict(self):
        """The plain-dict form stored in ``best_configuration.pkl``."""
        optimizer = {"fixed": dict(self.fixed_CPs)}
        if self.adapted_CP_space is not None:
            optimizer["adapted"] = self.adapted_CP_space
        config = {"identifier": self.identifier, "optimizer": optimizer}
        if self.adapter_params:
            config["adapter"] = dict(self.adapter_params)
        return config

    @classmethod
    def from_dict(cls, config):
        """Inverse of ``to_dict``; also reads ``best_configuration.pkl`` files."""
        optimizer = config["optimizer"]
        return cls(
            identifier=config.get("identifier"),
            fixed_CPs=dict(optimizer.get("fixed", {})),
            adapted_CP_space=optimizer.get("adapted"),
            adapter_params=dict(config.get("adapter") or {}),
        )
