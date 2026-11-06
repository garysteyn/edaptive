import numpy as np


class BaseSolver:
    """Runs an optimizer and, optionally, adapts its CPs during the run.

    Every solver is built the same way::

        Solver(optimizer_type, adapter_type, problem=..., config=..., rnd_seed=...)

    One ``np.random.Generator`` is created from ``rnd_seed`` and shared by the
    adapter, the solver and the optimizer. Construction order (and so the RNG
    consumption order) is: build the adapter, sample the initial adapted CPs, build
    the optimizer.

    Subclasses describe which types they accept and override the hooks
    ``initial_adapted_CPs`` and ``adapt``:

    - ``fixed_optimizer_type`` / ``fixed_adapter_type``: the solver always uses this
      type; callers may pass ``None`` or the same type.
    - ``uses_adapter``: whether the solver builds an adapter at all.
    """

    fixed_optimizer_type = None
    fixed_adapter_type = None
    uses_adapter = False

    def __init__(
        self, optimizer_type=None, adapter_type=None, *, problem, config, rnd_seed=0
    ):
        self.optimizer_type = self._resolve_optimizer_type(optimizer_type)
        self.adapter_type = self._resolve_adapter_type(adapter_type)

        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(rnd_seed)

        self.adapter = self._build_adapter()
        self.optimizer = self._build_optimizer(self.initial_adapted_CPs())

    @property
    def n_s(self):
        return self.config.fixed_CPs["n_s"]

    @classmethod
    def _resolve_optimizer_type(cls, optimizer_type):
        fixed = cls.fixed_optimizer_type
        if fixed is not None:
            if optimizer_type not in (None, fixed):
                raise ValueError(
                    f"{cls.__name__} always uses {fixed.__name__}, "
                    f"not {optimizer_type.__name__}"
                )
            return fixed
        if optimizer_type is None:
            raise ValueError(f"{cls.__name__} needs an optimizer_type")
        return optimizer_type

    @classmethod
    def _resolve_adapter_type(cls, adapter_type):
        if not cls.uses_adapter:
            if adapter_type is not None:
                raise ValueError(
                    f"{cls.__name__} takes no adapter, got {adapter_type.__name__}"
                )
            return None
        fixed = cls.fixed_adapter_type
        if fixed is not None:
            if adapter_type not in (None, fixed):
                raise ValueError(
                    f"{cls.__name__} always uses {fixed.__name__}, "
                    f"not {adapter_type.__name__}"
                )
            return fixed
        if adapter_type is None:
            raise ValueError(f"{cls.__name__} needs an adapter_type")
        return adapter_type

    def _build_adapter(self):
        if self.adapter_type is None:
            return None
        if self.config.adapted_CP_space is None:
            raise ValueError(
                f"{type(self).__name__} needs an adapted CP space in its config"
            )
        return self.adapter_type(
            CP_space=self.config.adapted_CP_space,
            **(self.config.adapter_params | {"rng": self.rng}),
        )

    def initial_adapted_CPs(self):
        """Initial values of the adapted CPs, ``{name: array of length n_s}``.

        By default the adapter samples them; without an adapter there are none.
        """
        if self.adapter is None:
            return {}
        samples = self.adapter.sample(N=self.n_s)
        return {name: samples[:, idx] for idx, name in enumerate(self.adapter.CP_names)}

    def _build_optimizer(self, initial_adapted_CPs):
        return self.optimizer_type(
            **(
                {"problem": self.problem}
                | {"rng": self.rng}
                | self.config.fixed_CPs
                | initial_adapted_CPs
            )
        )

    def adapt(self, t):
        """Called after every ``optimizer.step(t)``. Does nothing by default."""

    def run(self, max_iterations=5000):
        optimizer = self.optimizer

        for t in range(max_iterations):
            optimizer.step(t)
            self.adapt(t)

        optimizer.history["timings"] = optimizer._timings
        return (np.min(optimizer.history["f_best"]), optimizer.history)
