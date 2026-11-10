from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Adapts the CPs in ``CP_space`` (a subset of an optimizer's adaptable CPs).

    How an adapter changes its state is up to the subclass. Solvers drive it
    through:

    - ``required_data()``: names of optimizer attributes the adapter reads;
    - ``check_update``: decides whether an update should occur in this iteration.
      Solvers call it with the optimizer attributes named by ``required_data()``
      as keyword arguments. It returns the input for ``update`` if an update
      should occur, otherwise ``None``;
    - ``update``: changes the adapter's state. Solvers call it with the output of
      ``check_update``.

    The defaults read no data and never update, so subclasses that update override
    both ``check_update`` and ``update`` with the arguments they need.
    """

    def __init__(self, CP_space, **kwargs):
        self.CP_space = CP_space
        self.CP_lbs = CP_space.lower_bounds
        self.CP_ubs = CP_space.upper_bounds
        self.n_CPs = CP_space.dimension
        self.CP_names = tuple(parameter.name for parameter in CP_space)

        self._timings = {}

        self.initialize(**kwargs)

    def select_own_CPs(self, adaptable_CPs, adaptable_CP_names):
        """Columns of an optimizer's ``adaptable_CPs`` for this adapter's CPs, in
        ``CP_space`` order."""
        columns = [adaptable_CP_names.index(name) for name in self.CP_names]
        return adaptable_CPs[:, columns]

    def required_data(self):
        """Names of optimizer attributes this adapter needs for an update."""
        return []

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    def check_update(self):
        """Decide whether an update should occur in this iteration.

        Subclasses define the criteria (e.g. an update frequency, or a condition on
        the optimizer's state) and take the optimizer attributes named by
        ``required_data()`` as keyword arguments. Returns the input for ``update``
        if an update should occur, otherwise ``None``.
        """
        return None

    def update(self):
        """Change the adapter's state.

        Subclasses take the output of ``check_update`` as their argument.
        """
        return None


class EDAAdapter(BaseAdapter):
    """Samples CPs from a probability distribution.

    Adapters that learn the distribution return elite CPs from ``check_update`` and
    fit their model to them in ``update``.
    """

    def start(self, **data):
        """Record the optimizer's initial state. Does nothing by default.

        ``EDASolver`` calls it once after the optimizer is built, with the optimizer
        attributes named by ``required_data()`` as keyword arguments, so a model can
        take the initial population into account.
        """
        return None

    @abstractmethod
    def sample(self, N):
        """``N`` samples of this adapter's CPs, shape ``(N, n_CPs)``."""
