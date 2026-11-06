from functools import wraps
from time import perf_counter, process_time

CLOCKS = ("wall", "cpu")


def timed(func):
    """Accumulate the wall-clock and CPU time of each call of a method.

    Times are added to ``self._timings[func.__name__] = {"wall": s, "cpu": s}``
    (seconds; CPU time of the whole process). Time only methods that don't call
    other timed methods, so no time is counted twice.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        wall_start = perf_counter()
        cpu_start = process_time()
        result = func(self, *args, **kwargs)
        cpu = process_time() - cpu_start
        wall = perf_counter() - wall_start

        totals = self._timings.setdefault(func.__name__, {"wall": 0.0, "cpu": 0.0})
        totals["wall"] += wall
        totals["cpu"] += cpu

        return result

    return wrapper


def core_time(timings):
    """Total ``{"wall": s, "cpu": s}`` of a run's ``history["timings"]``, summed
    over every timed method of every component."""
    totals = {clock: 0.0 for clock in CLOCKS}
    for methods in timings.values():
        for times in methods.values():
            for clock in CLOCKS:
                totals[clock] += times[clock]
    return totals
