from functools import wraps
from time import perf_counter

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start

        # Store timing on the object
        if args and hasattr(args[0], "_timings"):
            args[0]._timings[func.__name__] = (
                args[0]._timings.get(func.__name__, 0.0) + elapsed
            )

        return result

    return wrapper