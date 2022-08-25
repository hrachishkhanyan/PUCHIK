from time import perf_counter
from functools import wraps


def timer(f):
    if callable(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            res = f(*args, **kwargs)
            end = perf_counter()
            print(f'Execution time of func {f.__name__}: {end - start} s')
            return res

        return wrapper

    raise TypeError(f'{f} if not callable.')
