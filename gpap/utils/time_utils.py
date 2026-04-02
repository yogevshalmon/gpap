import time

# function that time the execution of a function
def time_it(func, *args):
    start = time.time()
    res = func(*args)
    end = time.time()
    print(f"Execution time of {func.__name__}: {end - start} seconds")
    return res
