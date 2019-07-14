import pytest

import importlib


def run_sync(m, func, *args):
    res = func(*args)
    if m.__name__ == "cupy":
        m.cuda.Device().synchronize()
    return res


def gen_data_warmup(m, compute_func, data_func, shape):
    data = run_sync(m, data_func, shape)
    run_sync(m, compute_func, data)
    return data


def run_benchmark(benchmark, m, compute_func, data_func, shape):
    data = gen_data_warmup(m, compute_func, data_func, shape)

    return benchmark.pedantic(run_sync, args=(m, compute_func, data), rounds=5)
