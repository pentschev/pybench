import pytest

import importlib

from pybench import run_benchmark


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_fft(benchmark, module, shape):
    m = importlib.import_module(module)

    data_func = lambda shape: m.exp(2j * m.pi * m.random.random(shape))
    run_benchmark(benchmark, m, m.fft.fft, data_func, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_sum(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.sum, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_std(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.std, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_elementwise(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: m.sin(data)**2 + m.cos(data)**2
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_dot(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data.dot(data)
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000)])
def test_slicing(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data[::3]
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(500, 500), (1000, 1000)])
def test_svd(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.linalg.svd, m.random.random, shape)
