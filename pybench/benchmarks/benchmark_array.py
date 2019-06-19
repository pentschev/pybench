import pytest

import importlib
import numba, numba.cuda

from pybench import run_benchmark


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_fft(benchmark, module, shape):
    m = importlib.import_module(module)

    data_func = lambda shape: m.exp(2j * m.pi * m.random.random(shape))
    run_benchmark(benchmark, m, m.fft.fft, data_func, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_sum(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.sum, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_std(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.std, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_elementwise(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: m.sin(data)**2 + m.cos(data)**2
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_dot(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data.dot(data)
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_slicing(benchmark, module, shape):
    m = importlib.import_module(module)

    compute_func = lambda data: data[::3]
    run_benchmark(benchmark, m, compute_func, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 1000), (20000, 1000)])
def test_svd(benchmark, module, shape):
    m = importlib.import_module(module)

    run_benchmark(benchmark, m, m.linalg.svd, m.random.random, shape)


@pytest.mark.parametrize('module', ['numpy', 'cupy'])
@pytest.mark.parametrize('shape', [(1000, 1000), (10000, 10000), (20000, 20000)])
def test_stencil(benchmark, module, shape):

    m = importlib.import_module(module)
    @numba.stencil
    def _smooth(x):
        return (x[-1, -1] + x[-1, 0] + x[-1, 1] +
                x[ 0, -1] + x[ 0, 0] + x[ 0, 1] +
                x[ 1, -1] + x[ 1, 0] + x[ 1, 1]) // 9

    @numba.njit
    def smooth_cpu(x, out):
        out = _smooth(x)

    @numba.cuda.jit
    def _smooth_gpu(x, out):
        i, j = numba.cuda.grid(2)
        n, m = x.shape
        if 1 <= i < n - 1 and 1 <= j < m - 1:
            out[i, j] = (x[i - 1, j - 1] + x[i - 1, j] + x[i - 1, j + 1] +
                         x[i    , j - 1] + x[i    , j] + x[i    , j + 1] +
                         x[i + 1, j - 1] + x[i + 1, j] + x[i + 1, j + 1]) // 9

    def smooth_gpu(x, out):
        import math

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(x.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(x.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        _smooth_gpu[blockspergrid, threadsperblock](x, out)


    data_func = lambda shape: {'in': m.ones(shape, dtype='int8'), 'out': m.zeros(shape, dtype='int8')}
    f = smooth_cpu if module == 'numpy' else smooth_gpu
    compute_func = lambda data: f(data['in'], data['out'])

    run_benchmark(benchmark, m, compute_func, data_func, shape)
