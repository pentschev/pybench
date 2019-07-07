import pytest

import importlib
import numba, numba.cuda

from pybench import run_benchmark


def load_data(nrows, ncols, cached):
    import gzip
    import os
    import numpy as np, gzip, os
    import pandas as pd

    if os.path.exists(cached):
        print('use mortgage data')
        with gzip.open(cached) as f:
            X = np.load(f)
        X = X[np.random.randint(0,X.shape[0]-1,nrows),:ncols]
    else:
        # throws FileNotFoundError error if mortgage dataset is not present
        raise FileNotFoundError('Please download the required dataset or check the path')
    df = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])})
    return df


def load_mortgage(d):
    data = load_data(d['shape'][0], d['shape'][1], d['data'])
    if d['module'] == 'cuml':
        import cudf
        data = cudf.DataFrame.from_pandas(data)
    return {'module': d['module'], 'data': data}


@pytest.mark.parametrize('module', ['sklearn', 'cuml'])
@pytest.mark.parametrize('shape', [(int(2**14), 512), (int(2 ** 15), 512), (int(2 ** 16), 512)])
@pytest.mark.parametrize('data', ['data/mortgage.npy.gz'])
def test_PCA(benchmark, module, shape, data):
    if module == 'sklearn':
        m = importlib.import_module('sklearn.decomposition')
    else:
        m = importlib.import_module('cuml')

    def compute_func(data):
        kwargs = {
            'n_components': 10,
            'whiten': False,
            'random_state': 42,
            'svd_solver': 'full',
        }

        pca = m.PCA(**kwargs)

        pca.fit_transform(data['data'])

    run_benchmark(benchmark, m, compute_func, load_mortgage, {'module': module, 'shape': shape, 'data': data})


@pytest.mark.parametrize('module', ['sklearn', 'cuml'])
@pytest.mark.parametrize('shape', [(int(2**14), 512), (int(2 ** 15), 512), (int(2 ** 16), 512)])
@pytest.mark.parametrize('data', ['data/mortgage.npy.gz'])
def test_DBSCAN(benchmark, module, shape, data):
    if module == 'sklearn':
        m = importlib.import_module('sklearn.cluster')
    else:
        m = importlib.import_module('cuml')

    def compute_func(data):
        kwargs = {
            'eps': 3,
            'min_samples': 2,
        }

        if data['module'] == 'sklearn':
            kwargs['n_jobs'] = -1

        dbscan = m.DBSCAN(**kwargs)

        dbscan.fit(data['data'])

    run_benchmark(benchmark, m, compute_func, load_mortgage, {'module': module, 'shape': shape, 'data': data})


@pytest.mark.parametrize('module', ['sklearn', 'cuml'])
@pytest.mark.parametrize('shape', [(int(2**14), 512), (int(2 ** 15), 512), (int(2 ** 16), 512)])
@pytest.mark.parametrize('data', ['data/mortgage.npy.gz'])
def test_TSVD(benchmark, module, shape, data):
    if module == 'sklearn':
        m = importlib.import_module('sklearn.decomposition')
    else:
        m = importlib.import_module('cuml')

    def compute_func(data):
        kwargs = {
            'n_components': 10,
            'random_state': 42,
        }

        if data['module'] == 'sklearn':
            kwargs['algorithm'] = 'arpack'
        elif data['module'] == 'cuml':
            kwargs['algorithm'] = 'full'

        tsvd = m.TruncatedSVD(**kwargs)

        tsvd.fit_transform(data['data'])

    run_benchmark(benchmark, m, compute_func, load_mortgage, {'module': module, 'shape': shape, 'data': data})
