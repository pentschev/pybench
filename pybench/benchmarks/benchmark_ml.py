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


@pytest.mark.parametrize('module', ['sklearn', 'cuml'])
@pytest.mark.parametrize('shape', [(int(2 ** 15 * 1.5), 400)])
@pytest.mark.parametrize('data', ['data/mortgage.npy.gz'])
def test_PCA(benchmark, module, shape, data):
    if module == 'sklearn':
        m = importlib.import_module('sklearn.decomposition')
    else:
        m = importlib.import_module('cuml')

    def compute_func(data):
        n_components = 10
        whiten = False
        random_state = 42
        svd_solver = 'full'

        pca = m.PCA(n_components=n_components,svd_solver=svd_solver,
                       whiten=whiten, random_state=random_state)

        return pca.fit_transform(data['data'])

    def data_func(d):
        data = load_data(d['shape'][0], d['shape'][1], d['data'])
        if module == 'cuml':
            import cudf
            data = cudf.DataFrame.from_pandas(data)
        return {'module': module, 'data': data}

    run_benchmark(benchmark, m, compute_func, data_func, {'shape': shape, 'data': data})
