import pytest

import importlib

from pybench import run_benchmark


@pytest.mark.parametrize('module', ['pandas', 'cudf'])
@pytest.mark.parametrize('data_path', ['/datasets/nyc_taxi/2015/yellow_tripdata_2015-01.csv'])
@pytest.mark.parametrize('nrows', [10000000])
def test_Read_CSV(benchmark, module, data_path, nrows):
    m = importlib.import_module(module)

    compute_func = lambda data: m.read_csv(data['path'], nrows=data['nrows'])
    run_benchmark(benchmark, m, compute_func, lambda data: data, {'path': data_path, 'nrows': nrows})


@pytest.mark.parametrize('module', ['pandas', 'cudf'])
@pytest.mark.parametrize('data_path',
        [('/datasets/nyc_taxi/2015/yellow_tripdata_2015-01.csv',
          '/datasets/nyc_taxi/2015/yellow_tripdata_2015-02.csv')]
    )
@pytest.mark.parametrize('nrows', [50000])
def test_Merge_DataFrames(benchmark, module, data_path, nrows):
    m = importlib.import_module(module)

    data_func = lambda data: [m.read_csv(p, nrows=data['nrows']) for p in data['path']]
    compute_func = lambda data: data[0].merge(data[1], on='trip_distance')
    run_benchmark(benchmark, m, compute_func, data_func, {'path': data_path, 'nrows': nrows})
