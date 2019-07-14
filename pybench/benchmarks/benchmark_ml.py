import pytest

import importlib
import numba, numba.cuda
import numpy as np

from pybench import run_benchmark

_shapes = {
    "small": [(int(2 ** 14), 512), (int(2 ** 15), 512), (int(2 ** 16), 512)],
    "large": [(int(2 ** 20), 512), (int(2 ** 21), 512), (int(2 ** 22), 512)],
}


def load_data(nrows, ncols, cached, train_split=1.0, label_col=None):
    import gzip
    import os
    import numpy as np, gzip, os
    import pandas as pd

    train_rows = int(nrows * train_split)

    if os.path.exists(cached):
        with gzip.open(cached) as f:
            X = np.load(f)

        if train_split < 1.0 and label_col is not None:
            X = X[:, [i for i in range(X.shape[1]) if i != label_col]]
            y = X[:, label_col : label_col + 1]
            rindices = np.random.randint(0, X.shape[0] - 1, nrows)
            X = X[rindices, :ncols]
            y = y[rindices]
            df_y_train = pd.DataFrame(
                {"fea%d" % i: y[0:train_rows, i] for i in range(y.shape[1])}
            )
            df_y_test = pd.DataFrame(
                {"fea%d" % i: y[train_rows:, i] for i in range(y.shape[1])}
            )
        else:
            X = X[np.random.randint(0, X.shape[0] - 1, nrows), :ncols]

    else:
        # throws FileNotFoundError error if mortgage dataset is not present
        raise FileNotFoundError(
            "Please download the required dataset or check the path"
        )

    if train_split < 1.0 and label_col is not None:
        df_X_train = pd.DataFrame(
            {"fea%d" % i: X[0:train_rows, i] for i in range(X.shape[1])}
        )
        df_X_test = pd.DataFrame(
            {"fea%d" % i: X[train_rows:, i] for i in range(X.shape[1])}
        )

        return {
            "X_train": df_X_train,
            "X_test": df_X_test,
            "y_train": df_y_train,
            "y_test": df_y_test,
        }
    else:
        df = pd.DataFrame({"fea%d" % i: X[:, i] for i in range(X.shape[1])})
        return df


def load_mortgage(d):
    kwargs = {"nrows": d["shape"][0], "ncols": d["shape"][1], "cached": d["data"]}

    if "train_split" in d:
        kwargs["train_split"] = d["train_split"]
    if "label_col" in d:
        kwargs["label_col"] = d["label_col"]

    data = load_data(**kwargs)

    if d["module"] == "cuml":
        import cudf

        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = cudf.DataFrame.from_pandas(v)

            data["y_train"] = cudf.Series(data["y_train"]["fea0"])
        else:
            data = cudf.DataFrame.from_pandas(data)

    return {"module": d["module"], "data": data}


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_PCA(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.decomposition")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {
            "n_components": 10,
            "whiten": False,
            "random_state": 42,
            "svd_solver": "full",
        }

        pca = m.PCA(**kwargs)

        pca.fit_transform(data["data"])

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {"module": module, "shape": shape, "data": data},
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["small"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_DBSCAN(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.cluster")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {"eps": 3, "min_samples": 2}

        if data["module"] == "sklearn":
            kwargs["n_jobs"] = -1
            kwargs["algorithm"] = "brute"

        dbscan = m.DBSCAN(**kwargs)

        dbscan.fit(data["data"])

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {"module": module, "shape": shape, "data": data},
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_TSVD(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.decomposition")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {"n_components": 10, "random_state": 42}

        if data["module"] == "sklearn":
            kwargs["algorithm"] = "arpack"
        elif data["module"] == "cuml":
            kwargs["algorithm"] = "full"

        tsvd = m.TruncatedSVD(**kwargs)

        tsvd.fit_transform(data["data"])

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {"module": module, "shape": shape, "data": data},
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["small"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_KNN(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.neighbors")
    else:
        m = importlib.import_module("cuml.neighbors.nearest_neighbors")

    def compute_func(data):
        kwargs = {}

        n_neighbors = 10

        if data["module"] == "sklearn":
            kwargs["metric"] = "sqeuclidean"
            kwargs["n_jobs"] = -1

        knn = m.NearestNeighbors(**kwargs)

        knn.fit(data["data"])

        knn.kneighbors(data["data"], n_neighbors)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {"module": module, "shape": shape, "data": data},
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_SGD(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.linear_model")
    else:
        m = importlib.import_module("cuml.solvers")

    def compute_func(data):
        kwargs = {
            "learning_rate": "adaptive",
            "eta0": 0.07,
            "penalty": "elasticnet",
            "loss": "squared_loss",
            "tol": 0.0,
        }

        if data["module"] == "sklearn":
            kwargs["max_iter"] = 10
            kwargs["fit_intercept"] = True

            sgd = m.SGDRegressor(**kwargs)

        elif data["module"] == "cuml":
            kwargs["epochs"] = 10
            kwargs["batch_size"] = 512

            sgd = m.SGD(**kwargs)

        X_train = data["data"]["X_train"]
        y_train = data["data"]["y_train"]

        sgd.fit(X_train, y_train)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {
            "module": module,
            "shape": shape,
            "data": data,
            "train_split": 0.8,
            "label_col": 4,
        },
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_LinearRegression(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.linear_model")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {"fit_intercept": True, "normalize": True}

        if data["module"] == "cuml":
            kwargs["algorithm"] = "eig"

        X_train = data["data"]["X_train"]
        y_train = data["data"]["y_train"]

        lr = m.LinearRegression(**kwargs)

        lr.fit(X_train, y_train)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {
            "module": module,
            "shape": shape,
            "data": data,
            "train_split": 0.8,
            "label_col": 4,
        },
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_Ridge(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.linear_model")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {"fit_intercept": False, "normalize": True, "alpha": 0.1}

        if data["module"] == "cuml":
            kwargs["solver"] = "svd"

        X_train = data["data"]["X_train"]
        y_train = data["data"]["y_train"]

        ridge = m.Ridge(**kwargs)

        ridge.fit(X_train, y_train)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {
            "module": module,
            "shape": shape,
            "data": data,
            "train_split": 0.8,
            "label_col": 4,
        },
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_Lasso(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.linear_model")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {
            "alpha": np.array([0.001]),
            "fit_intercept": True,
            "normalize": False,
            "max_iter": 1000,
            "selection": "cyclic",
            "tol": 1e-10,
        }

        X_train = data["data"]["X_train"]
        y_train = data["data"]["y_train"]

        lasso = m.Lasso(**kwargs)

        lasso.fit(X_train, y_train)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {
            "module": module,
            "shape": shape,
            "data": data,
            "train_split": 0.8,
            "label_col": 4,
        },
    )


@pytest.mark.parametrize("module", ["sklearn", "cuml"])
@pytest.mark.parametrize("shape", _shapes["large"])
@pytest.mark.parametrize("data", ["data/mortgage.npy.gz"])
def test_ElasticNet(benchmark, module, shape, data):
    if module == "sklearn":
        m = importlib.import_module("sklearn.linear_model")
    else:
        m = importlib.import_module("cuml")

    def compute_func(data):
        kwargs = {
            "alpha": np.array([0.001]),
            "fit_intercept": True,
            "normalize": False,
            "max_iter": 1000,
            "selection": "cyclic",
            "tol": 1e-10,
        }

        X_train = data["data"]["X_train"]
        y_train = data["data"]["y_train"]

        lasso = m.Lasso(**kwargs)

        lasso.fit(X_train, y_train)

    run_benchmark(
        benchmark,
        m,
        compute_func,
        load_mortgage,
        {
            "module": module,
            "shape": shape,
            "data": data,
            "train_split": 0.8,
            "label_col": 4,
        },
    )
