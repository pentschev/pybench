import json
import pandas as pd


def benchmark_json_to_pandas(path):
    data = json.load(open(path))

    return pd.io.json.json_normalize(data=data["benchmarks"])


def compute_speedup(slow_df, fast_df, operation_list, param_list, stats_param):
    """Computes the speedup of fast operations against slow operations.

    Parameters
    ----------
    slow_df: pandas.DataFrame
        The dataframe containing the baseline for slow operations
    fast_df: pandas.DataFrame
        The dataframe containing the baseline for fast operations
    operations_list: list of str
        A list containing the names of operations for which to compute the
        speedup. pytest-benchmark populates a field ``name`` with the name of a
        pytest and the parameters used in that test, which is unique for all
        parameter combinations of a test. Each pytest function starts with the
        "test_" prefix and an operation name, such as "Sum", forming a name
        "test_Sum". This parameter should get a list of operation names, for
        example:
            ``["FFT", "Standard_Deviation", "Sum"]``
    param_list: list of str
        A list of parameters to identify matching benchmarks on the slow and
        fast dataframes. If a pytest-benchmark includes a parameter ``shape``
        which is composed of a list of 2-dimensional tuples, matching rows of
        the dataframes would have the same value for the two dimensions. The
        format for each element of that list should be
        ``"param_name.dimension``. An example for the case above would look
        like the following:
            ``["params.shape.0", "params.shape.1"]``
    stats_param: str
        The name of the parameter which we would like to use for computing
        speedups. Since pytest-benchmark computes different metrics, such as
        min, max, mean, median, etc., the metric desired has to be set.
        Usually, 'stats.median' is used.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing columns for the name of each operation,
        the parameters used to match slow and fast operations, and the speedup
        of the fast operation over the slow one.

    Example
    -------
    >>> compute_speedup(numpy_df, cupy_df,
    >>>     ["FFT", "Standard_Deviation", "Sum"],
    >>>     ["params.shape.0", "params.shape.1"],
    >>>     "median")
    """
    if not isinstance(operation_list, list):
        raise TypeError
    if not isinstance(param_list, list):
        raise TypeError
    if not isinstance(stats_param, str):
        raise TypeError

    slow_df = slow_df.loc[
        slow_df["name"].str.contains("|".join(operation_list))
    ]
    fast_df = fast_df.loc[
        fast_df["name"].str.contains("|".join(operation_list))
    ]

    slow_df["name"] = slow_df["name"].apply(
        lambda r: r.split("test_")[1].split("[")[0]
    )
    fast_df["name"] = fast_df["name"].apply(
        lambda r: r.split("test_")[1].split("[")[0]
    )

    slow_df = slow_df[["name", *param_list, stats_param]]
    fast_df = fast_df[["name", *param_list, stats_param]]

    speedup_df = slow_df.merge(fast_df, on=["name", *param_list])
    speedup_df["speedup"] = (
        speedup_df[stats_param + "_x"] / speedup_df[stats_param + "_y"]
    )
    speedup_df["speedup"] = speedup_df["speedup"].apply(
        lambda r: 1.0 / -r if r < 1 else r
    )
    speedup_df["speedup"] = speedup_df["speedup"].apply(
        significant_round, precision=2
    )

    speedup_df.drop(columns=[stats_param + "_x", stats_param + "_y"])

    return speedup_df


def filter_by_string_in_column(df, col, val):
    return df.loc[df[col].str.contains(val)]


def significant_round(x, precision):
    r = float(f"%.{precision - 1}e" % x)
    return r if r < 10.0 else round(r)


def split_params_list(df, params_name, columns=None):
    lst = df[params_name].to_list()
    lst = [[l] if not isinstance(l, list) else l for l in lst]
    ncols = max([len(l) for l in lst])
    if columns is None:
        columns = [params_name + "." + str(i) for i in range(ncols)]
    split_param = pd.DataFrame(lst, columns=columns)
    return df.join(split_param, how="outer")
