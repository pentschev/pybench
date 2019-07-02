import json
import pandas as pd


def benchmark_json_to_pandas(path):
    data = json.load(open(path))

    return pd.io.json.json_normalize(data=data['benchmarks'])


def filter_by_string_in_column(df, col, val):
    return df.loc[df[col].str.contains(val)]


def filter_by_value_in_column(df, col, val):
    return df.loc[df[col] == val]


def split_params_list(df, params_name, columns=None):
    lst = df[params_name].to_list()
    lst = [[l] if not isinstance(l, list) else l for l in lst]
    ncols = max([len(l) for l in lst])
    if columns is None:
        columns = [params_name + '.' + str(i) for i in range(ncols)]
    split_param = pd.DataFrame(lst, columns=columns)
    return df.join(split_param, how='outer')
