import json
import pandas as pd
import warnings


def benchmark_json_to_pandas(path):
    data = json.load(open(path))

    return pd.io.json.json_normalize(data=data['benchmarks'])


def compute_speedup(slow_df, fast_df, match_list, stats_param, label_list=None):
    res = []
    for k, v in slow_df.items():
        for _, row in v.iterrows():
            match_dict = {}
            n = v
            c = fast_df[k]
            for m in match_list:
                n = filter_by_value_in_column(n, m, row[m])
                c = filter_by_value_in_column(c, m, row[m])
                match_dict[m] = row[m]

            label_dict = {}
            if label_list is not None:
                for l in label_list:
                    label_dict[l] = row[l]

            if c.shape[0] == 0 or n.shape[0] == 0:
                empty_row = "slow" if n.shape[0] == 0 else "fast"
                message = "row for operation {0} with {1} not found on {2} dataframe, skipping operation".format(
                    k, str(match_dict), empty_row
                )
                warnings.warn(message, RuntimeWarning)
            else:
                n_med = n.iloc[0][stats_param]
                c_med = c.iloc[0][stats_param]

                res.append(pd.DataFrame(
                    {'operation': [k], 'speedup': n_med / c_med, **match_dict, **label_dict}
                ))

    return pd.concat(res, ignore_index=True)


def filter_by_string_in_column(df, col, val):
    return df.loc[df[col].str.contains(val)]


def filter_by_value_in_column(df, col, val):
    return df.loc[df[col] == val]


def significant_round(x, precision):
    r = float(f'%.{precision - 1}e' % x)
    return r if r < 10.0 else round(r)


def split_params_list(df, params_name, columns=None):
    lst = df[params_name].to_list()
    lst = [[l] if not isinstance(l, list) else l for l in lst]
    ncols = max([len(l) for l in lst])
    if columns is None:
        columns = [params_name + '.' + str(i) for i in range(ncols)]
    split_param = pd.DataFrame(lst, columns=columns)
    return df.join(split_param, how='outer')
