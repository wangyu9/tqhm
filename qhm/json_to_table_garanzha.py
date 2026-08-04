"""json_to_table_garanzha.m -- json_to_table with the columns renamed for the
[Garanzha et al. 2021] comparison table."""

import numpy as np


def json_to_table_garanzha(JS):
    import pandas as pd

    fn = list(JS.keys())

    variable_type = ['string', 'double', 'int64']
    variable_name = ['name', 'garanzha_time', 'garanzha_flips']

    table_results = pd.DataFrame({
        'name': pd.Series([''] * len(fn), dtype=object),
        'garanzha_time': pd.Series(np.zeros(len(fn)), dtype='float64'),
        'garanzha_flips': pd.Series(np.zeros(len(fn), dtype=np.int64), dtype='int64'),
    }, columns=variable_name)

    for k in range(len(fn)):
        name = fn[k]

        table_results.at[k, 'garanzha_time'] = float(np.sum(JS[name]['time']))

        table_results.at[k, 'name'] = name

        table_results.at[k, 'garanzha_flips'] = np.int64(
            np.ravel(JS[name]['nInverted'])[-1])

    return table_results
