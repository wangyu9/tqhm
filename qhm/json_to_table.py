"""json_to_table.m -- one row per mesh in a records JSON: total time and the
final flip count.

MATLAB `table` -> pandas DataFrame with the same column names, order and types
("string"/"double"/"int64" -> object/float64/int64). Row order follows the JSON
key order, matching MATLAB's `fieldnames`.

pandas is imported lazily so the solver modules never pull it in.
"""

import numpy as np


def json_to_table(JS):
    import pandas as pd

    fn = list(JS.keys())

    variable_type = ['string', 'double', 'int64']
    variable_name = ['name', 'time', 'flips']

    table_results = pd.DataFrame({
        'name': pd.Series([''] * len(fn), dtype=object),
        'time': pd.Series(np.zeros(len(fn)), dtype='float64'),
        'flips': pd.Series(np.zeros(len(fn), dtype=np.int64), dtype='int64'),
    }, columns=variable_name)

    for k in range(len(fn)):
        name = fn[k]

        table_results.at[k, 'time'] = float(np.sum(JS[name]['time']))

        table_results.at[k, 'name'] = name

        table_results.at[k, 'flips'] = np.int64(np.ravel(JS[name]['nInverted'])[-1])

    return table_results
