"""json_reader.m -- read a JSON file into a dict.

MATLAB's `jsondecode` mangles object keys that are not valid identifiers into
struct field names; `json.load` keeps them verbatim, which is closer to the file
and is what the callers (json_to_table*) want.
"""

import json


def json_reader(fname):
    if False:
        JS = json_reader('50lines_records.json')

        # MATLAB: JS.lucy_o_G.time ./ JS.lucy_o_G.LBFGSIternum
        import numpy as np
        np.asarray(JS['lucy_o_G']['time']) / np.asarray(JS['lucy_o_G']['LBFGSIternum'])

    #  fname = 'meta_Electronics.json'

    with open(fname, 'r') as fid:
        str_ = fid.read()
    return json.loads(str_)
