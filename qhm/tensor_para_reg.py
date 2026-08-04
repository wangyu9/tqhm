"""tensor_para_reg.m -- the (unfinished) regularizer factory for the tensor
parameterizations.

The MATLAB source is a stub: all four handles are the constant 0 and neither
`dim` nor `para_type` is read.  `Area = FA` is the last line and is assigned to a
local that is never returned, so it too is dead; it is kept as a comment-free
statement to stay line-for-line with the .m.

The live regularizer is reg_value_grad.py; tensor_para.py sets
`tp.reg_value`/`tp.reg_grad` to the same zero lambdas directly and never calls
this.
"""


def tensor_para_reg(FA, dim, para_type):
    # regularizer in at
    reg_value = lambda at: 0
    reg_grad = lambda at: 0

    # regularizer in au
    rau_value = lambda au: 0
    rau_grad = lambda au: 0

    Area = FA   # noqa: F841 -- dead in the MATLAB source too

    return reg_value, reg_grad, rau_value, rau_grad
