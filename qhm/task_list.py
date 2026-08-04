"""task_list.m -- a scratch script recording "what is in result 2".

Its single statement runs `figure_compare_tensor_para`, which does not exist
anywhere in the MATLAB source tree either, so the call is kept and fails on
import at call time.
"""


def task_list():
    # what is in result 2
    # --- figure_compare_tensor_para ---
    from figure_compare_tensor_para import figure_compare_tensor_para
    figure_compare_tensor_para()
