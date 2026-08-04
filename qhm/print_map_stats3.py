"""print_map_stats3.m -- one-line summary of a compute_map_stats3 result."""


def print_map_stats3(stats):
    print('#flips=%d, sigma_ratio: max=%g, mean=%g, MIPS-2=%g' % (
        stats['num_flips'],
        float(stats['sigma_ratio'].max()),
        float(stats['sigma_ratio'].mean()),
        float(stats['wavg_MIPS'].sum()) - 2,   # wavg_MIPS
    ))
