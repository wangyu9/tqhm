"""write_Du_2020_option.m -- write the options file for the TLC solver of
[Du et al. 2020].

Key/value pairs on alternating lines, then a `record` and a `save` section. All
defaults are hard-coded in the MATLAB source (the varargin is never read), so
they stay literals here; `%g` formatting matters to the C++ parser and is
reproduced exactly.
"""


def write_Du_2020_option(file, *varargin):
    form = 'Tutte'
    alphaRatio = 1.e-6
    alpha = -1
    ftol_abs = 1.e-8
    ftol_rel = 1.e-8
    xtol_abs = 1.e-8
    xtol_rel = 1.e-8
    gtol_abs = 1.e-8
    algorithm = 'Projected_Newton'
    maxeval = 75   # 10000
    stopCode = 'none'   # 'all_good'

    # 'record'

    vert = 1
    energy = 1
    minArea = 1
    gradient = 0
    gNorm = 0
    searchDirection = 0
    searchNorm = 1
    stepSize = 0
    stepNorm = 0

    # 'save'

    save_vert = 1

    with open(file, 'w') as fileID:
        fileID.write('form\n%s\n' % form)
        fileID.write('alphaRatio\n%g\n' % alphaRatio)
        fileID.write('alpha\n%g\n' % alpha)
        fileID.write('ftol_abs\n%g\n' % ftol_abs)
        fileID.write('ftol_rel\n%g\n' % ftol_rel)
        fileID.write('xtol_abs\n%g\n' % xtol_abs)
        fileID.write('xtol_rel\n%g\n' % xtol_rel)
        fileID.write('gtol_abs\n%g\n' % gtol_abs)
        fileID.write('algorithm\n%s\n' % algorithm)
        fileID.write('maxeval\n%d\n' % maxeval)
        fileID.write('stopCode\n%s\n' % stopCode)

        fileID.write('record\n')

        fileID.write('vert\t%d\n' % vert)
        fileID.write('energy\t%d\n' % energy)
        fileID.write('minArea\t%d\n' % minArea)
        fileID.write('gradient\t%d\n' % gradient)
        fileID.write('gNorm\t%d\n' % gNorm)
        fileID.write('searchDirection\t%d\n' % searchDirection)
        fileID.write('searchNorm\t%d\n' % searchNorm)
        fileID.write('stepSize\t%d\n' % stepSize)
        fileID.write('stepNorm\t%d\n' % stepNorm)

        fileID.write('save\n')

        fileID.write('vert\t%d' % save_vert)
