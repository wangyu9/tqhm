"""total_lifted_content.m -- run the TLC solver of [Du et al. 2020] out of process.

Shells out to the `findInjective_PN` binary and to the three IO scripts of the
`lifting_simplices_to_find_injectivity` repo; the hard-coded relative paths and
the anaconda python are kept verbatim, so this only works from the MATLAB working
directory with those siblings present. MATLAB's `[status,result] = system(cmd)`
becomes `subprocess.getstatusoutput`.

`writeOBJ` (the plain gptoolbox writer, not writeOBJ_safe) is not part of the
gptoolbox subset mirrored here, so the import is lazy and fails at call time.

MATLAB writes `B-1` to the handle file because the C++ tool wants 0-based
indices; `circular_laplacian` is already 0-based here, so `B` is written as is.
"""

import os
import subprocess
import sys

import numpy as np

from circular_laplacian import circular_laplacian
from write_Du_2020_option import write_Du_2020_option
from readDuMat import readDuMat


def total_lifted_content(V, F, VT, folder):
    folder = str(folder)

    os.makedirs(folder, exist_ok=True)

    folder = os.getcwd() + '/' + folder + '/'

    # folder = folder + '/'

    IO_path = '../lifting_simplices_to_find_injectivity/IO/'

    if sys.platform == 'darwin':
        python = '/Users/yu/opt/anaconda3/bin/python'
    else:
        python = '/home/wangyu/anaconda3/bin/python'

    convert_input_2D = python + ' ' + IO_path + 'convert_input_2D.py'

    extract_boundary_vert = python + ' ' + IO_path + 'extract_boundary_vert.py'

    get_result_mesh = python + ' ' + IO_path + 'get_result_mesh.py'

    input_mesh = folder + '/input.obj'

    handle = folder + '/handles.txt'

    # handle = '../Locally-Injective-Mappings-Benchmark/Simple/square_rot180/handles.txt'

    DuInputFormat = folder + 'DuInputFormat'

    DuOutput = folder + 'DuOutput'

    result_mesh = folder + 'result.obj'

    # writeOBJ_safe(input_mesh, V, F, VT[:, 0:2])
    from writeOBJ import writeOBJ
    writeOBJ(input_mesh, V, F, VT[:, 0:2])

    # if os.path.isfile(folder + '/input-meshlab.obj'):
    #     input_mesh = folder + '/input-meshlab.obj'

    # command0 = extract_boundary_vert + ' ' + input_mesh + ' ' + handle
    #
    # print('%s' % command0)
    # status, result = subprocess.getstatusoutput(command0)

    BE, _, _ = circular_laplacian(V.shape[0], F)
    B = np.sort(BE[:, 0])

    # dlmwrite adds a trailing '\n', which seems not to be a problem
    with open(handle, 'w') as fp:
        for b in B:
            fp.write('%d\n' % b)

    command1 = convert_input_2D + ' ' + input_mesh + ' ' + handle + ' ' + DuInputFormat

    print('%s' % command1)
    status, result = subprocess.getstatusoutput(command1)

    if status != 0:
        raise RuntimeError('')

    print(result)

    # ./extract_boundary_vert.py [inputMeshFile] [outputHandleFile]
    # ./convert_input_2D.py [inputObjFile] [handleFile] [outFile]

    # export fjPN=../Total-Lifted-Content-PN/build/findInjective_PN
    # $fjPN  $WD/DuInputFormat my_PN_solver_options0 $WD/Du-PN-result

    fjPN = '../Total-Lifted-Content-PN/build/findInjective_PN'

    config = folder + '/solver_option'
    # config = '../scripts/my_PN_solver_options0'

    write_Du_2020_option(config)

    command2 = fjPN + ' ' + DuInputFormat + ' ' + config + ' ' + DuOutput

    print('%s' % command2)
    status, result = subprocess.getstatusoutput(command2)

    if status != 0:
        raise RuntimeError('')

    print(result)

    command3 = get_result_mesh + ' ' + DuInputFormat + ' ' + DuOutput + ' ' + result_mesh

    print('%s' % command3)
    status, result = subprocess.getstatusoutput(command3)

    if status != 0:
        raise RuntimeError('')

    print(result)
    # ./get_result_mesh.py [inputFile] [resultFile] [outFile]

    from readOBJ import readOBJ
    VR = readOBJ(result_mesh)[0]

    n = V.shape[0]
    history = []

    VDD = None
    for ii in range(0, 100001):
        iter_file = folder + '/DuOutput_vert_iter' + str(ii)

        if os.path.isfile(iter_file):
            VDD = readDuMat(iter_file, n)

            history.append({'UV': VDD})

        else:
            break

    if False:
        from draw_uv_with_flips2 import draw_uv_with_flips2
        draw_uv_with_flips2(VDD[:, 0], VDD[:, 1], F, None)

    return VR, history
