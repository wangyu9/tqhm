"""total_lifted_content_QN.m -- the quasi-Newton sibling of total_lifted_content.

Only the binary (`findInjective` instead of `findInjective_PN`) and the config
file (a checked-in `my_QN_solver_options0` instead of one written by
write_Du_2020_option) differ; it also returns nothing and reads no iteration
history. See total_lifted_content.py for the path/`system` notes.
"""

import os
import subprocess
import sys

import numpy as np

from circular_laplacian import circular_laplacian


def total_lifted_content_QN(V, F, VT, folder):
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

    from writeOBJ import writeOBJ
    writeOBJ(input_mesh, V, F, VT[:, 0:2])

    # command0 = extract_boundary_vert + ' ' + input_mesh + ' ' + handle
    #
    # print('%s' % command0)
    # status, result = subprocess.getstatusoutput(command0)

    BE, _, _ = circular_laplacian(V.shape[0], F)
    B = np.sort(BE[:, 0])

    # dlmwrite adds a trailing '\n', which seems not to be a problem.
    # MATLAB writes B-1 here; circular_laplacian is already 0-based.
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

    # fjPN = '../Total-Lifted-Content-PN/build/findInjective_PN'
    # fjQN and config is the only difference.

    fjQN = '../lifting_simplices_to_find_injectivity/build/findInjective'

    config_QN = '../scripts/my_QN_solver_options0'

    command2 = fjQN + ' ' + DuInputFormat + ' ' + config_QN + ' ' + DuOutput

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
