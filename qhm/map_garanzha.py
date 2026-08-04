"""map_garanzha.m -- write the input mesh for the untangle2d binary of
[Garanzha et al. 2021] and print the command line that would run it.

The `system` call is commented out in the MATLAB source, so this only writes the
OBJ and prints; the hard-coded absolute paths are kept verbatim.

`writeOBJ` (the plain gptoolbox writer, not writeOBJ_safe) is not part of the
gptoolbox subset mirrored here, so the import is lazy and fails at call time.

See also: total_lifted_content.
"""


def map_garanzha(V, F, VT, folder, args):
    folder = str(folder)

    # mkdir(folder)

    input_mesh = folder + '/input.obj'

    from writeOBJ import writeOBJ
    writeOBJ(input_mesh, V, F, VT[:, 0:2])

    # linux
    # myPath = os.environ['PATH']
    myPath = ('/home/wangyu/workspace/tools/ws_moveit2/install/moveit_core/bin:'
              '/opt/ros/humble/bin:/home/wangyu/workspace/tools/anaconda3/bin:'
              '/home/wangyu/workspace/tools/anaconda3/condabin:'
              '/home/wangyu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:'
              '/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin')
    # os.system('export PATH=' + myPath + ' ; which gcc')

    cpath = ('export PATH=' + myPath + ' ; '
             '~/workspace/projects/qhm/invertible-maps/cpp/build/untangle2d '
             '../input.obj --save_iter 1 ' + str(args))

    cpath = ('~/workspace/projects/qhm/invertible-maps/cpp/build/untangle2d '
             '../input.obj --save_iter 1 ' + str(args))

    command1 = cpath + ' '

    print('%s' % command1)
    # status, result = subprocess.getstatusoutput(command1)
    #
    # if status != 0:
    #     raise RuntimeError('')
    # print(result)
