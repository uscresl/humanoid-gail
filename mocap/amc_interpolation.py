import numpy as np
from scipy.interpolate import CubicSpline
import sys


def interpolate(amc_file, output_file, src_framerate, dst_framerate):
    root = np.empty((0, 6))
    lowerback = np.empty((0, 3))
    upperback = np.empty((0, 3))
    thorax = np.empty((0, 3))
    lowerneck = np.empty((0, 3))
    upperneck = np.empty((0, 3))
    head = np.empty((0, 3))
    rclavicle = np.empty((0, 2))
    rhumerus = np.empty((0, 3))
    rradius = np.empty((0, 1))
    rwrist = np.empty((0, 1))
    rhand = np.empty((0, 2))
    rfingers = np.empty((0, 1))
    rthumb = np.empty((0, 2))
    lclavicle = np.empty((0, 2))
    lhumerus = np.empty((0, 3))
    lradius = np.empty((0, 1))
    lwrist = np.empty((0, 1))
    lhand = np.empty((0, 2))
    lfingers = np.empty((0, 1))
    lthumb = np.empty((0, 2))
    rfemur = np.empty((0, 3))
    rtibia = np.empty((0, 1))
    rfoot = np.empty((0, 2))
    rtoes = np.empty((0, 1))
    lfemur = np.empty((0, 3))
    ltibia = np.empty((0, 1))
    lfoot = np.empty((0, 2))
    ltoes = np.empty((0, 1))

    f1 = open(amc_file, 'r')
    f2 = open(output_file, 'w')

    joint_name = ['root'
        , 'lowerback'
        , 'upperback'
        , 'thorax'
        , 'lowerneck'
        , 'upperneck'
        , 'head'
        , 'rclavicle'
        , 'rhumerus'
        , 'rradius'
        , 'rwrist'
        , 'rhand'
        , 'rfingers'
        , 'rthumb'
        , 'lclavicle'
        , 'lhumerus'
        , 'lradius'
        , 'lwrist'
        , 'lhand'
        , 'lfingers'
        , 'lthumb'
        , 'rfemur'
        , 'rtibia'
        , 'rfoot'
        , 'rtoes'
        , 'lfemur'
        , 'ltibia'
        , 'lfoot'
        , 'ltoes']

    data = f1.readlines()
    frames = []
    k = 0
    for i in range(0, len(data)):
        if data[i].startswith('root'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            root = np.append(root, [vals[1:]], axis=0)
            frames.append(k)
            k += 1

        elif data[i].startswith('lowerback'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lowerback = np.append(lowerback, [vals[1:]], axis=0)

        elif data[i].startswith('upperback'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            upperback = np.append(upperback, [vals[1:]], axis=0)

        elif data[i].startswith('thorax'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            thorax = np.append(thorax, [vals[1:]], axis=0)

        elif data[i].startswith('lowerneck'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lowerneck = np.append(lowerneck, [vals[1:]], axis=0)

        elif data[i].startswith('upperneck'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            upperneck = np.append(upperneck, [vals[1:]], axis=0)

        elif data[i].startswith('head'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            head = np.append(head, [vals[1:]], axis=0)

        elif data[i].startswith('rclavicle'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rclavicle = np.append(rclavicle, [vals[1:]], axis=0)

        elif data[i].startswith('rhumerus'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rhumerus = np.append(rhumerus, [vals[1:]], axis=0)

        elif data[i].startswith('rradius'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rradius = np.append(rradius, [vals[1:]], axis=0)

        elif data[i].startswith('rwrist'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rwrist = np.append(rwrist, [vals[1:]], axis=0)

        elif data[i].startswith('rhand'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rhand = np.append(rhand, [vals[1:]], axis=0)

        elif data[i].startswith('rfingers'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rfingers = np.append(rfingers, [vals[1:]], axis=0)

        elif data[i].startswith('rthumb'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rthumb = np.append(rthumb, [vals[1:]], axis=0)

        elif data[i].startswith('lclavicle'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lclavicle = np.append(lclavicle, [vals[1:]], axis=0)

        elif data[i].startswith('lhumerus'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lhumerus = np.append(lhumerus, [vals[1:]], axis=0)

        elif data[i].startswith('lradius'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lradius = np.append(lradius, [vals[1:]], axis=0)

        elif data[i].startswith('lwrist'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lwrist = np.append(lwrist, [vals[1:]], axis=0)

        elif data[i].startswith('lhand'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lhand = np.append(lhand, [vals[1:]], axis=0)

        elif data[i].startswith('lfingers'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lfingers = np.append(lfingers, [vals[1:]], axis=0)

        elif data[i].startswith('lthumb'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lthumb = np.append(lthumb, [vals[1:]], axis=0)

        elif data[i].startswith('rfemur'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rfemur = np.append(rfemur, [vals[1:]], axis=0)

        elif data[i].startswith('rtibia'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rtibia = np.append(rtibia, [vals[1:]], axis=0)

        elif data[i].startswith('rfoot'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rfoot = np.append(rfoot, [vals[1:]], axis=0)

        elif data[i].startswith('rtoes'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            rtoes = np.append(rtoes, [vals[1:]], axis=0)

        elif data[i].startswith('lfemur'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lfemur = np.append(lfemur, [vals[1:]], axis=0)

        elif data[i].startswith('ltibia'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            ltibia = np.append(ltibia, [vals[1:]], axis=0)

        elif data[i].startswith('lfoot'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            lfoot = np.append(lfoot, [vals[1:]], axis=0)

        elif data[i].startswith('ltoes'):
            vals = data[i].split(' ')
            n = len(vals)
            vals[n - 1] = vals[n - 1][:len(vals[n - 1]) - 1]
            for i in range(1, n):
                vals[i] = float(vals[i])
            ltoes = np.append(ltoes, [vals[1:]], axis=0)
        elif data[i][:len(data[i]) - 1].isalnum() == False:
            f2.write(data[i])

    joints = [root
        , lowerback
        , upperback
        , thorax
        , lowerneck
        , upperneck
        , head
        , rclavicle
        , rhumerus
        , rradius
        , rwrist
        , rhand
        , rfingers
        , rthumb
        , lclavicle
        , lhumerus
        , lradius
        , lwrist
        , lhand
        , lfingers
        , lthumb
        , rfemur
        , rtibia
        , rfoot
        , rtoes
        , lfemur
        , ltibia
        , lfoot
        , ltoes]

    step = float(src_framerate / dst_framerate)
    redFrames = np.arange(1, len(frames), step)

    new_joints = []
    for i in range(0, len(joints)):
        cs = CubicSpline(frames, joints[i])
        jointsn = cs(redFrames)
        new_joints.append(jointsn)

    for i in range(0, len(redFrames)):
        f2.write(str(i + 1) + '\n')
        for j in range(0, len(new_joints)):
            f2.write(joint_name[j])
            for k in range(0, len(new_joints[j][i])):
                f2.write(' ' + str(float(new_joints[j][i][k])))
            f2.write('\n')

    f1.close()
    f2.close()
    print("Successfully sampled %s to %s from frame rate %.2fHz to %.2fHz"
          % (amc_file, output_file, src_framerate, dst_framerate))


if __name__ == "__main__":
    # if len(sys.argv) != 5:
    #     print("USAGE: %s INPUT_FILENAME OUTPUT_FILENAME SRC_FRAMERATE DST_FRAMERATE" % sys.argv[0])
    # else:
    import glob
    for filename in glob.glob("animations/running/*.amc"):
        interpolate(filename,
                    filename[:len("animations")] + "_resampled" + filename[len("animations"):],
                    120,
                    66.66)
        # interpolate(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
