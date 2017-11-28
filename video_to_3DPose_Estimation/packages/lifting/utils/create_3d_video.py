import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time


def plot_pose(poses,index):
    """Plot the 3D pose showing the joint connections."""
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

    def joint_color(j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """

        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.show(block=False)

    for move in poses:
        for pose in move:
            smallest = pose.min()
            largest = pose.max()
            break
        break



    for move in poses:
        for pose in move:
            for c in _CONNECTION:
                col = '#%02x%02x%02x' % joint_color(c[0])

                ax.plot([pose[0, c[0]], pose[0, c[1]]],
                        [pose[1, c[0]], pose[1, c[1]]],
                        [pose[2, c[0]], pose[2, c[1]]], c=col)
            for j in range(pose.shape[1]):
                col = '#%02x%02x%02x' % joint_color(j)
                ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                           c=col, marker='o', edgecolor=col)
            #ax.view_init(elev=2., azim=-100)
            ax.set_xlim3d(smallest, largest)
            ax.set_ylim3d(smallest, largest)
            ax.set_zlim3d(smallest, largest)
            #plt.pause(0.000001)
            fig.canvas.flush_events()
            ax.clear()


    return fig



if __name__ == '__main__':
    plt.ion() # turn on interactive mode, non-blocking `show`
    lis = []
    for loop in range(1,110):
        y = np.load('../../../joints/joint_list_'+ str(loop)+ '.npy')
        #for single_3D in y:
        lis.append(y)
    plot_pose(lis, 0)

         # show the figure, non-blocking
    #_ = raw_input("Press [enter] to continue.") # wait for input from the user
    plt.close()

