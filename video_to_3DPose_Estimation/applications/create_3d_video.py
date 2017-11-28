import cv2
import numpy as np
from .config import JOINT_DRAW_SIZE
from .config import LIMB_DRAW_SIZE
from .config import NORMALISATION_COEFFICIENT
import matplotlib.pyplot as plt
import math


def plot_pose(pose,index):
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

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                   c=col, marker='o', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)
    ax.view_init(elev=2., azim=-100)
    return fig



if __name__ == '__main__':
    x = [1, 2, 3]
    plt.ion() # turn on interactive mode, non-blocking `show`
    for loop in range(1,3):
        y = np.load('joints/joint_list_'+ str(loop))
        print (y)
        plot_pose(y)
        plt.show()     # show the figure, non-blocking
        _ = raw_input("Press [enter] to continue.") # wait for input from the user
        plt.close()
