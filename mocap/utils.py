import numpy as np


def rotation_matrix_axis(C_values):
    # Change coordinate system through matrix C
    rx = np.deg2rad(float(C_values[0]))
    ry = np.deg2rad(float(C_values[1]))
    rz = np.deg2rad(float(C_values[2]))

    Cx = np.matrix([[1, 0, 0],
                    [0, np.cos(rx), np.sin(rx)],
                    [0, -np.sin(rx), np.cos(rx)]])

    Cy = np.matrix([[np.cos(ry), 0, -np.sin(ry)],
                    [0, 1, 0],
                    [np.sin(ry), 0, np.cos(ry)]])

    Cz = np.matrix([[np.cos(rz), np.sin(rz), 0],
                    [-np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    C = Cx * Cy * Cz
    Cinv = np.linalg.inv(C)
    return C, Cinv


def rotation_matrix(bone, tx, ty, tz):
    # Construct rotation matrix M
    tx = np.deg2rad(tx)
    ty = np.deg2rad(ty)
    tz = np.deg2rad(tz)

    Mx = np.matrix([[1, 0, 0],
                    [0, np.cos(tx), np.sin(tx)],
                    [0, -np.sin(tx), np.cos(tx)]])

    My = np.matrix([[np.cos(ty), 0, -np.sin(ty)],
                    [0, 1, 0],
                    [np.sin(ty), 0, np.cos(ty)]])

    Mz = np.matrix([[np.cos(tz), np.sin(tz), 0],
                    [-np.sin(tz), np.cos(tz), 0],
                    [0, 0, 1]])
    M = Mx * My * Mz
    L = bone["Cinv"] * M * bone["C"]
    return L
