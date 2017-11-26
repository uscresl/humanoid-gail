from utils import rotation_matrix_axis, rotation_matrix
import numpy as np


class Skeleton(object):
    def __init__(self):
        self.hierarchy = {}
        self.bones = {}
        self.root = {}

    def compute_aux_matrices(self):
        for bone in self.bones.values():
            bone["C"], bone["Cinv"] = rotation_matrix_axis(bone["axis"])

        self.hierarchy["root"]["axis"] = np.zeros(3)
        self.hierarchy["root"]["C"], self.hierarchy["root"]["Cinv"] = rotation_matrix_axis(np.zeros(3))

    def compute_motion(self, frame):
        positions = {}

        def sanitize(vector):
            vector = vector.flatten()
            return np.array([vector[0, 2], vector[0, 0], vector[0, 1]])

        def dfs(position, node, stack):
            angles = np.zeros(3)
            direction = np.zeros(3)
            if "direction" in node:
                direction = node["direction"] * node["length"]
                if "axis" in node and node["name"] in frame:
                    dofs = ["rx", "ry", "rz"]
                    for d in range(len(node["dof"])):
                        dof = node["dof"][d]
                        angles[dofs.index(dof)] = frame[node["name"]][d]

            L = rotation_matrix(node, angles[0], angles[1], angles[2])
            stack.append(np.dot(L, stack[-1]))
            end_position = position + np.dot(direction, stack[-1])
            positions[node["name"]] = sanitize(end_position.copy())

            if "children" in node:
                for child in node["children"].values():
                    dfs(end_position, child, stack)

            stack.pop()

        root_matrix = rotation_matrix(self.hierarchy["root"], frame["root"][3],
                                      frame["root"][4], frame["root"][5])
        root_pos = frame["root"][:3]
        dfs(root_pos, self.hierarchy["root"], [root_matrix])

        return positions
