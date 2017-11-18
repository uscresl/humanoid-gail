from asf_parser import AsfParser
from amc_parser import AmcParser
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import csv, os


def normalize(vector):
    return vector / norm(vector)


def compute_features(positions):
    features = np.array([
        normalize(positions["head"] - positions["root"]),
        normalize(positions["lhand"] - positions["root"]),
        normalize(positions["rhand"] - positions["root"]),
        normalize(positions["lfoot"] - positions["root"]),
        normalize(positions["rfoot"] - positions["root"])
    ])
    return features.flatten()


def load_features(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                  amc_file=os.path.join(os.path.dirname(__file__), "examples/02_01.amc")):
    """Computes the 5 3D vectors over all animation frames of the AMC file."""
    parser = AsfParser()
    parser.parse(asf_file)
    amc = AmcParser()
    amc.parse(amc_file)
    skeleton = parser.skeleton
    features = []
    for frame in amc.frames:
        positions = skeleton.compute_motion(frame)
        features.append(compute_features(positions))
    features = np.array(features)
    return features


def main():
    plt.figure(1)
    features = []
    with open('examples/web_features.csv') as csvfile:
        featreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for idx, frame in enumerate(featreader):
            features.append(frame)
        features = np.array(features)
    for i in range(15):
        plt.plot(list(float(x) for x in features[:, i]))

    plt.figure(2)
    parser = AsfParser()
    parser.parse("examples/12.asf")
    amc = AmcParser()
    amc.parse("examples/02_01.amc")
    skeleton = parser.skeleton
    features = []
    for frame in amc.frames:
        positions = skeleton.compute_motion(frame)
        features.append(compute_features(positions))
    features = np.array(features)
    for i in range(15):
        plt.plot(features[:, i])
    plt.show()


if __name__ == "__main__":
    main()
