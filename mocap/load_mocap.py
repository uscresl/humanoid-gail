from asf_parser import AsfParser
from amc_parser import AmcParser
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import csv, os
from tqdm import tqdm


def normalize(vector):
    return vector / (norm(vector) + 1e-10)


def compute_features(positions):
    features = np.array([
        normalize(positions["head"] - positions["root"]),
        normalize(positions["lhand"] - positions["root"]),
        normalize(positions["rhand"] - positions["root"]),
        normalize(positions["lfoot"] - positions["root"]),
        normalize(positions["rfoot"] - positions["root"])
    ])
    return features.flatten()


def load_positions(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                  amc_file=os.path.join(os.path.dirname(__file__), "examples/02_01.amc")):
    """Computes the end-effector positions over all animation frames of the AMC file."""
    parser = AsfParser()
    parser.parse(asf_file)
    amc = AmcParser()
    amc.parse(amc_file)
    skeleton = parser.skeleton
    positions = []
    for frame in tqdm(amc.frames):
        positions.append(skeleton.compute_motion(frame))
    return positions


def load_features(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                  amc_file=os.path.join(os.path.dirname(__file__), "examples/02_01.amc"),
                  forward_vector_frames=10,
                  frames_per_feature=1):
    """Computes the 5 3D vectors over all animation frames of the AMC file. Returns a Tx15 matrix."""
    parser = AsfParser()
    parser.parse(asf_file)
    amc = AmcParser()
    amc.parse(amc_file)
    skeleton = parser.skeleton
    features = [np.zeros(18)] * (frames_per_feature-1)
    output_features = []
    positions = [{"root": np.zeros((1, 3))}] * forward_vector_frames
    print("There are %i frames" % len(amc.frames))
    for i, frame in tqdm(enumerate(amc.frames)):
        # Compute forward-facing unit vector from the average of previous frames
        forward_vector = np.zeros((1, 3))
        positions.append(skeleton.compute_motion(frame))
        previous_pos = positions[-forward_vector_frames]["root"]
        for j in range(-forward_vector_frames+1, 0):
            next_pos = positions[j]["root"]
            forward_vector += next_pos-previous_pos
            previous_pos = next_pos.copy()
        forward_vector /= forward_vector_frames

        pos = positions[i + forward_vector_frames]
        feat = compute_features(pos)
        feat = np.hstack((feat, forward_vector.flatten()))
        features.append(feat)
        output_features.append(np.array(features[-frames_per_feature:]).flatten())

    output_features = np.array(output_features)
    return output_features


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
