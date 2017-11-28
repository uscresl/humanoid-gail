import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def load_pickled_features(pkl_file):
    with open(pkl_file, "rb") as f:
        traj_data = pkl.load(f, encoding="latin1")
        return traj_data


def plot_time_features(features, title=None):
    xyz = "xyz"
    animation = {}
    #keys = {'Head', 'LFoot', 'RFoot', 'LWrist', 'RWrist'}
    keys = {'LWrist'}
    print(features[0])
    for key in keys:
        for d in xyz:
            animation["%s\n%s" % (key, d)] = []
    for frame in features:
        for key in keys:
            for i, d in enumerate(xyz):
                animation["%s\n%s" % (key, d)].append(frame[key][i]/1000.)

    for label, values in animation.items():
        plt.plot(values, label=label)

    plt.grid()

    if title is not None:
        plt.title(title)

    plt.legend(loc="right", ncol=3)
    plt.show()


if __name__ == "__main__":
    features = load_pickled_features("joints.pkl")
    plot_time_features(features, title="Video Pose Estimation")
