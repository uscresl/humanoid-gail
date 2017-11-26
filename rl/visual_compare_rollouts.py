import gym
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import featurized_env
import featurized_env.humanoid

plot_counter = 0


def sanitize_features(features):
    for i in range(0, 15, 3):
        features[:, i], features[:, i+1], features[:, i+2] = features[:, i+2].copy(), features[:, i].copy(), features[:, i+1].copy()
    return features


def plot_old_features(features, title=None):
    global plot_counter
    plot_counter += 1
    plt.figure(plot_counter)

    def label(i):
        parts = ["head", "left_hand", "right_hand", "left_foot", "right_foot"]
        xyz = "xyz"
        return "%s\n%s" % (parts[i // 3], xyz[i % 3])

    block_size = 40
    violins = []
    print(features.shape)
    for i in range(15):
        violins.append(features[:, i])
        # for j in range(0, features.shape[0], block_size):
        #     plt.plot(list(range(j, j+block_size)),
        #              list(features[j:j+block_size, i].mean() for x in range(block_size)), label=label(i))
    plt.violinplot(violins, showmeans=False, showmedians=True)
    plt.gca().set_xticks(range(1, 16))
    plt.gca().set_xticklabels([label(i) for i in range(15)])
    plt.grid()

    if title is not None:
        plt.title(title)

    plt.legend()


def plot_features(features, title=None):
    global plot_counter
    plot_counter += 1
    plt.figure(plot_counter)

    def label(i):
        parts = ["head", "left_hand", "right_hand", "left_foot", "right_foot", "delta_root"]
        xyz = "xyz"
        return "%s\n%s" % (parts[i // 3], xyz[i % 3])

    violins = []
    print(features.shape)
    for i in range(18):
        violins.append(features[i, :])
        # for j in range(0, features.shape[0], block_size):
        #     plt.plot(list(range(j, j+block_size)),
        #              list(features[j:j+block_size, i].mean() for x in range(block_size)), label=label(i))
    plt.violinplot(violins, showmeans=False, showmedians=True)
    plt.gca().set_xticks(range(1, 19))
    plt.gca().set_xticklabels([label(i) for i in range(18)])
    plt.grid()

    if title is not None:
        plt.title(title)

    plt.legend()


def load_pickled_features(pkl_file):
    with open(pkl_file, "rb") as f:
        traj_data = pkl.load(f)
        features = []
        for traj in traj_data:
            features.append(traj["features"])
        return np.array([v for ob in features for v in ob])


def load_pickled_features_buggy(pkl_file):
    with open(pkl_file, "rb") as f:
        traj_data = pkl.load(f)
        features = []
        for traj in traj_data:
            features.append(traj["features"])
        return np.array(features)


def load_pickled_positions(pkl_file):
    with open(pkl_file, "rb") as f:
        traj_data = pkl.load(f)
        positions = []
        for traj in traj_data:
            positions.append(traj["positions"])
        return [v for ob in positions for v in ob]


def plot_positions(positions, keys=["root"], title=None):
    global plot_counter
    plot_counter += 1
    plt.figure(plot_counter)
    violins = []
    xyz = "xyz"
    for key in keys:
        violinx, violiny, violinz = [], [], []
        for t, p in enumerate(positions):
            violinx.append(np.ravel(p[key])[0])
            violiny.append(np.ravel(p[key])[1])
            violinz.append(np.ravel(p[key])[2])
        violins.append(violinx)
        violins.append(violiny)
        violins.append(violinz)

        # for j in range(0, features.shape[0], block_size):
        #     plt.plot(list(range(j, j+block_size)),
        #              list(features[j:j+block_size, i].mean() for x in range(block_size)), label=label(i))
    plt.violinplot(violins, showmeans=False, showmedians=True)
    plt.gca().set_xticks(range(1, len(keys) * 3 + 1))

    def label(i):
        return "%s\n%s" % (keys[i // 3], xyz[i % 3])

    plt.gca().set_xticklabels([label(i) for i in range(len(keys) * 3)])
    plt.grid()

    if title is not None:
        plt.title(title)

    plt.legend()


def plot_time_positions(positions, keys=["root"], title=None):
    global plot_counter
    plot_counter += 1
    plt.figure(plot_counter)
    violins = []
    xyz = "xyz"
    for key in keys:
        violinx, violiny, violinz = [], [], []
        for t, p in enumerate(positions):
            violinx.append(np.ravel(p[key])[0])
            violiny.append(np.ravel(p[key])[1])
            violinz.append(np.ravel(p[key])[2])
        plt.plot(violinx, label="%s\n%s" % (key, "x"))
        plt.plot(violiny, label="%s\n%s" % (key, "y"))
        plt.plot(violinz, label="%s\n%s" % (key, "z"))

    plt.grid()

    if title is not None:
        plt.title(title)

    plt.legend()


def tabulate_features(features, title=None, scaling=10):
    target = np.zeros((features.shape[0] * scaling, features.shape[1]))
    for i in range(features.shape[0]):
        target[i*scaling:(i+1)*scaling,:] = features[i, :]
    plt.matshow(target)

    if title is not None:
        plt.title(title)


if __name__ == "__main__":
    env = gym.make("HumanoidFeaturized-v1")
    # trpo_features = load_pickled_features("/home/eric/.deep-rl-docker/gail-tf/rollout/stochastic.trpo.HumanoidFeaturized.0.00_sensical.pkl")
    trpo_features = load_pickled_features("/home/eric/.deep-rl-docker/gail-tf/rollout/stochastic.trpo.HumanoidFeaturized.0.00.pkl")
    mocap_features = load_pickled_features("/home/eric/.deep-rl-docker/gail-tf/rollout/mocap_trajectories.pkl")
    plot_features(trpo_features.T, title="TRPO")
    plot_features(mocap_features.T, title="Mocap")
    trpo_positions = load_pickled_positions(
        "/home/eric/.deep-rl-docker/gail-tf/rollout/stochastic.trpo.HumanoidFeaturized.0.00.pkl")[:2000]
    mocap_positions = load_pickled_positions("/home/eric/.deep-rl-docker/gail-tf/rollout/mocap_trajectories.pkl")[:2000]
    plot_time_positions(trpo_positions, keys=["root"], title="TRPO")
    plot_time_positions(mocap_positions, keys=["root"], title="Mocap")

    tabulate_features(trpo_features.T, title="TRPO")
    tabulate_features(mocap_features.T, title="Mocap")
    plt.show()
