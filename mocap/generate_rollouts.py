from load_mocap import load_features, load_positions
from glob import glob
import os
import pickle as pkl


def sanitize_features(features):
    for i in range(0, 15, 3):
        features[:, i], features[:, i+1], features[:, i+2] = features[:, i+2].copy(), features[:, i].copy(), features[:, i+1].copy()
    return features


def generate_rollouts(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                      amc_folder=os.path.join(os.path.dirname(__file__), "animations"),
                      output_file="mocap_trajectories.pkl"):
    sample_trajs = []
    total_features = 0
    for amc_file in sorted(glob(amc_folder + "/*.amc")):
        features = load_features(asf_file, amc_file)
        positions = load_positions(asf_file, amc_file)
        sample_trajs.append({
            "features": sanitize_features(features),
            "positions": positions
        })
        # traj = []
        # for feat, pos in zip(features, positions):
        #     traj.append({
        #         "features": fix_coordinates(feat)#,
        #         # "positions": pos
        #     })
        total_features += len(features)
        # sample_trajs.append(traj)
    pkl.dump(sample_trajs, open(output_file, "wb"))
    print("Saved %i trajectories totalling %i features to %s." % (len(sample_trajs), total_features, output_file))


if __name__ == "__main__":
    generate_rollouts(amc_folder="animations_resampled")
