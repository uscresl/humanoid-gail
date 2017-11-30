from load_mocap import load_features, load_positions
from glob import glob
import os
import pickle as pkl
import numpy as np


def generate_rollouts(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                      amc_folder=os.path.join(os.path.dirname(__file__), "animations"),
                      output_file="mocap_trajectories.pkl",
                      forward_vector_frames=10,
                      frames_per_feature=1):
    sample_trajs = []
    total_features = 0
    filenames = sorted(glob(amc_folder + "/*.amc"))
    for i, amc_file in enumerate(filenames):
        print("Processing file %i out of %i (%s)..." % (i+1, len(filenames), amc_file))
        features = load_features(asf_file,
                                 amc_file,
                                 forward_vector_frames=forward_vector_frames,
                                 frames_per_feature=frames_per_feature)
        positions = load_positions(asf_file, amc_file)
        sample_trajs.append({
            "features": features,
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
    print("Saved %i trajectories totalling %i features to %s."
          % (len(sample_trajs), total_features, output_file))


if __name__ == "__main__":
    generate_rollouts(amc_folder="animations_resampled/running")
