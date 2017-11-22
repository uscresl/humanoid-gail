from load_mocap import load_features
from glob import glob
import os
import pickle as pkl


def generate_rollouts(asf_file=os.path.join(os.path.dirname(__file__), "examples/12.asf"),
                      amc_folder=os.path.join(os.path.dirname(__file__), "animations"),
                      output_file="mocap_trajectories.pkl"):
    sample_trajs = []
    for amc_file in sorted(glob(amc_folder + "/*.amc")):
        features = load_features(asf_file, amc_file)
        for feat in features:
            sample_trajs.append({
                "features": feat
            })
    pkl.dump(sample_trajs, open(output_file, "wb"))
    print("Saved %i samples to %s." % (len(sample_trajs), output_file))


if __name__ == "__main__":
    generate_rollouts()