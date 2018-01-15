import numpy as np


def extract_features(env):
    """Extracts features from the current observation."""
    obs = env.task.get_observation(env.physics)
    return np.hstack((obs['extremities'], obs['torso_vertical']))


def extract_observations(env):
    """Observations include joint angles and features."""
    obs = env.task.get_observation(env.physics)
    return np.hstack((obs['joint_angles'], obs['extremities'], obs['torso_vertical']))
