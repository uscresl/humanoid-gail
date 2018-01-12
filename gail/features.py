import numpy as np


def features(env):
    obs = env.task.get_observation(env.physics)
    return np.hstack((obs['extremities'], obs['torso_vertical']))