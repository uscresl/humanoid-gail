from gym.envs.mujoco import HumanoidEnv
from gym.envs.mujoco import mujoco_env
from gym import utils
from sklearn.preprocessing import normalize

import os


class HumanoidFeatureEnv(HumanoidEnv):
    def __init__(self):
        # The MuJoCo XML definition has been modified so that head, hands and feet are denoted as <body> elements
        # so that we can obtain their COMs via self.get_body_com(body_name).
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'humanoid_featurized.xml'), 5)
        # self.model.opt.timestep = 0.005
        # self.frame_skip = 6
        print("*" * 50)
        print("Time step: %.4f" % self.model.opt.timestep)
        print("Frame skip: %i" % self.frame_skip)
        print("dt: %.4f" % self.dt)
        print("*" * 50)
        utils.EzPickle.__init__(self)

    def compute_features(self):
        features = [
            normalize(self.get_body_com("head") - self.get_body_com("torso")),
            normalize(self.get_body_com("left_hand") - self.get_body_com("torso")),
            normalize(self.get_body_com("right_hand") - self.get_body_com("torso")),
            normalize(self.get_body_com("left_foot") - self.get_body_com("torso")),
            normalize(self.get_body_com("right_foot") - self.get_body_com("torso"))
        ]
        return features
