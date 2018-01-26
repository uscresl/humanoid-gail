import gym
from gym.spaces import Box
from gail.rllab.rllab.envs.normalized_env import normalize
# from gail.rllab.rllab.envs.box2d.cartpole_env import CartpoleEnv
# from gail.rllab.rllab.envs.box2d.car_parking_env import CarParkingEnv
# from gail.rllab.rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
# from gail.rllab.rllab.envs.box2d.mountain_car_env import MountainCarEnv
from gail.rllab.rllab.envs.mujoco.ant_env import AntEnv
from gail.rllab.rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from gail.rllab.rllab.envs.mujoco.hopper_env import HopperEnv
from gail.rllab.rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from gail.rllab.rllab.envs.mujoco.point_env import PointEnv
from gail.rllab.rllab.envs.mujoco.humanoid_env import HumanoidEnv, SimpleHumanoidEnv
from gail.rllab.rllab.envs.mujoco.swimmer3d_env import Swimmer3DEnv, SwimmerEnv
from gail.rllab.rllab.envs.mujoco.walker2d_env import Walker2DEnv


_id_mapping = {
    # "cartpole": CartpoleEnv,
    # "car-parking": CarParkingEnv,
    # "double-pendulum": DoublePendulumEnv,
    # "mountain-car": MountainCarEnv,
    "ant": AntEnv,
    "half-cheetah": HalfCheetahEnv,
    "hopper": HopperEnv,
    "inverted-double-pendulum": InvertedDoublePendulumEnv,
    "point": PointEnv,
    "humanoid": HumanoidEnv,
    "simple-humanoid": SimpleHumanoidEnv,
    "swimmer": SwimmerEnv,
    "swimmer3d": Swimmer3DEnv,
    "walker2d": Walker2DEnv
}


def _convert_rl_space(rlspace):
    return Box(
            low=rlspace.low[0],
            high=rlspace.high[0],
            shape=rlspace.shape)


class RllabEnv(gym.Env):
    def __init__(self, id):
        self.rl_env = normalize(_id_mapping[id]())

        self.monitoring = False

        self.observation_space = _convert_rl_space(self.rl_env.observation_space)
        self.action_space = _convert_rl_space(self.rl_env.action_space)
        print('\tobservation space: %s (min: %.2f, max: %.2f)' %
              (str(self.observation_space.shape),
               self.observation_space.low[0], self.observation_space.high[0]))
        print('\taction space: %s (min: %.2f, max: %.2f)' %
              (str(self.action_space.shape), self.action_space.low[0],
               self.action_space.high[0]))
        self._force_reset = True

    def _reset(self):
        return self.rl_env.reset()

    def _step(self, action):
        s = self.rl_env.step(action)
        return s.observation, s.reward, s.done, s.info

    def render(self, mode='rgb_array', close=False):
        return self.rl_env.render(close=close, mode=mode)
