from gym.envs.registration import register
from .gym_roboschool_mocap_walker import RoboschoolMocapHumanoid
from .gym_mujoco_mocap_walker import MujocoMocapHumanoid

register(
    id='RoboschoolMocapHumanoid-v1',
    entry_point='mocap_env:RoboschoolMocapHumanoid',
    max_episode_steps=1000,
    reward_threshold=3500.0)
register(
    id='MujocoMocapHumanoid-v1',
    entry_point='mocap_env:MujocoMocapHumanoid',
    max_episode_steps=1000,
    reward_threshold=3500.0)
