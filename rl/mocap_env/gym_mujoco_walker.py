import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
from baselines import logger


def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]



class MujocoMocapHumanoid(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        print(os.path.join(os.path.dirname(__file__), 'humanoid_mocap.xml'))
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'humanoid_mocap.xml'), 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a):
        global plotting_initialized
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model)
        alive_bonus = 5.0
        qpos = self.model.data.qpos
        data = self.model.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        logger.logkv('reward', reward)
        logger.logkv('height', qpos[1])
        logger.logkv('lin_vel_cost', lin_vel_cost)
        done = bool((qpos[1] < .3) or (qpos[1] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
