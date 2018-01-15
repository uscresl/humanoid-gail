#!/usr/bin/env python3

import imageio, os

import sys
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger

from gym.spaces import box
import numpy as np
import tensorflow as tf

from gail.features import extract_features, extract_observations

from baselines.ppo1 import pposgd_simple, mlp_policy
import baselines.common.tf_util as U

from dm_control.rl import environment
from dm_control.suite import humanoid_CMU
from dm_control.mujoco import engine


class HumanoidEnv(gym.Env):
    def __init__(self):
        self.dm_env = humanoid_CMU.run()
        self.dm_env.reset()
        ob = self.observation()
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=ob.shape)
        action_spec = engine.action_spec(self.dm_env.physics)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=action_spec.shape)

    def _step(self, action):
        step = self.dm_env.step(action)
        last_step = (step.step_type == environment.StepType.LAST)
        ob = self.observation()
        return ob, step.reward, last_step, {}

    def _reset(self):
        self.dm_env.reset()
        return self.observation()

    def observation(self):
        ob = np.hstack([extract_features(self.dm_env),
                        extract_observations(self.dm_env)])
        return ob


def train(num_timesteps, num_cpu):
    rank = MPI.COMM_WORLD.Get_rank()
    ncpu = num_cpu
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = 123 + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = HumanoidEnv()

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=32, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)),
                        allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    def callback(locals, globals):
        if locals['iters_so_far'] % 50 == 0:
            print('Saving video and checkpoint for policy at iteration %i...' % locals['iters_so_far'])
            ob = env.reset()
            images = []
            for i in range(1000):
                ac, vpred = locals['pi'].act(False, ob)
                ob, rew, new, _ = env.step(ac)
                if new:
                    break
                images.append(env.env.dm_env.physics.render(400, 400, camera_id=1))
            imageio.mimsave("videos/iteration_%i.mp4" % locals['iters_so_far'], images, fps=60)
            env.reset()
            U.save_state(os.path.join("checkpoints", "CMUHumanoid_Run_%i" % locals['iters_so_far']))

    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=int(num_timesteps),
                        timesteps_per_actorbatch=256,
                        clip_param=0.2, entcoeff=0.01,
                        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                        gamma=0.99, lam=0.95,
                        schedule='linear',
                        callback=callback
                        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=4)
    args = parser.parse_args()
    train(num_timesteps=args.num_timesteps, num_cpu=args.num_cpu)


if __name__ == '__main__':
    main()
