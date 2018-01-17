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

from baselines.ppo1 import pposgd_simple, mlp_policy
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U

from dm_control.rl import environment
from dm_control import suite


class DMSuiteEnv(gym.Env):
    def __init__(self, domain_name="cartpole", task_name="swingup", visualize_reward=True):
        self.dm_env = suite.load(domain_name=domain_name,
                                 task_name=task_name,
                                 visualize_reward=visualize_reward)
        action_spec = self.dm_env.action_spec()
        self.action_space = gym.spaces.Box(low=action_spec.minimum[0],
                                           high=action_spec.maximum[0],
                                           shape=action_spec.shape)
        try:
            ob_spec = self.dm_env.task.observation_spec(self.dm_env.physics)
            self.observation_space = gym.spaces.Box(low=ob_spec.minimum[0],
                                                    high=ob_spec.maximum[0],
                                                    shape=ob_spec.shape)
        except NotImplementedError:
            print("Could not retrieve observation spec, min/max possibly incorrect.", file=sys.stderr)
            # sample observation and set range to [-10, 10]
            ob = self.dm_env.task.get_observation(self.dm_env.physics)
            # ob is an OrderedDict, iterate over all entries to determine overall flattened ob dim
            ob_dimension = 0
            for entry in ob.values():
                ob_dimension += len(entry.flatten())
            self.observation_space = gym.spaces.Box(low=-10,
                                                    high=10,
                                                    shape=(ob_dimension,))
        self.reward_range = (0, 1)
        print('Initialized %s - %s.' % (domain_name, task_name))
        print('\tobservation space: %s (min: %.2f, max: %.2f)' %
              (str(self.observation_space.shape), self.observation_space.low[0], self.observation_space.high[0]))
        print('\taction space: %s (min: %.2f, max: %.2f)' %
              (str(self.action_space.shape), self.action_space.low[0], self.action_space.high[0]))

    def _step(self, action):
        step = self.dm_env.step(action)
        last_step = (step.step_type == environment.StepType.LAST)
        ob = self.observe()
        return ob, step.reward, last_step, {}

    def _reset(self):
        self.dm_env.reset()
        return self.observe()

    def observe(self):
        src_ob = self.dm_env.task.get_observation(self.dm_env.physics)
        ob = np.hstack(entry.flatten() for entry in src_ob.values())
        return ob


def train(num_timesteps, num_cpu, method, domain, task):
    rank = MPI.COMM_WORLD.Get_rank()
    ncpu = num_cpu
    if sys.platform == 'darwin':
        ncpu //= 2
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

    env = DMSuiteEnv(domain, task)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=3)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)),
                        allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    def callback(locals, globals):
        if MPI.COMM_WORLD.Get_rank() == 0 and locals['iters_so_far'] % 50 == 0:
            print('Saving video and checkpoint for policy at iteration %i...' % locals['iters_so_far'])
            ob = env.reset()
            images = []
            for i in range(1000):
                ac, vpred = locals['pi'].act(False, ob)
                ob, rew, new, _ = env.step(ac)
                if new:
                    break
                images.append(env.env.dm_env.physics.render(400, 400, camera_id=1))
            imageio.mimsave("videos/%s_%s_%s_iteration_%i.mp4" % (domain, task, method, locals['iters_so_far']),
                            images, fps=60)
            env.reset()
            U.save_state(os.path.join("checkpoints", "%s_%s_%i" % (domain, task, locals['iters_so_far'])))

    if method == "ppo":
        pposgd_simple.learn(env, policy_fn,
                            max_timesteps=int(num_timesteps),
                            timesteps_per_actorbatch=256,
                            clip_param=0.2, entcoeff=0.01,
                            optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                            gamma=0.99, lam=0.95,
                            schedule='linear',
                            callback=callback
                            )
    elif method == "trpo":
        trpo_mpi.learn(env, policy_fn,
                       max_timesteps=int(num_timesteps),
                       timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,
                       callback=callback
                       )
    else:
        print('ERROR: Invalid "method" argument provided.', file=sys.stderr)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=4)
    parser.add_argument('--method', help='reinforcement learning algorithm to use (ppo/trpo)',
                        type=str, default='ppo')
    parser.add_argument('--domain', help='domain to use for the RL environment from DM Control Suite',
                        type=str, default='cartpole')
    parser.add_argument('--task', help='task to use for the RL environment from DM Control Suite',
                        type=str, default='swingup')
    args = parser.parse_args()
    train(num_timesteps=args.num_timesteps,
          num_cpu=args.num_cpu,
          method=args.method.lower(),
          domain=args.domain,
          task=args.task)


if __name__ == '__main__':
    main()
