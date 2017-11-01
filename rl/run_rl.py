#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
import tensorflow as tf
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
# from baselines import bench
from baselines.trpo_mpi import trpo_mpi
from baselines.trpo_mpi.trpo_mpi import traj_segment_generator
import baselines.common.tf_util as U
import sys

import mocap_env.gym_mocap_walker
import mocap_env.gym_mujoco_walker
import mocap_env

env = None
ENV_ID = 'MujocoMocapHumanoid-v1'


def learning_iteration(locals, globals):
    global env
    env.render()


def train(env_id, num_timesteps, seed):
    global env
    sess = U.make_session(3)
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=64, num_hid_layers=2)
    # env = bench.Monitor(env, logger.get_dir() and
        # osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, callback=learning_iteration)
    env.close()


def load(check_point):
    policy = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=64, num_hid_layers=2)
    saver = tf.train.Saver()
    sess = U.single_threaded_session()
    sess.__enter__()
    saver.restore(sess, check_point)

    seg_gen = traj_segment_generator(policy, env, 1024, True, human_render=True)
    while True:
        seg_gen.__next__()


def main():
    global ENV_ID
    global env
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=ENV_ID)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e5))
    parser.add_argument('--load', type=str, default="")
    args = parser.parse_args()
    ENV_ID = args.env
    env = gym.make(ENV_ID)
    if len(args.load) > 0:
        load(args.load)
        return

    logger.configure('./log', ['tensorboard'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    saver = tf.train.Saver()
    sess = tf.get_default_session()
    save_path = saver.save(sess, "./humanoid_policy.ckpt")


if __name__ == '__main__':
    main()