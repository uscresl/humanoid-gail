#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import mujoco_py  # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850

import argparse
import logging

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.mpi_fork import mpi_fork
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

import mocap_env.gym_roboschool_mocap_walker
import mocap_env.gym_mujoco_mocap_walker
import mocap_env

DEFAULT_ENV_ID = 'MujocoMocapHumanoid-v1'


def save(sess):
    saver = tf.train.Saver()
    saver.save(sess, "./checkpoint/humanoid_policy.ckpt")
    print("Saving policy checkpoint...")


def traj_segment_generator(pi, env, horizon, stochastic, human_render=False):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {
                "ob": obs,
                "rew": rews,
                "vpred": vpreds,
                "new": news,
                "ac": acs,
                "prevac": prevacs,
                "nextvpred": vpred * (1 - new),
                "ep_rets": ep_rets,
                "ep_lens": ep_lens
            }
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        if human_render:
            env.render()

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def train(env_id, num_timesteps, hidden_size, num_hidden_layers, seed, rank):
    with U.make_session(3) as sess:
        worker_seed = seed + 10000 * rank
        set_global_seeds(worker_seed)

        # env = bench.Monitor(env, logger.get_dir() and
        # osp.join(logger.get_dir(), str(rank)))

        try:
            env = gym.make(env_id)
            env.seed(worker_seed)

            # Rendering and saving callback
            episode = 0

            def episode_callback(locals, globals):
                nonlocal episode
                episode += 1
                print("----- Episode {} -----".format(episode))
                env.render()
                if episode % 20 == 0:
                    save(sess)

            # Policy function
            policy_fn = lambda name, ob_space, ac_space: MlpPolicy(
                name=name,
                ob_space=env.observation_space,
                ac_space=env.action_space,
                hid_size=hidden_size,
                num_hid_layers=num_hidden_layers
            )

            # Learning
            trpo_mpi.learn(
                env,
                policy_fn,
                timesteps_per_batch=1024,
                max_kl=0.01,
                cg_iters=10,
                cg_damping=0.1,
                max_timesteps=num_timesteps,
                gamma=0.99,
                lam=0.98,
                vf_iters=5,
                vf_stepsize=1e-3,
                callback=episode_callback)
        finally:
            env.close()


def load(checkpoint, env_id, hidden_size, num_hidden_layers):
    with U.single_threaded_session() as sess:
        try:
            env = gym.make(env_id)

            policy = MlpPolicy(
                name="pi",
                ob_space=env.observation_space,
                ac_space=env.action_space,
                hid_size=hidden_size,
                num_hid_layers=num_hidden_layers)

            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            seg_gen = traj_segment_generator(
                policy, env, 1024, True, human_render=True)

            # Generate trajectory segments until stopped
            for _ in seg_gen:
                pass
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--env_id', help='environment ID', default=DEFAULT_ENV_ID)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--hidden-size', type=int, default=int(64))
    parser.add_argument('--hidden-layers', type=int, default=int(2))
    parser.add_argument('--load_checkpoint', type=str, default="")
    args = parser.parse_args()

    # Option to load and execute a policy
    # TODO: move to a different script
    if args.load_checkpoint:
        load(
            checkpoint=args.load_checkpoint,
            env_id=args.env_id,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.hidden_layers)
        return

    # Configure logging
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure('./log', ['tensorboard'])
    else:
        logger.set_level(logger.DISABLED)
    gym.logger.setLevel(logging.WARN)

    # Main training loop
    train(
        env_id=args.env_id,
        num_timesteps=args.num_timesteps,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.hidden_layers,
        seed=args.seed,
        rank=rank)


if __name__ == '__main__':
    main()
