#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import gym
import logging
import tensorflow as tf
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
# from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U
import numpy as np

import mocap_env.gym_mocap_walker
import mocap_env.gym_mujoco_walker
import mocap_env

env = None
ENV_ID = 'MujocoMocapHumanoid-v1'
iteration = 0
hidden_size = 64
num_hidden_layers = 2


def save():
    saver = tf.train.Saver()
    sess = tf.get_default_session()
    saver.save(sess, "./humanoid_policy.ckpt")
    print("Saved policy.")


def learning_iteration(locals, globals):
    global iteration
    global env
    env.render()
    iteration += 1
    if iteration % 20 == 0:
        save()


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
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
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
                  hid_size=hidden_size, num_hid_layers=num_hidden_layers)
    # env = bench.Monitor(env, logger.get_dir() and
        # osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, callback=learning_iteration)
    env.close()


def load(check_point):
    policy = MlpPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=hidden_size, num_hid_layers=num_hidden_layers)
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
    global hidden_size
    global num_hidden_layers
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=ENV_ID)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--hidden-size', type=int, default=int(64))
    parser.add_argument('--hidden-layers', type=int, default=int(2))
    parser.add_argument('--load', type=str, default="")
    args = parser.parse_args()
    ENV_ID = args.env
    hidden_size = args.hidden_size
    num_hidden_layers = args.hidden_layers
    env = gym.make(ENV_ID)
    if len(args.load) > 0:
        load(args.load)
        return

    logger.configure('./log', ['tensorboard'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    save()


if __name__ == '__main__':
    main()