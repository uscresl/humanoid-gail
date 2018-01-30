#!/usr/bin/env python3

import imageio, os, sys, datetime, pathlib, json
from mpi4py import MPI
import os.path as osp
import gym, gym.spaces, logging

import tensorflow as tf

from gail.algos.acktr import acktr
from gail.baselines.baselines import bench, logger
from gail.baselines.baselines.acktr.policies import GaussianMlpPolicy
from gail.baselines.baselines.acktr.value_functions import NeuralNetValueFunction
from gail.baselines.baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from gail.baselines.baselines.ppo1 import pposgd_simple, mlp_policy
from gail.baselines.baselines.trpo_mpi import trpo_mpi
import gail.baselines.baselines.ddpg.training as ddpg_training
from gail.baselines.baselines.ddpg.models import Actor, Critic
from gail.baselines.baselines.ddpg.memory import Memory
from gail.baselines.baselines.ddpg.noise import *


def load_state(fname):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), fname)


def save_state(fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(tf.get_default_session(), fname)


def train(env_fn, environment, num_timesteps, num_cpu, method, noise_type, layer_norm, folder, load_policy,
          video_width, video_height, plot_rewards, **kwargs):
    if sys.platform == 'darwin':
        num_cpu //= 2
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=num_cpu,
        inter_op_parallelism_threads=num_cpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()

    worker_seed = 1234 + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(worker_seed)

    env = env_fn()
    env.seed(worker_seed)

    rank = MPI.COMM_WORLD.Get_rank()
    logger.info('rank {}: seed={}, logdir={}'.format(rank, worker_seed,
                                                     logger.get_dir()))

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        # failed attempt to transfer policy from rllab to DeepMind Control Suite
        # from gail.transfer_pi import TransferMlpPolicy
        # return TransferMlpPolicy(
        #     name=name,
        #     ob_space=ob_space,
        #     ac_space=ac_space,
        #     src_ob_space=gym.spaces.Box(-5, 5, shape=(142,)),  # rllab humanoid ob_space
        #     hid_size=150,
        #     num_hid_layers=3)
        return mlp_policy.MlpPolicy(
            name=name,
            ob_space=ob_space,
            ac_space=ac_space,
            hid_size=150,
            num_hid_layers=3)

    env = bench.Monitor(
        env,
        logger.get_dir() and osp.join(logger.get_dir(), str(rank)),
        allow_early_resets=True)
    gym.logger.setLevel(logging.INFO)

    def callback(locals, globals):
        if load_policy is not None and locals['iters_so_far'] == 0:
            # noinspection PyBroadException
            try:
                load_state(load_policy)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    logger.info("Loaded policy network weights from %s." % load_policy)
                    # save TensorFlow summary (contains at least the graph definition)
                    _ = tf.summary.FileWriter(folder, tf.get_default_graph())
            except:
                logger.error("Failed to load policy network weights from %s." % load_policy)
        if MPI.COMM_WORLD.Get_rank() == 0 and locals['iters_so_far'] % 50 == 0:
            print('Saving video and checkpoint for policy at iteration %i...' %
                  locals['iters_so_far'])
            ob = env.reset()
            images = []
            rewards = []
            max_reward = 1.  # if any reward > 1, we have to rescale
            lower_part = video_height // 4
            for i in range(1000):
                if isinstance(locals['pi'], GaussianMlpPolicy):
                    ac, _, _ = locals['pi'].act(np.concatenate((ob, ob)))
                else:
                    ac, _ = locals['pi'].act(False, ob)
                ob, rew, new, _ = env.step(ac)
                img = env.env.render(mode='rgb_array')
                if plot_rewards:
                    rewards.append(rew)
                    max_reward = max(rew, max_reward)
                images.append(img)
                if new:
                    break

            color = np.array([255, 163, 0])
            for i, img in enumerate(images):
                for j, r in enumerate(rewards[:i]):
                    rew_x = int(j / 1000. * video_width)
                    rew_y = int(r / max_reward * lower_part)
                    img[-lower_part, :10] = color
                    img[-lower_part, -10:] = color
                    img[-rew_y - 1:, rew_x] = color
            imageio.mimsave(
                os.path.join(folder, "videos", "%s_%s_iteration_%i.mp4" %
                             (environment, method, locals['iters_so_far'])),
                images,
                fps=60)
            env.reset()

            save_state(os.path.join(folder, "checkpoints", "%s_%i" %
                             (environment, locals['iters_so_far'])))

    if method == "ppo":
        pposgd_simple.learn(
            env,
            policy_fn,
            max_timesteps=int(num_timesteps),
            timesteps_per_actorbatch=1024,  # 256
            clip_param=0.2,
            entcoeff=0.01,
            optim_epochs=4,
            optim_stepsize=1e-3,  # 1e-3
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',  # 'linear'
            callback=callback)
    elif method == "trpo":
        trpo_mpi.learn(
            env,
            policy_fn,
            max_timesteps=int(num_timesteps),
            timesteps_per_batch=1024,
            max_kl=0.1,  # 0.01
            cg_iters=10,
            cg_damping=0.1,
            gamma=0.99,
            lam=0.98,
            vf_iters=5,
            vf_stepsize=1e-3,
            callback=callback)
    elif method == "acktr":
        with tf.Session(config=tf.ConfigProto()):
            ob_dim = env.observation_space.shape[0]
            ac_dim = env.action_space.shape[0]
            with tf.variable_scope("vf"):
                vf = NeuralNetValueFunction(ob_dim, ac_dim)
            with tf.variable_scope("pi"):
                policy = GaussianMlpPolicy(ob_dim, ac_dim)
            acktr.learn(
                env,
                pi=policy,
                vf=vf,
                gamma=0.99,
                lam=0.97,
                timesteps_per_batch=1024,
                desired_kl=0.01,  # 0.002
                num_timesteps=num_timesteps,
                animate=False,
                callback=callback)
    elif method == "ddpg":
        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = env.action_space.shape[-1]
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(
                    initial_stddev=float(stddev),
                    desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(
                    mu=np.zeros(nb_actions),
                    sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mu=np.zeros(nb_actions),
                    sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError(
                    'unknown noise type "{}"'.format(current_noise_type))

        # Configure components.
        memory = Memory(
            limit=int(1e6),
            action_shape=env.action_space.shape,
            observation_shape=env.observation_space.shape)
        critic = Critic(layer_norm=layer_norm)
        actor = Actor(nb_actions, layer_norm=layer_norm)

        ddpg_training.train(
            env=env,
            eval_env=None,
            param_noise=param_noise,
            render=False,
            render_eval=False,
            action_noise=action_noise,
            actor=actor,
            critic=critic,
            memory=memory,
            **kwargs)
    else:
        print('ERROR: Invalid "method" argument provided.', file=sys.stderr)
    env.close()


def main(**kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder_name = "{timestamp}-{kwargs[environment]}-{kwargs[method]}".format(
            timestamp=timestamp, kwargs=kwargs)
        folder_name = os.path.abspath(os.path.join(kwargs['logdir'], folder_name))
        pathlib.Path(folder_name, "videos").mkdir(parents=True, exist_ok=True)
        pathlib.Path(folder_name, "checkpoints").mkdir(
            parents=True, exist_ok=True)

        logger.configure(dir=folder_name, format_strs=['log', 'stdout'])

        run_json = {
            "time": timestamp,
            "settings": kwargs,
            "src_files": {
                "rl.py": "".join(open(sys.argv[0], "r"))  # TODO handle case with no sys.argv[0]
            }
        }

        # noinspection PyBroadException
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            run_json["git"] = {
                "head": repo.head.object.hexsha,
                "branch": str(repo.active_branch),
                "summary": repo.head.object.summary,
                "time": repo.head.object.committed_datetime.strftime("%Y-%m-%d-%H-%M-%S"),
                "author": {
                    "name": repo.head.object.author.name,
                    "email": repo.head.object.author.email
                }
            }
        except:
            print("Could not gather git repo information.", file=sys.stderr)

        json.dump(run_json, open(os.path.join(folder_name, "run.json"), "w"), indent=4)
    else:
        logger.configure(format_strs=[])
        folder_name = None

    def env_fn():
        ids = kwargs['environment'].split('-')
        framework = ids[0].lower()
        env_id = '-'.join(ids[1:])
        if framework == 'dm':
            from gail.envs.deepmind import DMSuiteEnv
            return DMSuiteEnv(env_id,
                              deterministic_reset=kwargs['deterministic_reset'],
                              render_camera=kwargs['render_camera'],
                              render_width=kwargs['video_width'],
                              render_height=kwargs['video_height'])
        elif framework == 'gym':
            return gym.make(env_id)
        elif framework == 'rllab':
            from gail.envs.rllab import RllabEnv
            return RllabEnv(env_id)

        raise LookupError("Could not find environment \"%s\"." % env_id)

    folder_name = MPI.COMM_WORLD.bcast(folder_name, root=0)
    train(env_fn=env_fn, folder=folder_name, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument(
        '--num-cpu', help='number of cpu to used', type=int, default=4)
    parser.add_argument(
        '--method',
        help='reinforcement learning algorithm to use (ppo/trpo/ddpg/acktr)',
        type=str,
        default='acktr')
    parser.add_argument(
        '--environment',
        help='environment ID prefixed by framework, e.g. dm-cartpole-swingup, gym-CartPole-v0, rllab-cartpole',
        type=str,
        default='dm-humanoid-run')
    # default='rllab-humanoid')

    parser.add_argument(
        '--logdir',
        help='folder where the logs will be stored',
        type=str,
        default='logs')

    # DDPG settings
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    boolean_flag(parser, 'layer-norm', default=True)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument(
        '--nb-epochs', type=int,
        default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument(
        '--nb-train-steps', type=int,
        default=50)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-eval-steps', type=int,
        default=100)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--nb-rollout-steps', type=int,
        default=100)  # per epoch cycle and MPI worker
    parser.add_argument(
        '--noise-type', type=str, default='adaptive-param_0.2'
    )  # choices are adaptive-param_xx, ou_xx, normal_xx, none

    # resolution of rendered videos
    parser.add_argument('--video-width', type=int, default=400)
    parser.add_argument('--video-height', type=int, default=400)
    boolean_flag(parser, 'plot-rewards', default=True)
    parser.add_argument('--render-camera', type=int, default=1)
    boolean_flag(parser, 'deterministic-reset', default=False)

    # load existing policy network weights
    parser.add_argument('--load-policy', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args))
