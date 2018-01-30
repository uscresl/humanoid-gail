from collections import namedtuple

import gym, sys, traceback
import numpy as np
from dm_control import suite
from dm_control.mujoco.wrapper import mjbindings
from gym.spaces import Box


class DMSuiteEnv(gym.Env):
    def __init__(self,
                 id="cartpole-swingup",
                 visualize_reward=True,
                 deterministic_reset=False,
                 render_camera=1,
                 render_width=400,
                 render_height=400):
        domain_name, task_name = tuple(id.split('-'))
        self.dm_env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            visualize_reward=visualize_reward)
        action_spec = self.dm_env.action_spec()
        self.action_space = Box(
            low=action_spec.minimum[0],
            high=action_spec.maximum[0],
            shape=action_spec.shape)
        self.deterministic_reset = deterministic_reset
        try:
            ob_spec = self.dm_env.task.observation_spec(self.dm_env.physics)
            self.observation_space = gym.spaces.Box(
                low=ob_spec.minimum[0],
                high=ob_spec.maximum[0],
                shape=ob_spec.shape)
        except NotImplementedError:
            print(
                "Could not retrieve observation spec, min/max possibly incorrect.",
                file=sys.stderr)
            # sample observation and set range to [-10, 10]
            # ob = self.dm_env.task.get_observation(self.dm_env.physics)
            # # ob is an OrderedDict, iterate over all entries to determine overall flattened ob dim
            # ob_dimension = 0
            # for entry in ob.values():
            #     ob_dimension += len(entry.flatten())
            ob = self.observe()
            ob_dimension = len(ob)
            self.observation_space = gym.spaces.Box(
                low=-10, high=10, shape=(ob_dimension, ))
        self.reward_range = (0, 1)
        print('Initialized %s: %s.' % (domain_name, task_name))
        print('\tobservation space: %s (min: %.2f, max: %.2f)' %
              (str(self.observation_space.shape),
               self.observation_space.low[0], self.observation_space.high[0]))
        print('\taction space: %s (min: %.2f, max: %.2f)' %
              (str(self.action_space.shape), self.action_space.low[0],
               self.action_space.high[0]))

        self.render_camera = render_camera
        self.render_width = render_width
        self.render_height = render_height

        env_spec = namedtuple('env_spec', ['id', 'timestep_limit'])
        self._spec = env_spec(id=id, timestep_limit=1000)

    def _step(self, action):
        # noinspection PyBroadException
        try:
            step = self.dm_env.step(action)
            done = step.last()
            reward = step.reward
            if reward is None:
                reward = 0

            physics = self.dm_env.physics

            # stop episode when falling
            _STAND_HEIGHT = 1.4
            # done = physics.head_height() < _STAND_HEIGHT/4.

            # compute custom reward for standing task
            # TODO remove
            # from dm_control.utils import rewards
            # physics = self.dm_env.physics
            # _STAND_HEIGHT = 1.4
            # standing = rewards.tolerance(physics.head_height(),
            #                              bounds=(_STAND_HEIGHT, float('inf')),
            #                              margin=_STAND_HEIGHT)
            # upright = rewards.tolerance(physics.torso_upright(),
            #                             bounds=(0.9, float('inf')), sigmoid='linear',
            #                             margin=1.9, value_at_margin=0)
            # stand_reward = standing * upright
            # small_control = rewards.tolerance(physics.control(), margin=1,
            #                                   value_at_margin=0,
            #                                   sigmoid='quadratic').mean()
            # small_control = (4 + small_control) / 5
            # reward = small_control * stand_reward
        except:
            # could only be dm_control.rl.control.PhysicsError?
            # reset environment for bad controls
            print(traceback.format_exc(), file=sys.stderr)
            self.dm_env.reset()
            done = True
            reward = 0

        ob = self.observe()
        return ob, reward, done, {}

    def _reset(self):
        self.dm_env.reset()

        if self.deterministic_reset:
            hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
            slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
            ball = mjbindings.enums.mjtJoint.mjJNT_BALL
            free = mjbindings.enums.mjtJoint.mjJNT_FREE

            physics = self.dm_env.physics

            for joint_id in range(physics.model.njnt):
                joint_name = physics.model.id2name(joint_id, 'joint')
                joint_type = physics.model.jnt_type[joint_id]
                is_limited = physics.model.jnt_limited[joint_id]
                range_min, range_max = physics.model.jnt_range[joint_id]

                if is_limited:
                    if joint_type == hinge or joint_type == slide:
                        self.dm_env.physics.named.data.qpos[
                            joint_name] = np.mean([
                                range_min, range_max
                            ])  # random.uniform(range_min, range_max)

                    elif joint_type == ball:
                        self.dm_env.physics.named.data.qpos[
                            joint_name] = np.mean([
                                range_min, range_max
                            ])  # random_limited_quaternion(random, range_max)

                else:
                    if joint_type == hinge:
                        self.dm_env.physics.named.data.qpos[joint_name] = 0.  # random.uniform(-np.pi, np.pi)

                    elif joint_type == ball:
                        quat = np.zeros(4)  # random.randn(4)
                        # quat /= np.linalg.norm(quat)
                        self.dm_env.physics.named.data.qpos[joint_name] = quat

                    elif joint_type == free:
                        quat = np.zeros(4)  # random.rand(4)
                        # quat /= np.linalg.norm(quat)
                        self.dm_env.physics.named.data.qpos[joint_name][3:] = quat
            self.dm_env.physics.after_reset()

        return self.observe()

    def _seed(self, seed=None):
        self.dm_env.random = np.random.RandomState(seed)
        return [seed]

    def observe(self):
        src_ob = self.dm_env.task.get_observation(self.dm_env.physics)
        ob = np.hstack(entry.flatten() for entry in src_ob.values())
        # # todo revert to DeepMind's ob
        # ob = np.concatenate([
        #     self.dm_env.physics.data.qpos[:].flat,
        #     self.dm_env.physics.data.qvel[:].flat,
        #     np.clip(self.dm_env.physics.data.cfrc_ext[:], -1, 1).flat,
        #     self.dm_env.physics.center_of_mass_position().flat,
        # ])
        return ob

    def render(self, mode='rgb_array', close=False):
        if mode != 'rgb_array':
            return
            # raise NotImplementedError('Render mode %s not implemented for the DM Control Suite environment.' % mode)

        return self.dm_env.physics.render(self.render_width, self.render_height, camera_id=self.render_camera)
