from .humanoid import HumanoidFeatureEnv
from gym.envs.registration import registry, register, make, spec

register(
    id='HumanoidFeaturized-v1',
    entry_point='featurized_env:HumanoidFeatureEnv',
    max_episode_steps=1000,
)