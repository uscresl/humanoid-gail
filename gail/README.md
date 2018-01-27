# Generative Adversarial Imitation Learning

## Requirements
* Python 3
* DeepMind Control Suite
* OpenAI Gym
* OpenAI Baselines

## Reinforcement Learning
Train a CMU Humanoid from DeepMind Control Suite via PPO:
```sh
export NUM_CPU=8
PYTHONPATH=..:$PYTHONPATH mpirun -np $NUM_CPU python3 rl.py \
        --num-cpu $NUM_CPU \
        --method trpo \
        --domain humanoid \
        --task run \
        --num-timesteps 90000000
```

### Supporting rllab
rllab is provided as a git submodule which requires additional dependencies that can be installed via
```
pip install cached_property path.py mako
```

## TODO
* Try [PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py)
* Make DDPG compatible with callback function to visualize and save policy during training
