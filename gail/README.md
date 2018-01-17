# Generative Adversarial Imitation Learning

## Requirements
* Python 3
* DeepMind Control Suite
* OpenAI Gym
* OpenAI Baselines

## Run
Train a CMU Humanoid from DeepMind Control Suite via PPO:
```sh
export NUM_CPU=8
PYTHONPATH=..:$PYTHONPATH mpirun -np $NUM_CPU python3 rl.py \
        --num_cpu $NUM_CPU \
        --method trpo \
        --domain humanoid \
        --task run \
        --num-timesteps 90000000
```

## TODO
* Try [PPO2](https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py)
