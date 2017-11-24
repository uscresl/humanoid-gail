# Learning how to walk

## Requirements
* Python 3.x
* Baselines (tested on GitHub state from 11/01/2017)
* OpenAI Gym 0.9.4
* Mujoco 1.31
* mujoco-py 0.5.7
* TensorFlow 1.x

## Getting started
First train a policy using
```bash
python3 run_rl.py
```
The resulting policy network is saved as TensorFlow checkpoint in the same path as a `*.ckpt` file.

To load a pretrained policy, issue
```bash
CUDA_VISIBLE_DEVICES="" python3 run_rl.py --load humanoid_policy.ckpt
```

## Example
Humanoid running in MuJoCo:
```bash
CUDA_VISIBLE_DEVICES="" python3 run_rl.py --env Humanoid-v1 --hidden-size 32 --load examples/trpo.Humanoid.0.00-10100
```
