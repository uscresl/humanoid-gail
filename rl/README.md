# Learning how to walk

## Getting started
First train a policy using
```bash
python3 run_rl.py
```
The resulting policy network is saved as TensorFlow checkpoint in the same path as a `*.ckpt` file.

To load a pretrained policy, issue
```bash
python3 run_rl.py --load humanoid_policy.ckpt
```