# Generative Adversarial Imitation Learning

## Requirements
* Python 3
* DeepMind Control Suite
* OpenAI Gym
* OpenAI Baselines

## Run
```python
export NUM_CPU=8
PYTHONPATH=..:$PYTHONPATH mpirun -np $NUM_CPU python3 rl.py --num_cpu $NUM_CPU
```