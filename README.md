# humanoid-gail
Humanoid behavior imitation using Generative Adversarial Imitation Learning (GAIL)

# Installation
1. Install Docker (We recommend the free CE edition): for [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/), for [Mac](https://docs.docker.com/docker-for-mac/install/), for [Windows](https://docs.docker.com/docker-for-windows/install/)
2. Build the Docker file:

    If your system does not have a NVIDIA® GPU that meets the [prerequisites](https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support):
    ```
    docker build -f Dockerfile -t uscresl/deep-rl-docker:tf1.3.0-gym0.9.3-baselines0.1.4-py3 .
    ```

    Else:
    ```
    docker build -f Dockerfile.gpu -t uscresl/deep-rl-docker:tf1.3.0-gym0.9.3-baselines0.1.4-gpu-py3 .
    ```
3. Execute

On macOS in terminal depending on whether you have a GPU version or not:
```
sh run.sh
```
or
```
sh run_gpu.sh
```

# Style
Please ensure that all your Python code conforms to the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard. If your question is not answered by PEP8, please revert to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Similarly, shell scripts should use the [Google Shell Style Guide](https://google.github.io/styleguide/shell.xml).
