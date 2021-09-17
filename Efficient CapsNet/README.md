<h1 align="center">Orginal and Efficient CapsNet evaluation on patch_camelyon</h1>
This folder contains a modified version of <a href="https://github.com/EscVM/Efficient-CapsNet"> Efficient-CapsNet</a> (<a href="https://arxiv.org/abs/2101.12491"><img src=(http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)></a>) by Vittorio Mazzia and Francesco Salvetti, extended with the purpose of benchmarking the patch_camelyon dataset against the Original and Efficent version of CapsNet.<br><br>


# 1.0 Installation

There are two ways to prepare an environment with the required package:

## 1.1 Docker container
It's the **reccomended way** because it doesn't require to manually install any python package (even python itself isn't needed) and any Nvida CUDA toolkit (potentially messing up installation between different version, compatibility problems etc.) 
1. Install Docker following the [official guide](https://docs.docker.com/get-docker/).

2. For GPU support on Linux (and WSL2 in the future) [install NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker).

3. Now you can procede in two different ways:
    * (**Reccomended**) Open the project folder with Visual Studio Code, install the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension, than press F1 to show all commands, find and select Remote-Containers: Reopen in Container.
    * Or build and run the docker image manually.
      ```bash
      docker build -t EffCapsNet .
      docker run --gpus all -it --rm EffCapsNet
      ```
      In the terminal you should be prompted to open a link in the browser that redirect to a Jupyter Notebook environment.

## 1.2 PyPI

 Python3 and Tensorflow 2.x are required and should be installed on the host machine following the [official guide](https://www.tensorflow.org/install). 

1. Install the required packages
   ```bash
   pip3 install -r requirements.txt
   ```
Maybe setup a virtual environment to left unchanged your base python installation.

# 2.0 Notebooks
The provided notebooks are ready to start training and testing the two model versions with the patch_camelyon dataset. All the model's hyperparameters (like batch size, epochs, learning rate etc.) are stored in config.json
