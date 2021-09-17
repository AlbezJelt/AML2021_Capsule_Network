# "Matrix Capsules with EM Routing" evaluation on patch_camelyon

This folder contains a modified version of [Implementation of "Matrix Capsules with EM Routing"](https://github.com/IBM/matrix-capsules-with-em-routing) by [Ashley Gritzman](https://github.com/ashleygritzman) from IBM Research AI. It's the best tensorflow implementation (in terms of test accuracy) for the network described in [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) by Geoffrey Hinton, Sara Sabour and Nicholas Frosst.

The original implementation has been extended with the following goals:
1. Benchmark the model against patch_camelyon dataset.
2. Improve the code structure (with python package style in mind).
3. Make the model usable from a Jupyter Nootebok.
4. Standardize the code to a more recent Tensorflow version (from 1.10.0 to 1.15.3).

All credits for the model implementation go to Ashley Gritzman, check out the original github repo (linked above) and the [paper](https://arxiv.org/pdf/1907.00652.pdf) and [blog](https://medium.com/@ashleygritzman/available-now-open-source-implementation-of-hintons-matrix-capsules-with-em-routing-e5601825ee2a).

# 1.0 Installation

There are two ways to prepare an environment with the required package:

## 1.1 Nvidia TF1 docker container
It's the **reccomended way** for multiple reasons:
- It comes with all packages and dipendencies installed, including Tensorflow, Nvida CUDA and the Jupyter kernel.
- **Tensorflow 1.15.\* is builded around CUDA 10.\* and can cause some compatibility issue with the most modern GPUs (like the ones based on Ampere architecture)**. Nvidia provides ad-hoc containers with a custom version of Tensorflow which provides best compatibility.

The only downside is that the image is very big in terms of size (~6Gb of space required).

Proceed in the following way:
1. Install Docker following the [official guide](https://docs.docker.com/get-docker/).

2. [Install NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker).

3. Now you can procede in two different ways:
    * (**Reccomended**) Open the project folder with Visual Studio Code, install the [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension, than press F1 to show all commands, find and select Remote-Containers: Reopen in Container.
    * Or build and run the docker image manually.
      ```bash
      docker build -t EM-Routing .
      docker run --gpus all -it --rm EM-Routing
      ```
      Than from the container's bash open a Jupyter Notebook session.

## 1.2 PyPI

 Python3 and Tensorflow 1.15.3 are required and should be installed on the host machine following the [official guide](https://www.tensorflow.org/install). 

1. Install the required packages
   ```bash
   pip3 install -r requirements.txt
   ```
Maybe setup a virtual environment to left unchanged your base python installation.

# 2.0 Notebooks
The provided notebooks are ready to start training and testing the model with the patch_camelyon dataset. All the model's hyperparameters (like batch size, epochs, learning rate etc.) are stored in *src/utils/config.py*

During the training the model is saved in *logs* folder and can be used to restore a stopped training session or to load a fully trained model for testing purpose.