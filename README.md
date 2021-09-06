# Capsule Network Tests on patch_camelyon dataset
Testing different CapsNet implementation using [patch_camelyon dataset from tensorflow_dataset](https://www.tensorflow.org/datasets/catalog/patch_camelyon).

Tested implementation:
- [Implementation of "Matrix Capsules with EM Routing"](https://github.com/IBM/matrix-capsules-with-em-routing) by [Ashley Gritzman](https://github.com/ashleygritzman).
- [Efficient-CapsNet](https://github.com/EscVM/Efficient-CapsNet) by [Vittorio Mazzia](https://github.com/EscVM) and [Francesco Salvetti](https://github.com/fsalv). They implemented the original CapsNet from Hinton and a new version that exploit self-attention as routing mechanism.

## Dataset
It consists of 327.680 RGB color images (96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue. It comes alredy splitted in training, validation and test set.

Tensorflow_dataset provide a [very interesting playground](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=patch_camelyon) to visualize and interact with the dataset.
## Download the dataset
1. Clone the git repository as [ZIP](https://github.com/AlbezJelt/AML2021_Capsule_Network/archive/refs/heads/main.zip) or with the command line:
   ```bash
   git clone https://github.com/AlbezJelt/AML2021_Capsule_Network
   ```
2. Install the required packages:
   ```bash
   pip install tensorflow tensorflow-datasets
   ```
3. Run the dedicated download script:
   ```bash
   cd AML2021_Capsule_Network
   python3 /data/download_dataset.py
   ```

## Installation
the various models require different versions of the same packages in order to run. Please refer to READMEs in dedicated folders (Efficient CapsNet and EM Routing).