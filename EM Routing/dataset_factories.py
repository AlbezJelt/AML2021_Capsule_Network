from config import FLAGS
import data_pipeline_norb as data_norb
import data_pipeline_pc as data_pc
import models as mod

# ------------------------------------------------------------------------------
# FACTORIES FOR DATASET
# ------------------------------------------------------------------------------
def get_dataset_path(dataset_name: str):
    options = {'smallNORB': 'data/smallNORB/tfrecord',
               'PatchCamelyon': 'tensorflow_datasets/patch_camelyon/2.0.0'}  # Patch camelyon dataset path
    path = FLAGS.storage + options[dataset_name]
    return path


def get_dataset_size_train(dataset_name: str):
    options = {'mnist': 55000,
               'smallNORB': 23400 * 2,
               'fashion_mnist': 55000,
               'cifar10': 50000,
               'cifar100': 50000,
               'PatchCamelyon': 4096 * 64}
    return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
    options = {'mnist': 10000,
               'smallNORB': 23400 * 2,
               'fashion_mnist': 10000,
               'cifar10': 10000,
               'cifar100': 10000,
               'PatchCamelyon': 4096 * 8}
    return options[dataset_name]


def get_dataset_size_validate(dataset_name: str):
    options = {'smallNORB': 23400 * 2,
               'PatchCamelyon': 4096 * 8}
    return options[dataset_name]


def get_num_classes(dataset_name: str):
    options = {'mnist': 10,
               'smallNORB': 5,
               'fashion_mnist': 10,
               'cifar10': 10,
               'cifar100': 100,
               'PatchCamelyon': 2}
    return options[dataset_name]


def get_create_inputs(dataset_name: str, mode="train"):

    if mode == "train":
        is_train = True
    else:
        is_train = False

    path = get_dataset_path(dataset_name)

    options = {'smallNORB': lambda: data_norb.create_inputs_norb(path, is_train),
               'PatchCamelyon': lambda: data_pc.create_inputs_PC(path, is_train)}
    return options[dataset_name]


def get_dataset_architecture(dataset_name: str):
    options = {'smallNORB': mod.build_arch_smallnorb,
               'PatchCamelyon': mod.build_arch_smallnorb, # Using the same architecture for PatchCamelyon
               'baseline': mod.build_arch_baseline}
    return options[dataset_name]
