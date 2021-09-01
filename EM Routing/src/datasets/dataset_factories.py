from src.utils.config import FLAGS
from src.datasets import data_pipeline_pc as data_pc
from src.models import model as mod

# ------------------------------------------------------------------------------
# FACTORIES FOR DATASET
# ------------------------------------------------------------------------------


def get_dataset_path(dataset_name: str):
    # Patch camelyon dataset path
    options = {'PatchCamelyon': 'tensorflow_datasets/patch_camelyon/2.0.0'}
    path = FLAGS.storage + options[dataset_name]
    return path


def get_dataset_size_train(dataset_name: str):
    options = {'PatchCamelyon': 4096 * 64}
    return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
    options = {'PatchCamelyon': 4096 * 8}
    return options[dataset_name]


def get_dataset_size_validate(dataset_name: str):
    options = {'PatchCamelyon': 4096 * 8}
    return options[dataset_name]


def get_num_classes(dataset_name: str):
    options = {'PatchCamelyon': 2}
    return options[dataset_name]


def get_create_inputs(dataset_name: str, mode="train"):

    path = get_dataset_path(dataset_name)

    options = {
        'PatchCamelyon': lambda: data_pc.create_inputs_PC(path, mode)}
    return options[dataset_name]


def get_dataset_architecture(dataset_name: str):
    options = {'PatchCamelyon': mod.build_arch_smallnorb}
    return options[dataset_name]
