import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow.data import AUTOTUNE

from utils import preprocessing

this_file_path = pathlib.Path(__file__).parent.resolve()
base_path = this_file_path.parent.parent


def load_train_val_dataset():
    (ds_train, ds_valid), ds_info = tfds.load(
        'patch_camelyon',
        data_dir=f'{base_path}/data/tensorflow_dataset',
        download=False,
        split=['train', 'validation'],
        shuffle_files=True,
        as_supervised=False,
        with_info=True)

    return ds_train, ds_valid


def load_test_dataset():
    (ds_test), ds_info = tfds.load(
        'patch_camelyon',
        data_dir=f'{base_path}/data/tensorflow_dataset',
        download=False,
        split=['test'],
        shuffle_files=False,
        as_supervised=False,
        with_info=True)

    return ds_test[0]


def preprocess_train(ds_train, batch_size):
    # Extract image and label from tf.Dataset
    ds_train = ds_train.map(
        preprocessing.pre_process, num_parallel_calls=AUTOTUNE)

    # Rescale to SCALE_PATCH_CAMELYON
    ds_train = ds_train.map(
        preprocessing.rescale, num_parallel_calls=AUTOTUNE)

    # Random crop to PATCH_PATCH_CAMELYON
    ds_train = ds_train.map(preprocessing.random_patches,
                            num_parallel_calls=AUTOTUNE)

    # Apply random brightness
    ds_train = ds_train.map(preprocessing.random_brightness,
                            num_parallel_calls=AUTOTUNE)

    # Apply random contrast
    ds_train = ds_train.map(preprocessing.random_contrast,
                            num_parallel_calls=AUTOTUNE)

    # Standardize the image
    ds_train = ds_train.map(
        preprocessing.normalize, num_parallel_calls=AUTOTUNE)

    # Batch
    ds_train = ds_train.batch(batch_size)

    # Prefetch
    ds_train = ds_train.prefetch(AUTOTUNE)

    return ds_train


def preprocess_validation(ds_val):
    # Extract image and label from tf.Dataset
    ds_val = ds_val.map(
        preprocessing.pre_process, num_parallel_calls=AUTOTUNE)

    # Rescale to SCALE_PATCH_CAMELYON
    ds_val = ds_val.map(
        preprocessing.rescale, num_parallel_calls=AUTOTUNE)

    # Center crop to PATCH_PATCH_CAMELYON
    ds_val = ds_val.map(preprocessing.test_patches,
                            num_parallel_calls=AUTOTUNE)

    # Standardize the image
    ds_val = ds_val.map(
        preprocessing.normalize, num_parallel_calls=AUTOTUNE)

    # Cache the data in memory. This dataset is smaller compared to the
    # training so it can be cached in memory for faster validation
    ds_val = ds_val.cache()

    # Batch
    ds_val = ds_val.batch(32)

    # Prefetch
    ds_val = ds_val.prefetch(AUTOTUNE)

    return ds_val
