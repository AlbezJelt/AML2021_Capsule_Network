import os

from numpy.lib.function_base import extract
from utils.pre_process_smallnorb import PATCH_SMALLNORB

import numpy as np
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tqdm.notebook import tqdm

# constants
INPUT_SHAPE = 96
SCALE_PATCH_CAMELYON = 64
PATCH_PATCH_CAMELYON = 48
N_CLASSES = 2
MAX_DELTA = 2.0
LOWER_CONTRAST = 0.5
UPPER_CONTRAST = 1.5


def pre_process(ds):
    return ds['image'], ds['label']


def normalize(image, label):
    label = tf.one_hot(label, 2)
    return tf.image.per_image_standardization(image), tf.cast(label, tf.float32)


def rescale(x, y):
    # return tf.image.resize(tensor['image'] , [48, 48]), tf.cast(tensor['label'], tf.float32)
    with tf.device("/cpu:0"):
        x = tf.image.resize(
            x, [SCALE_PATCH_CAMELYON, SCALE_PATCH_CAMELYON])
    return x, y
    # return tf.image.resize(tensor['image'] , [48, 48]), tensor['label']


def test_patches(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, PATCH_PATCH_CAMELYON, PATCH_PATCH_CAMELYON)
    return x, y


def generator(image, label):
    return (image, label), (label, image)


def random_patches(x, y):
    return tf.image.random_crop(x, [PATCH_PATCH_CAMELYON, PATCH_PATCH_CAMELYON, 3]), y


def random_brightness(x, y):
    return tf.image.random_brightness(x, max_delta=MAX_DELTA), y


def random_contrast(x, y):
    return tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST), y


def to_grayscale(x, y):
    return tf.image.rgb_to_grayscale(x), y


def generate_tf_data(dataset_train, dataset_val, batch_size):

    # =================== TRAINING SET ========================================

    # Extract image and label from tf.Dataset
    dataset_train = dataset_train.map(
        pre_process, num_parallel_calls=AUTOTUNE)

    # Rescale to SCALE_PATCH_CAMELYON
    dataset_train = dataset_train.map(
        rescale, num_parallel_calls=AUTOTUNE)

    # Random crop to PATCH_PATCH_CAMELYON
    dataset_train = dataset_train.map(random_patches,
                                      num_parallel_calls=AUTOTUNE)

    # Apply random brightness
    dataset_train = dataset_train.map(random_brightness,
                                      num_parallel_calls=AUTOTUNE)

    # Apply random contrast
    dataset_train = dataset_train.map(random_contrast,
                                      num_parallel_calls=AUTOTUNE)

    # Convert to grayscale
    # dataset_train = dataset_train.map(
    #     to_grayscale, num_parallel_calls=AUTOTUNE)

    # Standardize the image
    dataset_train = dataset_train.map(
        normalize, num_parallel_calls=AUTOTUNE)

    # Create the sample (image, label), (label, image)
    dataset_train = dataset_train.map(generator,
                                      num_parallel_calls=AUTOTUNE)

    # Batch
    dataset_train = dataset_train.batch(batch_size)

    # Prefetch
    dataset_train = dataset_train.prefetch(AUTOTUNE)

    # =========================================================================

    # =================== VALIDATION SET ======================================

    # Extract image and label from tf.Dataset
    dataset_val = dataset_val.map(
        pre_process, num_parallel_calls=AUTOTUNE)

    # Rescale to SCALE_PATCH_CAMELYON
    dataset_val = dataset_val.map(
        rescale, num_parallel_calls=AUTOTUNE)

    # Center crop to PATCH_PATCH_CAMELYON
    dataset_val = dataset_val.map(
        test_patches, num_parallel_calls=AUTOTUNE)

    # Convert to grayscale
    # dataset_test = dataset_test.map(
    #     to_grayscale, num_parallel_calls=AUTOTUNE)

    # Standardize the image
    dataset_val = dataset_val.map(
        normalize, num_parallel_calls=AUTOTUNE)

    # Create the sample (image, label), (label, image)
    dataset_val = dataset_val.map(generator,
                                  num_parallel_calls=AUTOTUNE)

    # Cache the data in memory. This dataset is smaller compared to the
    # training so it can be cached in memory for faster validation
    dataset_val = dataset_val.cache()

    # Batch
    dataset_val = dataset_val.batch(16)

    # Prefetch
    dataset_val = dataset_val.prefetch(AUTOTUNE)

    # =========================================================================

    return dataset_train, dataset_val


def generate_tf_test_data(dataset_test):

    # Extract image and label from tf.Dataset
    dataset_test = dataset_test.map(
        pre_process, num_parallel_calls=AUTOTUNE)

    # Rescale to SCALE_PATCH_CAMELYON
    dataset_test = dataset_test.map(
        rescale, num_parallel_calls=AUTOTUNE)

    # Center crop to PATCH_PATCH_CAMELYON
    dataset_test = dataset_test.map(
        test_patches, num_parallel_calls=AUTOTUNE)

    # Convert to grayscale
    # dataset_test = dataset_test.map(
    #     to_grayscale, num_parallel_calls=AUTOTUNE)

    # Standardize the image
    dataset_test = dataset_test.map(
        normalize, num_parallel_calls=AUTOTUNE)

    # Create the sample (image, label), (label, image)
    # dataset_test = dataset_test.map(generator,
    #                                 num_parallel_calls=AUTOTUNE)
    
    # Batch
    dataset_test = dataset_test.batch(32)

    # Prefetch
    dataset_test = dataset_test.prefetch(AUTOTUNE)

    return dataset_test
