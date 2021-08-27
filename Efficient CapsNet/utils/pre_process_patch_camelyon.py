import numpy as np
import tensorflow as tf
import os
from tqdm.notebook import tqdm


# constants
INPUT_SHAPE = 48
PATCH_PCAMELYON = 32
N_CLASSES = 5
MAX_DELTA = 2.0
LOWER_CONTRAST = 0.5
UPPER_CONTRAST = 1.5
PARALLEL_INPUT_CALLS = 16

# Unused, pass directly with tf.data.Dataset
def pre_process(ds):
    SAMPLES = int(ds.cardinality())

    X = np.empty((SAMPLES, INPUT_SHAPE, INPUT_SHAPE, 3))
    y = np.empty((SAMPLES,))

    with tf.device("/cpu:0"):    
        for index, d in tqdm(enumerate(ds.batch(1))):           
            img = tf.image.resize(d['image'] , [48, 48])
            X[index, :, :, :] = img / 255.0
            y[index] = d['label']
    return X, y

def normalize(image, label):
    label = tf.one_hot(label, 2)
    return tf.image.per_image_standardization(image), tf.cast(label, tf.float32)

def rescale(tensor):
    # return tf.image.resize(tensor['image'] , [48, 48]), tf.cast(tensor['label'], tf.float32)
    return tf.image.resize(tensor['image'] , [48, 48]), tensor['label']
# ?
def test_patches(x, y, config):
    res = (config['scale_smallnorb'] - config['patch_smallnorb']) // 2
    return x[:,res:-res,res:-res,:], y

def generator(image, label):
    return (image, label), (label, image)

def random_patches(x, y):
    return tf.image.random_crop(x, [PATCH_PCAMELYON, PATCH_PCAMELYON, 3]), y

def random_brightness(x, y):
    return tf.image.random_brightness(x, max_delta=MAX_DELTA), y

def random_contrast(x, y):
    return tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST), y

def generate_tf_data(dataset_train, dataset_test, batch_size):
    
    # Test set
    # dataset_train = dataset_train.map(random_patches,
    #     num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(rescale, num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(normalize, num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_brightness,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(random_contrast,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    # Random cropping is implemented as a layer
    # dataset_train = dataset_train.map(random_patches, num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(-1)

    # Validation set
    dataset_test = dataset_test.cache()
    dataset_test = dataset_test.map(rescale, num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.map(normalize, num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(1)
    dataset_test = dataset_test.prefetch(-1)
    
    return dataset_train, dataset_test
