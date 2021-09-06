import tensorflow as tf

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
    #label = tf.one_hot(label, 2)
    return tf.image.per_image_standardization(image), tf.cast(label, tf.float32)


def rescale(x, y):
    with tf.device("/cpu:0"):
        x = tf.image.resize(
            x, [SCALE_PATCH_CAMELYON, SCALE_PATCH_CAMELYON])
    return x, y


def test_patches(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, PATCH_PATCH_CAMELYON, PATCH_PATCH_CAMELYON)
    return x, y


def random_patches(x, y):
    return tf.image.random_crop(x, [PATCH_PATCH_CAMELYON, PATCH_PATCH_CAMELYON, 3]), y


def random_brightness(x, y):
    return tf.image.random_brightness(x, max_delta=MAX_DELTA), y


def random_contrast(x, y):
    return tf.image.random_contrast(x, lower=LOWER_CONTRAST, upper=UPPER_CONTRAST), y
