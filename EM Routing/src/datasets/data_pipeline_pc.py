import os
import re

import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from src.utils.config import FLAGS


def _parser(serialized_example):
    """Parse PatchCamelyon example from tfrecord.

    Args:
      serialized_example: serialized example from tfrecord
    Returns:
      img: image
      lab: label
    """

    # From patch_camelyon feature.json
    # features:
    # 	- "id" <tf.string>: id of the sample
    # 	- "image" <tf.string>: uint8-encoded pngs, stored as string
    # 	- "label" <tf.int64>: class of the sample, {0, 1}
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'id': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        })

    # Decode the image
    img = tf.image.decode_png(features['image'])
    # img = tf.cast(img, tf.float32)

    lab = tf.cast(features['label'], tf.int32)

    # sample id isn't usefull
    return img, lab


def _train_preprocess(img, lab):
    """Preprocessing for training.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped images.
    During test, we crop a 32 × 32 patch from the center of the image and
    achieve..."

    Args:
      img: this fn only works on the image
      lab: label
    Returns:
      img: image processed
      lab: label
    """

    img = tf.image.resize(img, [48, 48])
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_brightness(img, max_delta=2.0)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.per_image_standardization(img)

    return img, lab


def _test_and_val_preprocess(img, lab):
    """Preprocessing for validation/testing.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped
    images. During test, we crop a 32 × 32 patch from the center of the image
    and achieve..."

    Args:
      img: this fn only works on the image
      lab: label
    Returns:
      img: image processed
      lab: label
    """

    img = tf.image.resize(img, [48, 48])
    img = tf.slice(img, [8, 8, 0], [32, 32, 3])
    img = tf.image.per_image_standardization(img)

    return img, lab


def input_fn(path, mode='train'):
    """Input pipeline for PatchCamelyon using tf.data.

    Args:
      is_train:
    Returns:
      dataset: image tf.data.Dataset
    """

    if mode == 'train':
        CHUNK_RE = re.compile(
            r"patch_camelyon-train\.tfrecord-[0-9]+-of-[0-9]+")
    elif mode == 'validate':
        CHUNK_RE = re.compile(
            r"patch_camelyon-validation\.tfrecord-[0-9]+-of-[0-9]+")
    elif mode == 'test':
        CHUNK_RE = re.compile(
            r"patch_camelyon-test\.tfrecord-[0-9]+-of-[0-9]+")

    chunk_files = [os.path.join(path, fname)
                   for fname in os.listdir(path)
                   if CHUNK_RE.match(fname)]

    # 1. Create the dataset
    dataset = tf.data.TFRecordDataset(chunk_files)

    # 2. Map with the actual work (preprocessing, augmentation…) using multiple
    # parallel calls
    dataset = dataset.map(_parser, num_parallel_calls=AUTOTUNE)
    if mode == 'train':
        dataset = dataset.map(_train_preprocess,
                              num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(_test_and_val_preprocess,
                              num_parallel_calls=AUTOTUNE)

    # 3. Shuffle (with a big enough buffer size)
    capacity = 2000 + 3 * FLAGS.batch_size
    dataset = dataset.shuffle(buffer_size=capacity)

    # 4. Batch
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    # 5. Repeat
    dataset = dataset.repeat(count=FLAGS.epoch)

    # 6. Prefetch
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def create_inputs_PC(path, mode='train'):
    """Get a batch from the input pipeline.

    Args: 
      is_train:  
    Returns:
      img, lab
    """

    # Create batched dataset
    dataset = input_fn(path, mode)

    # Create one-shot iterator
    iterator = dataset.make_one_shot_iterator()

    img, lab = iterator.get_next()

    output_dict = {'image': img,
                   'label': lab}

    return output_dict
