import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def get_model() -> tf.keras.Model:

    # Input layer
    input = tf.keras.layers.Input((48, 48, 3))

    # First CNN layer with 32 5x5 filters, LeakyReLU activation and InstanceNormalization on outputs.
    x = tf.keras.layers.Conv2D(32, 5, activation=None)(input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)

    # Then the next two layers reduce the kernel size to 3x3, doubling the number of filters.
    # Both are followed by MaxPooling2D operation.
    x = tf.keras.layers.Conv2D(64, 3, activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Same as before, now with 256 filter, no pooling.
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)

    # The last convolutional layer flatten the outputs using Global Max Pooling so it can be attached to the fully connected part of the network.
    x = tf.keras.layers.Conv2D(128, 3, activation=None, padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tfa.layers.InstanceNormalization(axis=3,
                                         center=True,
                                         scale=True,
                                         beta_initializer="random_uniform",
                                         gamma_initializer="random_uniform")(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Here starts the fully connected part. Two dense layer with 512 units each, LeakyReLU activation and Dropout to increase generalization capability.
    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(256, activation=None)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # The network outputs is a single unit containing the raw prediction. No activation function (i.e sigmoid) is applied. Tensorflow documentation report better performance (in terms of training stability) using a loss function that takes raw logits as inputs, like tf.keras.losses.BinaryCrossentropy(**from_logits=True**)
    output = tf.keras.layers.Dense(1)(x)

    baseline = tf.keras.Model(input, output)

    return baseline


def evaluate_model(model: tf.keras.Model, ds_test):
    print('-'*30 + 'PATCH_CAMELYON Evaluation' + '-'*30)

    # Extract the lables first to compute metrics
    y_test = np.concatenate([label for img, label in ds_test], axis=0)

    y_pred = model.predict(ds_test)
    # Convert raw logits to the target class
    y_pred = tf.round(tf.nn.sigmoid(y_pred)) 

    acc = np.sum(np.concatenate(y_pred) == y_test)/y_test.shape[0]

    test_error = 1 - acc
    print('Test acc:', acc)
    print(f"Test error [%]: {(test_error):.4%}")

    print(
        f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")
