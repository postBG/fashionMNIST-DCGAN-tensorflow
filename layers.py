import tensorflow as tf

from data.fashion_mnist import IMAGE_SIZE


def leaky_relu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)


def model_inputs(image_width=IMAGE_SIZE, image_height=IMAGE_SIZE, image_channels=1, z_dim=100):
    inputs_real = tf.placeholder(tf.float32, shape=[None, image_width, image_height, image_channels],
                                 name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, shape=[None, z_dim], name='inputs_z')

    return inputs_real, inputs_z


def batch_norm(inputs, training=True):
    return tf.layers.batch_normalization(inputs, training=training)