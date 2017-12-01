import tensorflow as tf

from data.fashion_mnist import IMAGE_PIXELS


def model_inputs(real_dim=IMAGE_PIXELS, z_dim=100):
    inputs_real = tf.placeholder(tf.float32, shape=[None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, shape=[None, z_dim], name='inputs_z')

    return inputs_real, inputs_z
