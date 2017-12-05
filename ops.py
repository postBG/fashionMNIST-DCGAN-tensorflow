import tensorflow as tf

from data.fashion_mnist import IMAGE_PIXELS


def leaky_relu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)


def model_inputs(real_dim=IMAGE_PIXELS, z_dim=100):
    inputs_real = tf.placeholder(tf.float32, shape=[None, real_dim], name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, shape=[None, z_dim], name='inputs_z')

    return inputs_real, inputs_z


def batch_norm(inputs, training=True):
    return tf.layers.batch_normalization(inputs, training=training)


# TODO: How to collect this all variables
# TODO: Extract Hyper parameter
def generator(z, output_channel=1, reuse=False, training=True, kernel_size=4):
    """
    This function creates generator.
    
    :param z: random noise, ex) tensor shapes [None, 100]
    :param output_channel: number of channels of generated data, ex) 3 for SVHN, 1 for MNIST
    :param reuse: ...
    :param training: ...
    :param kernel_size: transposed conv layer's kernel size 
    
    :return: generated data(fake data) tensor, ex) tensor shapes [None, 28, 28, 1] for MNIST
    """
    with tf.variable_scope('generator', reuse=reuse):
        with tf.name_scope('layer1'):
            projected_z = tf.layers.dense(z, 7 * 7 * 256)
            reshaped_z = tf.reshape(projected_z, [-1, 7, 7, 256])
            layer1 = batch_norm(reshaped_z, training=training)
            layer1 = tf.nn.relu(layer1)

        with tf.name_scope('layer2'):
            layer2 = tf.layers.conv2d_transpose(layer1, 128, kernel_size, strides=2, padding='same')
            layer2 = batch_norm(layer2, training=training)
            layer2 = tf.nn.relu(layer2)

        with tf.name_scope('layer3'):
            layer3 = tf.layers.conv2d_transpose(layer2, 64, kernel_size, strides=2, padding='same')
            layer3 = batch_norm(layer3, training=training)
            layer3 = tf.nn.relu(layer3)

        with tf.name_scope('output'):
            logits = tf.layers.conv2d_transpose(layer3, output_channel, kernel_size, strides=1, padding='same')
            output = tf.nn.tanh(logits)

        return output


# TODO: Extract Hyper parameter and Care about initializer
def discriminator(images, reuse=False, alpha=0.2):
    """
    Create Discriminator
    
    :param images: tensor shapes [None, 28, 28, 1] 
    :param reuse: ...
    :param alpha: leaky relu alpha
    :return: discriminator output and logits
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('layer1'):
            layer1 = tf.layers.conv2d(images, 64, 4, strides=2, padding='same')
            layer1 = leaky_relu(layer1, alpha)
            # 14x14x64

        with tf.name_scope('layer2'):
            layer2 = tf.layers.conv2d(layer1, 128, 4, strides=2, padding='same')
            layer2 = batch_norm(layer2, training=True)
            layer2 = leaky_relu(layer2, alpha)
            # 7x7x128

        with tf.name_scope('layer3'):
            layer3 = tf.layers.conv2d(layer2, 256, 4, strides=2, padding='same')
            layer3 = batch_norm(layer3, training=True)
            layer3 = leaky_relu(layer3, alpha)
            # 4x4x256

        # TODO: Make robust to tensor shapes using tensor's get_shape method
        with tf.name_scope('output'):
            flatten = tf.reshape(layer3, [-1, 4 * 4 * 256])
            logits = tf.layers.dense(flatten, 1)
            output = tf.nn.sigmoid(logits)

        return output, logits
