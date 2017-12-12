import tensorflow as tf

from ops import DISCRIMINATOR, GENERATOR


def model_optimizer(g_loss, d_loss, learning_rate, beta1):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith(DISCRIMINATOR)]
    g_vars = [var for var in t_vars if var.name.startswith(GENERATOR)]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_optimizer, g_optimizer


if __name__ == '__main__':
    tf.app.run()
