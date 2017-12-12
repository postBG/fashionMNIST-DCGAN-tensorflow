import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")


def main(unused_args):
    pass


if __name__ == '__main__':
    tf.app.run()
