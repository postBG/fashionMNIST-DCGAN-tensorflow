import tensorflow as tf

from data.fashion_mnist import IMAGE_SIZE
from layers import model_inputs
from models import DCGAN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
tf.app.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.app.flags.DEFINE_float("alpha", 0.2, "Alpha for leaky relu [0.2]")
tf.app.flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
tf.app.flags.DEFINE_integer("kernel_size", 4, "The size of kernel [4]")
tf.app.flags.DEFINE_integer("z_dim", 100, "The dimension of noise z [100]")


def main(unused_args):
    inputs_real, inputs_z = model_inputs(IMAGE_SIZE, IMAGE_SIZE, 1, FLAGS.z_dim)
    dcgan = DCGAN(inputs_real, inputs_z, FLAGS)


if __name__ == '__main__':
    tf.app.run()
