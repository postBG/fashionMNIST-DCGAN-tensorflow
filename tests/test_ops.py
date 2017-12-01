import tensorflow as tf
import unittest

from ops import model_inputs, leaky_relu
from data.fashion_mnist import IMAGE_PIXELS


class TestModelInputs(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_model_inputs(self):
        inputs_real, inputs_z = model_inputs(real_dim=IMAGE_PIXELS, z_dim=100)

        self.assertEquals([None, IMAGE_PIXELS], inputs_real.get_shape().as_list(),
                          'Incorrect Image Shape.  Found {} shape'.format(inputs_real.get_shape().as_list()))
        self.assertEquals('inputs_real:0', inputs_real.name,
                          'Incorrect Name.  Found {}'.format(inputs_real.name))

        self.assertEquals([None, 100], inputs_z.get_shape().as_list(),
                          'Incorrect Image Shape.  Found {} shape'.format(inputs_z.get_shape().as_list()))
        self.assertEquals('inputs_z:0', inputs_z.name,
                          'Incorrect Name.  Found {}'.format(inputs_z.name))


class TestLeakyRelu(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_identity_when_input_is_bigger_than_zero(self):
        inputs = tf.constant(1.)
        leaky = leaky_relu(inputs, 0.2)

        with tf.Session() as sess:
            output = sess.run(leaky)
            self.assertEquals(1., output,
                              'Incorrect Value. Found {}'.format(output))

    def test_identity_when_input_is_zero(self):
        inputs = tf.constant(0.)
        leaky = leaky_relu(inputs, 0.2)

        with tf.Session() as sess:
            output = sess.run(leaky)
            self.assertEquals(0., output,
                              'Incorrect Value. Found {}'.format(output))

    def test_leak_when_input_is_smaller_than_zero(self):
        inputs = tf.constant(-1.)
        leaky = leaky_relu(inputs, 0.2)

        with tf.Session() as sess:
            output = sess.run(leaky)
            self.assertAlmostEqual(-0.2, output,
                                   delta=0.0001,
                                   msg='Incorrect Value. Found {}'.format(output))
