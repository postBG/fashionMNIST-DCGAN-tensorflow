import tensorflow as tf
import unittest

from ops import model_inputs
from data.fashion_mnist import IMAGE_PIXELS


class TestLayers(unittest.TestCase):
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
