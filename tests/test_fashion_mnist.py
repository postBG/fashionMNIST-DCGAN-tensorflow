import unittest
import numpy as np

from data.fashion_mnist import DCGANFashionMNIST, IMAGE_SIZE


class TestFashionMNIST(unittest.TestCase):
    mnist = DCGANFashionMNIST()

    def test_image_shape(self):
        first_image = self.mnist.images[0]
        self.assertTupleEqual((IMAGE_SIZE, IMAGE_SIZE, 1), first_image.shape,
                              'Incorrect Image Shape.  Found {} shape'.format(first_image.shape))

    def test_image_type(self):
        first_image = self.mnist.images[0]
        self.assertEquals(np.float32, first_image.dtype,
                          'Incorrect Image type.  Found {}'.format(type(first_image)))
