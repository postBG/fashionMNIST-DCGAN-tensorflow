import numpy as np
import matplotlib.pyplot as plt

from .fashion_mnist import IMAGE_SIZE


def gen_image(arr):
    two_d = np.reshape(arr, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
    img_plot = plt.imshow(two_d, interpolation='nearest')

    return img_plot


def show_image(arr):
    img_plot = gen_image(arr)
    plt.show(img_plot)
