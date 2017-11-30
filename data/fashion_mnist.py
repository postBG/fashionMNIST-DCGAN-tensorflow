import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset/fashionmnist/fashion-mnist_train.csv')

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


class FashionMNIST:
    def __init__(self):
        self.raw_data = pd.read_csv(DATA_PATH)
        self._images = self.raw_data.drop('label', axis=1).values
        self._labels = self.raw_data['label'].values

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
