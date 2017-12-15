import os
import pandas as pd
import numpy as np

# We Don't need to use test data
DATA_PATH = os.path.join(os.path.dirname(__file__), '../dataset/fashionmnist/fashion-mnist_train.csv')

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# TODO: Normalize images between -1 to 1
class FashionMNIST:
    def __init__(self, shuffle=True):
        self.loaded_data = pd.read_csv(DATA_PATH)
        if shuffle:
            self.loaded_data = self.loaded_data.sample(frac=1).reset_index(drop=True)

        self._images = self.loaded_data.drop('label', axis=1).values\
            .reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1)).astype(np.float32)
        self._labels = self.loaded_data['label'].values

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels


class DCGANFashionMNIST(FashionMNIST):
    def __init__(self):
        super().__init__()

    def to_batches(self, batch_size=128):
        batch_num = len(self.labels) // batch_size
        for i in range(0, len(self.labels), batch_num):
            yield self.images[i:i + batch_size].astype(np.float32)


if __name__ == "__main__":
    mnist = DCGANFashionMNIST()
    batches = mnist.to_batches(batch_size=300)
