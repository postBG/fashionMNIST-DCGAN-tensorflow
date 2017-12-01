import tensorflow as tf


class DCGAN:
    def __init__(self, inputs_real, inputs_z, configs):
        self.inputs_real = inputs_real
        self.inputs_z = inputs_z
        self.configs = configs
