from ops import model_loss
from train import model_optimizer


class DCGAN:
    def __init__(self, inputs_real, inputs_z, learning_rate, configs):
        self.inputs_real = inputs_real
        self.inputs_z = inputs_z
        self.learning_rate = learning_rate
        self.configs = configs

        self.d_loss, self.g_loss = model_loss(inputs_real, inputs_z,
                                              output_channel=1,
                                              kernel_size=self.configs.kernel_size,
                                              alpha=self.configs.alpha)
        self.d_optimizer, self.g_optimizer = model_optimizer(self.g_loss, self.d_loss,
                                                             learning_rate=self.learning_rate,
                                                             beta1=self.configs.beta1)
