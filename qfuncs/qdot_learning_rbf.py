import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qdot_base import QdotBase
from rbfnet import RBFNet

class ComplexRBFNet():
    """
    RBF network class for representing complex value
    """

    def __init__(self, input_size, hidden_size, weight_lr, mu_lr, sigma_lr):
        self.real_part = RBFNet(input_size, hidden_size, weight_lr, mu_lr, sigma_lr)
        self.imaginary_part = RBFNet(input_size, hidden_size, weight_lr, mu_lr, sigma_lr)

    def outputs(self, input):
        real = self.real_part.outputs(input)
        imaginary = self.imaginary_part.outputs(input)
        return complex(real, imaginary)

    def update(self, input, td_error):
        error_real = td_error.real
        error_imag = td_error.imag
        self.real_part.update(input, error_real)
        self.imaginary_part.update(input, error_imag)

class Qfunc(QdotBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        # initialize Q network
        self.Q_network = []
        for _ in range(env.action_size):
            self.Q_network.append(ComplexRBFNet(env.observation_size, config.hidden_size, config.lr_o, config.lr_h, config.lr_h))

    def get_q_o(self, observation):
        return np.array([q_net.outputs(observation) for q_net in self.Q_network])

    def update_q(self, observation, action, q_target):
        td_error = q_target - self.get_q_o(observation)[action]
        self.Q_network[action].update(observation, td_error)
