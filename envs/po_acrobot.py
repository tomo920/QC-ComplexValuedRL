import numpy as np
from numpy import pi
import sys
import os
sys.path.append(os.path.dirname(__file__))

from acrobot import Env

class PoEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        if config.is_continuous:
            self.observation_size = 2
        else:
            self.observation_size = config.n_equal_part**2
            if config.is_special_quantization:
                self.observation_size = 10**2
                q_rate = np.array([0., 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.])
                o_list = 2. * pi * q_rate - pi
                self.o1_list = o_list
                self.o2_list = o_list

    def get_observation(self):
        theta1 = self.env.state[0]
        theta2 = self.env.state[1]
        if self.config.is_continuous:
            return np.array([theta1, theta2])
        else:
            o1 = self.discret(theta1, self.o1_list)
            o2 = self.discret(theta2, self.o2_list)
            return o1 + 10 * (o2 - 1)
