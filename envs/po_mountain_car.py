import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from mountain_car import Env

class PoEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        if config.is_continuous:
            self.observation_size = 1
        else:
            self.observation_size = config.n_equal_part

    def get_observation(self):
        position = self.env.state[0]
        if self.config.is_continuous:
            if self.config.algorithm == 'qdot_learning_qc':
                position = 2 * ((position - self.o_min) / (self.o_max - self.o_min)) - 1
            return np.array([position])
        else:
            o1 = self.discret(position, self.o1_list)
            if self.config.algorithm == 'qdot_learning_qc':
                return [o1]
            else:
                return o1
