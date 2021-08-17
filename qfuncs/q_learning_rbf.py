import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from q_base import QBase
from rbfnet import RBFNet

class Qfunc(QBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        # initialize Q network
        self.Q_network = []
        for _ in range(env.action_size):
            self.Q_network.append(RBFNet(env.observation_size, config.hidden_size, config.lr_o, config.lr_h, config.lr_h))

    def get_q_o(self, observation):
        return np.array([q_net.outputs(observation) for q_net in self.Q_network])

    def update_q(self, observation, action, q_target):
        td_error = q_target - self.get_q_o(observation)[action]
        self.Q_network[action].update(observation, td_error)
