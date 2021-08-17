import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from q_base import QBase

class Qfunc(QBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        # initialize Q table
        self.Q = {}
        for observation in range(1, env.observation_size+1):
            self.Q[observation] = np.zeros(env.action_size)

    def get_q_o(self, observation):
        return self.Q[observation]

    def update_q(self, observation, action, q_target):
        self.Q[observation][action] = self.Q[observation][action] \
            + self.config.alpha * (q_target - self.Q[observation][action])
