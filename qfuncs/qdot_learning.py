import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qdot_base import QdotBase

class Qfunc(QdotBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        # initialize Q table
        self.Q = {}
        for observation in range(1, env.observation_size+1):
            self.Q[observation] = np.array([complex(0, 0) for _ in range(env.action_size)])

    def get_q_o(self, observation):
        return self.Q[observation]

    def update_q(self, observation, action, q_target):
        self.Q[observation][action] = self.Q[observation][action] \
            + self.config.alpha * (q_target - self.Q[observation][action])

    def get_params(self):
        return self.Q

    def set_params(self, params):
        for observation in range(1, self.env.observation_size+1):
            for action in range(self.env.action_size):
                self.Q[observation][action] = params.item()[observation][action]
