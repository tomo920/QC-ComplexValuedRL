import gym
import numpy as np
from numpy import cos, pi
import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_base import EnvBase

h0 = 0.5    # The goal height the agent should swing up the end effector.

class Env(EnvBase):
    def __init__(self, config):
        self.env = gym.make('Acrobot-v1').env
        self.env.seed(config.seed)
        if config.is_continuous:
            observation_size = 4
        else:
            observation_size = config.n_equal_part**4
            self.o1_list = np.linspace(-pi, pi, config.n_equal_part+1)
            self.o2_list = np.linspace(-pi, pi, config.n_equal_part+1)
            self.o3_list = np.linspace(-self.env.MAX_VEL_1, self.env.MAX_VEL_1, config.n_equal_part+1)
            self.o4_list = np.linspace(-self.env.MAX_VEL_2, self.env.MAX_VEL_2, config.n_equal_part+1)
        action_size = 3
        max_step = config.max_steps
        action_list = self.env.AVAIL_TORQUE
        super().__init__(config, observation_size, action_size, max_step, action_list)

    def get_observation(self):
        theta1 = self.env.state[0]
        theta2 = self.env.state[1]
        thetadot1 = self.env.state[2]
        thetadot2 = self.env.state[3]
        if self.config.is_continuous:
            return np.array([theta1, theta2, thetadot1, thetadot2])
        else:
            o1 = self.discret(theta1, self.o1_list)
            o2 = self.discret(theta2, self.o2_list)
            o3 = self.discret(thetadot1, self.o3_list)
            o4 = self.discret(thetadot2, self.o4_list)
            return o1 + 10 * (o2 - 1) + 100 * (o3 - 1) + 1000 * (o4 - 1)

    def reset_state(self):
        self.env.reset()

    def change_state(self, action):
        self.env.step(action)

    def check_goal(self):
        s = self.env.state
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > h0)
