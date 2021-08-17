import gym
import numpy as np
from numpy import cos
import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_base import EnvBase

action_list = [-1.0, 0, 1.0]

class Env(EnvBase):
    def __init__(self, config):
        self.env = gym.make('MountainCar-v0').env
        self.env.seed(config.seed)
        if config.is_continuous:
            observation_size = 2
        else:
            observation_size = config.n_equal_part**2
            self.o1_list = np.linspace(self.env.min_position, self.env.max_position, config.n_equal_part+1)
            self.o2_list = np.linspace(-1 * self.env.max_speed, self.env.max_speed, config.n_equal_part+1)
        if config.algorithm == 'qdot_learning_qc':
            self.o_max = self.env.max_position
            self.o_min = self.env.min_position
        action_size = 3
        max_step = config.max_steps
        super().__init__(config, observation_size, action_size, max_step, action_list)

    def get_observation(self):
        position = self.env.state[0]
        velocity = self.env.state[1]
        if self.config.is_continuous:
            return np.array([position, velocity])
        else:
            o1 = self.discret(position, self.o1_list)
            o2 = self.discret(velocity, self.o2_list)
            return o1 + 10 * (o2 - 1)

    def reset_state(self):
        self.env.reset()

    def change_state(self, action):
        _, _, self.done, _ = self.env.step(action)

    def check_goal(self):
        return self.done
