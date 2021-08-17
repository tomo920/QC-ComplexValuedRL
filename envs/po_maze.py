import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from env_base import EnvBase
from maze_types import get_maze

'''
action 0 -> right
action 1 -> left
action 2 -> up
action 3 -> down
'''
action_list = [np.array([1.0, 0.0]),
               np.array([-1.0, 0.0]),
               np.array([0.0, 1.0]),
               np.array([0.0, -1.0])]

class PoEnv(EnvBase):
    def __init__(self, config):
        self.legal_states, _, self.observation_list, self.o_num, \
        self.start_state, self.goal_state, self.legal_action_list = get_maze(config.maze_type)
        if config.is_continuous:
            observation_size = 1
        else:
            observation_size = self.o_num
        action_size = 4
        max_step = config.max_steps
        action_list = [0.0, 1.0, 2.0, 3.0]
        if config.algorithm == 'qdot_learning_qc':
            action_list = 2 * ((np.array(action_list) - 0) / 3) - 1
        self.legal_action = config.is_legal_action
        super().__init__(config, observation_size, action_size, max_step, action_list)

    def get_observation(self):
        state = self.state.tostring()
        o = self.observation_list[state]
        if self.config.is_continuous:
            if self.config.algorithm == 'qdot_learning_qc':
                o = 2 * ((o - 1) / (self.o_num - 1)) - 1
            return np.array([o])
        else:
            if self.config.algorithm == 'qdot_learning_qc':
                return [o]
            else:
                return o

    def reset_state(self):
        self.state = self.start_state

    def change_state(self, action):
        c_state = self.state
        self.state = self.state+action_list[action]
        if self.legal_action:
            if not self.check_legal():
                print('error')
                sys.exit()
        else:
            if not self.check_legal():
                self.state = c_state

    def check_legal(self):
        state = self.state.tostring()
        if state in self.legal_states:
            return True
        else:
            return False

    def check_goal(self):
        return self.state[0] == self.goal_state[0] and self.state[1] == self.goal_state[1]

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state
