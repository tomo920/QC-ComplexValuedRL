import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qdot_base import QdotBase
from complexnet import CNnet

class MultiCNnet(CNnet):
    def __init__(self, layer_config):
        super().__init__(layer_config)

    def train(self, input, n, target, loss_type):
        pred = self.outputs(input)
        if loss_type == 'square_error':
            grad = np.zeros(len(pred)).astype(np.complex128)
            grad[n] = pred[n] - target
        for layer in reversed(self.layers):
            grad = layer.back_propagate(grad)
            layer.update()

class Qfunc(QdotBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.action_size = env.action_size
        self.action_list = env.action_list
        # neural network configulation
        if config.action_encoding:
            input_size = env.observation_size + 1
            output_size = 1
            nnet = CNnet
        else:
            input_size = env.observation_size
            output_size = env.action_size
            nnet = MultiCNnet
        layer_config = []
        # set hidden layer config
        for i in range(config.hidden_layer_num):
            layer_config.append({
                'input_size': input_size if i == 0 else config.hidden_size,
                'output_size': config.hidden_size,
                'activation': 'tanh',
                'lr': config.lr_h,
                'trainable': True
            })
        # set output layer config
        layer_config.append({
            'input_size': input_size if config.hidden_layer_num == 0 else config.hidden_size,
            'output_size': output_size,
            'activation': 'linear',
            'lr': config.lr_o,
            'trainable': not config.not_trainable_output_layer
        })
        # initialize Q network
        self.Q_network = nnet(layer_config)

    def get_q_o_a(self, observation, action):
        action = np.array([self.action_list[action]])
        input = np.concatenate([observation, action])
        return self.Q_network.outputs(input.astype(np.complex128))

    def get_q_o(self, observation):
        if self.config.action_encoding:
            return np.concatenate([self.get_q_o_a(observation, action) for action in range(self.action_size)])
        else:
            return self.Q_network.outputs(observation.astype(np.complex128))

    def update_q(self, observation, action, q_target):
        if self.config.action_encoding:
            action = np.array([self.action_list[action]])
            input = np.concatenate([observation, action])
            self.Q_network.train(input.astype(np.complex128), q_target, 'square_error')
        else:
            self.Q_network.train(observation.astype(np.complex128), action, q_target, 'square_error')

    def get_params(self):
        return self.Q_network.get_params()

    def set_params(self, params):
        self.Q_network.set_params(params)
