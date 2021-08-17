import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qlstm_base import QlstmBase
from lstmnet import LSTMNet

class Qfunc(QlstmBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        torch.manual_seed(config.seed)
        if self.config.is_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print('GPU is not available')
                sys.exit()
        else:
            device = torch.device("cpu")
        # initialize Q network
        self.Q_network = LSTMNet(env.observation_size + 1, env.action_size, config.hidden_size).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(self.Q_network.parameters(), lr=config.lr_o)

    def get_q_hi(self, hidden_state, input_seq):
        return self.Q_network.forward(input_seq.permute(1, 0, 2), hidden_state)[0]

    def update_q(self, input_seq, action_seq, target_seq):
        hidden_state = (torch.zeros(1, 1, self.config.hidden_size).float(), torch.zeros(1, 1, self.config.hidden_size).float())
        q = self.Q_network.forward(input_seq.permute(1, 0, 2), hidden_state)[0]
        q_sa = torch.gather(q, 2, action_seq.permute(1, 0, 2))
        # compute loss
        loss = self.loss_fn(q_sa, target_seq.permute(1, 0, 2))
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_hidden_state(self, hidden_state, input_seq):
        return self.Q_network.forward(input_seq.permute(1, 0, 2), hidden_state)[1]
