import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

class LSTMNet(nn.Module):
    """
    LSTM network class for deep recurrent q learning
    Only batch size 1 is supported
    """

    def __init__(self, input_size, action_size, hidden_size):
        super(LSTMNet, self).__init__()
        # lstm layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        # output layer
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden):
        lstm_output, next_hidden = self.lstm(x, hidden)
        q = self.fc(lstm_output)
        return q, next_hidden
