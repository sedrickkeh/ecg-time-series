import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, 5)

    def forward(self, x):
        out = self.lstm(x.unsqueeze(2))
        out = out[0][:,-1,:]
        out = self.linear(out)
        return out