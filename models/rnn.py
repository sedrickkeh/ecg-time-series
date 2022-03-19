import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=3, batch_first=True)
        self.linear = nn.Linear(192, 5)

    def forward(self, x):
        out = self.rnn(x.unsqueeze(2))
        out = out[1].transpose(0,1).flatten(start_dim=1)
        out = self.linear(out)
        return out