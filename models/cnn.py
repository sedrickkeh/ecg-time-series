import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_of_class):
        super(CNNModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),

        )

        self.linear = nn.Sequential(
            nn.Linear(2944,500),
            nn.LeakyReLU(inplace=True),
            nn.Linear(500, num_of_class),

        )


    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x