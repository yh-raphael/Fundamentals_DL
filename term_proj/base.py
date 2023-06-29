import torch
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self, inputSize = int, outputSize = int):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = inputSize                             # 132
        layers = [
            nn.Linear(in_size, 128),                    # 132 -> Input layer    => 128
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, 256),                        # 128 -> Hidden layer 1 => 256
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(256, 128),                        # 256 -> Hidden layer 2 => 128
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, outputSize),                 # 128 -> Output layer   => 4
        ]
        self.laysers = nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x