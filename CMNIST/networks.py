# Code adopted from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Just an MLP"""
    def __init__(self, input_dim=2 * 14 * 14, hidden_dim=390, output_dim=1, n_hidden_layers=2, use_xavier_init=True,
                 dropout=0.2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hiddens = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.output = nn.Linear(hidden_dim, output_dim)

        if use_xavier_init:
            for layer in [self.input, *self.hiddens, self.output]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST. Adopted from Domainbed.
    """

    def __init__(self, input_shape, n_outputs=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, n_outputs, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(n_outputs, n_outputs, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(n_outputs, n_outputs, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, n_outputs)
        self.bn2 = nn.GroupNorm(8, n_outputs)
        self.bn3 = nn.GroupNorm(8, n_outputs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x
