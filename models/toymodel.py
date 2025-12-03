"""
MNIST MLP Model for Handwritten Digit Recognition
"""

import torch.nn as nn
from .lib.network.mlp import MLP

class ToyModel(nn.Module):
    """Simple MLP for MNIST digit recognition"""
    def __init__(self):
      super(ToyModel, self).__init__()
      self.net = MLP(in_features=1*28*28,
                     hidden_layers=[20, 10],
                     activation='relu',
                     bn=True, dropout=0.1)

    def forward(self, x):
        return self.net(x)