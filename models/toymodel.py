"""
MNIST MLP Model for Handwritten Digit Recognition
"""

import torch.nn as nn
from .lib.network.mlp import MLP

class ToyModel(nn.Module):
    """Simple MLP for MNIST digit recognition"""
    def __init__(self, cfg):
      super(ToyModel, self).__init__()
      self.net = MLP(in_features=cfg.INPUT_DIM,
                     hidden_layers=cfg.HIDDEN_DIM,
                     out_features=cfg.OUTPUT_DIM,
                     activation='relu',
                     bn=True, dropout=0.1)

    def forward(self, x):
        return self.net(x)