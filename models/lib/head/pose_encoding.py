import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding layer using sinusoidal functions.

    This class generates positional encodings based on the input dimension and maximum sequence length.
    The encodings are added to the input tensor to provide positional information for sequence modeling tasks.

    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the output.
        pe (torch.Tensor): Precomputed positional encodings stored as a buffer.
    """

    def __init__(self, d_model, dropout=0.1, max_len=600):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimension of the input embeddings.
            dropout (float): Dropout probability applied to the output. Default is 0.1.
            max_len (int): The maximum length of the input sequence. Default is 600.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encodings added, of the same shape as the input.
        """
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)