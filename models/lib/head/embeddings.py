import math
import torch
import torch.nn as nn


def modulate(x, scale, shift):
    """
    Modulates input tensor x with scale and shift parameters.
    
    Args:
        x: Input tensor to be modulated
        scale: Scaling factor tensor
        shift: Shifting factor tensor
    
    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds timestep values into high-dimensional representations using sinusoidal embeddings
    followed by an MLP. Commonly used in diffusion models to condition on timesteps.
    """
    
    def __init__(self, dim, nfreq=256):
        """
        Initialize the TimestepEmbedder.
        
        Args:
            dim: Output dimension of the embedding
            nfreq: Frequency dimension for sinusoidal encoding (default: 256)
        """
        super().__init__()
        # MLP to process sinusoidal embeddings: Linear -> SiLU activation -> Linear
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings similar to the positional encoding
        used in Transformers.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
            dim: Dimension of the output embedding
            max_period: Maximum period for the sinusoidal functions (default: 10000)
        
        Returns:
            Tensor of shape (batch_size, dim) containing sinusoidal embeddings
        """
        half_dim = dim // 2
        # Compute frequency bands using exponential decay
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        # Compute arguments for sin and cos functions
        args = t[:, None].float() * freqs[None]
        # Concatenate cosine and sine embeddings
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Pad with zeros if dim is odd
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        """
        Forward pass to convert timesteps to embeddings.
        
        Args:
            t: Input timestep tensor (typically normalized between 0 and 1)
        
        Returns:
            Embedded timestep tensor of shape (batch_size, dim)
        """
        # Scale timestep to [0, 1000] range
        t = t * 1000
        # Generate sinusoidal frequency embeddings
        t_freq = self.timestep_embedding(t, self.nfreq)
        # Pass through MLP to get final embedding
        t_emb = self.mlp(t_freq)
        return t_emb