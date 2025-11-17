import torch.nn as nn
from ..lib.head.pose_encoding import PositionalEncoding

class StyleEncoder(nn.Module):
    """
    StyleEncoder is a neural network module designed to encode motion coefficients into a feature representation.
    It uses a Transformer-based architecture for feature extraction, with positional encoding applied to the input data.

    Args:
        args: A namespace containing the following attributes:
            - rot_repr (str): Rotation representation type ('aa' for axis-angle).
            - no_head_pose (bool): Whether to exclude head pose information.
            - feature_dim (int): Dimensionality of the feature space.
            - n_heads (int): Number of attention heads in the Transformer.
            - n_layers (int): Number of Transformer encoder layers.
            - mlp_ratio (float): Ratio for the feedforward layer dimensionality in the Transformer.
    """
    def __init__(self, cfg):
        super().__init__()

        # Model parameters
        self.motion_coef_dim = cfg.BACKBONE.IN_DIM if cfg.BACKBONE.IN_DIM else 50 # Base dimension for motion coefficients
        if cfg.HEAD.ROT_REPR == 'aa':
            # Adjust motion coefficient dimension based on head pose inclusion
            self.motion_coef_dim += 1 if cfg.HEAD.NO_HEAD_POSE else 4
        else:
            raise ValueError(f'Unknown rotation representation {cfg.HEAD.ROT_REPR}!')

        self.feature_dim = cfg.BACKBONE.HIDDEN_SIZE  # Dimensionality of the feature space
        self.n_heads = cfg.BACKBONE.NUM_ATTENTION_HEADS  # Number of attention heads
        self.n_layers = cfg.BACKBONE.NUM_HIDDEN_LAYERS  # Number of Transformer layers
        self.mlp_ratio = cfg.TAIL.MLP_RATIO  # Feedforward layer dimensionality ratio

        # Transformer for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,  # Input feature dimension
            nhead=self.n_heads,  # Number of attention heads
            dim_feedforward=self.mlp_ratio * self.feature_dim,  # Feedforward layer dimension
            activation='gelu',  # Activation function
            batch_first=True  # Input shape is (batch_size, seq_len, feature_dim)
        )

        # Positional encoding for sequence data
        self.PE = PositionalEncoding(self.feature_dim)

        # Transformer encoder with motion projection
        self.encoder = nn.ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),  # Linear projection for motion coefficients
            'transformer': nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers),  # Transformer encoder
        })

    @property
    def device(self):
        """
        Returns the device on which the model's parameters are located.
        """
        return next(self.parameters()).device

    def forward(self, motion_coef):
        """
        Forward pass of the StyleEncoder.

        Args:
            motion_coef (torch.Tensor): Input motion coefficients of shape (batch_size, seq_len, motion_coef_dim).

        Returns:
            torch.Tensor: Encoded feature representation of shape (batch_size, feature_dim).
        """
        batch_size, seq_len, _ = motion_coef.shape  # Extract batch size and sequence length

        # Project motion coefficients to feature space
        motion_feat = self.encoder['motion_proj'](motion_coef)
        # Apply positional encoding
        motion_feat = self.PE(motion_feat)

        # Pass through Transformer encoder
        feat = self.encoder['transformer'](motion_feat)  # Output shape: (batch_size, seq_len, feature_dim)

        # Pooling over the sequence dimension to get a single feature vector per batch
        feat = feat.mean(dim=1)  # Output shape: (batch_size, feature_dim)

        return feat