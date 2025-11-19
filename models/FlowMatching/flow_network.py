"""
Flow Denoising Network for FlowMatching.
Adapted from DenoisingNetwork in DiffPoseTalk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lib.head.pose_encoding import PositionalEncoding
from ..lib.common import enc_dec_mask


class FlowDenoisingNetwork(nn.Module):
    """
    Network to predict velocity/flow for Flow Matching.
    Similar to DenoisingNetwork but adapted for continuous time flow matching.
    """
    
    def __init__(self, cfg):
        super().__init__()

        # Model parameters
        self.use_style = True if cfg.ADD.STYLE_ENC_CKPT else False
        self.motion_feat_dim = 51 if cfg.MODEL.HEAD.NO_HEAD_POSE else 54
        self.shape_feat_dim = 100
        if self.use_style:
            self.style_feat_dim = cfg.MODEL.HEAD.STYLE_DIM
            self.person_feat_dim = self.shape_feat_dim + self.style_feat_dim
        else:
            self.person_feat_dim = self.shape_feat_dim
        self.use_indicator = cfg.MODEL.HEAD.USE_INDICATOR

        # Transformer
        self.feature_dim = cfg.MODEL.BACKBONE.HIDDEN_SIZE
        self.n_heads = cfg.MODEL.BACKBONE.NUM_ATTENTION_HEADS
        self.n_layers = cfg.MODEL.BACKBONE.NUM_HIDDEN_LAYERS
        self.mlp_ratio = cfg.MODEL.TAIL.MLP_RATIO
        self.align_mask_width = cfg.MODEL.HEAD.ALIGN_MASK_WIDTH
        self.use_learnable_pe = cfg.MODEL.HEAD.USE_LEARNABLE_PE
        
        # Sequence length
        self.n_prev_motions = cfg.DATASET.HDTF_TFHP.N_PREV_MOTIONS
        self.n_motions = cfg.DATASET.HDTF_TFHP.MOTIONS

        # Time embedding for continuous time t in [0, 1]
        # Use sinusoidal embeddings similar to diffusion but for continuous time
        self.time_embed = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Sinusoidal time encoding (similar to PositionalEncoding but for time)
        self.register_buffer('freqs', torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, self.feature_dim, 2) / self.feature_dim
        ))

        if self.use_learnable_pe:
            # Learnable positional encoding
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        self.person_proj = nn.Linear(self.person_feat_dim, self.feature_dim)

        # Transformer decoder 
        self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0), self.feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        
        if self.align_mask_width > 0:
            motion_len = self.n_prev_motions + self.n_motions
            alignment_mask = enc_dec_mask(motion_len, motion_len, 1, self.align_mask_width - 1)
            # Note: No padding needed here since person_feat is in memory, not in target sequence
            self.register_buffer('alignment_mask', alignment_mask)
        else:
            self.alignment_mask = None

        # Flow/velocity decoder
        self.flow_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim)
        )

    def get_time_embedding(self, t):
        """
        Get sinusoidal time embedding for continuous time t in [0, 1].
        
        Args:
            t: (N,) time steps in [0, 1]
            
        Returns:
            time_emb: (N, feature_dim) time embeddings
        """
        # Scale time to a reasonable range
        t_scaled = t * 1000.0  # Scale to [0, 1000]
        t_scaled = t_scaled.unsqueeze(-1)  # (N, 1)
        
        # Compute sinusoidal embeddings
        freqs = self.freqs.unsqueeze(0)  # (1, feature_dim/2)
        args = t_scaled * freqs  # (N, feature_dim/2)
        
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, feature_dim)
        
        return emb

    def forward(self, motion_feat, audio_feat, person_feat, prev_motion_feat, prev_audio_feat, t, indicator=None):
        """
        Predict the flow/velocity at time t.
        
        Args:
            motion_feat: (N, L, d_motion) - Motion at time t (interpolated between x0 and x1)
            audio_feat: (N, L, feature_dim) - Audio features
            person_feat: (N, 1, d_person) - Person features (shape + style)
            prev_motion_feat: (N, L_p, d_motion) - Previous motion coefficients
            prev_audio_feat: (N, L_p, d_audio) - Previous audio features
            t: (N,) - Continuous time steps in [0, 1]
            indicator: (N, L) - 0/1 indicator for real (unpadded) motion
            
        Returns:
            flow: (N, L_p + L, d_motion) - Predicted flow/velocity
        """
        # Time embedding
        time_emb = self.get_time_embedding(t)  # (N, feature_dim)
        time_emb = self.time_embed(time_emb).unsqueeze(1)  # (N, 1, feature_dim)

        # Person feature projection
        person_feat = self.person_proj(person_feat)  # (N, 1, feature_dim)
        person_feat = person_feat + time_emb

        if indicator is not None:
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)  # (N, L_p + L)
            indicator = indicator.unsqueeze(-1)  # (N, L_p + L, 1)

        # Concat features
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion + 1)

        # Project to feature dimension
        feats_in = self.feature_proj(feats_in)  # (N, L_p + L, feature_dim)
        
        # Add positional encoding
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE[:, 1:, :]  # Skip the person feat position
        else:
            feats_in = self.PE(feats_in)

        # Prepare audio features
        audio_feats = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (N, L_p + L, feature_dim)
        if self.use_learnable_pe:
            audio_feats = audio_feats + self.PE[:, 1:, :]
        else:
            audio_feats = self.PE(audio_feats)

        # Concatenate person feat with audio
        memory = torch.cat([person_feat, audio_feats], dim=1)  # (N, 1 + L_p + L, feature_dim)

        # Transformer
        if self.alignment_mask is not None:
            feats_out = self.transformer(feats_in, memory, tgt_mask=self.alignment_mask)
        else:
            feats_out = self.transformer(feats_in, memory)

        # Decode flow/velocity
        flow = self.flow_dec(feats_out)  # (N, L_p + L, d_motion)

        return flow
