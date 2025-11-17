import torch
import torch.nn as nn
from ..head.pose_encoding import PositionalEncoding
from ..common import enc_dec_mask
import torch.nn.functional as F

class DenoisingNetwork(nn.Module):
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
        # sequence length
        self.n_prev_motions = cfg.DATASET.HDTF_TFHP.N_PREV_MOTIONS
        self.n_motions = cfg.DATASET.HDTF_TFHP.MOTIONS

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=cfg.MODEL.BACKBONE.N_STEPS + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

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
            alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
            self.register_buffer('alignment_mask', alignment_mask)
        else:
            self.alignment_mask = None

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim)
        )

    def forward(self, motion_feat, audio_feat, person_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature
            audio_feat: (N, L, feature_dim)
            person_feat: (N, 1, d_person)
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion coefficients or feature
            prev_audio_feat: (N, L_p, d_audio). Padded previous motion coefficients or feature
            step: (N,)
            indicator: (N, L). 0/1 indicator for the real (unpadded) motion feature

        Returns:
            motion_feat_target: (N, L_p + L, d_motion)
        """
        # Diffusion time step embedding
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)  # (N, 1, diff_step_dim)

        person_feat = self.person_proj(person_feat)  # (N, 1, feature_dim)
        person_feat = person_feat + diff_step_embedding

        if indicator is not None:
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)  # (N, L_p + L)
            indicator = indicator.unsqueeze(-1)  # (N, L_p + L, 1)

        # Concat features and embeddings
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion + d_audio + 1)

        feats_in = self.feature_proj(feats_in)  # (N, L_p + L, feature_dim)
        feats_in = torch.cat([person_feat, feats_in], dim=1)  # (N, 1 + L_p + L, feature_dim)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # Transformer
        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (N, L_p + L, d_audio)
        feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)

        # Decode predicted motion feature noise / sample
        motion_feat_target = self.motion_dec(feat_out[:, 1:])  # (N, L_p + L, d_motion)

        return motion_feat_target