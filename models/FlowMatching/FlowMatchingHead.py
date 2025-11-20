"""
Flow Matching based Talking Head model.
Adapted from MeanAudio and DiffPoseTalk for 3DTalkingHeadCodeBase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lib.network.wav2vec import Wav2Vec2Model
from ..lib.network.hubert import HubertModel
from ..lib.common import pad_audio
from .flow_matching import FlowMatching
from .flow_network import FlowDenoisingNetwork


class FlowMatchingHead(nn.Module):
    """Flow Matching model for talking head generation."""
    
    def __init__(self, cfg):
        super().__init__()

        # Model parameters
        self.use_style = True if cfg.ENV.EXTRA.STYLE_ENC_CKPT else False
        self.motion_feat_dim = 50
        if cfg.MODEL.HEAD.ROT_REPR == 'aa':
            self.motion_feat_dim += 1 if cfg.MODEL.HEAD.NO_HEAD_POSE else 4
        else:
            raise ValueError(f'Unknown rotation representation {cfg.MODEL.HEAD.ROT_REPR}!')

        self.fps = cfg.DATASET.HDTF_TFHP.COEF_FPS
        self.n_motions = cfg.DATASET.HDTF_TFHP.MOTIONS
        self.n_prev_motions = cfg.DATASET.HDTF_TFHP.N_PREV_MOTIONS
        if self.use_style:
            self.style_feat_dim = cfg.MODEL.HEAD.STYLE_DIM

        # Audio encoder
        self.audio_model = cfg.MODEL.HEAD.AUDIO_MODEL
        if self.audio_model == 'wav2vec2':
            self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h') 
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            self.audio_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960')
            self.audio_encoder.feature_extractor._freeze_parameters()

            frozen_layers = [0, 1]
            for name, param in self.audio_encoder.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        self.audio_feature_map = nn.Linear(768, cfg.MODEL.BACKBONE.HIDDEN_SIZE)
        self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, cfg.MODEL.BACKBONE.HIDDEN_SIZE))
        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim))

        # Flow Matching components
        self.flow_matching = FlowMatching(
            min_sigma=cfg.ALGORITHM.FLOWMATCHING.get('MIN_SIGMA', 0.0),
            inference_mode=cfg.ALGORITHM.FLOWMATCHING.get('INFERENCE_MODE', 'euler'),
            num_steps=cfg.ALGORITHM.FLOWMATCHING.get('NUM_STEPS', 25),
            reverse_flow=cfg.ALGORITHM.FLOWMATCHING.get('REVERSE_FLOW', True)
        )
        
        # Flow denoising network (similar to diffusion denoising network but for flow)
        self.flow_net = FlowDenoisingNetwork(cfg)

        # Classifier-free guidance settings
        self.cfg_mode = cfg.MODEL.CFG_MODE
        guiding_conditions = cfg.MODEL.GUIDING_CONDITIONS.split(',') if cfg.MODEL.GUIDING_CONDITIONS else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['style', 'audio']]
        if 'style' in self.guiding_conditions:
            if not self.use_style:
                raise ValueError('Cannot use style guiding without enabling it!')
            self.null_style_feat = nn.Parameter(torch.randn(1, 1, self.style_feat_dim))
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = cfg.MODEL.BACKBONE.HIDDEN_SIZE
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim))
        
        # Log-normal sampling parameters for time
        self.log_normal_mean = cfg.MODEL.BACKBONE.get('LOG_NORMAL_MEAN', 0.0)
        self.log_normal_std = cfg.MODEL.BACKBONE.get('LOG_NORMAL_STD', 1.0)

    def forward(self, motion_feat, audio_or_feat, shape_feat, style_feat=None,
                prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Forward pass for training.
        
        Args:
            motion_feat: (N, L, d_coef) motion coefficients
            audio_or_feat: (N, L_audio) raw audio or (N, L, d_audio) audio features
            shape_feat: (N, d_shape) or (N, 1, d_shape)
            style_feat: (N, d_style) style features
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,) time steps (if None, will be sampled)
            indicator: (N, L) 0/1 indicator of real motions
            
        Returns:
            predicted_flow: (N, L, d_motion) predicted flow/velocity
            target_flow: (N, L, d_motion) target flow/velocity
            motion_feat: (N, L, d_motion) clean motion (for caching)
            audio_feat: (N, L, d_audio) audio features (for caching)
        """
        if self.use_style:
            assert style_feat is not None, 'Missing style features!'

        batch_size, device = motion_feat.shape[0], motion_feat.device

        # Extract audio features
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        # Reshape features
        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance during training
        if len(self.guiding_conditions) > 0:
            null_cond_prob = 0.1
            if 'style' in self.guiding_conditions:
                mask_style = torch.rand(batch_size, device=device) < null_cond_prob
                style_feat = torch.where(mask_style.view(-1, 1, 1),
                                         self.null_style_feat.expand(batch_size, -1, -1),
                                         style_feat)
            if 'audio' in self.guiding_conditions:
                mask_audio = torch.rand(batch_size, device=device) < null_cond_prob
                audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                         self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                         audio_feat)

        # Prepare person features
        if style_feat is None:
            person_feat = shape_feat
        else:
            person_feat = torch.cat([shape_feat, style_feat], dim=-1)

        # Sample time step if not provided
        if time_step is None:
            # Log-normal sampling for better distribution
            time_step = self._log_normal_sample(batch_size, device)

        # Flow matching forward process
        x1 = motion_feat  # data (clean motion)
        x0 = torch.randn_like(x1)  # prior (noise)
        
        # Get interpolated state xt
        t_expanded = time_step[:, None, None].expand_as(x1)
        if self.flow_matching.reverse_flow:
            xt = (1 - t_expanded) * x1 + t_expanded * x0
            target_v = x0 - x1
        else:
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            target_v = x1 - x0

        # Predict flow/velocity
        predicted_v = self.flow_net(xt, audio_feat, person_feat,
                                    prev_motion_feat, prev_audio_feat, time_step, indicator)

        # Predict motion samples by integrating the learned flow from noise to data
        with torch.no_grad():
            prev_len = prev_motion_feat.shape[1]
            static_prev_motion = prev_motion_feat.detach()
            ode_init = torch.cat([static_prev_motion, x0.detach()], dim=1)

            def ode_func(t, x):
                if not torch.is_tensor(t):
                    t_tensor = torch.tensor(t, device=x.device, dtype=x.dtype)
                else:
                    t_tensor = t.to(device=x.device, dtype=x.dtype)
                t_batch = t_tensor.expand(x0.shape[0])

                # Enforce static history and extract current segment
                current_motion = x[:, prev_len:]
                flow_full = self.flow_net(current_motion, audio_feat, person_feat,
                                          static_prev_motion, prev_audio_feat, t_batch, indicator)
                return flow_full

            motion_pre = self.flow_matching.to_data(ode_func, ode_init)

        return predicted_v, target_v, motion_pre, motion_feat.detach(), audio_feat_saved.detach()

    def _log_normal_sample(self, batch_size, device):
        """Sample time steps from log-normal distribution."""
        u = torch.randn(batch_size, device=device)
        log_t = self.log_normal_mean + self.log_normal_std * u
        t = torch.sigmoid(log_t)  # Ensure t in [0, 1]
        return t

    def extract_audio_feature(self, audio, frame_num=None):
        """Extract audio features using pre-trained audio encoder."""
        frame_num = frame_num or self.n_motions

        # BackResample strategy
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)

        audio_feat = self.audio_feature_map(hidden_states)
        return audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, shape_feat, style_feat=None, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_start=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, 
               ret_traj=False):
        """
        Sample motion sequences using flow matching.
        
        Args:
            audio_or_feat: Audio or audio features
            shape_feat: Shape features
            style_feat: Style features
            prev_motion_feat: Previous motion features
            prev_audio_feat: Previous audio features
            motion_at_start: Initial noise (if None, will be sampled)
            indicator: Motion indicator
            cfg_mode: Classifier-free guidance mode
            cfg_cond: CFG conditions
            cfg_scale: CFG scale
            ret_traj: Whether to return full trajectory
            
        Returns:
            Sampled motion coefficients
        """
        batch_size, device = audio_or_feat.shape[0], audio_or_feat.device

        # Check CFG conditions
        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'style']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # Extract and prepare features
        if audio_or_feat.ndim == 2:
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_start is None:
            motion_at_start = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(device)

        # Prepare CFG inputs
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        if 'style' in cfg_cond:
            person_feat_null = torch.cat([shape_feat, self.null_style_feat.expand(batch_size, -1, -1)], dim=-1)
        else:
            if self.use_style and style_feat is not None:
                person_feat_null = torch.cat([shape_feat, style_feat], dim=-1)
            else:
                person_feat_null = shape_feat

        # Prepare multiple condition entries for CFG
        audio_feat_in = [audio_feat_null]
        person_feat_in = [person_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)
                person_feat_in.append(person_feat_null)
            elif cond == 'style':
                if cfg_mode == 'independent':
                    audio_feat_in.append(audio_feat_null)
                elif cfg_mode == 'incremental':
                    audio_feat_in.append(audio_feat)
                person_feat_in.append(torch.cat([shape_feat, style_feat], dim=-1))

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        person_feat_in = torch.cat(person_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        # Define ODE function with CFG
        def ode_func(t, x):
            t_batch = torch.full((batch_size * n_entries,), t, device=device, dtype=x.dtype)
            pred_v = self.flow_net(x, audio_feat_in, person_feat_in,
                                  prev_motion_feat_in, prev_audio_feat_in, t_batch, indicator_in)
            
            # Apply CFG
            pred_v_chunks = pred_v.chunk(n_entries)
            v_uncond = pred_v_chunks[0][:batch_size]
            v_final = v_uncond
            
            for i, scale in enumerate(cfg_scale):
                if cfg_mode == 'independent':
                    v_final = v_final + scale * (pred_v_chunks[i+1][:batch_size] - v_uncond)
                elif cfg_mode == 'incremental':
                    v_final = v_final + scale * (pred_v_chunks[i+1][:batch_size] - pred_v_chunks[i][:batch_size])
            
            return torch.cat([v_final] * n_entries, dim=0)

        # Solve ODE to generate samples
        x_start = torch.cat([motion_at_start] * n_entries, dim=0)
        x_final = self.flow_matching.to_data(ode_func, x_start)
        x_final = x_final[:batch_size]  # Only take the first batch

        if ret_traj:
            return x_final, motion_at_start, audio_feat, None
        else:
            return x_final, motion_at_start, audio_feat
