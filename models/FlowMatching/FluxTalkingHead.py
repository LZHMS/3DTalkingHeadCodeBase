import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lib.network.wav2vec import Wav2Vec2Model
from ..lib.network.hubert import HubertModel
from ..lib.head.diffusion_schedule import DiffusionSchedule
from ..lib.network.denoising_network import DenoisingNetwork

from ..lib.common import pad_audio


class FluxTalkingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Model parameters
        self.target = cfg.MODEL.TAIL.TYARGET
        self.use_style = True if cfg.ADD.STYLE_ENC_CKPT else False
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
            # wav2vec 2.0 weights initialization
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

        # Diffusion model
        self.denoising_net = DenoisingNetwork(cfg)
        # diffusion schedule
        self.diffusion_sched = DiffusionSchedule(cfg.MODEL.BACKBONE.N_STEPS, cfg.MODEL.BACKBONE.DIFF_SCHEDULE)

        # Classifier-free settings
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

    def forward(self, motion_feat, audio_or_feat, shape_feat, style_feat=None,
                prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or audio feature
            shape_feat: (N, d_shape) or (N, 1, d_shape)
            style_feat: (N, d_style)
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion coefficients or feature
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients

        Returns:
           motion_feat_noise: (N, L, d_motion)
        """
        if self.use_style:
            assert style_feat is not None, 'Missing style features!'

        batch_size, device = motion_feat.shape[0], motion_feat.device

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance
        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
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
            else:
                # len(self.guiding_conditions) > 1 and self.cfg_mode == 'incremental'
                # full (0.45), w/o style (0.45), w/o style or audio (0.1)
                mask_flag = torch.rand(batch_size, device=device)
                if 'style' in self.guiding_conditions:
                    mask_style = mask_flag > 0.55
                    style_feat = torch.where(mask_style.view(-1, 1, 1),
                                             self.null_style_feat.expand(batch_size, -1, -1),
                                             style_feat)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag > 0.9
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

        if style_feat is None:
            # The model only accepts audio and shape features, i.e., self.use_style = False
            person_feat = shape_feat
        else:
            person_feat = torch.cat([shape_feat, style_feat], dim=-1)

        if time_step is None:
            # Sample time step
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # (N,)

        # The forward diffusion process
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (N, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (N, 1, 1)

        eps = torch.randn_like(motion_feat)  # (N, L, d_motion)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # The reverse diffusion process
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, person_feat,
                                                prev_motion_feat, prev_audio_feat, time_step, indicator)

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions

        # # Strategy 1: resample during audio feature extraction
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

        # Strategy 2: resample after audio feature extraction (BackResample)
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

        audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
        return audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, shape_feat, style_feat=None, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size, device = audio_or_feat.shape[0], audio_or_feat.device

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'style']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'style'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if 'style' in cfg_cond:
            assert self.use_style and style_feat is not None

        if self.use_style:
            if style_feat is None:  # use null style feature
                style_feat = self.null_style_feat.expand(batch_size, -1, -1)
        else:
            assert style_feat is None, 'This model does not support style feature input!'

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(device)

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        if 'style' in cfg_cond:
            person_feat_null = torch.cat([shape_feat, self.null_style_feat.expand(batch_size, -1, -1)], dim=-1)
        else:
            if self.use_style:
                person_feat_null = torch.cat([shape_feat, style_feat], dim=-1)
            else:
                person_feat_null = shape_feat

        audio_feat_in = [audio_feat_null]
        person_feat_in = [person_feat_null]  # one frame condition: shape template + style
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)
                person_feat_in.append(person_feat_null)
            elif cond == 'style':
                if cfg_mode == 'independent':
                    audio_feat_in.append(audio_feat_null)
                elif cfg_mode == 'incremental':
                    audio_feat_in.append(audio_feat)
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
                person_feat_in.append(torch.cat([shape_feat, style_feat], dim=-1))

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        person_feat_in = torch.cat(person_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, person_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)

            # Unconditional target (CFG) or the conditional target (non-CFG)
            target_theta = results[0][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat