"""
TalkerEvaluator Module

This module provides evaluation functionality for 3D talking head generation models.
It includes metrics computation, geometric loss calculation, and video rendering
capabilities for evaluating the quality of generated facial animations.

Classes:
    TDTalkerEvaluator: Main evaluator class for talking head generation.
"""

import os
import cv2
import tempfile
import os.path as osp
import numpy as np
import soundfile as sf
from functools import reduce

import torch
import torch.nn.functional as F

from base.base_evaluator import EVALUATOR_REGISTRY, EvaluatorBase
from models.avatar.flame import FLAME, build_flame_config
from utils.loss import calc_vq_loss, calc_logit_loss, nt_xent_loss
from utils.media import combine_video_and_audio, convert_video, reencode_audio

import logging
logger: logging.Logger


@EVALUATOR_REGISTRY.register()
class TDTalkerEvaluator(EvaluatorBase):
    """
    Evaluator for 3D talking head generation.
    
    This evaluator handles the computation of various geometric losses,
    coefficient processing, and video rendering for talking head models.
    It uses the FLAME model for 3D face representation.
    
    Attributes:
        coef_stats (dict): Statistics for coefficient normalization/denormalization.
        rot_repr (str): Rotation representation format ('aa' for axis-angle).
        no_head_pose (bool): Whether to ignore head pose in computations.
        motion_len (int): Length of motion sequence.
        pre_motion_len (int): Length of previous motion for conditioning.
        device (str): Computation device ('cpu' or 'cuda').
        flame (FLAME): FLAME model instance for face representation.
        mesh_render: Mesh renderer for visualization.
    """

    def __init__(self, cfg, coef_stats=None, 
                 rot_repr='aa', no_head_pose=False, 
                 motion_len=100, pre_motion_len=10, 
                 audio_sr=16000, coef_fps=25, device='cpu'):
        """
        Initialize the TDTalkerEvaluator.
        
        Args:
            cfg: Configuration object containing model and training parameters.
            coef_stats (dict, optional): Statistics for coefficient normalization.
            rot_repr (str): Rotation representation, default 'aa' (axis-angle).
            no_head_pose (bool): If True, ignore global head pose. Default False.
            motion_len (int): Length of motion sequence. Default 100.
            pre_motion_len (int): Length of conditioning motion. Default 10.
            device (str): Device for computation. Default 'cpu'.
        """
        super().__init__(cfg)
        self.coef_stats = coef_stats
        self.rot_repr, self.no_head_pose = rot_repr, no_head_pose
        self.motion_len, self.pre_motion_len = motion_len, pre_motion_len
        self.cfg, self.device, self.mask = cfg, device, None
        self.audio_sr = audio_sr
        self.coef_fps = coef_fps  # default fps for video rendering
        self.render_size = cfg.RENDER.REND_SIZE  # video frame size
        
        # Initialize FLAME avatar model for loss computation
        self.flame = FLAME(build_flame_config(cfg.TDMM.FLAME.ROOT)).to(self.device)
        logger.info(f"Loaded FLAME model for loss computation.")

        # Build loss criterion based on configuration
        self.criterion = self.build_loss_metrics(cfg.LOSS.NAME)

        # Load renderer if specified in config
        if cfg.LOAD_RENDER:
            from psbody.mesh import Mesh
            from utils.renderer import PyMeshRenderer
            self.Mesh = Mesh
            self.uv_coords = np.load(osp.join(cfg.TDMM.FLAME.ROOT, 'uv_coords.npz'))
            self.mesh_render = self.setup_mesh_renderer(cfg.RENDER.NAME,
                                                        cfg.RENDER.REND_SIZE,
                                                        cfg.RENDER.BLACK_BG)
          
    def setup_mesh_renderer(self, render_name, size, black_bg):
        """
        Set up the mesh renderer for visualization.
        
        Args:
            render_name (str): Name of the renderer to use.
            size (tuple): Render resolution (width, height).
            black_bg (bool): Whether to use black background.
            
        Returns:
            Renderer: Initialized mesh renderer instance.
            
        Raises:
            ValueError: If unknown renderer name is specified.
        """
        if render_name == "PyMeshRenderer":
            return PyMeshRenderer(size, black_bg=black_bg)
        else:
            raise ValueError(f"Unknown mesh renderer: {render_name}")

    def reset(self, shape_coef=None):
        """
        Reset evaluator state for new evaluation sequence.
        
        Args:
            shape_coef (torch.Tensor, optional): Shape coefficients for the
                                                  identity being evaluated.
        """
        self.clip_id, self.shape_coef = 0, shape_coef

    def build_loss_metrics(self, loss_fc_name):
        """
        Build loss function based on configuration name.
        
        Args:
            loss_fc_name (str): Name of the loss function to use.
                               Supported: 'VQLoss', 'LogitLoss', 'NTXentLoss',
                               'L2Loss', 'L1Loss'.
                               
        Returns:
            callable: Loss function to be used for metric computation.
        """
        if loss_fc_name == "VQLoss":
            logger.info("Using VQ loss function for metrics ...")
            return calc_vq_loss
        elif loss_fc_name == "LogitLoss":
            logger.info("Using Logit loss function for metrics ...")
            return calc_logit_loss
        elif loss_fc_name == "NTXentLoss":
            logger.info("Using NT-Xent loss function for metrics ...")
            return nt_xent_loss
        elif loss_fc_name == "L2Loss":
            return F.mse_loss
        elif loss_fc_name == "L1Loss":
            return F.l1_loss
    
    def geometric_losses(self, motion_coef_gt, motion_pre, end_idx=None):
        """
        Compute geometric losses between ground truth and predicted motions.
        
        This method calculates various geometric losses including vertex loss,
        velocity loss, smoothness loss, and head pose losses.
        
        Args:
            motion_coef_gt (torch.Tensor): Ground truth motion coefficients.
            motion_pre (torch.Tensor): Predicted motion coefficients.
            end_idx (torch.Tensor, optional): End indices for variable length sequences.
            
        Returns:
            dict: Dictionary containing all computed geometric losses with keys:
                  'vert', 'vel', 'smooth', 'head_angle', 'head_vel', 'head_smooth'.
        """
        loss_cfg = self.cfg.LOSS.GEOMETRIC
        
        # Convert coefficients to FLAME format
        coef_gt = self.get_coef_dict(motion_coef_gt, self.shape_coef, self.coef_stats, \
                                     with_global_pose=False, rot_repr=self.rot_repr)
        coef_pred = self.get_coef_dict(motion_pre, self.shape_coef, self.coef_stats, \
                                       with_global_pose=False, rot_repr=self.rot_repr)
        
        # Compute vertices from coefficients using FLAME model
        verts_gt, _, _ = self.flame(coef_gt['shape'].view(-1, 100), coef_gt['exp'].view(-1, 50),
                                coef_gt['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
        verts_pred, _, _ = self.flame(coef_pred['shape'].view(-1, 100), coef_pred['exp'].view(-1, 50),
                                    coef_pred['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
        verts_gt, verts_pred = verts_gt.view(-1, motion_pre.shape[1], 5023, 3), verts_pred.view(-1, motion_pre.shape[1], 5023, 3)

        # Get mask for handling padded sequences
        mask = self.fetch_mask(motion_pre.shape[0], end_idx)

        geometric_losses = {}
        
        # Vertex reconstruction loss
        if loss_cfg.W_VERTEX > 0:
            geometric_losses['vert'] = self.simple_loss(verts_gt, verts_pred,
                                                         w=self.cfg.LOSS.GEOMETRIC.W_VERTEX, end_idx=end_idx, mask=mask)
        # Velocity consistency loss
        if loss_cfg.W_VELOCITY > 0:
            geometric_losses['vel'] = self.velocity_loss(verts_gt, verts_pred,
                                                         w=self.cfg.LOSS.GEOMETRIC.W_VELOCITY, end_idx=end_idx, mask=mask)
        # Temporal smoothness loss
        if loss_cfg.W_SMOOTH > 0:
            geometric_losses['smooth'] = self.smooth_loss(verts_pred, w=self.cfg.LOSS.GEOMETRIC.W_SMOOTH, end_idx=end_idx, mask=mask)

        # Head pose specific losses
        if not self.no_head_pose:
            head_pose_gt, head_pose_pred = motion_coef_gt[:, :, 50:53], motion_pre[:, :, 50:53]
            
            # Head angle reconstruction loss
            if loss_cfg.HEAD.W_ANGLE > 0:
                geometric_losses['head_angle'] = self.simple_loss(head_pose_gt, head_pose_pred, 
                                                                   w=self.cfg.LOSS.GEOMETRIC.HEAD.W_ANGLE, end_idx=end_idx, mask=mask)
            # Head velocity loss
            if loss_cfg.HEAD.W_VELOCITY > 0:
                geometric_losses['head_vel'] = self.velocity_loss(head_pose_gt, head_pose_pred,
                                                                  w=self.cfg.LOSS.GEOMETRIC.HEAD.W_VELOCITY, end_idx=end_idx, mask=mask)
            # Head smoothness loss
            if loss_cfg.HEAD.W_SMOOTH > 0:
                geometric_losses['head_smooth'] = self.smooth_loss(head_pose_pred, w=self.cfg.LOSS.GEOMETRIC.HEAD.W_SMOOTH, end_idx=end_idx, mask=mask)

            # Transition smoothness for non-initial clips
            if self.clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                # Constrain only the predicted current motions (x_{0} ~ x_{2})
                # Concatenate previous GT frames with predicted frames for smooth transition
                head_pose_trans = torch.cat([head_pose_gt[:, self.pre_motion_len - 3 : self.pre_motion_len],
                                            head_pose_pred[:, self.pre_motion_len : self.pre_motion_len + 3]], dim=1)

                # Velocity constraint for x_{-2|0} ~ x_{1}
                geometric_losses['head_smooth'] += self.velocity_loss(head_pose_trans[:, 1:4], head_pose_trans[:, 2:5],
                                                         mask=mask[:, self.pre_motion_len - 1: self.pre_motion_len + 2])
                # Smoothness constraint for x_{-3|0} ~ x_{2}
                geometric_losses['head_smooth'] += self.smooth_loss(head_pose_trans, mask=mask[:, self.pre_motion_len - 3: self.pre_motion_len + 3])

        self.clip_id += 1  # Update clip id for next iteration
        return geometric_losses
    
    def fetch_mask(self, batch_size, end_idx=None):
        """
        Generate mask for handling variable length sequences.
        
        This method creates boolean masks to exclude padded positions
        from loss computation.
        
        Args:
            batch_size (int): Number of samples in the batch.
            end_idx (torch.Tensor, optional): End indices for each sequence.
            
        Returns:
            torch.Tensor: Boolean mask tensor where True indicates valid positions.
        """
        # Create mask based on sequence end indices
        if end_idx is None:
            noise_mask = torch.ones((batch_size, self.motion_len), dtype=torch.bool, device=self.device)
        else:
            noise_mask = torch.arange(self.motion_len, device=self.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

        # Return basic mask for initial clip or noise target
        if self.clip_id == 0 or self.cfg.TARGET == 'noise':
            return noise_mask
        
        # For 'sample' target and non-initial clip, handle previous motion masking
        if self.cfg.NO_CONSTRAIN_PREV:
            # Don't constrain previous motions
            sample_mask = torch.cat([torch.zeros((batch_size, self.pre_motion_len), dtype=torch.bool, device=self.device), noise_mask], dim=1)
        else:
            # Include previous motions in constraint
            sample_mask = torch.cat([torch.ones((batch_size, self.pre_motion_len), dtype=torch.bool, device=self.device), noise_mask], dim=1)

        return sample_mask
    
    def simple_loss(self, y_gt, y_pre, w=1, reduction='none', end_idx=None, mask=None):
        """
        Compute simple reconstruction loss between ground truth and prediction.
        
        Args:
            y_gt (torch.Tensor): Ground truth tensor.
            y_pre (torch.Tensor): Predicted tensor.
            w (float): Loss weight multiplier. Default 1.
            reduction (str): Reduction mode for criterion. Default 'none'.
            end_idx (torch.Tensor, optional): End indices for masking.
            mask (torch.Tensor, optional): Pre-computed mask tensor.
            
        Returns:
            torch.Tensor: Weighted and masked loss value.
        """
        simple_loss = self.criterion(y_gt, y_pre, reduction=reduction)
        if mask is None:
            mask = self.fetch_mask(y_pre.shape[0], end_idx)

        return (simple_loss[mask].mean() / 2) * w
    
    def velocity_loss(self, coef_gt, coef_pre, w=1, reduction='none', end_idx=None, mask=None):
        """
        Compute velocity (first-order derivative) loss.
        
        This loss encourages the predicted motion to have similar
        frame-to-frame changes as the ground truth.
        
        Args:
            coef_gt (torch.Tensor): Ground truth tensor.
            coef_pre (torch.Tensor): Predicted tensor.
            w (float): Loss weight multiplier. Default 1.
            reduction (str): Reduction mode for criterion. Default 'none'.
            end_idx (torch.Tensor, optional): End indices for masking.
            mask (torch.Tensor, optional): Pre-computed mask tensor.
            
        Returns:
            torch.Tensor: Weighted velocity loss value.
        """
        # Compute velocities as frame differences
        vel_gt = coef_gt[:, 1:] - coef_gt[:, :-1]
        vel_pred = coef_pre[:, 1:] - coef_pre[:, :-1]

        vel_loss = self.criterion(vel_gt, vel_pred, reduction=reduction)
        if mask is None:
            mask = self.fetch_mask(coef_pre.shape[0], end_idx)

        # Adjust mask for velocity (one less frame than position)
        return (vel_loss[mask[:, 1:]].mean() / 2) * w

    def smooth_loss(self, coef_pre, w=1, reduction='none', end_idx=None, mask=None):
        """
        Compute smoothness (second-order derivative) loss.
        
        This loss penalizes acceleration, encouraging smooth motion
        by minimizing changes in velocity.
        
        Args:
            coef_pre (torch.Tensor): Predicted tensor.
            w (float): Loss weight multiplier. Default 1.
            reduction (str): Reduction mode for criterion. Default 'none'.
            end_idx (torch.Tensor, optional): End indices for masking.
            mask (torch.Tensor, optional): Pre-computed mask tensor.
            
        Returns:
            torch.Tensor: Weighted smoothness loss value.
        """
        # Compute velocity
        vel_pred = coef_pre[:, 1:] - coef_pre[:, :-1]
        # Compute acceleration (change in velocity)
        smooth_loss = self.criterion(vel_pred[:, 1:], vel_pred[:, :-1], reduction=reduction)
        if mask is None:
            mask = self.fetch_mask(coef_pre.shape[0], end_idx)
        
        # Adjust mask for acceleration (two less frames than position)
        return (smooth_loss[mask[:, 2:]].mean() / 2) * w
    
    def save_coef_file(self, coef, out_path):
        """
        Save FLAME coefficients to compressed numpy file.
        
        Args:
            coef (dict): Dictionary of coefficient tensors.
            out_path (str): Output file path (without extension).
        """
        os.mkdir(out_path, exist_ok=True)
        coef_np = {k: v.detach().cpu().numpy() for k, v in coef.items()}
        np.savez_compressed(out_path, **coef_np)

    def coef_dict_to_vertices(self, coef_dict, flame_batch_size=512):
        """
        Convert FLAME coefficients to 3D mesh vertices.
        
        Args:
            coef_dict (dict): Dictionary containing FLAME coefficients
                              ('exp', 'pose', 'shape').
            flame_batch_size (int): Batch size for FLAME forward pass.
                                   Default 512.
                                   
        Returns:
            torch.Tensor: Vertex positions with shape (..., 5023, 3).
        """
        shape = coef_dict['exp'].shape[:-1]
        coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
        n_samples = reduce(lambda x, y: x * y, shape, 1)

        # Process in batches to avoid memory issues
        vert_list = []
        for i in range(0, n_samples, flame_batch_size):
            batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
            if self.rot_repr == 'aa':
                vert, _, _ = self.flame(
                    batch_coef_dict['shape'], batch_coef_dict['exp'], batch_coef_dict['pose'],
                    pose2rot=True, ignore_global_rot=self.no_head_pose, return_lm2d=False, return_lm3d=False)
            else:
                raise ValueError(f'Unknown rot_repr: {self.rot_repr}')
            vert_list.append(vert)

        # Concatenate and reshape to original dimensions
        vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
        vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

        return vert_list

    def get_coef_dict(self, motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
        """
        Extract and organize FLAME coefficients from motion tensor.
        
        Args:
            motion_coef (torch.Tensor): Motion coefficients tensor containing
                                        expression and pose parameters.
            shape_coef (torch.Tensor, optional): Shape coefficients (identity).
            denorm_stats (dict, optional): Statistics for denormalization.
            with_global_pose (bool): Whether to include global head rotation.
            rot_repr (str): Rotation representation format.
            
        Returns:
            dict: Dictionary with keys 'exp', 'pose', and optionally 'shape'.
            
        Raises:
            ValueError: If unknown rotation representation is specified.
        """
        coef_dict = {
            'exp': motion_coef[..., :50]  # First 50 dims are expression
        }
        if rot_repr == 'aa':
            if with_global_pose:
                coef_dict['pose'] = motion_coef[..., 50:]
            else:
                # Use placeholder zeros for global rotation
                placeholder = torch.zeros_like(motion_coef[..., :3])
                coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
            # Append zeros for rotation around y and z axes
            coef_dict['pose'] = torch.cat([coef_dict['pose'], torch.zeros_like(motion_coef[..., :2])], dim=-1)
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

        # Handle shape coefficients with proper broadcasting
        if shape_coef is not None:
            if motion_coef.ndim == 3:
                if shape_coef.ndim == 2:
                    shape_coef = shape_coef.unsqueeze(1)
                if shape_coef.shape[1] == 1:
                    shape_coef = shape_coef.expand(-1, motion_coef.shape[1], -1)

            coef_dict['shape'] = shape_coef

        # Apply denormalization if statistics provided
        if denorm_stats is not None:
            coef_dict = {k: coef_dict[k] * denorm_stats[f'{k}_std'].to(shape_coef.device) + denorm_stats[f'{k}_mean'].to(shape_coef.device) for k in coef_dict}

        # Zero out global rotation if not needed
        if not with_global_pose:
            if rot_repr == 'aa':
                coef_dict['pose'][..., :3] = 0
            else:
                raise ValueError(f'Unknown rotation representation {rot_repr}!')

        return coef_dict

    def render_and_save(self, name, motion_coef, audio, clip_id, output_dir, shape_coef=None, texture=None):
        """
        Render motion coefficients to video and save.
        
        This method converts motion coefficients to mesh vertices,
        renders them to video frames, and optionally combines with audio.
        
        Args:
            name (str): Base name for output files.
            motion_coef (torch.Tensor): Motion coefficients to render.
            audio: Audio data (numpy array, tensor, or file path).
            clip_id (int): Clip identifier for file naming.
            output_dir (str): Directory for output files.
            shape_coef (torch.Tensor, optional): Shape coefficients for identity.
            texture: Texture image or path for mesh rendering.
        """
        # Use provided shape_coef or fall back to self.shape_coef
        shape = shape_coef if shape_coef is not None else self.shape_coef
        coef_dict = self.get_coef_dict(motion_coef, shape, self.coef_stats, \
                                       with_global_pose=True, rot_repr=self.rot_repr)
        verts_list = self.coef_dict_to_vertices(coef_dict).detach().cpu().numpy()

        # Optionally save coefficients
        if self.cfg.SAVE_COEF:
            coef_path = osp.join(output_dir, "coefficients", f"{name}_clip{clip_id}_coef")
            self.save_coef_file({k: v[0] for k, v in coef_dict.items()}, coef_path)
        
        # Render to video file
        render_path = osp.join(output_dir, "renderings", f"{name}_clip{clip_id}_render.mp4")
        self.render_to_video(verts_list[0], render_path, audio=audio, texture=texture)
    
    def render_to_video(self, verts_list, out_path, audio=None, texture=None, sample_rate=16000):
        """
        Render mesh vertices to video file.
        
        This method takes a sequence of vertex positions and renders them
        as a video, optionally adding audio track.
        
        Args:
            verts_list (np.ndarray): Vertex positions with shape (L, 5023, 3)
                                     where L is the number of frames.
            out_path (str): Output video file path.
            audio: Audio data (numpy array, tensor, or file path string).
            texture: Texture image or path for mesh rendering.
            sample_rate (int): Audio sample rate in Hz. Default 16000.
            
        Raises:
            AssertionError: If renderer is not loaded (LOAD_RENDER is False).
        """
        assert self.cfg.LOAD_RENDER, 'Renderer is not loaded.'
        
        faces = self.flame.faces_tensor.detach().cpu().numpy()
        
        # Load texture from file if path provided
        if isinstance(texture, str):
            texture = cv2.cvtColor(cv2.imread(str(texture)), cv2.COLOR_BGR2RGB)

        # Create output directory if needed
        parent_dir = os.path.dirname(out_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Create temporary video file for rendering
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=parent_dir, delete=False)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.coef_fps, self.render_size)

        # Compute center for camera positioning
        center = np.mean(verts_list, axis=(0, 1))
        
        # Render each frame
        for verts in verts_list:
            mesh = self.Mesh(verts, faces)
            rendered, _ = self.mesh_render.render_mesh(mesh, center, tex_img=texture, tex_uv=self.uv_coords)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        # Handle audio track if provided
        if audio is not None:
            # Determine if audio is file path or data
            if isinstance(audio, str):
                audio_path = audio
                tmp_audio_wav = None
            else:
                # Convert tensor to numpy if needed
                if hasattr(audio, 'cpu'):
                    audio = audio.cpu().numpy()
                # Save audio data to temporary wav file
                tmp_audio_wav = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=parent_dir, delete=False)
                sf.write(tmp_audio_wav.name, audio, samplerate=sample_rate)
                audio_path = tmp_audio_wav.name
            
            # Re-encode audio to AAC format to prevent audio-video desync
            tmp_audio_aac = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=parent_dir, delete=False)
            reencode_audio(audio_path, tmp_audio_aac.name)
            combine_video_and_audio(tmp_video_file.name, tmp_audio_aac.name, out_path, copy_audio=False)
            
            # Clean up temporary audio files
            os.remove(tmp_audio_aac.name)
            if tmp_audio_wav is not None:
                os.remove(tmp_audio_wav.name)
        else:
            # No audio - just convert video format
            convert_video(tmp_video_file.name, out_path)
        
        # Clean up temporary video file
        os.remove(tmp_video_file.name)
