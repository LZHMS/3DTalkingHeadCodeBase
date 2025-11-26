
import os
import cv2
import tempfile
import os.path as osp
import numpy as np
from tqdm import tqdm
from functools import reduce

import torch
import torch.nn.functional as F
from psbody.mesh import Mesh

from base import EVALUATOR_REGISTRY, EvaluatorBase
from models import FLAME, build_flame_config
from utils import PyMeshRenderer, calc_vq_loss, calc_logit_loss, nt_xent_loss, combine_video_and_audio, convert_video, reencode_audio

import logging
logger: logging.Logger

@EVALUATOR_REGISTRY.register()
class TalkerEvaluator(EvaluatorBase):
    """Evaluator for talking head generation."""

    def __init__(self, cfg, coef_stats=None, 
                 rot_repr='aa', no_head_pose=False, 
                 motion_len=100, pre_motion_len=10, device='cpu'):
        super().__init__(cfg)
        self.coef_stats = coef_stats
        self.rot_repr, self.no_head_pose = rot_repr, no_head_pose
        self.motion_len, self.pre_motion_len = motion_len, pre_motion_len
        self.cfg, self.device = cfg, device
        
        # Avatar model for loss computation
        self.flame = FLAME(build_flame_config(cfg.TDMM.FLAME.ROOT)).to(self.device)
        logger.info(f"Loaded FLAME model for loss computation.")

        self.criterion = self.build_loss_metrics(cfg.LOSS.NAME)

        if cfg.LOAD_RENDER:
            # Set environment for offscreen rendering
            os.environ["PYOPENGL_PLATFORM"] = cfg.RENDER.PYOPENGL_PLATFORM # osmesa or egl
            self.Mesh = Mesh
            self.uv_coords = np.load(osp.join(cfg.TDMM.FLAME.ROOT, 'uv_coords.npz'))
            self.mesh_render = self.setup_mesh_renderer(cfg.RENDER.NAME,
                                                        cfg.RENDER.REND_SIZE,
                                                        cfg.RENDER.BLACK_BG)
          
    def setup_mesh_renderer(self, render_name, size, black_bg):
        if render_name == "PyMeshRenderer":
            return PyMeshRenderer(size, black_bg=black_bg)
        else:
            raise ValueError(f"Unknown mesh renderer: {render_name}")

    def coef_dict_to_vertices(self, coef_dict, flame_batch_size=512):
        shape = coef_dict['exp'].shape[:-1]
        coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
        n_samples = reduce(lambda x, y: x * y, shape, 1)

        # Convert to vertices
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

        vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
        vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

        return vert_list

    def get_coef_dict(self, motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
        coef_dict = {
            'exp': motion_coef[..., :50]
        }
        if rot_repr == 'aa':
            if with_global_pose:
                coef_dict['pose'] = motion_coef[..., 50:]
            else:
                placeholder = torch.zeros_like(motion_coef[..., :3])
                coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
            # Add back rotation around y, z axis
            coef_dict['pose'] = torch.cat([coef_dict['pose'], torch.zeros_like(motion_coef[..., :2])], dim=-1)
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

        if shape_coef is not None:
            if motion_coef.ndim == 3:
                if shape_coef.ndim == 2:
                    shape_coef = shape_coef.unsqueeze(1)
                if shape_coef.shape[1] == 1:
                    shape_coef = shape_coef.expand(-1, motion_coef.shape[1], -1)

            coef_dict['shape'] = shape_coef

        if denorm_stats is not None:
            coef_dict = {k: coef_dict[k] * denorm_stats[f'{k}_std'].to(shape_coef.device) + denorm_stats[f'{k}_mean'].to(shape_coef.device) for k in coef_dict}

        if not with_global_pose:
            if rot_repr == 'aa':
                coef_dict['pose'][..., :3] = 0
            else:
                raise ValueError(f'Unknown rotation representation {rot_repr}!')

        return coef_dict

    def build_loss_metrics(self, loss_fc_name):
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
        seq_len, loss_cfg = motion_pre.shape[1], self.cfg.EVALUATE.LOSS.GEOMETRIC
        # get the vertices
        coef_gt = self.get_coef_dict(motion_coef_gt, self.shape_coef, self.coef_stats, \
                                     with_global_pose=False, rot_repr=self.rot_repr)
        coef_pred = self.get_coef_dict(motion_pre, self.shape_coef, self.coef_stats, \
                                       with_global_pose=False, rot_repr=self.rot_repr)
        
        verts_gt, _, _ = self.flame(coef_gt['shape'].view(-1, 100), coef_gt['exp'].view(-1, 50),
                                coef_gt['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
        verts_pred, _, _ = self.flame(coef_pred['shape'].view(-1, 100), coef_pred['exp'].view(-1, 50),
                                    coef_pred['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
        verts_gt, verts_pred = verts_gt.view(-1, seq_len, 5023, 3), verts_pred.view(-1, seq_len, 5023, 3)

        
        geometric_losses = {}
        if loss_cfg.W_VERTEX > 0:   # vetices loss
            geometric_losses['vert'] = self.vetices_loss(verts_gt, verts_pred)
        if loss_cfg.W_VELOCITY > 0:  # velocity loss
            geometric_losses['vel'] = self.velocity_loss(verts_gt, verts_pred)
        if loss_cfg.W_SMOOTH > 0:    # smoothness loss
            geometric_losses['smooth'] = self.smooth_loss(verts_pred)

        # head pose losses
        if not self.no_head_pose:
            head_pose_gt, head_pose_pred = motion_coef_gt[:, :, 50:53], motion_pre[:, :, 50:53]
            if loss_cfg.HEAD.W_ANGLE > 0:
                geometric_losses['head_angle'] = self.vetices_loss(head_pose_gt, head_pose_pred)
            if loss_cfg.HEAD.W_VELOCITY > 0:
                geometric_losses['head_vel'] = self.velocity_loss(head_pose_gt, head_pose_pred)
            if loss_cfg.HEAD.W_SMOOTH > 0:
                geometric_losses['head_smooth'] = self.smooth_loss(head_pose_pred)

            if self.clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})
                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})
                head_pose_trans = torch.cat([head_pose_gt[:, self.pre_motion_len - 3 : self.pre_motion_len],
                                            head_pose_pred[:, self.pre_motion_len : self.pre_motion_len + 3]], dim=1)
                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]

                # will constrain x_{-2|0} ~ x_{1}
                loss_head_trans_vel = self.vetices_loss(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3])
                # will constrain x_{-3|0} ~ x_{2}
                loss_head_trans_accel = self.smooth_loss(head_vel_pred)

        # Mask handling
        if end_idx is None:
            mask = torch.ones((motion_pre.shape[0], self.motion_len.MOTIONS), dtype=torch.bool, device=motion_pre.device)
        else:
            mask = torch.arange(self.motion_len, device=motion_pre.device).expand(motion_pre.shape[0], -1) < end_idx.unsqueeze(1)
        mask = torch.cat([torch.ones_like(mask[:, :self.pre_motion_len]), mask], dim=1) if self.clip_id != 0 else mask

        # Final meaning and scaling
        if 'vert' in geometric_losses:
            geometric_losses['vert'] = (geometric_losses['vert'][mask].mean() / 2) * loss_cfg.W_VERTEX
        if 'vel' in geometric_losses and torch.numel(geometric_losses['vel']) > 0:
            geometric_losses['vel'] = (geometric_losses['vel'][mask[:, 1:]].mean() / 2) * loss_cfg.W_VELOCITY
        if 'smooth' in geometric_losses and torch.numel(geometric_losses['smooth']) > 0:
            geometric_losses['smooth'] = (geometric_losses['smooth'][mask[:, 2:]].mean() / 2) * loss_cfg.W_SMOOTH
        if 'head_angle' in geometric_losses:
            geometric_losses['head_angle'] = (geometric_losses['head_angle'][mask].mean() / 2) * loss_cfg.HEAD.W_ANGLE
        if 'head_vel' in geometric_losses and torch.numel(geometric_losses['head_vel']) > 0:
            geometric_losses['head_vel'] = (geometric_losses['head_vel'][mask[:, 1:]].mean() / 2) * loss_cfg.HEAD.W_VELOCITY
        if 'head_smooth' in geometric_losses and torch.numel(geometric_losses['head_smooth']) > 0:
            geometric_losses['head_smooth'] = (geometric_losses['head_smooth'][mask[:, 2:]].mean() / 2)
        if self.clip_id != 0 and loss_cfg.HEAD.W_TRANS > 0:
            vel_mask = mask[:, self.pre_motion_len : self.pre_motion_len + 2]
            accel_mask = mask[:, self.pre_motion_len : self.pre_motion_len + 3]
            geometric_losses['head_smooth'] += (loss_head_trans_vel[vel_mask].mean() + loss_head_trans_accel[accel_mask].mean())
            geometric_losses['head_smooth'] *= loss_cfg.HEAD.W_SMOOTH

        self.clip_id += 1   # update clip id for next clip
        return geometric_losses

    def vetices_loss(self, coef_gt, coef_pre, reduction='none'):
        return self.criterion(coef_gt, coef_pre, reduction=reduction)
    
    def velocity_loss(self, coef_gt, coef_pre, reduction='none'):
        vel_gt = coef_gt[:, 1:] - coef_gt[:, :-1]
        vel_pred = coef_pre[:, 1:] - coef_pre[:, :-1]
        
        return self.criterion(vel_gt, vel_pred, reduction=reduction)

    def smooth_loss(self, coef_pre, reduction='none'):
        vel_pred = coef_pre[:, 1:] - coef_pre[:, :-1]
        return self.criterion(vel_pred[:, 1:], vel_pred[:, :-1], reduction=reduction)
        
    def reset(self, shape_coef=None):
        self.clip_id, self.shape_coef = 0, shape_coef
    
    def save_coef_file(self, coef, out_path):
        os.mkdir(out_path, exist_ok=True)
        coef_np = {k: v.detach().cpu().numpy() for k, v in coef.items()}
        np.savez_compressed(out_path, **coef_np)

    def render_and_save(self, name, coef_dict, clip_id, output_dir):
        verts_list = self.coef_dict_to_vertices(coef_dict).detach().cpu().numpy()

        if self.cfg.SAVE_COEF:
            coef_path = osp.join(output_dir, "coefficients", f"{name}_clip{clip_id}_coef")
            self.save_coef_file({k: v[0] for k, v in coef_dict.items()}, coef_path)
        render_path = osp.join(output_dir, "renderings", f"{name}_clip{clip_id}_render.mp4")
        self.render_to_video(verts_list[0], render_path, audio_path, tex_path)
    
    def render_to_video(self, verts_list, out_path, audio_path=None, texture=None):
        """
        Args:
            verts_list (np.ndarray): (L, 5023, 3)
        """
        assert self.cfg.LOAD_RENDER, 'Renderer is not loaded.'
        faces = self.flame.faces_tensor.detach().cpu().numpy()
        if isinstance(texture, str):
            texture = cv2.cvtColor(cv2.imread(str(texture)), cv2.COLOR_BGR2RGB)

        os.makedirs(out_path, exist_ok=True)
        parent_dir = os.path.dirname(out_path)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=parent_dir)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)

        center = np.mean(verts_list, axis=(0, 1))
        for verts in tqdm(verts_list, desc='Rendering'):
            mesh = self.Mesh(verts, faces)
            rendered, _ = self.renderer.render_mesh(mesh, center, tex_img=texture, tex_uv=self.uv_coords)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        if audio_path is not None:
            # needs to re-encode audio to AAC format first, or the audio will be ahead of the video!
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=parent_dir)
            reencode_audio(audio_path, tmp_audio_file.name)
            combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_path, copy_audio=False)
            tmp_audio_file.close()
        else:
            convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()
