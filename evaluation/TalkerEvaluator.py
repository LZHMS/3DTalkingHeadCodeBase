import numpy as np
import os
import os.path as osp
from functools import reduce
from collections import OrderedDict, defaultdict
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
from sklearn.metrics import f1_score, confusion_matrix

from base import EVALUATOR_REGISTRY, EvaluatorBase
from models import FLAME, build_flame_config
from utils import PyMeshRenderer, calc_vq_loss, calc_logit_loss, nt_xent_loss

import logging
logger: logging.Logger

@EVALUATOR_REGISTRY.register()
class TalkerEvaluator(EvaluatorBase):
    """Evaluator for talking head generation."""

    def __init__(self, cfg, flame_model=None, device='cpu'):
        super().__init__(cfg)
        self._total = 0
        self.flame_model = flame_model
        self.rot_repr = cfg.MODEL.HEAD.ROT_REPR
        self.no_head_pose = cfg.MODEL.HEAD.NO_HEAD_POSE
        self.device = device

        # Avatar model for loss computation
        self.flame = FLAME(build_flame_config(cfg.TDMM.FLAME.ROOT)).to(self.device)
        logger.info(f"Loaded FLAME model for loss computation.")

        self.criterion = self.build_loss_metrics(cfg.LOSS.NAME)

        if cfg.EVALUATE.LOAD_RENDER:
            # Set environment for offscreen rendering
            os.environ["PYOPENGL_PLATFORM"] = cfg.EVALUATE.PYOPENGL_PLATFORM # osmesa or egl

            self.Mesh = Mesh
            self.uv_coords = np.load(osp.join(cfg.TDMM.FLAME.ROOT, 'uv_coords.npz'))
            self.mesh_render = self.setup_mesh_renderer(cfg.EVALUATE.MESH_RENDER,
                                                        cfg.EVALUATE.REND_SIZE,
                                                        cfg.EVALUATE.BLACK_BG)
        
            
    def setup_mesh_renderer(self, render_name, size, black_bg):
        if render_name == "PyMeshRenderer":
            return PyMeshRenderer(size, black_bg=black_bg)
        else:
            raise ValueError(f"Unknown mesh renderer: {render_name}")

    def coef_dict_to_vertices(self, coef_dict, flame, flame_batch_size=512):
        shape = coef_dict['exp'].shape[:-1]
        coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
        n_samples = reduce(lambda x, y: x * y, shape, 1)

        # Convert to vertices
        vert_list = []
        for i in range(0, n_samples, flame_batch_size):
            batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
            if self.rot_repr == 'aa':
                vert, _, _ = flame(
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
    
    def loss_calculate(self):
        pass

    def reset(self):
        self.clip_id = 0
