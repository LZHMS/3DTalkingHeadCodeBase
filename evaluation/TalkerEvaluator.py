import numpy as np
import os
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from psbody.mesh import Mesh
from sklearn.metrics import f1_score, confusion_matrix

from base import EVALUATOR_REGISTRY, EvaluatorBase
from utils import MeshRenderer

@EVALUATOR_REGISTRY.register()
class TalkerEvaluator(EvaluatorBase):
    """Evaluator for talking head generation."""

    def __init__(self, cfg, flame_model=None, device='0'):
        super().__init__(cfg)
        self._total = 0
        self.flame_model = flame_model
        if cfg.EVALUATE.LOAD_RENDER:
            # Set environment for offscreen rendering
            os.environ['EGL_DEVICE_ID'] = device
            os.environ["PYOPENGL_PLATFORM"] = cfg.EVALUATE.PYOPENGL_PLATFORM # osmesa or egl

            self.Mesh = Mesh
            self.uv_coords = np.load(osp.join(cfg.TDMM.FLAME.ROOT, 'uv_coords.npz'))
            self.mesh_render = self.setup_mesh_renderer(cfg.EVALUATE.MESH_RENDER,
                                                        cfg.EVALUATE.REND_SIZE,
                                                        cfg.EVALUATE.BLACK_BG)
            
    
    def setup_mesh_renderer(self, render_name, size, black_bg):
        if render_name == "PyMeshRenderer":
            return MeshRenderer.PyMeshRenderer(size, black_bg=black_bg)
        else:
            raise ValueError(f"Unknown mesh renderer: {render_name}")

    def evaluate(self, model, data_loader):
        model.eval()
        for batch_idx, batch in enumerate(data_loader):

    
    def reset(self):
        self._correct = 0
        pass

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)