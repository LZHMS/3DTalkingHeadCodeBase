import torch
from .vqae import VQAutoEncoder


class StyleVQAE(VQAutoEncoder):
  def __init__(self, cfg):
    super().__init__(cfg)

  def encode(self, x, x_a=None):
    h = self.encoder(x) ## x --> z'
    h = h.view(x.shape[0], -1, self.cfg.BACKBONE.FACE_QUAN_NUM, self.cfg.HEAD.ZQUANT_DIM)
    h = h.view(x.shape[0], -1, self.cfg.HEAD.ZQUANT_DIM)
    quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
    return quant, emb_loss, info

  def forward(self, coefficients):
    x = torch.concat(coefficients, dim=2)   # B, L, C

    ### x.shape: [B, L C]
    quant, emb_loss, info = self.encode(x)
    ### quant [B, C, L]
    dec = self.decode(quant)
    return dec, emb_loss, info