import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lib.quantizer import VectorQuantizer
from models.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from base import BaseModel


class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """

    def __init__(self, cfg):
        super().__init__()
        self.encoder = TransformerEncoder(cfg.BACKBONE)
        self.decoder = TransformerDecoder(cfg.BACKBONE, cfg.BACKBONE.IN_DIM)
        self.quantize = VectorQuantizer(cfg.HEAD.N_EMBED,
                                        cfg.HEAD.ZQUANT_DIM,
                                        beta=0.25)
        self.cfg = cfg

    def encode(self, x, x_a=None):
        h = self.encoder(x) ## x --> z'
        h = h.view(x.shape[0], -1, self.cfg.BACKBONE.FACE_QUAN_NUM, self.cfg.HEAD.ZQUANT_DIM)
        h = h.view(x.shape[0], -1, self.cfg.HEAD.ZQUANT_DIM)
        quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        return quant, emb_loss, info


    def decode(self, quant):
        #BCL
        quant = quant.permute(0,2,1)
        quant = quant.view(quant.shape[0], -1, self.cfg.BACKBONE.FACE_QUAN_NUM, self.cfg.HEAD.ZQUANT_DIM).contiguous()
        quant = quant.view(quant.shape[0], -1,  self.cfg.BACKBONE.FACE_QUAN_NUM*self.cfg.HEAD.ZQUANT_DIM).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = self.decoder(quant) ## z' --> x

        return dec

    def forward(self, x, template):
        template = template.unsqueeze(1) # B, V*3 -> B, 1, V*3
        x = x - template

        ### x.shape: [B, L C]
        quant, emb_loss, info = self.encode(x)
        ### quant [B, C, L]
        dec = self.decode(quant)

        dec = dec + template
        return dec, emb_loss, info


    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0
        ##########
        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    size = cfg.IN_DIM
    dim = cfg.HIDDEN_SIZE
    self.vertice_mapping = nn.Sequential(nn.Linear(size, dim), nn.LeakyReLU(cfg.NEG, True))
    if cfg.QUANT_FACTOR == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim, dim, 5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(cfg.NEG, True),
                    nn.InstanceNorm1d(dim, affine=cfg.INAFFINE)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(cfg.NEG, True),
                    nn.InstanceNorm1d(dim, affine=cfg.INAFFINE)
                    )]
        for _ in range(1, cfg.QUANT_FACTOR):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(cfg.NEG, True),
                        nn.InstanceNorm1d(dim, affine=cfg.INAFFINE),
                        nn.MaxPool1d(2)
                        )]
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=cfg.HIDDEN_SIZE,
        hidden_size=cfg.HIDDEN_SIZE,
        num_hidden_layers=\
                cfg.NUM_HIDDEN_LAYERS,
        num_attention_heads=\
                cfg.NUM_ATTENTION_HEADS,
        intermediate_size=\
                cfg.INTERMEDIATE_SIZE)
    self.encoder_pos_embedding = PositionalEncoding(
        cfg.HIDDEN_SIZE)
    self.encoder_linear_embedding = LinearEmbedding(
        cfg.HIDDEN_SIZE,
        cfg.HIDDEN_SIZE)

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.vertice_mapping(inputs)
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, cfg, out_dim, is_audio=False):
    super().__init__()
    self.cfg = cfg
    size=self.cfg.HIDDEN_SIZE
    dim=self.cfg.HIDDEN_SIZE
    self.expander = nn.ModuleList()
    if cfg.QUANT_FACTOR == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.cfg.NEG, True),
                    nn.InstanceNorm1d(dim, affine=cfg.INAFFINE)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.cfg.NEG, True),
                    nn.InstanceNorm1d(dim, affine=cfg.INAFFINE)
                            ))                      
        num_layers = cfg.QUANT_FACTOR+2 \
            if is_audio else cfg.QUANT_FACTOR

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.cfg.NEG, True),
                                nn.InstanceNorm1d(dim, affine=cfg.INAFFINE),
                                ))
    self.decoder_transformer = Transformer(
        in_size=self.cfg.HIDDEN_SIZE,
        hidden_size=self.cfg.HIDDEN_SIZE,
        num_hidden_layers=\
            self.cfg.NUM_HIDDEN_LAYERS,
        num_attention_heads=\
            self.cfg.NUM_ATTENTION_HEADS,
        intermediate_size=\
            self.cfg.INTERMEDIATE_SIZE)
    self.decoder_pos_embedding = PositionalEncoding(
        self.cfg.HIDDEN_SIZE)
    self.decoder_linear_embedding = LinearEmbedding(
        self.cfg.HIDDEN_SIZE,
        self.cfg.HIDDEN_SIZE)

    self.vertice_map_reverse = nn.Linear(cfg.HIDDEN_SIZE,out_dim)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)
    inputs = inputs.permute(0,2,1) #BLC
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.vertice_map_reverse(decoder_features)
    return pred_recon