import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss

def nt_xent_loss(feature_a, feature_b, temperature):
    """
    Normalized temperature-scaled cross entropy loss.

    (Adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py)

    Args:
        feature_a (torch.Tensor): shape (batch_size, feature_dim)
        feature_b (torch.Tensor): shape (batch_size, feature_dim)
        temperature (float): temperature scaling factor

    Returns:
        torch.Tensor: scalar
    """
    batch_size = feature_a.shape[0]
    device = feature_a.device

    features = torch.cat([feature_a, feature_b], dim=0)

    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)

    # select the positives and negatives
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels)
    return loss