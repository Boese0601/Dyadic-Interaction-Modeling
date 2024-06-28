#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn


def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

def calc_vq_loss_AV(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    # audio dim: 768, visual dim: 56
    # pred: [pred_v, pred_a]
    pred_v, pred_a = pred[:, :, :56], pred[:, :, 56:]
    target_v, target_a = target[:, :, :56], target[:, :, 56:]
    rec_loss = nn.L1Loss()(pred_v, target_v) + nn.L1Loss()(pred_a, target_a)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]

def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss
