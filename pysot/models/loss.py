# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


# #############CenterNet Loss Function############# #

def _neg_loss(preds, targets):
    """ Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    """
    pos_inds = targets.eq(1).float().cuda()
    neg_inds = targets.lt(1).float().cuda()
    neg_weights = torch.pow(1 - targets, 4).cuda()

    loss = 0

    preds = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds
    neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)


def focal_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x 1 x h x w)
      gt (batch x h x w)
    """
    n, h, w = gt.size()
    gt = gt.view(n, -1, h, w)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss / n
    else:
        loss = loss - (pos_loss + neg_loss) / n

    return loss


def _reg_loss(regs, gt_regs):
    loss = sum(F.l1_loss(regs, gt_regs, reduction='sum') / (1 + 1e-4))
    return loss / len(regs)


def mask_l1_loss(pred, gt, loss_mask):
    b, _, sh, sw = pred.size()
    diff = (pred - gt).abs()
    diff = diff.sum(dim=1).view(b, sh, sw)
    loss = diff * loss_mask
    return loss.sum().div(b)


def center_kp_criterion(hmap, regs, w_h_, label_hmap, label_regs, label_w_h_, offset_wh_mask_gt):
    hm = _sigmoid(hmap)
    hmap_loss = focal_loss(hm, label_hmap)
    regs_loss = mask_l1_loss(regs, label_regs, offset_wh_mask_gt)
    w_h__loss = mask_l1_loss(w_h_, label_w_h_, offset_wh_mask_gt)
    total_loss = hmap_loss + 1 * regs_loss + 0.1 * w_h__loss

    outputs = {'total_loss': total_loss,
               'hmap_loss': hmap_loss,
               'regs_loss': regs_loss,
               'w_h__loss': w_h__loss}
    return outputs

# ************** ori function *************** #
def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)