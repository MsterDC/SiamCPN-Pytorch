# Copyright (c) LLVISION. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from pysot.core.config_new import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head import get_afpn_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        # build afpn head
        if cfg.AFPN.AFPN:
            self.afpn_head = get_afpn_head(cfg.AFPN.TYPE,
                                           **cfg.AFPN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        hmap, regs, w_h_ = self.afpn_head(self.zf, xf)
        return {
            'hmap': hmap,
            'regs': regs,
            'w_h_': w_h_
        }

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_hmap = data['label_hmap'].cuda()
        label_regs = data['label_regs'].cuda()
        label_w_h_ = data['label_w_h_'].cuda()
        offset_wh_mask_gt = data['offset_wh_mask_gt'].cuda()
        hmap, regs, w_h_ = None, None, None

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.AFPN.AFPN:
            hmap, regs, w_h_ = self.afpn_head(zf, xf)

        return hmap, regs, w_h_, label_hmap, label_regs, label_w_h_, offset_wh_mask_gt

    def init_weights(self):
        # self.backbone.apply(ModelBuilder.op_weights_init)
        if cfg.ADJUST.ADJUST:
            self.neck.apply(ModelBuilder.op_weights_init)
        self.afpn_head.apply(ModelBuilder.op_weights_init)
        self.afpn_head.apply(ModelBuilder.head_weights_fill)

    @staticmethod
    def op_weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def head_weights_fill(m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                shape = m.weight.data.shape
                if shape[0] == 1:
                    # fill hm to make sigmoid(hm)=0.01 in the phase of initialization
                    m.bias.data.fill_(-2.19)
