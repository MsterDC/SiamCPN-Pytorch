# Copyright (c) LLVISION. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.xcorr import xcorr_depthwise


class AFPN(nn.Module):
    def __init__(self):
        super(AFPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


# class DepthwiseXCorr(nn.Module):
#     def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
#         super(DepthwiseXCorr, self).__init__()
#         self.conv_kernel = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_search = nn.Sequential(
#             nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.ReLU(inplace=True),
#         )
#         self.head = nn.Sequential(
#             nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, out_channels, kernel_size=1)
#         )
#
#     def forward(self, kernel, search):
#         kernel = self.conv_kernel(kernel)
#         search = self.conv_search(search)
#         feature = xcorr_depthwise(search, kernel)
#         out = self.head(feature)
#         return out
#
#
# class DepthwiseAFPN(AFPN):
#     def __init__(self, in_channels=256, out_channels=256):
#         super(DepthwiseAFPN, self).__init__()
#         self.hmap = DepthwiseXCorr(in_channels, out_channels, 1)
#         self.regs = DepthwiseXCorr(in_channels, out_channels, 2)
#         self.w_h_ = DepthwiseXCorr(in_channels, out_channels, 2)
#
#     def forward(self, z_f, x_f):
#         hmap = self.hmap(z_f, x_f)
#         regs = self.regs(z_f, x_f)
#         w_h_ = self.w_h_(z_f, x_f)
#         return hmap, regs, w_h_


# class MultiAFPN(AFPN):
#     def __init__(self, in_channels, weighted=False):
#         super(MultiAFPN, self).__init__()
#         self.weighted = weighted
#         for i in range(len(in_channels)):
#             self.add_module('afpn' + str(i + 2),
#                             DepthwiseAFPN(in_channels[i], in_channels[i]))
#         if self.weighted:
#             self.hmap_weight = nn.Parameter(torch.ones(len(in_channels)))
#             self.regs_weight = nn.Parameter(torch.ones(len(in_channels)))
#             self.w_h__weight = nn.Parameter(torch.ones(len(in_channels)))

# **************** Yao's RPN Head implementation **************** #
class AdaptNonShared(nn.Module):
    def __init__(self, in_channels, hidden, rpn_index, center_crop_size=0):
        super(AdaptNonShared, self).__init__()
        self.center_crop_size = center_crop_size
        self.adapt = self.adapt_non_shared(in_channels, hidden, rpn_index)
        # anti-aliasing & reduce
        self.dw_reduce = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

    def adapt_non_shared(self, in_channels, hidden, rpn_index):
        layers = []
        layers += [nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)]
        layers += [nn.BatchNorm2d(hidden)]
        layers += [nn.ReLU(inplace=True)]

        # up_layers = [nn.ConvTranspose2d(hidden, hidden, kernel_size=3, stride=2, padding=0, groups=hidden, bias=False),
        #              nn.BatchNorm2d(in_channels),
        #              ]
        # if rpn_index == 1:
        #     layers += up_layers
        # elif rpn_index == 2:
        #     layers += up_layers
        #     layers += up_layers

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.adapt(x)

        if self.center_crop_size:
            l = (x.size(3) - self.center_crop_size) // 2
            r = l + self.center_crop_size
            x = x[:, :, l:r, l:r]

        x = self.dw_reduce(x)
        return x


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, rpn_index):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = AdaptNonShared(in_channels, hidden, rpn_index, center_crop_size=7)
        self.conv_search = AdaptNonShared(in_channels, hidden, rpn_index)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)

        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseAFPN(AFPN):
    def __init__(self, in_channels, hidden, rpn_index):
        super(DepthwiseAFPN, self).__init__()
        self.hmap = DepthwiseXCorr(in_channels, hidden, 1, rpn_index)
        self.regs = DepthwiseXCorr(in_channels, hidden, 2, rpn_index)
        self.w_h_ = DepthwiseXCorr(in_channels, hidden, 2, rpn_index)

    def forward(self, z_f, x_f):
        hmap = self.hmap(z_f, x_f)
        regs = self.regs(z_f, x_f)
        w_h_ = self.w_h_(z_f, x_f)
        return hmap, regs, w_h_


class MultiAFPN(AFPN):
    def __init__(self, in_channels, weighted=False):
        super(MultiAFPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('afpn' + str(i + 2),
                            DepthwiseAFPN(in_channels[i], in_channels[i], rpn_index=i))
        if self.weighted:
            self.hmap_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.regs_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.w_h__weight = nn.Parameter(torch.ones(len(in_channels)))

# ************************** [END] **************************** #
    def forward(self, z_fs, x_fs):
        hmap = []
        regs = []
        w_h_ = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            afpn = getattr(self, 'afpn' + str(idx))
            h, r, w = afpn(z_f, x_f)  # get the output of DepthwiseAFPN
            hmap.append(h)
            regs.append(r)
            w_h_.append(w)

        if self.weighted:
            hmap_weight = F.softmax(self.hmap_weight, 0)
            regs_weight = F.softmax(self.regs_weight, 0)
            w_h__weight = F.softmax(self.w_h__weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(hmap, hmap_weight), weighted_avg(regs, regs_weight), weighted_avg(w_h_, w_h__weight)
        else:
            return avg(hmap), avg(regs), avg(w_h_)
