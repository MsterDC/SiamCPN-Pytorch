# Copyright (c) LLVISION. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from pysot.core.config_new import cfg
from pysot.tracker.base_tracker import SiameseTracker


class SiamAFPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamAFPNTracker, self).__init__()
        self.model = model
        self.model.eval()

    def _decode_bbox(self, offset_wh, best_idx):
        regs, w_h_ = offset_wh[0], offset_wh[1]
        output_w = regs.size(3)
        stride = 8
        regs = regs.view(2, -1)
        w_h_ = w_h_.view(2, -1)
        regs = regs.data.cpu().numpy()
        w_h_ = w_h_.data.cpu().numpy()

        cx = best_idx % output_w
        cy = best_idx // output_w

        cx = cx + regs[0, best_idx]
        cy = cy + regs[1, best_idx]
        w = w_h_[0, best_idx] / 0.1
        h = w_h_[1, best_idx] / 0.1

        bbox = np.array([cx, cy, w, h]) * stride
        c = cfg.TRACK.INSTANCE_SIZE // 2
        bbox[0:2] = bbox[0:2] + 31 - c
        return bbox

    def _convert_score(self, score):
        score = score.view(-1)
        score = torch.sigmoid(score).data.cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox, the ROI of first frame.
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        # add padding to template patch
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop: resize &  padding & crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['hmap'])

        best_idx = np.argmax(score)
        best_score = score[best_idx]

        offset_wh = [outputs['regs'], outputs['w_h_']]

        bbox = self._decode_bbox(offset_wh, best_idx) / scale_z

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        if best_score >= 0.1:
            self.center_pos = np.array([cx, cy])
            self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
            'best_score': best_score
        }
