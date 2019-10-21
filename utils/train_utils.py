from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def build_optimizer(weights, biases, opt):
    if opt.optim == 'adam':
        return optim.Adam([
                    {'params': weights},
                    {'params': biases, 'weight_decay': 0}],
                    lr=opt.learning_rate,
                    betas=(opt.optim_alpha, opt.optim_beta),
                    eps=opt.optim_epsilon,
                    weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union


def computeLabels(gd_boxes, det_boxes, threshold=0.5):
    labels = np.zeros([len(gd_boxes), len(det_boxes)])
    for i, gd_box in enumerate(gd_boxes):
        for j, det_box in enumerate(det_boxes):
            l = computeIoU(gd_box, det_box)
            l = l if l >= threshold else 0
            labels[i, j] = l

    labels = torch.from_numpy(labels).cuda().float()
    return labels
