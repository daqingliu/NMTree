from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import utils.train_utils as train_utils
from utils.loader import Loader


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


def eval_gt_split(loader, model, crit, split, opt, is_dump_json=False):
    # initialize
    model.eval()
    loader.reset_iterator(split)

    # get the predict result
    pred_sent = {}
    total_loss = 0.0
    iterations = 0.0
    while True:
        data = loader.get_data(split)

        # forward
        scores = model(data)
        if crit:
            loss = crit(scores, data['gts']) if crit else 0
            total_loss += loss.data.cpu().numpy()
            
        iterations += 1

        scores = scores.data.cpu().numpy()
        pred_ix = np.argmax(scores, axis=1)

        # get the predict result
        ann_ids = data['ann_ids']
        for ix, sent_id in enumerate(data['sent_ids']):
            pred_sent[sent_id] = {'sent_id': sent_id,
                                  'ann_id': ann_ids[pred_ix[ix]],
                                  'candidates': ann_ids,
                                  'box': loader.Anns[ann_ids[pred_ix[ix]]]['box']}

        # if used up
        if data['bounds']['wrapped']:
            break

    # compute accuracy
    n = 0.0
    acc = 0.0
    for _, ref in loader.Refs.items():
        if ref['split'] == split:
            for sent_id in ref['sent_ids']:
                n += 1
                if pred_sent[sent_id]['ann_id'] == ref['ann_id']:
                    acc += 1
                # check out the candidates are right, for fair accuracy
                assert loader.Images[ref['image_id']]['ann_ids'] == pred_sent[sent_id]['candidates']
        else:
            continue

    # save the predict result if get better result
    if is_dump_json:
        checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_split_by'] + '_' + opt['id'])
        json.dump(pred_sent, open(osp.join(checkpoint_dir, split+'_gt_res.json'), 'w'))

    # restore the model to train
    model.train()

    return acc, n, total_loss/iterations


def eval_det_split(loader, model, crit, split, opt, is_dump_json=False):
    model.eval()
    loader.reset_iterator(split)

    pred_sent = {}
    n = 0.0
    acc = 0.0
    total_loss = 0.0
    iterations = 0.0
    while True:
        data = loader.get_data(split)
        
        scores = model(data)
    
        if crit:
            target = train_utils.computeLabels(data['gd_boxes'], \
                [loader.Dets[a]['box'] for a in data['det_ids']])
            loss = crit(scores, F.normalize(target, p=1, dim=-1))
            total_loss += loss.data.cpu().numpy()
        iterations += 1

        scores = scores.data.cpu().numpy()
        pred_ix = np.argmax(scores, axis=1)

        det_ids = data['det_ids']
        sent_ids = data['sent_ids']

        for ix, sent_id in enumerate(sent_ids):
            pred_det_id = det_ids[pred_ix[ix]]
            pred_box = loader.Dets[pred_det_id]['box']
            gd_box = data['gd_boxes'][ix]

            if computeIoU(pred_box, gd_box) >= 0.5:
                acc += 1
            n += 1

            pred_sent[sent_id] = {'sent_id': sent_id,
                                  'pred_det_id': pred_det_id,
                                  'boxes': loader.Dets[pred_det_id]['box']}

        # if used up
        if data['bounds']['wrapped']:
            break

    # save the predict result if get better result
    if is_dump_json:
        checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_split_by'] + '_' + opt['id'])
        json.dump(pred_sent, open(osp.join(checkpoint_dir, split+'_det_res.json'), 'w'))

    return acc, n, total_loss/iterations


def eval_gt_split_by_length(loader, model, crit, split, opt, is_dump_json=False):
    # initialize
    model.eval()
    loader.reset_iterator(split)

    data_json = 'data/feats/refcocog_umd/data_plain.json'
    ori_loader = Loader(data_json)

    # get the predict result
    pred_sent = {}
    total_loss = 0.0
    iterations = 0.0
    while True:
        data = loader.get_data(split)

        # forward
        scores = model(data)
        if crit:
            loss = crit(scores, data['gts']) if crit else 0
            total_loss += loss.data.cpu().numpy()
            
        iterations += 1

        scores = scores.data.cpu().numpy()
        pred_ix = np.argmax(scores, axis=1)

        # get the predict result
        ann_ids = data['ann_ids']
        for ix, sent_id in enumerate(data['sent_ids']):
            pred_sent[sent_id] = {'sent_id': sent_id,
                                  'ann_id': ann_ids[pred_ix[ix]],
                                  'candidates': ann_ids,
                                  'box': loader.Anns[ann_ids[pred_ix[ix]]]['box']}

        # if used up
        if data['bounds']['wrapped']:
            break

    # compute accuracy
    n = {}
    acc = {}
    for _, ref in loader.Refs.items():
        if ref['split'] == split:
            for sent_id in ref['sent_ids']:
                sent_len = len(ori_loader.Sentences[sent_id]['tokens'])
                n[sent_len] = n.get(sent_len, 0) + 1
                if pred_sent[sent_id]['ann_id'] == ref['ann_id']:
                    acc[sent_len] = acc.get(sent_len, 0) + 1
                # check out the candidates are right, for fair accuracy
                assert loader.Images[ref['image_id']]['ann_ids'] == pred_sent[sent_id]['candidates']
        else:
            continue

    # save the predict result if get better result
    if is_dump_json:
        checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_split_by'] + '_' + opt['id'])
        json.dump(pred_sent, open(osp.join(checkpoint_dir, split+'_gt_res.json'), 'w'))

    # restore the model to train
    model.train()

    return acc, n, total_loss/iterations
