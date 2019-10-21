from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse

import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))

import models
from utils.gt_loader import GtLoader
import utils.eval_utils as eval_utils


def parse_eval_opt():
    parser = argparse.ArgumentParser()

    # General Settings
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='refcocog')
    parser.add_argument('--split_by', type=str, default='umd')
    parser.add_argument('--load_best', type=bool, default=True)
    parser.add_argument('--split', type=str, default='all', help="val/test/all")
    parser.add_argument('--visual_feat_file', type=str, default='matt_res_gt_feats.pth')

    parser.add_argument('--dump_images', type=bool, default=False)
    parser.add_argument('--dump_json', type=bool, default=True)

    args = parser.parse_args()
    return args


def show_acc(acc, n, split='test'):
    eval_accuracy = acc / n
    print("%s set evaluated. acc = %d / %d = %.2f%%" % (split, acc, n, eval_accuracy*100))


def split_eval(opt):
    # Set path
    infos_file = 'infos-best.json' if opt.load_best else 'info.json'
    model_file = 'model-best.pth' if opt.load_best else 'model.pth'
    infos_path = os.path.join(opt.log_path, infos_file)
    model_path = os.path.join(opt.log_path, model_file)

    # Load infos
    with open(infos_path, 'rb') as f:
        infos = json.load(f)

    ignore = ['visual_feat_file', 'dataset', 'split_by']
    for k in infos['opt'].keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == infos['opt'][k], k + ' option not consistent'
            else:
                vars(opt).update({k: infos['opt'][k]})  # copy over options from model

    # set up loader
    data_json = os.path.join(opt.feats_path, opt.dataset+'_'+opt.split_by, opt.data_file+'.json')
    visual_feats_dir = os.path.join(opt.feats_path, opt.dataset+'_'+opt.split_by, opt.visual_feat_file)
    data_pth = os.path.join(opt.feats_path, opt.dataset+'_'+opt.split_by, opt.data_file + '.pth')
    visual_feats_dir = os.path.join(opt.feats_path, opt.dataset+'_'+opt.split_by, opt.visual_feat_file)

    if os.path.isfile(data_pth):
        loader = GtLoader(data_json, visual_feats_dir, opt, data_pth)
        opt.tag_vocab_size = loader.tag_vocab_size
        opt.dep_vocab_size = loader.dep_vocab_size
    else:
        loader = GtLoader(data_json, visual_feats_dir, opt)

    opt.word_vocab_size = loader.word_vocab_size
    opt.vis_dim = loader.vis_dim

    # Print out the option variables
    print("*" * 20)
    for k, v in opt.__dict__.items():
        print("%r: %r" % (k, v))
    print("*" * 20)

    # set up model and criterion
    model = models.setup(opt, loader).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate all sets
    acc = {}
    print("Start evaluating %s" % opt.dataset+'_'+opt.split_by)
    if opt.split in ['all', 'val']:
        acc, n, _ = eval_utils.eval_gt_split(loader, model, None, 'val', vars(opt), opt.dump_json)
        show_acc(acc, n, split='val')

    if opt.split in ['all', 'test']:
        if opt.dataset in ['refcoco', 'refcoco+']:
            acc, n, _ = eval_utils.eval_gt_split(loader, model, None, 'testA', vars(opt), opt.dump_json)
            show_acc(acc, n, split='testA')
            acc, n, _ = eval_utils.eval_gt_split(loader, model, None, 'testB', vars(opt), opt.dump_json)
            show_acc(acc, n, split='testB')
        else:
            acc, n, _ = eval_utils.eval_gt_split(loader, model, None, 'test', vars(opt), opt.dump_json)
            show_acc(acc, n, split='test')

    print("All sets evaluated.")

    return acc


if __name__ == '__main__':
    opt = parse_eval_opt()
    acc = split_eval(opt)
