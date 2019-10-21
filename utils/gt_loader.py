from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from utils.loader import Loader

import torch
import pickle
from functools import partial


class GtLoader(Loader):

    def __init__(self, data_json, visual_feats_dir, opt, tree_pth=None):
        # parent loader instance, see loader.py
        Loader.__init__(self, data_json)

        self.opt = opt
        self.batch_size = opt.batch_size
        self.vis_dim = 2048 + 512 + 512

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}

        # load ann feats
        print('loading visual feats ...')
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        self.visual_feats = torch.load(visual_feats_dir,
            map_location=lambda storage, loc: storage, pickle_module=pickle)
        print('loaded')

        if tree_pth:
            self.trees = torch.load(tree_pth, 'r')
        else:
            self.trees = None

        for image_id, image in self.Images.items():
            split = self.Refs[image['ref_ids'][0]]['split']

            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0

            # add sentences to each subsets
            sent_ids = []
            for ref_id in self.Images[image_id]['ref_ids']:
                sent_ids += self.Refs[ref_id]['sent_ids']
            self.split_ix[split].append(image_id)

        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' % (len(v), k))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def reset_iterator(self, split):
        self.iterators[split] = 0

    # get one of data
    def get_data(self, split):
        # options
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1  # don't forget to -1
        wrapped = False

        # get information about this batch image
        image_id = split_ix[self.iterators[split]]
        ann_ids = self.Images[image_id]['ann_ids']

        # fetch sentences
        sent_ids = []
        gt_ixs = []
        sents = []
        trees = []
        vis = []
        for ref_id in self.Images[image_id]['ref_ids']:
            for sent_id in self.Refs[ref_id]['sent_ids']:
                sent_ids += [sent_id]
                gt_ixs += [ann_ids.index(self.Refs[ref_id]['ann_id'])]
                vis += [self.visual_feats[sent_id]]

                if self.trees:
                    trees += [self.trees[sent_id]['tree']]
                    sents += [self.trees[sent_id]]
                else:
                    sents += [self.Sentences[sent_id]]

        # convert to tensor
        vis = torch.from_numpy(np.asarray(vis)).cuda()
        gt_ixs = torch.from_numpy(np.asarray(gt_ixs)).cuda()

        # update iter status
        if self.iterators[split] + 1 > max_index:
            self.iterators[split] = 0
            wrapped = True
        else:
            self.iterators[split] += 1

        # return
        data = {}
        data['vis'] = vis.float()  # (num_bboxs, fc7_dim)
        data['sents'] = sents
        if self.trees:
            data['trees'] = trees

        data['gts'] = gt_ixs.long()   # (num_sents, )
        data['sent_ids'] = sent_ids
        data['ann_ids'] = ann_ids
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index,
                          'wrapped': wrapped}
        return data
