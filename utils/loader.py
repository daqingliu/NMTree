"""
data_json has 
0. refs        : list of {ref_id, ann_id, box, image_id, split, category_id, sent_ids}
1. images      : list of {image_id, ref_ids, ann_ids, file_name, width, height, h5_id}
2. anns        : list of {ann_id, category_id, image_id, box, h5_id}
3. sentences   : list of {sent_id, tokens}
4: word_to_ix  : word->ix
5: cat_to_ix   : cat->ix
Note, box in [xywh] format
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


class Loader(object):

    def __init__(self, data_json):
        # load the json file which contains info about the dataset
        print('Loader loading data.json: ', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix']
        
        if 'tag_to_ix' in self.info:
            self.tag_to_ix = self.info['tag_to_ix']
            self.dep_to_ix = self.info['dep_to_ix']

        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        print('word vocab size is ', self.word_vocab_size)
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object category size is ', len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        print('we have %s images.' % len(self.images))
        print('we have %s anns.' % len(self.anns))
        print('we have %s refs.' % len(self.refs))
        print('we have %s sentences.' % len(self.sentences))

        # construct mapping
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs for sent_id in ref['sent_ids']}

    @property
    def word_vocab_size(self):
        return len(self.word_to_ix)

    @property
    def tag_vocab_size(self):
        return len(self.tag_to_ix)

    @property
    def dep_vocab_size(self):
        return len(self.dep_to_ix)
